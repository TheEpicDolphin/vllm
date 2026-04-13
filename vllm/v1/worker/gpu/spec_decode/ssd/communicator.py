# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
NCCL P2P communication for Speculative Speculative Decoding (SSD).

The SSD speculator device and each base model device exchange:
  - base -> speculator: num_accepted tokens per request
  - speculator -> base: K×K draft tokens per request

Communication is split into two phases to allow the rejection sampler
and speculator to each own half of the protocol:
  Phase 1 (rejection sampler): base sends num_accepted, speculator receives
  Phase 2 (speculator.propose): speculator sends K×K drafts, base receives

Workers in different engine-core processes are in separate NCCL worlds
(non-MoE DP resets data_parallel_size to 1).  To communicate, we create
a private stateless NCCL process group that spans all DP ranks.

The stateless PG is NOT registered in PyTorch's _world.pg_map, so we
must use the low-level ``group.send`` / ``group.recv`` methods instead
of ``torch.distributed.send`` / ``torch.distributed.recv``.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from vllm.distributed.utils import (
    stateless_init_torch_distributed_process_group,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


class SSDCommunicator:
    """Handles NCCL P2P send/recv between base ranks and the SSD
    speculator rank using a private stateless NCCL process group."""

    def __init__(
        self,
        is_speculator: bool,
        dp_size: int,
        dp_rank: int,
        num_speculative_tokens: int,
        max_num_reqs: int,
        device: torch.device,
        master_ip: str,
        nccl_port: int,
    ):
        self.is_speculator = is_speculator
        self.dp_size = dp_size
        self.dp_rank = dp_rank
        self.num_speculative_tokens = num_speculative_tokens
        self.max_num_reqs = max_num_reqs
        self.device = device
        self._master_ip = master_ip
        self._nccl_port = nccl_port

        self.K = num_speculative_tokens
        self.speculator_dp_rank = dp_size - 1
        self.num_base_ranks = dp_size - 1

        if is_speculator:
            self.recv_num_reqs = [
                torch.zeros(1, dtype=torch.int32, device=device)
                for _ in range(self.num_base_ranks)
            ]
            self.recv_num_accepted = [
                torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
                for _ in range(self.num_base_ranks)
            ]
            self.send_draft_tokens = [
                torch.zeros(
                    max_num_reqs,
                    self.K,
                    self.K,
                    dtype=torch.int64,
                    device=device,
                )
                for _ in range(self.num_base_ranks)
            ]
            self._last_num_reqs: list[int] = [0] * self.num_base_ranks
        else:
            self.send_num_reqs = torch.zeros(
                1,
                dtype=torch.int32,
                device=device,
            )
            self.send_num_accepted = torch.zeros(
                max_num_reqs,
                dtype=torch.int32,
                device=device,
            )
            self.recv_draft_tokens = torch.zeros(
                max_num_reqs,
                self.K,
                self.K,
                dtype=torch.int64,
                device=device,
            )

        # Create NCCL group eagerly — both engine-core workers construct
        # the communicator at roughly the same time during model loading,
        # so the TCP rendezvous succeeds within the default timeout.
        logger.info(
            "SSD: creating stateless NCCL group (rank=%d, world=%d, ip=%s, port=%d)",
            self.dp_rank,
            self.dp_size,
            self._master_ip,
            self._nccl_port,
        )
        self._ssd_pg: dist.ProcessGroup = (
            stateless_init_torch_distributed_process_group(
                host=self._master_ip,
                port=self._nccl_port,
                rank=self.dp_rank,
                world_size=self.dp_size,
                backend="nccl",
                group_name="ssd_nccl",
            )
        )

        # Communication is disabled until the engine loop explicitly
        # enables it (after warmup/profiling complete).
        self.enabled = False

    @property
    def ssd_group(self) -> dist.ProcessGroup:
        return self._ssd_pg

    def _send(self, tensor: torch.Tensor, dst: int) -> None:
        """Low-level blocking send that bypasses dist.send wrappers."""
        self._ssd_pg.send([tensor], dst, 0).wait()

    def _recv(self, tensor: torch.Tensor, src: int) -> None:
        """Low-level blocking recv that bypasses dist.recv wrappers."""
        self._ssd_pg.recv([tensor], src, 0).wait()

    # ------------------------------------------------------------------
    # Phase 1: acceptance info  (called from SSDRejectionSampler)
    # ------------------------------------------------------------------

    def base_send_accepted(
        self,
        num_accepted: torch.Tensor,
        num_reqs: int,
    ) -> None:
        """Base -> speculator: send num_reqs and num_accepted."""
        self.send_num_reqs[0] = num_reqs
        self._send(self.send_num_reqs, self.speculator_dp_rank)

        if num_reqs > 0:
            self.send_num_accepted[:num_reqs].copy_(num_accepted[:num_reqs])
            self._send(self.send_num_accepted[:num_reqs], self.speculator_dp_rank)

    def speculator_recv_accepted(
        self,
    ) -> list[tuple[torch.Tensor, int]]:
        """Speculator <- bases: receive num_reqs and num_accepted from all."""
        results = []
        for base_dp_rank in range(self.num_base_ranks):
            self._recv(self.recv_num_reqs[base_dp_rank], base_dp_rank)
            num_reqs = self.recv_num_reqs[base_dp_rank].item()
            self._last_num_reqs[base_dp_rank] = num_reqs

            if num_reqs > 0:
                self._recv(
                    self.recv_num_accepted[base_dp_rank][:num_reqs],
                    base_dp_rank,
                )

            results.append(
                (
                    self.recv_num_accepted[base_dp_rank],
                    num_reqs,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Phase 2: draft tokens  (called from SSDSpeculator.propose)
    # ------------------------------------------------------------------

    def speculator_send_drafts(
        self,
        draft_tokens: torch.Tensor | None,
    ) -> None:
        """Speculator -> bases: send K×K draft tokens to each base rank.

        Args:
            draft_tokens: [max_num_reqs, K, K] or None (first step).
        """
        for base_dp_rank in range(self.num_base_ranks):
            num_reqs = self._last_num_reqs[base_dp_rank]

            if num_reqs > 0:
                buf = self.send_draft_tokens[base_dp_rank]
                if draft_tokens is not None:
                    buf[:num_reqs].copy_(draft_tokens[:num_reqs])
                else:
                    buf[:num_reqs].zero_()
                self._send(buf[:num_reqs].reshape(-1), base_dp_rank)

    def base_recv_drafts(self, num_reqs: int) -> torch.Tensor:
        """Base <- speculator: receive K×K draft tokens.

        Args:
            num_reqs: number of active requests on this base.

        Returns:
            [max_num_reqs, K, K] tensor of draft tokens.
        """
        if num_reqs > 0:
            flat_recv = self.recv_draft_tokens[:num_reqs].reshape(-1)
            self._recv(flat_recv, self.speculator_dp_rank)

        return self.recv_draft_tokens

    # ------------------------------------------------------------------
    # Legacy combined methods (kept for backward compat)
    # ------------------------------------------------------------------

    def base_sync(
        self,
        num_accepted: torch.Tensor,
        num_reqs: int,
    ) -> torch.Tensor:
        self.base_send_accepted(num_accepted, num_reqs)
        return self.base_recv_drafts(num_reqs)

    def speculator_sync(
        self,
        draft_tokens_per_base: list[torch.Tensor | None],
    ) -> list[tuple[torch.Tensor, int]]:
        results = self.speculator_recv_accepted()
        draft = (
            draft_tokens_per_base[0]
            if draft_tokens_per_base and draft_tokens_per_base[0] is not None
            else None
        )
        self.speculator_send_drafts(draft)
        return results
