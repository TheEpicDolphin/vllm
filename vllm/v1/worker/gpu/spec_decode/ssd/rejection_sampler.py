# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SSD Rejection Sampler.

On the speculator device: receives num_accepted from each base rank via NCCL
during __call__, producing a dummy SamplerOutput. The actual token generation
happens later in SSDSpeculator.propose().

On the base device: delegates to the standard RejectionSampler, behaving
identically to non-SSD speculative decoding. When the base has no requests
(input_batch is None), handles the empty NCCL sync so the speculator doesn't
deadlock.
"""
from __future__ import annotations

import torch

from vllm.config import SpeculativeConfig
from vllm.logger import init_logger
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
from vllm.v1.worker.gpu.spec_decode.ssd.communicator import SSDCommunicator

logger = init_logger(__name__)


class SSDRejectionSampler:
    """Rejection sampler for Speculative Speculative Decoding.

    Checks its role (speculator vs base) and behaves accordingly:
    - Speculator: receives num_accepted from base ranks via NCCL, returns
      a dummy SamplerOutput. Stores received acceptance data for use by
      SSDSpeculator.propose().
    - Base: delegates entirely to the standard RejectionSampler.
    """

    def __init__(
        self,
        sampler: Sampler,
        spec_config: SpeculativeConfig,
        is_speculator: bool,
        ssd_comm: SSDCommunicator,
    ):
        self.is_speculator = is_speculator
        self.ssd_comm = ssd_comm
        self.num_speculative_steps = spec_config.num_speculative_tokens

        if not is_speculator:
            self.rejection_sampler = RejectionSampler(sampler, spec_config)
        else:
            self.rejection_sampler = None

        # Stored by the speculator side after receiving from bases.
        # List of (num_accepted_tensor, num_reqs) per base rank.
        self.last_base_results: list[tuple[torch.Tensor, int]] = []
        self.draft_logits: torch.Tensor | None = None

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch | None,
        draft_logits: torch.Tensor | None = None,
    ) -> SamplerOutput:
        if not self.is_speculator:
            if input_batch is None:
                # Base with no requests: send num_reqs=0 so the speculator
                # knows this rank has no work, then recv (discard) drafts.
                self.ssd_comm.base_send_accepted(
                    torch.zeros(0, dtype=torch.int32, device=logits.device),
                    num_reqs=0,
                )
                self.ssd_comm.base_recv_drafts(num_reqs=0)
                K = self.num_speculative_steps
                return SamplerOutput(
                    sampled_token_ids=torch.zeros(
                        0, K + 1, dtype=torch.int64, device=logits.device),
                    logprobs_tensors=None,
                    num_nans=None,
                    num_sampled=torch.zeros(
                        0, dtype=torch.int32, device=logits.device),
                )
            assert self.rejection_sampler is not None
            return self.rejection_sampler(logits, input_batch, draft_logits)

        # Speculator device: receive num_accepted from each base rank.
        # The speculator doesn't do real rejection sampling — it just needs
        # to know what the bases accepted so propose() can act on it.
        self.last_base_results = self.ssd_comm.speculator_recv_accepted()

        for base_dp_rank, (num_accepted, num_reqs) in enumerate(
            self.last_base_results
        ):
            if num_reqs > 0:
                logger.debug(
                    "SSD speculator: base dp_rank=%d, num_reqs=%d, "
                    "avg_accepted=%.1f",
                    base_dp_rank, num_reqs,
                    num_accepted[:num_reqs].float().mean().item(),
                )

        # Return a dummy SamplerOutput. The speculator doesn't produce
        # user-facing tokens — it only generates drafts.
        K = self.num_speculative_steps
        dummy_sampled = torch.zeros(0, K + 1, dtype=torch.int64,
                                    device=logits.device)
        dummy_num_sampled = torch.zeros(0, dtype=torch.int32,
                                        device=logits.device)
        return SamplerOutput(
            sampled_token_ids=dummy_sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=dummy_num_sampled,
        )
