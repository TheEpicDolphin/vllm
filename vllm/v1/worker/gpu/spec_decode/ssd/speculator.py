# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Speculative Speculative Decoding (SSD) Speculator.

A single role-aware class that checks whether it is the speculator device
or a base device and behaves accordingly:

  Speculator device:
    propose() generates K×K draft tokens via a standalone draft model, then
    sends them to each base rank via NCCL.

  Base device:
    propose() sends num_accepted to the speculator via NCCL, receives K×K
    draft tokens, selects the K continuations from the accepted position,
    and returns them as [num_reqs, K].

The draft model must be a standalone autoregressive model (e.g.
Llama-3.2-1B).  EAGLE/EAGLE-3 methods are NOT supported because they
require hidden states from the base model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import init_attn_backend
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.ssd.communicator import SSDCommunicator

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.spec_decode.ssd.rejection_sampler import (
        SSDRejectionSampler,
    )

logger = init_logger(__name__)


@triton.jit
def _update_draft_inputs_kernel(
    input_ids_ptr,
    positions_ptr,
    input_hidden_states_ptr,
    input_hidden_states_stride,
    seq_lens_ptr,
    max_model_len,
    draft_tokens_ptr,
    output_hidden_states_ptr,
    output_hidden_states_stride,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)

    # Draft token -> Input ID.
    draft_token = tl.load(draft_tokens_ptr + req_idx)
    tl.store(input_ids_ptr + req_idx, draft_token)

    # Output hidden states -> Input hidden states.
    for i in range(0, hidden_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < hidden_size
        output_hidden_states = tl.load(
            output_hidden_states_ptr + req_idx * output_hidden_states_stride + block,
            mask=mask,
        )
        tl.store(
            input_hidden_states_ptr + req_idx * input_hidden_states_stride + block,
            output_hidden_states,
            mask=mask,
        )

    # Increment position and seq_lens.
    # NOTE(woosuk): To prevent out-of-range access, we clamp these values
    # if they reach the max model length.
    position = tl.load(positions_ptr + req_idx)
    position = tl.minimum(position + 1, max_model_len - 1)
    tl.store(positions_ptr + req_idx, position)

    seq_len = tl.load(seq_lens_ptr + req_idx)
    seq_len = tl.minimum(seq_len + 1, max_model_len)
    tl.store(seq_lens_ptr + req_idx, seq_len)


def update_draft_inputs(
    draft_tokens: torch.Tensor,
    output_hidden_states: torch.Tensor,
    input_buffers: InputBuffers,
    hidden_states: torch.Tensor,
    max_model_len: int,
):
    num_reqs, hidden_size = output_hidden_states.shape
    _update_draft_inputs_kernel[(num_reqs,)](
        input_buffers.input_ids,
        input_buffers.positions,
        hidden_states,
        hidden_states.stride(0),
        input_buffers.seq_lens,
        max_model_len,
        draft_tokens,
        output_hidden_states,
        output_hidden_states.stride(0),
        hidden_size,
        BLOCK_SIZE=1024,
    )


class SSDSpeculator:
    """Role-aware speculator for Speculative Speculative Decoding.

    On the speculator device: generates K×K draft tokens per step and
    sends them to base ranks.
    On a base device: receives K×K drafts from the speculator, selects
    the relevant continuations, and returns [num_reqs, K].
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        is_speculator: bool,
        ssd_comm: SSDCommunicator,
    ):
        self.vllm_config = vllm_config
        self.device = device
        self.is_speculator = is_speculator
        self.ssd_comm = ssd_comm

        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None
        self.K = self.speculative_config.num_speculative_tokens
        self.draft_model_config = self.speculative_config.draft_model_config

        self.scheduler_config = vllm_config.scheduler_config
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.vocab_size = self.draft_model_config.get_vocab_size()
        self.dtype = vllm_config.model_config.dtype

        self.input_buffers = InputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=device,
        )

        # K×K draft tokens: [max_num_reqs, K, K]
        self.draft_tokens = torch.zeros(
            self.max_num_reqs,
            self.K,
            self.K,
            dtype=torch.int64,
            device=device,
        )

        # Primary draft tokens from the previous round: [max_num_reqs, K]
        self.primary_draft_tokens = torch.zeros(
            self.max_num_reqs,
            self.K,
            dtype=torch.int64,
            device=device,
        )

        self.temperature = torch.zeros(
            self.max_num_reqs,
            dtype=torch.float32,
            device=device,
        )
        self.seeds = torch.zeros(
            self.max_num_reqs,
            dtype=torch.int64,
            device=device,
        )
        self.idx_mapping = torch.zeros(
            self.max_num_reqs,
            dtype=torch.int32,
            device=device,
        )

        self.model: nn.Module | None = None
        self.draft_logits: torch.Tensor | None = None
        self.supports_mm_inputs = False
        self.cudagraph_manager = None

        # Set by model_runner.__init__ after construction.
        self.ssd_rejection_sampler: SSDRejectionSampler | None = None

    def load_model(self, target_model: nn.Module) -> None:
        if not self.is_speculator:
            return

        target_attn_layer_names = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,
        ).keys()

        from vllm.compilation.backends import set_model_tag

        orig_model_config = self.vllm_config.model_config
        self.vllm_config.model_config = self.draft_model_config
        try:
            with set_model_tag("draft_model"):
                self.model = get_model(
                    vllm_config=self.vllm_config,
                    model_config=self.draft_model_config,
                    prefix="draft_model",
                )
        finally:
            self.vllm_config.model_config = orig_model_config

        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,
        ).keys()
        self.draft_attn_layer_names = set(all_attn_layers) - set(
            target_attn_layer_names
        )

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
    ) -> None:
        if not self.is_speculator:
            return

        self.model_state = model_state
        self.kv_cache_config = kv_cache_config
        _, self.attn_groups, _ = init_attn_backend(
            kv_cache_config,
            self.vllm_config,
            self.device,
            active_layer_names=self.draft_attn_layer_names,
        )
        self.block_tables = block_tables

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        pass

    def capture_model(self) -> None:
        pass

    @torch.inference_mode()
    def run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the standalone draft model forward pass (speculator only)."""
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=batch_descriptor,
        ):
            hidden_states = self.model(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
            )
        return hidden_states

    def generate_drafts(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        pos = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        idx_mapping = self.idx_mapping[:num_reqs]
        for step in range(1, self.num_speculative_steps):
            # Run the eagle model.
            last_hidden_states, hidden_states = self.run_model(
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )
            last_hidden_states = last_hidden_states[:num_reqs]
            hidden_states = hidden_states[:num_reqs]
            logits = self.model.compute_logits(last_hidden_states)

            # NOTE(woosuk): We must add 1 to the positions to match the Gumbel noise
            # used for draft and target sampling.
            draft_tokens = gumbel_sample(
                logits,
                idx_mapping,
                self.temperature,
                self.seeds,
                pos + 1,
                apply_temperature=True,
                processed_logits_out=self.draft_logits[:, step]
                if self.draft_logits is not None
                else None,
            )
            self.draft_tokens[:num_reqs, step] = draft_tokens

            if step < self.num_speculative_steps - 1:
                # Update the inputs for the next step.
                update_draft_inputs(
                    draft_tokens,
                    hidden_states,
                    self.input_buffers,
                    self.hidden_states,
                    self.max_model_len,
                )
                if attn_metadata is not None:
                    self.block_tables.compute_slot_mappings(
                        idx_mapping, query_start_loc, pos, num_tokens_padded
                    )

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        if dummy_run or not self.ssd_comm.enabled:
            # Dummy/profiling/warmup runs are per-engine-core and
            # uncoordinated across DP ranks — skip NCCL to avoid deadlock.
            num_reqs = input_batch.num_reqs
            return self.primary_draft_tokens[:num_reqs]

        if self.is_speculator:
            self._propose_speculator(
                input_batch,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                temperature,
                seeds,
            )
        else:
            return self._propose_base(input_batch, num_sampled)

    # ------------------------------------------------------------------
    # Speculator device: generate K×K drafts + send to bases
    # ------------------------------------------------------------------

    def _propose_speculator(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        num_tokens_across_dp: torch.Tensor | None,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
    ) -> torch.Tensor:
        """Generate K×K draft tokens and send to all base ranks."""
        num_reqs = input_batch.num_reqs
        num_tokens = input_batch.num_tokens_after_padding

        self.temperature.copy_(temperature)
        self.seeds.copy_(seeds)
        idx_mapping = self.idx_mapping[:num_reqs]
        idx_mapping.copy_(input_batch.idx_mapping)

        # Draft prefill.
        last_hidden_states = self.run_model(
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp=num_tokens_across_dp,
        )
        sample_hidden_states = last_hidden_states[input_batch.logits_indices]
        logits = self.model.compute_logits(sample_hidden_states)

        positions = input_batch.positions[input_batch.logits_indices]
        self.draft_tokens[:num_reqs, 0, 0] = gumbel_sample(
            logits,
            idx_mapping,
            self.temperature,
            self.seeds,
            positions + 1,
            apply_temperature=True,
        )

        # Round i
        # Base
        # d0 - d1 - d2 - d3
        #                X
        # Speculator
        # d1 - |d2| - d3
        # d2   |d3|   d4
        # d3   |d4|   d5
        # d4   |d5|   d6
        #
        # Round i+1
        # Base
        # d2 - d3 - d4 - d5
        #      X
        # Speculator
        # |d2| - d3 - d4
        # |d3|   d4   d5
        # |d4|   d5   d6
        # |d5|   d6   d7

        # Each request produces exactly 1 token per draft decode step,
        # enabling FULL cudagraph.
        decode_batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
            self.cudagraph_manager,
            num_reqs,
            num_reqs,
            uniform_token_count=1,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
        )

        attn_metadata_updated = None
        slot_mappings_updated = None
        if not (dummy_run and skip_attn_for_dummy_run):
            # Build attention metadata and slot mappings for the draft
            # decode steps. It is necessary to rebuild the attention
            # metadata even when replaying the FULL cudagraph so that
            # any attention metadata builder state is updated.
            slot_mappings = self.block_tables.compute_slot_mappings(
                idx_mapping,
                self.input_buffers.query_start_loc[: num_reqs + 1],
                pos,
                decode_batch_desc.num_tokens,
            )
            slot_mappings_updated = build_slot_mappings_by_layer(
                slot_mappings, self.kv_cache_config
            )
            attn_metadata_updated = self._build_draft_attn_metadata(
                num_reqs=num_reqs,
                num_reqs_padded=decode_batch_desc.num_reqs or num_reqs,
                num_tokens_padded=decode_batch_desc.num_tokens,
                max_query_len=1,
            )

        if decode_batch_desc.cg_mode == CUDAGraphMode.FULL:
            assert self.cudagraph_manager is not None
            self.cudagraph_manager.run_fullgraph(decode_batch_desc)
        else:
            self.generate_drafts(
                num_reqs,
                decode_batch_desc.num_tokens,
                attn_metadata_updated,
                slot_mappings_updated,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=decode_batch_desc.cg_mode,
            )
        return self.draft_tokens[:num_reqs]

        # Phase 2: send K×K draft tokens to all base ranks.
        self.ssd_comm.speculator_send_drafts(self.draft_tokens)

        return self.primary_draft_tokens[:num_reqs]

    # ------------------------------------------------------------------
    # Base device: send num_accepted + receive K×K + select continuations
    # ------------------------------------------------------------------

    def _propose_base(
        self,
        input_batch: InputBatch,
        num_sampled: torch.Tensor,
    ) -> torch.Tensor:
        """Send acceptance, receive K×K drafts, return [num_reqs, K]."""
        num_reqs = input_batch.num_reqs
        K = self.K

        # Phase 1: send num_accepted to speculator.
        self.ssd_comm.base_send_accepted(num_sampled, num_reqs)

        # Phase 2: receive K×K draft tokens from speculator.
        kxk_draft_tokens = self.ssd_comm.base_recv_drafts(num_reqs)

        # Select the K continuations from the last accepted position.
        accepted_pos = torch.clamp(
            num_sampled[:num_reqs] - 1,
            min=0,
            max=K - 1,
        )
        accepted_pos_expanded = (
            accepted_pos.long().unsqueeze(1).unsqueeze(2).expand(-1, 1, K)
        )
        selected = torch.gather(
            kxk_draft_tokens[:num_reqs],
            dim=1,
            index=accepted_pos_expanded,
        ).squeeze(1)  # [num_reqs, K]

        return selected
