# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
    init_attn_backend,
)
from vllm.v1.worker.gpu.block_table import (
    BlockTables,
    _compute_slot_id,
    _load_ptr,
)
from vllm.v1.worker.gpu.cudagraph_utils import (
    get_uniform_token_count,
)
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.eagle.cudagraph import (
    EagleCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model

logger = init_logger(__name__)


class EagleSpeculator:
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.device = device

        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None
        self.method = self.speculative_config.method
        self.num_speculative_steps = self.speculative_config.num_speculative_tokens
        self.draft_model_config = self.speculative_config.draft_model_config

        self.scheduler_config = vllm_config.scheduler_config
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        # We need to get the hidden size from the draft model config because
        # the draft model's hidden size can be different from the target model's
        # hidden size (e.g., Llama 3.3 70B).
        self.hidden_size = self.draft_model_config.get_hidden_size()
        self.vocab_size = self.draft_model_config.get_vocab_size()
        self.dtype = vllm_config.model_config.dtype

        # DP configuration
        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank

        self.prefill_input_buffers = InputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=device,
        )
        self.decode_input_buffers: InputBuffers | None = None
        self.decode_slot_mappings: torch.Tensor | None = None
        if self.num_speculative_steps > 1:
            # Decode input buffers are only needed when drafting more
            # than one token.
            self.decode_input_buffers = InputBuffers(
                max_num_reqs=self.max_num_reqs,
                max_num_tokens=self.max_num_reqs,
                device=device,
            )
        self.hidden_states = torch.zeros(
            self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device
        )
        self.idx_mapping = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self.temperature = torch.zeros(
            self.max_num_reqs, dtype=torch.float32, device=device
        )
        self.seeds = torch.zeros(self.max_num_reqs, dtype=torch.int64, device=device)
        self.draft_tokens = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )
        self.last_token_indices = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )

        self.supports_mm_inputs = MULTIMODAL_REGISTRY.supports_multimodal_inputs(
            self.draft_model_config
        )
        if self.supports_mm_inputs:
            self.inputs_embeds = torch.zeros(
                self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device
            )

        self.draft_logits: torch.Tensor | None = None
        if self.speculative_config.rejection_sample_method == "probabilistic":
            self.draft_logits = torch.zeros(
                self.max_num_reqs,
                self.num_speculative_steps,
                self.vocab_size,
                dtype=torch.float32,
                device=device,
            )

        self.prefill_cudagraph_manager: EagleCudaGraphManager | None = None
        self.decode_cudagraph_manager: EagleCudaGraphManager | None = None

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        cudagraph_mode = self.vllm_config.compilation_config.cudagraph_mode
        # Initialize cudagraph manager for draft prefill (draft position 0).
        self.prefill_cudagraph_manager = EagleCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            self.num_speculative_steps + 1,
        )

        # PIECEWISE cudagraphs are not supported for eagle draft decodes.
        # PIECEWISE pads num_tokens to the next capture size without padding
        # num_reqs, which can cause attention backends to read past the
        # valid per-request metadata (e.g. FlashInfer's kv_indptr buffer).
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        else:
            cudagraph_mode = CUDAGraphMode.NONE

        # Initialize cudagraph manager for draft decodes (draft positions > 0).
        self.decode_cudagraph_manager = EagleCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            decode_query_len=1,
        )
        # Share a single pool between prefill and decode since they never
        # execute concurrently.
        self.decode_cudagraph_manager.pool = self.prefill_cudagraph_manager.pool

    def load_model(self, target_model: nn.Module) -> None:
        target_attn_layer_names = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        ).keys()

        self.model = load_eagle_model(target_model, self.vllm_config)

        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
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
        self.model_state = model_state
        self.kv_cache_config = kv_cache_config
        _, self.attn_groups, _ = init_attn_backend(
            kv_cache_config,
            self.vllm_config,
            self.device,
            active_layer_names=self.draft_attn_layer_names,
        )
        self.block_tables = block_tables

        # Dedicated slot mappings tensor for decode steps so that writing
        # decode slot mappings in the fused kernel doesn't corrupt the
        # shared block_tables.slot_mappings used by the prefill attention.
        if self.num_speculative_steps > 1:
            self.decode_slot_mappings = torch.full(
                block_tables.slot_mappings.shape,
                PAD_SLOT_ID,
                dtype=torch.int64,
                device=self.device,
            )

    @torch.inference_mode()
    def run_model(
        self,
        input_buffers: InputBuffers,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=batch_descriptor,
        ):
            inputs_embeds = None
            if self.supports_mm_inputs:
                # Merge multimodal embeddings with input ids.
                mm_embeds, is_mm_embed = mm_inputs or (None, None)
                num_input_tokens = (
                    is_mm_embed.shape[0] if is_mm_embed is not None else num_tokens
                )
                self.inputs_embeds[:num_input_tokens] = self.model.embed_input_ids(
                    input_buffers.input_ids[:num_input_tokens],
                    multimodal_embeddings=mm_embeds,
                    is_multimodal=is_mm_embed,
                )
                inputs_embeds = self.inputs_embeds[:num_tokens]

            ret_hidden_states = self.model(
                input_ids=input_buffers.input_ids[:num_tokens],
                positions=input_buffers.positions[:num_tokens],
                hidden_states=self.hidden_states[:num_tokens],
                inputs_embeds=inputs_embeds,
            )
        if self.method == "mtp":
            last_hidden_states = ret_hidden_states
            hidden_states = ret_hidden_states
        else:
            last_hidden_states, hidden_states = ret_hidden_states
        return last_hidden_states, hidden_states

    def prefill(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        last_token_indices = self.last_token_indices[:num_reqs]
        pos = self.prefill_input_buffers.positions[last_token_indices]
        idx_mapping = self.idx_mapping[:num_reqs]

        last_hidden_states, hidden_states = self.run_model(
            self.prefill_input_buffers,
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            mm_inputs=mm_inputs,
        )
        sample_hidden_states = last_hidden_states[last_token_indices]
        logits = self.model.compute_logits(sample_hidden_states)

        # NOTE(woosuk): We must add 1 to the positions to match the Gumbel noise
        # used for draft and target sampling.
        self.draft_tokens[:num_reqs, 0] = gumbel_sample(
            logits,
            idx_mapping,
            self.temperature,
            self.seeds,
            pos + 1,
            apply_temperature=True,
            processed_logits_out=self.draft_logits[:, 0]
            if self.draft_logits is not None
            else None,
        )
        self.hidden_states[:num_reqs] = hidden_states[last_token_indices]

    def generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        assert self.decode_input_buffers is not None
        pos = self.decode_input_buffers.positions[:num_reqs]
        idx_mapping = self.idx_mapping[:num_reqs]
        for step in range(1, self.num_speculative_steps):
            last_hidden_states, hidden_states = self.run_model(
                self.decode_input_buffers,
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
                update_eagle_decode_inputs(
                    draft_tokens,
                    hidden_states,
                    self.decode_input_buffers,
                    self.hidden_states,
                    self.max_model_len,
                    idx_mapping,
                    self.block_tables if attn_metadata is not None else None,
                )

    def _build_draft_attn_metadata(
        self,
        num_reqs: int,
        num_reqs_padded: int,
        num_tokens_padded: int,
        max_query_len: int,
        input_buffers: InputBuffers,
    ) -> dict[str, Any] | None:
        if not self.draft_attn_layer_names:
            return None

        query_start_loc_cpu = (
            torch.arange(num_reqs_padded + 1, dtype=torch.int32, device="cpu").clamp_(
                max=num_reqs
            )
            * max_query_len
        )
        block_tables = [
            x[:num_reqs_padded] for x in self.block_tables.input_block_tables
        ]
        slot_mappings = self.block_tables.slot_mappings[:, :num_tokens_padded]
        attn_metadata = build_attn_metadata(
            attn_groups=self.attn_groups,
            num_reqs=num_reqs_padded,
            num_tokens=num_tokens_padded,
            query_start_loc_gpu=input_buffers.query_start_loc[: num_reqs_padded + 1],
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_buffers.seq_lens[:num_reqs_padded],
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=self.kv_cache_config,
        )
        return attn_metadata

    def capture_model(self) -> None:
        logger.info("Capturing model for Eagle speculator...")
        # Reset indices to zeros to prevent stale values from prior
        # dummy runs to cause out-of-bounds indexing during capture.
        self.last_token_indices.zero_()

        # Capture the prefill routine (model forward + compute_logits +
        # gumbel_sample).
        # For FULL graphs, the entire routine is recorded as one graph.
        # For PIECEWISE, only the model's compiled regions are captured
        # and the rest (compute_logits, gumbel_sample) runs eagerly.
        assert self.prefill_cudagraph_manager is not None
        self.prefill_cudagraph_manager.capture(
            self.prefill,
            self.model_state,
            self.prefill_input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            progress_bar_desc="Capturing eagle prefill CUDA graphs",
        )

        if self.num_speculative_steps == 1:
            return

        # Capture the decode draft generation loop (model forward +
        # compute_logits + gumbel_sample + update_eagle_inputs, for
        # each step). For FULL graphs, the entire multi-step loop is
        # recorded as one graph.
        assert self.decode_cudagraph_manager is not None
        assert self.decode_input_buffers is not None
        self.decode_cudagraph_manager.capture(
            self.generate_draft,
            self.model_state,
            self.decode_input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            progress_bar_desc="Capturing eagle decode CUDA graphs",
        )

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        # [num_tokens, hidden_size]
        last_hidden_states: torch.Tensor,
        # num_layers x [num_tokens, hidden_size]
        aux_hidden_states: list[torch.Tensor] | None,
        # [num_reqs]
        num_sampled: torch.Tensor,
        # [num_reqs]
        num_rejected: torch.Tensor,
        # [max_num_reqs]
        last_sampled: torch.Tensor,
        # [max_num_reqs]
        next_prefill_tokens: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        num_tokens = input_batch.num_tokens_after_padding
        num_reqs = input_batch.num_reqs
        max_query_len = input_batch.num_scheduled_tokens.max()

        # NOTE(woosuk): To avoid CPU-GPU synchronization without CPU knowing the
        # number of rejected tokens, we maintain the size of eagle's input_ids and
        # hidden_states the same as the target model's. This means, we pad each
        # request's query length to include any rejected positions. By doing so,
        # we can also reuse the attention metadata (e.g., query_start_loc,
        # seq_lens) of the target model.
        if aux_hidden_states:
            assert self.method == "eagle3"
            hidden_states = self.model.combine_hidden_states(
                torch.cat(aux_hidden_states, dim=-1)
            )
        else:
            hidden_states = last_hidden_states
        self.hidden_states[:num_tokens].copy_(hidden_states)

        # Prepare all eagle inputs in a single fused kernel:
        # 1. Prefill input_ids/positions/query_start_loc/seq_lens
        # 2. Temperature, seeds, idx_mapping copies
        # 3. Decode positions/seq_lens/query_start_loc (precomputed)
        # 4. First decode step slot mappings
        # Steps 3 & 4 are compiled out when num_speculative_steps <= 1.
        prepare_eagle_inputs(
            self.last_token_indices,
            self.prefill_input_buffers,
            self.decode_input_buffers,
            self.idx_mapping,
            self.temperature,
            self.seeds,
            input_batch,
            temperature,
            seeds,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            self.block_tables if not (dummy_run and skip_attn_for_dummy_run) else None,
            self.decode_slot_mappings,
            self.max_model_len,
            self.max_num_reqs,
            self.num_speculative_steps,
        )

        # When all requests are decoding (no true prefills), each has
        # num_speculative_steps + 1 tokens, enabling FULL graph replay.
        # Mixed or prefill-only batches fall back to PIECEWISE.
        prefill_batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
            self.prefill_cudagraph_manager,
            num_reqs,
            num_tokens,
            get_uniform_token_count(num_reqs, num_tokens, max_query_len),
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
        )

        if prefill_batch_desc.cg_mode == CUDAGraphMode.FULL:
            # It is necessary to rebuild the attention metadata when
            # replaying the FULL graph so that any attention metadata
            # builder state is updated.
            self._build_draft_attn_metadata(
                num_reqs=num_reqs,
                num_reqs_padded=prefill_batch_desc.num_reqs or num_reqs,
                num_tokens_padded=prefill_batch_desc.num_tokens,
                max_query_len=self.num_speculative_steps + 1,
                input_buffers=self.prefill_input_buffers,
            )
            # Replay the full graph for draft prefill.
            assert self.prefill_cudagraph_manager is not None
            self.prefill_cudagraph_manager.run_fullgraph(prefill_batch_desc)
        else:
            # The target model's attention metadata and slot mappings
            # can directly be used for draft prefill, because of the
            # identical batch shape and KV cache layout.
            self.prefill(
                num_reqs,
                prefill_batch_desc.num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=prefill_batch_desc.cg_mode,
                mm_inputs=mm_inputs,
            )

        if self.num_speculative_steps == 1:
            return self.draft_tokens[:num_reqs, :1]

        assert self.decode_input_buffers is not None
        # Copy the first draft token into decode input_ids.
        self.decode_input_buffers.input_ids[:num_reqs].copy_(
            self.draft_tokens[:num_reqs, 0].to(torch.int32)
        )

        # Each request produces exactly 1 token per draft generation step,
        # enabling FULL graph replay.
        decode_batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
            self.decode_cudagraph_manager,
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
            assert self.decode_slot_mappings is not None
            # Copy decode slot mappings (computed by the fused kernel into
            # a dedicated buffer) back into block_tables.slot_mappings now
            # that the prefill is done and no longer reading from it.
            num_decode_tokens = decode_batch_desc.num_tokens
            self.block_tables.slot_mappings[:, :num_decode_tokens].copy_(
                self.decode_slot_mappings[:, :num_decode_tokens]
            )
            slot_mappings_updated = build_slot_mappings_by_layer(
                self.block_tables.slot_mappings[:, :num_decode_tokens],
                self.kv_cache_config,
            )
            attn_metadata_updated = self._build_draft_attn_metadata(
                num_reqs=num_reqs,
                num_reqs_padded=decode_batch_desc.num_reqs or num_reqs,
                num_tokens_padded=num_decode_tokens,
                max_query_len=1,
                input_buffers=self.decode_input_buffers,
            )

        if decode_batch_desc.cg_mode == CUDAGraphMode.FULL:
            # Replay the full graph for draft generation.
            assert self.decode_cudagraph_manager is not None
            self.decode_cudagraph_manager.run_fullgraph(decode_batch_desc)
        else:
            self.generate_draft(
                num_reqs,
                decode_batch_desc.num_tokens,
                attn_metadata_updated,
                slot_mappings_updated,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=decode_batch_desc.cg_mode,
            )
        return self.draft_tokens[:num_reqs]


@triton.jit
def _store_slot_mappings(
    position,
    req_state_idx,
    req_idx,
    block_table_ptrs,
    block_table_strides,
    block_sizes,
    slot_mappings_ptr,
    slot_mappings_stride,
    num_kv_cache_groups,
    cp_rank,
    CP_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    PAD_ID: tl.constexpr,
):
    """Compute and store slot mappings for a single request across all KV cache
    groups.  No-op when num_kv_cache_groups == 0."""
    for group_id in range(num_kv_cache_groups):
        block_table_ptr = _load_ptr(block_table_ptrs + group_id, tl.int32)
        block_table_stride = tl.load(block_table_strides + group_id)
        block_size = tl.load(block_sizes + group_id)
        slot_id = _compute_slot_id(
            position,
            req_state_idx,
            block_table_ptr,
            block_table_stride,
            block_size,
            cp_rank,
            CP_SIZE,
            CP_INTERLEAVE,
            PAD_ID,
        )
        tl.store(
            slot_mappings_ptr + group_id * slot_mappings_stride + req_idx,
            slot_id,
        )


@triton.jit
def _prepare_eagle_inputs_kernel(
    # Prefill outputs
    last_token_indices_ptr,
    prefill_input_ids_ptr,
    prefill_positions_ptr,
    prefill_query_start_loc_ptr,
    prefill_seq_lens_ptr,
    # Decode outputs (used when NUM_SPEC_STEPS > 1)
    decode_positions_ptr,
    decode_query_start_loc_ptr,
    decode_seq_lens_ptr,
    # Slot mapping outputs
    slot_mappings_ptr,
    slot_mappings_stride,
    # Shared outputs
    eagle_idx_mapping_ptr,
    eagle_temperature_ptr,
    eagle_seeds_ptr,
    # Prefill sources
    target_input_ids_ptr,
    target_positions_ptr,
    idx_mapping_ptr,
    temperature_ptr,
    seeds_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    # Block table params (used when NUM_SPEC_STEPS > 1)
    block_table_ptrs,
    block_table_strides,
    block_sizes,
    num_kv_cache_groups,
    cp_rank,
    # Shared scalars
    max_model_len,
    max_num_reqs,
    max_num_batched_tokens,
    # Constexprs
    NUM_SPEC_STEPS: tl.constexpr,
    CP_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    PAD_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_reqs = tl.num_programs(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    temperature = tl.load(temperature_ptr + req_idx)
    seed = tl.load(seeds_ptr + req_idx)

    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + req_idx)

    num_rejected = tl.load(num_rejected_ptr + req_idx)
    query_len -= num_rejected

    num_sampled = tl.load(num_sampled_ptr + req_idx)
    if num_sampled > 0:
        next_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        next_token = tl.load(next_prefill_tokens_ptr + req_state_idx)

    # Shift target_input_ids by one.
    for i in range(1, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        input_ids = tl.load(target_input_ids_ptr + query_start + block, mask=mask)
        tl.store(prefill_input_ids_ptr + query_start + block - 1, input_ids, mask=mask)

    last_token_index = query_start + query_len - 1
    tl.store(last_token_indices_ptr + req_idx, last_token_index)
    tl.store(prefill_input_ids_ptr + last_token_index, next_token)

    # Copy positions.
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        target_pos = tl.load(target_positions_ptr + query_start + block, mask=mask)
        tl.store(prefill_positions_ptr + query_start + block, target_pos, mask=mask)

    tl.store(prefill_query_start_loc_ptr + req_idx, query_start)
    tl.store(prefill_seq_lens_ptr + req_idx, seq_len)

    # NOTE(woosuk): For draft sampling, we only consider the temperature
    # and ignore the other sampling parameters such as top_k and top_p,
    # for simplicity and performance.
    tl.store(eagle_temperature_ptr + req_idx, temperature)
    tl.store(eagle_seeds_ptr + req_idx, seed)
    tl.store(eagle_idx_mapping_ptr + req_idx, req_state_idx)

    # Precompute decode inputs + slot mappings.
    if NUM_SPEC_STEPS > 1:
        last_pos = tl.load(target_positions_ptr + last_token_index)
        decode_pos = tl.minimum(last_pos + 1, max_model_len - 1)
        tl.store(decode_positions_ptr + req_idx, decode_pos)

        decode_seq_len = tl.minimum(seq_len - num_rejected + 1, max_model_len)
        tl.store(decode_query_start_loc_ptr + req_idx, req_idx)
        tl.store(decode_seq_lens_ptr + req_idx, decode_seq_len)

        # Compute slot mappings for the first decode step.
        # When block_tables is None (no draft attention layers),
        # num_kv_cache_groups is 0 so this is a no-op.
        _store_slot_mappings(
            decode_pos,
            req_state_idx,
            req_idx,
            block_table_ptrs,
            block_table_strides,
            block_sizes,
            slot_mappings_ptr,
            slot_mappings_stride,
            num_kv_cache_groups,
            cp_rank,
            CP_SIZE,
            CP_INTERLEAVE,
            PAD_ID,
        )

    # Padding for cudagraph (handled by the last program).
    if req_idx == (num_reqs - 1):
        # Pad prefill buffers.
        for i in range(num_reqs, max_num_reqs + 1, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs + 1
            tl.store(prefill_query_start_loc_ptr + block, query_end, mask=mask)
        for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(prefill_seq_lens_ptr + block, 0, mask=mask)
        for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(last_token_indices_ptr + block, 0, mask=mask)

        if NUM_SPEC_STEPS > 1:
            # Pad decode buffers.
            for i in range(num_reqs, max_num_reqs + 1, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_reqs + 1
                tl.store(decode_query_start_loc_ptr + block, num_reqs, mask=mask)
            for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_reqs
                tl.store(decode_seq_lens_ptr + block, 0, mask=mask)
            for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_reqs
                tl.store(decode_positions_ptr + block, 0, mask=mask)
            # Pad slot mappings.
            for group_id in range(num_kv_cache_groups):
                sm_ptr = slot_mappings_ptr + group_id * slot_mappings_stride
                for i in range(num_reqs, max_num_batched_tokens, BLOCK_SIZE):
                    block = i + tl.arange(0, BLOCK_SIZE)
                    mask = block < max_num_batched_tokens
                    tl.store(sm_ptr + block, PAD_ID, mask=mask)


def _get_block_table_args(
    block_tables: BlockTables | None,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    int,
    int,
]:
    if block_tables is None:
        return (
            torch.zeros(1, dtype=torch.uint64, device=device),
            torch.zeros(1, dtype=torch.int64, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
            0,  # num_kv_cache_groups
            0,  # cp_rank
            1,  # cp_size
            1,  # cp_interleave
        )

    return (
        block_tables.block_table_ptrs,
        block_tables.block_table_strides,
        block_tables.block_sizes_tensor,
        block_tables.num_kv_cache_groups,
        block_tables.cp_rank,
        block_tables.cp_size,
        block_tables.cp_interleave,
    )


def prepare_eagle_inputs(
    last_token_indices: torch.Tensor,
    prefill_input_buffers: InputBuffers,
    decode_input_buffers: InputBuffers | None,
    idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    input_batch: InputBatch,
    input_temperature: torch.Tensor,
    input_seeds: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    block_tables: BlockTables | None,
    decode_slot_mappings: torch.Tensor | None,
    max_model_len: int,
    max_num_reqs: int,
    num_speculative_steps: int,
) -> torch.Tensor:
    num_reqs = input_batch.num_reqs

    (
        block_table_ptrs,
        block_table_strides,
        block_sizes,
        num_kv_cache_groups,
        cp_rank,
        cp_size,
        cp_interleave,
    ) = _get_block_table_args(block_tables, idx_mapping.device)

    # When decode_input_buffers is None (num_speculative_steps == 1),
    # the decode path is compiled out, so these pointers are never
    # accessed. The prefill buffers are used as safe dummies.
    decode_buffers = decode_input_buffers or prefill_input_buffers
    if decode_slot_mappings is None:
        decode_slot_mappings = torch.zeros(
            1, dtype=torch.int64, device=idx_mapping.device
        )
    max_num_batched_tokens = decode_slot_mappings.shape[-1]

    _prepare_eagle_inputs_kernel[(num_reqs,)](
        last_token_indices,
        prefill_input_buffers.input_ids,
        prefill_input_buffers.positions,
        prefill_input_buffers.query_start_loc,
        prefill_input_buffers.seq_lens,
        decode_buffers.positions,
        decode_buffers.query_start_loc,
        decode_buffers.seq_lens,
        decode_slot_mappings,
        decode_slot_mappings.stride(0),
        idx_mapping,
        temperature,
        seeds,
        input_batch.input_ids,
        input_batch.positions,
        input_batch.idx_mapping,
        input_temperature,
        input_seeds,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        input_batch.query_start_loc,
        input_batch.seq_lens,
        block_table_ptrs,
        block_table_strides,
        block_sizes,
        num_kv_cache_groups,
        cp_rank,
        max_model_len,
        max_num_reqs,
        max_num_batched_tokens,
        NUM_SPEC_STEPS=num_speculative_steps,
        CP_SIZE=cp_size,
        CP_INTERLEAVE=cp_interleave,
        PAD_ID=PAD_SLOT_ID,
        BLOCK_SIZE=1024,
    )
    return last_token_indices


@triton.jit
def _update_eagle_decode_inputs_kernel(
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
    # Slot mapping params
    idx_mapping_ptr,
    block_table_ptrs,
    block_table_strides,
    block_sizes,
    slot_mappings_ptr,
    slot_mappings_stride,
    num_kv_cache_groups,
    cp_rank,
    CP_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    PAD_ID: tl.constexpr,
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

    # Compute slot mappings for the updated position.
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    _store_slot_mappings(
        position,
        req_state_idx,
        req_idx,
        block_table_ptrs,
        block_table_strides,
        block_sizes,
        slot_mappings_ptr,
        slot_mappings_stride,
        num_kv_cache_groups,
        cp_rank,
        CP_SIZE,
        CP_INTERLEAVE,
        PAD_ID,
    )


def update_eagle_decode_inputs(
    draft_tokens: torch.Tensor,
    output_hidden_states: torch.Tensor,
    input_buffers: InputBuffers,
    hidden_states: torch.Tensor,
    max_model_len: int,
    idx_mapping: torch.Tensor,
    block_tables: BlockTables | None,
):
    num_reqs, hidden_size = output_hidden_states.shape

    (
        block_table_ptrs,
        block_table_strides,
        block_sizes,
        num_kv_cache_groups,
        cp_rank,
        cp_size,
        cp_interleave,
    ) = _get_block_table_args(block_tables, idx_mapping.device)

    if block_tables is not None:
        slot_mappings = block_tables.slot_mappings
    else:
        slot_mappings = torch.zeros(1, dtype=torch.int64, device=idx_mapping.device)

    _update_eagle_decode_inputs_kernel[(num_reqs,)](
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
        idx_mapping,
        block_table_ptrs,
        block_table_strides,
        block_sizes,
        slot_mappings,
        slot_mappings.stride(0),
        num_kv_cache_groups,
        cp_rank,
        CP_SIZE=cp_size,
        CP_INTERLEAVE=cp_interleave,
        PAD_ID=PAD_SLOT_ID,
        BLOCK_SIZE=1024,
    )
