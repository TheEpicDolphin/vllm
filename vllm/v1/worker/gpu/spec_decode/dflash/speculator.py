# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig, replace
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    get_kv_cache_spec_sliding_window,
)
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cp_utils import cp_local_slot
from vllm.v1.worker.gpu.dp_utils import DPSyncState, dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager
from vllm.v1.worker.gpu.spec_decode.dflash.utils import load_dflash_model
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    get_eagle3_aux_layers_from_config,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.gpu.spec_decode.utils import get_parallel_drafting_token_id
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


class DFlashSpeculator(DraftModelSpeculator):
    _speculator_name = "DFlash"  # For logging, so we can share methods with subclasses

    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, req_states: RequestState
    ):
        self.req_states = req_states
        parallel_config = vllm_config.parallel_config
        if parallel_config.prefill_context_parallel_size > 1:
            vllm_config = copy.copy(vllm_config)
            vllm_config.parallel_config = replace(
                parallel_config,
                prefill_context_parallel_size=1,
            )
        super().__init__(vllm_config, device)

        self.hidden_states = torch.zeros(
            self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device
        )
        self.aux_hidden_states: torch.Tensor | None = None

        # Multimodal inputs not currently supported.
        self.supports_mm_inputs = False

        # Each request emits exactly (bonus + N mask) query tokens per step.
        self.num_query_per_req = 1 + self.num_speculative_steps

        self.parallel_drafting_token_id = get_parallel_drafting_token_id(
            self.draft_model_config.hf_config
        )

        from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal

        self.requires_non_causal = dflash_has_any_non_causal(
            self.draft_model_config.hf_config
        )

        # Whether the anchor query position is itself a prediction. DFlash default uses
        # the anchor as the bonus token (only mask tokens predict); DSpark samples from
        # the anchor and the N-1 mask token positions. See _prepare_dflash_inputs_kernel
        dflash_config = (
            getattr(self.draft_model_config.hf_config, "dflash_config", None) or {}
        )
        if dflash_config.get("sample_from_anchor", False):
            raise ValueError(
                "sample_from_anchor=True is not supported for DFlash. "
                "DFlash uses a fixed 1+N query layout where the anchor "
                "is the bonus token."
            )
        self.sample_from_anchor = False

        # Context positions for the K/V precompute. Populated by
        # prepare_dflash_inputs, and processed by the model's
        # precompute_and_store_context_kv method. NOT captured by CUDA graphs.
        self.context_positions = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )

        # Per-mask-token sampling buffers. Flattened from (num_reqs, num_spec_tokens).
        max_num_sampled_tokens = self.max_num_reqs * self.num_speculative_steps
        self.sample_indices = torch.zeros(
            max_num_sampled_tokens, dtype=torch.int64, device=device
        )
        self.sample_pos = torch.zeros(
            max_num_sampled_tokens, dtype=torch.int64, device=device
        )
        # -1 marks an inert sampling row. CUDA graph capture can execute the
        # full buffer before a real batch has populated it, so zero would make
        # every padding row scatter into request slot 0.
        self.sample_idx_mapping = torch.full(
            (max_num_sampled_tokens,), -1, dtype=torch.int32, device=device
        )
        # [0, 1, ..., N-1, 0, 1, ..., N-1, ...] -> the per-token column index into
        # draft_logits[req, step, :].
        self.sample_col = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        ).repeat(self.max_num_reqs)

        self.query_cudagraph_manager: DFlashCudaGraphManager | None = None
        self.draft_kv_cache_group_id: int = -1
        # Upper bound on the trailing context rows staged per request for the
        # K/V precompute, or None to stage the full context. Resolved in set_attn.
        self.max_sliding_window: int | None = None

    @property
    def attn_vllm_config(self) -> VllmConfig:
        # The draft's attention differs from the target's in causality.
        config = copy.copy(super().attn_vllm_config)
        config.attention_config = replace(
            self.vllm_config.attention_config,
            use_non_causal=self.requires_non_causal,
        )
        return config

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        wants_full = cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
        supports_full = (
            self.attn_cg_support.min_cg_support.value
            >= AttentionCGSupport.UNIFORM_BATCH.value
        )
        if wants_full and not supports_full:
            logger.warning(
                "%s draft attention (%s) does not support full CUDA graphs; "
                "running the draft eagerly.",
                self._speculator_name,
                self.attn_cg_support.min_cg_attn_backend,
            )
        # PIECEWISE cudagraphs are not supported for dflash.
        if wants_full and supports_full:
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        else:
            cudagraph_mode = CUDAGraphMode.NONE

        self.query_cudagraph_manager = DFlashCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            decode_query_len=self.num_query_per_req,
        )

    def capture(self) -> None:
        logger.info("Capturing model for %s speculator...", self._speculator_name)
        # Padded sample rows must not scatter into a live request during capture.
        self.sample_indices.zero_()
        self.sample_pos.zero_()
        self.sample_idx_mapping.fill_(-1)
        assert self.query_cudagraph_manager is not None
        self.query_cudagraph_manager.capture(
            self._generate_draft,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            self.max_model_len,
            causal=self._group_causal,
            progress_bar_desc=f"Capturing {self._speculator_name.lower()} CUDA graphs",
        )

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        return load_dflash_model(target_model, self.vllm_config)

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        # The target emits one [num_tokens, hidden] aux tensor per layer that
        # set_eagle3_aux_hidden_state_layers resolves, via the same lookup.
        aux_layers = get_eagle3_aux_layers_from_config(self.speculative_config)
        if not aux_layers:
            aux_layers = target_model.get_eagle3_default_aux_hidden_state_layers()
        # Concatenated aux hidden states of the trimmed context, feeding
        # combine_hidden_states.
        self.aux_hidden_states = torch.empty(
            self.max_num_tokens,
            len(aux_layers) * self.vllm_config.model_config.get_hidden_size(),
            dtype=self.dtype,
            device=self.device,
        )

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        target_input_buffers: InputBuffers,
        target_attn_groups: list[list[AttentionGroup]],
    ) -> None:
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )

        # FlashAttention's AOT split schedule is wrong for a windowed drafter,
        # and `_get_sliding_window_configs` leaves it on or off depending on
        # whether the target also runs FlashAttention. Decide it here instead.
        windows: list[int] = []
        has_full_attention = False
        for groups in self.attn_groups:
            for group in groups:
                builder = group.get_metadata_builder()
                window = get_kv_cache_spec_sliding_window(builder.kv_cache_spec)
                if window is None:
                    has_full_attention = True
                    continue
                windows.append(window)
                if getattr(builder, "aot_schedule", False):
                    # `aot_schedule` belongs to FlashAttention's builder, not
                    # to the base class this loop is typed against.
                    builder.aot_schedule = False  # type: ignore[attr-defined]
        self.max_sliding_window = (
            max(windows) + self.num_query_per_req
            if windows and not has_full_attention
            else None
        )
        if envs.VLLM_DFLASH_DISABLE_CONTEXT_TRIM:
            self.max_sliding_window = None
        logger.info_once(
            "%s context K/V precompute trimmed to %s trailing rows per request.",
            self._speculator_name,
            self.max_sliding_window,
        )

        self.draft_kv_cache_group_ids = [
            gid for gid, g in enumerate(self.attn_groups) if g
        ]
        assert self.draft_kv_cache_group_ids, "No draft attention groups found."
        self.draft_kv_cache_group_id = self.draft_kv_cache_group_ids[0]

        # Per-group context slot buffers for the precompute (one row per group).
        self._context_slot_mappings = torch.zeros(
            len(self.draft_kv_cache_group_ids),
            self.max_num_tokens,
            dtype=torch.int64,
            device=self.device,
        )

        # Map each draft decoder layer to the index (within draft_kv_cache_group_ids)
        # of the kv-cache group its cache belongs to. Models that share a single group
        # leave this as None and share one context slot mapping.
        self._layer_group_idx: list[int] | None = None
        # Per-KV-group causal, falling back to whether the drafter is all-causal.
        self._group_causal: dict[int, bool] | bool = not self.requires_non_causal
        if hasattr(self.model, "get_draft_kv_cache_layer_names"):
            layer_names = self.model.get_draft_kv_cache_layer_names()
            name_to_gid = {
                ln: gid
                for gid, group in enumerate(kv_cache_config.kv_cache_groups)
                for ln in group.layer_names
            }
            gid_to_idx = {gid: i for i, gid in enumerate(self.draft_kv_cache_group_ids)}
            self._layer_group_idx = [
                gid_to_idx[name_to_gid[name]] for name in layer_names
            ]
            if hasattr(self.model, "get_draft_attn_causal"):
                self._group_causal = {
                    name_to_gid[name]: layer_causal
                    for name, layer_causal in zip(
                        layer_names, self.model.get_draft_attn_causal()
                    )
                }

    def _copy_context_full(
        self,
        num_target_tokens: int,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
    ) -> None:
        # NOTE: To avoid CPU-GPU synchronization without CPU knowing the
        # number of rejected tokens, we maintain the size of input_ids and
        # hidden_states the same as the target model's. This means, we pad each
        # request's query length to include any rejected positions.
        if aux_hidden_states:
            assert self.aux_hidden_states is not None
            torch.cat(
                [states[:num_target_tokens] for states in aux_hidden_states],
                dim=-1,
                out=self.aux_hidden_states[:num_target_tokens],
            )
            hidden_states = self.model.combine_hidden_states(
                self.aux_hidden_states[:num_target_tokens]
            )
        else:
            hidden_states = last_hidden_states[:num_target_tokens]
        self.hidden_states[:num_target_tokens].copy_(hidden_states)

    def _copy_context_window(
        self,
        input_batch: InputBatch,
        prefill_lens: torch.Tensor,
        num_target_tokens: int,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
    ) -> None:
        assert self.max_sliding_window is not None
        if aux_hidden_states:
            assert self.aux_hidden_states is not None
            for i, layer_hidden_states in enumerate(aux_hidden_states):
                hidden_size = layer_hidden_states.shape[-1]
                _gather_context_tail(
                    self.aux_hidden_states[:, hidden_size * i : hidden_size * (i + 1)],
                    layer_hidden_states,
                    input_batch,
                    prefill_lens,
                    self.max_sliding_window,
                )
            hidden_states = self.model.combine_hidden_states(
                self.aux_hidden_states[:num_target_tokens]
            )
            self.hidden_states[:num_target_tokens].copy_(hidden_states)
        else:
            _gather_context_tail(
                self.hidden_states,
                last_hidden_states,
                input_batch,
                prefill_lens,
                self.max_sliding_window,
            )

    def _copy_context(
        self,
        input_batch: InputBatch,
        prefill_lens: torch.Tensor | None,
        num_target_tokens: int,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
    ) -> int:
        """Stage the target hidden states the context K/V precompute reads and
        return the number of context rows.

        A windowed drafter only reads context within context_tail_len of the
        prompt end: drafts proposed mid-prompt are discarded by the scheduler,
        so a prefill chunk contributes only its rows inside that final window,
        and a chunk ending before it contributes none. prepare_dflash_inputs
        lays out positions and slots by the same rule (_get_context_tail_length),
        computed from device-side inputs; the host mirror below only decides
        whether anything is trimmed and bounds the compact row count. The
        layout is the identity when no span exceeds its tail.
        """
        if self.max_sliding_window is None:
            self._copy_context_full(
                num_target_tokens, last_hidden_states, aux_hidden_states
            )
            return num_target_tokens
        assert prefill_lens is not None

        num_reqs = input_batch.num_reqs
        query_lens = input_batch.num_scheduled_tokens[:num_reqs]
        num_computed_prefill = (
            input_batch.num_computed_prefill_tokens_np[:num_reqs] + query_lens
        )
        window_start = input_batch.prefill_len_np[:num_reqs] - self.max_sliding_window
        context_lens = np.minimum(
            query_lens,
            np.where(
                input_batch.is_prefilling_np[:num_reqs],
                np.clip(num_computed_prefill - window_start, 0, None),
                self.max_sliding_window,
            ),
        )
        if (context_lens == query_lens).all():
            self._copy_context_full(
                num_target_tokens, last_hidden_states, aux_hidden_states
            )
            return num_target_tokens

        # Host upper bound on the compact row count. Adaptive verification
        # reassigns spans on the GPU, so rows past the true total must be
        # inert: PAD slots write no KV and position 0 keeps RoPE in range.
        # prepare_dflash_inputs writes the real rows on top of this fill.
        num_target_tokens = int(context_lens.sum())
        if num_target_tokens > 0:
            self._copy_context_window(
                input_batch,
                prefill_lens,
                num_target_tokens,
                last_hidden_states,
                aux_hidden_states,
            )
        return num_target_tokens

    @torch.inference_mode()
    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> torch.Tensor:
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
            last_hidden_states = self.model(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
                inputs_embeds=None,
            )
        return last_hidden_states

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        last_hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        num_sample = num_reqs * self.num_speculative_steps
        sample_hidden_states = last_hidden_states[self.sample_indices[:num_sample]]
        # sample_pos is the predicted token's position P. Sampling keys a draw
        # by the position before the sampled token, P-1.
        draft_tokens = self.sample_draft(
            sample_hidden_states,
            self.sample_pos[:num_sample] - 1,
            self.sample_idx_mapping[:num_sample],
            self.temperature,
            self.seeds,
            self.sample_col[:num_sample],
            self.draft_logits,
        )
        self.draft_tokens[:num_reqs] = draft_tokens.view(
            num_reqs, self.num_speculative_steps
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
        dp_sync: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        num_target_tokens = input_batch.num_tokens
        num_query_tokens = num_reqs * self.num_query_per_req
        max_seq_len = input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()
        self.draft_max_seq_len = min(
            max_seq_len + self.num_query_per_req, self.max_model_len
        )

        if dummy_run and skip_attn_for_dummy_run:
            # Memory profiling path: block_tables / kv_cache_config are not initialized.
            # Since DFlash needs to build its own attention metadata, we must skip the
            # preparation in this path and run a minimal forward pass.
            self._copy_context_full(
                num_target_tokens, last_hidden_states, aux_hidden_states
            )
            self.model.precompute_and_store_context_kv(
                self.hidden_states[:num_target_tokens],
                self.context_positions[:num_target_tokens],
            )
            # DFlash processes all speculative tokens in one forward pass,
            # so the real token count is num_query_tokens.
            self._prepare_eplb_forward(num_query_tokens)
            self._generate_draft(
                num_reqs,
                num_query_tokens,
                attn_metadata=None,
                slot_mappings=None,
                num_tokens_across_dp=None,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            return self.draft_tokens[:num_reqs]

        if self.pcp_manager is not None and not dummy_run:
            self.block_tables.gather_block_tables(
                input_batch.idx_mapping, num_reqs_padded=num_reqs
            )

        prefill_len = self.req_states.prefill_len.gpu
        num_target_tokens = self._copy_context(
            input_batch,
            prefill_len,
            num_target_tokens,
            last_hidden_states,
            aux_hidden_states,
        )
        self.context_positions[:num_target_tokens].zero_()
        self._context_slot_mappings[:, :num_target_tokens].fill_(PAD_SLOT_ID)

        # The query slot mapping is written into the shared BlockTables slot_mappings.
        # That buffer's address is what the captured CUDA graph reads from at replay.
        assert self.draft_kv_cache_group_id >= 0
        # Support multiple draft KV cache groups by preparing inputs once for each
        for i, gid in enumerate(self.draft_kv_cache_group_ids):
            prepare_dflash_inputs(
                self.input_buffers,
                self.block_tables.slot_mappings[gid],
                self.context_positions,
                self._context_slot_mappings[i],
                self.sample_indices,
                self.sample_pos,
                self.sample_idx_mapping,
                self.temperature,
                self.seeds,
                input_batch,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                temperature,
                seeds,
                self.block_tables.input_block_tables[gid],
                self.block_tables.kernel_block_sizes[gid],
                self.block_tables.cp_rank,
                self.block_tables.cp_size,
                self.block_tables.cp_interleave,
                self.parallel_drafting_token_id,
                self.num_query_per_req,
                self.num_speculative_steps,
                self.max_num_reqs,
                self.max_num_tokens,
                self.max_model_len,
                self.sample_from_anchor,
                prefill_len=prefill_len,
                context_tail_len=self.max_sliding_window,
            )

        # Pre-insert context K/V into the cache. Runs eagerly outside the captured graph
        # because the context shape varies per step. During dummy runs the block tables
        # are placeholders, so we skip the cache write to avoid clobbering real entries.
        # Each layer uses the context slots of its own kv-cache group.
        if dummy_run:
            context_slots: torch.Tensor | list[torch.Tensor | None] | None = None
        elif self._layer_group_idx is not None:
            context_slots = [
                self._context_slot_mappings[gidx][:num_target_tokens]
                for gidx in self._layer_group_idx
            ]
        else:
            context_slots = self._context_slot_mappings[0][:num_target_tokens]
        if num_target_tokens > 0:
            self.model.precompute_and_store_context_kv(
                self.hidden_states[:num_target_tokens],
                self.context_positions[:num_target_tokens],
                context_slots,
            )

        batch_sync, num_batch_tokens = (
            self._build_uniform_batch_dp_sync(dp_sync, num_reqs, self.num_query_per_req)
            if dp_sync is not None
            else (None, num_query_tokens)
        )
        # Every DFlash step has exactly num_query_per_req tokens, so we can use FULL CGs
        batch_desc, batch_sync = dispatch_cg_and_sync_dp(
            self.query_cudagraph_manager,
            num_reqs,
            num_batch_tokens,
            uniform_token_count=self.num_query_per_req,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
            dp_sync=batch_sync,
        )
        num_tokens_padded = batch_desc.num_tokens
        num_tokens_across_dp = (
            batch_sync.num_tokens_across_dp if batch_sync is not None else None
        )

        # Rebuild the draft attention metadata even when replaying the FULL
        # graph so that any attention metadata builder state is updated.
        draft_attn_metadata = self._build_uniform_attn_metadata(
            num_reqs=num_reqs,
            batch_desc=batch_desc,
            num_query_per_req=self.num_query_per_req,
            seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
            step=self.num_query_per_req,
            causal=self._group_causal,
        )
        draft_slot_mappings_by_layer = build_slot_mappings_by_layer(
            self.block_tables.slot_mappings[:, :num_tokens_padded],
            self.kv_cache_config,
        )

        # DFlash processes all speculative tokens in one forward pass,
        # so the real token count is num_query_tokens.
        self._prepare_eplb_forward(num_query_tokens)

        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            assert self.query_cudagraph_manager is not None
            self.query_cudagraph_manager.run_fullgraph(batch_desc)
        else:
            self._generate_draft(
                num_reqs,
                num_tokens_padded,
                draft_attn_metadata,
                draft_slot_mappings_by_layer,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=batch_desc.cg_mode,
            )

        return self.draft_tokens[:num_reqs]


# Requests summed per iteration when deriving a request's compact row offset.
_COMPACT_BLOCK_R = 256


@triton.jit
def _context_tail_length(
    query_start_loc_ptr,
    positions_ptr,
    idx_mapping_ptr,
    prefill_lens_ptr,
    req_idx,
    req_mask,
    max_context_tail_len,
):
    query_start = tl.load(query_start_loc_ptr + req_idx, mask=req_mask, other=0)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1, mask=req_mask, other=0)
    query_len = query_end - query_start
    first_pos = tl.load(positions_ptr + query_start, mask=req_mask, other=0).to(
        tl.int32
    )
    req_state = tl.load(idx_mapping_ptr + req_idx, mask=req_mask, other=0)
    prefill_len = tl.load(prefill_lens_ptr + req_state, mask=req_mask, other=0)
    window_rows = first_pos + query_len - (prefill_len - max_context_tail_len)
    tail_cap = tl.where(
        first_pos < prefill_len, tl.maximum(window_rows, 0), max_context_tail_len
    )
    return tl.minimum(query_len, tail_cap)


@triton.jit
def _cumulative_context_tail_length(
    query_start_loc_ptr,
    positions_ptr,
    idx_mapping_ptr,
    prefill_len_ptr,
    req_idx,
    max_context_tail_len,
    BLOCK_R: tl.constexpr,
):
    acc = tl.zeros((BLOCK_R,), dtype=tl.int32)
    for start in range(0, req_idx, BLOCK_R):
        req_block = start + tl.arange(0, BLOCK_R)
        req_mask = req_block < req_idx
        acc += _context_tail_length(
            query_start_loc_ptr,
            positions_ptr,
            idx_mapping_ptr,
            prefill_len_ptr,
            req_block,
            req_mask,
            max_context_tail_len,
        )
    return tl.sum(acc)


@triton.jit
def _gather_context_tail_kernel(
    out_hidden_states_ptr,
    out_hidden_states_stride,
    hidden_states_ptr,
    hidden_states_stride,
    query_start_loc_ptr,
    positions_ptr,
    idx_mapping_ptr,
    prefill_lens_ptr,
    max_context_tail_len,
    hidden_size,
    BLOCK_Q: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    req_idx = tl.program_id(0)
    query_block_idx = tl.program_id(1)
    dim_block_idx = tl.program_id(2)
    ctx_tail_len = _context_tail_length(
        query_start_loc_ptr,
        positions_ptr,
        idx_mapping_ptr,
        prefill_lens_ptr,
        req_idx,
        req_idx >= 0,
        max_context_tail_len,
    )
    ctx_src_start = tl.load(query_start_loc_ptr + req_idx + 1) - ctx_tail_len
    ctx_dst_start = _cumulative_context_tail_length(
        query_start_loc_ptr,
        positions_ptr,
        idx_mapping_ptr,
        prefill_lens_ptr,
        req_idx,
        max_context_tail_len,
        BLOCK_R,
    )
    query_block = query_block_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    dim_block = dim_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = (query_block < ctx_tail_len)[:, None] & (dim_block < hidden_size)[None, :]
    hidden_states = tl.load(
        hidden_states_ptr
        + (ctx_src_start + query_block).to(tl.int64)[:, None] * hidden_states_stride
        + dim_block[None, :],
        mask=mask,
    )
    tl.store(
        out_hidden_states_ptr
        + (ctx_dst_start + query_block).to(tl.int64)[:, None] * out_hidden_states_stride
        + dim_block[None, :],
        hidden_states,
        mask=mask,
    )


def _gather_context_tail(
    out_hidden_states: torch.Tensor,
    hidden_states: torch.Tensor,
    input_batch: InputBatch,
    prefill_len: torch.Tensor,
    max_context_tail_len: int,
) -> None:
    hidden_size = hidden_states.shape[1]
    query_block_size = 16
    hidden_block_size = 256
    grid = (
        input_batch.num_reqs,
        triton.cdiv(max_context_tail_len, query_block_size),
        triton.cdiv(hidden_size, hidden_block_size),
    )
    _gather_context_tail_kernel[grid](
        out_hidden_states,
        out_hidden_states.stride(0),
        hidden_states,
        hidden_states.stride(0),
        input_batch.query_start_loc,
        input_batch.positions,
        input_batch.idx_mapping,
        prefill_len,
        max_context_tail_len,
        hidden_size,
        BLOCK_Q=query_block_size,
        BLOCK_H=hidden_block_size,
        BLOCK_R=_COMPACT_BLOCK_R,
    )


@triton.jit
def _prepare_dflash_inputs_kernel(
    # Outputs
    out_input_ids_ptr,
    out_query_positions_ptr,
    out_query_start_loc_ptr,
    out_seq_lens_ptr,
    out_query_slot_mapping_ptr,
    out_context_positions_ptr,
    out_context_slot_mapping_ptr,
    out_sample_indices_ptr,
    out_sample_pos_ptr,
    out_sample_idx_mapping_ptr,
    out_temperature_ptr,
    out_seeds_ptr,
    # Inputs from target batch
    target_positions_ptr,
    target_query_start_loc_ptr,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    # Sampling params
    temperature_ptr,
    seeds_ptr,
    # Block table for slot mapping lookup.
    block_table_ptr,
    block_table_stride,
    # Scalars
    parallel_drafting_token_id,
    block_size,
    num_query_per_req,
    num_speculative_steps,
    max_num_reqs,
    max_num_tokens,
    max_model_len,
    cp_rank,
    prefill_lens_ptr,
    max_context_tail_len,
    SAMPLE_FROM_ANCHOR: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
    CP_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TRIM_CONTEXT: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    num_reqs = tl.num_programs(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)

    ctx_start = tl.load(target_query_start_loc_ptr + req_idx)
    ctx_end = tl.load(target_query_start_loc_ptr + req_idx + 1)
    num_ctx = ctx_end - ctx_start
    if TRIM_CONTEXT:
        # A windowed drafter only reads context near the prompt end, so each
        # request stores only that tail of its span, packed after the tails of
        # the requests before it. The hidden-state gather derives the same
        # layout from the same inputs.
        ctx_tail_len = _context_tail_length(
            target_query_start_loc_ptr,
            target_positions_ptr,
            idx_mapping_ptr,
            prefill_lens_ptr,
            req_idx,
            req_idx >= 0,
            max_context_tail_len,
        )
        ctx_dst_start = _cumulative_context_tail_length(
            target_query_start_loc_ptr,
            target_positions_ptr,
            idx_mapping_ptr,
            prefill_lens_ptr,
            req_idx,
            max_context_tail_len,
            BLOCK_R,
        )
    else:
        ctx_tail_len = num_ctx
        ctx_dst_start = ctx_start
    num_trimmed = num_ctx - ctx_tail_len

    num_rejected = tl.load(num_rejected_ptr + req_idx)
    valid_ctx_end = ctx_end - num_rejected
    num_valid_ctx = valid_ctx_end - ctx_start

    num_sampled = tl.load(num_sampled_ptr + req_idx)
    if num_sampled > 0:
        bonus_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        # Chunked prefilling: splice in the next prefill token.
        bonus_token = tl.load(next_prefill_tokens_ptr + req_state_idx).to(tl.int32)

    last_valid_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)
    query_base = req_idx * num_query_per_req

    j = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    is_ctx = j < num_ctx
    is_valid_ctx = j < num_valid_ctx
    is_query = (j >= num_valid_ctx) & (j < num_valid_ctx + num_query_per_req)
    query_off = j - num_valid_ctx

    # --- Context positions / slots ---
    ctx_pos_idx = ctx_start + tl.where(is_ctx, j, 0)
    ctx_pos = tl.load(target_positions_ptr + ctx_pos_idx, mask=is_valid_ctx, other=0)
    ctx_block_num = ctx_pos // (block_size * CP_SIZE)
    ctx_block_num = tl.minimum(ctx_block_num, block_table_stride - 1)
    ctx_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + ctx_block_num,
        mask=is_valid_ctx,
        other=0,
    ).to(tl.int64)
    # Block 0 is the null block. Old sliding-window context positions can map
    # to it after eviction; rejected suffix rows are invalid context as well.
    # Neither kind of row may write draft KV into physical block 0.
    ctx_resident = is_valid_ctx & (ctx_block_id != 0)
    local_ctx_slot = cp_local_slot(
        ctx_pos, ctx_block_id, block_size, cp_rank, CP_SIZE, CP_INTERLEAVE, PAD_SLOT_ID
    )
    ctx_slot = tl.where(
        ctx_resident,
        local_ctx_slot,
        PAD_SLOT_ID,
    )
    # Stored over the kept tail of the span while the loads above are masked to
    # [0, num_valid_ctx): the rejected suffix rows (always inside the tail) get
    # position 0 and PAD_SLOT_ID. That is intentional — those rows write no KV
    # and their positions are never consumed, but the tail must stay fully
    # initialized so a stale value from an earlier batch is never observed.
    is_kept = is_ctx & (j >= num_trimmed)
    ctx_dst = ctx_dst_start + j - num_trimmed
    tl.store(out_context_positions_ptr + ctx_dst, ctx_pos, mask=is_kept)
    tl.store(out_context_slot_mapping_ptr + ctx_dst, ctx_slot, mask=is_kept)

    # --- Query positions / input_ids / slots ---
    query_pos = last_valid_pos + 1 + query_off
    query_idx = query_base + query_off
    is_bonus = is_query & (query_off == 0)
    input_id = tl.where(is_bonus, bonus_token, parallel_drafting_token_id)

    q_block_num = query_pos // (block_size * CP_SIZE)
    q_block_num = tl.minimum(q_block_num, block_table_stride - 1)
    q_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + q_block_num,
        mask=is_query,
        other=0,
    ).to(tl.int64)
    # A null block is never a writable cache slot. This can occur when a
    # sliding-window block table contains evicted/global padding entries.
    q_resident = is_query & (q_block_id != 0)
    local_q_slot = cp_local_slot(
        query_pos,
        q_block_id,
        block_size,
        cp_rank,
        CP_SIZE,
        CP_INTERLEAVE,
        PAD_SLOT_ID,
    )
    q_slot = tl.where(
        q_resident,
        local_q_slot,
        PAD_SLOT_ID,
    )

    tl.store(out_input_ids_ptr + query_idx, input_id, mask=is_query)
    clamped_query_pos = tl.minimum(query_pos, max_model_len - 1)
    tl.store(out_query_positions_ptr + query_idx, clamped_query_pos, mask=is_query)
    tl.store(out_query_slot_mapping_ptr + query_idx, q_slot, mask=is_query)

    # --- Sample indices / positions / idx_mapping ---
    # When SAMPLE_FROM_ANCHOR (DSpark), so we sample at EVERY query position
    # and each position k predicts the NEXT token (sampled position = query_pos + 1).
    # Otherwise (DFlash default) the anchor is the bonus token and only the mask tokens
    # at offsets > 0 are sampled from, each AT its own position.
    sample_off = 0 if SAMPLE_FROM_ANCHOR else 1
    is_sample = is_query & (query_off >= sample_off)
    sample_idx = req_idx * num_speculative_steps + (query_off - sample_off)
    sample_pos = query_pos + 1 if SAMPLE_FROM_ANCHOR else query_pos
    tl.store(out_sample_indices_ptr + sample_idx, query_idx, mask=is_sample)
    tl.store(out_sample_pos_ptr + sample_idx, sample_pos, mask=is_sample)
    tl.store(out_sample_idx_mapping_ptr + sample_idx, req_state_idx, mask=is_sample)

    if block_idx == 0:
        tl.store(out_query_start_loc_ptr + req_idx, query_base)
        # seq_lens is the absolute sequence length the draft attention
        # reads up to (context + query), not just the count of accepted
        # tokens this step.
        tl.store(
            out_seq_lens_ptr + req_idx,
            tl.minimum(last_valid_pos + 1 + num_query_per_req, max_model_len),
        )
        # Copy sampling state.
        tl.store(
            out_temperature_ptr + req_state_idx,
            tl.load(temperature_ptr + req_state_idx),
        )
        tl.store(out_seeds_ptr + req_state_idx, tl.load(seeds_ptr + req_state_idx))
        if req_idx == num_reqs - 1:
            # Pad per-request buffers to max_num_reqs for CUDA graph safety.
            last_query_end = num_reqs * num_query_per_req
            for i in range(num_reqs, max_num_reqs + 1, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_reqs + 1
                tl.store(out_query_start_loc_ptr + block, last_query_end, mask=mask)
            for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_reqs
                tl.store(out_seq_lens_ptr + block, 0, mask=mask)
            # Padded sample slots point at query index 0 (a valid row in
            # last_hidden_states) so CG replay never reads OOB. Padded
            # sample idx mappings point to -1, which is ignored during
            # sampling to prevent writing stale values to draft logits.
            pad_start = num_reqs * num_speculative_steps
            pad_end = max_num_reqs * num_speculative_steps
            for i in range(pad_start, pad_end, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < pad_end
                tl.store(out_sample_indices_ptr + block, 0, mask=mask)
                tl.store(out_sample_pos_ptr + block, 0, mask=mask)
                tl.store(out_sample_idx_mapping_ptr + block, -1, mask=mask)
            # Pad query slot mappings past num_query_tokens with PAD so the
            # captured CG sees PAD slots (no K/V write) for replay sizes
            # larger than the current request count.
            q_pad_start = num_reqs * num_query_per_req
            for i in range(q_pad_start, max_num_tokens, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_tokens
                tl.store(out_query_slot_mapping_ptr + block, PAD_SLOT_ID, mask=mask)


def prepare_dflash_inputs(
    input_buffers: InputBuffers,
    query_slot_mapping: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor,
    sample_indices: torch.Tensor,
    sample_pos: torch.Tensor,
    sample_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    input_batch: InputBatch,
    # [num_reqs]
    num_sampled: torch.Tensor,
    # [num_reqs]
    num_rejected: torch.Tensor,
    # [max_num_reqs]
    last_sampled: torch.Tensor,
    # [max_num_reqs]
    next_prefill_tokens: torch.Tensor,
    # [max_num_reqs]
    input_temperature: torch.Tensor,
    # [max_num_reqs]
    input_seeds: torch.Tensor,
    # [max_num_reqs, max_num_blocks]
    block_table: torch.Tensor,
    block_size: int,
    cp_rank: int,
    cp_size: int,
    cp_interleave: int,
    parallel_drafting_token_id: int,
    num_query_per_req: int,
    num_speculative_steps: int,
    max_num_reqs: int,
    max_num_tokens: int,
    max_model_len: int,
    sample_from_anchor: bool = False,
    prefill_len: torch.Tensor | None = None,
    context_tail_len: int | None = None,
) -> None:
    num_reqs = input_batch.num_reqs
    assert num_reqs > 0
    trim_context = context_tail_len is not None
    assert not trim_context or prefill_len is not None
    # Cover the longest possible per-request span (ctx + query). Use the max
    # per-request query length, not the total token count across the batch.
    max_target_query_len = int(input_batch.num_scheduled_tokens.max())
    max_tokens_per_req = max_target_query_len + num_query_per_req
    BLOCK_SIZE = min(256, triton.next_power_of_2(max(1, max_tokens_per_req)))
    num_blocks = triton.cdiv(max_tokens_per_req, BLOCK_SIZE)
    _prepare_dflash_inputs_kernel[(num_reqs, num_blocks)](
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        query_slot_mapping,
        context_positions,
        context_slot_mapping,
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        temperature,
        seeds,
        input_batch.positions,
        input_batch.query_start_loc,
        input_batch.idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        input_temperature,
        input_seeds,
        block_table,
        block_table.stride(0),
        parallel_drafting_token_id,
        block_size,
        num_query_per_req,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        max_model_len,
        cp_rank,
        # Never dereferenced unless TRIM_CONTEXT; any valid pointer will do.
        prefill_len if prefill_len is not None else input_batch.query_start_loc,
        context_tail_len or 0,
        SAMPLE_FROM_ANCHOR=sample_from_anchor,
        PAD_SLOT_ID=PAD_SLOT_ID,
        CP_SIZE=cp_size,
        CP_INTERLEAVE=cp_interleave,
        BLOCK_SIZE=BLOCK_SIZE,
        TRIM_CONTEXT=trim_context,
        BLOCK_R=_COMPACT_BLOCK_R,
    )
