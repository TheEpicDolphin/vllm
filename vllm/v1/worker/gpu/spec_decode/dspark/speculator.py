# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculator: semi-autoregressive parallel drafting.

DSpark drafts a block of ``num_speculative_tokens`` tokens in one parallel pass
(reusing the DFlash machinery: context-KV precompute + a query-block forward),
then injects intra-block dependency with a lightweight sequential Markov head.

Differences from DFlash:
  * Anchor-as-first-prediction: each request emits exactly ``N =
    num_speculative_tokens`` query tokens (anchor + N-1 noise), NOT ``1 + N``.
    Every query position is a prediction (the anchor predicts the first draft
    token), so we sample at all N positions and ``sample_pos = query_pos + 1``
    (standard next-token), whereas DFlash's masks sit AT the predicted position.
    This is the ``sample_from_anchor`` path in the shared prepare-inputs kernel.
    Speculators-format checkpoints instead use the DFlash ``1 + N`` fill-in
    layout (anchor is the bonus token).
  * Sequential Markov sampling: instead of DFlash's single parallel sample, we
    sample left-to-right, adding a prefix-dependent Markov bias derived from the
    previously sampled token at each step.

CUDA graphs (FULL, mirroring DFlash) cover the whole draft step: the parallel
backbone forward AND the sequential Markov sampling.

Under batch-sharded sampling, large enough FULL graph steps also shard the
sequential sampling across TP ranks: each rank runs the Markov loop only for
the requests it owns (the same owners as verification), see
``_sample_sequential_sharded``.
"""

from functools import partial
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.triton_utils import triton
from vllm.utils.math_utils import cdiv
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.batch_shard import (
    build_fixed_shard_plan,
    max_num_reqs_per_rank,
)
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model

logger = init_logger(__name__)

# Smallest padded batch whose draft step shards the sequential sampling. Below
# it, the extra all-to-all costs more latency than the smaller loop saves.
_MIN_SHARDED_LOOP_REQS = 16


class _ShardedLoopCudaGraphManager(DFlashCudaGraphManager):
    """Captures the batch-sharded variant of the DSpark draft step.

    It captures only batches of at least ``min_num_reqs`` requests. Dispatch
    with the replicated manager, then replay here if the descriptor was
    captured.
    """

    def __init__(self, *args: Any, min_num_reqs: int, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._capture_descs = {
            mode: [d for d in descs if (d.num_reqs or 0) >= min_num_reqs]
            for mode, descs in self._capture_descs.items()
        }


class DSparkSpeculator(DFlashSpeculator):
    _speculator_name = "DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        # Whether to sample from the anchor position. When True, uses anchor-as-first
        # (N slots, each position predicts the next token). When False, uses 1+N
        # fill-in block (anchor is a bonus token).
        self.sample_from_anchor = getattr(
            self.draft_model_config.hf_config, "sample_from_anchor", True
        )
        if self.sample_from_anchor:
            self.num_query_per_req = self.num_speculative_steps
        else:
            self.num_query_per_req = 1 + self.num_speculative_steps

        # DSpark consumes mean-pooled target aux hidden states at the target
        # layers, combined to hidden_size via main_proj. Store that combined
        # main_x (hidden_size wide). DSpark does not use the same pre-allocated buffer
        # that DeepSeek-V4's MTP uses.
        draft_hidden = self.draft_model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            self.max_num_tokens, draft_hidden, dtype=self.dtype, device=device
        )

        self._step_cols = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        )

        self._anchor_idx = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )

        # Reduced-vocab probabilistic drafting only; set in load_draft_model.
        self._d2t_scatter_index: torch.Tensor | None = None
        self._draft_scatter_buf: torch.Tensor | None = None
        self._draft_topk: int | None = getattr(
            self.draft_model_config.hf_config, "dspark_draft_topk", None
        )

        self.use_confidence_head: bool = False

        # Set when the sequential sampling can be batch-sharded across TP.
        self._loop_tp_size = 1
        self._loop_tp_rank = 0
        self._loop_pynccl: PyNcclCommunicator | None = None
        self._padded_max_num_reqs = triton.next_power_of_2(self.max_num_reqs)
        self.sharded_loop_cudagraph_manager: _ShardedLoopCudaGraphManager | None = None

    def supports_sharded_draft_logits(self) -> bool:
        # A model's own sample_sequential kernels write the full-size cache.
        return not hasattr(self.model, "sample_sequential")

    def enable_batch_sharded_sampling(self, tp_size: int, tp_rank: int) -> bool:
        sharded_draft_logits = super().enable_batch_sharded_sampling(tp_size, tp_rank)
        if tp_size == 1:
            return sharded_draft_logits
        lm_head = getattr(self.model, "lm_head", None)
        pynccl = getattr(get_tp_group().device_communicator, "pynccl_comm", None)
        reason = None
        if self.draft_logits is not None and not sharded_draft_logits:
            reason = "the draft logits cache is replicated"
        elif self.use_acceptance_estimator:
            reason = (
                "the online acceptance estimator fits on every request's "
                "draft logits on every rank"
            )
        elif self._draft_topk is not None:
            reason = "top-k Markov sampling is not supported"
        elif hasattr(self.model, "sample_sequential"):
            reason = "the draft model fuses its own sequential sampling"
        elif not hasattr(self.model, "compute_draft_logits_local"):
            reason = "the draft model lacks compute_draft_logits_local"
        elif getattr(lm_head, "tp_size", None) != tp_size:
            reason = "the draft lm_head is not vocab-sharded over the TP group"
        elif pynccl is None or pynccl.disabled:
            reason = "PyNCCL is unavailable"
        if reason is not None:
            logger.info(
                "DSpark sequential sampling stays replicated across TP ranks: %s.",
                reason,
            )
            return sharded_draft_logits
        self._loop_tp_size = tp_size
        self._loop_tp_rank = tp_rank
        self._loop_pynccl = pynccl
        logger.info(
            "DSpark sequential sampling is batch-sharded across %d TP ranks for "
            "CUDA graph batches of at least %d requests.",
            tp_size,
            _MIN_SHARDED_LOOP_REQS,
        )
        return sharded_draft_logits

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        super().init_cudagraph_manager(cudagraph_mode)
        self.sharded_loop_cudagraph_manager = None
        assert self.query_cudagraph_manager is not None
        if self._loop_tp_size == 1 or not self.query_cudagraph_manager.cudagraph_mode:
            return
        self.sharded_loop_cudagraph_manager = _ShardedLoopCudaGraphManager(
            self.vllm_config,
            self.device,
            self.query_cudagraph_manager.cudagraph_mode,
            decode_query_len=self.num_query_per_req,
            min_num_reqs=_MIN_SHARDED_LOOP_REQS,
        )

    def capture(self) -> None:
        super().capture()
        if self.sharded_loop_cudagraph_manager is not None:
            self._capture_graphs(
                self.sharded_loop_cudagraph_manager,
                partial(self._generate_draft, sharded=True),
                "Capturing dspark batch-sharded CUDA graphs",
            )

    def _shard_capacity(self, num_reqs: int) -> int:
        """Requests each rank can own in a sharded step of num_reqs requests."""
        per_rank = cdiv(num_reqs, self._loop_tp_size)
        # Slot-based ownership is not perfectly balanced; allow 25% headroom.
        return min(num_reqs, per_rank + cdiv(per_rank, 4))

    def _run_fullgraph(
        self, batch_desc: BatchExecutionDescriptor, input_batch: InputBatch
    ) -> None:
        manager = self.sharded_loop_cudagraph_manager
        # Every TP rank sees the same idx_mapping, so they all pick the same
        # graph and their collectives line up.
        assert batch_desc.num_reqs is not None
        if (
            manager is None
            or batch_desc not in manager.graphs
            or max_num_reqs_per_rank(input_batch.idx_mapping_np, self._loop_tp_size)
            > self._shard_capacity(batch_desc.num_reqs)
        ):
            super()._run_fullgraph(batch_desc, input_batch)
            return
        manager.run_fullgraph(batch_desc)

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        # Reduced draft vocab: probabilistic rejection sampling indexes draft
        # logits by target id, so precompute the draft->target column map and a
        # scratch buffer to scatter logits into target vocab before sampling.
        if self.draft_logits is not None and model.draft_id_to_target_id is not None:
            d2t = model.draft_id_to_target_id
            self._d2t_scatter_index = (
                torch.arange(d2t.shape[0], device=d2t.device) + d2t
            )
            # -inf once; the per-step scatter overwrites the draft->target
            # columns. Kept separate from draft_logits to avoid aliasing.
            self._draft_scatter_buf = torch.full(
                (self.max_num_reqs, self.vocab_size),
                float("-inf"),
                dtype=self.draft_logits.dtype,
                device=self.device,
            )
        self.use_confidence_head = (
            self.enable_adaptive_verification
            and model.model.confidence_head is not None
        )
        if self.use_confidence_head:
            # The acceptance estimator is not needed when a trained confidence head
            # is available.
            self.use_acceptance_estimator = False
        return model

    def _sample_logits(
        self,
        logits: torch.Tensor,
        idx_map: torch.Tensor,
        sample_pos: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        self._maybe_predict_acceptance(logits, idx_map, self._step_cols[step])
        if self.draft_logits is None:
            draft_ids = logits.argmax(dim=-1)
            return self.model.map_draft_to_target(draft_ids)

        # Probabilistic sampling and rejection operate in target-vocabulary
        # space. A reduced draft vocabulary is scattered into its target rows.
        if self._d2t_scatter_index is not None:
            assert self._draft_scatter_buf is not None
            buf = self._draft_scatter_buf[: logits.shape[0]]
            buf.index_copy_(1, self._d2t_scatter_index, logits.to(buf.dtype))
            logits = buf

        # sample_pos is the predicted token's position P. Sampling keys a draw
        # by the position before the sampled token, P-1.
        if self.draft_watermarker is not None:
            # Watermarked drafts keep a replicated draft logits cache.
            return self.draft_watermarker.sample(
                logits,
                idx_map,
                self.temperature,
                self.seeds,
                sample_pos - 1,
                apply_temperature=True,
                is_drafting=True,
                logits_cache=self.draft_logits,
                logits_cache_col=self._step_cols[step],
                use_fp64=self.use_fp64_gumbel,
            )
        return gumbel_sample(
            logits,
            idx_map,
            self.temperature,
            self.seeds,
            sample_pos - 1,
            apply_temperature=True,
            is_drafting=True,
            logits_cache=self.draft_logits,
            logits_cache_col=self._step_cols[step],
            use_fp64=self.use_fp64_gumbel,
            logits_cache_tp_size=self.draft_logits_tp_size,
            logits_cache_tp_rank=self.draft_logits_tp_rank,
        )

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        if self._draft_topk is not None:
            self._sample_sequential_topk(num_reqs, head_hidden)
            return
        # A draft model can run the Markov steps below with its own kernels.
        # It returns False, having done nothing, for a step that the loop
        # below must run.
        sample_sequential = getattr(self.model, "sample_sequential", None)
        if sample_sequential is not None and sample_sequential(
            self, num_reqs, head_hidden
        ):
            return

        # Sequential Markov sampling over the backbone's output hidden states.
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        # Per-(req, position) head hidden, ordered (req, step).
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        # Draft-vocab logits; sampled ids are remapped to target vocab below.
        base_logits = self.model.compute_draft_logits(sample_hidden)
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)

        # Anchor (bonus) token per request = the input id at query offset 0,
        # read via the precomputed persistent index (fixed buffer for capture).
        markov_embeds = self._sample_rows(
            base_logits,
            self.input_buffers.input_ids[self._anchor_idx[:num_reqs]],
            self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec),
            self.sample_pos[:num_sample].view(num_reqs, n_spec),
            self.draft_tokens[:num_reqs],
        )

        if self.use_confidence_head:
            confidence = self.model.compute_confidence(
                sample_hidden, torch.stack(markov_embeds, dim=1).flatten(0, 1)
            )
            self.draft_token_confidence_probs[:num_reqs] = confidence.view(
                num_reqs, n_spec
            )

    def _sample_rows(
        self,
        base_logits: torch.Tensor,
        prev: torch.Tensor,
        idx_map: torch.Tensor,
        sample_pos: torch.Tensor,
        out_tokens: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Sample each row's draft tokens left to right.

        Args:
            base_logits: [R, n_spec, V] backbone draft-vocab logits.
            prev: [R] anchor token of each row.
            idx_map: [R, n_spec] request state index, -1 for an inert row.
            sample_pos: [R, n_spec] predicted token positions.
            out_tokens: [R, n_spec] receives the sampled target-vocab ids.

        Returns:
            The per-step Markov embeddings if the confidence head needs them.

        """
        markov_embeds = []
        for i in range(self.num_speculative_steps):
            # Sequential stage: Markov bias from the previously sampled token.
            markov_embed = self.model.markov_embed(prev)
            if self.use_confidence_head:
                markov_embeds.append(markov_embed)
            bias = self.model.markov_bias(markov_embed)
            logits_i = base_logits[:, i] + bias
            if hasattr(self.model, "apply_output_logit_transform"):
                logits_i = self.model.apply_output_logit_transform(logits_i)
            draft_sampled_i = self._sample_logits(
                logits_i, idx_map[:, i], sample_pos[:, i], i
            )
            out_tokens[:, i] = draft_sampled_i
            prev = draft_sampled_i
        return markov_embeds

    def _sample_sequential_sharded(
        self, num_reqs: int, head_hidden: torch.Tensor
    ) -> None:
        """_sample_sequential with each TP rank sampling only its own requests.

        Each rank computes vocab-shard base logits for every request, in
        owner-sorted order with a fixed number of rows per owner. An all-to-all
        then gives each rank the full-vocab rows of the requests it owns, the
        Markov loop runs on just those rows, and an all-gather returns the
        drafts in batch order. All shapes depend only on num_reqs, so the step
        can be captured; the host checks that no rank owns more than the
        capacity before replaying it.
        """
        assert self._loop_pynccl is not None
        tp_size, tp_rank = self._loop_tp_size, self._loop_tp_rank
        n_spec = self.num_speculative_steps
        capacity = self._shard_capacity(num_reqs)
        idx_map = self.sample_idx_mapping[: num_reqs * n_spec].view(num_reqs, n_spec)
        send_reqs, gathered_src_indices = build_fixed_shard_plan(
            idx_map[:, 0].contiguous(), capacity, tp_size, self._padded_max_num_reqs
        )

        # Empty slots read request 0's rows; nothing they produce is kept.
        send_rows = send_reqs.clamp(min=0).view(-1, 1) * n_spec + self._step_cols
        send_hidden = head_hidden[self.sample_indices[send_rows.flatten()]]
        local_logits = self.model.compute_draft_logits_local(send_hidden).contiguous()
        recv_logits = torch.empty_like(local_logits)
        self._loop_pynccl.all_to_all(recv_logits, local_logits)
        vocab_shard_size = recv_logits.shape[-1]
        base_logits = (
            recv_logits.view(tp_size, capacity * n_spec, vocab_shard_size)
            .permute(1, 0, 2)
            .reshape(capacity, n_spec, tp_size * vocab_shard_size)
        )[..., : self.model.logits_processor.org_vocab_size]

        local_reqs = send_reqs.view(tp_size, capacity)[tp_rank]
        local_rows = send_rows.view(tp_size, capacity, n_spec)[tp_rank]
        local_tokens = self.draft_tokens.new_empty(capacity, n_spec)
        markov_embeds = self._sample_rows(
            base_logits,
            self.input_buffers.input_ids[self._anchor_idx[local_reqs.clamp(min=0)]],
            # Empty slots are inert: no draft logits cache writes.
            torch.where(
                local_reqs.view(-1, 1) >= 0, self.sample_idx_mapping[local_rows], -1
            ),
            self.sample_pos[local_rows],
            local_tokens,
        )

        # Pack the drafts (and fp32 confidences, bitcast) for one all-gather.
        packed = local_tokens
        if self.use_confidence_head:
            local_hidden = send_hidden.view(tp_size, capacity * n_spec, -1)[tp_rank]
            confidence = self.model.compute_confidence(
                local_hidden, torch.stack(markov_embeds, dim=1).flatten(0, 1)
            ).view(capacity, n_spec)
            packed = torch.cat(
                [local_tokens, confidence.float().view(torch.int32).long()], dim=1
            )
        gathered = packed.new_empty(tp_size * capacity, packed.shape[1])
        self._loop_pynccl.all_gather(gathered, packed.contiguous())
        rows = gathered[gathered_src_indices]
        self.draft_tokens[:num_reqs] = rows[:, :n_spec]
        if self.use_confidence_head:
            self.draft_token_confidence_probs[:num_reqs] = (
                rows[:, n_spec:].int().view(torch.float32)
            )

    def _sample_sequential_topk(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        """Apply the sequential Markov head only to top-k base-logit candidates.

        Candidate selection is done once for all draft positions. At each
        sequential step, the selected logits are corrected in place and every
        other entry is set to ``-inf``. The normal dense sampling and rejection
        paths then consume that truncated distribution unchanged.
        """
        assert self._draft_topk is not None
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        base_logits = self.model.compute_draft_logits(sample_hidden)
        base_logits = base_logits.view(num_reqs, n_spec, -1)
        base_values, draft_indices = base_logits.topk(self._draft_topk, dim=-1)
        # Reuse the dense backbone output as the normal sampler's input. Fill
        # once for all positions, then scatter only the corrected candidates
        # during the sequential loop.
        base_logits.fill_(float("-inf"))
        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        confidence_markov_embeds = []
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        for i in range(n_spec):
            markov_embed = self.model.markov_embed(prev)
            if self.use_confidence_head:
                confidence_markov_embeds.append(markov_embed)
            logits_i = self.model.apply_markov_bias_gathered(
                markov_embed,
                base_logits[:, i],
                base_values[:, i],
                draft_indices[:, i],
            )
            draft_sampled_i = self._sample_logits(
                logits_i, idx_map[:, i], sample_pos[:, i], i
            )
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

        if self.use_confidence_head:
            confidence = self.model.compute_confidence(
                sample_hidden,
                torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
            )
            self.draft_token_confidence_probs[:num_reqs] = confidence.view(
                num_reqs, n_spec
            )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        sharded: bool = False,
    ) -> None:
        # Full draft step (captured under CUDA graph): parallel backbone forward
        # then sequential Markov sampling over its hidden state outputs.
        head_hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        if sharded:
            self._sample_sequential_sharded(num_reqs, head_hidden)
        else:
            self._sample_sequential(num_reqs, head_hidden)
