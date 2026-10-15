# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EAGLE-4 Llama drafter.

Same layout as EAGLE-3 plus a low-rank candidate LM head: on each TP rank a
rank-r projection scores the rank's vocab shard and its top ``num_candidates``
tokens get exact logits from the ``lm_head`` shard. The ranks' candidates are
gathered and the best ``num_candidates`` by exact logit are kept. The
sliding-window draft layer needs no changes here; ``LlamaAttention`` reads it
from ``layer_types`` / ``sliding_window``.
"""

import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm.model_executor.layers.logits_processor import _topk
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.models.common.ops.sequence_parallel import sp_all_gather
from vllm.triton_utils import tl, triton

from .utils import maybe_prefix


@triton.jit
def _compute_topk_logits(
    out_logits_ptr,
    out_token_ids_ptr,
    out_stride,
    hidden_states_ptr,
    hidden_states_stride,
    topk_indices_ptr,
    topk_indices_stride,
    lm_head_ptr,
    lm_head_stride,
    hidden_size,
    vocab_start,
    num_candidates,
    HIDDEN_BLOCK_SIZE: tl.constexpr,
    CANDIDATE_BLOCK_SIZE: tl.constexpr,
    TOP_TOKENS: tl.constexpr,
    ROUND_DTYPE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    candidate_block_idx = tl.program_id(1)
    c_block = candidate_block_idx * CANDIDATE_BLOCK_SIZE + tl.arange(
        0, CANDIDATE_BLOCK_SIZE
    )
    c_mask = c_block < num_candidates
    # [c_block], rows of this rank's lm_head shard
    local_rows = tl.load(
        topk_indices_ptr + token_idx * topk_indices_stride + c_block,
        mask=c_mask,
        other=0,
    ).to(tl.int64)

    # [c_block, h_block]
    acc = tl.zeros((CANDIDATE_BLOCK_SIZE, HIDDEN_BLOCK_SIZE), dtype=tl.float32)
    for i in range(0, hidden_size, HIDDEN_BLOCK_SIZE):
        h_block = i + tl.arange(0, HIDDEN_BLOCK_SIZE)
        h_mask = h_block < hidden_size
        # [h_block]
        hidden_states = tl.load(
            hidden_states_ptr + token_idx * hidden_states_stride + h_block,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        # [c_block, h_block]
        topk_lm_head = tl.load(
            lm_head_ptr + local_rows[:, None] * lm_head_stride + h_block[None, :],
            mask=c_mask[:, None] & h_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += topk_lm_head * hidden_states[None, :]
    topk_logits = tl.sum(acc, axis=1)
    out_logits = out_logits_ptr + token_idx * out_stride
    out_token_ids = out_token_ids_ptr + token_idx * out_stride
    token_ids = local_rows + vocab_start
    if TOP_TOKENS:
        # This block's best (logit, token id), the logit rounded to the head
        # dtype and ties to the lowest id as an argmax over compute_logits'
        # output.
        logits = tl.where(
            c_mask, topk_logits.to(ROUND_DTYPE).to(tl.float32), float("-inf")
        )
        best = tl.max(logits, axis=0)
        best_id = tl.min(tl.where(logits == best, token_ids, 1 << 62), axis=0)
        tl.store(out_logits + candidate_block_idx, best)
        tl.store(out_token_ids + candidate_block_idx, best_id.to(tl.float32))
    else:
        tl.store(out_logits + c_block, topk_logits, mask=c_mask)
        # float32 holds the token ids exactly.
        tl.store(out_token_ids + c_block, token_ids.to(tl.float32), mask=c_mask)


@triton.jit
def _merge_candidates(
    logits_ptr,
    logits_stride,
    candidate_logits_ptr,
    candidate_token_ids_ptr,
    candidate_rank_stride,
    candidate_row_stride,
    num_candidates,
    total_num_candidates,
    BLOCK_SIZE: tl.constexpr,
):
    # Keep the num_candidates best of every rank's candidates (ties to the
    # lowest gathered position) and scatter them into the -inf logits row.
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_num_candidates
    candidate_offsets = (
        (offsets // num_candidates) * candidate_rank_stride
        + row * candidate_row_stride
        + offsets % num_candidates
    )
    candidate_logits = tl.load(
        candidate_logits_ptr + candidate_offsets, mask=mask, other=float("-inf")
    )
    token_ids = tl.load(
        candidate_token_ids_ptr + candidate_offsets, mask=mask, other=0.0
    ).to(tl.int64)

    sorted_logits = tl.sort(candidate_logits, descending=True)
    threshold = tl.max(
        tl.where(offsets == num_candidates - 1, sorted_logits, float("-inf")), axis=0
    )
    above = candidate_logits > threshold
    tied = mask & (candidate_logits == threshold)
    num_above = tl.sum(above.to(tl.int32), axis=0)
    tie_rank = tl.cumsum(tied.to(tl.int32), axis=0) - 1
    keep = above | (tied & (tie_rank < num_candidates - num_above))
    tl.store(
        logits_ptr + row * logits_stride + token_ids,
        candidate_logits.to(logits_ptr.dtype.element_ty),
        mask=keep,
    )


@triton.jit
def _select_top_token(
    top_tokens_ptr,
    block_logits_ptr,
    block_token_ids_ptr,
    block_rank_stride,
    block_row_stride,
    num_blocks,
    total_num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    # The best of every rank's per-block (logit, token id), ties to the
    # lowest id.
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_num_blocks
    block_offsets = (
        (offsets // num_blocks) * block_rank_stride
        + row * block_row_stride
        + offsets % num_blocks
    )
    logits = tl.load(block_logits_ptr + block_offsets, mask=mask, other=float("-inf"))
    token_ids = tl.load(block_token_ids_ptr + block_offsets, mask=mask, other=0.0).to(
        tl.int64
    )
    best = tl.max(logits, axis=0)
    token_id = tl.min(tl.where(mask & (logits == best), token_ids, 1 << 62), axis=0)
    tl.store(top_tokens_ptr + row, token_id)


_CANDIDATE_BLOCK_SIZE = 8

_TL_DTYPES = {
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
    torch.float32: tl.float32,
}


class Eagle4LlamaForCausalLM(Eagle3LlamaForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        head_config = getattr(self.config, "efficient_lm_head_config", None) or {}
        if head_config.get("type", "low_rank") != "low_rank":
            raise ValueError(
                f"Unsupported EAGLE-4 efficient_lm_head_config: {head_config}"
            )
        self.num_candidates: int = head_config.get("num_candidates", 0)
        if not self.num_candidates:
            # No candidate head: plain EAGLE-3 logits.
            return

        self.head_dtype = vllm_config.model_config.head_dtype
        rank = head_config["rank"]
        self.efficient_lm_head_down_proj = ReplicatedLinear(
            self.config.hidden_size,
            rank,
            bias=False,
            quant_config=None,
            prefix=maybe_prefix(prefix, "efficient_lm_head_down_proj"),
            return_bias=False,
        )
        self.efficient_lm_head = ParallelLMHead(
            self.config.vocab_size,
            rank,
            quant_config=None,
            prefix=maybe_prefix(prefix, "efficient_lm_head"),
        )

    def _gather_candidates(
        self,
        hidden_states: torch.Tensor,
        width: int,
        top_tokens: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Every rank's candidates as exact logits and global token ids (as
        float32), [tp_size, num_tokens, width] views of one gathered buffer.
        Each rank contributes its num_candidates candidates, or with
        ``top_tokens`` the best candidate of each candidate block."""
        if not isinstance(
            self.lm_head.quant_method,
            (UnquantizedEmbeddingMethod, UnquantizedLinearMethod),
        ):
            raise NotImplementedError(
                "EAGLE-4 candidate logits require an unquantized lm_head."
            )

        num_tokens = hidden_states.shape[0]
        # [num_tokens, rank]
        down_projected = self.efficient_lm_head_down_proj(hidden_states)
        # [num_tokens, shard_vocab_size]
        scores = self.efficient_lm_head.quant_method.apply(
            self.efficient_lm_head, down_projected
        )
        shard_indices = self.lm_head.shard_indices
        if shard_indices.num_org_vocab_padding:
            scores[:, -shard_indices.num_org_vocab_padding :] = float("-inf")

        # [num_tokens, num_candidates], rows of this rank's lm_head shard
        topk_indices = _topk(scores, self.num_candidates)[1]
        # [num_tokens, logits | token ids]
        buffer = hidden_states.new_empty(num_tokens, 2 * width, dtype=torch.float32)
        num_candidate_blocks = triton.cdiv(self.num_candidates, _CANDIDATE_BLOCK_SIZE)
        _compute_topk_logits[(num_tokens, num_candidate_blocks)](
            buffer[:, :width],
            buffer[:, width:],
            buffer.stride(0),
            hidden_states,
            hidden_states.stride(0),
            topk_indices,
            topk_indices.stride(0),
            self.lm_head.weight,
            self.lm_head.weight.stride(0),
            hidden_states.shape[1],
            shard_indices.org_vocab_start_index,
            self.num_candidates,
            HIDDEN_BLOCK_SIZE=1024,
            CANDIDATE_BLOCK_SIZE=_CANDIDATE_BLOCK_SIZE,
            TOP_TOKENS=top_tokens,
            ROUND_DTYPE=_TL_DTYPES[self.head_dtype],
            num_warps=2,
        )
        if self.lm_head.tp_size > 1:
            buffer = sp_all_gather(buffer)
        buffer = buffer.view(self.lm_head.tp_size, num_tokens, 2 * width)
        return buffer[..., :width], buffer[..., width:]

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.num_candidates:
            return super().compute_logits(hidden_states)

        num_tokens = hidden_states.shape[0]
        # [num_tokens, vocab_size]
        logits = torch.full(
            (num_tokens, self.config.vocab_size),
            float("-inf"),
            dtype=self.head_dtype,
            device=hidden_states.device,
        )
        if num_tokens == 0:
            return logits

        candidate_logits, candidate_token_ids = self._gather_candidates(
            hidden_states, self.num_candidates, top_tokens=False
        )
        total_num_candidates = self.lm_head.tp_size * self.num_candidates
        _merge_candidates[(num_tokens,)](
            logits,
            logits.stride(0),
            candidate_logits,
            candidate_token_ids,
            candidate_logits.stride(0),
            candidate_logits.stride(1),
            self.num_candidates,
            total_num_candidates,
            BLOCK_SIZE=triton.next_power_of_2(total_num_candidates),
        )
        return logits

    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.num_candidates:
            return super().get_top_tokens(hidden_states)

        num_tokens = hidden_states.shape[0]
        top_tokens = hidden_states.new_empty(num_tokens, dtype=torch.int64)
        if num_tokens == 0:
            return top_tokens

        num_blocks = triton.cdiv(self.num_candidates, _CANDIDATE_BLOCK_SIZE)
        block_logits, block_token_ids = self._gather_candidates(
            hidden_states, num_blocks, top_tokens=True
        )
        total_num_blocks = self.lm_head.tp_size * num_blocks
        _select_top_token[(num_tokens,)](
            top_tokens,
            block_logits,
            block_token_ids,
            block_logits.stride(0),
            block_logits.stride(1),
            num_blocks,
            total_num_blocks,
            BLOCK_SIZE=triton.next_power_of_2(total_num_blocks),
        )
        return top_tokens
