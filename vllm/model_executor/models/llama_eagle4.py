# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EAGLE-4 Llama drafter.

Same layout as EAGLE-3 plus a low-rank candidate LM head: a rank-r projection
scores the whole vocab, the top ``num_candidates`` tokens are kept, and exact
logits are computed for those tokens only from the full ``lm_head``. The
sliding-window draft layer needs no changes here; ``LlamaAttention`` reads it
from ``layer_types`` / ``sliding_window``.
"""

import torch

from vllm.config import VllmConfig
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm.model_executor.layers.logits_processor import _topk
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.triton_utils import tl, triton

from .utils import maybe_prefix


@triton.jit
def _compute_topk_logits(
    logits_ptr,
    logits_stride,
    hidden_states_ptr,
    hidden_states_stride,
    topk_indices_ptr,
    topk_indices_stride,
    lm_head_ptr,
    lm_head_stride,
    hidden_size,
    vocab_start,
    vocab_end,
    num_candidates,
    HIDDEN_BLOCK_SIZE: tl.constexpr,
    CANDIDATE_BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    candidate_block_idx = tl.program_id(1)
    c_block = candidate_block_idx * CANDIDATE_BLOCK_SIZE + tl.arange(
        0, CANDIDATE_BLOCK_SIZE
    )
    c_mask = c_block < num_candidates
    # [c_block]
    topk_idxs = tl.load(
        topk_indices_ptr + token_idx * topk_indices_stride + c_block,
        mask=c_mask,
        other=0,
    ).to(tl.int64)
    owned_mask = c_mask & (topk_idxs >= vocab_start) & (topk_idxs < vocab_end)
    local_rows = tl.where(owned_mask, topk_idxs - vocab_start, 0)

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
            mask=owned_mask[:, None] & h_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += topk_lm_head * hidden_states[None, :]
    topk_logits = tl.sum(acc, axis=1)
    tl.store(
        logits_ptr + token_idx * logits_stride + c_block,
        topk_logits,
        mask=c_mask,
    )


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
        self.efficient_lm_head = ReplicatedLinear(
            rank,
            self.config.vocab_size,
            bias=False,
            quant_config=None,
            prefix=maybe_prefix(prefix, "efficient_lm_head"),
            return_bias=False,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.num_candidates:
            return super().compute_logits(hidden_states)

        if not isinstance(
            self.lm_head.quant_method,
            (UnquantizedEmbeddingMethod, UnquantizedLinearMethod),
        ):
            raise NotImplementedError(
                "EAGLE-4 candidate logits require an unquantized lm_head."
            )

        num_tokens, hidden_size = hidden_states.shape
        # [num_tokens, vocab_size]
        logits = torch.full(
            (num_tokens, self.config.vocab_size),
            float("-inf"),
            dtype=self.head_dtype,
            device=hidden_states.device,
        )
        if num_tokens == 0:
            return logits

        # [num_tokens, rank]
        down_projected = self.efficient_lm_head_down_proj(hidden_states)
        # [num_tokens, vocab_size]
        effective_logits = self.efficient_lm_head(down_projected)
        # [num_tokens, num_candidates]
        _, topk_indices = _topk(effective_logits, self.num_candidates)

        CANDIDATE_BLOCK_SIZE = 8
        num_candidate_blocks = triton.cdiv(self.num_candidates, CANDIDATE_BLOCK_SIZE)
        shard_indices = self.lm_head.shard_indices
        topk_logits = logits.new_empty(num_tokens, self.num_candidates)
        _compute_topk_logits[(num_tokens, num_candidate_blocks)](
            topk_logits,
            topk_logits.stride(0),
            hidden_states,
            hidden_states.stride(0),
            topk_indices,
            topk_indices.stride(0),
            self.lm_head.weight,
            self.lm_head.weight.stride(0),
            hidden_size,
            shard_indices.org_vocab_start_index,
            shard_indices.org_vocab_end_index,
            self.num_candidates,
            HIDDEN_BLOCK_SIZE=1024,
            CANDIDATE_BLOCK_SIZE=CANDIDATE_BLOCK_SIZE,
            num_warps=2,
        )
        topk_logits = tensor_model_parallel_all_reduce(topk_logits)
        logits.scatter_(1, topk_indices.long(), topk_logits)
        return logits
