# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tensor-parallel linear layers shared by diffusion transformers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn import Parameter
from vllm.model_executor.layers.linear import RowParallelLinear

from vllm_omni.quantization.component_config import ComponentQuantizationConfig

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

# Quantization methods whose row-parallel parameters are either indexed by the input
# dimension (and so can be re-ordered by ``MergedRowParallelLinear``) or carry no input
# dimension at all (per-tensor / per-output-channel scales). Models fall back to their
# replicated projections for anything else; see ``merged_row_parallel_supported``.
_VALIDATED_QUANT_METHODS = frozenset({"fp8"})

# Parameters whose *values* are positions along the input dimension (GPTQ/AWQ act-order
# group indices). Re-ordering the columns would leave the positions pointing at the
# wrong groups, so these are rejected instead of being loaded wrongly.
_POSITIONAL_INPUT_PARAMS = ("g_idx", "weight_g_idx")


def merged_row_parallel_supported(quant_config: QuantizationConfig | None, prefix: str = "") -> bool:
    """Whether ``MergedRowParallelLinear`` can load checkpoints for ``quant_config``.

    Callers use this to choose between sharding a concatenated-input projection and
    keeping it replicated (``disable_tp=True``) at ``tensor_parallel_size > 1``.
    ``prefix`` resolves a ``ComponentQuantizationConfig`` to the layer's own config.
    """
    if isinstance(quant_config, ComponentQuantizationConfig):
        quant_config = quant_config.resolve(prefix)
    return quant_config is None or quant_config.get_name() in _VALIDATED_QUANT_METHODS


def to_rank_major(tensor: torch.Tensor, dim: int, input_sizes: list[int], tp_size: int, name: str = "") -> torch.Tensor:
    """Re-order ``[W_0 | W_1 | ...]`` along ``dim`` into ``[W_0^r0 W_1^r0 | W_0^r1 W_1^r1 | ...]``.

    After this, a contiguous per-rank narrow of width ``width / tp_size`` yields
    ``cat([W_0^r, W_1^r, ...])`` for rank ``r``, i.e. the layout
    ``MergedRowParallelLinear`` keeps on each rank. The width along ``dim`` may differ
    from ``sum(input_sizes)`` (block/group scales, packed integer weights) as long as
    every segment maps to a whole number of elements on every rank.
    """
    if tp_size == 1:
        return tensor
    total = sum(input_sizes)
    width = tensor.shape[dim]
    seg_widths = []
    for size in input_sizes:
        if (size * width) % total != 0 or (size * width // total) % tp_size != 0:
            raise ValueError(
                f"{name or 'MergedRowParallelLinear'}: cannot split an input dimension of width "
                f"{width} into segments {input_sizes} across tensor_parallel_size={tp_size}"
            )
        seg_widths.append(size * width // total)
    segments = tensor.split(seg_widths, dim=dim)
    return torch.cat(
        [segment.chunk(tp_size, dim=dim)[rank] for rank in range(tp_size) for segment in segments], dim=dim
    )


class MergedRowParallelLinear(RowParallelLinear):
    """Row-parallel linear over a concatenation of independently sharded inputs.

    The input-dimension counterpart of ``MergedColumnParallelLinear``. Parallel
    attention + MLP blocks (FLUX-style single-stream blocks) compute

        out = proj_out(cat([attn, mlp], dim=-1))

    where ``attn`` and ``mlp`` are each sharded on their own feature dimension
    across TP ranks. A plain ``RowParallelLinear`` would split the concatenated
    input into contiguous chunks, which does not match the per-rank ``attn``/``mlp``
    shards. This layer stores, on rank ``r``, the columns
    ``cat([W_0[:, r-th shard], W_1[:, r-th shard], ...])`` so that

        out = all_reduce(cat([attn_r, mlp_r], dim=-1) @ W_r.T) + bias

    is one GEMM and one all-reduce per call. Forward, bias handling (added once, on
    rank 0) and ``reduce_results`` are inherited from ``RowParallelLinear``.

    Checkpoints keep the unsharded ``[W_0 | W_1 | ...]`` tensor under the layer's own
    name; the weight loader re-orders every input-dimension parameter (weights, block
    or group scales, packed weights) segment by segment. Parameters with no input
    dimension (bias, per-tensor and per-channel scales) load unchanged.

    Args:
        input_sizes: Logical width of each concatenated input segment. Each must be
            divisible by the TP size.
        output_size: Output width.
        Remaining arguments are as for ``RowParallelLinear``; the input is always
        expected to be already sharded (``input_is_parallel=True``).
    """

    def __init__(
        self,
        input_sizes: list[int],
        output_size: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        reduce_results: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        self.input_sizes = list(input_sizes)
        super().__init__(
            sum(self.input_sizes),
            output_size,
            bias=bias,
            input_is_parallel=True,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )
        for size in self.input_sizes:
            if size % self.tp_size != 0:
                raise ValueError(
                    f"{prefix or type(self).__name__}: input segment size {size} is not "
                    f"divisible by tensor_parallel_size={self.tp_size}"
                )
        if self.tp_size > 1:
            for name in _POSITIONAL_INPUT_PARAMS:
                if getattr(self, name, None) is not None:
                    raise ValueError(
                        f"{prefix or type(self).__name__}: '{name}' holds positions along the "
                        f"input dimension and cannot be re-ordered per input segment; use "
                        f"replicated projections for this quantization method"
                    )

    def _to_rank_major(self, param: Parameter, loaded_weight: torch.Tensor) -> torch.Tensor:
        input_dim = getattr(param, "input_dim", None)
        if input_dim is None or getattr(param, "is_sharded_weight", False) or loaded_weight.dim() == 0:
            return loaded_weight
        return to_rank_major(loaded_weight, input_dim, self.input_sizes, self.tp_size, self.prefix)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        super().weight_loader(param, self._to_rank_major(param, loaded_weight))

    def weight_loader_v2(self, param: Parameter, loaded_weight: torch.Tensor):
        super().weight_loader_v2(param, self._to_rank_major(param, loaded_weight))
