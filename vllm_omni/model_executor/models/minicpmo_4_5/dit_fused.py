# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused Code2Wav CFM DiT block forward (~15 kernels vs ~47 eager ops per block)."""

from __future__ import annotations

import functools
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import qkv_head_layer_norm, residual_layer_norm

# adaLN chunk order of ``DiTBlock.adaLN_modulation``.
_SHIFT_MSA, _SCALE_MSA, _GATE_MSA, _SHIFT_MLP, _SCALE_MLP, _GATE_MLP, _SHIFT_CONV, _SCALE_CONV, _GATE_CONV = range(9)


def supports_fused_body(estimator: nn.Module) -> bool:
    """Whether ``blocks_forward_chunk_fused`` computes this estimator's blocks."""
    blocks = getattr(estimator, "blocks", None)
    final_layer = getattr(estimator, "final_layer", None)
    if not blocks or not hasattr(estimator, "in_proj") or final_layer is None:
        return False
    if not isinstance(getattr(final_layer, "norm_final", None), nn.LayerNorm) or not hasattr(final_layer, "linear"):
        return False
    for block in blocks:
        attn = getattr(block, "attn", None)
        conv_block = getattr(getattr(block, "conv", None), "block", None)
        mlp = getattr(block, "mlp", None)
        if attn is None or conv_block is None or mlp is None:
            return False
        if not all(hasattr(attn, name) for name in ("to_q", "to_k", "to_v", "proj", "num_heads", "head_dim")):
            return False
        if not (isinstance(attn.q_norm, nn.LayerNorm) and isinstance(attn.k_norm, nn.LayerNorm)):
            return False
        if not all(isinstance(getattr(block, name, None), nn.LayerNorm) for name in ("norm1", "norm2", "norm3")):
            return False
        if len(conv_block) < 7 or not isinstance(conv_block[3], nn.LayerNorm) or not isinstance(conv_block[4], nn.Mish):
            return False
        for conv in (conv_block[1], conv_block[6]):
            if not isinstance(conv, nn.Conv1d) or conv.stride != (1,) or conv.dilation != (1,) or conv.groups != 1:
                return False
            if conv.padding != (0,) or conv.kernel_size[0] < 2:
                return False
        if not (hasattr(mlp, "fc1") and hasattr(mlp, "fc2") and isinstance(mlp.act, nn.GELU)):
            return False
        if not isinstance(getattr(mlp, "norm", nn.Identity()), nn.Identity):
            return False
    return True


def dit_modulation(estimator: nn.Module, time_embedding: torch.Tensor) -> torch.Tensor:
    """adaLN table ``(depth + 1, 9, C)`` for one timestep; scales stored as ``1 + scale``."""
    embedding = time_embedding.reshape(1, 1, -1)
    table = torch.stack([block.adaLN_modulation(embedding).reshape(9, -1) for block in estimator.blocks], dim=0)
    table[:, (_SCALE_MSA, _SCALE_MLP, _SCALE_CONV)] += 1.0
    final_shift, final_scale = estimator.final_layer.adaLN_modulation(embedding).reshape(2, -1)
    final = table.new_zeros((1, 9, int(table.shape[2])))
    # Where a block keeps its MSA shift/scale, so the last block's closing LayerNorm reads it like a next block.
    final[0, _SHIFT_MSA] = final_shift
    final[0, _SCALE_MSA] = final_scale + 1.0
    return torch.cat((table, final), dim=0).contiguous()


def _cached(module: nn.Module, key: str, sources: tuple[torch.Tensor, ...], build) -> Any:
    version = tuple((t.data_ptr(), t._version, t.dtype) for t in sources)
    cached = module.__dict__.get(key)
    if cached is None or cached[0] != version:
        with torch.no_grad():
            cached = (version, build())
        module.__dict__[key] = cached
    return cached[1]


def packed_qkv(attn: nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``[to_q; to_k; to_v]`` weight and bias as one projection, rebuilt only when a source changes."""
    sources = (attn.to_q.weight, attn.to_k.weight, attn.to_v.weight)

    def build():
        weight = torch.cat(sources, dim=0).contiguous()
        bias = None
        if attn.to_q.bias is not None:
            bias = torch.cat((attn.to_q.bias, attn.to_k.bias, attn.to_v.bias), dim=0).contiguous()
        return weight, bias

    return _cached(attn, "_packed_qkv", sources, build)


def _conv_taps_weight(conv: nn.Conv1d) -> torch.Tensor:
    """``(C_out, C_in, K)`` -> ``(K * C_out, C_in)``: tap ``k`` is output rows ``[k*C_out, (k+1)*C_out)``."""
    return _cached(
        conv,
        "_fused_taps_weight",
        (conv.weight,),
        lambda: conv.weight.detach().permute(2, 0, 1).reshape(-1, conv.in_channels).contiguous(),
    )


def _conv_history(
    old_cache: torch.Tensor | None, rows: int, width: int, frames: int, channels: int, like: torch.Tensor
) -> torch.Tensor:
    """``(N, width + frames, C)``: the cached frames, then room for the current ones."""
    history = like.new_empty((rows, width + frames, channels))
    if old_cache is None:
        history[:, :width].zero_()
    else:
        history[:, :width].copy_(old_cache.transpose(1, 2))
    return history


def causal_cache_index(lengths: torch.Tensor, width: int) -> torch.Tensor:
    """``(N, width)`` history frames that end each row's valid frames, for ``gather_causal_cache``."""
    return lengths[:, None] + torch.arange(width, device=lengths.device)[None, :]


def gather_causal_cache(history: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """The ``index`` frames of ``history`` (N, width + T, C), as the (N, C, width) causal cache."""
    return history.gather(1, index[:, :, None].expand(-1, -1, int(history.shape[2]))).transpose(1, 2)


def _mlp(mlp: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    rows = hidden.reshape(-1, int(hidden.shape[-1]))
    if mlp.act.approximate == "tanh" and rows.is_cuda and mlp.fc1.bias is not None:
        # cuBLASLt applies bias + tanh-GELU in the GEMM epilogue.
        inner = torch._addmm_activation(mlp.fc1.bias, rows, mlp.fc1.weight.t(), use_gelu=True)
    else:
        inner = mlp.act(F.linear(rows, mlp.fc1.weight, mlp.fc1.bias))
    return F.linear(inner, mlp.fc2.weight, mlp.fc2.bias).view(*hidden.shape[:-1], -1)


@functools.cache
def tiled_attention_supported(device_index: int) -> bool:
    """``cfm_attention``'s tf32x3 dots need NVIDIA tensor cores with TF32 (SM80+)."""
    return torch.version.hip is None and torch.cuda.get_device_capability(device_index) >= (8, 0)


def _attention(
    attn: nn.Module,
    hidden: torch.Tensor,
    att_cache: torch.Tensor | None,
    attn_mask: torch.Tensor | None,
    kv: torch.Tensor,
    slots: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """``attn.forward_chunk`` over ``hidden``, writing new keys/values into ``kv``."""
    batch, frames, _ = hidden.shape
    weight, bias = packed_qkv(attn)
    qkv = F.linear(hidden.reshape(batch * frames, -1), weight, bias).view(batch, frames, -1)
    rows, positions = slots if slots is not None else (None, None)
    if slots is None:
        # ``att_cache`` behind the current frames: left where it is when it
        # already sits there (the Whole-Euler arena's aliasing), else copied once.
        cached = 0 if att_cache is None else int(att_cache.shape[2])
        kv = kv[:, :, : frames + cached]
        behind = kv[:, :, frames:]
        if cached and (att_cache.data_ptr() != behind.data_ptr() or att_cache.stride() != behind.stride()):
            behind.copy_(att_cache)
    q = qkv_head_layer_norm(
        qkv,
        kv,
        num_heads=int(attn.num_heads),
        head_dim=int(attn.head_dim),
        q_norm=attn.q_norm,
        k_norm=attn.k_norm,
        rows=rows,
        positions=positions,
    )
    head_dim = int(q.shape[-1])
    keys, values = kv[..., :head_dim], kv[..., head_dim:]
    if rows is not None or (
        kv.is_cuda and kv.dtype == q.dtype == torch.float32 and tiled_attention_supported(kv.get_device())
    ):
        from .cfm_attention import cfm_attention

        out = cfm_attention(q, keys, values, attn_mask, rows)
    else:
        out = F.scaled_dot_product_attention(
            q, keys, values, attn_mask=None if attn_mask is None else attn_mask.unsqueeze(1)
        )
    out = out.transpose(1, 2).reshape(batch * frames, -1)
    return F.linear(out, attn.proj.weight, attn.proj.bias).view(batch, frames, -1)


def blocks_forward_chunk_fused(
    estimator: nn.Module,
    estimator_input: torch.Tensor,
    time_embedding: torch.Tensor,
    attn_mask: torch.Tensor | None,
    cnn_cache: Any,
    att_cache: Any,
    cnn_cache_buffer: torch.Tensor,
    att_cache_buffer: torch.Tensor,
    valid_lengths: list[int] | torch.Tensor,
    modulation: torch.Tensor | None = None,
    slots: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """``BatchedToken2Wav._blocks_forward_chunk_ragged`` with fused norms/GEMM/attention."""
    if isinstance(valid_lengths, torch.Tensor):
        lengths = valid_lengths
    else:
        lengths = torch.tensor((*valid_lengths, *valid_lengths), device=estimator_input.device, dtype=torch.long)
    if modulation is None:
        modulation = dit_modulation(estimator, time_embedding[:1])
    blocks = estimator.blocks
    depth = len(blocks)
    # Causal cache gather index per conv width, shared by the blocks.
    cache_index: dict[int, torch.Tensor] = {}

    x = estimator.in_proj(estimator_input.transpose(1, 2)).contiguous()
    rows, frames, _ = x.shape
    first = modulation[0]
    hidden = residual_layer_norm(x, weight=first[_SCALE_MSA], bias=first[_SHIFT_MSA], eps=blocks[0].norm1.eps)
    for block_index, block in enumerate(blocks):
        mod = modulation[block_index]
        x_att = _attention(block.attn, hidden, att_cache[block_index], attn_mask, att_cache_buffer[block_index], slots)

        # CausalConvBlock: [conv1, LayerNorm, Mish, conv2] on (N, width + T, C) histories.
        conv_layers = block.conv.block
        conv1, conv_norm, conv2 = conv_layers[1], conv_layers[3], conv_layers[6]
        width = int(conv1.kernel_size[0]) - 1
        old_cnn = cnn_cache[block_index]
        split = conv1.in_channels
        old_cnn1, old_cnn2 = (None, None) if old_cnn is None else (old_cnn[:, :split], old_cnn[:, split:])
        history = _conv_history(old_cnn1, rows, width, frames, split, x)
        # x += gate_msa * attention; the conv block's modulated LayerNorm lands in the history.
        residual_layer_norm(
            x,
            x_att,
            gate=mod[_GATE_MSA],
            weight=mod[_SCALE_CONV],
            bias=mod[_SHIFT_CONV],
            eps=block.norm3.eps,
            residual_out=x,
            out=history[:, width:],
        )
        taps1 = F.linear(history.view(-1, conv1.in_channels), _conv_taps_weight(conv1))
        history2 = _conv_history(old_cnn2, rows, width, frames, conv2.in_channels, x)
        residual_layer_norm(
            None,
            taps1.view(rows, width + frames, -1),
            taps=width + 1,
            y_bias=conv1.bias,
            weight=conv_norm.weight,
            bias=conv_norm.bias,
            eps=conv_norm.eps,
            activation="mish",
            out=history2[:, width:],
        )
        taps2 = F.linear(history2.view(-1, conv2.in_channels), _conv_taps_weight(conv2))
        # x += gate_conv * conv2; the MLP's modulated LayerNorm.
        mlp_in = residual_layer_norm(
            x,
            taps2.view(rows, width + frames, -1),
            taps=width + 1,
            y_bias=conv2.bias,
            gate=mod[_GATE_CONV],
            weight=mod[_SCALE_MLP],
            bias=mod[_SHIFT_MLP],
            eps=block.norm2.eps,
            residual_out=x,
        )
        mlp_out = _mlp(block.mlp, mlp_in)
        # x += gate_mlp * MLP, then the next block's (or the final layer's) modulated LayerNorm:
        # table row ``depth`` holds the final layer's shift and 1 + scale where a block keeps its MSA pair.
        last = block_index + 1 == depth
        hidden = residual_layer_norm(
            x,
            mlp_out,
            gate=mod[_GATE_MLP],
            weight=modulation[block_index + 1][_SCALE_MSA],
            bias=modulation[block_index + 1][_SHIFT_MSA],
            eps=estimator.final_layer.norm_final.eps if last else blocks[block_index + 1].norm1.eps,
            residual_out=None if last else x,
        )

        if width not in cache_index:
            cache_index[width] = causal_cache_index(lengths, width)
        cnn_out = cnn_cache_buffer[block_index]
        cnn_out[:, :split].copy_(gather_causal_cache(history, cache_index[width]))
        cnn_out[:, split:].copy_(gather_causal_cache(history2, cache_index[width]))

    return estimator.final_layer.linear(hidden).transpose(1, 2)
