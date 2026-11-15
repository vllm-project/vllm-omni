# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Inference-only MiniCPM DiT projections and attention on Ascend."""

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni.model_executor.models.minicpmo_4_5.dit_fused import (
    _GATE_CONV,
    _GATE_MLP,
    _GATE_MSA,
    _SCALE_CONV,
    _SCALE_MLP,
    _SCALE_MSA,
    _SHIFT_CONV,
    _SHIFT_MLP,
    _SHIFT_MSA,
    _mlp,
    causal_cache_index,
    dit_modulation,
    gather_causal_cache,
    supports_fused_body,
)

from .cosyvoice2_dit_attn import _expand_attn_mask_for_npu


def supports_packed_qkv(attn: nn.Module) -> bool:
    projections: list[nn.Linear] = []
    for name in ("to_q", "to_k", "to_v"):
        projection = getattr(attn, name, None)
        if type(projection) is not nn.Linear:
            return False
        projections.append(cast(nn.Linear, projection))
    reference = projections[0]
    return all(p.weight.shape == reference.weight.shape for p in projections) and (
        all(p.bias is None for p in projections) or all(p.bias is not None for p in projections)
    )


def _same_view(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.data_ptr() == b.data_ptr()
        and a.shape == b.shape
        and a.stride() == b.stride()
        and a.dtype == b.dtype
        and a.device == b.device
    )


def prepare_estimator(estimator: nn.Module, *, pack_convs: bool = False) -> None:
    """Establish weight storage before either eager-body or fused-body capture."""
    pack_convs = pack_convs and supports_fused_body(estimator)
    for block in getattr(estimator, "blocks", ()):
        attn = getattr(block, "attn", None)
        if attn is not None and not attn.training and supports_packed_qkv(attn):
            if attn.to_q.weight.device.type == "npu" and attn.to_q.weight.dtype == torch.float32:
                packed_qkv(attn)
        if pack_convs:
            for conv in (block.conv.block[1], block.conv.block[6]):
                if not conv.training and conv.weight.device.type == "npu" and conv.weight.dtype == torch.float32:
                    packed_conv_columns(conv)


def packed_qkv(attn: nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pack frozen projections once, retaining original Parameters as views.

    In-place weight loads then update the packed storage too. Replacing a
    parameter or moving the module rebuilds the pack on the next eager call.
    As with other device graphs, reloads require recapture. This also supports
    inference tensors, which have no version counter. Warm up before capture.
    """
    projections = (attn.to_q, attn.to_k, attn.to_v)
    cached = attn.__dict__.get("_npu_packed_qkv")
    rows = projections[0].out_features
    if cached is not None:
        weight, bias = cached
        if all(
            _same_view(p.weight, weight[i * rows : (i + 1) * rows])
            and (
                (p.bias is None and bias is None)
                or (p.bias is not None and bias is not None and _same_view(p.bias, bias[i * rows : (i + 1) * rows]))
            )
            for i, p in enumerate(projections)
        ):
            return cached
    if projections[0].weight.device.type == "npu" and torch.npu.is_current_stream_capturing():
        raise RuntimeError("MiniCPM NPU QKV weights must be packed during warmup, before graph capture")
    with torch.no_grad():
        weight = torch.cat([p.weight for p in projections])
        bias = None if projections[0].bias is None else torch.cat([p.bias for p in projections])
        for i, p in enumerate(projections):
            p.weight.data = weight[i * rows : (i + 1) * rows]
            if bias is not None:
                p.bias.data = bias[i * rows : (i + 1) * rows]
    attn.__dict__["_npu_packed_qkv"] = weight, bias
    return weight, bias


def attention(attn, x, att_cache, attn_mask, kv, *, min_cache_frames: int):
    """One QKV GEMM, preserving contiguous attention for short histories."""
    batch, frames, _ = x.shape
    weight, bias = packed_qkv(attn)
    q, k, v = F.linear(x, weight, bias).chunk(3, dim=-1)
    q, k, v = attn.q_norm(attn.to_heads(q)), attn.k_norm(attn.to_heads(k)), attn.to_heads(v)
    cached = 0 if att_cache is None else int(att_cache.shape[2])
    dest = kv[:, :, : frames + cached]
    if cached >= min_cache_frames:
        behind = dest[:, :, frames:]
        if att_cache.data_ptr() != behind.data_ptr() or att_cache.stride() != behind.stride():
            behind.copy_(att_cache)
        dim = k.shape[-1]
        dest[:, :, :frames, :dim].copy_(k)
        dest[:, :, :frames, dim:].copy_(v)
        k, v = dest.chunk(2, dim=-1)
    else:
        if att_cache is not None:
            old_k, old_v = att_cache.chunk(2, dim=-1)
            k, v = torch.cat((k, old_k), dim=2), torch.cat((v, old_v), dim=2)
        dest.copy_(torch.cat((k, v), dim=-1))
    mask = _expand_attn_mask_for_npu(attn_mask, frames, frames + cached)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return attn.proj(out.transpose(1, 2).reshape(batch, frames, -1))


def packed_conv_columns(conv: nn.Conv1d) -> torch.Tensor:
    """``(C_out, K * C_in)`` weight for frames-major im2col columns ``[k * C_in + c]``.

    The Parameter is retained as a view of the pack, so in-place loads update
    it, and replacing/moving a Parameter rebuilds it. Inference tensors have
    no version counter. Graphs must be recaptured after weight reloads, as
    with the rest of the body.
    """
    shape = (conv.out_channels, conv.kernel_size[0], conv.in_channels)
    packed = conv.__dict__.get("_npu_packed_conv_columns")
    if packed is not None and _same_view(conv.weight, packed.view(shape).permute(0, 2, 1)):
        return packed
    if conv.weight.device.type == "npu" and torch.npu.is_current_stream_capturing():
        raise RuntimeError("MiniCPM NPU convolution weights must be packed during warmup, before graph capture")
    with torch.no_grad():
        packed = conv.weight.detach().permute(0, 2, 1).reshape(conv.out_channels, -1).contiguous()
        conv.weight.data = packed.view(shape).permute(0, 2, 1)
    conv.__dict__["_npu_packed_conv_columns"] = packed
    return packed


def _norm_affine(norm: nn.LayerNorm, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight, bias = norm.weight, norm.bias
    if weight is not None and bias is not None:
        return weight, bias
    cached = norm.__dict__.get("_npu_norm_affine")
    if cached is None or cached[0].device != like.device or cached[0].dtype != like.dtype:
        cached = (like.new_ones(like.shape[-1]), like.new_zeros(like.shape[-1]))
        norm.__dict__["_npu_norm_affine"] = cached
    return (cached[0] if weight is None else weight), (cached[1] if bias is None else bias)


def _add_layer_norm_op(x: torch.Tensor):
    """ACLNN ``npu_add_layer_norm`` when it can run on this tensor, else ``None`` for the native fallback."""
    if x.device.type != "npu":
        return None
    try:
        import torch_npu
    except ImportError:
        return None
    return getattr(torch_npu, "npu_add_layer_norm", None)


def gated_residual_norm(x, gate, branch, norm):
    """Return both the residual stream and its normalization with ACLNN AddLayerNorm."""
    op = _add_layer_norm_op(x)
    if op is None:
        residual = torch.addcmul(x, gate, branch)
        return residual, norm(residual)
    weight, bias = _norm_affine(norm, x)
    normalized, _, _, residual = op(x, gate * branch, weight, bias, epsilon=norm.eps, additional_output=True)
    return residual, normalized


def _add_layer_norm(x, branch, weight, bias, eps):
    """``(x + branch, LayerNorm(x + branch) * weight + bias)`` in one ACLNN AddLayerNorm."""
    op = _add_layer_norm_op(x)
    if op is None:
        added = x + branch
        return added, F.layer_norm(added, (int(added.shape[-1]),), weight, bias, eps)
    normalized, _, _, added = op(x, branch, weight, bias, epsilon=float(eps), additional_output=True)
    return added, normalized


def _causal_conv(conv: nn.Conv1d, history: torch.Tensor, frames: int) -> torch.Tensor:
    """``conv`` over contiguous frames-major ``history`` (N, W + T, C_in): one im2col copy, one biased GEMM."""
    rows, _, channels = history.shape
    taps = int(conv.kernel_size[0])
    columns = history.as_strided((rows, frames, taps * channels), (int(history.stride(0)), channels, 1))
    out = F.linear(columns.reshape(rows * frames, taps * channels), packed_conv_columns(conv), conv.bias)
    return out.view(rows, frames, -1)


def _frames_major_cache(cnn_cache, width: int):
    """All blocks' ``(N, C, W)`` caches as one contiguous ``(depth, N, W, C)`` tensor: one transpose per body."""
    if width <= 0:
        return None
    if not isinstance(cnn_cache, torch.Tensor):
        caches = list(cnn_cache)
        if all(cache is None for cache in caches):
            return None
        if any(cache is None for cache in caches):
            raise ValueError("NPU fused body needs either every block's CNN cache or none")
        if any(cache.shape != caches[0].shape for cache in caches):
            return [cache.transpose(1, 2) for cache in caches]
        cnn_cache = torch.stack(caches)
    return cnn_cache.transpose(-1, -2).contiguous()


def _attach_history(old: torch.Tensor | None, current: torch.Tensor, width: int) -> torch.Tensor:
    """``(N, W + T, C)`` from frames-major cached frames and the current ones."""
    if width <= 0:
        return current
    if old is None:
        return F.pad(current, (0, 0, width, 0))
    return torch.cat((old, current), dim=1)


def attention_npu(attn, hidden, att_cache, attn_mask, kv, slots=None):
    """Packed-QKV attention using the shared ``[K | V]`` cache layout."""
    if slots is not None:
        raise RuntimeError("MiniCPM NPU fused body does not support the CFM attention slot pool")
    batch, frames, _ = hidden.shape
    if supports_packed_qkv(attn):
        weight, bias = packed_qkv(attn)
        qkv = F.linear(hidden.reshape(batch * frames, -1), weight, bias).view(batch, frames, -1)
    else:
        qkv = torch.cat([attn.to_q(hidden), attn.to_k(hidden), attn.to_v(hidden)], dim=-1)
    cached = 0 if att_cache is None else int(att_cache.shape[2])
    cache_capacity = int(kv.shape[2]) - frames
    logical_cache = cached if attn_mask is None else int(attn_mask.shape[-1]) - frames
    if logical_cache < cached or logical_cache > cache_capacity:
        raise ValueError(
            f"NPU fused attention cache/mask exceeds output capacity: "
            f"cache={cached}, mask_cache={logical_cache}, capacity={cache_capacity}"
        )
    # Whole-Euler reserves query padding between the live queries and cache.
    # Keep this capacity in the attention sequence so the capture mask remains
    # aligned with the static cache buffer on every replay.
    kv = kv[:, :, : frames + logical_cache]
    behind = kv[:, :, frames:]
    if cached and (att_cache.data_ptr() != behind.data_ptr() or att_cache.stride() != behind.stride()):
        behind[:, :, :cached].copy_(att_cache)
    packed = qkv.view(batch, frames, 3, int(attn.num_heads), int(attn.head_dim))
    query = attn.q_norm(packed[:, :, 0].transpose(1, 2))
    current_keys = attn.k_norm(packed[:, :, 1].transpose(1, 2))
    current_values = packed[:, :, 2].transpose(1, 2)
    head_dim = int(query.shape[-1])
    # The shared cache format stores K in the first half and V in the second.
    # A masked gap past the real cache must stay finite: an uninitialized key
    # poisons real rows through ``0 * NaN`` in the attention epilogue.
    if logical_cache > cached:
        behind[:, :, cached:].zero_()
    current = kv[:, :, :frames]
    if current.is_contiguous():
        torch.cat((current_keys, current_values), dim=-1, out=current)
    else:
        current[..., :head_dim].copy_(current_keys)
        current[..., head_dim:].copy_(current_values)
    keys, values = kv[..., :head_dim], kv[..., head_dim:]
    mask = _expand_attn_mask_for_npu(attn_mask, frames, int(keys.shape[-2]))
    if mask is not None and int(mask.shape[-1]) != int(keys.shape[-2]):
        raise ValueError(
            f"NPU fused attention mask/cache mismatch: mask has {mask.shape[-1]} keys, cache has {keys.shape[-2]}"
        )
    out = F.scaled_dot_product_attention(query, keys, values, attn_mask=mask)
    out = out.transpose(1, 2).reshape(batch * frames, -1)
    return F.linear(out, attn.proj.weight, attn.proj.bias).view(batch, frames, -1)


def fused_body_accepts(attn_mask: torch.Tensor | None, att_cache_buffer: torch.Tensor) -> bool:
    """Whether the key mask describes the same sequence as the static output cache (Whole-Euler layout)."""
    return attn_mask is None or int(attn_mask.shape[-1]) == int(att_cache_buffer.shape[-2])


def blocks_forward_chunk_fused_npu(
    estimator: nn.Module,
    estimator_input: torch.Tensor,
    time_embedding: torch.Tensor,
    attn_mask: torch.Tensor | None,
    cnn_cache,
    att_cache,
    cnn_cache_buffer: torch.Tensor,
    att_cache_buffer: torch.Tensor,
    valid_lengths,
    modulation: torch.Tensor | None = None,
    slots: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """``BatchedToken2Wav._blocks_forward_chunk_ragged`` scheduled for Ascend.

    The CUDA fused body's math with AddLayerNorm residual updates, packed QKV,
    im2col convolutions with biased GEMMs, and frames-major CNN caches that are
    transposed once per body instead of per block.
    """
    if slots is not None:
        raise RuntimeError("MiniCPM NPU fused body does not support the CFM attention slot pool")
    if not fused_body_accepts(attn_mask, att_cache_buffer):
        raise RuntimeError("NPU fused body mask does not match the captured cache layout")
    if isinstance(valid_lengths, torch.Tensor):
        lengths = valid_lengths
    else:
        lengths = torch.tensor((*valid_lengths, *valid_lengths), device=estimator_input.device, dtype=torch.long)
    if modulation is None:
        modulation = dit_modulation(estimator, time_embedding[:1])
    blocks = estimator.blocks
    depth = len(blocks)
    conv1, conv2 = blocks[0].conv.block[1], blocks[0].conv.block[6]
    width = int(conv1.kernel_size[0]) - 1
    split = conv1.in_channels

    x = estimator.in_proj(estimator_input.transpose(1, 2))
    rows, frames, channels = x.shape
    old = _frames_major_cache(cnn_cache, width)
    # Both histories' causal caches land frames-major in one staging tensor;
    # a single transpose at the end writes the caller's (N, C, W) layout.
    staged = (
        width > 0
        and isinstance(cnn_cache_buffer, torch.Tensor)
        and conv1.in_channels == conv2.in_channels
        and all(b.conv.block[1].kernel_size == b.conv.block[6].kernel_size == conv1.kernel_size for b in blocks)
    )
    if staged:
        staging = x.new_empty((depth, 2, rows * width, split))
        offsets = torch.arange(rows, device=x.device, dtype=lengths.dtype) * (width + frames)
        rows_index = (offsets + lengths)[:, None] + torch.arange(width, device=x.device, dtype=lengths.dtype)
        rows_index = rows_index.reshape(-1)
    cache_index: dict[int, torch.Tensor] = {}

    first = modulation[0]
    hidden = F.layer_norm(x, (channels,), first[_SCALE_MSA], first[_SHIFT_MSA], blocks[0].norm1.eps)
    for block_index, block in enumerate(blocks):
        mod = modulation[block_index]
        x_att = attention_npu(block.attn, hidden, att_cache[block_index], attn_mask, att_cache_buffer[block_index])

        # CausalConvBlock: [conv1, LayerNorm, Mish, conv2] over (N, W + T, C) histories.
        conv_layers = block.conv.block
        conv1, conv_norm, conv2 = conv_layers[1], conv_layers[3], conv_layers[6]
        block_width = int(conv1.kernel_size[0]) - 1
        old1 = None if old is None else old[block_index][..., : conv1.in_channels]
        old2 = None if old is None else old[block_index][..., conv1.in_channels :]
        x, conv_in = _add_layer_norm(x, mod[_GATE_MSA] * x_att, mod[_SCALE_CONV], mod[_SHIFT_CONV], block.norm3.eps)
        history = _attach_history(old1, conv_in, block_width)
        conv_hidden = _causal_conv(conv1, history, frames)
        conv_weight, conv_bias = _norm_affine(conv_norm, conv_hidden)
        conv_hidden = F.mish(
            F.layer_norm(conv_hidden, (int(conv_hidden.shape[-1]),), conv_weight, conv_bias, conv_norm.eps)
        )
        history2 = _attach_history(old2, conv_hidden, block_width)
        x, mlp_in = _add_layer_norm(
            x,
            mod[_GATE_CONV] * _causal_conv(conv2, history2, frames),
            mod[_SCALE_MLP],
            mod[_SHIFT_MLP],
            block.norm2.eps,
        )
        mlp_out = _mlp(block.mlp, mlp_in)
        # Table row ``depth`` holds the final layer's shift and 1 + scale where a block keeps its MSA pair.
        last = block_index + 1 == depth
        x, hidden = _add_layer_norm(
            x,
            mod[_GATE_MLP] * mlp_out,
            modulation[block_index + 1][_SCALE_MSA],
            modulation[block_index + 1][_SHIFT_MSA],
            estimator.final_layer.norm_final.eps if last else blocks[block_index + 1].norm1.eps,
        )

        if staged:
            torch.index_select(history.view(-1, split), 0, rows_index, out=staging[block_index, 0])
            torch.index_select(history2.view(-1, split), 0, rows_index, out=staging[block_index, 1])
        else:
            if block_width not in cache_index:
                cache_index[block_width] = causal_cache_index(lengths, block_width)
            cnn_out = cnn_cache_buffer[block_index]
            cnn_out[:, : conv1.in_channels].copy_(gather_causal_cache(history, cache_index[block_width]))
            cnn_out[:, conv1.in_channels :].copy_(gather_causal_cache(history2, cache_index[block_width]))

    if staged:
        cnn_cache_buffer.unflatten(2, (2, split)).copy_(
            staging.view(depth, 2, rows, width, split).permute(0, 2, 1, 4, 3)
        )
    return estimator.final_layer.linear(hidden).transpose(1, 2)
