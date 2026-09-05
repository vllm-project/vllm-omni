# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: N803
"""Bit-exact data-movement kernels for the diffusers Wan causal VAE decoder.

Adapted from SGLang PR #34125 (https://github.com/sgl-project/sglang/pull/34125).

The kernels in this module only move values, fill zeros, or perform the same
two-operand addition as the PyTorch reference path, so the decoder numerics are
unchanged. They preserve the input memory layout (contiguous or
channels_last_3d) instead of selecting a new one, so enabling them does not
change the cuDNN algorithm or accumulation order of the surrounding
convolutions.

Every public wrapper returns ``None`` when it declines an input (rank, dtype,
device, layout mismatch, or too many elements); callers then run the reference
PyTorch expression.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

_MAX_INT32 = 2**31 - 1
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_BLOCK_SIZE = 512
# Elements handled per program by the row/pixel-oriented cat+pad kernels: with
# 4 warps this is 32 elements per thread, enough independent 16-byte loads to
# keep HBM busy without register pressure.
_TILE_ELEMENTS = 4096
_NUM_WARPS = 4
# The channels-first tile holds int64 pointer tiles for source, output and cache
# refresh; 8 warps halve the per-thread register footprint versus 4.
_ROWS_BLOCK_W = 256
_ROWS_NUM_WARPS = 8
# Plane copies: 4096 elements per program on 8 warps = 16 elements (two
# 16-byte vectors) per thread.
_PLANE_BLOCK = 4096
# Column blocks for the row-tiled shortcut add (720p stage rows are 320/640/1280 wide).
_ROW_BLOCK_WIDTHS = (64, 128, 256)
# Square tile edge for assembling channels-first input into channels-last output.
_TRANSPOSE_BLOCK = 64
_HAS_INTERLEAVE = HAS_TRITON and hasattr(tl, "interleave")

if HAS_TRITON:

    @triton.jit
    def _cat_time_planes_kernel(
        x_ptr,
        cache_ptr,
        out_ptr,
        keep_ptr,
        plane_size,
        channels_outer,
        out_frames,
        cache_frames,
        zero_front_frames,
        keep_start,
        keep_frames,
        stride_x_batch,
        stride_x_channel,
        stride_x_frame,
        stride_cache_batch,
        stride_cache_channel,
        stride_cache_frame,
        has_cache: tl.constexpr,
        has_keep: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Temporal concat ``[zeros | cache | x]`` as whole-plane copies.

        A plane is one frame of one channel (channels-first) or one frame of all
        channels (channels-last): a contiguous run of ``plane_size`` elements
        whose start is aligned, so every load and store vectorizes. No spatial
        padding is written here; the consumer lets cuDNN pad inside the
        convolution.
        """
        plane = tl.program_id(0).to(tl.int64)
        offsets = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < plane_size
        frame = plane % out_frames
        rest = plane // out_frames
        channel = rest % channels_outer
        batch = rest // channels_outer
        source_frame = frame - zero_front_frames

        values = tl.zeros([BLOCK], dtype=x_ptr.dtype.element_ty)
        if source_frame >= cache_frames:
            x_plane = (
                x_ptr
                + batch * stride_x_batch
                + channel * stride_x_channel
                + (source_frame - cache_frames) * stride_x_frame
            )
            values = tl.load(x_plane + offsets, mask=mask, other=0)
        else:
            if has_cache:
                if source_frame >= 0:
                    cache_plane = (
                        cache_ptr
                        + batch * stride_cache_batch
                        + channel * stride_cache_channel
                        + source_frame * stride_cache_frame
                    )
                    values = tl.load(cache_plane + offsets, mask=mask, other=0)
        tl.store(out_ptr + plane * plane_size + offsets, values, mask=mask)
        if has_keep:
            if frame >= keep_start:
                keep_plane = (batch * channels_outer + channel) * keep_frames + (frame - keep_start)
                tl.store(keep_ptr + keep_plane * plane_size + offsets, values, mask=mask)

    @triton.jit
    def _cat_pad_rows_kernel(
        x_ptr,
        cache_ptr,
        out_ptr,
        keep_ptr,
        channels,
        frames,
        height,
        width,
        cache_frames,
        out_frames,
        out_height,
        out_width,
        zero_front_frames,
        pad_height,
        pad_width,
        keep_start,
        keep_frames,
        row_blocks,
        stride_x_batch,
        stride_x_channel,
        stride_x_frame,
        stride_x_height,
        stride_x_width,
        stride_cache_batch,
        stride_cache_channel,
        stride_cache_frame,
        stride_cache_height,
        stride_cache_width,
        has_cache: tl.constexpr,
        has_keep: tl.constexpr,
        ROWS: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        """Channels-first output: one program fills ``ROWS`` consecutive rows of one ``(batch, channel, frame)`` plane.

        All rows of a program share one source (x, cache or causal zero padding),
        selected once with scalar control flow, so each ``[ROWS, BLOCK_W]`` tile is a
        single masked load and a single store; zero frames are stored without
        loading. Row and column coordinates come from the program id and
        ``tl.arange``; there is no per-element division.
        """
        pid = tl.program_id(0).to(tl.int64)
        row_block = pid % row_blocks
        rest = pid // row_blocks
        frame = rest % out_frames
        rest = rest // out_frames
        channel = rest % channels
        batch = rest // channels

        out_rows = row_block * ROWS + tl.arange(0, ROWS)
        row_mask = out_rows < out_height
        in_rows = out_rows - pad_height
        row_valid = row_mask & (in_rows >= 0) & (in_rows < height)
        source_frame = frame - zero_front_frames
        plane = (batch * channels + channel) * out_frames + frame
        out_row_offsets = (plane * out_height + out_rows) * out_width
        if has_keep:
            keep_plane = (batch * channels + channel) * keep_frames + (frame - keep_start)
            keep_row_offsets = (keep_plane * height + in_rows) * width

        for w0 in range(0, out_width, BLOCK_W):
            cols = w0 + tl.arange(0, BLOCK_W)
            col_mask = cols < out_width
            in_cols = cols - pad_width
            interior = row_valid[:, None] & (col_mask & (in_cols >= 0) & (in_cols < width))[None, :]
            values = tl.zeros([ROWS, BLOCK_W], dtype=x_ptr.dtype.element_ty)
            if source_frame >= cache_frames:
                x_plane = (
                    x_ptr
                    + batch * stride_x_batch
                    + channel * stride_x_channel
                    + (source_frame - cache_frames) * stride_x_frame
                )
                values = tl.load(
                    x_plane + in_rows[:, None] * stride_x_height + in_cols[None, :] * stride_x_width,
                    mask=interior,
                    other=0,
                )
            else:
                if has_cache:
                    if source_frame >= 0:
                        cache_plane = (
                            cache_ptr
                            + batch * stride_cache_batch
                            + channel * stride_cache_channel
                            + source_frame * stride_cache_frame
                        )
                        cache_tile = cache_plane + in_rows[:, None] * stride_cache_height
                        values = tl.load(cache_tile + in_cols[None, :] * stride_cache_width, mask=interior, other=0)
            store_mask = row_mask[:, None] & col_mask[None, :]
            tl.store(out_ptr + out_row_offsets[:, None] + cols[None, :], values, mask=store_mask)
            if has_keep:
                if frame >= keep_start:
                    tl.store(keep_ptr + keep_row_offsets[:, None] + in_cols[None, :], values, mask=interior)

    @triton.jit
    def _cat_pad_pixels_kernel(
        x_ptr,
        cache_ptr,
        out_ptr,
        keep_ptr,
        channels,
        frames,
        height,
        width,
        cache_frames,
        out_frames,
        out_height,
        out_width,
        zero_front_frames,
        pad_height,
        pad_width,
        keep_start,
        keep_frames,
        width_blocks,
        stride_x_batch,
        stride_x_channel,
        stride_x_frame,
        stride_x_height,
        stride_x_width,
        stride_cache_batch,
        stride_cache_channel,
        stride_cache_frame,
        stride_cache_height,
        stride_cache_width,
        stride_keep_batch,
        stride_keep_channel,
        stride_keep_frame,
        stride_keep_height,
        stride_keep_width,
        has_cache: tl.constexpr,
        has_keep: tl.constexpr,
        BLOCK_W: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """Channels-last output: one program assembles ``BLOCK_W`` pixels x ``BLOCK_C`` channels of one output row.

        The output is written as contiguous channel spans per pixel. The input and
        the cache refresh go through explicit strides, so a channels-first ``x`` can
        be assembled straight into a channels-last output (the tile is loaded
        pixel-contiguous and stored channel-contiguous, a fused transpose) while the
        cache keeps ``x``'s layout.
        """
        pid = tl.program_id(0).to(tl.int64)
        width_block = pid % width_blocks
        rest = pid // width_blocks
        out_row = rest % out_height
        rest = rest // out_height
        frame = rest % out_frames
        batch = rest // out_frames
        in_row = out_row - pad_height
        row_valid = (in_row >= 0) & (in_row < height)
        source_frame = frame - zero_front_frames

        cols = width_block * BLOCK_W + tl.arange(0, BLOCK_W)
        col_mask = cols < out_width
        in_col = cols - pad_width
        interior = col_mask & row_valid & (in_col >= 0) & (in_col < width)
        out_pixels = (((batch * out_frames + frame) * out_height + out_row) * out_width + cols) * channels
        if has_keep:
            keep_pixels = (
                batch * stride_keep_batch
                + (frame - keep_start) * stride_keep_frame
                + in_row * stride_keep_height
                + in_col * stride_keep_width
            )
        for c0 in range(0, channels, BLOCK_C):
            chans = c0 + tl.arange(0, BLOCK_C)
            chan_mask = chans < channels
            load_mask = interior[:, None] & chan_mask[None, :]
            values = tl.zeros([BLOCK_W, BLOCK_C], dtype=x_ptr.dtype.element_ty)
            if source_frame >= cache_frames:
                x_frame = source_frame - cache_frames
                x_base = x_ptr + batch * stride_x_batch + x_frame * stride_x_frame + in_row * stride_x_height
                values = tl.load(
                    x_base + in_col[:, None] * stride_x_width + chans[None, :] * stride_x_channel,
                    mask=load_mask,
                    other=0,
                )
            else:
                if has_cache:
                    if source_frame >= 0:
                        cache_base = (
                            cache_ptr
                            + batch * stride_cache_batch
                            + source_frame * stride_cache_frame
                            + in_row * stride_cache_height
                        )
                        values = tl.load(
                            cache_base + in_col[:, None] * stride_cache_width + chans[None, :] * stride_cache_channel,
                            mask=load_mask,
                            other=0,
                        )
            store_mask = col_mask[:, None] & chan_mask[None, :]
            tl.store(out_ptr + out_pixels[:, None] + chans[None, :], values, mask=store_mask)
            if has_keep:
                if frame >= keep_start:
                    keep_offsets = keep_pixels[:, None] + chans[None, :] * stride_keep_channel
                    tl.store(keep_ptr + keep_offsets, values, mask=load_mask)

    @triton.jit
    def _dup_up3d_add_rows_kernel(
        main_ptr,
        source_ptr,
        bias_ptr,
        out_ptr,
        out_channels,
        out_frames,
        out_height,
        out_width,
        source_width,
        frame_offset,
        row_blocks,
        stride_main_batch,
        stride_main_channel,
        stride_main_frame,
        stride_main_height,
        stride_source_batch,
        stride_source_channel,
        stride_source_frame,
        stride_source_height,
        FT: tl.constexpr,
        FS: tl.constexpr,
        REPEATS: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        ROWS: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        """Channels-first ``main + DupUp3D(source)``: one program owns ``ROWS`` rows of one output plane.

        The output plane ``(batch, channel, frame)`` maps to one source plane per row
        (``repeats >= FS`` makes the source channel independent of the column), so
        each source row is loaded once as a contiguous span and interleaved with
        itself ``FS`` times; the main tile is loaded and the result stored with
        contiguous rows. Rows are contiguous in ``main``/``out``/``source`` but the
        other strides are arbitrary (frame-major views included).
        """
        pid = tl.program_id(0).to(tl.int64)
        row_block = pid % row_blocks
        rest = pid // row_blocks
        frame = rest % out_frames
        rest = rest // out_frames
        channel = rest % out_channels
        batch = rest // out_channels

        unsliced_frame = frame + frame_offset
        source_frame = unsliced_frame // FT
        temporal_remainder = unsliced_frame % FT
        out_rows = row_block * ROWS + tl.arange(0, ROWS)
        row_mask = out_rows < out_height
        source_rows = out_rows // FS
        height_remainder = out_rows % FS
        source_channel = (((channel * FT + temporal_remainder) * FS + height_remainder) * FS) // REPEATS

        main_rows = (
            batch * stride_main_batch
            + channel * stride_main_channel
            + frame * stride_main_frame
            + out_rows * stride_main_height
        )
        source_row_offsets = (
            batch * stride_source_batch
            + source_channel * stride_source_channel
            + source_frame * stride_source_frame
            + source_rows * stride_source_height
        )
        dtype = main_ptr.dtype.element_ty
        if HAS_BIAS:
            bias = tl.load(bias_ptr + channel).to(tl.float32)
        for w0 in range(0, out_width, BLOCK_W):
            cols = w0 + tl.arange(0, BLOCK_W)
            mask = row_mask[:, None] & (cols < out_width)[None, :]
            offsets = main_rows[:, None] + cols[None, :]
            main = tl.load(main_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            if HAS_BIAS:
                # ATen's in-place conv bias ``add_``: fp32 opmath, rounded once.
                main = (main + bias).to(dtype).to(tl.float32)
            source_cols = w0 // FS + tl.arange(0, BLOCK_W // FS)
            source_mask = row_mask[:, None] & (source_cols < source_width)[None, :]
            source = tl.load(
                source_ptr + (source_row_offsets[:, None] + source_cols[None, :]), mask=source_mask, other=0.0
            )
            if FS >= 2:
                source = tl.interleave(source, source)
            if FS >= 4:
                source = tl.interleave(source, source)
            value = main + source.to(tl.float32)
            tl.store(out_ptr + offsets, value.to(dtype), mask=mask)

    @triton.jit
    def _dup_up3d_add_pixels_kernel(
        main_ptr,
        source_ptr,
        bias_ptr,
        out_ptr,
        out_channels,
        out_frames,
        out_height,
        out_width,
        source_width,
        frame_offset,
        width_blocks,
        stride_main_batch,
        stride_main_frame,
        stride_main_height,
        stride_main_width,
        stride_source_batch,
        stride_source_frame,
        stride_source_height,
        stride_source_width,
        FT: tl.constexpr,
        FS: tl.constexpr,
        REPEATS: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_W: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """Channels-last ``main + DupUp3D(source)``: one program owns ``BLOCK_W`` pixels x all channels of one row.

        Every output pixel reads one source pixel (``col // FS``) whose channel
        vector is gathered with the per-channel map ``source_channel(channel)``
        (a contiguous span when in/out channels match, stride 2 when the block
        halves the channel count); main and out are contiguous along channels.
        """
        pid = tl.program_id(0).to(tl.int64)
        width_block = pid % width_blocks
        rest = pid // width_blocks
        out_row = rest % out_height
        rest = rest // out_height
        frame = rest % out_frames
        batch = rest // out_frames

        unsliced_frame = frame + frame_offset
        source_frame = unsliced_frame // FT
        temporal_remainder = unsliced_frame % FT
        source_row = out_row // FS
        height_remainder = out_row % FS
        cols = width_block * BLOCK_W + tl.arange(0, BLOCK_W)
        col_mask = cols < out_width
        source_cols = cols // FS

        main_pixels = (
            batch * stride_main_batch
            + frame * stride_main_frame
            + out_row * stride_main_height
            + cols * stride_main_width
        )
        source_pixels = (
            batch * stride_source_batch
            + source_frame * stride_source_frame
            + source_row * stride_source_height
            + source_cols * stride_source_width
        )
        dtype = main_ptr.dtype.element_ty
        for c0 in range(0, out_channels, BLOCK_C):
            chans = c0 + tl.arange(0, BLOCK_C)
            chan_mask = chans < out_channels
            mask = col_mask[:, None] & chan_mask[None, :]
            offsets = main_pixels[:, None] + chans[None, :]
            main = tl.load(main_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            if HAS_BIAS:
                bias = tl.load(bias_ptr + chans, mask=chan_mask, other=0.0).to(tl.float32)
                main = (main + bias[None, :]).to(dtype).to(tl.float32)
            source_channels = (((chans * FT + temporal_remainder) * FS + height_remainder) * FS) // REPEATS
            source = tl.load(source_ptr + (source_pixels[:, None] + source_channels[None, :]), mask=mask, other=0.0).to(
                tl.float32
            )
            tl.store(out_ptr + offsets, (main + source).to(dtype), mask=mask)

    @triton.jit
    def _add_bias_residual_kernel(
        x_ptr,
        bias_x_ptr,
        h_ptr,
        bias_h_ptr,
        out_ptr,
        total,
        channels,
        spatial,
        has_bias_x: tl.constexpr,
        has_bias_h: tl.constexpr,
        channels_inner: tl.constexpr,
        index_64bit: tl.constexpr,
        block_size_const: tl.constexpr,
    ):
        """``round(round(x + bias_x) + round(h + bias_h))`` for dense ``(N, C, *spatial)`` tensors.

        Reproduces ATen's per-conv ``output.add_(bias)`` (fp32 opmath, one
        rounding) followed by the residual ``x + h`` (fp32 opmath, one rounding)
        without materializing the biased tensors.
        """
        if index_64bit:
            offsets = tl.program_id(0).to(tl.int64) * block_size_const + tl.arange(0, block_size_const).to(tl.int64)
        else:
            offsets = tl.program_id(0) * block_size_const + tl.arange(0, block_size_const)
        mask = offsets < total
        if channels_inner:
            channel = offsets % channels
        else:
            channel = (offsets // spatial) % channels
        dtype = x_ptr.dtype.element_ty

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        if has_bias_x:
            bias_x = tl.load(bias_x_ptr + channel, mask=mask, other=0.0).to(tl.float32)
            x = (x + bias_x).to(dtype).to(tl.float32)
        h = tl.load(h_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        if has_bias_h:
            bias_h = tl.load(bias_h_ptr + channel, mask=mask, other=0.0).to(tl.float32)
            h = (h + bias_h).to(dtype).to(tl.float32)
        tl.store(out_ptr + offsets, (x + h).to(dtype), mask=mask)


def _uses_channels_last_3d(tensor: torch.Tensor) -> bool:
    return tensor.shape[1] > 1 and tensor.stride(1) == 1


def _next_power_of_2(value: int) -> int:
    return 1 << max(0, value - 1).bit_length()


def _pick_block_width(width: int) -> int:
    """The column block (64/128/256) that pads ``width`` the least; ties go to the wider block."""
    best, best_padded = _ROW_BLOCK_WIDTHS[0], None
    for block in _ROW_BLOCK_WIDTHS:
        padded = -(-width // block) * block
        if best_padded is None or padded <= best_padded:
            best, best_padded = block, padded
    return best


def cat_pad_5d(
    x: torch.Tensor,
    cache_x: torch.Tensor | None,
    padding: list[int] | tuple[int, ...],
    keep_cache_frames: int = 0,
    *,
    channels_last_output: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    """Fuse temporal concat, constant padding, and optional cache refresh.

    ``padding`` follows the Wan causal Conv3d convention
    ``(w_left, w_right, h_top, h_bottom, t_front, t_back)``. The output uses
    the same dense layout that the reference ``cat``/``pad`` path selects, or
    ``channels_last_3d`` when ``channels_last_output`` is set (a fused transpose
    for channels-first ``x``; the cache refresh still follows ``x``'s layout).
    With ``keep_cache_frames > 0`` the last frames of the assembled input
    (without spatial padding) are also written to a second tensor that
    replaces the reference ``x[:, :, -CACHE_T:].clone()`` cache refresh.
    ``cache_x`` may use either layout. Unsupported inputs return ``None`` so
    callers can execute the reference implementation.
    """
    if not HAS_TRITON or len(padding) != 6:
        return None
    pad_width_left, pad_width_right, pad_height_top, pad_height_bottom, pad_front, pad_back = padding
    if any(value < 0 for value in padding):
        return None
    if pad_width_left != pad_width_right or pad_height_top != pad_height_bottom or pad_back != 0:
        return None
    if x.dim() != 5 or not x.is_cuda or x.dtype not in _SUPPORTED_DTYPES:
        return None
    if keep_cache_frames < 0:
        return None

    channels_inner = _uses_channels_last_3d(x)
    out_channels_inner = channels_inner or channels_last_output
    cache_frames = 0
    if cache_x is not None:
        if (
            cache_x.dim() != 5
            or cache_x.dtype != x.dtype
            or cache_x.device != x.device
            or cache_x.shape[0] != x.shape[0]
            or cache_x.shape[1] != x.shape[1]
            or cache_x.shape[3:] != x.shape[3:]
        ):
            return None
        cache_frames = cache_x.shape[2]

    zero_front_frames = pad_front - cache_frames
    if zero_front_frames < 0:
        return None

    batch, channels, frames, height, width = x.shape
    out_frames = pad_front + frames
    out_height = height + 2 * pad_height_top
    out_width = width + 2 * pad_width_left
    keep_frames = min(keep_cache_frames, out_frames)
    out = torch.empty(
        (batch, channels, out_frames, out_height, out_width),
        device=x.device,
        dtype=x.dtype,
        memory_format=torch.channels_last_3d if out_channels_inner else torch.contiguous_format,
    )
    if out.numel() == 0:
        return None

    if keep_frames:
        keep_arg = torch.empty(
            (batch, channels, keep_frames, height, width),
            device=x.device,
            dtype=x.dtype,
            memory_format=torch.channels_last_3d if channels_inner else torch.contiguous_format,
        )
    else:
        keep_arg = out

    if cache_x is None:
        cache_arg = x
        cache_strides = (0, 0, 0, 0, 0)
    else:
        cache_arg = cache_x
        cache_strides = cache_x.stride()

    common = (
        channels,
        frames,
        height,
        width,
        cache_frames,
        out_frames,
        out_height,
        out_width,
        zero_front_frames,
        pad_height_top,
        pad_width_left,
        out_frames - keep_frames,
        keep_frames,
    )
    with torch.get_device_module().device(x.device):
        if out_channels_inner:
            if channels_inner:
                block_c = min(_next_power_of_2(channels), 1024)
                block_w = max(1, _TILE_ELEMENTS // block_c)
            else:
                # Transposing tile: 64 pixels x 64 channels reads 128-byte pixel runs
                # per channel and writes 128-byte channel runs per pixel.
                block_c = min(_next_power_of_2(channels), _TRANSPOSE_BLOCK)
                block_w = _TRANSPOSE_BLOCK
            width_blocks = triton.cdiv(out_width, block_w)
            grid = (batch * out_frames * out_height * width_blocks,)
            _cat_pad_pixels_kernel[grid](
                x,
                cache_arg,
                out,
                keep_arg,
                *common,
                width_blocks,
                *x.stride(),
                *cache_strides,
                *keep_arg.stride(),
                has_cache=cache_x is not None,
                has_keep=keep_frames > 0,
                BLOCK_W=block_w,
                BLOCK_C=block_c,
                num_warps=_NUM_WARPS,
            )
        else:
            # 256-wide column chunks keep 84% of the lanes useful on the 642-wide
            # 720p rows (a 1024-wide block would waste 37%); ROWS rows share the
            # chunk so a program still moves a 4096-element tile per iteration.
            block_w = min(_next_power_of_2(out_width), _ROWS_BLOCK_W)
            rows_per_program = max(1, _TILE_ELEMENTS // block_w)
            row_blocks = triton.cdiv(out_height, rows_per_program)
            grid = (batch * channels * out_frames * row_blocks,)
            _cat_pad_rows_kernel[grid](
                x,
                cache_arg,
                out,
                keep_arg,
                *common,
                row_blocks,
                *x.stride(),
                *cache_strides,
                has_cache=cache_x is not None,
                has_keep=keep_frames > 0,
                ROWS=rows_per_program,
                BLOCK_W=block_w,
                num_warps=_ROWS_NUM_WARPS,
            )

    if keep_cache_frames:
        return out, keep_arg
    return out


def _plane_layout(x: torch.Tensor) -> tuple[int, int, int] | None:
    """``(channels_outer, plane_size, channel_stride)`` if ``x``'s frames are contiguous planes.

    Channels-first (including the frame-major view ``WanResample`` emits): each
    ``(batch, channel, frame)`` plane is ``H * W`` contiguous elements.
    Channels-last: each ``(batch, frame)`` plane is ``H * W * C`` contiguous elements.
    """
    _, channels, _, height, width = x.shape
    if x.stride(4) == 1 and x.stride(3) == width:
        return channels, height * width, x.stride(1)
    if channels > 1 and x.stride(1) == 1 and x.stride(4) == channels and x.stride(3) == width * channels:
        return 1, height * width * channels, 0
    return None


def cat_time_5d(
    x: torch.Tensor,
    cache_x: torch.Tensor | None,
    pad_front: int,
    keep_cache_frames: int = 0,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None:
    """Assemble ``[zeros(pad_front - Tc) | cache | x]`` along time, without spatial padding.

    Returns the assembled tensor (same memory format as ``x``) and, with
    ``keep_cache_frames > 0``, its last frames as the next feature-cache entry.
    Because only whole aligned planes move, this runs at memory bandwidth in
    both layouts; the spatial zero padding is left to the convolution
    (``padding=(0, ph, pw)``), whose bitwise agreement with a pre-padded input
    the caller verifies. Unsupported inputs return ``None``.
    """
    if not HAS_TRITON or x.dim() != 5 or not x.is_cuda or x.dtype not in _SUPPORTED_DTYPES:
        return None
    if pad_front < 0 or keep_cache_frames < 0:
        return None
    layout = _plane_layout(x)
    if layout is None:
        return None
    channels_outer, plane_size, x_channel_stride = layout
    channels_last = channels_outer == 1

    cache_frames = 0
    cache_strides = (0, 0, 0)
    if cache_x is not None:
        cache_layout = _plane_layout(cache_x) if cache_x.dim() == 5 else None
        if (
            cache_layout is None
            or cache_layout[:2] != (channels_outer, plane_size)
            or cache_x.dtype != x.dtype
            or cache_x.device != x.device
            or cache_x.shape[0] != x.shape[0]
            or cache_x.shape[1] != x.shape[1]
            or cache_x.shape[3:] != x.shape[3:]
        ):
            return None
        cache_frames = cache_x.shape[2]
        cache_strides = (cache_x.stride(0), cache_layout[2], cache_x.stride(2))
    zero_front_frames = pad_front - cache_frames
    if zero_front_frames < 0:
        return None

    batch, channels, frames, height, width = x.shape
    out_frames = pad_front + frames
    keep_frames = min(keep_cache_frames, out_frames)
    memory_format = torch.channels_last_3d if channels_last else torch.contiguous_format
    out = torch.empty(
        (batch, channels, out_frames, height, width), device=x.device, dtype=x.dtype, memory_format=memory_format
    )
    if out.numel() == 0:
        return None
    if keep_frames:
        keep_arg = torch.empty(
            (batch, channels, keep_frames, height, width), device=x.device, dtype=x.dtype, memory_format=memory_format
        )
    else:
        keep_arg = out

    grid = (batch * channels_outer * out_frames, triton.cdiv(plane_size, _PLANE_BLOCK))
    with torch.get_device_module().device(x.device):
        _cat_time_planes_kernel[grid](
            x,
            x if cache_x is None else cache_x,
            out,
            keep_arg,
            plane_size,
            channels_outer,
            out_frames,
            cache_frames,
            zero_front_frames,
            out_frames - keep_frames,
            keep_frames,
            x.stride(0),
            x_channel_stride,
            x.stride(2),
            *cache_strides,
            has_cache=cache_x is not None,
            has_keep=keep_frames > 0,
            BLOCK=_PLANE_BLOCK,
            num_warps=_ROWS_NUM_WARPS,
        )
    if keep_cache_frames:
        return out, keep_arg
    return out


def _rows_layout(x: torch.Tensor) -> bool:
    """Rows are contiguous: channels-first tensors and frame-major views."""
    return x.stride(4) == 1


def _pixels_layout(x: torch.Tensor) -> bool:
    """Channels are contiguous and pixels are whole channel vectors (channels_last_3d)."""
    return x.shape[1] > 1 and x.stride(1) == 1 and x.stride(4) == x.shape[1]


def dup_up3d_add(
    main: torch.Tensor,
    source: torch.Tensor,
    factor_temporal: int,
    factor_spatial: int,
    repeats: int,
    drop_first_frames: bool,
    main_bias: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Evaluate ``(main + main_bias) + DupUp3D(source)`` without materializing the shortcut.

    ``DupUp3D`` is ``repeat_interleave`` + an 8-D ``permute().contiguous()``
    + an optional leading-frame slice; every output element is one input
    element at a closed-form index, so the gather is fused with the residual
    add. ``main_bias`` is the un-added per-channel bias of the convolution that
    produced ``main`` (the up block's resample ``Conv2d``); it is applied with
    ATen's rounding before the add. The output keeps ``main``'s strides, as
    the reference add would. Declines (``None``) exotic factors, ``repeats <
    factor_spatial`` (the source channel would vary along a row) and mixed
    layouts.
    """
    if not HAS_TRITON or main.dim() != 5 or source.dim() != 5:
        return None
    if factor_temporal <= 0 or factor_spatial not in (1, 2, 4) or repeats <= 0:
        return None
    if factor_spatial > 1 and not _HAS_INTERLEAVE:
        return None
    if factor_temporal & (factor_temporal - 1) or repeats & (repeats - 1) or repeats < factor_spatial:
        return None
    if not main.is_cuda or not source.is_cuda:
        return None
    if main.dtype != source.dtype or main.dtype not in _SUPPORTED_DTYPES or main.device != source.device:
        return None

    batch, source_channels, frames, height, width = source.shape
    frame_offset = factor_temporal - 1 if drop_first_frames else 0
    numerator = source_channels * repeats
    denominator = factor_temporal * factor_spatial * factor_spatial
    if numerator % denominator:
        return None
    out_channels = numerator // denominator
    expected_shape = (
        batch,
        out_channels,
        frames * factor_temporal - frame_offset,
        height * factor_spatial,
        width * factor_spatial,
    )
    if tuple(main.shape) != expected_shape or main.numel() == 0:
        return None
    if main_bias is not None and (
        main_bias.dim() != 1
        or main_bias.numel() != out_channels
        or main_bias.device != main.device
        or main_bias.dtype not in _SUPPORTED_DTYPES
    ):
        return None

    out = torch.empty_like(main)
    if out.stride() != main.stride():
        return None
    bias_arg = main if main_bias is None else main_bias.contiguous()
    _, _, out_frames, out_height, out_width = expected_shape
    factors = dict(FT=factor_temporal, FS=factor_spatial, REPEATS=repeats, HAS_BIAS=main_bias is not None)

    with torch.get_device_module().device(main.device):
        if _rows_layout(main) and _rows_layout(source):
            block_w = _pick_block_width(out_width)
            rows_per_program = max(1, _TILE_ELEMENTS // block_w)
            row_blocks = triton.cdiv(out_height, rows_per_program)
            grid = (batch * out_channels * out_frames * row_blocks,)
            _dup_up3d_add_rows_kernel[grid](
                main,
                source,
                bias_arg,
                out,
                out_channels,
                out_frames,
                out_height,
                out_width,
                width,
                frame_offset,
                row_blocks,
                *main.stride()[:4],
                *source.stride()[:4],
                **factors,
                ROWS=rows_per_program,
                BLOCK_W=block_w,
                num_warps=_NUM_WARPS,
            )
            return out
        if _pixels_layout(main) and _pixels_layout(source):
            block_c = min(_next_power_of_2(out_channels), 1024)
            block_w = max(1, _TILE_ELEMENTS // block_c)
            width_blocks = triton.cdiv(out_width, block_w)
            grid = (batch * out_frames * out_height * width_blocks,)
            main_strides = (main.stride(0), main.stride(2), main.stride(3), main.stride(4))
            source_strides = (source.stride(0), source.stride(2), source.stride(3), source.stride(4))
            _dup_up3d_add_pixels_kernel[grid](
                main,
                source,
                bias_arg,
                out,
                out_channels,
                out_frames,
                out_height,
                out_width,
                width,
                frame_offset,
                width_blocks,
                *main_strides,
                *source_strides,
                **factors,
                BLOCK_W=block_w,
                BLOCK_C=block_c,
                num_warps=_NUM_WARPS,
            )
            return out
    return None


def _bias_ok(bias: torch.Tensor | None, x: torch.Tensor) -> bool:
    if bias is None:
        return True
    return (
        bias.device == x.device
        and bias.dtype in _SUPPORTED_DTYPES
        and bias.dim() == 1
        and bias.numel() == x.shape[1]
        and bias.is_contiguous()
    )


def add_bias_residual(
    x: torch.Tensor,
    bias_x: torch.Tensor | None,
    h: torch.Tensor,
    bias_h: torch.Tensor | None,
) -> torch.Tensor | None:
    """``(x + bias_x) + (h + bias_h)`` with ATen's rounding points, or ``None`` to decline.

    Folds the per-channel convolution bias adds (``output.add_(bias)``) of a
    residual block's ``conv2`` and ``conv_shortcut`` into the residual add.
    ``x`` and ``h`` must share shape, dtype and a dense layout (both contiguous,
    or both channels-last); either bias may be ``None`` when that operand
    already carries its bias.
    """
    if not HAS_TRITON or not x.is_cuda or x.dim() not in (4, 5) or x.dtype not in _SUPPORTED_DTYPES:
        return None
    if h.shape != x.shape or h.dtype != x.dtype or h.device != x.device:
        return None
    if x.is_contiguous() and h.is_contiguous():
        channels_inner = False
    else:
        memory_format = torch.channels_last_3d if x.dim() == 5 else torch.channels_last
        channels_last = x.is_contiguous(memory_format=memory_format) and h.is_contiguous(memory_format=memory_format)
        if x.shape[1] < 2 or not channels_last:
            return None
        channels_inner = True
    if not (_bias_ok(bias_x, x) and _bias_ok(bias_h, x)):
        return None
    total = x.numel()
    if total == 0 or total > _MAX_INT32 * 4:
        return None
    channels = x.shape[1]
    spatial = total // (x.shape[0] * channels)

    out = torch.empty_like(x)
    grid = (triton.cdiv(total, _BLOCK_SIZE),)
    with torch.get_device_module().device(x.device):
        _add_bias_residual_kernel[grid](
            x,
            x if bias_x is None else bias_x,
            h,
            x if bias_h is None else bias_h,
            out,
            total,
            channels,
            spatial,
            has_bias_x=bias_x is not None,
            has_bias_h=bias_h is not None,
            channels_inner=channels_inner,
            index_64bit=total >= _MAX_INT32,
            block_size_const=_BLOCK_SIZE,
            enable_fp_fusion=False,
        )
    return out


__all__ = ["add_bias_residual", "cat_pad_5d", "cat_time_5d", "dup_up3d_add"]
