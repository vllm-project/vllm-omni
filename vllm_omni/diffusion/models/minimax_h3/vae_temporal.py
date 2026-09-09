# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Memory-optimized forks of the checkpoint's temporal VAE loops.

Two method replacements installed on the checkpoint's ``AutoencoderKLLegacy``
at adapter construction:

* ``encode_temporal`` pads misaligned frame counts by concatenating repeated
  last frames onto the *whole* video before the chunk loop (~4GB copy for a
  15s clip). The fork pads inside the tail chunk instead: the chunks see
  bit-identical frames while the copy shrinks to a single clip.

* ``_decode_temporal_streaming`` accumulates the whole decoded video in a
  floating-point buffer that is denormalized, clamped, and quantized to
  uint8 only after the final chunk lands. The fork fuses the adapter's
  in-place revert and the output quantizer into ``write_part`` and
  accumulates directly into a uint8 buffer -- the same per-element op order
  (``(x - mean) / std -> clamp(0, 1) -> *255 -> round -> uint8``), so the
  delivered video is bit-identical while the resident decode buffer shrinks
  4x.

Set ``VLLM_OMNI_VAE_LEGACY_TEMPORAL=1`` to keep the checkpoint's methods.
"""

from __future__ import annotations

import os
from types import MethodType

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_LEGACY_TEMPORAL_ENV = "VLLM_OMNI_VAE_LEGACY_TEMPORAL"


def _legacy_temporal_enabled() -> bool:
    value = os.environ.get(_LEGACY_TEMPORAL_ENV, "0")
    return value.strip().lower() not in ("", "0", "false", "off")


def _encode_temporal_tail_pad(self, x):
    """Drop-in fork of the checkpoint's ``encode_temporal``.

    Frame handling is identical to the original; only the padding site
    moves from a whole-video ``torch.cat`` into the tail chunk.
    """
    offset_frame = 1 if self.isolated_first_frame and self.frame_pre_padding == 0 else 0

    orig_frames = int(x.shape[2])
    pad_size = (offset_frame - orig_frames) % self.clip_length
    padded_frames = orig_frames + pad_size
    num_chunks = (padded_frames - offset_frame) // self.clip_length

    z_list = []
    for i in range(num_chunks):
        start_idx = i * self.clip_length + offset_frame
        end_idx = (i + 1) * self.clip_length + offset_frame
        clip_x = x[:, :, start_idx : min(end_idx, orig_frames), :, :]
        if end_idx > orig_frames:
            # The original pads the whole video up front by repeating the
            # last frame; repeating it inside the tail chunk feeds the
            # identical frames to the identical encodes.
            clip_x = torch.cat(
                [clip_x, clip_x[:, :, -1:].repeat(1, 1, end_idx - orig_frames, 1, 1)],
                dim=2,
            )

        if self.isolated_key_frame:
            key_frame = clip_x[:, :, :1, :, :]
            z_key = self._adaptive_encode(key_frame)

            if clip_x.shape[2] > 1:
                video_frames = clip_x[:, :, 1:, :, :]
                z_video = self._adaptive_encode(video_frames)
                z = torch.cat([z_key, z_video], dim=2)
            else:
                z = z_key
        else:
            z = self._adaptive_encode(clip_x)

        z_list.append(z)

    z = torch.cat(z_list, dim=2)
    if self.token_drop > 0:
        z = z[:, :, : -self.token_drop]

    if self.isolated_first_frame:
        input_first_frame = x[:, :, :1, :, :]
        z_first_frame = self._adaptive_encode(input_first_frame)

        if self.frame_pre_padding == 0:
            z = torch.cat([z_first_frame, z], dim=2)
        else:
            z = torch.cat([z_first_frame, z[:, :, 1:, :, :]], dim=2)

    if self.isolated_last_frame:
        # The original indexes into the padded video: pad frames replicate
        # the last real frame, out-of-range indexes produce an empty slice
        # (a no-op), and negative indexes wrap. Reproduce that mapping
        # against the unpadded tensor instead of materializing the pad.
        last_frame_idx = padded_frames - self.frame_drop + offset_frame
        if last_frame_idx < 0:
            last_frame_idx += padded_frames
        if 0 <= last_frame_idx < padded_frames:
            src_idx = min(last_frame_idx, orig_frames - 1)
            input_last_frame = x[:, :, src_idx : src_idx + 1, :, :]
            z_last_frame = self._adaptive_encode(input_last_frame)
            z = torch.cat([z, z_last_frame], dim=2)

    return z


def _decode_temporal_streaming_uint8(self, z, z_head, z_tail, num_chunks, pad_tokens, temporal_cat_dtype):
    """Drop-in fork of the checkpoint's ``_decode_temporal_streaming``.

    The chunk loop, overlap blending, frame plan, and assertions are
    identical to the original. ``write_part`` additionally runs the
    denormalize -> clamp -> quantize chain in place (the same op order the
    pipeline applies after decode) and accumulates into a uint8 buffer.
    """
    total_frames, pad_frames, output_frames = self._decode_temporal_output_frame_plan(
        z, z_head, z_tail, num_chunks, pad_tokens
    )
    if output_frames <= 0:
        raise ValueError(
            f"decode_temporal streaming planned non-positive output_frames={output_frames} "
            f"total_frames={total_frames} pad_frames={pad_frames}"
        )

    chunk_dec = self.tokens_chunk_size * self.vae_ratio_t
    split_count = int(self.token_drop > 0) + 1
    dec = None
    dec_overlap = None
    norm = None
    write_pos = 0
    logical_frames = 0
    dropped_frames = 0
    decoded_count = 0

    def write_part(part):
        nonlocal dec, dec_overlap, norm, write_pos, logical_frames, dropped_frames
        part_frames = int(part.shape[2])
        if part_frames <= 0:
            return
        logical_frames += part_frames
        if dec is None:
            out_shape = list(part.shape)
            out_shape[2] = output_frames
            dec = torch.empty(out_shape, dtype=torch.uint8, device=part.device)
            transform_rev = self.processor.transform_rev
            mean = torch.as_tensor(transform_rev.mean, dtype=part.dtype, device=part.device).view(1, 3, 1, 1, 1)
            std = torch.as_tensor(transform_rev.std, dtype=part.dtype, device=part.device).view(1, 3, 1, 1, 1)
            norm = (mean, std)
        # revert_tensor ((x - mean) / std, clamp) followed by the output
        # quantizer (*255, round, uint8), in place on the part: the blended
        # raw values are consumed before this point, and every write into
        # ``dec`` goes through here, so the final buffer matches the legacy
        # post-decode chain element for element. Denormalization and clamping
        # run in the decoded dtype exactly like the legacy adapter; the
        # quantizer then runs in FP32 because the legacy adapter returned
        # ``frames.float()`` before the pipeline multiplied and rounded —
        # rounding fp16/bf16 directly flips borderline pixels by one.
        mean_t, std_t = norm
        part.sub_(mean_t).div_(std_t).clamp_(0.0, 1.0)
        part = part.float()
        part.mul_(255.0).round_()

        remaining = int(dec.shape[2]) - write_pos
        copy_frames = min(part_frames, max(0, remaining))
        if copy_frames > 0:
            dec[:, :, write_pos : write_pos + copy_frames, :, :].copy_(part[:, :, :copy_frames, :, :])
            write_pos += copy_frames
        dropped_frames += part_frames - copy_frames

    for i in range(num_chunks):
        t_start_idx = i * self.tokens_chunk_size
        t_end_idx = t_start_idx + self.tokens_chunk_size + self.token_overlap
        clip_z = z[:, :, t_start_idx:t_end_idx, :, :]

        if i == 0 and z_head is not None:
            clip_z = torch.cat([z_head, clip_z], dim=2)

        if i == num_chunks - 1 and z_tail is not None:
            clip_z = torch.cat([clip_z, z_tail], dim=2)

        clip_dec = self._adaptive_decode(clip_z)
        decoded_count += 1
        if temporal_cat_dtype is not None and clip_dec.dtype != temporal_cat_dtype:
            clip_dec = clip_dec.to(temporal_cat_dtype)
        if clip_dec.device != z.device:
            clip_dec = clip_dec.to(z.device)

        dec_tail = None
        if i == 0 and z_head is not None:
            write_part(clip_dec[:, :, self.vae_ratio_t - 1 : self.vae_ratio_t, :, :])
            clip_dec = clip_dec[:, :, self.vae_ratio_t :, :, :]

        if i == num_chunks - 1 and z_tail is not None:
            dec_tail = clip_dec[:, :, -1:, :, :]
            clip_dec = clip_dec[:, :, : -self.vae_ratio_t, :, :]

        for j in range(split_count):
            f_start_idx = j * chunk_dec
            f_end_idx = min(f_start_idx + chunk_dec, clip_dec.shape[2])
            clip_dec_chunk = clip_dec[:, :, f_start_idx:f_end_idx, :, :]
            clip_dec_chunk = clip_dec_chunk[:, :, self.frame_pre_padding :, :, :]

            if j == 0:
                if dec_overlap is not None:
                    clip_dec_chunk = self.blend(dec_overlap, clip_dec_chunk, self.frame_overlap, dim=-3)
                    dec_overlap = None
                write_part(clip_dec_chunk)
            else:
                # Break the view's reference to the full decoded clip so earlier
                # temporal chunks can be released before the final output exists.
                dec_overlap = clip_dec_chunk.contiguous()

        if i == num_chunks - 1:
            if dec_overlap is not None:
                write_part(dec_overlap)
                dec_overlap = None
            if dec_tail is not None:
                write_part(dec_tail)

        del clip_dec, clip_z

    if dec is None:
        raise RuntimeError("decode_temporal streaming produced no output tensor")
    if logical_frames != total_frames or dropped_frames != pad_frames or write_pos != output_frames:
        raise RuntimeError(
            "decode_temporal streaming frame plan mismatch: "
            f"logical_frames={logical_frames} total_frames={total_frames} "
            f"dropped_frames={dropped_frames} pad_frames={pad_frames} "
            f"write_pos={write_pos} output_frames={output_frames}"
        )

    return dec


def _processor_denorm_available(model) -> bool:
    transform_rev = getattr(getattr(model, "processor", None), "transform_rev", None)
    mean = getattr(transform_rev, "mean", None)
    std = getattr(transform_rev, "std", None)
    return mean is not None and std is not None and len(mean) == 3 and len(std) == 3


def install_temporal_stream_patches(model) -> None:
    """Install the memory-optimized temporal forks on the checkpoint model.

    Each fork is installed only when the checkpoint contract it mirrors is
    discoverable; otherwise the checkpoint's own method keeps running (the
    adapter's post-decode revert already covers that path).
    """
    if _legacy_temporal_enabled():
        logger.info("MiniMax-H3 VAE temporal stream patches disabled by %s", _LEGACY_TEMPORAL_ENV)
        return
    if callable(getattr(model, "encode_temporal", None)) and hasattr(model, "clip_length"):
        model.encode_temporal = MethodType(_encode_temporal_tail_pad, model)
        logger.info("MiniMax-H3 VAE encode_temporal tail-padding patch installed")
    else:
        logger.warning("MiniMax-H3 VAE encode_temporal contract not found; keeping checkpoint method")
    if callable(getattr(model, "_decode_temporal_streaming", None)) and _processor_denorm_available(model):
        model._decode_temporal_streaming = MethodType(_decode_temporal_streaming_uint8, model)
        logger.info("MiniMax-H3 VAE streaming uint8 decode patch installed")
    else:
        logger.warning("MiniMax-H3 VAE streaming decode contract not found; keeping checkpoint method")


__all__ = ["install_temporal_stream_patches"]
