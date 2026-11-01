# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for the temporal VAE forks and the uint8 output transfer.

The encode fork must feed bit-identical frames to the checkpoint's chunk
loop (padding moves from a whole-video cat into the tail chunk), the decode
fork must produce the same video as the legacy fp32-buffer chain
(revert -> clamp -> *255 -> round -> uint8), and the output transfer must
match the previous ``.to(uint8, contiguous)`` conversion.
"""

from typing import Any, cast

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
    _prepare_minimax_h3_video_output,
)
from vllm_omni.diffusion.models.minimax_h3.vae_temporal import (
    _decode_temporal_streaming_uint8,
    _encode_temporal_tail_pad,
    install_temporal_stream_patches,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

CLIP = 4


# ---------------------------------------------------------------------------
# encode_temporal fork
# ---------------------------------------------------------------------------


class _EncodeModel:
    """Minimal stand-in implementing the checkpoint's encode_temporal."""

    # Attached per test case as capability probes for the install checks;
    # annotations only, the fork must keep working when they are absent.
    encode_temporal: Any
    _decode_temporal_streaming: Any
    processor: Any

    def __init__(self, *, isolated_first=False, isolated_last=False, key_frame=False, frame_drop=0):
        self.clip_length = CLIP
        self.isolated_first_frame = isolated_first
        self.frame_pre_padding = 0
        self.isolated_key_frame = key_frame
        self.token_drop = 0
        self.isolated_last_frame = isolated_last
        self.frame_drop = frame_drop
        self.encoded = []

    def _adaptive_encode(self, clip):
        self.encoded.append(clip.clone())
        return clip  # identity: z_list entries are the clips themselves

    def reference_encode_temporal(self, x):
        """The checkpoint's original: whole-video pad, then chunk."""
        offset_frame = 1 if self.isolated_first_frame and self.frame_pre_padding == 0 else 0
        if x.shape[2] % self.clip_length != offset_frame:
            pad_size = (offset_frame - x.shape[2]) % self.clip_length
            pad_frames = x[:, :, -1:].repeat(1, 1, pad_size, 1, 1)
            x = torch.cat([x, pad_frames], dim=2)
        num_chunks = (x.shape[2] - offset_frame) // self.clip_length
        z_list = []
        for i in range(num_chunks):
            start_idx = i * self.clip_length + offset_frame
            end_idx = (i + 1) * self.clip_length + offset_frame
            clip_x = x[:, :, start_idx:end_idx, :, :]
            if self.isolated_key_frame:
                key_frame = clip_x[:, :, :1, :, :]
                z_key = self._adaptive_encode(key_frame)
                if clip_x.shape[2] > 1:
                    z_video = self._adaptive_encode(clip_x[:, :, 1:, :, :])
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
            z_first = self._adaptive_encode(x[:, :, :1, :, :])
            z = torch.cat([z_first, z], dim=2)
        if self.isolated_last_frame:
            frame_num = x.shape[2]
            last_idx = frame_num - self.frame_drop + offset_frame
            z_last = self._adaptive_encode(x[:, :, last_idx : last_idx + 1, :, :])
            z = torch.cat([z, z_last], dim=2)
        return z


def _video_x(t, c=3, h=4, w=4):
    return torch.arange(t * c * h * w, dtype=torch.float32).reshape(1, c, t, h, w) / 255.0


@pytest.mark.parametrize("num_frames", [8, 10, 11, 13])
@pytest.mark.parametrize(
    "flags",
    [
        {},
        {"isolated_first": True},
        {"isolated_last": True},
        {"isolated_first": True, "isolated_last": True},
        {"key_frame": True},
    ],
)
def test_encode_fork_matches_whole_video_pad_bitwise(num_frames, flags):
    x = _video_x(num_frames)
    ref_model = _EncodeModel(**flags)
    expected = ref_model.reference_encode_temporal(x.clone())
    # Empty-slice encodes (out-of-range isolated-last indexes are no-ops in
    # the checkpoint) carry no content; compare only real calls.
    ref_calls = [c.clone() for c in ref_model.encoded if c.shape[2] > 0]

    fork_model = _EncodeModel(**flags)
    out = _encode_temporal_tail_pad(fork_model, x.clone())

    assert out.shape == expected.shape
    assert torch.equal(out, expected)
    assert len(fork_model.encoded) == len(ref_calls)
    for got, want in zip(fork_model.encoded, ref_calls):
        assert torch.equal(got, want)


def test_encode_fork_avoids_whole_video_pad_for_misaligned_input():
    # 375 frames with an asymmetric config (isolated last frame only):
    # the checkpoint always pads (375 % 17 != 0); the fork must encode the
    # tail chunk with an in-chunk pad instead of concatenating the video.
    model = _EncodeModel(isolated_last=True, frame_drop=16)
    model.clip_length = 17
    x = _video_x(375, c=1, h=2, w=2)
    _encode_temporal_tail_pad(model, x)
    # 375 % 17 == 1 → pad 16 inside the tail chunk; the isolated last frame
    # resolves to the final real frame (pad frames replicate it).
    assert len(model.encoded) == (375 // 17) + 1 + 1
    assert torch.equal(model.encoded[-1].flatten(), x[:, :, -1:, :, :].flatten())


# ---------------------------------------------------------------------------
# _decode_temporal_streaming uint8 fork
# ---------------------------------------------------------------------------

DENORM_MEAN = (-2.11628412, -2.03571429, -1.80444444)
DENORM_STD = (4.36681223, 4.46428571, 4.44444444)


class _FakeDenorm:
    def __init__(self):
        self.mean = DENORM_MEAN
        self.std = DENORM_STD


class _FakeProcessor:
    def __init__(self):
        self.transform_rev = _FakeDenorm()


class _DecodeModel:
    """Minimal stand-in for the checkpoint's streaming decode contract."""

    def __init__(self, *, token_drop=0, frame_overlap=0):
        self.tokens_chunk_size = 2
        self.vae_ratio_t = 1
        self.token_overlap = 1 if token_drop else 0
        self.token_drop = token_drop
        self.frame_pre_padding = 0
        self.frame_overlap = frame_overlap
        self.processor = _FakeProcessor()

    def _adaptive_decode(self, clip_z):
        # Deterministic pseudo-decode: map latent tokens to frames 1:1.
        return torch.sin(clip_z * 3.0)

    def blend(self, a, b, overlap, dim=-3):
        return (a + b) / 2.0

    def _decode_temporal_pad_frames(self, z, pad_tokens):
        return int(pad_tokens)

    def _decode_temporal_output_frame_plan(self, z, z_head, z_tail, num_chunks, pad_tokens):
        total = num_chunks * self.tokens_chunk_size * self.vae_ratio_t
        if z_head is not None:
            total += 1
        if z_tail is not None:
            total += 1
        if self.token_drop > 0:
            total += self.frame_overlap
        return int(total), int(pad_tokens), int(total - pad_tokens)

    def legacy_streaming(self, z, z_head, z_tail, num_chunks, pad_tokens, temporal_cat_dtype):
        """The checkpoint's original: accumulate in the cat dtype, quantize later."""
        total_frames, _, output_frames = self._decode_temporal_output_frame_plan(
            z, z_head, z_tail, num_chunks, pad_tokens
        )
        chunk_dec = self.tokens_chunk_size * self.vae_ratio_t
        split_count = int(self.token_drop > 0) + 1
        dec = None
        dec_overlap = None
        write_pos = 0

        def write_part(part):
            nonlocal dec, write_pos
            if dec is None:
                out_shape = list(part.shape)
                out_shape[2] = output_frames
                dec = torch.empty(out_shape, dtype=part.dtype)
            n = min(part.shape[2], dec.shape[2] - write_pos)
            if n > 0:
                dec[:, :, write_pos : write_pos + n].copy_(part[:, :, :n])
                write_pos += n

        for i in range(num_chunks):
            t_start = i * self.tokens_chunk_size
            t_end = t_start + self.tokens_chunk_size + self.token_overlap
            clip_z = z[:, :, t_start:t_end]
            clip_dec = self._adaptive_decode(clip_z)
            if temporal_cat_dtype is not None and clip_dec.dtype != temporal_cat_dtype:
                clip_dec = clip_dec.to(temporal_cat_dtype)
            for j in range(split_count):
                chunk = clip_dec[:, :, j * chunk_dec : min(j * chunk_dec + chunk_dec, clip_dec.shape[2])]
                if j == 0:
                    if dec_overlap is not None:
                        chunk = self.blend(dec_overlap, chunk, self.frame_overlap, dim=-3)
                        dec_overlap = None
                    write_part(chunk)
                else:
                    dec_overlap = chunk.contiguous()
            if i == num_chunks - 1 and dec_overlap is not None:
                write_part(dec_overlap)
                dec_overlap = None
        # Legacy post-decode chain: revert in place in the decoded dtype, then
        # the adapter's ``frames.float()`` before the pipeline's quantizer.
        assert dec is not None
        mean = torch.tensor(DENORM_MEAN, dtype=dec.dtype).view(1, 3, 1, 1, 1)
        std = torch.tensor(DENORM_STD, dtype=dec.dtype).view(1, 3, 1, 1, 1)
        dec.sub_(mean).div_(std).clamp_(0.0, 1.0)
        dec = dec.float()
        dec.mul_(255.0).round_()
        return dec.to(torch.uint8)


@pytest.mark.parametrize("token_drop,frame_overlap", [(0, 0), (1, 1)])
@pytest.mark.parametrize("temporal_cat_dtype", [None, torch.float16, torch.bfloat16])
def test_decode_fork_matches_legacy_chain_bitwise(token_drop, frame_overlap, temporal_cat_dtype):
    # The quantizer must see FP32 no matter the cat dtype: the legacy adapter
    # returned frames.float() before the pipeline multiplied and rounded, so
    # an fp16/bf16 cat dtype only narrows denormalization, never the rounding.
    torch.manual_seed(0)
    tokens = 2 * 2 + (1 if token_drop else 0)  # num_chunks * tokens_chunk_size
    z = torch.randn(1, 3, tokens, 4, 4)
    legacy = _DecodeModel(token_drop=token_drop, frame_overlap=frame_overlap)
    expected = legacy.legacy_streaming(z.clone(), None, None, 2, 0, temporal_cat_dtype)

    fork = _DecodeModel(token_drop=token_drop, frame_overlap=frame_overlap)
    out = _decode_temporal_streaming_uint8(fork, z.clone(), None, None, 2, 0, temporal_cat_dtype)
    assert out.dtype == torch.uint8
    assert out.shape == expected.shape
    assert torch.equal(out, expected)


def test_decode_fork_rejects_empty_plan():
    model = _DecodeModel()
    model.tokens_chunk_size = 0
    with pytest.raises(ValueError, match="non-positive output_frames"):
        _decode_temporal_streaming_uint8(model, torch.randn(1, 3, 0, 4, 4), None, None, 0, 0, None)


# ---------------------------------------------------------------------------
# install + escape hatch
# ---------------------------------------------------------------------------


def test_install_binds_both_forks():
    model = _EncodeModel()
    # cast(Any, ...) keeps mypy from narrowing the probes to the lambda type,
    # which has no __func__ for the bound-method assertions below.
    model.encode_temporal = cast(Any, lambda x: x)  # placeholder for the capability check
    model._decode_temporal_streaming = cast(Any, lambda *a: None)
    model.processor = _FakeProcessor()
    install_temporal_stream_patches(model)
    assert model.encode_temporal.__func__ is _encode_temporal_tail_pad
    assert model._decode_temporal_streaming.__func__ is _decode_temporal_streaming_uint8


def test_install_skips_decode_fork_without_denorm_contract():
    model = _EncodeModel()
    model.encode_temporal = cast(Any, lambda x: x)
    model._decode_temporal_streaming = cast(Any, lambda *a: None)
    install_temporal_stream_patches(model)  # no processor -> keep original
    assert model.encode_temporal.__func__ is _encode_temporal_tail_pad
    assert not hasattr(model._decode_temporal_streaming, "__func__")


def test_install_respects_legacy_env(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_VAE_LEGACY_TEMPORAL", "1")
    model = _EncodeModel()
    install_temporal_stream_patches(model)
    assert not hasattr(getattr(model, "encode_temporal", None), "__func__")


# ---------------------------------------------------------------------------
# _prepare_minimax_h3_video_output
# ---------------------------------------------------------------------------


def test_prepare_output_matches_previous_conversion():
    video = torch.randn(1, 3, 4, 6, 8).clamp_(-1, 2)
    legacy = (
        video.detach()
        .float()
        .clamp(0, 1)
        .mul(255)
        .round()
        .permute(0, 2, 3, 4, 1)
        .to(dtype=torch.uint8, memory_format=torch.contiguous_format)
    )
    out = _prepare_minimax_h3_video_output(video.clone())
    assert out.dtype == torch.uint8
    assert out.shape == (1, 4, 6, 8, 3)
    assert out.is_contiguous()
    assert torch.equal(out, legacy)


def test_prepare_output_uint8_passthrough():
    video = torch.randint(0, 256, (1, 3, 4, 6, 8), dtype=torch.uint8)
    out = _prepare_minimax_h3_video_output(video)
    assert out.dtype == torch.uint8
    assert out.is_contiguous()
    assert torch.equal(out, video.permute(0, 2, 3, 4, 1).contiguous())
