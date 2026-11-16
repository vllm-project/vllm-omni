# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for the optional TAEH3 tiny video decoder of MiniMax-H3."""

from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

from vllm_omni.diffusion.models.interface import supports_chunked_vae_decode
from vllm_omni.diffusion.models.minimax_h3 import taeh3 as taeh3_module
from vllm_omni.diffusion.models.minimax_h3 import vae as vae_module
from vllm_omni.diffusion.models.minimax_h3.taeh3 import (
    TAEH3Decoder,
    _MemBlock,
    decode_taeh3_chunks,
    resolve_h3_tae_decoder_path,
    taeh3_decoded_frames,
)
from vllm_omni.diffusion.models.minimax_h3.time_request import _frame_count_from_video_latent_t, _video_latent_t
from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE, _VideoVAEPartProxy

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

CPU = torch.device("cpu")


def _random_state() -> dict[str, torch.Tensor]:
    """Random weights in the published layout: FP16, with the unused encoder alongside."""
    generator = torch.Generator().manual_seed(0)
    shapes: dict[str, tuple[int, ...]] = {"decoder.1.weight": (256, 24, 3, 3), "decoder.1.bias": (256,)}
    for first, channels in ((3, 256), (9, 128), (15, 64)):
        for block in range(first, first + 3):
            shapes[f"decoder.{block}.conv.0.weight"] = (channels, 2 * channels, 3, 3)
            shapes[f"decoder.{block}.conv.0.bias"] = (channels,)
            for conv in (2, 4):
                shapes[f"decoder.{block}.conv.{conv}.weight"] = (channels, channels, 3, 3)
                shapes[f"decoder.{block}.conv.{conv}.bias"] = (channels,)
    shapes.update(
        {
            "decoder.7.conv.weight": (256, 256, 1, 1),
            "decoder.8.weight": (128, 256, 3, 3),
            "decoder.13.conv.weight": (256, 128, 1, 1),
            "decoder.14.weight": (64, 128, 3, 3),
            "decoder.19.conv.weight": (128, 64, 1, 1),
            "decoder.20.weight": (64, 64, 3, 3),
            "decoder.22.weight": (12, 64, 3, 3),
            "decoder.22.bias": (12,),
            "encoder.0.weight": (64, 12, 3, 3),
        }
    )
    state = {}
    for name, shape in shapes.items():
        fan_in = max(1, int(torch.tensor(shape[1:]).prod())) if len(shape) > 1 else 1
        state[name] = (torch.randn(shape, generator=generator) * fan_in**-0.5).half()
    return state


@pytest.fixture(scope="module")
def checkpoint_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("taeh3") / "taeh3.safetensors"
    save_file(_random_state(), str(path))
    return path


@pytest.fixture(scope="module")
def decoder(checkpoint_path):
    return TAEH3Decoder(checkpoint_path, device=CPU)


def _latent(frames: int, *, batch: int = 1, height: int = 2, width: int = 3, seed: int = 0) -> torch.Tensor:
    return torch.randn(batch, 24, frames, height, width, generator=torch.Generator().manual_seed(seed))


def _decode_all_at_once(decoder: TAEH3Decoder, latent: torch.Tensor) -> torch.Tensor:
    """The published reference formulation: one pass over the whole video, then trim."""
    batch = latent.shape[0]
    value = latent.transpose(1, 2).flatten(0, 1)
    for block in decoder.decoder:
        if isinstance(block, _MemBlock):
            frames = value.unflatten(0, (batch, -1))
            past = F.pad(frames, (0, 0, 0, 0, 0, 0, 1, 0))[:, : frames.shape[1]].flatten(0, 1)
            value = block(value, past)
        else:
            value = block(value)
    frames = value.unflatten(0, (batch, -1))
    frames = F.pad(frames, (0, 0, 0, 0, 0, 0, 0, -frames.shape[1] % 20))
    frames = frames.unflatten(1, (-1, 20))[:, :, 3:].flatten(1, 2)[:, :-12]
    return F.pixel_shuffle(frames.flatten(0, 1), 2).unflatten(0, (batch, -1)).clamp(0, 1).transpose(1, 2)


@pytest.mark.parametrize("frames", [5, 22, 39, 124, 362])
def test_frame_count_matches_the_pipeline_latent_contract(frames):
    latent_frames = _video_latent_t(frames)
    assert taeh3_decoded_frames(latent_frames) == frames
    assert taeh3_decoded_frames(latent_frames) == _frame_count_from_video_latent_t(latent_frames)


@pytest.mark.parametrize("latent_frames", [0, 1, 3, 5, 6, 10, -3])
def test_latent_lengths_outside_the_h3_clip_contract_are_rejected(latent_frames):
    with pytest.raises(ValueError, match="5n - 3 latent frames"):
        taeh3_decoded_frames(latent_frames)


def test_weights_in_the_published_layout_load_as_fp32_on_cpu_without_the_encoder(decoder):
    assert decoder.dtype == torch.float32
    assert decoder.device == CPU
    assert not decoder.training
    assert all(not parameter.requires_grad for parameter in decoder.parameters())
    assert not any(name.startswith("encoder") for name, _ in decoder.named_parameters())
    assert sum(parameter.numel() for parameter in decoder.parameters()) == 9_868_236


@pytest.mark.parametrize(("latent_frames", "clips"), [(2, [5]), (7, [17, 5]), (12, [17, 17, 5])])
def test_decode_returns_unit_range_frames_clip_by_clip(decoder, latent_frames, clips):
    latent = _latent(latent_frames)

    video = decoder.decode(latent)
    streamed = list(decoder.iter_clips(latent))

    assert video.shape == (1, 3, sum(clips), 32, 48)
    assert video.dtype == torch.float32
    assert 0.0 <= float(video.min()) and float(video.max()) <= 1.0
    assert [clip.shape[2] for clip in streamed] == clips
    assert torch.equal(torch.cat(streamed, dim=2), video)


@pytest.mark.parametrize(("latent_frames", "batch"), [(2, 1), (12, 1), (7, 2)])
def test_clip_by_clip_decode_matches_the_all_at_once_formulation(decoder, latent_frames, batch):
    """Memory carried between clips must reproduce decoding the video in one pass."""
    latent = _latent(latent_frames, batch=batch, seed=latent_frames)

    torch.testing.assert_close(decoder.decode(latent), _decode_all_at_once(decoder, latent), rtol=0, atol=1e-5)


def test_later_clips_depend_on_earlier_ones(decoder):
    """The second clip of a video is not the first clip of a shorter one."""
    latent = _latent(7, seed=3)

    in_context = list(decoder.iter_clips(latent))[1]
    alone = decoder.decode(latent[:, :, 5:])

    assert in_context.shape == alone.shape
    assert not torch.allclose(in_context, alone, atol=1e-4)


def test_decode_does_not_modify_its_input(decoder):
    latent = _latent(7, seed=5)
    before = latent.clone()
    decoder.decode(latent)
    assert torch.equal(latent, before)


@pytest.mark.parametrize(
    "shape",
    [(24, 7, 2, 2), (1, 16, 7, 2, 2), (1, 7, 24, 2, 2)],
)
def test_decode_rejects_latents_of_the_wrong_layout(decoder, shape):
    with pytest.raises(ValueError, match=r"TAEH3 expects \[B, 24, T, H, W\] latents"):
        decoder.decode(torch.zeros(shape))


def test_checkpoint_validation(tmp_path, checkpoint_path):
    with pytest.raises(ValueError, match="must name a local .safetensors file"):
        TAEH3Decoder(tmp_path / "missing.safetensors", device=CPU)
    with pytest.raises(ValueError, match="must name a local .safetensors file"):
        TAEH3Decoder(tmp_path, device=CPU)

    encoder_only = tmp_path / "encoder.safetensors"
    save_file({"encoder.0.weight": torch.zeros(64, 12, 3, 3)}, str(encoder_only))
    with pytest.raises(ValueError, match="holds no TAEH3 decoder weights"):
        TAEH3Decoder(encoder_only, device=CPU)

    state = _random_state()
    other_family = tmp_path / "taew2_2.safetensors"
    save_file({**state, "decoder.1.weight": torch.zeros(256, 48, 3, 3)}, str(other_family))
    with pytest.raises(ValueError, match="does not take 24-channel latents"):
        TAEH3Decoder(other_family, device=CPU)

    truncated = tmp_path / "truncated.safetensors"
    save_file({name: value for name, value in state.items() if name != "decoder.22.bias"}, str(truncated))
    with pytest.raises(RuntimeError, match="Missing key"):
        TAEH3Decoder(truncated, device=CPU)


def test_optional_path_and_explicit_invalid_values():
    assert resolve_h3_tae_decoder_path(None) is None
    assert resolve_h3_tae_decoder_path({}) is None
    assert resolve_h3_tae_decoder_path({"tae_decoder_path": "/weights/taeh3.safetensors"}) == (
        "/weights/taeh3.safetensors"
    )
    for value in (None, "", "  ", 1, True):
        with pytest.raises(ValueError, match="tae_decoder_path must be a nonempty local file path"):
            resolve_h3_tae_decoder_path({"tae_decoder_path": value})


@pytest.mark.parametrize("other", ["video_vae_decoder", "lightvae_decoder_path"])
def test_tae_rejects_conflicting_decoder_selection(other):
    with pytest.raises(ValueError, match=f"cannot be combined with {other}"):
        resolve_h3_tae_decoder_path({"tae_decoder_path": "taeh3.safetensors", other: "other.safetensors"})
    assert resolve_h3_tae_decoder_path({"tae_decoder_path": "taeh3.safetensors", other: None}) == "taeh3.safetensors"


# --- the VAE adapter -------------------------------------------------------


class _FakeRemoteModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(8, 8)
        self.quant_conv = nn.Linear(4, 4)
        self.post_quant_conv = nn.Linear(4, 4)
        self.decoder = nn.Linear(16, 16)

    def decode_base(self, latent):
        pytest.fail("the checkpoint decoder must not run when the tiny decoder is selected")


class _FakeRemote(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _FakeRemoteModel()


def _patch_checkpoint(monkeypatch) -> _FakeRemote:
    remote = _FakeRemote()
    config = {"latent_channels": 24, "latents_mean": [0.5] * 24, "latents_std": [2.0] * 24}
    monkeypatch.setattr(vae_module, "_load_component_config", lambda _: config)
    monkeypatch.setattr(vae_module, "_load_remote_component", lambda *args, **kwargs: remote)
    return remote


@pytest.mark.parametrize("decode_only", [False, True])
def test_adapter_replaces_the_decode_half_with_the_tiny_decoder(monkeypatch, checkpoint_path, decode_only):
    remote = _patch_checkpoint(monkeypatch)
    optimize = Mock()
    monkeypatch.setattr(vae_module, "install_h3_vae_optimizations", optimize)

    vae = MiniMaxH3VideoVAE(
        "unused", device=CPU, decode_only=decode_only, trust_remote_code=True, tae_decoder_path=str(checkpoint_path)
    )

    assert isinstance(vae.tae_decoder, TAEH3Decoder)
    # The checkpoint's decode half is gone, and nothing was installed on it.
    assert not hasattr(remote.model, "decoder")
    assert not hasattr(remote.model, "post_quant_conv")
    assert hasattr(remote.model, "encoder") is (not decode_only)
    optimize.assert_not_called()
    # The tiny decoder is not a registered submodule: staging and offload of
    # the checkpoint halves must not move it.
    assert "tae_decoder" not in dict(vae.named_children())
    assert all("tae_decoder" not in name for name, _ in vae.named_parameters())
    assert supports_chunked_vae_decode(vae)

    latent = _latent(7, seed=11)
    video = vae.decode_latent(latent)
    # Model-space latents go straight in: no latents_mean / latents_std.
    assert torch.equal(video, vae.tae_decoder.decode(latent))
    assert video.shape == (1, 3, 22, 32, 48)

    chunks: list[torch.Tensor] = []
    vae.decode_with_chunks(latent, on_chunk=chunks.append)
    assert [chunk.shape[2] for chunk in chunks] == [17, 5]
    assert all(chunk.dtype == torch.float32 and chunk.is_contiguous() for chunk in chunks)
    assert torch.equal(torch.cat(chunks, dim=2), video)
    low, high = vae.chunk_value_range
    assert low <= float(video.min()) and float(video.max()) <= high


def test_adapter_without_the_option_keeps_the_checkpoint_decoder(monkeypatch):
    remote = _patch_checkpoint(monkeypatch)
    monkeypatch.setattr(vae_module, "install_h3_vae_optimizations", Mock())

    vae = MiniMaxH3VideoVAE("unused", device=CPU, trust_remote_code=True)

    assert vae.tae_decoder is None
    assert isinstance(remote.model.decoder, nn.Module)
    assert isinstance(remote.model.post_quant_conv, nn.Module)


@pytest.mark.parametrize(
    "conflict",
    [{"decoder_source": "decoder-nvfp4.safetensors"}, {"lightvae_decoder_path": "lite.safetensors"}],
)
def test_adapter_rejects_conflicting_decoders_before_loading(monkeypatch, conflict):
    def unexpected_load(*args, **kwargs):
        pytest.fail("Conflicting decoder selections must fail before loading assets")

    monkeypatch.setattr(vae_module, "_load_component_config", unexpected_load)
    with pytest.raises(ValueError, match="tae_decoder_path cannot be combined with"):
        MiniMaxH3VideoVAE("unused", device=CPU, tae_decoder_path="taeh3.safetensors", **conflict)


def test_encode_only_adapter_ignores_the_tiny_decoder(monkeypatch):
    remote = nn.Module()
    remote.model = nn.Module()
    remote.model.encoder = nn.Linear(2, 2)
    monkeypatch.setattr(
        vae_module,
        "_load_component_config",
        lambda _: {"latent_channels": 2, "latents_mean": [0.0, 0.0], "latents_std": [1.0, 1.0]},
    )
    monkeypatch.setattr(vae_module, "_load_video_vae_encoder", lambda *args: remote)

    vae = MiniMaxH3VideoVAE("unused", device=CPU, encode_only=True, tae_decoder_path="does-not-exist.safetensors")

    assert vae.tae_decoder is None
    with pytest.raises(RuntimeError, match="encode-only video VAE cannot decode"):
        vae.decode_latent(torch.zeros(1, 24, 2, 1, 1))


def test_decoder_residency_requests_do_not_stage_the_encoder(decoder):
    """With the decode half gone, the remaining stager covers the encoder only."""
    vae = object.__new__(MiniMaxH3VideoVAE)
    nn.Module.__init__(vae)
    vae._device_target = CPU
    vae.remote = _FakeRemote()
    vae.model = vae.remote.model
    vae._encoder_stager = None
    vae._decoder_stager = None
    vae._stager = Mock()
    object.__setattr__(vae, "tae_decoder", decoder)

    component = _VideoVAEPartProxy(vae, "decoder")
    component.load_to_device()
    component.offload_to_cpu()
    vae._stager.load.assert_not_called()
    vae._stager.offload.assert_not_called()

    _VideoVAEPartProxy(vae, "encoder").load_to_device()
    vae._stager.load.assert_called_once()


# --- chunk publication across a VAE group ----------------------------------


def _patch_group(monkeypatch, *, rank: int, owner_failed: bool = False) -> list[int]:
    """Simulate one rank of a three-rank VAE group; return what the rank broadcast."""
    sent: list[int] = []

    def fake_broadcast(tensor, src=0, group=None):
        del src, group
        if rank == 0:
            sent.append(int(tensor.item()))
        else:
            tensor.fill_(int(owner_failed))

    monkeypatch.setattr(taeh3_module.dist, "get_rank", lambda _group: rank)
    monkeypatch.setattr(taeh3_module.dist, "get_global_rank", lambda _group, _rank: 0)
    monkeypatch.setattr(taeh3_module.dist, "broadcast", fake_broadcast)
    return sent


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_only_rank_zero_decodes_and_publishes_chunks(monkeypatch, decoder, rank):
    sent = _patch_group(monkeypatch, rank=rank)
    spy = Mock(wraps=decoder.iter_clips)
    monkeypatch.setattr(decoder, "iter_clips", spy)
    seen: list[int] = []

    decode_taeh3_chunks(decoder, _latent(7), lambda frames: seen.append(frames.shape[2]), group=object())

    assert seen == ([17, 5] if rank == 0 else [])
    assert spy.call_count == (1 if rank == 0 else 0)
    assert sent == ([0] if rank == 0 else [])


def test_callback_failure_stops_the_owner_and_is_reported_to_peers(monkeypatch, decoder):
    seen: list[int] = []

    def fail(frames):
        seen.append(frames.shape[2])
        raise RuntimeError("sink failed")

    with pytest.raises(RuntimeError, match="sink failed"):
        decode_taeh3_chunks(decoder, _latent(12), fail, group=None)
    assert seen == [17]

    sent = _patch_group(monkeypatch, rank=0)
    with pytest.raises(RuntimeError, match="sink failed"):
        decode_taeh3_chunks(decoder, _latent(12), fail, group=object())
    assert sent == [1]


def test_peer_ranks_raise_when_the_owner_callback_failed(monkeypatch, decoder):
    _patch_group(monkeypatch, rank=2, owner_failed=True)

    with pytest.raises(RuntimeError, match="chunk callback failed on rank zero"):
        decode_taeh3_chunks(decoder, _latent(7), lambda _frames: None, group=object())


def test_adapter_chunk_decode_keeps_the_group_validation(monkeypatch, decoder):
    """A distributed adapter without a usable group must fail, not publish from every rank."""
    vae = object.__new__(MiniMaxH3VideoVAE)
    nn.Module.__init__(vae)
    vae.parallel_size = 4
    object.__setattr__(vae, "tae_decoder", decoder)
    monkeypatch.setattr(vae_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(MiniMaxH3VideoVAE, "_native_parallel_state", lambda self: {"sp_process_group": None})

    with pytest.raises(RuntimeError, match="invalid spatial-parallel group"):
        vae.decode_with_chunks(_latent(2), on_chunk=lambda _frames: pytest.fail("must not publish"))
