#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AuK VAE parity: the numeric oracle for the vendored BigVGAN-flow codec.

Checks ``AuKVAE`` against the AuK reference implementation on real speech, with
both models loaded from the same released ``vae.safetensors``:

  * encoder statistics (mean / log-std) agree,
  * ``encode(sample=False)`` reproduces the reference mean path,
  * ``encode(sample=True)`` reproduces ``encoding_and_normalization`` for a
    matching RNG seed,
  * ``decode`` reproduces ``denormalize`` + ``inference_from_latents``, in fp32
    and (reported, not gated tightly) under bf16 autocast,
  * the waveform survives a mean-path round trip.

Runs on GPU in fp32 and needs both the released checkpoint and the reference
``auk`` package importable, so it is a local_model test rather than CI. Point
``AUK_CKPT_DIR`` at the directory holding ``vae.safetensors`` and ``AUK_REF_WAV``
at a speech clip (default ``$AUK_CKPT_DIR/ref.wav``)::

    AUK_CKPT_DIR=... AUK_REF_WAV=... python -m pytest \
        tests/diffusion/models/auk/test_auk_vae_parity.py -v -s

Unlike its sibling directories this one carries no ``__init__.py`` on purpose: it would put
``tests/diffusion/models`` on ``sys.path`` ahead of site-packages, and this directory's own name
would then shadow the reference ``auk`` package the parity check imports.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import soundfile as sf
import torch
import torchaudio.functional as AF

from vllm_omni.diffusion.models.auk.auk_vae import AuKVAE

pytestmark = [pytest.mark.local_model, pytest.mark.diffusion, pytest.mark.gpu, pytest.mark.cuda]

# The released model.vae.model_init_kwargs, shared by both models under test.
VAE_CONFIG = {
    "upsample_rates": [5, 4, 3, 2, 2, 2],
    "upsample_kernel_sizes": [10, 8, 6, 4, 4, 4],
    "upsample_initial_channel": 1536,
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "downsample_rates": [2, 2, 2, 3, 4, 5],
    "downsample_channels": [12, 24, 48, 96, 192, 384, 768],
    "snake_logscale": True,
    "latent_dim": 64,
    "use_vae": True,
    "causal": True,
    "flow_hidden_channels": 256,
    "act_causal": True,
}

SAMPLE_RATE = 24000
HOP_SIZE = 480
LATENT_DIM = 64
SEED = 1234
# Keep the clip short: the parity run shares its GPU with other jobs.
MAX_SECONDS = 4.0
TOLERANCE = 1e-5
DEVICE = "cuda"

_CKPT_DIR = os.environ.get("AUK_CKPT_DIR")
_VAE_PATH = Path(_CKPT_DIR) / "vae.safetensors" if _CKPT_DIR else None
_REF_WAV = Path(os.environ.get("AUK_REF_WAV") or (Path(_CKPT_DIR) / "ref.wav" if _CKPT_DIR else "unset"))


def _reference_importable() -> bool:
    """Resolve ``auk.model``, not ``auk``.

    An ``__init__.py`` in this directory would make its own name resolve as ``auk`` and satisfy a
    top-level probe, leaving the fixtures to fail at setup instead of skipping. The submodule
    probe fails in that case, which is the honest answer.
    """
    try:
        return importlib.util.find_spec("auk.model") is not None
    except (ImportError, ValueError):
        return False


def _skip_reason() -> str | None:
    if not _CKPT_DIR:
        return "set AUK_CKPT_DIR to the directory holding vae.safetensors"
    if _VAE_PATH is None or not _VAE_PATH.is_file():
        return f"no vae.safetensors under {_CKPT_DIR}"
    if not _REF_WAV.is_file():
        return f"no reference clip at {_REF_WAV} (set AUK_REF_WAV)"
    if not _reference_importable():
        return "cannot import the reference auk.model package (an __init__.py here would shadow it)"
    if not torch.cuda.is_available():
        return "parity runs in fp32 on GPU"
    return None


pytestmark.append(pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or ""))


def _max_abs(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return (lhs.float() - rhs.float()).abs().max().item()


def _snr_db(signal: torch.Tensor, estimate: torch.Tensor) -> float:
    noise = (signal.float() - estimate.float()).pow(2).sum()
    return (10.0 * torch.log10(signal.float().pow(2).sum() / noise)).item()


@pytest.fixture(scope="module")
def clip() -> torch.Tensor:
    """Mono 24 kHz speech, trimmed to a whole number of latent frames."""
    audio, sample_rate = sf.read(str(_REF_WAV), dtype="float32", always_2d=True)  # (T, C)
    wav = torch.from_numpy(audio).mean(dim=1, keepdim=True).transpose(0, 1)
    if sample_rate != SAMPLE_RATE:
        wav = AF.resample(wav, sample_rate, SAMPLE_RATE)
    samples = min(int(MAX_SECONDS * SAMPLE_RATE), wav.size(-1)) // HOP_SIZE * HOP_SIZE
    wav = wav[:, :samples].contiguous().to(DEVICE)
    print(f"\n[clip] {_REF_WAV.name}: {samples} samples, {samples / SAMPLE_RATE:.2f} s, {samples // HOP_SIZE} frames")
    return wav


@pytest.fixture(scope="module")
def reference(clip: torch.Tensor) -> dict[str, torch.Tensor]:
    """Reference tensors from the upstream model, which is then released so one model fits at a time."""
    from auk.model.vae import load_vae_model
    from auk.model.vae.bigvgan_flow_vae import BigVGANFlowVAEConfig

    model = load_vae_model(
        vae_name="BigVGANFlowVAE",
        vae_cfg=BigVGANFlowVAEConfig.from_dict(VAE_CONFIG),
        vae_ckpt=str(_VAE_PATH),
        map_location="cpu",
    )
    model = model.to(DEVICE).eval()
    model.requires_grad_(False)

    with torch.no_grad():
        mean, log_std = model.audio_encoder(clip.unsqueeze(1)).chunk(2, 1)
        scale = torch.sqrt(model.global_log_std.float())
        mean_latents = (mean.transpose(1, 2).float() - model.global_mean.float()) / scale
        torch.manual_seed(SEED)
        sampled_latents, _ = model.encoding_and_normalization(clip.unsqueeze(1))
        decoded = model.inference_from_latents(model.denormalize(mean_latents).permute(0, 2, 1))

    tensors = {
        "mean": mean.clone(),
        "log_std": log_std.clone(),
        "mean_latents": mean_latents.clone(),
        "sampled_latents": sampled_latents.clone(),
        "decoded": decoded.squeeze(1).clone(),
    }
    del model
    torch.accelerator.empty_cache()
    return tensors


@pytest.fixture(scope="module")
def port(reference: dict[str, torch.Tensor]) -> tuple[AuKVAE, list[str], list[str]]:
    """The vendored module, loaded from the same checkpoint. Depends on ``reference`` for load ordering.

    Placed on the accelerator before loading so weight norm is folded there; see ``load_weights``.
    """
    model = AuKVAE.from_config(VAE_CONFIG).to(DEVICE)
    missing, unexpected = model.load_weights(_VAE_PATH)
    model = model.eval()
    model.requires_grad_(False)
    return model, missing, unexpected


def test_checkpoint_covers_every_inference_tensor(port: tuple[AuKVAE, list[str], list[str]]) -> None:
    model, missing, unexpected = port
    print(f"[keys] missing={len(missing)} unexpected={len(unexpected)}")
    print(f"[keys] unexpected prefixes={sorted({key.split('.')[0] for key in unexpected})}")
    assert missing == []
    assert unexpected and all(key.startswith("flow.") for key in unexpected)
    assert (model.hop_size, model.sample_rate, model.latent_dim) == (HOP_SIZE, SAMPLE_RATE, LATENT_DIM)


def test_encoder_statistics_match(
    clip: torch.Tensor, reference: dict[str, torch.Tensor], port: tuple[AuKVAE, list[str], list[str]]
) -> None:
    model = port[0]
    with torch.no_grad():
        mean, log_std = model.audio_encoder(clip.unsqueeze(1)).chunk(2, 1)
    mean_diff = _max_abs(mean, reference["mean"])
    log_std_diff = _max_abs(log_std, reference["log_std"])
    print(f"[encoder] max|dmean|={mean_diff:.3e} max|dlog_std|={log_std_diff:.3e}")
    assert mean_diff < TOLERANCE
    assert log_std_diff < TOLERANCE
    # Folded and hook-computed weights are bit-identical on one device, so the whole encoder is.
    assert (mean_diff, log_std_diff) == (0.0, 0.0)


def test_encode_mean_path_matches(
    clip: torch.Tensor, reference: dict[str, torch.Tensor], port: tuple[AuKVAE, list[str], list[str]]
) -> None:
    model = port[0]
    with torch.no_grad():
        latents = model.encode(clip)
    assert latents.shape == (1, clip.size(-1) // HOP_SIZE, LATENT_DIM)
    diff = _max_abs(latents, reference["mean_latents"])
    print(f"[encode mean] shape={tuple(latents.shape)} max|d|={diff:.3e}")
    assert diff == 0.0


def test_encode_sampled_matches_seeded_reference(
    clip: torch.Tensor, reference: dict[str, torch.Tensor], port: tuple[AuKVAE, list[str], list[str]]
) -> None:
    model = port[0]
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(SEED)
    with torch.no_grad():
        latents = model.encode(clip, sample=True, generator=generator)
    diff = _max_abs(latents, reference["sampled_latents"])
    spread = _max_abs(latents, reference["mean_latents"])
    print(f"[encode sample] max|d vs seeded reference|={diff:.3e} max|d vs mean path|={spread:.3e}")
    assert diff < TOLERANCE
    assert diff == 0.0  # a fresh device Generator replays the default one after torch.manual_seed
    assert spread > TOLERANCE  # the draw actually moved the latents


def test_decode_matches(reference: dict[str, torch.Tensor], port: tuple[AuKVAE, list[str], list[str]]) -> None:
    model = port[0]
    latents = reference["mean_latents"]
    with torch.no_grad():
        wav = model.decode(latents)
    assert wav.shape == (1, latents.size(1) * HOP_SIZE)
    diff = _max_abs(wav, reference["decoded"])
    print(f"[decode fp32] shape={tuple(wav.shape)} max|d|={diff:.3e}")
    assert diff < TOLERANCE
    assert diff == 0.0

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        wav_bf16 = model.decode(latents)
    bf16_diff = _max_abs(wav_bf16, reference["decoded"])
    print(f"[decode bf16 autocast] max|d| vs fp32 reference={bf16_diff:.3e} snr={_snr_db(wav, wav_bf16):.2f} dB")
    assert torch.isfinite(wav_bf16).all()
    assert bf16_diff < 0.5  # sanity only: bf16 is reported, not held to fp32 parity


def test_round_trip_snr(clip: torch.Tensor, port: tuple[AuKVAE, list[str], list[str]]) -> None:
    model = port[0]
    with torch.no_grad():
        wav = model.decode(model.encode(clip))
    assert wav.shape == clip.shape
    snr = _snr_db(clip, wav)
    print(f"[round trip] snr={snr:.2f} dB peak={wav.abs().max().item():.4f}")
    assert snr > 10.0
