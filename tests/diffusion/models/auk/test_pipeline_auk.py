# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AuK stage-1 pipeline: request parsing on CPU, generation parity on a GPU.

The CPU tests stub the transformer, the codec and the ODE so they exercise only
what the pipeline itself owns: how a request turns into a target length, a
sampling schedule and a noise generator.

The parity test replays the saved upstream reference for the zero-shot TTS case
and reports latent, per-frame and waveform distances rather than asserting bit
equality, because two differences live in the conditioning rather than in the
port: the saved text condition is an fp32 thinker forward while the reference
generated under bf16 autocast, and the reference's stochastic VAE draw came from
the process-global RNG seeded at 1234.

It replays with ``vae_sample=False``. That is the deliberate choice: the
reference re-seeds the global RNG immediately before drawing the ODE noise, so a
per-request generator reproduces that noise exactly (verified bit-identical) as
long as nothing else has drawn from it. Sampling the VAE posterior consumes the
generator first, which moves the noise and produces a different, equally valid
realization; comparing latents then measures the noise, not the port. The
posterior mean costs a reference latent that is 0.045 MSE (0.969 cosine) away
from the draw the reference used, which is the smaller of the two errors.
Run it with::

    AUK_OMNI_CKPT_DIR=/path/to/auk-omni-base \
    AUK_PARITY_REF=/path/to/parity_ref/base \
    CUDA_VISIBLE_DEVICES=6 python -m pytest -s \
        tests/diffusion/models/auk/test_pipeline_auk.py

``AUK_OMNI_CKPT_DIR`` is the assembled directory this pipeline loads, which is
not the same thing as the released AuK snapshot that the transformer and codec
parity tests in this directory read from ``AUK_CKPT_DIR``. ``AUK_CKPT_DIR`` is
accepted as a fallback, and the test skips rather than fails when the directory
it resolves to is not an assembled one.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch import nn

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.auk import pipeline_auk
from vllm_omni.diffusion.models.auk.pipeline_auk import AuKPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

LATENT_DIM = 64
HOP = 480
SAMPLE_RATE = 24000
TEXT_HIDDEN_DIM = 2048

# The distilled student's grid, exact values from the released checkpoint.
FLASH_T_GRID = [0.0, 0.07612049579620361, 0.2928932309150696, 0.6173166036605835, 1.0]

CKPT_DIR = os.environ.get("AUK_OMNI_CKPT_DIR") or os.environ.get("AUK_CKPT_DIR")
PARITY_REF = os.environ.get("AUK_PARITY_REF")
# An assembled directory is the only thing this pipeline can load.
IS_ASSEMBLED = bool(CKPT_DIR) and (Path(CKPT_DIR).expanduser() / "config.json").is_file()

# Measured on the released base checkpoint: mean frame cosine 0.961 to 0.968,
# relative latent MSE 0.051 to 0.063, log-mel distance 0.34. That spread is
# bf16 against fp32 DiT weights plus where the codec folds its weight norm,
# neither of which is worth gating on. A port that has lost the conditioning
# sits at 0.110 cosine, 1.82 relative MSE and 2.88 log-mel, so these gates
# still separate the two by an order of magnitude.
PARITY_MIN_FRAME_COSINE = 0.93
PARITY_MAX_RELATIVE_MSE = 0.15
PARITY_MAX_MEL_DISTANCE = 0.6


class _StubTransformer(nn.Module):
    """Records its construction arguments; owns one weight so strict loading runs."""

    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        dim_head: int,
        ff_mult: float,
        latent_dim: int,
        text_hidden_dim: int,
        num_layers: int,
        num_single_layers: int,
        attn_mask_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.init_kwargs = {
            "dim": dim,
            "heads": heads,
            "dim_head": dim_head,
            "ff_mult": ff_mult,
            "latent_dim": latent_dim,
            "text_hidden_dim": text_hidden_dim,
            "num_layers": num_layers,
            "num_single_layers": num_single_layers,
            "attn_mask_enabled": attn_mask_enabled,
        }
        self.latent_dim = latent_dim
        self.proj = nn.Linear(latent_dim, latent_dim, bias=False)


class _StubVAE(nn.Module):
    """Deterministic stand-in for the codec, one latent frame per hop."""

    hop_size = HOP
    sample_rate = SAMPLE_RATE
    latent_dim = LATENT_DIM

    def __init__(self) -> None:
        super().__init__()
        self.encode_calls: list[dict[str, Any]] = []
        self.decode_calls = 0
        self.weights_path: str | None = None
        self.config: dict[str, Any] = {}

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> _StubVAE:
        vae = cls()
        vae.config = dict(cfg)
        return vae

    def load_weights(self, path: str, **_: Any) -> tuple[list[str], list[str]]:
        self.weights_path = str(path)
        return [], []

    def encode(
        self,
        wav: torch.Tensor,
        *,
        sample: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        frames = wav.shape[-1] // HOP
        self.encode_calls.append({"samples": int(wav.shape[-1]), "sample": sample, "generator": generator})
        if sample:
            return torch.randn(1, frames, LATENT_DIM, generator=generator, device=wav.device)
        return torch.zeros(1, frames, LATENT_DIM, device=wav.device)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        self.decode_calls += 1
        return torch.zeros(1, latents.shape[1] * HOP, device=latents.device)


def _stub_sampler(calls: list[dict[str, Any]]):
    """Record the ODE arguments and return noise drawn from the request generator."""

    def sample_latents(dit: nn.Module, **kwargs: Any) -> torch.Tensor:
        calls.append(kwargs)
        shape = (1, kwargs["gen_frames"], kwargs["latent_dim"])
        generator = kwargs.get("generator")
        if generator is None:
            return torch.zeros(*shape, device=kwargs["device"], dtype=kwargs["dtype"])
        return torch.randn(*shape, generator=generator, device=kwargs["device"], dtype=kwargs["dtype"])

    return sample_latents


def _write_checkpoint(root: Path, variant: str) -> Path:
    """Write a tiny assembled checkpoint the stubs can load."""

    from safetensors.torch import save_file

    model_dir = root / f"auk-{variant}"
    model_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "auk",
        "architectures": ["AuKForConditionalGeneration"],
        "dit": {
            "dim": 16,
            "heads": 2,
            "dim_head": 8,
            "ff_mult": 2,
            "text_hidden_dim": TEXT_HIDDEN_DIM,
            "num_layers": 1,
            "num_single_layers": 1,
            "attn_mask_enabled": True,
        },
        "vae": {
            "latent_dim": LATENT_DIM,
            "downsample_rate": HOP,
            "target_sample_rate": SAMPLE_RATE,
            "model_init_kwargs": {"latent_dim": LATENT_DIM},
        },
        "variant": variant,
        "flash_t_grid": FLASH_T_GRID,
        "defaults": {"nfe": 32, "cfg": 2.0, "sway": -1.0},
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    save_file(
        {
            "transformer.proj.weight": torch.zeros(LATENT_DIM, LATENT_DIM),
            "layer_weights": torch.zeros(36),
            "layer_scale": torch.ones(1),
        },
        str(model_dir / "auk.safetensors"),
    )
    save_file({"global_mean": torch.zeros(LATENT_DIM)}, str(model_dir / "vae.safetensors"))
    return model_dir


@pytest.fixture
def build_pipeline(tmp_path, monkeypatch):
    """Build an AuKPipeline whose transformer, codec and ODE are stubs."""

    monkeypatch.setattr(pipeline_auk, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(pipeline_auk, "AuKTransformer", _StubTransformer)
    monkeypatch.setattr(pipeline_auk, "AuKVAE", _StubVAE)

    def _build(variant: str = "base") -> tuple[AuKPipeline, list[dict[str, Any]]]:
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(pipeline_auk, "sample_latents", _stub_sampler(calls))
        od_config = OmniDiffusionConfig(
            model=str(_write_checkpoint(tmp_path, variant)),
            dtype=torch.float32,
            model_class_name="AuKPipeline",
        )
        return AuKPipeline(od_config=od_config), calls

    return _build


def _prompt(
    *,
    tokens: int = 8,
    audio: Any = None,
    knobs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the diffusion prompt the stage input processor would emit."""

    prompt: dict[str, Any] = {
        "prompt": "",
        "prompt_embeds": torch.zeros(tokens, TEXT_HIDDEN_DIM),
        "additional_information": {"auk": dict(knobs or {})},
    }
    if audio is not None:
        prompt["multi_modal_data"] = {"audio": audio}
    return prompt


def _batch(prompt: dict[str, Any], **sampling: Any) -> DiffusionRequestBatch:
    request = OmniDiffusionRequest(
        prompt=prompt,
        sampling_params=OmniDiffusionSamplingParams(**sampling),
        request_id="auk-test-0",
    )
    return DiffusionRequestBatch(requests=[request])


def _silence(seconds: float, sample_rate: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    return np.zeros(int(seconds * sample_rate), dtype=np.float32), sample_rate


@pytest.mark.core_model
@pytest.mark.cpu
class TestRequestParsing:
    def test_pipeline_declares_audio_output(self, build_pipeline):
        pipeline, _ = build_pipeline()

        assert pipeline.support_audio_output is True
        assert pipeline.audio_sample_rate == SAMPLE_RATE
        assert pipeline.supports_request_batch is False
        # The warmup request cannot carry an encoder-stage text condition.
        assert pipeline.dummy_run_num_frames == 0

    def test_config_reaches_the_transformer(self, build_pipeline):
        pipeline, _ = build_pipeline()

        assert pipeline.dit.init_kwargs["latent_dim"] == LATENT_DIM
        assert pipeline.dit.init_kwargs["text_hidden_dim"] == TEXT_HIDDEN_DIM
        assert pipeline.dit.init_kwargs["attn_mask_enabled"] is True

    def test_gen_seconds_sets_the_target_length(self, build_pipeline):
        pipeline, calls = build_pipeline()

        outputs = pipeline.forward(_batch(_prompt(audio=_silence(2.0), knobs={"gen_seconds": 6.0}), seed=1))

        assert calls[0]["gen_frames"] == math.ceil(6.0 * SAMPLE_RATE / HOP)
        assert calls[0]["ref"].shape == (1, 100, LATENT_DIM)
        assert calls[0]["ref_mask"].shape == (1, 100)
        assert calls[0]["c_mask"].shape == (1, 8)
        assert outputs[0].output.shape == (300 * HOP,)
        assert outputs[0].output.dtype is torch.float32

    def test_target_length_falls_back_to_the_source_clip(self, build_pipeline):
        pipeline, calls = build_pipeline()

        pipeline.forward(_batch(_prompt(audio=_silence(2.0), knobs={"gen_seconds": None}), seed=1))

        assert calls[0]["gen_frames"] == 100

    def test_text_only_request_without_a_duration_is_rejected(self, build_pipeline):
        pipeline, calls = build_pipeline()

        with pytest.raises(ValueError, match="gen_seconds"):
            pipeline.forward(_batch(_prompt(knobs={"gen_seconds": None}), seed=1))
        assert calls == []

    def test_text_only_request_gets_an_empty_reference(self, build_pipeline):
        pipeline, calls = build_pipeline()

        pipeline.forward(_batch(_prompt(knobs={"gen_seconds": 3.5}), seed=1))

        assert calls[0]["ref"].shape == (1, 0, LATENT_DIM)
        assert calls[0]["gen_frames"] == 175

    def test_source_clip_is_resampled_to_the_codec_rate(self, build_pipeline):
        pipeline, _ = build_pipeline()

        pipeline.forward(_batch(_prompt(audio=_silence(1.0, 16000), knobs={"gen_seconds": 1.0}), seed=1))

        assert pipeline.vae.encode_calls[0]["samples"] == SAMPLE_RATE

    def test_stereo_source_is_mixed_to_mono(self, build_pipeline):
        pipeline, _ = build_pipeline()
        stereo = np.zeros((2, SAMPLE_RATE), dtype=np.float32)

        pipeline.forward(_batch(_prompt(audio=(stereo, SAMPLE_RATE), knobs={"gen_seconds": 1.0}), seed=1))

        assert pipeline.vae.encode_calls[0]["samples"] == SAMPLE_RATE

    def test_missing_text_condition_is_rejected(self, build_pipeline):
        pipeline, _ = build_pipeline()
        prompt = _prompt(knobs={"gen_seconds": 1.0})
        del prompt["prompt_embeds"]

        with pytest.raises(ValueError, match="prompt_embeds"):
            pipeline.forward(_batch(prompt, seed=1))

    def test_base_variant_uses_the_requested_schedule(self, build_pipeline):
        pipeline, calls = build_pipeline()

        pipeline.forward(
            _batch(
                _prompt(audio=_silence(1.0), knobs={"gen_seconds": 1.0, "sway": -1.0}),
                seed=1,
                num_inference_steps=16,
                guidance_scale=3.0,
            )
        )

        assert calls[0]["nfe"] == 16
        assert calls[0]["cfg_strength"] == 3.0
        assert calls[0]["sway_sampling_coef"] == -1.0
        assert calls[0]["t_grid"] is None

    def test_schedule_defaults_come_from_the_checkpoint(self, build_pipeline):
        pipeline, calls = build_pipeline()

        pipeline.forward(_batch(_prompt(audio=_silence(1.0), knobs={"gen_seconds": 1.0, "sway": None}), seed=1))

        assert calls[0]["nfe"] == 32
        assert calls[0]["cfg_strength"] == 2.0
        assert calls[0]["sway_sampling_coef"] == -1.0

    def test_flash_variant_locks_the_distilled_recipe(self, build_pipeline):
        pipeline, calls = build_pipeline("flash")

        pipeline.forward(
            _batch(
                _prompt(audio=_silence(1.0), knobs={"gen_seconds": 1.0, "sway": -1.0}),
                seed=1,
                num_inference_steps=32,
                guidance_scale=2.0,
            )
        )

        assert calls[0]["nfe"] == 4
        assert calls[0]["cfg_strength"] == 0.0
        assert calls[0]["sway_sampling_coef"] is None
        assert calls[0]["t_grid"] == FLASH_T_GRID

    def test_noise_comes_from_a_seeded_request_generator(self, build_pipeline):
        pipeline, calls = build_pipeline()

        outputs = pipeline.forward(
            _batch(_prompt(knobs={"gen_seconds": 1.0}), seed=0, output_type="latent"),
        )

        expected = torch.randn(1, 50, LATENT_DIM, generator=torch.Generator().manual_seed(0))
        assert torch.equal(outputs[0].output, expected)
        # The global RNG is never seeded on the pipeline's behalf.
        assert calls[0]["seed"] is None
        assert isinstance(calls[0]["generator"], torch.Generator)

    def test_latent_output_type_skips_the_decoder(self, build_pipeline):
        pipeline, _ = build_pipeline()

        outputs = pipeline.forward(_batch(_prompt(knobs={"gen_seconds": 1.0}), seed=1, output_type="latent"))

        assert outputs[0].output.shape == (1, 50, LATENT_DIM)
        assert pipeline.vae.decode_calls == 0

    def test_vae_sample_draws_from_the_request_generator(self, build_pipeline):
        pipeline, _ = build_pipeline()

        pipeline.forward(_batch(_prompt(audio=_silence(1.0), knobs={"gen_seconds": 1.0, "vae_sample": True}), seed=1))

        call = pipeline.vae.encode_calls[0]
        assert call["sample"] is True
        assert isinstance(call["generator"], torch.Generator)

    def test_one_request_per_forward(self, build_pipeline):
        pipeline, _ = build_pipeline()
        batch = _batch(_prompt(knobs={"gen_seconds": 1.0}), seed=1)
        batch.requests.append(batch.requests[0])

        with pytest.raises(AssertionError, match="one request per forward"):
            pipeline.forward(batch)


def _reference_audio_path(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        for item in message.get("content") or ():
            if isinstance(item, dict) and item.get("type") == "audio":
                path = item.get("audio") or item.get("audio_url")
                if path:
                    return str(path)
    raise AssertionError("the saved reference case carries no source clip")


def _log_mel_distance(left: torch.Tensor, right: torch.Tensor, sample_rate: int) -> float:
    """Mean absolute log-mel difference between two mono waveforms."""

    mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, hop_length=256, n_mels=80)
    length = min(left.shape[-1], right.shape[-1])
    spectra = [torch.log(mel(wav[..., :length].float()) + 1e-5) for wav in (left, right)]
    return float((spectra[0] - spectra[1]).abs().mean())


@pytest.mark.local_model
@pytest.mark.diffusion
@pytest.mark.skipif(
    not (IS_ASSEMBLED and PARITY_REF),
    reason="needs AUK_PARITY_REF and AUK_OMNI_CKPT_DIR pointing at an assembled checkpoint directory",
)
def test_zero_shot_tts_tracks_the_upstream_reference():
    """Replay the saved zero-shot TTS case and report the distances."""

    parity_dir = Path(PARITY_REF)
    fusion_dir = Path(os.environ.get("AUK_FUSION_REF") or parity_dir.parent.parent / "fusion_ref")
    reference = torch.load(parity_dir / "zs_tts_en.pt", map_location="cpu", weights_only=False)
    fusion = torch.load(fusion_dir / "zs_tts_audio.pt", map_location="cpu", weights_only=False)

    # soundfile rather than torchaudio.load: the repo's test convention, and
    # torchaudio's decoder needs an ffmpeg the port environment does not have.
    samples, source_rate = sf.read(_reference_audio_path(reference["messages"]), dtype="float32", always_2d=True)
    source = torch.from_numpy(samples).transpose(0, 1)
    knobs = {"gen_seconds": reference["gen_seconds"], "sway": -1.0, "t_grid": None, "vae_sample": False}
    prompt = {
        "prompt": "",
        # The encoder stage emits bf16; the saved fusion is an fp32 forward.
        "prompt_embeds": fusion["fused"].to(torch.bfloat16),
        "multi_modal_data": {"audio": (source, int(source_rate))},
        "additional_information": {"auk": knobs},
    }
    od_config = OmniDiffusionConfig(model=CKPT_DIR, dtype=torch.bfloat16, model_class_name="AuKPipeline")
    pipeline = AuKPipeline(od_config=od_config)

    def run(output_type: str) -> torch.Tensor:
        batch = _batch(
            dict(prompt),
            seed=reference["seed"],
            num_inference_steps=32,
            guidance_scale=2.0,
            output_type=output_type,
        )
        return pipeline.forward(batch)[0].output

    # Two deterministic runs: the ODE is seeded, so the latents belong to the
    # waveform even though the decode happens in the second call.
    latents = run("latent")
    waveform = run("pt")

    ref_frames = int(reference["ref_latent_lens"][0])
    expected = reference["generated"][:, ref_frames:, :].float()
    assert latents.shape == expected.shape
    assert torch.isfinite(latents).all()
    assert torch.isfinite(waveform).all()
    assert waveform.shape[-1] == expected.shape[1] * HOP

    cosine = F.cosine_similarity(latents[0], expected[0], dim=-1)
    metrics = {
        "gen_frames": int(expected.shape[1]),
        "ref_frames": ref_frames,
        "latent_mse": float(torch.mean((latents - expected) ** 2)),
        "latent_relative_mse": float(torch.mean((latents - expected) ** 2) / expected.var()),
        "frame_cosine_mean": float(cosine.mean()),
        "frame_cosine_min": float(cosine.min()),
        "log_mel_distance": _log_mel_distance(waveform, reference["audio"].reshape(-1), int(reference["sr"])),
        "rms_port": float(waveform.pow(2).mean().sqrt()),
        "rms_reference": float(reference["audio"].reshape(-1).pow(2).mean().sqrt()),
    }
    print("\nAuK zero-shot TTS parity:", json.dumps(metrics, indent=1))

    assert metrics["frame_cosine_mean"] > PARITY_MIN_FRAME_COSINE
    assert metrics["latent_relative_mse"] < PARITY_MAX_RELATIVE_MSE
    assert metrics["log_mel_distance"] < PARITY_MAX_MEL_DISTANCE
