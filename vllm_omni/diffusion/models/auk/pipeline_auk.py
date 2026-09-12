# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""AuK audio-editing pipeline for vLLM-Omni.

Turns a layer-fused text condition (produced by the encoder stage) plus an
optional source clip into a 24 kHz waveform: the source clip is encoded to VAE
latents, a rectified-flow DiT integrates the target latents conditioned on
both, and the BigVGAN-flow decoder renders the waveform.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from typing import Any, ClassVar

import numpy as np
import torch
import torchaudio
from safetensors import safe_open
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.auk.auk_transformer import AuKTransformer, dit_state_dict, sample_latents
from vllm_omni.diffusion.models.auk.auk_vae import AuKVAE
from vllm_omni.diffusion.models.interface import (
    SupportAudioInput,
    SupportAudioOutput,
    SupportsComponentDiscovery,
)
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.model_extras.auk import resolve_gen_frames

logger = init_logger(__name__)

# Longest ref + target latent sequence the released checkpoints accept.
MAX_LATENT_FRAMES = 65536

# Weight-file layout: the DiT lives under this prefix; the layer-fusion
# parameters sit next to it but belong to the encoder stage.
_DIT_PREFIX = "transformer."
_FUSION_KEYS = frozenset({"layer_weights", "layer_scale"})

# The distilled variant only reproduces its training recipe at these settings.
_FLASH_NFE = 4
_FLASH_CFG = 0.0
# Decorrelates the VAE posterior draw from the initial latent for the same request seed.
_VAE_SEED_OFFSET = 0x5EED_0A0C


def get_auk_post_process_func(od_config: OmniDiffusionConfig):
    """Create the post-processing function for AuK audio output.

    Tensor output types pass through; anything else becomes a numpy waveform.
    The sample rate is not attached here: the output formatter reads
    ``AuKPipeline.audio_sample_rate`` for audio-output pipelines.
    """

    del od_config  # The conversion does not depend on the config.

    def post_process_func(
        audio: torch.Tensor,
        output_type: str = "np",
    ):
        if output_type in ("pt", "latent"):
            return audio
        return audio.cpu().float().numpy()

    return post_process_func


def _prompt_mapping(prompt: Any) -> dict[str, Any]:
    """Return the prompt dict, or an empty mapping for a bare text prompt."""

    if isinstance(prompt, dict):
        return prompt
    return {}


def _unwrap_single(value: Any) -> Any:
    """Unwrap the one-element lists the serving layer sometimes produces."""

    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _split_audio(audio: Any, default_sample_rate: int) -> tuple[Any, int]:
    """Split a multimodal audio item into raw samples and a sample rate."""

    if isinstance(audio, (tuple, list)) and len(audio) == 2 and isinstance(audio[1], (int, float)):
        return audio[0], int(audio[1])
    return audio, default_sample_rate


class AuKPipeline(nn.Module, SupportAudioInput, SupportAudioOutput, SupportsComponentDiscovery):
    """Instruction-driven audio generation and editing with AuK.

    One request per forward: the rectified-flow ODE runs over the whole target
    span and the batch dimension carries the CFG branches, so there is nothing
    to share between requests yet.

    Args:
        od_config: OmniDiffusion configuration. ``od_config.model`` must be an
            assembled AuK directory (``config.json``, ``auk.safetensors``,
            ``vae.safetensors``), as produced by
            ``tools/prepare_auk_checkpoint.py``.
        prefix: Unused; kept for the pipeline construction contract.
    """

    supports_request_batch = False

    # Picked up by ``supports_audio_output`` in the diffusion engine so the
    # default stage metadata reports ``final_output_type="audio"`` and the
    # ``multimodal_output`` payload includes the sample rate.
    support_audio_output: ClassVar[bool] = True
    support_audio_input: ClassVar[bool] = True
    audio_sample_rate: ClassVar[int] = 24000

    # The DiT cannot run without an encoder-stage text condition, which the
    # engine's synthetic warmup request has no way to produce.
    dummy_run_num_frames: ClassVar[int] = 0

    _dit_modules: ClassVar[list[str]] = ["dit"]
    _encoder_modules: ClassVar[list[str]] = []
    _vae_modules: ClassVar[list[str]] = ["vae"]

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        del prefix  # Weights are not namespaced: one checkpoint, one pipeline.
        self.od_config = od_config
        self.device = get_local_device()
        self.dtype = getattr(od_config, "dtype", None) or torch.bfloat16

        model_dir = od_config.model
        if not model_dir or not os.path.isdir(model_dir):
            raise ValueError(
                f"AuK needs an assembled local checkpoint directory, got {model_dir!r}. "
                "Build one with tools/prepare_auk_checkpoint.py."
            )
        config = _read_config(model_dir)

        vae_config = dict(config["vae"])
        self.latent_dim = int(vae_config["latent_dim"])
        self.hop_size = int(vae_config["downsample_rate"])
        self.sample_rate = int(vae_config["target_sample_rate"])

        self.variant = str(config.get("variant", "base"))
        self.is_flash = self.variant == "flash"
        self.flash_t_grid = [float(t) for t in config.get("flash_t_grid") or ()]
        if self.is_flash and len(self.flash_t_grid) != _FLASH_NFE + 1:
            raise ValueError(f"The flash variant needs a {_FLASH_NFE + 1}-point flash_t_grid, got {self.flash_t_grid}.")
        defaults = dict(config.get("defaults") or {})
        self.default_nfe = int(defaults.get("nfe", 32))
        self.default_cfg = float(defaults.get("cfg", 2.0))
        self.default_sway = defaults.get("sway", -1.0)
        self._flash_lock_logged = False

        # The codec is small and numerically sensitive: it stays in fp32. It
        # must sit on the inference device BEFORE load_weights folds weight
        # norm: a CPU fold lands one ULP off the accelerator's and the encoder
        # amplifies that into a 1e-2 latent drift.
        self.vae = AuKVAE.from_config(vae_config["model_init_kwargs"]).to(device=self.device, dtype=torch.float32)
        self.vae.load_weights(os.path.join(model_dir, "vae.safetensors"))
        self.vae = self.vae.eval()
        self.vae.requires_grad_(False)
        self._check_vae_geometry()

        # The transformer owns every key the checkpoint tool writes into `dit`,
        # `attn_mask_enabled` included, so the section passes straight through.
        self.dit = AuKTransformer(latent_dim=self.latent_dim, **config["dit"])
        self.dit = self.dit.to(dtype=self.dtype)
        self.dit.load_state_dict(_read_dit_weights(model_dir, self.dtype), strict=True)
        self.dit = self.dit.to(device=self.device).eval()
        self.dit.requires_grad_(False)

        logger.info(
            "AuK pipeline ready: variant=%s dtype=%s latent_dim=%d hop=%d sample_rate=%d",
            self.variant,
            self.dtype,
            self.latent_dim,
            self.hop_size,
            self.sample_rate,
        )

    # The assembled checkpoint is not a diffusers layout: __init__ reads
    # auk.safetensors and vae.safetensors directly, so the loader has no
    # component sources to stream and load_weights only reports what is loaded.
    weights_sources: ClassVar[tuple] = ()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stray = [name for name, _ in weights]
        if stray:
            logger.warning("AuKPipeline ignores %d loader-provided weights (e.g. %s)", len(stray), stray[:3])
        return {name for name, _ in self.named_parameters()}

    def _check_vae_geometry(self) -> None:
        """Fail fast when config.json and the VAE disagree on latent geometry."""

        declared = {
            "latent_dim": (self.latent_dim, int(self.vae.latent_dim)),
            "downsample_rate": (self.hop_size, int(self.vae.hop_size)),
            "target_sample_rate": (self.sample_rate, int(self.vae.sample_rate)),
        }
        bad = {name: pair for name, pair in declared.items() if pair[0] != pair[1]}
        if bad:
            raise ValueError(f"config.json and the AuK VAE disagree (config, vae): {bad}")
        if self.sample_rate != self.audio_sample_rate:
            raise ValueError(
                f"AuK advertises {self.audio_sample_rate} Hz output but the checkpoint is {self.sample_rate} Hz."
            )

    def _resolve_generator(self, sampling_params: Any) -> torch.Generator | None:
        """Per-request generator; the process-global RNG is never seeded."""

        generator = sampling_params.generator
        if isinstance(generator, list):
            if len(generator) > 1:
                logger.warning(
                    "AuKPipeline runs one request per forward; using the first of %d generators", len(generator)
                )
            generator = generator[0] if generator else None
        if generator is None and sampling_params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(sampling_params.seed))
        return generator

    def _prepare_waveform(self, audio: Any) -> torch.Tensor:
        """Return mono float32 samples ``[1, T]`` at the VAE sample rate.

        ``T`` is truncated to a whole number of latent frames. The encoder is
        causal, so dropping a trailing partial frame leaves every emitted frame
        unchanged.
        """

        data, sample_rate = _split_audio(audio, self.sample_rate)
        if isinstance(data, np.ndarray):
            wav = torch.from_numpy(np.ascontiguousarray(data))
        else:
            wav = torch.as_tensor(data)
        wav = wav.detach().to(device="cpu", dtype=torch.float32)

        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        elif wav.ndim == 2:
            # Accept both [channels, samples] and soundfile's [samples, channels];
            # the channel axis is the short one (at most 8 wide).
            if wav.shape[0] > wav.shape[1] and wav.shape[1] <= 8:
                wav = wav.transpose(0, 1)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
        else:
            raise ValueError(f"AuK audio input must be [samples] or [channels, samples], got {tuple(wav.shape)}.")

        if wav.shape[-1] == 0:
            raise ValueError("AuK audio input is empty.")
        if not torch.isfinite(wav).all():
            raise ValueError("AuK audio input contains NaN or Inf.")

        if sample_rate != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sample_rate, self.sample_rate)

        frames = wav.shape[-1] // self.hop_size
        if frames < 1:
            raise ValueError(
                f"AuK audio input is shorter than one latent frame ({self.hop_size} samples at {self.sample_rate} Hz)."
            )
        return wav[..., : frames * self.hop_size]

    def _encode_source(
        self,
        audio: Any,
        *,
        sample: bool,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Encode the source clip to normalized latents ``[1, np, latent_dim]``."""

        if audio is None:
            return torch.zeros(1, 0, self.latent_dim, device=self.device, dtype=torch.float32)
        wav = self._prepare_waveform(_unwrap_single(audio)).to(self.device)
        return self.vae.encode(wav, sample=sample, generator=generator)

    def _target_frames(self, gen_seconds: Any, ref_frames: int) -> int:
        """Target latent length, capped by the remaining context."""

        frames = resolve_gen_frames(
            None if gen_seconds is None else float(gen_seconds),
            ref_frames,
            sample_rate=self.sample_rate,
            hop=self.hop_size,
        )
        budget = MAX_LATENT_FRAMES - ref_frames
        if budget < 1:
            raise ValueError(
                f"The source clip already fills the {MAX_LATENT_FRAMES}-frame context ({ref_frames} frames)."
            )
        if frames > budget:
            logger.warning("AuK target of %d frames exceeds the remaining context; clamping to %d.", frames, budget)
            frames = budget
        return frames

    def _resolve_schedule(
        self,
        sampling_params: Any,
        knobs: dict[str, Any],
    ) -> tuple[int, float, float | None, list[float] | None]:
        """Resolve (nfe, cfg, sway, t_grid), locking the distilled recipe on Flash.

        A knob present but ``None`` counts as unspecified: the stage input
        processor fills every key it knows about, whether the caller set it or
        not.
        """

        steps = sampling_params.num_inference_steps
        nfe = self.default_nfe if steps is None else int(steps)
        if nfe < 1:
            raise ValueError(f"num_inference_steps must be >= 1 for AuK; got {steps}")
        cfg = float(sampling_params.guidance_scale) if sampling_params.guidance_scale_provided else self.default_cfg
        sway = knobs.get("sway")
        sway = self.default_sway if sway is None else float(sway)
        raw_grid = knobs.get("t_grid")
        t_grid = [float(t) for t in raw_grid] if raw_grid else None

        if not self.is_flash:
            return nfe, cfg, sway, t_grid

        locked = (_FLASH_NFE, _FLASH_CFG, None, self.flash_t_grid)
        if (nfe, cfg, sway, t_grid) != locked and not self._flash_lock_logged:
            self._flash_lock_logged = True
            logger.warning(
                "AuK-Flash ignores per-request sampling knobs (asked nfe=%s cfg=%s sway=%s t_grid=%s); "
                "using the distilled recipe nfe=%d cfg=%s with the checkpoint time grid.",
                nfe,
                cfg,
                sway,
                t_grid,
                _FLASH_NFE,
                _FLASH_CFG,
            )
        return locked

    def _dit_autocast(self):
        """Match the reference, which runs the DiT under bf16 autocast."""

        return torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.dtype in (torch.bfloat16, torch.float16),
        )

    def forward(self, req: DiffusionRequestBatch) -> list[DiffusionOutput]:
        """Generate one waveform.

        Args:
            req: Request batch holding exactly one request. The prompt carries
                ``prompt_embeds`` (the fused text condition ``[nt, 2048]``), an
                optional ``multi_modal_data["audio"]`` source clip, and
                ``additional_information["auk"]`` with ``gen_seconds``,
                ``sway``, ``t_grid`` and ``vae_sample``. ``num_inference_steps``,
                ``guidance_scale`` and ``seed``/``generator`` come from the
                sampling params.

        Returns:
            One ``DiffusionOutput`` whose ``output`` is a float32 mono waveform
            ``[T]`` at 24 kHz, or the normalized target latents
            ``[1, gen_frames, latent_dim]`` when ``output_type`` is ``latent``.
        """

        assert req.num_reqs == 1, f"AuKPipeline runs one request per forward, got {req.num_reqs}."
        prompt = _prompt_mapping(req.prompts[0])
        sampling_params = req.sampling_params
        knobs = dict((prompt.get("additional_information") or {}).get("auk") or {})

        text = _unwrap_single(prompt.get("prompt_embeds"))
        if text is None:
            raise ValueError("AuK stage 1 needs `prompt_embeds`, the fused text condition from the encoder stage.")
        text = torch.as_tensor(text).to(device=self.device, dtype=self.dtype)
        if text.ndim == 3 and text.shape[0] == 1:
            text = text[0]
        if text.ndim != 2:
            raise ValueError(f"AuK `prompt_embeds` must be [nt, text_hidden_dim], got {tuple(text.shape)}.")
        text = text.unsqueeze(0)
        c_mask = torch.ones(text.shape[:2], dtype=torch.bool, device=self.device)

        generator = self._resolve_generator(sampling_params)
        audio = (prompt.get("multi_modal_data") or {}).get("audio")
        output_type = sampling_params.output_type or "np"

        with torch.inference_mode():
            # The VAE posterior draw gets its own generator, seeded with a
            # fixed offset from the request seed: the initial latent below then
            # still equals the reference's fresh manual_seed draw, and the two
            # noise streams are not copies of each other.
            vae_generator = None
            if knobs.get("vae_sample") and generator is not None:
                vae_seed = (int(generator.initial_seed()) + _VAE_SEED_OFFSET) % (2**63)
                vae_generator = torch.Generator(device=self.device).manual_seed(vae_seed)
            ref = self._encode_source(audio, sample=bool(knobs.get("vae_sample")), generator=vae_generator)
            ref_frames = ref.shape[1]
            ref_mask = torch.ones(ref.shape[:2], dtype=torch.bool, device=self.device)
            gen_frames = self._target_frames(knobs.get("gen_seconds"), ref_frames)
            nfe, cfg, sway, t_grid = self._resolve_schedule(sampling_params, knobs)

            with self._dit_autocast():
                latents = sample_latents(
                    self.dit,
                    text=text,
                    c_mask=c_mask,
                    ref=ref,
                    ref_mask=ref_mask,
                    gen_frames=gen_frames,
                    nfe=nfe,
                    cfg_strength=cfg,
                    sway_sampling_coef=sway,
                    t_grid=t_grid,
                    seed=None,
                    latent_dim=self.latent_dim,
                    device=self.device,
                    dtype=torch.float32,
                    generator=generator,
                )

            latents = latents.float()
            if not torch.isfinite(latents).all():
                raise RuntimeError("AuK generated latents contain NaN or Inf.")
            if output_type == "latent":
                return [DiffusionOutput(output=latents.detach().cpu())]

            wav = self.vae.decode(latents)

        # One mono waveform per request; the formatter expects [T].
        wav = wav.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        if not torch.isfinite(wav).all():
            raise RuntimeError("AuK generated audio contains NaN or Inf.")
        return [DiffusionOutput(output=wav)]


def _read_config(model_dir: str) -> dict[str, Any]:
    """Read the assembled AuK ``config.json``."""

    path = os.path.join(model_dir, "config.json")
    try:
        with open(path) as handle:
            config = json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"AuK checkpoint is missing config.json: {path}") from exc
    except json.JSONDecodeError as exc:
        logger.error("AuK config.json is not valid JSON: %s", path)
        raise ValueError(f"AuK config.json is not valid JSON: {path}") from exc

    missing = [key for key in ("dit", "vae") if key not in config]
    if missing:
        raise ValueError(f"AuK config.json is missing required sections {missing}: {path}")
    return config


def _read_dit_weights(model_dir: str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Read the DiT state dict from ``auk.safetensors``, cast to ``dtype``.

    Tensors are cast one at a time so the fp32 checkpoint is never fully
    resident alongside the model.
    """

    path = os.path.join(model_dir, "auk.safetensors")
    if not os.path.isfile(path):
        raise ValueError(f"AuK checkpoint is missing auk.safetensors: {path}")

    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        keys = list(checkpoint.keys())
        state = dit_state_dict(((key, checkpoint.get_tensor(key).to(dtype)) for key in keys), _DIT_PREFIX)

    if not state:
        raise ValueError(f"AuK checkpoint has no '{_DIT_PREFIX}*' weights: {path}")
    unexpected = sorted(key for key in keys if key not in _FUSION_KEYS and not key.startswith(_DIT_PREFIX))
    if unexpected:
        logger.warning("Ignoring %d unrecognized keys in %s, e.g. %s", len(unexpected), path, unexpected[:5])
    return state
