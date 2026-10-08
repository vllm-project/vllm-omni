# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""AuK audio-editing pipeline for vLLM-Omni.

Turns a layer-fused text condition (produced by the encoder stage) plus an
optional source clip into a 24 kHz waveform: the source clip is encoded to VAE
latents, a rectified-flow DiT integrates the target latents conditioned on
both, and the BigVGAN-flow decoder renders the waveform.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import torch
import torchaudio
from safetensors import safe_open
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from vllm.logger import init_logger

from vllm_omni.diffusion.compile import regionally_compile
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.auk.auk_transformer import (
    AuKTransformer,
    build_time_grid,
    dit_state_dict,
    sample_latents,
)
from vllm_omni.diffusion.models.auk.auk_vae import AuKVAE
from vllm_omni.diffusion.models.auk.cudagraph_wrapper import AuKCUDAGraphWrapper
from vllm_omni.diffusion.models.auk.fp8_linear import fp8_supported, quantize_block_linears
from vllm_omni.diffusion.models.auk.vae_cudagraph import AuKVAEDecodeGraph
from vllm_omni.diffusion.models.interface import (
    SupportAudioInput,
    SupportAudioOutput,
    SupportsComponentDiscovery,
)
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.model_extras.auk import resolve_gen_frames
from vllm_omni.quantization.component_config import ComponentQuantizationConfig, resolve_component_quant_config

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
# Startup DiT warm-up shapes: target frames of 3 / 6 / 12 s requests, with no
# reference (instruct TTS) and a 3 s one (zero-shot). The first call compiles
# the blocks; each shape then captures its graph bucket. Other shapes still
# capture lazily, at about a second each.
_WARMUP_TARGET_FRAMES = (150, 300, 600)
_WARMUP_REF_FRAMES = (0, 150)
_WARMUP_TEXT_TOKENS = 96
# Reference clips whose posterior-mean latents are kept. Voice cloning reuses a
# small set of reference voices, so repeat requests skip the VAE encode.
# ``model_config.auk_ref_cache_size`` overrides it; 0 disables the cache.
_REF_CACHE_SIZE = 16
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


def _dit_fp8_config(quant_config: Any) -> Any | None:
    """The stage's FP8 config for the DiT, or None when the DiT stays unquantized.

    A per-component config is resolved to its ``dit`` entry first. Only the
    DiT can be quantized, so a component config naming any other part of the
    pipeline, any method other than ``fp8``, and FP8 options this pipeline
    cannot honour are rejected rather than silently ignored.
    """
    if quant_config is None:
        return None
    if isinstance(quant_config, ComponentQuantizationConfig):
        components = quant_config.component_configs
        # The whole DiT takes one config; per-layer opt-outs go through ignored_layers.
        scoped = [p for p in components if p != "dit" and p.split(".")[0] == "dit"]
        if scoped:
            raise ValueError(
                f"AuK quantizes the DiT as one component and does not support the sub-scopes {scoped}; "
                "configure 'dit' and list the linears to keep in the model dtype in its ignored_layers"
            )
        others = [p for p, c in components.items() if c is not None and p != "dit"]
        if others:
            raise ValueError(f"AuK quantizes only the DiT; set components {others} to null")
    quant_config = resolve_component_quant_config(quant_config, "dit")
    if quant_config is None:
        return None
    name = quant_config.get_name() if hasattr(quant_config, "get_name") else str(quant_config)
    if name != "fp8":
        raise ValueError(f"AuK supports only 'fp8' quantization of the DiT, got {name!r}")
    unsupported = []
    if getattr(quant_config, "is_checkpoint_fp8_serialized", False):
        unsupported.append("is_checkpoint_fp8_serialized")
    if getattr(quant_config, "weight_block_size", None):
        unsupported.append("weight_block_size")
    if getattr(quant_config, "activation_scheme", "dynamic") != "dynamic":
        unsupported.append("activation_scheme")
    if unsupported:
        raise ValueError(f"AuK's online FP8 DiT does not support the FP8 options {unsupported}")
    return quant_config


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


_ScheduleKey = tuple[int, float, float | None, tuple[float, ...] | None, int]


@dataclass
class _RequestCondition:
    index: int
    text: torch.Tensor
    ref: torch.Tensor
    generator: torch.Generator | None
    output_type: str


def _pad_conditions(tensors: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad request-local conditions and mask every padded token."""
    if len(tensors) == 1:
        batch = tensors[0].unsqueeze(0)
        return batch, torch.ones(batch.shape[:2], dtype=torch.bool, device=batch.device)
    batch = pad_sequence(tensors, batch_first=True)
    lengths = torch.tensor([tensor.shape[0] for tensor in tensors], device=batch.device)
    mask = torch.arange(batch.shape[1], device=batch.device)[None, :] < lengths[:, None]
    return batch, mask


class AuKPipeline(nn.Module, SupportAudioInput, SupportAudioOutput, SupportsComponentDiscovery):
    """Instruction-driven audio generation and editing with AuK.

    Compatible requests share a DiT/VAE pass while retaining their own
    conditioning, noise generator and output position.

    Args:
        od_config: OmniDiffusion configuration. ``od_config.model`` must be an
            assembled AuK directory (``config.json``, ``auk.safetensors``,
            ``vae.safetensors``), as produced by
            ``tools/prepare_auk_checkpoint.py``.
        prefix: Unused; kept for the pipeline construction contract.
    """

    supports_request_batch = True

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
        max_dit_graphs = od_config.model_config.get("max_dit_graphs", 32)
        if isinstance(max_dit_graphs, bool) or not isinstance(max_dit_graphs, int) or max_dit_graphs < 1:
            raise ValueError("AuK max_dit_graphs must be a positive integer")
        ref_cache_size = od_config.model_config.get("auk_ref_cache_size", _REF_CACHE_SIZE)
        if isinstance(ref_cache_size, bool) or not isinstance(ref_cache_size, int) or ref_cache_size < 0:
            raise ValueError("AuK auk_ref_cache_size must be a non-negative integer (0 disables the cache)")
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
        self.text_hidden_dim = int(config["dit"]["text_hidden_dim"])
        self.dit = AuKTransformer(latent_dim=self.latent_dim, **config["dit"])
        self.dit = self.dit.to(dtype=self.dtype)
        self.dit.load_state_dict(_read_dit_weights(model_dir, self.dtype), strict=True)
        self.dit = self.dit.to(device=self.device).eval()
        self.dit.requires_grad_(False)
        # Opt-in FP8 GEMMs for the block linears (stage `diffusion_quantization_config: fp8`).
        # The swap happens before compilation and graph capture, which then see the FP8 modules.
        self.dit_fp8 = False
        fp8_config = _dit_fp8_config(getattr(od_config, "quantization_config", None))
        if fp8_config is not None:
            if fp8_supported(self.device):
                count = quantize_block_linears(
                    self.dit,
                    ignored_layers=list(getattr(fp8_config, "ignored_layers", None) or ()),
                    match_mode=getattr(fp8_config, "ignored_layers_match_mode", "exact"),
                )
                self.dit_fp8 = True
                logger.info("AuK DiT: %d block linears run as FP8 GEMMs", count)
            else:
                logger.warning(
                    "AuK FP8 quantization needs an Ada or Hopper CUDA device; the DiT stays %s on %s",
                    self.dtype,
                    self.device,
                )
        self.cudagraph_wrapper = AuKCUDAGraphWrapper(
            self.dit, enabled=not od_config.enforce_eager, max_graphs=max_dit_graphs
        )
        # The compiled decode buckets are warmed by setup_compile(), which the
        # model runner calls at startup unless the stage is enforce_eager. The
        # bucket list and the plain-graph cache size come from the stage's
        # model_config, like the other codec graph wrappers' knobs.
        model_config = getattr(od_config, "model_config", None) or {}
        vae_decode_kwargs: dict[str, Any] = {}
        if model_config.get("auk_vae_compile_shapes") is not None:
            vae_decode_kwargs["compile_shapes"] = [int(size) for size in model_config["auk_vae_compile_shapes"]]
        if model_config.get("auk_vae_max_graphs") is not None:
            vae_decode_kwargs["max_graphs"] = int(model_config["auk_vae_max_graphs"])
        if model_config.get("auk_vae_tile_frames") is not None:
            vae_decode_kwargs["tile_frames"] = int(model_config["auk_vae_tile_frames"])
        self.vae_decode = AuKVAEDecodeGraph(self.vae, enabled=not od_config.enforce_eager, **vae_decode_kwargs)
        self._vae_warmup_batches = self._vae_warmup_batch_sizes(od_config, model_config)
        self._ref_cache_size = ref_cache_size
        self._ref_cache: OrderedDict[str, torch.Tensor] = OrderedDict()

        logger.info(
            "AuK pipeline ready: variant=%s dtype=%s latent_dim=%d hop=%d sample_rate=%d",
            self.variant,
            self.dtype,
            self.latent_dim,
            self.hop_size,
            self.sample_rate,
        )

    def setup_compile(self) -> None:
        """Compile the DiT as configured, then capture the codec decode buckets.

        Defining this hook replaces the runner's generic transformer compile,
        so the DiT's ``diffusion_compile_granularity`` is honoured here: full
        compiles the whole denoise step;
        regional compiles the double- and single-stream blocks. Compilation is
        lazy and happens in the DiT CUDA graph's warm-up, before capture, so
        each graph replays the compiled kernels. The startup cost worth paying
        is the VAE decode, whose buckets are compiled and captured before the
        first request.
        """
        granularity = self.od_config.diffusion_compile_granularity
        dynamic = self.od_config.diffusion_compile_dynamic
        if granularity == "full":
            # The samplers drive prepare() and step() directly rather than
            # __call__, so the per-step body is what gets compiled.
            self.dit.step = torch.compile(self.dit.step, dynamic=dynamic)
        else:
            regionally_compile(self.dit, dynamic=dynamic)
        logger.info("AuK DiT configured for lazy %s torch.compile with dynamic=%s", granularity, dynamic)
        self._warmup_dit()
        self.vae_decode.warmup(self.device, batch_sizes=self._vae_warmup_batches)

    @staticmethod
    def _vae_warmup_batch_sizes(od_config: OmniDiffusionConfig, model_config: dict[str, Any]) -> tuple[int, ...]:
        """B=1 compiled graphs plus a compiled graph at max_num_seqs (or an explicit list)."""
        explicit = model_config.get("auk_vae_warmup_batches")
        if explicit is not None:
            return tuple(int(b) for b in explicit if int(b) > 0)
        max_bs = max(1, int(od_config.max_num_seqs))
        return (1, max_bs) if max_bs > 1 else (1,)

    def _warmup_dit(self) -> None:
        """Compile the DiT step and capture the common graph buckets before serving.

        The engine's synthetic warm-up request cannot reach this pipeline
        (``dummy_run_num_frames = 0``), so without this the first real request
        pays the whole torch.compile. Runs the variant's default guidance path
        on zero conditioning; ``model_config.auk_dit_warmup_frames`` overrides
        the target lengths and ``[]`` disables the warm-up.

        Serving 4-step TTS is a different graph key from the checkpoint's
        default 32-step recipe, and C=128 is a different batch from B=1. Those
        shapes are captured here so the first packet is a replay, not a capture.
        """
        if not self.cudagraph_wrapper.enabled:
            return
        model_config = getattr(self.od_config, "model_config", None) or {}
        frames = model_config.get("auk_dit_warmup_frames", _WARMUP_TARGET_FRAMES)
        if not frames:
            logger.info("AuK DiT startup warm-up disabled")
            return
        cfg = _FLASH_CFG if self.is_flash else self.default_cfg
        if self.is_flash:
            default_grid = torch.tensor(self.flash_t_grid, device=self.device, dtype=torch.float32)
        else:
            sway = None if self.default_sway is None else float(self.default_sway)
            default_grid = build_time_grid(
                nfe=self.default_nfe, sway_sampling_coef=sway, t_grid=None, device=self.device
            )
        start = time.perf_counter()
        n_shapes = 0
        with torch.inference_mode(), self._dit_autocast():
            for ref_frames in _WARMUP_REF_FRAMES:
                for target_frames in frames:
                    self._run_dit_warmup_shape(
                        batch=1,
                        target_frames=int(target_frames),
                        ref_frames=int(ref_frames),
                        text_tokens=_WARMUP_TEXT_TOKENS,
                        grid=default_grid,
                        cfg=cfg,
                    )
                    n_shapes += 1

            serve_nfe = int(model_config.get("auk_dit_warmup_nfe", self.default_nfe))
            extra_nfe = (not self.is_flash) and serve_nfe > 0 and serve_nfe != self.default_nfe
            if extra_nfe:
                sway = None if self.default_sway is None else float(self.default_sway)
                serve_grid = build_time_grid(nfe=serve_nfe, sway_sampling_coef=sway, t_grid=None, device=self.device)
                serve_cfg = self.default_cfg
                for target_frames in frames:
                    self._run_dit_warmup_shape(
                        batch=1,
                        target_frames=int(target_frames),
                        ref_frames=0,
                        text_tokens=_WARMUP_TEXT_TOKENS,
                        grid=serve_grid,
                        cfg=serve_cfg,
                    )
                    n_shapes += 1
            else:
                serve_grid = default_grid
                serve_cfg = cfg

            dit_batches = model_config.get("auk_dit_warmup_batches")
            if dit_batches is not None:
                extra_batches = sorted({int(b) for b in dit_batches if int(b) > 1})
            else:
                extra_batches = [int(b) for b in self._vae_warmup_batches if int(b) > 1]
            serve_frames = int(frames[0])
            for batch in extra_batches:
                self._run_dit_warmup_shape(
                    batch=batch,
                    target_frames=serve_frames,
                    ref_frames=0,
                    text_tokens=_WARMUP_TEXT_TOKENS,
                    grid=serve_grid,
                    cfg=serve_cfg,
                )
                n_shapes += 1
        if self.device.type != "cpu":
            torch.accelerator.synchronize(self.device)
        logger.info(
            "AuK DiT warm-up: compiled and captured %d shapes in %.1f s",
            n_shapes,
            time.perf_counter() - start,
        )

    def _run_dit_warmup_shape(
        self,
        *,
        batch: int,
        target_frames: int,
        ref_frames: int,
        text_tokens: int,
        grid: torch.Tensor,
        cfg: float,
    ) -> None:
        """Capture one DiT CUDA-graph shape (full Euler loop when the wrapper has one)."""
        text = torch.zeros(batch, text_tokens, self.text_hidden_dim, device=self.device, dtype=self.dtype)
        c_mask = torch.ones(text.shape[:2], dtype=torch.bool, device=self.device)
        ref = torch.zeros(batch, ref_frames, self.latent_dim, device=self.device, dtype=torch.float32)
        ref_mask = torch.ones(ref.shape[:2], dtype=torch.bool, device=self.device)
        x = torch.zeros(batch, target_frames, self.latent_dim, device=self.device, dtype=torch.float32)
        self.cudagraph_wrapper.sample_loop(
            x=x,
            text=text,
            c_mask=c_mask,
            ref=ref,
            ref_mask=ref_mask,
            timesteps=grid,
            cfg_strength=cfg,
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

    def _resolve_generator(self, sampling_params: Any) -> torch.Generator:
        """Per-request generator; unseeded requests draw deterministically in request order."""

        generator = sampling_params.generator
        if isinstance(generator, list):
            if len(generator) > 1:
                logger.warning(
                    "AuKPipeline runs one request per forward; using the first of %d generators", len(generator)
                )
            generator = generator[0] if generator else None
        if generator is None:
            if sampling_params.seed is not None:
                seed = int(sampling_params.seed)
            else:
                # Draw seed in request arrival order before schedule grouping so
                # unseeded requests draw independent noise regardless of schedule order.
                seed = int(torch.empty((), dtype=torch.int64).random_().item()) & 0x7FFFFFFFFFFFFFFF
            generator = torch.Generator(device=self.device).manual_seed(seed)
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
        wav = self._prepare_waveform(_unwrap_single(audio))
        # The posterior mean is a pure function of the prepared waveform, so it
        # is cached by content. A posterior draw depends on the generator and
        # is always recomputed. Callers must not modify the returned tensor.
        key = None
        if not sample and self._ref_cache_size:
            digest = hashlib.blake2b(wav.numpy().tobytes(), digest_size=16).hexdigest()
            key = f"{wav.shape[-1]}:{digest}"
            cached = self._ref_cache.get(key)
            if cached is not None:
                self._ref_cache.move_to_end(key)
                return cached
        latents = self.vae.encode(wav.to(self.device), sample=sample, generator=generator)
        if key is not None:
            self._ref_cache[key] = latents
            while len(self._ref_cache) > self._ref_cache_size:
                self._ref_cache.popitem(last=False)
        return latents

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

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch) -> list[DiffusionOutput]:
        """Batch requests with matching target length and Euler schedule.

        Text and reference lengths may differ; their padding is masked. Seeds,
        posterior sampling and output types remain request-local.
        """
        groups: dict[_ScheduleKey, list[_RequestCondition]] = {}
        for index, request in enumerate(req.requests):
            prompt = _prompt_mapping(request.prompt)
            sampling_params = request.sampling_params
            knobs = dict((prompt.get("additional_information") or {}).get("auk") or {})
            raw_text = _unwrap_single(prompt.get("prompt_embeds"))
            if raw_text is None:
                raise ValueError("AuK stage 1 needs `prompt_embeds`, the fused text condition from the encoder stage.")
            text = torch.as_tensor(raw_text).to(device=self.device, dtype=self.dtype)
            if text.ndim == 3 and text.shape[0] == 1:
                text = text[0]
            if text.ndim != 2:
                raise ValueError(f"AuK `prompt_embeds` must be [nt, text_hidden_dim], got {tuple(text.shape)}.")

            generator = self._resolve_generator(sampling_params)
            # Keep the VAE posterior draw independent of the request's DiT noise.
            vae_generator = None
            if knobs.get("vae_sample") and generator is not None:
                vae_seed = (int(generator.initial_seed()) + _VAE_SEED_OFFSET) % (2**63)
                vae_generator = torch.Generator(device=self.device).manual_seed(vae_seed)
            ref = self._encode_source(
                (prompt.get("multi_modal_data") or {}).get("audio"),
                sample=bool(knobs.get("vae_sample")),
                generator=vae_generator,
            )
            gen_frames = self._target_frames(knobs.get("gen_seconds"), ref.shape[1])
            nfe, cfg, sway, t_grid = self._resolve_schedule(sampling_params, knobs)
            key = (nfe, cfg, sway, tuple(t_grid) if t_grid is not None else None, gen_frames)
            groups.setdefault(key, []).append(
                _RequestCondition(index, text, ref, generator, sampling_params.output_type or "np")
            )

        outputs: dict[int, DiffusionOutput] = {}
        for (nfe, cfg, sway, t_grid, gen_frames), items in groups.items():
            text, c_mask = _pad_conditions([item.text for item in items])
            if len(items) == 1:
                ref = items[0].ref
                ref_mask = torch.ones(ref.shape[:2], dtype=torch.bool, device=ref.device)
            else:
                ref, ref_mask = _pad_conditions([item.ref[0] for item in items])
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
                    t_grid=list(t_grid) if t_grid is not None else None,
                    seed=None,
                    latent_dim=self.latent_dim,
                    device=self.device,
                    dtype=torch.float32,
                    generator=[item.generator for item in items] if len(items) > 1 else items[0].generator,
                    sampler=self.cudagraph_wrapper,
                ).float()
            if not torch.isfinite(latents).all():
                raise RuntimeError("AuK generated latents contain NaN or Inf.")

            wav = None
            if any(item.output_type != "latent" for item in items):
                # Transfer the group once rather than copying each waveform separately.
                wav = self.vae_decode(latents).detach().to(device="cpu", dtype=torch.float32)
                if not torch.isfinite(wav).all():
                    raise RuntimeError("AuK generated audio contains NaN or Inf.")
            cpu_latents = latents.detach().cpu() if any(item.output_type == "latent" for item in items) else None
            for row, item in enumerate(items):
                if item.output_type == "latent":
                    assert cpu_latents is not None
                    output = cpu_latents[row : row + 1]
                else:
                    assert wav is not None
                    output = wav[row].reshape(-1)
                outputs[item.index] = DiffusionOutput(output=output)
        return [outputs[index] for index in range(req.num_reqs)]


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
