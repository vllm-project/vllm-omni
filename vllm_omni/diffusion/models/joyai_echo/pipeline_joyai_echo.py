"""JoyAI-Echo (LTX-2.3 derivative) T2V+Audio pipeline.

Subclasses :class:`LTX23Pipeline` so that the entire denoising loop
(``forward``, CFG, scheduler stepping, prompt encoding, etc.) is inherited
as-is. The constructor is fully overridden because JoyAI ships a single
monolithic ``JoyAI-Echo-release.safetensors`` (no diffusers ``vae/`` /
``audio_vae/`` / ``vocoder/`` / ``connectors/`` subdirectories) and the
component architectures do not match ``AutoencoderKLLTX2Video`` /
``AutoencoderKLLTX2Audio`` / ``LTX2VocoderWithBWE``.

Runtime adaptation:

- ``self.transformer`` is :class:`JoyAIEchoTransformer` (extends
  :class:`LTX2VideoTransformer3DModel` with caption projection + integrated
  embeddings connectors + gated attention).
- ``self.connectors`` is a no-op passthrough -- the integrated connectors
  live inside the transformer subclass, so the parent's
  ``self.connectors(prompt_embeds, ...)`` call simply forwards the raw
  Gemma 188160-dim stack.
- ``self.vae`` / ``self.audio_vae`` / ``self.vocoder`` wrap the upstream
  Lightricks ``VideoEncoder``/``VideoDecoder`` etc. (vendored under
  ``ltx_core_vae/``) behind a diffusers-compatible facade so the parent's
  ``forward()`` can use them transparently.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from types import SimpleNamespace

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.video_processor import VideoProcessor
from safetensors import safe_open
from torch import nn
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.joyai_echo.joyai_echo_transformer import (
    JoyAIEchoTransformer,
    _translate_joyai_config,
)
from vllm_omni.diffusion.models.joyai_echo.ltx_core_vae.audio_vae import (
    AudioDecoder,
    AudioDecoderConfigurator,
    AudioEncoderConfigurator,
    VocoderConfigurator,
)
from vllm_omni.diffusion.models.joyai_echo.ltx_core_vae.video_vae import (
    VideoDecoder,
    VideoDecoderConfigurator,
    VideoEncoderConfigurator,
)
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23Pipeline

logger = init_logger(__name__)

# Threshold above which a tokenizer's ``model_max_length`` is treated as "huge
# / effectively unbounded" (e.g. Gemma3 reports ``1e30``) and we fall back to
# the encoder config's ``max_position_embeddings``. Mirrors the same constant
# inlined in ``LTX23Pipeline.__init__`` (``pipeline_ltx2_3.py:249``).
_HUGE_TOKENIZER_MODEL_MAX = 100_000


# ---------------------------------------------------------------------------
# Default scheduler config for JoyAI-Echo (LTX-2.3 family).
# JoyAI ships no scheduler/ subdirectory.
#
# The default DMD few-step path provides 9 explicit sigmas per-call (or 8 via
# ``OmniDiffusionSamplingParams.sigmas``), in which case
# ``scheduler.set_timesteps(sigmas=...)`` is invoked and the
# ``use_dynamic_shifting`` / ``base_shift`` / ``max_shift`` knobs below are
# *bypassed entirely*. They only take effect on the (untested in PR1) non-DMD
# fallback path where someone calls the pipeline without overriding sigmas.
# ---------------------------------------------------------------------------
_DEFAULT_SCHEDULER_CONFIG = {
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "use_dynamic_shifting": True,
    "base_shift": 0.5,
    "max_shift": 1.15,
    "base_image_seq_len": 256,
    "max_image_seq_len": 4096,
}


def _resolve_checkpoint_path(model: str) -> str:
    """Find the JoyAI-Echo monolithic safetensors file under a model directory."""
    if os.path.isfile(model):
        return model
    candidate = os.path.join(model, "JoyAI-Echo-release.safetensors")
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(
        f"Could not locate JoyAI-Echo monolithic safetensors under {model!r}; expected {candidate!r}."
    )


# ---------------------------------------------------------------------------
# Gemma-3 text encoder resolution
# ---------------------------------------------------------------------------
# The official ``jdopensource/JoyAI-Echo`` Hugging Face release does **not**
# bundle the Gemma-3-12B-IT tokenizer / weights -- the upstream
# ``inference.py`` reads a separate ``paths.gemma_path`` from its YAML config
# (see ``JoyAI-Echo/configs/inference.yaml``). We mirror that by accepting an
# explicit override via the ``JOYAI_ECHO_GEMMA_PATH`` environment variable
# (matching the existing ``INTERNVLA_A1_COSMOS_ENCODER_PATH`` precedent in
# ``vllm_omni/diffusion/models/internvla_a1/model_internvla_a1.py``).
#
# Resolution order:
#
# 1. ``$JOYAI_ECHO_GEMMA_PATH`` — explicit override; expected to point at a
#    standalone Gemma-3-12B-IT checkpoint directory.
# 2. ``<model>/text_encoder/`` — backward-compatible subfolder layout used
#    by mirrors that bundle Gemma alongside the JoyAI checkpoint.
# 3. Otherwise raise ``FileNotFoundError`` with an actionable message.

_GEMMA_ENV_VAR = "JOYAI_ECHO_GEMMA_PATH"
_REQUIRED_GEMMA_FILES = ("config.json", "tokenizer_config.json")


def _resolve_gemma_dir(model: str) -> tuple[str, str | None]:
    """Resolve the directory + ``from_pretrained`` ``subfolder`` for Gemma-3.

    Returns a ``(model_or_path, subfolder)`` pair compatible with
    ``AutoTokenizer.from_pretrained`` / ``Gemma3ForConditionalGeneration.from_pretrained``.
    """

    def _has_gemma_files(path: str) -> bool:
        return all(os.path.exists(os.path.join(path, f)) for f in _REQUIRED_GEMMA_FILES)

    env_path = os.environ.get(_GEMMA_ENV_VAR)
    if env_path:
        env_path = os.path.expanduser(env_path)
        if not os.path.isdir(env_path):
            raise FileNotFoundError(f"${_GEMMA_ENV_VAR} is set to {env_path!r} but the directory does not exist.")
        if not _has_gemma_files(env_path):
            raise FileNotFoundError(
                f"${_GEMMA_ENV_VAR}={env_path!r} does not look like a Gemma-3 checkpoint "
                f"(missing one of {_REQUIRED_GEMMA_FILES})."
            )
        return env_path, None

    if os.path.isdir(model):
        subfolder_path = os.path.join(model, "text_encoder")
        if os.path.isdir(subfolder_path) and _has_gemma_files(subfolder_path):
            return model, "text_encoder"

    raise FileNotFoundError(
        "JoyAI-Echo requires a Gemma-3-12B-IT checkpoint, which the official "
        "`jdopensource/JoyAI-Echo` release does not bundle. Either:\n"
        f"  (a) set ${_GEMMA_ENV_VAR} to a local Gemma-3-12B-IT directory, or\n"
        f"  (b) place a Gemma-3-12B-IT checkpoint under {os.path.join(str(model), 'text_encoder')!r}.\n"
        "See examples/offline_inference/joyai_echo/README.md for details."
    )


# ---------------------------------------------------------------------------
# Adapter classes -- expose diffusers-style VAE/vocoder API on top of the
# vendored Lightricks ``ltx_core_vae`` modules so that ``LTX23Pipeline.forward``
# can use them unchanged.
# ---------------------------------------------------------------------------


class _VideoVAEAdapter(nn.Module):
    """diffusers-compatible facade around ltx_core ``VideoDecoder`` (+ optional encoder).

    The parent pipeline calls ``self.vae.decode(latents, timestep, return_dict=False)[0]``.
    JoyAI's :class:`VideoDecoder` already accepts ``(latent, timestep_emb)`` and
    returns a pixel tensor, so we simply repackage the call.
    """

    def __init__(
        self,
        decoder: VideoDecoder,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
        *,
        encoder=None,
        scaling_factor: float = 1.0,
        timestep_conditioning: bool = True,
        spatial_compression_ratio: int = 32,
        temporal_compression_ratio: int = 8,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        if encoder is not None:
            self.encoder = encoder
        # Diffusers compatibility surface
        self.register_buffer("latents_mean", latents_mean.to(torch.float32), persistent=False)
        self.register_buffer("latents_std", latents_std.to(torch.float32), persistent=False)
        self.config = SimpleNamespace(
            scaling_factor=scaling_factor,
            timestep_conditioning=timestep_conditioning,
        )
        self.spatial_compression_ratio = spatial_compression_ratio
        self.temporal_compression_ratio = temporal_compression_ratio

    @property
    def dtype(self) -> torch.dtype:
        for p in self.decoder.parameters():
            return p.dtype
        return torch.float32

    def decode(self, latents: torch.Tensor, timestep: torch.Tensor | None = None, return_dict: bool = True):
        # ltx_core VideoDecoder forward signature: decoder(latent, timestep)
        out = self.decoder(latents, timestep)
        if return_dict:
            return SimpleNamespace(sample=out)
        return (out,)


class _AudioVAEAdapter(nn.Module):
    """diffusers-compatible facade around ltx_core ``AudioDecoder``."""

    def __init__(
        self,
        decoder: AudioDecoder,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
        *,
        encoder=None,
        sample_rate: int = 16000,
        mel_hop_length: int = 160,
        mel_bins: int = 64,
        latent_channels: int = 8,
        mel_compression_ratio: int = 4,
        temporal_compression_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        if encoder is not None:
            self.encoder = encoder
        self.register_buffer("latents_mean", latents_mean.to(torch.float32), persistent=False)
        self.register_buffer("latents_std", latents_std.to(torch.float32), persistent=False)
        self.config = SimpleNamespace(
            sample_rate=sample_rate,
            mel_hop_length=mel_hop_length,
            mel_bins=mel_bins,
            latent_channels=latent_channels,
        )
        self.mel_compression_ratio = mel_compression_ratio
        self.temporal_compression_ratio = temporal_compression_ratio

    @property
    def dtype(self) -> torch.dtype:
        for p in self.decoder.parameters():
            return p.dtype
        return torch.float32

    def decode(self, latents: torch.Tensor, return_dict: bool = True):
        out = self.decoder(latents)
        if return_dict:
            return SimpleNamespace(sample=out)
        return (out,)


class _JoyAIPassthroughConnectors(nn.Module):
    """No-op connector. JoyAI's connectors are integrated into the transformer.

    The parent pipeline calls
    ``self.connectors(prompt_embeds, attn_mask, padding_side=...)`` and expects
    a ``(video_embeds, audio_embeds, attention_mask)`` triple. We return the
    raw 188160-dim Gemma stack twice; ``JoyAIEchoTransformer.forward`` then
    splits it via the dedicated video/audio caption projections + connectors.
    """

    def forward(
        self,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor | None,
        *,
        padding_side: str = "left",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        return prompt_embeds, prompt_embeds, prompt_attention_mask


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class JoyAIEchoPipeline(LTX23Pipeline):
    """JoyAI-Echo single-shot T2V+Audio pipeline."""

    # Disable warmup (mirrors ``Wan2_2S2VPipeline``). The Omni engine's
    # dummy-run path is therefore not exercised for JoyAI in PR1, so any
    # latent-shape-dependent AdaLN issues at small canvases would only be
    # caught at real inference time. Re-enabled in PR2 alongside the paired
    # audio/video shape inference work (see RFC #4193 §3.3).
    dummy_run_num_frames = 0

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        # NOTE: We intentionally bypass ``LTX23Pipeline.__init__`` because it
        # hard-codes the diffusers subfolder layout, which JoyAI does not have.
        nn.Module.__init__(self)

        self.od_config = od_config
        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)
        if not isinstance(dtype, torch.dtype):
            dtype = torch.bfloat16
        self._dtype = dtype

        model = od_config.model
        ckpt_path = _resolve_checkpoint_path(model)

        # We own all weight loading; vllm's loader will receive an empty stream.
        self.weights_sources: list = []

        # ---- Read the bundled config dict from safetensors metadata ----
        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            md = f.metadata() or {}
        if "config" not in md:
            raise RuntimeError(
                "JoyAI checkpoint metadata is missing the 'config' field. "
                "This pipeline relies on the bundled vae/audio_vae/vocoder configs."
            )
        bundled_config = json.loads(md["config"])
        self._bundled_config = bundled_config

        # ---- Tokenizer + Text encoder ----
        # The official JoyAI release does not bundle Gemma-3-12B-IT under
        # ``<model>/text_encoder/``; resolve a standalone Gemma checkpoint
        # via ``$JOYAI_ECHO_GEMMA_PATH`` first and fall back to the
        # subfolder layout for backward compatibility (see
        # ``_resolve_gemma_dir`` above).
        gemma_root, gemma_subfolder = _resolve_gemma_dir(model)
        gemma_local_files_only = os.path.isdir(gemma_root)
        tokenizer_kwargs: dict = {"local_files_only": gemma_local_files_only}
        if gemma_subfolder is not None:
            tokenizer_kwargs["subfolder"] = gemma_subfolder
        self.tokenizer = AutoTokenizer.from_pretrained(gemma_root, **tokenizer_kwargs)
        with torch.device("cpu"):
            text_encoder_kwargs: dict = {
                "local_files_only": gemma_local_files_only,
                "torch_dtype": dtype,
            }
            if gemma_subfolder is not None:
                text_encoder_kwargs["subfolder"] = gemma_subfolder
            self.text_encoder = Gemma3ForConditionalGeneration.from_pretrained(gemma_root, **text_encoder_kwargs)

        # ---- Connectors: passthrough (JoyAI integrates them into transformer) ----
        self.connectors = _JoyAIPassthroughConnectors()

        # ---- VAE + audio VAE + vocoder via JoyAI configurators (CPU init) ----
        with torch.device("cpu"):
            video_decoder: VideoDecoder = VideoDecoderConfigurator.from_config(bundled_config)
            video_encoder = VideoEncoderConfigurator.from_config(bundled_config)
            audio_decoder: AudioDecoder = AudioDecoderConfigurator.from_config(bundled_config)
            audio_encoder = AudioEncoderConfigurator.from_config(bundled_config)
            raw_vocoder = VocoderConfigurator.from_config(bundled_config)

        # Compression ratios -- read from bundled config instead of from
        # diffusers attributes (JoyAI VAE classes don't expose these).
        vae_cfg = bundled_config.get("vae", {})
        avae_cfg = bundled_config.get("audio_vae", {})
        spatial_compression = int(vae_cfg.get("patch_size", 4)) * 8  # 32 for JoyAI
        temporal_compression = int(vae_cfg.get("temporal_compression_ratio", 8))
        # Audio VAE: latent shape = mel_bins / mel_compression_ratio in mel-axis,
        # T / temporal_compression_ratio in time axis.
        audio_mel_compression = int(avae_cfg.get("mel_compression_ratio", 4))
        audio_temporal_compression = int(avae_cfg.get("temporal_compression_ratio", 4))

        # ---- Per-channel statistics ----
        # The Lightricks ``VideoEncoder``/``VideoDecoder`` (and audio counterparts)
        # already perform (de)normalization *internally* via their own
        # ``per_channel_statistics`` submodule (see ``video_vae.py:712`` and
        # ``video_vae.py:320``). The checkpoint's
        # ``vae.per_channel_statistics.{mean,std}-of-means`` tensors are routed
        # into those submodules by ``_load_vae_state_dict``.
        #
        # The parent ``LTX23Pipeline._normalize_latents`` /
        # ``_denormalize_latents`` would also (de)normalize using
        # ``self.vae.latents_{mean,std}``. To avoid double application, we
        # expose **identity** stats here (mean=0, std=1, scaling_factor=1.0)
        # so the parent's calls become no-ops and only the in-decoder/encoder
        # normalization is effective.
        #
        # Note: the parent's ``_denormalize_audio_latents`` runs on **packed**
        # audio latents whose last (channel) dim equals
        # ``z_channels * latent_mel_bins`` (==128 for JoyAI: 8 * 16). We
        # therefore size the identity audio stats to match the actual packed
        # per-channel-statistics buffer shape from the checkpoint, not the
        # nominal ``z_channels``.
        video_latent_channels = int(vae_cfg.get("latent_channels", 128))
        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            avae_pcs_shape = f.get_slice("audio_vae.per_channel_statistics.mean-of-means").get_shape()
        audio_latent_channels = int(avae_pcs_shape[0]) if avae_pcs_shape else 128
        identity_video_mean = torch.zeros(video_latent_channels)
        identity_video_std = torch.ones(video_latent_channels)
        identity_audio_mean = torch.zeros(audio_latent_channels)
        identity_audio_std = torch.ones(audio_latent_channels)

        # Wrap in diffusers-compatible adapters
        self.vae = _VideoVAEAdapter(
            decoder=video_decoder,
            encoder=video_encoder,
            latents_mean=identity_video_mean,
            latents_std=identity_video_std,
            scaling_factor=1.0,
            timestep_conditioning=bool(vae_cfg.get("timestep_conditioning", True)),
            spatial_compression_ratio=spatial_compression,
            temporal_compression_ratio=temporal_compression,
        ).to(dtype)

        self.audio_vae = _AudioVAEAdapter(
            decoder=audio_decoder,
            encoder=audio_encoder,
            latents_mean=identity_audio_mean,
            latents_std=identity_audio_std,
            sample_rate=int(avae_cfg.get("sample_rate", 16000)),
            mel_hop_length=int(avae_cfg.get("mel_hop_length", 160)),
            mel_bins=int(avae_cfg.get("mel_bins", 64)),
            latent_channels=int(avae_cfg.get("latent_channels", 8)),
            mel_compression_ratio=audio_mel_compression,
            temporal_compression_ratio=audio_temporal_compression,
        ).to(dtype)

        # Vocoder: use JoyAI's raw VocoderWithBWE (or Vocoder); LTX23Pipeline's
        # forward calls it as ``self.vocoder(mel) -> waveform``.
        self.vocoder = raw_vocoder.to(dtype)

        # ---- Transformer ----
        transformer_config = bundled_config.get("transformer", {})
        ltx2_kwargs, joyai_kwargs = _translate_joyai_config(transformer_config)
        quant_config = getattr(self.od_config, "quantization_config", None)
        self.transformer = JoyAIEchoTransformer(
            **joyai_kwargs,
            **ltx2_kwargs,
            quant_config=quant_config,
        )

        # ---- Scheduler (no scheduler/ subdir; use defaults) ----
        # JoyAI ships a ``RectifiedFlowScheduler`` config in the metadata which
        # is Lightricks-internal and not API-compatible with diffusers.
        # We use the LTX-2.3 default ``FlowMatchEulerDiscreteScheduler``
        # parameters here; the actual sigma schedule is provided per-call by
        # the caller (``OmniDiffusionSamplingParams``) anyway.
        self.scheduler = FlowMatchEulerDiscreteScheduler(**_DEFAULT_SCHEDULER_CONFIG)

        # ---- Derived attributes (mirror parent ``__init__`` lines 230-263) ----
        self.vae_spatial_compression_ratio = self.vae.spatial_compression_ratio
        self.vae_temporal_compression_ratio = self.vae.temporal_compression_ratio
        self.audio_vae_mel_compression_ratio = self.audio_vae.mel_compression_ratio
        self.audio_vae_temporal_compression_ratio = self.audio_vae.temporal_compression_ratio
        self.transformer_spatial_patch_size = self.transformer.config.patch_size if self.transformer is not None else 1
        self.transformer_temporal_patch_size = (
            self.transformer.config.patch_size_t if self.transformer is not None else 1
        )
        self.audio_sampling_rate = self.audio_vae.config.sample_rate
        self.audio_hop_length = self.audio_vae.config.mel_hop_length

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio)

        # Tokenizer max length. Mirrors ``LTX23Pipeline.__init__``
        # (``pipeline_ltx2_3.py:246-256``) verbatim, including the
        # ``_HUGE_TOKENIZER_MODEL_MAX`` sentinel: Gemma3 tokenizers report
        # ``model_max_length = 1e30`` because their context is effectively
        # unbounded (rope_scaling). We fall back to the encoder's actual
        # ``max_position_embeddings`` (128K for Gemma3-12B-IT) in that case,
        # then to a safe 1024 default.
        tokenizer_max_length = int(getattr(self.tokenizer, "model_max_length", 0) or 0)
        if not tokenizer_max_length or tokenizer_max_length > _HUGE_TOKENIZER_MODEL_MAX:
            encoder_config = getattr(self.text_encoder, "config", None)
            tokenizer_max_length = (
                getattr(encoder_config, "max_position_embeddings", None)
                or getattr(encoder_config, "max_seq_len", None)
                or 1024
            )
        self.tokenizer_max_length = int(tokenizer_max_length)

        # Pipeline state mirroring parent
        self._guidance_scale = None
        self._attention_kwargs = None
        self._interrupt = False
        self._num_timesteps = None
        self._current_timestep = None

        # ---- Defensive device placement ----
        # The outer ``DiffusersPipelineLoader`` wraps init in
        # ``with target_device:`` (CUDA), but the ``with torch.device("cpu"):``
        # blocks above intentionally override that for memory-friendly CPU
        # construction. The parent ``LTX23Pipeline.forward`` then re-issues
        # ``self.{text_encoder,connectors,vae,audio_vae,vocoder}.to(device)``
        # before each component is exercised. That swap pattern is observed
        # to be unreliable on some transformers / accelerate versions when a
        # ``Gemma3ForConditionalGeneration`` is loaded under a torch.device
        # CPU context (PR #4203 reviewer-reproduced ``F.embedding`` device
        # mismatch). Pin every component on ``self.device`` here so the
        # parent's per-call ``.to`` calls become no-ops on the first
        # invocation. Memory accounting is unchanged: the parent still moves
        # ``text_encoder`` / ``connectors`` back to CPU after each forward,
        # so subsequent generations resume the original swap pattern.
        self.text_encoder.to(self.device)
        self.vae.to(self.device)
        self.audio_vae.to(self.device)
        self.vocoder.to(self.device)

    # ------------------------------------------------------------------
    # Normalization overrides (explicit no-ops)
    # ------------------------------------------------------------------
    # JoyAI's vendored Lightricks ``VideoDecoder`` / ``VideoEncoder`` (and
    # the audio counterparts) (de)normalize latents *internally* via their
    # ``per_channel_statistics`` submodule in ``forward`` (see
    # ``ltx_core_vae/video_vae/video_vae.py:712`` and ``:320``). The
    # checkpoint's ``vae.per_channel_statistics.*`` buffers are routed into
    # those submodules by ``_load_vae_state_dict`` below.
    #
    # The parent ``LTX23Pipeline._{normalize,denormalize}_{,audio_}latents``
    # would normally apply the *same* statistics a second time, which would
    # silently corrupt outputs. We override them as explicit no-ops here
    # rather than relying on the more fragile "identity stats on the adapter"
    # trick (mean=0, std=1, scaling_factor=1) -- that trick still works as a
    # belt-and-braces backstop but the explicit no-op makes the intent
    # obvious in stack traces and survives parent refactors.
    @staticmethod
    def _normalize_latents(
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
        scaling_factor: float = 1.0,
    ) -> torch.Tensor:
        return latents

    @staticmethod
    def _normalize_audio_latents(
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
    ) -> torch.Tensor:
        return latents

    @staticmethod
    def _denormalize_latents(
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
        scaling_factor: float = 1.0,
    ) -> torch.Tensor:
        return latents

    @staticmethod
    def _denormalize_audio_latents(
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
    ) -> torch.Tensor:
        return latents

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str] | None:
        """Read the JoyAI monolithic safetensors and dispatch per-component.

        Returns ``None`` (and not the loaded-key set) on purpose: the JoyAI
        checkpoint is a *single* file packed with five top-level namespaces
        (``model.diffusion_model.*`` / ``vae.*`` / ``audio_vae.*`` /
        ``vocoder.*`` / ``text_embedding_projection.*``) and the vllm strict
        ``weights_not_loaded`` audit only knows about the transformer's
        parameter dict -- the VAE/audio_vae/vocoder keys would all show as
        "unexpected" even though we explicitly route them ourselves. Returning
        ``None`` opts out of that audit. The per-component loads below log
        their own missing/unexpected key counts at INFO so we keep coverage.
        """
        # Drain any weights vllm's loader passed (will be empty since
        # ``weights_sources`` is empty).
        for _ in weights:
            pass

        ckpt_path = _resolve_checkpoint_path(self.od_config.model)

        transformer_w: list[tuple[str, torch.Tensor]] = []
        vae_state: dict[str, torch.Tensor] = {}
        audio_vae_state: dict[str, torch.Tensor] = {}
        vocoder_state: dict[str, torch.Tensor] = {}
        ignored_prefixes: dict[str, int] = {}

        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if key.startswith("model.diffusion_model."):
                    transformer_w.append((key[len("model.diffusion_model.") :], tensor))
                elif key.startswith("text_embedding_projection."):
                    transformer_w.append((key, tensor))
                elif key.startswith("vae."):
                    vae_state[key[len("vae.") :]] = tensor
                elif key.startswith("audio_vae."):
                    audio_vae_state[key[len("audio_vae.") :]] = tensor
                elif key.startswith("vocoder."):
                    vocoder_state[key[len("vocoder.") :]] = tensor
                else:
                    # Track unknown top-level prefixes so we can surface them
                    # in the per-component INFO summary instead of silently
                    # dropping at debug level.
                    top = key.split(".", 1)[0]
                    ignored_prefixes[top] = ignored_prefixes.get(top, 0) + 1

        if ignored_prefixes:
            logger.info(
                "JoyAI: ignored top-level checkpoint prefixes (counts): %s",
                ignored_prefixes,
            )

        # ---- Transformer ----
        loaded = self.transformer.load_weights(iter(transformer_w))
        logger.info(
            "JoyAI: transformer loaded %d / %d keys", len(loaded), len(dict(self.transformer.named_parameters()))
        )

        # ---- VAE: feed the entire ``vae.*`` state into the encoder + decoder
        # adapter. The adapter holds them as submodules ``self.encoder`` /
        # ``self.decoder`` etc., so we route via name prefix. ----
        self._load_vae_state_dict(self.vae, vae_state, name="vae")
        self._load_vae_state_dict(self.audio_vae, audio_vae_state, name="audio_vae")

        # ---- Vocoder ----
        missing, unexpected = self.vocoder.load_state_dict(vocoder_state, strict=False)
        if missing:
            logger.warning("JoyAI: vocoder missing keys: %s", missing[:10])
        if unexpected:
            logger.warning("JoyAI: vocoder unexpected keys: %s", unexpected[:10])
        logger.info(
            "JoyAI: vocoder loaded %d keys (missing=%d, unexpected=%d)",
            len(vocoder_state) - len(unexpected),
            len(missing),
            len(unexpected),
        )

        # ---- Defensive device placement (mirror ``__init__``) ----
        # ``self.transformer`` was constructed under the outer
        # ``DiffusersPipelineLoader``'s ``with target_device:`` context, so
        # parameters are normally on CUDA. After ``load_weights`` rewires the
        # tensors via ``state_dict`` -> ``load_state_dict``, however, the
        # destination device follows the source tensors (which we read with
        # ``safe_open(..., device="cpu")`` above). Re-pin everything to
        # ``self.device`` so the first forward avoids any cross-device dispatch.
        self.transformer.to(self.device)
        self.vae.to(self.device)
        self.audio_vae.to(self.device)
        self.vocoder.to(self.device)

        return None  # opt out of strict audit

    @staticmethod
    def _load_vae_state_dict(adapter: nn.Module, state: dict[str, torch.Tensor], *, name: str) -> None:
        """Route ``encoder.*`` / ``decoder.*`` / ``per_channel_statistics.*`` into the adapter.

        ``per_channel_statistics.*`` is a top-level group in JoyAI's checkpoint
        (``vae.per_channel_statistics.{mean,std}-of-means``) but the
        Lightricks ``VideoEncoder`` / ``VideoDecoder`` classes each *contain*
        a ``self.per_channel_statistics`` submodule (used to (de)normalize
        latents inside their own ``forward``). We therefore route these stats
        into both encoder and decoder.
        """
        encoder_sd: dict[str, torch.Tensor] = {}
        decoder_sd: dict[str, torch.Tensor] = {}
        for k, v in state.items():
            if k.startswith("encoder."):
                encoder_sd[k[len("encoder.") :]] = v
            elif k.startswith("decoder."):
                decoder_sd[k[len("decoder.") :]] = v
            elif k.startswith("per_channel_statistics."):
                # Replicate into both encoder and decoder.
                encoder_sd[k] = v
                decoder_sd[k] = v
            else:
                logger.debug("JoyAI: %s ignoring unknown sub-key %s", name, k)

        if encoder_sd and getattr(adapter, "encoder", None) is not None:
            mm, ee = adapter.encoder.load_state_dict(encoder_sd, strict=False)
            if mm:
                logger.warning("JoyAI: %s encoder missing %d keys: %s", name, len(mm), list(mm)[:5])
            if ee:
                logger.warning("JoyAI: %s encoder unexpected %d keys: %s", name, len(ee), list(ee)[:5])
            logger.info(
                "JoyAI: %s encoder loaded %d keys (missing=%d, unexpected=%d)",
                name,
                len(encoder_sd) - len(ee),
                len(mm),
                len(ee),
            )
        if decoder_sd:
            mm, ee = adapter.decoder.load_state_dict(decoder_sd, strict=False)
            if mm:
                logger.warning("JoyAI: %s decoder missing %d keys: %s", name, len(mm), list(mm)[:5])
            if ee:
                logger.warning("JoyAI: %s decoder unexpected %d keys: %s", name, len(ee), list(ee)[:5])
            logger.info(
                "JoyAI: %s decoder loaded %d keys (missing=%d, unexpected=%d)",
                name,
                len(decoder_sd) - len(ee),
                len(mm),
                len(ee),
            )


# ---------------------------------------------------------------------------
# Post-process registration helper. Hardcodes ``audio_sample_rate=24000``
# (JoyAI vocoder output) instead of reading vocoder/config.json (absent).
# ---------------------------------------------------------------------------


def get_joyai_echo_post_process_func(od_config: OmniDiffusionConfig):
    """Factory used by ``vllm_omni/diffusion/registry.py:187``."""

    def _normalize_video(video):
        """Strip leading batch dim (if singleton) and convert float [0,1] →
        uint8 [0,255]. Pipeline returns ``(B, T, H, W, 3)`` numpy float32 (when
        ``output_type='np'``); examples / tests expect ``(T, H, W, 3) uint8``.
        """
        # Strip batch
        if hasattr(video, "ndim") and hasattr(video, "shape") and video.ndim == 5 and video.shape[0] == 1:
            video = video[0]
        # Convert float → uint8
        import numpy as _np  # local import to avoid polluting module globals

        if isinstance(video, _np.ndarray) and video.dtype != _np.uint8:
            v = video
            if _np.issubdtype(v.dtype, _np.floating):
                # diffusers VideoProcessor returns float32 in [0, 1]
                v = (v * 255.0).clip(0, 255).astype(_np.uint8)
            else:
                v = v.clip(0, 255).astype(_np.uint8)
            video = v
        elif isinstance(video, torch.Tensor) and video.dtype != torch.uint8:
            v = video.detach().cpu()
            if v.is_floating_point():
                v = (v * 255.0).clamp(0, 255).to(torch.uint8)
            else:
                v = v.clamp(0, 255).to(torch.uint8)
            video = v
        return video

    def _normalize_audio(audio):
        """Convert audio tensor to float32 cpu and strip leading singleton
        batch dim if present. Pipeline returns ``(B, C, N)`` or ``(B, N)``;
        downstream PyAV mux expects ``(C, N)`` or ``(N,)``.
        """
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().to(torch.float32).cpu()
            if audio.ndim == 3 and audio.shape[0] == 1:
                audio = audio[0]
        else:
            import numpy as _np

            if isinstance(audio, _np.ndarray) and audio.ndim == 3 and audio.shape[0] == 1:
                audio = audio[0]
        return audio

    def post_process_func(output):
        if isinstance(output, tuple) and len(output) == 2:
            video, audio = output
            return {"video": _normalize_video(video), "audio": _normalize_audio(audio), "audio_sample_rate": 24000}
        if isinstance(output, DiffusionOutput):
            video, audio = output.output
            return {"video": _normalize_video(video), "audio": _normalize_audio(audio), "audio_sample_rate": 24000}
        return output

    return post_process_func
