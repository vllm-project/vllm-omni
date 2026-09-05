# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
F5 TTS Pipeline for vLLM-Omni.

This module provides text-to-audio generation using the F5 TTS model,
integrated with the vLLM-Omni diffusion framework.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.f5_tts.audio_utils import (
    MelSpec,
    MelSpecConfig,
    load_vocoder,
    process_audio,
    run_vocoder,
)
from vllm_omni.diffusion.models.f5_tts.cuda_graph_dit_wrapper import F5TTSDiTCUDAGraphWrapper
from vllm_omni.diffusion.models.f5_tts.f5_tts_transformer import F5TTSDiTModel
from vllm_omni.diffusion.models.f5_tts.text_utils import (
    Tokenizer,
    estimate_duration,
    load_tokenizer,
    pad_and_batch,
    process_text,
)
from vllm_omni.diffusion.models.interface import SupportAudioOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)


def get_epss_timesteps(
    n: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Returns a 1-D tensor of ``n + 1`` timestep values in [0, 1].
    For supported values of *n* the schedule is taken from a table of
    hand-tuned timesteps; otherwise falls back to uniform spacing.
    """
    dt = 1.0 / 32.0
    predefined_timesteps: dict[int, list[int]] = {
        5: [0, 2, 4, 8, 16, 32],
        6: [0, 2, 4, 6, 8, 16, 32],
        7: [0, 2, 4, 6, 8, 16, 24, 32],
        10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    }
    if n in predefined_timesteps:
        return dt * torch.tensor(predefined_timesteps[n], device=device, dtype=dtype)
    else:
        return torch.linspace(0, 1, n + 1, device=device, dtype=dtype)


def euler_ode_solve(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    y0: torch.Tensor,
    ts: torch.Tensor,
) -> torch.Tensor:
    """
    Euler ODE solver for flow-matching models.

    Integrates ``dy/dt = fn(t, y)`` from ``ts[0]`` to ``ts[-1]`` using
    the explicit Euler method.  Time-stepping arithmetic is performed in
    float32 for numerical stability; the solution ``y`` is kept in the
    original dtype so the model sees its expected activation dtype (e.g.
    bfloat16).

    Args:
        fn: The ODE right-hand-side  ``fn(t, y) -> dy/dt``.
        y0: Initial condition, shape ``[B, T, D]``.
        ts: 1-D tensor of monotonically increasing timestep values.

    Returns:
        The solution at ``ts[-1]``, same shape as *y0*.
    """
    y = y0
    orig_dtype = y0.dtype
    for i in range(len(ts) - 1):
        t_prev = ts[i].to(torch.float32)
        t_next = ts[i + 1].to(torch.float32)
        h = t_next - t_prev
        pred = fn(t_prev, y)
        y = (y.float() + h * pred.float()).to(orig_dtype)
    return y


def seq_lens_to_mask(seq_lens: torch.Tensor, max_len: int) -> torch.Tensor:
    """Create a boolean mask ``[B, T]`` where ``mask[b, t] = (t < seq_lens[b])``."""
    seq = torch.arange(max_len, device=seq_lens.device)
    return seq.unsqueeze(0) < seq_lens.unsqueeze(1)


# ---------------------------------------------------------------------------
# Post-process helper (registered in the diffusion registry)
# ---------------------------------------------------------------------------


def get_f5_tts_post_process_func(
    od_config: OmniDiffusionConfig,
):
    """
    Create post-processing function for F5 TTS output.

    Converts the audio waveform tensor to a numpy array.
    """

    def post_process_func(
        audio: torch.Tensor,
        output_type: str = "np",
    ):
        if output_type == "latent":
            return audio
        if output_type == "pt":
            return audio
        audio_np = audio.cpu().float().numpy()
        return audio_np

    return post_process_func


def _resolve_request_value(
    primary: object | None,
    fallback: object | None = None,
    *,
    default: object | None = None,
) -> object | None:
    """Normalize request fields that may arrive as a scalar or single-item list."""
    value = primary if primary is not None else fallback
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value


class F5TTSPipeline(nn.Module, CFGParallelMixin, SupportAudioOutput):
    """
    Pipeline for text-to-speech generation using F5 TTS with flow-matching.

    This pipeline generates audio waveforms from text using a DiT-based
    flow-matching model (F5 TTS architecture).  Sampling is done via Euler
    ODE integration with optional sway-sampling and Classifier-Free Guidance.

    The sole entry point is ``forward()``, called by the vLLM-Omni framework
    via ``DiffusionModelRunner.execute_model()``.  It handles the full
    pipeline: audio preprocessing, text tokenization, flow-matching sampling,
    vocoder decoding, and returns a ``DiffusionOutput`` containing an audio
    waveform tensor.

    Internally, ``sample()`` performs the low-level flow-matching ODE solve
    and returns a raw mel spectrogram.

    Args:
        od_config: OmniDiffusion configuration object
        prefix: Weight prefix for loading (default: "")
    """

    support_audio_output = True

    # Default Euler steps when the request does not set num_inference_steps.
    # Exposed for the runner's per-request cache refresh
    # (``_refresh_cache_for_requests``).
    default_num_inference_steps = 32

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()

        # Read transformer architecture from od_config.tf_model_config
        # (populated from transformer/config.json by OmniDiffusion)
        tf_cfg = od_config.tf_model_config
        dim = tf_cfg.get("dim", 1152)
        depth = tf_cfg.get("depth", 24)
        heads = tf_cfg.get("heads", 18)
        dim_head = tf_cfg.get("dim_head", 64)
        ff_mult = tf_cfg.get("ff_mult", 2)
        dropout = tf_cfg.get("dropout", 0.0)
        mel_dim = tf_cfg.get("mel_dim", 100)
        text_num_embeds = tf_cfg.get("text_num_embeds", 3244)
        text_dim = tf_cfg.get("text_dim", 512)
        conv_layers = tf_cfg.get("text_conv_layers", 4)
        attn_mask_enabled = tf_cfg.get("attn_mask_enabled", True)
        long_skip_connection = tf_cfg.get("long_skip_connection", False)
        text_mask_padding = tf_cfg.get("text_mask_padding", True)
        pe_attn_head = tf_cfg.get("pe_attn_head")
        conv_pos_embed_groups = tf_cfg.get("conv_pos_embed_groups", 1)

        hf_repo_id = tf_cfg.get("hf_repo_id")
        hf_subfolder = tf_cfg.get("hf_subfolder")
        hf_weight_filename = tf_cfg.get("hf_weight_filename")
        if not hf_repo_id or not hf_subfolder or not hf_weight_filename:
            raise ValueError("F5 support requires hf_repo_id, hf_subfolder, and hf_weight_filename in tf_model_config.")

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=hf_repo_id,
                subfolder=hf_subfolder,
                revision=od_config.revision,
                prefix="",
                fall_back_to_pt=True,
                allow_patterns_overrides=[hf_weight_filename],
            ),
        ]

        # Initialize our custom DiT transformer with config-driven params
        quant_config = od_config.quantization_config
        self.transformer = F5TTSDiTModel(
            od_config=od_config,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            ff_mult=ff_mult,
            mel_dim=mel_dim,
            text_num_embeds=text_num_embeds,
            text_dim=text_dim,
            text_mask_padding=text_mask_padding,
            pe_attn_head=pe_attn_head,
            conv_layers=conv_layers,
            attn_mask_enabled=attn_mask_enabled,
            long_skip_connection=long_skip_connection,
            conv_pos_embed_groups=conv_pos_embed_groups,
            quant_config=quant_config,
        )

        # Cache backend (set by worker if needed)
        self._cache_backend = None

        # DiT CUDA Graph replay is opt-in from od_config.extras, matching the
        # GLM-TTS `use_dit_cuda_graphs` convention (deploy/f5_tts.yaml enables
        # it explicitly). Cache backend + graph capture is separately gated
        # until validated, because per-step caching decisions must not be
        # baked into a graph.
        self._dit_cudagraph: F5TTSDiTCUDAGraphWrapper | None = None
        f5_graph_cfg = getattr(od_config, "extras", {}) or {}
        self._dit_cudagraph_requested = F5TTSDiTCUDAGraphWrapper.parse_bool(
            f5_graph_cfg.get("f5_dit_cudagraph"), default=False
        )
        self._dit_cudagraph_with_cache = F5TTSDiTCUDAGraphWrapper.parse_bool(
            f5_graph_cfg.get("f5_dit_cudagraph_with_cache"), default=False
        )
        raw_max_graphs = f5_graph_cfg.get("f5_dit_cudagraph_max_graphs", 8)
        self._dit_cudagraph_max_graphs = 8 if raw_max_graphs is None else int(raw_max_graphs)

        # Mel normalization parameters (optional, loaded from config / checkpoint)
        self._mel_mean: torch.Tensor | None = None
        self._mel_std: torch.Tensor | None = None
        self._mel_norm_eps: float = 1e-6
        self._mel_norm_mode: str = "mean_std"

        # Inference components (initialized by init_inference)
        self._mel_spec: MelSpec | None = None
        self._vocoder: Any = None
        self._vocoder_name: str | None = None
        self._tokenizer: Tokenizer | None = None
        tf_cfg = self.od_config.tf_model_config

        # Mel spec config
        mel_dict = tf_cfg.get("mel_spec", {}) or {}

        mel_spec_config = MelSpecConfig(
            sample_rate=mel_dict.get("sample_rate", 24000),
            hop_length=mel_dict.get("hop_length", 256),
            win_length=mel_dict.get("win_length", 1024),
            n_mel_channels=mel_dict.get("n_mel_channels", 100),
            n_fft=mel_dict.get("n_fft", 1024),
            mel_spec_type=mel_dict.get("mel_spec_type", "vocos"),
            wav_norm=mel_dict.get("wav_norm", False),
        )

        # Tokenizer
        tokenizer_type = tf_cfg.get("tokenizer_type")
        tokenizer_path = tf_cfg.get("tokenizer_path")
        if not tokenizer_type or not tokenizer_path:
            raise ValueError("F5 support requires tokenizer_type and tokenizer_path in tf_model_config.")

        # Vocoder
        vocoder_name = tf_cfg.get("vocoder_name", mel_spec_config.mel_spec_type)

        # Mel spec extractor
        self._mel_spec = MelSpec(mel_spec_config)

        # Tokenizer
        self._tokenizer = load_tokenizer(tokenizer_type, tokenizer_path)

        # Vocoder
        voc_name = vocoder_name or mel_spec_config.mel_spec_type
        self._vocoder_name = voc_name
        self._vocoder = load_vocoder(
            voc_name,
            sample_rate=mel_spec_config.sample_rate,
            device=self.device,
        )

    def _enable_dit_cudagraph_if_requested(self) -> None:
        """Lazily construct the F5 DiT CUDA Graph wrapper if configured.

        The wrapper is lazy so cache backends can first mutate/wrap the
        transformer during runner initialization. Graph capture is disabled
        whenever a cache backend is active (Cache-DiT, TeaCache, ...) unless
        explicitly forced: per-step cache decisions must not be baked into a
        captured graph. It is also disabled under HSDP (parameter all-gathers
        inside the forward) and, at runtime, under tensor parallelism
        (``RowParallelLinear`` collectives) — those stay eager.
        """
        if self._dit_cudagraph is not None or not self._dit_cudagraph_requested:
            return
        # Stage-level enforce_eager implies no graph capture unless the user
        # explicitly configured f5_dit_cudagraph in extras (deploy/f5_tts.yaml
        # documents the two as orthogonal: eager gates the in-stage vLLM
        # engine compile, the wrapper is a pipeline-owned DiT capture).
        f5_graph_cfg = getattr(self.od_config, "extras", {}) or {}
        if self.od_config.enforce_eager and "f5_dit_cudagraph" not in f5_graph_cfg:
            logger.info_once(
                "F5-TTS DiT CUDA Graph skipped under enforce_eager; set extras.f5_dit_cudagraph to force it."
            )
            return
        # HSDP shards weights and all-gathers parameters inside the forward —
        # not capturable by the manual graph wrapper (same for TP collectives,
        # which _can_graph detects at runtime).
        parallel_cfg = getattr(self.od_config, "parallel_config", None)
        if getattr(parallel_cfg, "use_hsdp", False):
            logger.info_once(
                "F5-TTS DiT CUDA Graph skipped: use_hsdp=True gathers parameters inside the forward."
            )
            return
        cache_backend = (self.od_config.cache_backend or "none").lower()
        if cache_backend not in ("", "none") and not self._dit_cudagraph_with_cache:
            logger.info_once(
                "F5-TTS DiT CUDA Graph requested but skipped with cache_backend=%s; "
                "set f5_dit_cudagraph_with_cache=true after validation to force it.",
                cache_backend,
            )
            return
        self._dit_cudagraph = F5TTSDiTCUDAGraphWrapper(
            self.transformer,
            max_graphs=self._dit_cudagraph_max_graphs,
        )
        logger.info(
            "F5-TTS DiT CUDA Graph enabled (max_graphs=%d, cache_backend=%s)",
            self._dit_cudagraph_max_graphs,
            cache_backend,
        )

    def predict_noise(self, **kwargs) -> torch.Tensor:
        """CFGParallelMixin interface: single transformer forward pass."""
        self._enable_dit_cudagraph_if_requested()
        if self._dit_cudagraph is not None:
            return self._dit_cudagraph(**kwargs)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            return self.transformer(**kwargs)

    def combine_cfg_noise(
        self,
        positive_noise_pred,
        negative_noise_pred,
        true_cfg_scale: float,
        cfg_normalize: bool = False,
    ) -> torch.Tensor:
        """F5-TTS CFG: pred_cond + (pred_cond - pred_uncond) * strength."""
        p = positive_noise_pred if isinstance(positive_noise_pred, torch.Tensor) else positive_noise_pred[0]
        n = negative_noise_pred if isinstance(negative_noise_pred, torch.Tensor) else negative_noise_pred[0]
        return p + (p - n) * true_cfg_scale

    def set_mel_stats(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        eps: float = 1e-6,
    ) -> None:
        """Set mel normalization statistics (call after loading weights if needed)."""
        self._mel_mean = mean.to(self.device)
        self._mel_std = std.to(self.device)
        self._mel_norm_eps = eps

    def _normalize_mel(self, x: torch.Tensor) -> torch.Tensor:
        if self._mel_mean is None:
            return x
        x_dtype = x.dtype
        mean = self._mel_mean.float()
        std = self._mel_std.float()
        if self._mel_norm_mode == "mean_std":
            x = (x.float() - mean) / (std + self._mel_norm_eps)
        elif self._mel_norm_mode == "mean":
            x = x.float() - mean
        else:
            raise ValueError(f"invalid mel_norm_mode={self._mel_norm_mode!r}")
        return x.to(x_dtype)

    def _denormalize_mel(self, x: torch.Tensor) -> torch.Tensor:
        if self._mel_mean is None:
            return x
        x_dtype = x.dtype
        mean = self._mel_mean.float()
        std = self._mel_std.float()
        if self._mel_norm_mode == "mean_std":
            x = x.float() * (std + self._mel_norm_eps) + mean
        elif self._mel_norm_mode == "mean":
            x = x.float() + mean
        else:
            raise ValueError(f"invalid mel_norm_mode={self._mel_norm_mode!r}")
        return x.to(x_dtype)

    @torch.no_grad()
    def sample(
        self,
        cond_audio: torch.Tensor,
        cond_audio_lens: torch.Tensor,
        cond_text: torch.Tensor,
        sample_lens: torch.Tensor,
        *,
        steps: int = 32,
        cfg_strength: float = 1.0,
        sway_sampling_coef: float | None = None,
        use_epss: bool = True,
        drop_ref_audio: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Generate mel spectrogram via Euler ODE sampling with flow-matching.

        This is the PyTorch equivalent of ``f5tts.model.cfm.CFM.sample``.

        Args:
            cond_audio: Reference/conditioning mel spectrogram ``[B, T, D]``.
            cond_audio_lens: Number of valid frames per batch element ``[B]``.
            cond_text: Text token IDs ``[B, T]``.
            sample_lens: Total output sequence length per element ``[B]``.
            steps: Number of Euler integration steps.
            cfg_strength: Classifier-Free Guidance scale. Set to 0 to disable.
            sway_sampling_coef: Sway sampling coefficient (e.g. -1.0).
                ``None`` disables sway sampling.
            use_epss: Whether to use EPSS timestep schedule.
            drop_ref_audio: If ``True``, zero out reference audio (unconditional).
            generator: Optional torch Generator for reproducible noise.

        Returns:
            Mel spectrogram ``[B, T, D]``.
        """
        device = cond_audio.device
        batch, seq_len, mel_dim = cond_audio.shape

        # Create masks
        cond_mask = seq_lens_to_mask(cond_audio_lens, seq_len)  # [B, T]
        # Upstream F5 skips the audio padding mask for batch-size 1 inference.
        # That keeps the text/audio sequence lengths aligned at the quantized
        # model width while still preserving the conditioning span via cond_mask.
        padding_mask = seq_lens_to_mask(sample_lens, seq_len) if batch > 1 else None

        # Normalize mel
        cond_audio = self._normalize_mel(cond_audio)
        if drop_ref_audio:
            cond_audio = torch.zeros_like(cond_audio)

        # Conditioning: keep cond region, zero elsewhere
        step_cond_audio = torch.where(cond_mask.unsqueeze(-1), cond_audio, torch.zeros_like(cond_audio))

        do_cfg = cfg_strength >= 1e-5

        def fn(t_val: torch.Tensor, noisy_audio: torch.Tensor) -> torch.Tensor:
            t = t_val.expand(batch)
            drop_mask_false = torch.zeros(batch, dtype=torch.bool, device=device)

            positive_kwargs = dict(
                noisy_audio=noisy_audio,
                cond_audio=step_cond_audio,
                cond_text=cond_text,
                timestep=t,
                mask=padding_mask,
                drop_audio_mask=drop_mask_false,
                drop_text_mask=drop_mask_false,
            )

            if not do_cfg:
                return self.predict_noise(**positive_kwargs)

            drop_mask_true = torch.ones(batch, dtype=torch.bool, device=device)
            negative_kwargs = dict(
                noisy_audio=noisy_audio,
                cond_audio=step_cond_audio,
                cond_text=cond_text,
                timestep=t,
                mask=padding_mask,
                drop_audio_mask=drop_mask_true,
                drop_text_mask=drop_mask_true,
            )

            return self.predict_noise_maybe_with_cfg(
                do_true_cfg=True,
                true_cfg_scale=cfg_strength,
                positive_kwargs=positive_kwargs,
                negative_kwargs=negative_kwargs,
                cfg_normalize=False,
            )

        # Ensure generator lives on the target device.
        if generator is not None and generator.device != device:
            seed = generator.initial_seed()
            generator = torch.Generator(device=device).manual_seed(seed)

        # Initial noise
        y0 = torch.randn(
            step_cond_audio.shape,
            dtype=step_cond_audio.dtype,
            device=device,
            generator=generator,
        )

        # Timestep schedule
        if use_epss:
            ts = get_epss_timesteps(steps, device=device, dtype=torch.float32)
        else:
            ts = torch.linspace(0, 1, steps + 1, device=device, dtype=torch.float32)

        if sway_sampling_coef is not None:
            ts = ts + sway_sampling_coef * (torch.cos(math.pi / 2.0 * ts) - 1.0 + ts)

        # Euler ODE solve
        sampled = euler_ode_solve(fn, y0, ts)

        # Replace conditioning region with original cond audio
        out = torch.where(cond_mask.unsqueeze(-1), cond_audio, sampled)

        return self._denormalize_mel(out)

    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        """
        Generate audio waveform from an OmniDiffusionRequest.

        This is the sole entry point, called by the vLLM-Omni framework
        (``DiffusionModelRunner.execute_model`` -> ``pipeline.forward``).

        The full pipeline is:

        1.  Extract data from the request.
        2.  Audio preprocessing (mel extraction and RMS normalization).
        3.  Text preprocessing (tokenization, duration estimation).
        4.  Flow-matching sampling (Euler ODE with EPSS / sway / CFG).
        5.  Vocoder decoding (mel -> waveform).

        Expected prompt format::

            prompts = [{
                "prompt": "<target text to synthesize>",
                "additional_information": {
                    "cond_text": "<transcript of conditioning audio>",
                    "ref_audio": <conditioning audio bytes>,
                    "lang": "<ISO language code>",
                },
            }]

        TTS-specific parameters (via ``req.sampling_params.extra_args``):
            speed (float): Speed factor (default 1.0).
            target_rms (float | None): RMS normalization target (default 0.1).
            sway_sampling_coef (float | None): Sway coefficient (default -1.0).
            use_epss (bool): Use EPSS schedule (default True).
            drop_ref_audio (bool): Drop reference audio (default False).

        Returns:
            DiffusionOutput containing the generated audio waveform.
        """
        device = self.device
        extra = req.sampling_params.extra_args if hasattr(req.sampling_params, "extra_args") else {}

        # Extract data from request prompts
        prompt_data = req.prompts[0] if req.prompts else {}

        if isinstance(prompt_data, str):
            target_text = prompt_data
            cond_audio_raw = None
            cond_text_str = ""
            lang = "en"
        else:
            target_text = prompt_data.get("prompt", "")
            additional = prompt_data.get("additional_information") or {}
            cond_text_str = _resolve_request_value(
                additional.get("ref_text"),
                additional.get("cond_text", extra.get("ref_text", extra.get("cond_text"))),
                default="",
            )
            cond_audio_raw = _resolve_request_value(
                additional.get("ref_audio"),
                extra.get("ref_audio"),
                default=None,
            )
            lang = _resolve_request_value(
                additional.get("lang"),
                extra.get("lang"),
                default="en",
            )

            cond_text_str = str(cond_text_str) if cond_text_str is not None else ""
            lang = str(lang) if lang is not None else "en"

        if cond_audio_raw is None:
            # No conditioning audio (e.g. warmup dummy run)
            logger.warning("No conditioning audio provided in request; returning empty DiffusionOutput.")
            return DiffusionOutput(output=torch.zeros(1, 1, 1, device=device))

        if self._mel_spec is None or self._tokenizer is None or self._vocoder is None:
            raise RuntimeError("Required components (mel_spec, tokenizer, vocoder) are not initialized.")

        speed = extra.get("speed", 1.0)
        target_rms = extra.get("target_rms", 0.1)

        # Audio preprocessing
        cond_mel, cond_rms = process_audio(
            cond_audio_raw,
            mel_spec=self._mel_spec,
            target_rms=target_rms,
        )

        # Text preprocessing
        cond_text_str, target_text = process_text(cond_text_str, target_text, lang)
        text_token_ids = self._tokenizer.encode(cond_text_str + target_text, lang=lang)

        cond_mel_len = cond_mel.shape[0]
        total_mel_len = estimate_duration(cond_mel_len, cond_text_str, target_text, speed)
        total_mel_len = max(total_mel_len, len(text_token_ids), cond_mel_len) + 1

        assert len(text_token_ids) <= total_mel_len, (
            f"number of text tokens ({len(text_token_ids)}) must be <= number of mel frames ({total_mel_len})"
        )

        # Padding and batching
        cond_audio, cond_text_tensor, seq_len, _ = pad_and_batch(cond_mel, text_token_ids, total_mel_len)
        # Batch-size-1 F5 inference runs without a padding mask to match F5-TTS.
        # In that mode the model can emit meaningful speech into the quantized tail,
        # so decoding only up to the heuristic `total_mel_len` can cut words off.
        effective_total_mel_len = seq_len if cond_audio.shape[0] == 1 else total_mel_len

        cond_audio = cond_audio.to(device)
        cond_text_tensor = cond_text_tensor.to(device)
        cond_audio_lens = torch.tensor([cond_mel_len], device=device)
        sample_lens = torch.tensor([effective_total_mel_len], device=device)

        # Sampling parameters
        steps = req.sampling_params.num_inference_steps or self.default_num_inference_steps
        cfg_strength = req.sampling_params.guidance_scale if req.sampling_params.guidance_scale_provided else 2.0
        sway_sampling_coef = extra.get("sway_sampling_coef", -1.0)
        use_epss = extra.get("use_epss", True)
        drop_ref_audio = extra.get("drop_ref_audio", False)

        # Generator for reproducibility
        generator = req.sampling_params.generator
        if generator is None and req.sampling_params.seed is not None:
            generator = torch.Generator(device=device).manual_seed(req.sampling_params.seed)

        # Flow-matching sampling
        mel_out = self.sample(
            cond_audio=cond_audio,
            cond_audio_lens=cond_audio_lens,
            cond_text=cond_text_tensor,
            sample_lens=sample_lens,
            steps=steps,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            use_epss=use_epss,
            drop_ref_audio=drop_ref_audio,
            generator=generator,
        )

        # Vocoder decoding (mel -> waveform)
        # Slice out only the generated portion (exclude conditioning region)
        gen_mel = mel_out[:, cond_mel_len:effective_total_mel_len, :]

        # Permute for vocoder: [B, T, D] -> [B, D, T]
        gen_mel = gen_mel.permute(0, 2, 1)

        # Run vocoder
        audio = run_vocoder(gen_mel, self._vocoder, self._vocoder_name)

        # RMS rescaling: if we scaled up the input, scale down the output
        if target_rms is not None and cond_rms < target_rms:
            audio = audio * cond_rms / target_rms

        return DiffusionOutput(output=audio)

    @staticmethod
    def _strip_ema_prefix(name: str) -> str:
        """Strip EMA / state-dict wrapper prefixes from checkpoint keys."""
        prefixes = ("ema_model_state_dict.", "model_state_dict.", "ema_model.")
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if name.startswith(p):
                    name = name[len(p):]
                    changed = True
        return name

    @staticmethod
    def _iter_checkpoint_tensors(name: str, value: Any) -> Iterable[tuple[str, torch.Tensor]]:
        if isinstance(value, torch.Tensor):
            yield name, value
            return
        if isinstance(value, Mapping):
            for nested_name, nested_value in value.items():
                yield from F5TTSPipeline._iter_checkpoint_tensors(
                    f"{name}.{nested_name}", nested_value,
                )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from F5-TTS PyTorch/EMA checkpoints.

        Delegates transformer weights to ``self.transformer.load_weights()``
        so that QKV stacking and FFN name mapping are applied correctly
        under Tensor Parallelism.
        """
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # Mark vocoder params as loaded — initialized via load_vocoder().
        if self._vocoder is not None and isinstance(self._vocoder, nn.Module):
            for name in params_dict:
                if name.startswith("_vocoder."):
                    loaded_params.add(name)

        # Build a set of top-level prefixes owned by the transformer so we
        # can route weights without per-key linear scans.
        tf_prefixes = {n.split(".")[0] for n, _ in self.transformer.named_parameters()}

        # Flatten & strip EMA prefixes, then split into transformer vs pipeline.
        transformer_weights: list[tuple[str, torch.Tensor]] = []
        pipeline_weights: list[tuple[str, torch.Tensor]] = []

        for raw_name, raw_value in weights:
            for flat_name, tensor in self._iter_checkpoint_tensors(raw_name, raw_value):
                clean = self._strip_ema_prefix(flat_name)
                # F5 checkpoints nest weights under "transformer." but
                # named_parameters() on the sub-module omits that prefix.
                if clean.startswith("transformer."):
                    clean = clean[len("transformer."):]
                top = clean.split(".")[0]
                if top in tf_prefixes:
                    transformer_weights.append((clean, tensor))
                elif clean in params_dict:
                    pipeline_weights.append((clean, tensor))
                else:
                    logger.debug("Skipping unmatched F5 weight: %s", flat_name)

        # Delegate to transformer's load_weights (handles QKV stacking, FFN mapping)
        if transformer_weights:
            tf_loaded = self.transformer.load_weights(transformer_weights)
            for tf_name in tf_loaded:
                loaded_params.add(f"transformer.{tf_name}")

        # Load remaining pipeline-level params directly
        for name, tensor in pipeline_weights:
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, tensor)
                loaded_params.add(name)

        # Log loading summary
        all_params = set(params_dict.keys())
        missing = all_params - loaded_params
        if missing:
            logger.warning(
                "Parameters not loaded from checkpoint (%d/%d): %s",
                len(missing),
                len(all_params),
                sorted(missing)[:20],
            )
        else:
            logger.info(
                "All %d parameters loaded successfully from checkpoint.",
                len(loaded_params),
            )

        # F5-TTS upstream assumes float32 everywhere (sinusoidal embeddings,
        # positional convolutions, etc.).  Keep the transformer in float32 so
        # these paths stay correct; we use torch.amp.autocast in predict_noise
        # to let matmul / attention run in bfloat16 for FlashAttn + speed.
        if next(self.transformer.parameters()).dtype != torch.float32:
            self.transformer.float()

        return loaded_params
