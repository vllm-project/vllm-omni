# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SongGeneration v2 Stage 1 — Flow1dVAE diffusion decoder (code2wav).

Consumes vocal + bgm token streams from Stage 0 and decodes them into
48 kHz stereo waveform via CFM diffusion + Stable Audio VAE.
Wraps the upstream Tango decoder with the LLM_GENERATION stage contract.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .configuration_songgeneration_v2 import (
    SongGenerationV2Config,
    SongGenerationV2Flow1dVAESeparateConfig,
)
from .runtime_assets import RUNTIME_REPO_ENV, resolve_songgeneration_runtime_assets

logger = init_logger(__name__)


def _resolve_upstream_repo(explicit: str | None = None) -> str:
    """Resolve the upstream SongGeneration repo path.

    Priority: SONGGENERATION_REPO env var > explicit model path if it contains
    runtime assets > common local paths > HuggingFace runtime repo.
    """
    try:
        return resolve_songgeneration_runtime_assets(explicit)
    except Exception as exc:
        raise RuntimeError(
            "Could not resolve SongGeneration runtime assets for Stage 1. "
            "Set a local/HF runtime path with "
            f"{RUNTIME_REPO_ENV}, or use the default HuggingFace runtime repo."
        ) from exc


def _ensure_upstream_on_path(repo_root: str) -> None:
    """Prepend the upstream repo and its Flow1dVAE dir to sys.path.

    Both prefixes are needed because upstream uses absolute imports against
    either root (e.g. `from third_party.demucs...`) and against the
    `Flow1dVAE/` dir (e.g. `from tools.get_1dvae_large import...`).

    The upstream repo's own .venv site-packages is intentionally not added: it
    could shadow the server's transformers/torch. Install the upstream leaf
    deps into the active environment instead (see the example README).
    """
    flow_dir = os.path.join(repo_root, "codeclm", "tokenizer", "Flow1dVAE")
    for p in (flow_dir, repo_root):
        if p not in sys.path:
            sys.path.insert(0, p)


def _strip_audio_tokenizer_prefix(checkpoint_field: str) -> str:
    """Strip the `Flow1dVAESeparate_` dispatch prefix used in config.yaml."""
    prefix = "Flow1dVAESeparate_"
    if checkpoint_field.startswith(prefix):
        return checkpoint_field[len(prefix) :]
    return checkpoint_field


def _resolve_checkpoint(repo_root: str, relpath: str) -> str:
    if os.path.isabs(relpath):
        return relpath
    if relpath.startswith("./"):
        relpath = relpath[2:]
    return os.path.join(repo_root, relpath)


class SongGenerationV2Flow1dVAESeparateDecoder(nn.Module):
    """Stage 1 code2wav decoder for SongGeneration v2-large.

    Follows the same contract as FishSpeechDACDecoder / Qwen3TTSCode2Wav:
    - `forward()` receives flat stream-major token IDs, returns OmniOutput
    - Lazy weight loading via `_ensure_codec_loaded()`
    - `load_weights()` drains iterator (weights load from upstream ckpt)
    """

    input_modalities = "audio"

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.vllm_config = vllm_config
        hf_config = vllm_config.model_config.hf_config

        if isinstance(hf_config, SongGenerationV2Config):
            self.decoder_config: SongGenerationV2Flow1dVAESeparateConfig = hf_config.decoder_config
        elif isinstance(hf_config, SongGenerationV2Flow1dVAESeparateConfig):
            self.decoder_config = hf_config
        else:
            raise TypeError(
                f"hf_config has unexpected type {type(hf_config).__name__}; "
                f"expected SongGenerationV2Config or SongGenerationV2Flow1dVAESeparateConfig."
            )

        dev_cfg = getattr(vllm_config, "device_config", None)
        self._target_device: str | torch.device = (
            getattr(dev_cfg, "device", None) if dev_cfg is not None else None
        ) or "cuda:0"

        self.upstream_repo_path: str = _resolve_upstream_repo(vllm_config.model_config.model)

        # Stage runner flags (same as FishSpeechDACDecoder)
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.requires_raw_input_tokens = True
        self.enable_update_additional_information = True

        # Lazy-loaded upstream Tango instance
        self._tango: Any = None
        self.register_parameter(
            "_load_marker",
            nn.Parameter(torch.zeros(0), requires_grad=False),
        )

        # DefaultModelLoader needs a pattern to avoid downloading the full repo
        self.allow_patterns_overrides = ["songgeneration_v2_large/model.pt"]
        self.fall_back_to_pt_during_load = False

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict[str, Any]]:
        return [{"meta": {"gen_type": "mixed"}, "_is_dummy": True} for _ in range(num_reqs)]

    # ---- Lazy weight loading ------------------------------------------------

    def _ensure_codec_loaded(self) -> None:
        if self._tango is not None:
            return

        _ensure_upstream_on_path(self.upstream_repo_path)

        from .upstream_transformers_compat import ensure_upstream_transformers_compat

        ensure_upstream_transformers_compat()

        from generate_septoken import Tango  # type: ignore

        cfg = self.decoder_config
        ckpt_path = _resolve_checkpoint(
            self.upstream_repo_path,
            _strip_audio_tokenizer_prefix(cfg.audio_tokenizer_checkpoint_sep),
        )
        vae_config_path = _resolve_checkpoint(self.upstream_repo_path, cfg.vae_config)
        vae_model_path = _resolve_checkpoint(self.upstream_repo_path, cfg.vae_model)

        for p, label in [
            (ckpt_path, "septoken checkpoint"),
            (vae_config_path, "VAE config"),
            (vae_model_path, "VAE model"),
        ]:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"SongGeneration v2 {label} not found at {p}. Set SONGGENERATION_REPO or pass upstream_repo_path."
                )

        self._tango = Tango(
            model_path=ckpt_path,
            vae_config=vae_config_path,
            vae_model=vae_model_path,
            device=str(self._target_device),
        )
        logger.info(
            "SongGeneration v2 Tango decoder loaded (device=%s, sr=%d)",
            self._target_device,
            int(cfg.sample_rate),
        )

    # ---- vLLM stage hooks ---------------------------------------------------

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> None:
        return None

    # ---- Forward (LLM_GENERATION contract) ----------------------------------

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        """Decode stream-major codec tokens into stereo 48kHz audio.

        Input layout per request: flat [K * num_frames] tokens in
        stream-major order [s0_f0..s0_fT, s1_f0..s1_fT, s2_f0..s2_fT].
        Stream 0 (mixed) is kept for K=3 parity but unused by the
        diffusion decoder.
        """
        sr_val = int(self.decoder_config.sample_rate)
        sr_tensor = torch.tensor(sr_val, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)

        # Dummy forward for profiling
        if runtime_additional_information and all(
            isinstance(x, dict) and x.get("_is_dummy") for x in runtime_additional_information
        ):
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty], "sr": [sr_tensor]},
            )

        if input_ids is None or input_ids.numel() == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty], "sr": [sr_tensor]},
            )

        K = int(self.decoder_config.audio_tokenizer_code_depth_sep) + 1
        request_codes = self._split_requests(input_ids, K, kwargs.get("seq_token_counts"))

        audios: list[torch.Tensor] = []
        srs: list[torch.Tensor] = []
        for i, codes_BKT in enumerate(request_codes):
            info = (
                runtime_additional_information[i]
                if runtime_additional_information is not None and i < len(runtime_additional_information)
                else {}
            )
            meta = (info.get("meta", {}) or {}) if isinstance(info, dict) else {}
            gen_type = meta.get("gen_type", "mixed")
            left_context_frames = int(meta.get("left_context_size", 0))
            prompt_v = info.get("prompt_vocal_wav") if isinstance(info, dict) else None
            prompt_b = info.get("prompt_bgm_wav") if isinstance(info, dict) else None

            wav = self._decode_codes(
                codes_BKT,
                gen_type=gen_type,
                prompt_vocal_wav=prompt_v,
                prompt_bgm_wav=prompt_b,
            )

            # Trim left-context overlap for streaming chunks (same as Fish Speech).
            if left_context_frames > 0 and wav.numel() > 0:
                total_frames = codes_BKT.shape[-1]
                denom = max(int(total_frames), 1)
                samples_to_cut = int(left_context_frames / denom * wav.shape[-1])
                samples_to_cut = max(0, min(samples_to_cut, wav.shape[-1]))
                if samples_to_cut < wav.shape[-1]:
                    wav = wav[..., samples_to_cut:]
                else:
                    logger.warning(
                        "left_context trim %d >= decoded length %d; returning empty",
                        samples_to_cut,
                        wav.shape[-1],
                    )
                    wav = empty

            audios.append(wav.detach().to("cpu").to(torch.float32))
            srs.append(sr_tensor.clone())

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audios, "sr": srs},
        )

    # ---- Decode logic -------------------------------------------------------

    @torch.no_grad()
    def _decode_codes(
        self,
        codes: torch.Tensor,
        *,
        gen_type: str = "mixed",
        prompt_vocal_wav: torch.Tensor | None = None,
        prompt_bgm_wav: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode [1, K, T] codes to stereo waveform via Tango.code2sound.

        Stream 0 is dropped; streams 1/2 become vocal/bgm. For pure-track
        modes the unwanted stream is filled with a constant replacement token.
        """
        self._ensure_codec_loaded()
        assert self._tango is not None

        assert codes.dim() == 3 and codes.size(1) >= 3, f"expected [B, K>=3, T] codes; got {tuple(codes.shape)}"
        codes_vocal = codes[:, [1], :]
        codes_bgm = codes[:, [2], :]

        if gen_type == "bgm":
            codes_vocal = torch.full_like(codes_vocal, int(self.decoder_config.vocal_replacement_token_id))
        elif gen_type == "vocal":
            codes_bgm = torch.full_like(codes_bgm, int(self.decoder_config.bgm_replacement_token_id))

        ns = int(self.decoder_config.num_steps)
        gs = float(self.decoder_config.guidance_scale)

        # code2sound's "duration" is the diffusion window size
        # (min_samples = duration * frame_rate frames), not the output length.
        # Derive it from the configured window instead of a magic 40; with the
        # defaults (1000 frames / 25 fps) this stays 40 but now tracks config.
        frame_rate = int(getattr(self.decoder_config, "frame_rate", 25)) or 25
        window_frames = int(getattr(self.decoder_config, "decode_window_frames", 1000))
        duration_s = max(1, round(window_frames / frame_rate))

        wav = self._tango.code2sound(
            (codes_vocal, codes_bgm),
            prompt_vocal=prompt_vocal_wav,
            prompt_bgm=prompt_bgm_wav,
            duration=duration_s,
            guidance_scale=gs,
            num_steps=ns,
            disable_progress=True,
        )
        return wav

    # ---- Weight loading (registry contract) ---------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Drain vLLM iterator, then eager-load Tango/Flow1dVAE from upstream ckpt.

        Eager loading ensures:
        - Memory profiler sees the real Stage 1 footprint (~5 GB)
        - Checkpoint/path errors surface at startup, not first request
        - No OOM surprise on the first forward() call
        """
        for _ in weights:
            pass
        # Tango/upstream expects float32 weights; vLLM may set default dtype to
        # bf16 during model init. Temporarily restore float32 for codec loading.
        prev_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            self._ensure_codec_loaded()
        finally:
            torch.set_default_dtype(prev_dtype)
        return {name for name, _ in self.named_parameters()}

    # ---- Helpers ------------------------------------------------------------

    def _split_requests(
        self,
        ids: torch.Tensor,
        code_depth: int,
        seq_token_counts: list[int] | None,
    ) -> list[torch.Tensor]:
        """Split flat input_ids into per-request [1, K, T] tensors."""
        ids = ids.reshape(-1).to(torch.long)
        if seq_token_counts is None:
            seq_token_counts = [int(ids.numel())]
        boundaries = [0]
        for c in seq_token_counts:
            boundaries.append(boundaries[-1] + int(c))

        out: list[torch.Tensor] = []
        for i, count in enumerate(seq_token_counts):
            slab = ids[boundaries[i] : boundaries[i] + count]
            if slab.numel() == 0 or slab.numel() % code_depth != 0:
                raise ValueError(f"request {i}: {slab.numel()} codes not divisible by code_depth={code_depth}")
            T = slab.numel() // code_depth
            out.append(slab.view(1, code_depth, T))
        return out


__all__ = ["SongGenerationV2Flow1dVAESeparateDecoder"]
