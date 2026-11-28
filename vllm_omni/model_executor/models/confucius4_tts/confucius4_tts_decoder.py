# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright 2026 NetEase Youdao
"""Confucius4-TTS Stage 1: semantic-to-acoustic diffusion and BigVGAN."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.indextts2.indextts2_s2mel_decoder import (
    _load_bigvgan,
    _resolve_bigvgan_source,
)
from vllm_omni.model_executor.models.indextts2.preprocess_utils import (
    resolve_model_file,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

from .configuration_confucius4_tts import S2A_MAX_POSITIONS, Confucius4TTSConfig
from .confucius4_tts_latent_recompute import Confucius4TTST2SRecompute
from .flow.flow import MaskedDiffWithXvec, MaskedDiffWithXvecConfig

logger = init_logger(__name__)


def _unwrap_state_dict(raw: Any) -> Mapping[str, torch.Tensor]:
    state = raw
    for key in ("state_dict", "model", "module"):
        if isinstance(state, Mapping) and key in state and isinstance(state[key], Mapping):
            state = state[key]
    if not isinstance(state, Mapping):
        raise TypeError("Confucius4-TTS S2A checkpoint is not a state dict")

    tensor_state = {str(name): value for name, value in state.items() if isinstance(value, torch.Tensor)}
    if not tensor_state:
        raise ValueError("Confucius4-TTS S2A checkpoint contains no tensors")

    for prefix in ("module.", "model."):
        if all(name.startswith(prefix) for name in tensor_state):
            tensor_state = {name[len(prefix) :]: value for name, value in tensor_state.items()}
    return tensor_state


class Confucius4TTSS2ADecoder(nn.Module):
    """Decode completed semantic sequences into 22.05 kHz waveforms."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        del prefix
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model
        self.config: Confucius4TTSConfig = (  # type: ignore[assignment]
            vllm_config.model_config.hf_config
        )

        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True

        cfg = self.config
        # The budget only limits requests; the rotary table keeps the checkpoint's
        # size, so a lower budget cannot change a parameter shape before loading.
        if not 1 <= cfg.s2a_max_positions <= S2A_MAX_POSITIONS:
            raise ValueError(
                f"Confucius4-TTS s2a_max_positions must be in [1, {S2A_MAX_POSITIONS}], got {cfg.s2a_max_positions}"
            )
        s2a_config = MaskedDiffWithXvecConfig(
            input_size=cfg.s2a_input_size,
            output_size=cfg.s2a_output_size,
            spk_embed_dim=cfg.s2a_spk_embed_dim,
            semantic_embed_dim=cfg.s2a_semantic_embed_dim,
            lm_latent_dim=cfg.s2a_lm_latent_dim,
            semantic_codebook_size=cfg.s2a_semantic_codebook_size,
            semantic_codebook_dim=cfg.s2a_semantic_codebook_dim,
            semantic_output_dim=cfg.s2a_semantic_output_dim,
            lr_channels=cfg.s2a_lr_channels,
            lr_sampling_ratios=tuple(cfg.s2a_lr_sampling_ratios),
            lr_out_channels=cfg.s2a_lr_out_channels,
            lr_groups=cfg.s2a_lr_groups,
            lr_in_channels=cfg.s2a_lr_in_channels,
            cfm_inference_cfg_rate=cfg.s2a_cfm_inference_cfg_rate,
            cfm_t_scheduler=cfg.s2a_cfm_t_scheduler,
            estimator_hidden_dim=cfg.s2a_estimator_hidden_dim,
            estimator_num_heads=cfg.s2a_estimator_num_heads,
            estimator_depth=cfg.s2a_estimator_depth,
            estimator_mel_dim=cfg.s2a_estimator_mel_dim,
            estimator_cond_dim=cfg.s2a_estimator_cond_dim,
            estimator_style_dim=cfg.s2a_estimator_style_dim,
            estimator_mlp_ratio=cfg.s2a_estimator_mlp_ratio,
            estimator_dropout=cfg.s2a_estimator_dropout,
            estimator_max_seq_len=S2A_MAX_POSITIONS,
            estimator_long_skip_connection=(cfg.s2a_estimator_long_skip_connection),
            estimator_final_layer=cfg.s2a_estimator_final_layer,
            estimator_wavenet_hidden_dim=(cfg.s2a_estimator_wavenet_hidden_dim),
            estimator_wavenet_kernel_size=(cfg.s2a_estimator_wavenet_kernel_size),
            estimator_wavenet_dilation_rate=(cfg.s2a_estimator_wavenet_dilation_rate),
            estimator_wavenet_num_layers=(cfg.s2a_estimator_wavenet_num_layers),
            estimator_wavenet_dropout=(cfg.s2a_estimator_wavenet_dropout),
        )
        self.s2a_model = MaskedDiffWithXvec(s2a_config).float()
        self._resolved_vocoder_source: str | None = None
        # Stage 0 no longer ships per-step hidden rows: the latent for the
        # finished semantic sequence is recomputed here in float32, matching
        # the official ``[BOS | codes | EOS]`` pass. Weights load in
        # ``load_weights`` from the same talker checkpoint Stage 0 uses.
        self.t2s_recompute = Confucius4TTST2SRecompute(cfg).float().eval()

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        **_: Any,
    ) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty(
                (0, 1),
                device=input_ids.device,
                dtype=torch.float32,
            )
        return torch.zeros(
            (input_ids.shape[0], 1),
            device=input_ids.device,
            dtype=torch.float32,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> None:
        return None

    @staticmethod
    def _tensor(
        info: Mapping[str, Any],
        key: str,
        *,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        value = info.get(key)
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            raise ValueError(f"Confucius4-TTS Stage 1 requires non-empty {key}")
        return value.to(device=device, dtype=dtype, non_blocking=True)

    def _recompute_latent(
        self,
        info: Mapping[str, Any],
        semantic: torch.Tensor,
    ) -> torch.Tensor:
        """Rebuild the talker latent from the finished semantic sequence."""
        device = semantic.device
        text_ids = self._tensor(info, "text_token_ids", device=device).to(torch.long).reshape(-1)
        condition = self._tensor(info, "condition_vector", device=device, dtype=torch.float32)
        return self.t2s_recompute(
            text_ids,
            condition,
            semantic,
            self.config.bos_token_id,
            self.config.eos_token_id,
        )

    def _prepare_request(
        self,
        info: Mapping[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int | None,
    ]:
        semantic = self._tensor(
            info,
            "semantic_codes",
            device=device,
        ).to(torch.long)
        ref_mel = self._tensor(
            info,
            "ref_mel",
            device=device,
            dtype=dtype,
        )
        style = self._tensor(
            info,
            "style",
            device=device,
            dtype=dtype,
        )

        semantic = semantic.reshape(-1)
        # Validate before any embedding lookup: an out-of-range id there is a
        # device-side assert that takes down the whole engine.
        if torch.any((semantic < 0) | (semantic >= self.config.s2a_semantic_codebook_size)):
            raise ValueError(
                f"Confucius4-TTS semantic code is outside [0, {self.config.s2a_semantic_codebook_size - 1}]"
            )
        latent = self._recompute_latent(info, semantic)
        if latent.shape[1] != self.config.s2a_lm_latent_dim:
            raise ValueError(
                "Confucius4-TTS latent width mismatch: "
                f"expected={self.config.s2a_lm_latent_dim} "
                f"actual={latent.shape[1]}"
            )

        if ref_mel.ndim == 2:
            ref_mel = ref_mel.unsqueeze(0)
        if ref_mel.ndim != 3 or ref_mel.shape[0] != 1 or ref_mel.shape[2] != self.config.s2a_output_size:
            raise ValueError(f"Confucius4-TTS ref_mel must have shape [1, T_ref, {self.config.s2a_output_size}]")
        if style.ndim == 1:
            style = style.unsqueeze(0)
        if style.ndim != 2 or style.shape[0] != 1 or style.shape[1] != self.config.s2a_spk_embed_dim:
            raise ValueError(f"Confucius4-TTS style must have shape [1, {self.config.s2a_spk_embed_dim}]")

        seed = info.get("seed")
        return (
            semantic.unsqueeze(0),
            latent.unsqueeze(0),
            ref_mel,
            style,
            int(seed) if seed is not None else None,
        )

    def _seed_context(
        self,
        device: torch.device,
        seed: int | None,
    ):
        if seed is None:
            return nullcontext()
        devices = (
            [device.index if device.index is not None else torch.accelerator.current_device_index()]
            if device.type == "cuda"
            else []
        )
        return torch.random.fork_rng(devices=devices)

    def _fit_reference_to_positions(self, ref_mel: torch.Tensor, target_length: int) -> torch.Tensor | None:
        """Keep reference plus target frames inside the S2A rotary table.

        The DiT sees ``[reference mel | target frames]`` and only has
        ``s2a_max_positions`` rotary positions. A reference of up to 30 s (about
        2600 frames) next to a long target overflows it, and the resulting shape
        error in a worker kills the whole stage. Trim the reference prompt to its
        most recent frames, the ones adjacent to the target, so every position
        stays in the range the checkpoint was trained on.

        Args:
            ref_mel: Reference mel prompt, shape (1, T_ref, n_mels).
            target_length: Number of generated mel frames.

        Returns:
            The reference, trimmed if needed, or ``None`` when the target alone
            does not fit (unreachable with the 1518-token budget).
        """
        budget = self.config.s2a_max_positions - target_length
        if budget < 1:
            return None
        if ref_mel.shape[1] > budget:
            logger.info(
                "Confucius4-TTS trims the reference prompt from %d to %d mel frames to fit %d S2A positions",
                ref_mel.shape[1],
                budget,
                self.config.s2a_max_positions,
            )
            ref_mel = ref_mel[:, -budget:, :]
        return ref_mel

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        del positions, intermediate_tensors, inputs_embeds
        infos = kwargs.get("model_intermediate_buffer")
        if infos is None:
            infos = runtime_additional_information
        request_infos = [info for info in (infos or []) if isinstance(info, dict)]
        sample_rate = torch.tensor(
            self.config.output_sample_rate,
            dtype=torch.int32,
        )
        if not request_infos:
            return OmniOutput(
                text_hidden_states=torch.zeros(1, device=input_ids.device),
                multimodal_outputs={
                    "audio": torch.zeros(1, device=input_ids.device),
                    "sr": self.config.output_sample_rate,
                },
            )

        device = input_ids.device
        model_dtype = next(self.s2a_model.parameters()).dtype
        if self._resolved_vocoder_source is None:
            self._resolved_vocoder_source = _resolve_bigvgan_source(
                self.model_path,
                self.config.vocoder_name,
            )
        bigvgan = _load_bigvgan(
            self._resolved_vocoder_source,
            device,
            torch.float32,
        )

        waveforms: list[torch.Tensor] = []
        for info in request_infos:
            codes = info.get("semantic_codes")
            if isinstance(codes, torch.Tensor) and codes.numel() == 0:
                # Stage 0 produced no semantic code (immediate EOS): nothing to
                # synthesize, and this must stay a per-request outcome.
                waveforms.append(torch.zeros(0, dtype=torch.float32))
                continue
            semantic, latent, ref_mel, style, seed = self._prepare_request(
                info,
                device=device,
                dtype=model_dtype,
            )
            target_length = max(
                1,
                int(semantic.shape[1] * self.config.mel_length_ratio),
            )
            target_lengths = torch.tensor(
                [target_length],
                device=device,
                dtype=torch.long,
            )
            ref_mel = self._fit_reference_to_positions(ref_mel, target_length)
            if ref_mel is None:
                logger.error(
                    "Confucius4-TTS target of %d mel frames exceeds the %d S2A positions; returning empty audio",
                    target_length,
                    self.config.s2a_max_positions,
                )
                waveforms.append(torch.zeros(0, dtype=torch.float32))
                continue
            with self._seed_context(device, seed):
                if seed is not None:
                    torch.manual_seed(seed)
                mel = self.s2a_model.inference(
                    semantic_token=semantic,
                    lm_latent=latent,
                    prompt_feat=ref_mel,
                    embedding=style,
                    target_feat_len=target_lengths,
                    n_timesteps=self.config.diffusion_steps,
                    inference_cfg_rate=self.config.inference_cfg_rate,
                )
            waveform = bigvgan(mel.float()).squeeze(0).squeeze(0)
            waveforms.append(waveform.float().reshape(-1).detach().cpu().contiguous())

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": waveforms,
                "sr": [sample_rate] * len(waveforms),
            },
        )

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        del weights
        checkpoint = resolve_model_file(
            self.model_path,
            self.config.s2a_checkpoint,
        )
        if checkpoint is None:
            raise FileNotFoundError(f"Confucius4-TTS S2A checkpoint {self.config.s2a_checkpoint!r} not found")
        raw = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        state = _unwrap_state_dict(raw)
        incompatible = self.s2a_model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Confucius4-TTS S2A checkpoint mismatch: "
                f"missing={incompatible.missing_keys[:8]} "
                f"unexpected={incompatible.unexpected_keys[:8]}"
            )
        loaded = {f"s2a_model.{name}" for name in state}
        logger.info(
            "Loaded %d Confucius4-TTS S2A weights",
            len(loaded),
        )

        t2s_checkpoint = resolve_model_file(
            self.model_path,
            self.config.t2s_checkpoint,
        )
        if t2s_checkpoint is None:
            raise FileNotFoundError(f"Confucius4-TTS checkpoint {self.config.t2s_checkpoint!r} not found")
        talker_state = load_file(t2s_checkpoint, device="cpu")
        loaded.update(f"t2s_recompute.{name}" for name in self.t2s_recompute.load_state(talker_state))
        return loaded
