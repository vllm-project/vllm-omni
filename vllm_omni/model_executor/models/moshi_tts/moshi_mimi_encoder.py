"""Moshi Mimi Encoder -- Stage 0: waveform -> audio codes.

Encodes raw waveform input with the HF Mimi codec and emits frame-major
codes for downstream AR conditioning.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = logging.getLogger(__name__)

DEFAULT_FRAME_RATE = 12.5


class MoshiMimiEncoder(nn.Module):
    """Stage-0 Mimi encoder for Moshi (GenerationModelRunner)."""

    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:  # noqa: ARG002
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True

        self._mimi: nn.Module | None = None
        self._forward_call_count: int = 0
        self._audio_vocab_size: int = 2048
        self._text_card: int = 32000
        self._num_codebooks: int = 8
        self._target_sample_rate: int = 24000
        self._frame_rate: float = DEFAULT_FRAME_RATE
        self._blank_user_codes: list[int] = [self._audio_vocab_size] * self._num_codebooks

    def _find_checkpoint_files(self) -> list[str]:
        import glob

        patterns = [
            os.path.join(self.model_path, "*.safetensors"),
            os.path.join(self.model_path, "model*.safetensors"),
        ]
        files: list[str] = []
        for pat in patterns:
            files.extend(glob.glob(pat))

        if files:
            return sorted(set(files))

        try:
            from huggingface_hub import snapshot_download

            cache_dir = snapshot_download(self.model_path)
            files = glob.glob(os.path.join(cache_dir, "*.safetensors"))
            return sorted(files)
        except Exception:
            pass

        raise FileNotFoundError(f"No safetensors files found for {self.model_path}")

    def _ensure_mimi_loaded(self) -> None:
        if self._mimi is not None:
            return

        from safetensors import safe_open
        from transformers import AutoModel, MoshiConfig

        config = MoshiConfig.from_pretrained(self.model_path)
        audio_config = config.audio_encoder_config
        mimi = AutoModel.from_config(audio_config)  # type: ignore[no-untyped-call]

        checkpoint_files = self._find_checkpoint_files()
        audio_state: dict[str, torch.Tensor] = {}

        hf_prefix = "audio_encoder."
        for ckpt_file in checkpoint_files:
            with safe_open(ckpt_file, framework="pt") as f:  # type: ignore[no-untyped-call]
                for key in f.keys():  # noqa: SIM118
                    if key.startswith(hf_prefix):
                        audio_state[key[len(hf_prefix) :]] = f.get_tensor(key)

        if not audio_state:
            mimi_file = os.path.join(self.model_path, "mimi.safetensors")
            if os.path.exists(mimi_file):
                with safe_open(mimi_file, framework="pt") as f:  # type: ignore[no-untyped-call]
                    for key in f.keys():  # noqa: SIM118
                        audio_state[key] = f.get_tensor(key)
                logger.info("Loaded %d Mimi weights from mimi.safetensors", len(audio_state))

        model_keys = set(mimi.state_dict().keys())
        if audio_state and not (set(audio_state.keys()) & model_keys):
            from .mimi_remap import remap_kyutai_mimi_keys

            logger.info("Remapping Kyutai Mimi keys to HF format...")
            audio_state = remap_kyutai_mimi_keys(audio_state)
            matched = set(audio_state.keys()) & model_keys
            logger.info("  Remapped: %d/%d keys match", len(matched), len(audio_state))

        missing, unexpected = mimi.load_state_dict(audio_state, strict=False)
        if missing:
            logger.warning("Mimi encoder: missing keys: %s", missing[:10])
        if unexpected:
            logger.warning("Mimi encoder: unexpected keys: %s", unexpected[:10])

        device = self.vllm_config.device_config.device
        mimi = mimi.to(device=device, dtype=torch.float32).eval()
        self._mimi = mimi

        self._audio_vocab_size = int(getattr(config, "audio_vocab_size", 2048))
        vocab_size = int(getattr(config, "vocab_size", 32000))
        self._text_card = vocab_size - 1 if vocab_size % 2 == 1 else vocab_size
        self._num_codebooks = int(getattr(config, "num_codebooks", 8))
        self._target_sample_rate = int(getattr(audio_config, "sampling_rate", 24000))
        self._frame_rate = float(getattr(audio_config, "frame_rate", DEFAULT_FRAME_RATE))

        samples_per_frame = int(self._target_sample_rate / self._frame_rate)
        blank_audio = torch.zeros(1, 1, samples_per_frame, dtype=torch.float32, device=device)
        with torch.no_grad():
            blank_codes = self._mimi.encode(blank_audio, num_quantizers=self._num_codebooks)[0]  # type: ignore[operator]
        self._blank_user_codes = blank_codes.squeeze(0).squeeze(-1).to(torch.long).tolist()

        logger.info(
            "Mimi encoder loaded from %s (sr=%d, frame_rate=%.2f, q=%d)",
            self.model_path,
            self._target_sample_rate,
            self._frame_rate,
            self._num_codebooks,
        )

    @staticmethod
    def _info_value(info: dict[str, Any], *keys: str, default: Any = None) -> Any:
        for key in keys:
            if key in info:
                return info[key]
        return default

    @staticmethod
    def _to_mono_waveform(raw_audio: Any, device: torch.device) -> torch.Tensor:
        if raw_audio is None:
            return torch.zeros((1, 0), dtype=torch.float32, device=device)
        if isinstance(raw_audio, torch.Tensor):
            wave = raw_audio.to(device=device, dtype=torch.float32)
        else:
            wave = torch.tensor(raw_audio, dtype=torch.float32, device=device)

        if wave.ndim == 0:
            wave = wave.reshape(1, 1)
        elif wave.ndim == 1:
            wave = wave.reshape(1, -1)
        elif wave.ndim >= 2:
            wave = wave.mean(dim=0, keepdim=True) if wave.shape[0] > 1 else wave[:1]
            wave = wave.reshape(1, -1)

        return wave.contiguous()

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,  # noqa: ARG002
        sampling_metadata: Any = None,  # noqa: ARG002
    ) -> None:
        return None

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,  # noqa: ARG002
        positions: torch.Tensor | None = None,  # noqa: ARG002
        intermediate_tensors: Any = None,  # noqa: ARG002
        inputs_embeds: torch.Tensor | None = None,  # noqa: ARG002
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> OmniOutput:
        self._forward_call_count += 1
        call_n = self._forward_call_count
        self._ensure_mimi_loaded()
        assert self._mimi is not None

        info = runtime_additional_information[0] if runtime_additional_information else {}
        logger.debug("[MimiEncoder forward #%d] has_audio=%s", call_n, "raw_audio" in info or "waveform" in info)
        device_any = self.vllm_config.device_config.device
        if isinstance(device_any, str):
            device = torch.device("cpu" if device_any == "auto" else device_any)
        elif device_any is None:
            device = torch.device("cpu")
        else:
            device = torch.device(device_any.type)

        raw_audio = self._info_value(info, "raw_audio", "waveform")
        sample_rate = int(
            self._info_value(
                info,
                "raw_audio_sample_rate",
                "sample_rate",
                default=self._target_sample_rate,
            )
        )

        waveform = self._to_mono_waveform(raw_audio, device=device)

        if waveform.numel() > 0 and sample_rate != self._target_sample_rate:
            import torchaudio

            waveform = torchaudio.transforms.Resample(sample_rate, self._target_sample_rate).to(device)(waveform)

        if waveform.numel() > 0:
            codes = self._mimi.encode(waveform.unsqueeze(0), num_quantizers=self._num_codebooks)[0]  # type: ignore[operator]
            user_audio_frames = codes.squeeze(0).transpose(0, 1).to(torch.long).cpu()
        else:
            user_audio_frames = torch.zeros((0, self._num_codebooks), dtype=torch.long)

        logger.debug(
            "[MimiEncoder forward #%d] encoded user_audio_frames=%d",
            call_n,
            user_audio_frames.shape[0],
        )
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "user_audio_frames": user_audio_frames,
                "blank_user_codes": torch.tensor(
                    self._blank_user_codes,
                    dtype=torch.long,
                ),
                "audio_vocab_size": torch.tensor(self._audio_vocab_size, dtype=torch.long),
                "frame_rate": torch.tensor(self._frame_rate, dtype=torch.float32),
            },
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:  # noqa: ARG002
        """Weights are loaded lazily in _ensure_mimi_loaded."""
        return set()
