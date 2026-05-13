"""Moshi Mimi Decoder — Stage 1: audio codes → waveform.

Loads the Mimi audio codec from the HF Moshi checkpoint and decodes
multi-codebook audio tokens to a 24kHz waveform.

Analogous to FishSpeechDACDecoder in the Fish Speech pipeline.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = logging.getLogger(__name__)

MIMI_SAMPLE_RATE = 24000


class MoshiMimiDecoder(nn.Module):
    """Stage-1 Mimi decoder for Moshi (GenerationModelRunner).

    Consumes frame-aligned audio codes from input_ids and decodes waveform
    via the HF transformers MimiModel decoder.
    """

    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True

        self._mimi: nn.Module | None = None
        # Read num_codebooks from config; default 8 for Moshi, 16 for Hibiki
        hf_config = vllm_config.model_config.hf_config
        self._num_codebooks = getattr(hf_config, "num_codebooks", 8)
        self._output_sample_rate: int = MIMI_SAMPLE_RATE

    def _ensure_mimi_loaded(self) -> None:
        if self._mimi is not None:
            return

        from transformers import AutoModel, MoshiConfig

        # Load the full Moshi config to get audio_encoder_config
        config = MoshiConfig.from_pretrained(self.model_path)
        audio_config = config.audio_encoder_config

        # Instantiate MimiModel from config
        mimi = AutoModel.from_config(audio_config)

        # Load audio_encoder weights from the checkpoint.
        # Two formats: HF (keys prefixed with "audio_encoder.") or Kyutai
        # (raw keys in a separate mimi.safetensors file).
        import os

        from safetensors import safe_open

        checkpoint_files = self._find_checkpoint_files()
        audio_state: dict[str, torch.Tensor] = {}

        # Try 1: HF format (audio_encoder.X in model shards)
        hf_prefix = "audio_encoder."
        for ckpt_file in checkpoint_files:
            with safe_open(ckpt_file, framework="pt") as f:
                for key in f.keys():
                    if key.startswith(hf_prefix):
                        audio_state[key[len(hf_prefix) :]] = f.get_tensor(key)

        # Try 2: Kyutai format (raw keys in mimi.safetensors)
        if not audio_state:
            mimi_file = os.path.join(self.model_path, "mimi.safetensors")
            if os.path.exists(mimi_file):
                with safe_open(mimi_file, framework="pt") as f:
                    for key in f.keys():
                        audio_state[key] = f.get_tensor(key)
                logger.info("Loaded %d Mimi weights from mimi.safetensors", len(audio_state))

        # Remap Kyutai key names to HF MimiModel if needed
        model_keys = set(mimi.state_dict().keys())
        if audio_state and not (set(audio_state.keys()) & model_keys):
            from .mimi_remap import remap_kyutai_mimi_keys

            logger.info("Remapping Kyutai Mimi keys to HF format...")
            audio_state = remap_kyutai_mimi_keys(audio_state)
            matched = set(audio_state.keys()) & model_keys
            logger.info("  Remapped: %d/%d keys match", len(matched), len(audio_state))

        missing, unexpected = mimi.load_state_dict(audio_state, strict=False)
        if missing:
            logger.warning("Mimi: missing keys: %s", missing[:10])
        if unexpected:
            logger.warning("Mimi: unexpected keys: %s", unexpected[:10])
        if not missing:
            logger.info("Mimi decoder: all keys loaded successfully")

        device = self.vllm_config.device_config.device
        mimi = mimi.to(device=device, dtype=torch.float32).eval()
        self._mimi = mimi

        # Update sample rate from loaded config
        if hasattr(audio_config, "sampling_rate"):
            self._output_sample_rate = audio_config.sampling_rate

        logger.info(
            "Mimi codec loaded from %s (device=%s, sample_rate=%d)",
            self.model_path,
            device,
            self._output_sample_rate,
        )

    def _find_checkpoint_files(self) -> list[str]:
        """Find safetensors checkpoint files for the model."""
        import glob
        import os

        # Try local path first
        patterns = [
            os.path.join(self.model_path, "*.safetensors"),
            os.path.join(self.model_path, "model*.safetensors"),
        ]
        files = []
        for pat in patterns:
            files.extend(glob.glob(pat))

        if files:
            return sorted(set(files))

        # Try HF cache
        try:
            from huggingface_hub import snapshot_download

            cache_dir = snapshot_download(self.model_path)
            files = glob.glob(os.path.join(cache_dir, "*.safetensors"))
            return sorted(files)
        except Exception:
            pass

        raise FileNotFoundError(f"No safetensors files found for {self.model_path}")

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros(
            (input_ids.shape[0], 1),
            device=input_ids.device,
            dtype=torch.float32,
        )

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
        return None

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
        """Decode audio codes into waveform.

        input_ids layout: flat codes [num_codebooks * num_frames],
        codebook-major: [cb0_f0, cb0_f1, ..., cb0_fN, cb1_f0, ...].
        """
        self._ensure_mimi_loaded()
        assert self._mimi is not None

        q = self._num_codebooks
        sr_val = self._output_sample_rate
        sr_tensor = torch.tensor(sr_val, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)

        if input_ids is None or input_ids.numel() == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": [empty],
                    "sr": [sr_tensor],
                },
            )

        ids = input_ids.reshape(-1).to(dtype=torch.long)
        total = ids.numel()

        if total % q != 0:
            # Truncate to multiple of num_codebooks
            total = (total // q) * q
            ids = ids[:total]

        num_frames = total // q
        if num_frames == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": [empty],
                    "sr": [sr_tensor],
                },
            )

        # Reshape: flat codebook-major → [1, num_codebooks, num_frames]
        codes = ids.reshape(q, num_frames).unsqueeze(0)  # [1, Q, T]

        # Decode via Mimi
        device = codes.device
        codes = codes.to(device=device)
        audio_output = self._mimi.decode(codes)

        # MimiModel.decode returns an object with audio_values
        if hasattr(audio_output, "audio_values"):
            waveform = audio_output.audio_values  # [1, 1, samples]
        else:
            waveform = audio_output  # Fallback

        waveform = waveform.reshape(-1).to(dtype=torch.float32)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                # "audio" key for /v1/chat/completions (_create_audio_choice)
                "audio": waveform,
                # "model_outputs" key for /v1/audio/speech (fallback path)
                "model_outputs": [waveform],
                "sr": [sr_tensor],
            },
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Weights are loaded lazily in _ensure_mimi_loaded."""
        return set()
