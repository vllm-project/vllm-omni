# Copyright 2025 vLLM-Omni Team
"""Stage 1: Flow-matching audio detokenizer and BigVGAN vocoder for Kimi Audio.

This replaces the earlier stub DiT/HiFi-GAN with a checkpoint-compatible
flow-matching DiT and BigVGAN vocoder so that generated semantic tokens are
correctly converted to 24kHz speech.
"""

import json
import os

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.indextts2.s2mel.modules.bigvgan import (
    BigVGAN,
)
from vllm_omni.model_executor.models.indextts2.s2mel.modules.commons import (
    AttrDict,
)
from vllm_omni.model_executor.models.kimi_audio.constants import (
    KIMI_AUDIO_OUTPUT_SAMPLE_RATE,
    KIMI_AUDIO_SEMANTIC_VOCAB_SIZE,
    KIMI_AUDIO_TOKEN_OFFSET,
)
from vllm_omni.model_executor.models.kimi_audio.detokenizer.dit import (
    KimiAudioFlowMatchingDiT,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


def _load_yaml_detokenizer_config(model_path: str) -> dict:
    """Load the Kimi Audio audio_detokenizer config.yaml if present."""
    config_path = os.path.join(model_path, "audio_detokenizer", "config.yaml")
    if os.path.exists(config_path):
        import yaml

        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        # The checkpoint's cfg_scale is a training value; the reference streaming
        # detokenizer does not use classifier-free guidance at inference.
        cfg["cfg_scale"] = 1.0
        cfg.setdefault("ode_steps", 30)
        cfg.setdefault("normalize_mel", False)
        cfg.setdefault("mel_mean", 0.0)
        cfg.setdefault("mel_std", 1.0)
        return cfg
    return {
        "model": {
            "dit": {
                "hidden_size": 2304,
                "depth": 16,
                "num_heads": 18,
                "semantic_vocab_size": KIMI_AUDIO_SEMANTIC_VOCAB_SIZE,
                "input_size": 80,
                "output_size": 80,
                "use_rope": True,
                "position_embedding_type": "skip",
                "max_seq_len": 4096,
                "mlp_ratio": 4.0,
                "rope_params": {
                    "max_position_embeddings": 4096,
                    "rope_base": 10000.0,
                },
            }
        },
        "ode_steps": 30,
        "cfg_scale": 1.0,
        "normalize_mel": False,
        "mel_mean": 0.0,
        "mel_std": 1.0,
    }


class KimiAudioDetokenizerForConditionalGeneration(nn.Module):
    """Stage 1: Flow-matching DiT → BigVGAN vocoder → 24kHz waveform."""

    # Mark as generative model so vllm's runner validation passes
    is_text_generation_model = True
    # Mark as producing multimodal outputs (audio waveform)
    have_multimodal_outputs = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.dtype = vllm_config.model_config.dtype
        self.model_path = vllm_config.model_config.model

        detokenizer_config = _load_yaml_detokenizer_config(self.model_path)

        # DiT
        self.dit = KimiAudioFlowMatchingDiT(detokenizer_config)
        self._load_dit_weights()

        # BigVGAN vocoder
        vocoder_config_path = os.path.join(self.model_path, "vocoder", "config.json")
        with open(vocoder_config_path) as f:
            vocoder_hparams = AttrDict(json.load(f))
        self.vocoder = BigVGAN(vocoder_hparams)
        self._load_vocoder_weights()

        self.vocoder = self.vocoder.to(self.dtype)

        # Inference hyperparameters
        self.ode_steps = detokenizer_config.get("ode_steps", 30)
        self.cfg_scale = detokenizer_config.get("cfg_scale", 1.0)
        self.normalize_mel = detokenizer_config.get("normalize_mel", False)
        self.mel_mean = detokenizer_config.get("mel_mean", 0.0)
        self.mel_std = detokenizer_config.get("mel_std", 1.0)
        logger.info(
            "Kimi Audio detokenizer loaded: ode_steps=%d cfg_scale=%s normalize_mel=%s mel_mean=%s mel_std=%s",
            self.ode_steps,
            self.cfg_scale,
            self.normalize_mel,
            self.mel_mean,
            self.mel_std,
        )

    def _load_dit_weights(self) -> None:
        dit_weights_path = os.path.join(self.model_path, "audio_detokenizer", "model.pt")
        if not os.path.exists(dit_weights_path):
            return

        checkpoint = torch.load(dit_weights_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint)
        speech_model_params = {
            k.replace("speech_model.", ""): v for k, v in state_dict.items() if k.startswith("speech_model.")
        }
        missing, unexpected = self.dit.load_state_dict(speech_model_params, strict=False)
        if missing:
            raise RuntimeError(f"Kimi Audio DiT missing keys: {missing[:5]}")
        # The reference checkpoint may contain extra conditioning/resampler keys that
        # are not used in the TTS path; unexpected keys are expected.
        if unexpected:
            pass

    def _load_vocoder_weights(self) -> None:
        vocoder_weights_path = os.path.join(self.model_path, "vocoder", "model.pt")
        if not os.path.exists(vocoder_weights_path):
            return

        checkpoint = torch.load(vocoder_weights_path, map_location="cpu", weights_only=True)
        generator_state = checkpoint.get("generator", checkpoint)
        try:
            self.vocoder.load_state_dict(generator_state, strict=False)
        except RuntimeError:
            self.vocoder.remove_weight_norm()
            self.vocoder.load_state_dict(generator_state, strict=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        multimodal_embeddings: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        sampling_metadata: torch.Tensor | None = None,
        logits_index: int | None = None,
        sampler=None,
        additional_information: dict | None = None,
        **kwargs,
    ) -> OmniOutput:
        """Convert audio tokens to waveform.

        Args:
            input_ids: [batch, seq_len] - mixed audio stream token IDs.  The
                stream contains special tokens (message starts, BLANK, EOD) as
                well as real semantic audio tokens.  Only tokens at or above
                KIMI_AUDIO_TOKEN_OFFSET are valid semantic IDs.
            positions: Position indices (not used; kept for protocol compatibility).

        Returns:
            OmniOutput with waveform.
        """
        # vLLM V1 flattens the scheduled batch into a 1-D token tensor
        # [total_tokens] (sequence boundaries live in attn_metadata.seq_lens,
        # not in a batch dimension). Stage 1 runs with max_num_seqs=1, so the
        # whole tensor is a single audio sequence. Promote it to [1, L] so the
        # batch loop below treats it as ONE sequence instead of L independent
        # one-token "sequences" (which would render each token in isolation and
        # produce gibberish).
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        # Filter out special/illegal tokens and map the remainder into the DiT
        # semantic range.  This mirrors the reference implementation which keeps
        # only tokens with t >= kimia_token_offset before detokenizing.
        semantic_tokens = []
        max_len = 0
        for b in range(input_ids.shape[0]):
            valid = input_ids[b][input_ids[b] >= KIMI_AUDIO_TOKEN_OFFSET]
            valid = valid - KIMI_AUDIO_TOKEN_OFFSET
            valid = valid.clamp(0, KIMI_AUDIO_SEMANTIC_VOCAB_SIZE - 1)
            semantic_tokens.append(valid)
            max_len = max(max_len, valid.shape[0])

        if max_len == 0:
            # No valid semantic tokens; return silence.
            waveform = torch.zeros(
                input_ids.shape[0],
                1,
                device=input_ids.device,
                dtype=self.dtype,
            )
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "audio": waveform,
                    "sr": KIMI_AUDIO_OUTPUT_SAMPLE_RATE,
                },
            )

        # Pad each sequence to the batch max length.  The extra embedding index
        # (semantic_vocab_size) is used as padding; generate() clamps to it.
        padded_ids = torch.full(
            (input_ids.shape[0], max_len),
            KIMI_AUDIO_SEMANTIC_VOCAB_SIZE,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        for b, valid in enumerate(semantic_tokens):
            padded_ids[b, : valid.shape[0]] = valid

        # Each semantic token represents 4 vocoder frames (12.5 semantic Hz vs
        # 50 mel Hz).  The reference repeats tokens with upsample_factor=4.
        padded_ids = padded_ids.repeat_interleave(4, dim=1)

        mel_spectrogram = self.dit.generate(
            padded_ids,
            ode_steps=self.ode_steps,
            cfg_scale=self.cfg_scale,
            dtype=self.dtype,
            normalize_mel=self.normalize_mel,
            mel_mean=self.mel_mean,
            mel_std=self.mel_std,
        )

        # Vocoder expects [B, num_mels, seq_len].
        waveform = self.vocoder(mel_spectrogram.transpose(1, 2))
        waveform = waveform.squeeze(1)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "audio": waveform,
                "sr": KIMI_AUDIO_OUTPUT_SAMPLE_RATE,
            },
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed input IDs (not used for detokenizer, stub for Protocol)."""
        return torch.zeros(1, device=input_ids.device)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute logits (not used for detokenizer, stub for Protocol)."""
        return None

    def load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        """Load weights from audio_detokenizer/ and vocoder/ subfolders.

        Weights are already loaded in __init__ because the Kimi Audio checkpoint
        layout differs from vLLM's standard weight loader.  The ``weights``
        argument is accepted only for protocol compatibility.
        """
        self._load_dit_weights()
        self._load_vocoder_weights()
