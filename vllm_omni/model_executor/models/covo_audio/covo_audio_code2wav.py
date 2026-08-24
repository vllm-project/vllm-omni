# Copyright 2026 Tencent.
import os
from collections.abc import Iterable

import numpy as np
import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.models import SupportsPP
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from .config_covo_audio import CovoAudioCode2WavConfig
from .token2wav import JsonHParams, Token2WavDecoder


class CovoAudioCode2WavForConditionalGeneration(nn.Module, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        model_name = vllm_config.model_config.model
        if os.path.isdir(model_name):
            model_path = model_name
        else:
            from huggingface_hub import snapshot_download

            model_path = snapshot_download(model_name)
        token2wav_path = os.path.join(model_path, "token2wav")

        code2wav_config = CovoAudioCode2WavConfig()
        model_config = JsonHParams(
            **{
                "token2latent": code2wav_config.token2latent,
                "wavegan": code2wav_config.wavegan,
                "wav_input_sr": code2wav_config.wav_input_sr,
                "global_mean_var": os.path.join(token2wav_path, "global_mean_var.npy"),
            }
        )

        self.decoder = Token2WavDecoder(model_config)
        ckpt = torch.load(
            os.path.join(token2wav_path, "model.pt"),
            map_location="cpu",
            weights_only=True,
        )
        self.decoder.load_state_dict(ckpt)
        self.decoder.eval()
        self.infer_config = code2wav_config.inference
        self._decoder_moved = False
        self._sampler = Sampler()

        # Load bundled speaker prompt files for voice timbre control.
        prompt_dir = os.path.join(os.path.dirname(__file__), "speaker_prompt")
        self._prompt_token = torch.from_numpy(np.load(os.path.join(prompt_dir, "prompt_token.npy"))).long()
        self._prompt_latent = torch.from_numpy(np.load(os.path.join(prompt_dir, "prompt_latent.npy"))).float()
        self._spkr_embed = torch.from_numpy(np.load(os.path.join(prompt_dir, "speaker_embed.npy"))).float()

    def _ensure_on_device(self, device: torch.device) -> None:
        """Move decoder and prompt tensors to *device* once."""
        if self._decoder_moved:
            return
        self.decoder = self.decoder.to(device)
        self._prompt_token = self._prompt_token.to(device)
        self._prompt_latent = self._prompt_latent.to(device)
        self._spkr_embed = self._spkr_embed.to(device)
        self._decoder_moved = True

    def forward(self, input_ids: torch.Tensor, **kwargs):
        self._ensure_on_device(input_ids.device)
        data = {
            "target_token": input_ids.unsqueeze(0),
            "sample_rate": 24000,
            "prompt_token": self._prompt_token,
            "prompt_latent": self._prompt_latent,
            "spkr_embed": self._spkr_embed,
        }
        audio = self.decoder.inference(data, **self.infer_config)
        return audio

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal: bool = False,
    ) -> torch.Tensor:
        hidden_size = self.vllm_config.model_config.get_hidden_size()
        return torch.zeros(
            input_ids.numel(),
            hidden_size,
            dtype=self.vllm_config.model_config.dtype,
            device=input_ids.device,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        if hidden_states.numel() == 0:
            return None
        batch_size = hidden_states.shape[0] if hidden_states.ndim > 0 else 1
        vocab_size = self.config.llm_config.vocab_size
        return torch.zeros(
            (batch_size, vocab_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        if logits is None or logits.numel() == 0:
            return None
        return self._sampler(logits=logits, sampling_metadata=sampling_metadata)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        # Weights are loaded in __init__ via torch.load
        # Return all decoder state_dict keys so vLLM knows they are initialized
        return {f"decoder.{k}" for k in self.decoder.state_dict()}
