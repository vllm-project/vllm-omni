"""MiniCPM-o 4.5 Token2Wav: waveform passthrough from talker output."""
import logging
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import SupportsPP

logger = logging.getLogger(__name__)


class MiniCPMO45OmniT2WForConditionalGeneration(nn.Module, SupportsPP):
    """No-op passthrough: talker already produces waveform, just forward it."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.have_multimodal_outputs = True

    def forward(self, input_ids=None, positions=None, **kwargs):
        device = input_ids.device if input_ids is not None else torch.device("cuda")
        runtime_info = kwargs.get("runtime_additional_information")
        if runtime_info and isinstance(runtime_info, list) and len(runtime_info) > 0:
            info = runtime_info[0] if isinstance(runtime_info[0], dict) else {}
            waveform = info.get("waveform") or info.get("mel_spec")
            if isinstance(waveform, torch.Tensor) and waveform.dim() == 1 and waveform.numel() > 100:
                from vllm_omni.model_executor.models.output_templates import OmniOutput
                dummy = torch.zeros(1, 1, device=device)
                return OmniOutput(text_hidden_states=dummy, multimodal_outputs={"model_outputs": [waveform]})
        return torch.zeros(1, 1, device=device)

    def compute_logits(self, hidden_states, *args, **kwargs):
        return torch.zeros(1, 2, device=hidden_states.device if isinstance(hidden_states, torch.Tensor) else "cuda")

    def sample(self, logits, sampling_metadata):
        return None

    def load_weights(self, weights):
        for k, v in weights:
            pass
        return set()

    def get_input_embeddings(self, input_ids, multimodal_embeddings=None, **kwargs):
        return torch.zeros(input_ids.shape[0], 1, device=input_ids.device)

    def embed_input_ids(self, input_ids, **kwargs):
        return self.get_input_embeddings(input_ids, **kwargs)
