# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact-shape replay of the existing Torch CAM++ speaker model."""

from copy import copy

import torch
from vllm.config import VllmConfig
from vllm.config.multimodal import MultiModalConfig

from vllm_omni.model_executor.models.cosyvoice3.speaker_embedding_torch import CampplusTorch
from vllm_omni.model_executor.models.minicpmo_4_5.encoder_cuda_graph import EncoderCudaGraph


def _precision_policy() -> tuple[bool, ...]:
    return (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.enabled,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.are_deterministic_algorithms_enabled(),
    )


class SpeakerCudaGraph:
    """Keep the shared eager model untouched and own this consumer's graphs.

    Shapes are not padded. A changed precision policy runs eagerly instead of
    replaying kernels captured with another policy. The existing encoder graph
    manager bounds shape admission, isolates streams, and returns owned outputs.
    """

    def __init__(self, speaker: CampplusTorch, vllm_config: VllmConfig, *, max_graphs: int = 4):
        self.speaker = speaker
        self.device = speaker.device
        # Code2Wav is a generation stage, so its model config need not carry
        # multimodal limits. The local exact-shape adapter needs only defaults.
        config = copy(vllm_config)
        config.model_config = copy(config.model_config)
        if config.model_config.multimodal_config is None:
            config.model_config.multimodal_config = MultiModalConfig()
        self.graph = EncoderCudaGraph(speaker.model.forward, config, max_graphs=max_graphs)
        self._policy: tuple[bool, ...] | None = None

    @torch.inference_mode()
    def __call__(self, feat: torch.Tensor) -> torch.Tensor:
        if self.device.type != "cuda" or torch.is_autocast_enabled("cuda"):
            return self.speaker(feat)
        policy = _precision_policy()
        if self._policy is None:
            self._policy = policy
        elif policy != self._policy:
            return self.speaker(feat)
        feat = feat.to(self.device, non_blocking=True).unsqueeze(0).float()
        return self.graph(feat)
