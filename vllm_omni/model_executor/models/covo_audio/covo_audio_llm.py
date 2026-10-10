# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright 2026 Tencent.
from collections.abc import Hashable, Iterable
from typing import Any

import torch
from torch import nn
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.encoder_cudagraph_defs import (
    EncoderCudaGraphCaptureInputs,
    EncoderCudaGraphConfig,
    EncoderCudaGraphReplayBuffers,
    EncoderItemSpec,
)

_MAX_AUDIO_TOKENS = 188


class DownsampleLayer(nn.Module):
    """
    Downsample layer with 1D convolution and linear layers.
    """

    def __init__(self, input_dim, output_dim, hidden_dim=2048):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=2, padding=1)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        # x: (B, T, C)
        x = x.transpose(1, 2)  # -> (B, C, T)
        x = self.conv1d(x)  # -> (B, C, T // 2)
        x = x.transpose(1, 2)  # -> (B, T // 2, C)
        x = self.relu1(x)
        x = self.linear1(x)  # -> (B, T // 2, hidden_dim)
        x = self.relu2(x)
        x = self.linear2(x)  # -> (B, T // 2, output_dim)
        return x


class AudioAdapter(nn.Module):
    """
    Audio adapter with downsample layers.
    """

    def __init__(self, input_dim, output_dim, downsample=8):
        """
        Args:
            input_dim (int): input feature dimension (number of channels)
            output_dim (int): output feature dimension
            downsample (int): total downsampling factor, must be a power of 2
        """
        super().__init__()
        assert downsample >= 2 and (downsample & (downsample - 1)) == 0, "downsample must be a power of 2"

        num_layers = (
            downsample.bit_length() - 1
        )  # calculate how many downsampling steps are needed to reach the target factor

        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            is_last = i == num_layers - 1
            out_dim = output_dim if is_last else input_dim
            layers.append(DownsampleLayer(in_dim, out_dim))
            in_dim = out_dim

        self.downsample_layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        Args:
            x (Tensor): shape (B, T, C),C=input_dim
        Returns:
            Tensor: shape (B, T // downsample, output_dim)
        """
        for layer in self.downsample_layers:
            x = layer(x)
        return x


class CovoAudioLLMForConditionalGeneration(nn.Module, SupportsPP, SupportsMultiModal):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.llm = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "llm"),
            architectures=["Qwen2ForCausalLM"],
        )
        self.encoder = WhisperEncoder(config.encoder_config)
        self.audio_adapter = AudioAdapter(
            config.whisper_feats_dim, config.llm_config.hidden_size, config.adapter_downsample
        )
        self.make_empty_intermediate_tensors = self.llm.make_empty_intermediate_tensors

    def get_language_model(self) -> torch.nn.Module:
        return self.llm

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        return super().embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_features = kwargs["audio_features"]
        audio_num_tokens = kwargs.get("audio_num_tokens")
        # Match encoder dtype (bfloat16) since audio_features from WhisperFeatureExtractor are float32
        encoder_dtype = next(self.encoder.parameters()).dtype
        audio_features = audio_features.to(dtype=encoder_dtype)
        feats = self.encoder(audio_features).last_hidden_state
        features = self.audio_adapter(feats)
        # Truncate each audio to its actual length (Whisper pads all to 30s)
        # Return a list of per-item tensors since lengths may differ.
        batch_size = features.shape[0]
        result = []
        for i in range(batch_size):
            if audio_num_tokens is not None:
                n = int(audio_num_tokens[i])
                result.append(features[i, :n, :])
            else:
                result.append(features[i])
        return result

    def get_encoder_cudagraph_config(self) -> EncoderCudaGraphConfig:
        return EncoderCudaGraphConfig(
            modalities=["audio"],
            buffer_keys=["audio_features"],
            out_hidden_size=self.audio_adapter.downsample_layers[-1].linear2.out_features,
        )

    def get_encoder_cudagraph_budget_range(self, vllm_config: VllmConfig) -> tuple[int, int]:
        return _MAX_AUDIO_TOKENS, 2 * _MAX_AUDIO_TOKENS

    def get_encoder_cudagraph_item_specs(self, mm_kwargs: dict[str, Any]) -> list[EncoderItemSpec]:
        audio_features = mm_kwargs["audio_features"]
        return [
            EncoderItemSpec(input_size=audio_features.shape[-1], output_tokens=_MAX_AUDIO_TOKENS)
            for _ in range(audio_features.shape[0])
        ]

    def select_encoder_cudagraph_items(self, mm_kwargs: dict[str, Any], indices: list[int]) -> dict[str, Any]:
        return {key: value[indices] for key, value in mm_kwargs.items()}

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
        device: torch.device,
        dtype: torch.dtype,
        path: str = "default",
        axis_keys: tuple[Hashable, ...] | None = None,
    ) -> EncoderCudaGraphCaptureInputs:
        batch_size = min(token_budget // _MAX_AUDIO_TOKENS, max_batch_size)
        if batch_size < 1:
            raise ValueError("CoVo encoder CUDA graph budget must fit at least one audio item")
        encoder_dtype = next(self.encoder.parameters()).dtype
        return EncoderCudaGraphCaptureInputs(
            {"audio_features": torch.zeros((batch_size, 128, 3000), device=device, dtype=encoder_dtype)}
        )

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ) -> EncoderCudaGraphReplayBuffers:
        audio_features = mm_kwargs["audio_features"]
        return EncoderCudaGraphReplayBuffers({"audio_features": audio_features})

    def encoder_cudagraph_forward(self, inputs: dict[str, torch.Tensor], path: str = "default") -> torch.Tensor:
        feats = self.encoder(inputs["audio_features"]).last_hidden_state
        return self.audio_adapter(feats)

    def encoder_eager_forward(self, mm_kwargs: dict[str, Any], path: str = "default") -> torch.Tensor:
        audio_features = mm_kwargs["audio_features"]
        encoder_dtype = next(self.encoder.parameters()).dtype
        return self.encoder_cudagraph_forward({"audio_features": audio_features.to(dtype=encoder_dtype)}, path)

    def postprocess_encoder_output(
        self,
        outputs: dict[str, torch.Tensor],
        indices: list[int],
        per_item_out_tokens: list[int],
        dest: dict[int, torch.Tensor] | list[torch.Tensor | None],
        clone: bool = False,
        batch_mm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        assert batch_mm_kwargs is not None
        audio_num_tokens = batch_mm_kwargs.get("audio_num_tokens")
        features = outputs["default"]
        for batch_idx, original_idx in enumerate(indices):
            n = int(audio_num_tokens[batch_idx]) if audio_num_tokens is not None else _MAX_AUDIO_TOKENS
            item = features[batch_idx, :n, :]
            dest[original_idx] = item.clone() if clone else item

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_state = self.llm(input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds)
        return hidden_state

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.llm.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
