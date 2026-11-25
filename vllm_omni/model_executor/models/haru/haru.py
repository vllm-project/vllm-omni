# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from https://github.com/higurashi-ai/Haru (Apache-2.0), haru_vllm/haru_vllm/model.py and
# haru/haru/models/haru.py @ d3612b0.
"""Haru-4B: the Qwen3.5 backbone on vLLM's engine, the audio parts in model hooks.

One language-model position is one 133 ms frame. Its input is the sum of three
streams at that frame: the model's text token, the model's audio latent and the
user's audio latent. The hidden state at frame ``t`` predicts frame ``t + 1``:
the text token through the tied output embedding and the audio latent through
the MeanFlow head.

The MRv2 model state resets the batch, preprocesses each scheduled conversation,
and applies Haru's text-stream rules to sampling logits. After text sampling,
the sampler generates the audio for each completed frame and saves the feedback
used in the next frame's embeddings.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper

from vllm_omni.model_executor.models.haru.meanflow_head import MeanFlowHead
from vllm_omni.model_executor.models.haru.runtime import HaruRuntime
from vllm_omni.model_executor.models.haru.vibevoice_codec import VibeVoiceCodec

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache

    from vllm_omni.model_executor.models.haru.model_state import HaruModelState


class AudioProjector(nn.Module):
    """Audio latent -> LM embedding space."""

    def __init__(self, latent_dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden)
        self.norm = nn.RMSNorm(hidden, eps=1e-6)
        self.fc2 = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.norm(self.fc1(x)))


class HaruForConditionalGeneration(Qwen3_5ForCausalLM):
    has_preprocess = True
    have_multimodal_outputs = True
    omni_pooler_payload_include_hidden = False
    # A conversation's prompt grows by one frame per step: keep its batch row.
    resume_streaming_rows_in_place = True

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"lm.": "model."})

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _check_supported(vllm_config)
        self.haru_config = spec = vllm_config.model_config.hf_config.spec
        self.model_audio_proj = AudioProjector(spec.latent_dim, spec.hidden_size)
        self.user_audio_proj = AudioProjector(spec.latent_dim, spec.hidden_size)
        self.head = MeanFlowHead(spec.head)
        self.codec = VibeVoiceCodec(spec.codec)
        self.register_buffer("latent_scale", torch.ones(()))
        self.register_buffer("latent_bias", torch.zeros(()))
        self.runtime = HaruRuntime(self, model_path=str(vllm_config.model_config.model))

    # ------------------------------------------------------------ latents

    def scale_latents(self, raw: torch.Tensor) -> torch.Tensor:
        return (raw.float() + self.latent_bias.float()) * self.latent_scale.float()

    def unscale_latents(self, normalized: torch.Tensor) -> torch.Tensor:
        return normalized.float() / self.latent_scale.float() - self.latent_bias.float()

    def embed_streams(
        self,
        tokens: torch.Tensor,
        model_latents: torch.Tensor,
        user_latents: torch.Tensor,
        model_mask: torch.Tensor | None = None,
        user_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sum of the three streams: ``tokens [T]``, raw latents ``[T, D]`` -> ``[T, H]``."""
        x = self.model.embed_input_ids(tokens)
        for proj, latents, mask in (
            (self.model_audio_proj, model_latents, model_mask),
            (self.user_audio_proj, user_latents, user_mask),
        ):
            y = proj(self.scale_latents(latents).to(x.dtype))
            if mask is not None:
                y = y * mask[:, None].to(y.dtype)
            x = x + y
        return x

    # ------------------------------------------------------- runner hooks

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        self.runtime.release(finished_req_ids)

    def create_mrv2_model_state(
        self, vllm_config: VllmConfig, encoder_cache: EncoderCache | None, device: torch.device
    ) -> HaruModelState:
        from vllm_omni.model_executor.models.haru.model_state import HaruModelState

        return HaruModelState(vllm_config, self, encoder_cache, device)

    def preprocess(
        self, input_ids: torch.Tensor, input_embeds: torch.Tensor | None, **info: Any
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        embeds = self.runtime.add_row(input_ids, input_embeds, info)
        return input_ids, embeds, {}

    # ------------------------------------------------------------ weights

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        scalars = {"latent_scale": self.latent_scale, "latent_bias": self.latent_bias}
        loaded: set[str] = set()

        def tensors() -> Iterator[tuple[str, torch.Tensor]]:
            for name, tensor in weights:
                if name in scalars:
                    scalars[name].copy_(tensor.reshape(()).to(scalars[name]))
                    loaded.add(name)
                else:
                    yield name, tensor

        loader = AutoWeightsLoader(self, ignore_unexpected_prefixes=["lm_head."])
        loaded |= loader.load_weights(tensors(), mapper=self.hf_to_vllm_mapper)
        self.runtime.warm_up()
        return loaded


def _check_supported(vllm_config: VllmConfig) -> None:
    """Fail at startup on engine options the per-step audio hooks do not support."""
    parallel = vllm_config.parallel_config
    unsupported = {
        "async_scheduling": vllm_config.scheduler_config.async_scheduling,
        "enable_prefix_caching": vllm_config.cache_config.enable_prefix_caching,
        "speculative decoding": vllm_config.speculative_config is not None,
        "tensor/pipeline parallelism": parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1,
    }
    enabled = [name for name, active in unsupported.items() if active]
    if enabled:
        raise ValueError(f"Haru does not support: {', '.join(enabled)}")
