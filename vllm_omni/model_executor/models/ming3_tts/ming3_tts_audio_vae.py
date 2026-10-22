# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Streaming Ming3 AudioVAE stage using the shared Ming implementation."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.model_executor.models.common.ming.audio_vae import AudioVAE
from vllm_omni.model_executor.models.ming_tts.ming_tts_audio_vae import (
    MingAudioVAEModel,
)

from .config_ming3_tts import Ming3TTSRuntimeConfig


class Ming3TTSAudioVAEModel(MingAudioVAEModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Keep the mature streaming decode/cache implementation but construct it
        # from Ming3's audio_config rather than Ming's audio_tokenizer_config.
        super(MingAudioVAEModel, self).__init__()
        del prefix
        self.vllm_config = vllm_config
        self.ming_config = Ming3TTSRuntimeConfig.from_hf_config(vllm_config.model_config.hf_config)
        self.ming_config.validate()
        self.ming3_config = self.ming_config
        self.audio = AudioVAE(self.ming_config.audio_tokenizer_config, build_encoder=False)
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True
        self._past_key_values: dict[str, Any] = {}
        self._stream_state: dict[str, tuple[Any, Any, Any]] = {}
        self._patch_totals: dict[str, int] = {}
        self._sample_totals: dict[str, int] = {}
        self._state_access_times: OrderedDict[str, float] = OrderedDict()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = list(weights)
        params = dict(self.named_parameters(remove_duplicate=False))
        buffers = dict(self.named_buffers(remove_duplicate=False))
        targets = set(params) | set(buffers)
        required = set(self.state_dict())
        unknown = [name for name, _ in weights if name not in targets]
        if unknown:
            raise RuntimeError(f"Ming3 AudioVAE received {len(unknown)} unknown checkpoint tensors: {unknown[:8]}")
        loaded: set[str] = set()
        for name, value in weights:
            if name in params:
                parameter = params[name]
                loader = getattr(parameter, "weight_loader", default_weight_loader)
                loader(parameter, value)
            else:
                buffer = buffers[name]
                buffer.copy_(value.to(device=buffer.device, dtype=buffer.dtype))
            loaded.add(name)
        missing = {name for name in required if name.startswith("audio.")} - loaded
        if missing:
            raise RuntimeError(
                f"Ming3 AudioVAE left {len(missing)} checkpoint-backed tensors unloaded: {sorted(missing)[:8]}"
            )
        return loaded

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        for request_id in finished_req_ids:
            self._clear_request_state(request_id)


__all__ = ["Ming3TTSAudioVAEModel"]
