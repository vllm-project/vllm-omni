# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ming3 continuous-latent AR stage."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.models.bailing_moe_v3 import BailingMoeV3ForCausalLM
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid, SupportsPP
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix

from vllm_omni.model_executor.models.common.ming.aggregator import Aggregator
from vllm_omni.model_executor.models.common.ming.cfm_head import FlowLoss
from vllm_omni.model_executor.models.common.ming.continuous_tts import (
    STOP_REASON_CODES,
    STOP_REASON_CONTINUE,
    resolve_stop_decision,
)
from vllm_omni.model_executor.models.ming_tts.config_ming_tts import (
    KEY_MAX_DECODE_STEPS,
    KEY_MIN_DECODE_STEPS,
    KEY_REQUEST_ID,
)
from vllm_omni.model_executor.models.ming_tts.ming_tts_llm import (
    MingLLMModel,
    _resolve_ming_runtime_dtype,
    _warn_missing_prefix,
)
from vllm_omni.model_executor.models.ming_tts.patch_emission import (
    _resolve_optional_runtime_int,
    _resolve_runtime_int,
)

from .config_ming3_tts import (
    Ming3TTSRuntimeConfig,
    get_ming3_backbone_config,
)
from .constants import (
    KEY_STOP_REASON,
    SPEAKER_EMBEDDING_DIM,
)


def _assert_all_parameter_tensors_loaded(module: nn.Module, loaded_names: set[str], *, label: str) -> None:
    aliases_by_tensor: dict[int, set[str]] = {}
    for name, parameter in module.named_parameters(remove_duplicate=False):
        aliases_by_tensor.setdefault(id(parameter), set()).add(name)
    missing = [sorted(aliases) for aliases in aliases_by_tensor.values() if aliases.isdisjoint(loaded_names)]
    if missing:
        raise RuntimeError(
            f"Ming3 {label}: {len(missing)} distinct parameter tensors were not loaded. "
            f"First few aliases: {missing[:5]}"
        )


class Ming3TTSLLMModel(MingLLMModel, HasInnerState, IsHybrid, SupportsPP):
    packed_modules_mapping = BailingMoeV3ForCausalLM.packed_modules_mapping
    latent_patch_key = "ming3_latent_patch"
    next_embeds_key = "ming3_next_embeds"
    new_history_key = "ming3_new_history"
    decode_step_key = "ming3_decode_step"
    has_patch_key = "ming3_has_patch"
    stop_reason_key = KEY_STOP_REASON
    stop_probability_key = "ming3_stop_prob"
    stop_reason_codes = STOP_REASON_CODES
    vectorize_cfm_decode = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.ming_config = Ming3TTSRuntimeConfig.from_hf_config(vllm_config.model_config.hf_config)
        self.ming_config.validate()
        self.ming3_config = self.ming_config
        self.vllm_config = vllm_config
        self.prefix = prefix
        self.fm_dtype = _resolve_ming_runtime_dtype(vllm_config)
        self.model = BailingMoeV3ForCausalLM(
            vllm_config=get_ming3_backbone_config(vllm_config),
            prefix=maybe_prefix(prefix, "model"),
        )
        self.linear_proj_audio = Aggregator(
            in_channels=self.ming_config.latent_dim,
            llm_input_dim=self.ming_config.llm_hidden_size,
            **self.ming_config.aggregator_config,
        )
        self.flowloss = FlowLoss(
            z_channels=self.ming_config.latent_dim,
            llm_cond_dim=self.ming_config.llm_hidden_size,
            **self.ming_config.ditar_config,
        )
        self.stop_head = nn.Linear(self.ming_config.llm_hidden_size, 2, bias=True)
        self.spk_head = nn.Linear(SPEAKER_EMBEDDING_DIM, self.ming_config.llm_hidden_size, bias=True)
        for module in (
            self.flowloss,
            self.linear_proj_audio,
            self.stop_head,
            self.spk_head,
        ):
            module.to(dtype=self.fm_dtype)
        self._pending_postprocess_updates: dict[str, dict[str, Any]] = {}
        self._last_ming_next_token_ids = None
        self._last_text_mode = False
        self._init_cfm_graph_state()
        self._request_generators: dict[str, torch.Generator] = {}

    def make_empty_intermediate_tensors(self, *args: Any, **kwargs: Any):
        return self.model.make_empty_intermediate_tensors(*args, **kwargs)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        return BailingMoeV3ForCausalLM.get_mamba_state_shape_from_config(get_ming3_backbone_config(vllm_config))

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        return BailingMoeV3ForCausalLM.get_mamba_state_dtype_from_config(get_ming3_backbone_config(vllm_config))

    @classmethod
    def get_mamba_state_copy_func(cls):
        return BailingMoeV3ForCausalLM.get_mamba_state_copy_func()

    def _validate_decode_window(self, request_infos: list[dict[str, Any]]) -> None:
        for index, info in enumerate(request_infos):
            maximum = _resolve_runtime_int(info, KEY_MAX_DECODE_STEPS, self.ming_config.max_decode_steps)
            # The shared Ming loop reads this field with a zero default; resolve Ming3's fallback here.
            info.setdefault(KEY_MIN_DECODE_STEPS, self.ming_config.stop_head_min_steps)
            minimum = max(1, _resolve_optional_runtime_int(info, KEY_MIN_DECODE_STEPS, 0))
            if maximum < minimum:
                raise ValueError(
                    f"Ming3 request idx={index}: maximum decode steps {maximum} is smaller than minimum {minimum}"
                )

    def _resolve_stop_decision(
        self,
        *,
        step: int,
        stop_prob: float,
        stop_threshold: float,
        min_stop_step: int,
        min_decode_steps: int,
        max_decode_steps: int,
        audio_dummy_token_id: int,
        text_eos_token_id: int,
    ) -> tuple[str, bool, bool, int, int]:
        decision = resolve_stop_decision(
            emitted_steps=step + 1,
            stop_probability=stop_prob,
            stop_threshold=stop_threshold,
            minimum_steps=max(1, min_decode_steps),
            maximum_steps=max_decode_steps,
            continue_token_id=audio_dummy_token_id,
            stop_token_id=text_eos_token_id,
        )
        return (
            decision.reason,
            decision.should_stop,
            decision.forced,
            decision.minimum_steps,
            decision.next_token_id,
        )

    def _get_request_generator(self, req_info: dict[str, Any], device: torch.device) -> torch.Generator | None:
        request_id = req_info.get(KEY_REQUEST_ID, req_info.get("request_id"))
        if not isinstance(request_id, str) or not request_id:
            return None
        generator = self._request_generators.get(request_id)
        if generator is not None:
            return generator
        raw_seed = req_info.get("tts_local_seed")
        if raw_seed is None and req_info.get("prompt_waveform") is None:
            return None
        generator = torch.Generator(device=device)
        if raw_seed is None:
            generator.seed()
        else:
            generator.manual_seed(int(raw_seed))
        self._request_generators[request_id] = generator
        return generator

    def _on_stop_decision(self, req_info: dict[str, Any], stop_reason: str) -> None:
        if stop_reason != STOP_REASON_CONTINUE:
            request_id = req_info.get(KEY_REQUEST_ID)
            if isinstance(request_id, str):
                self._request_generators.pop(request_id, None)

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        for request_id in finished_req_ids:
            self._request_generators.pop(request_id, None)
            self._pending_postprocess_updates.pop(request_id, None)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = list(weights)
        unknown = [
            name
            for name, _ in weights
            if not name.startswith(("model.", "linear_proj_audio.", "flowloss.", "stop_head.", "spk_head."))
        ]
        if unknown:
            raise RuntimeError(f"Ming3 Stage-0 received {len(unknown)} unclassified tensors: {unknown[:8]}")

        backbone_weights = [(name[len("model.") :], value) for name, value in weights if name.startswith("model.")]
        if not backbone_weights:
            raise RuntimeError("Ming3 Stage-0 received no model.* backbone tensors")
        loaded_backbone = self.model.load_weights(backbone_weights)
        _assert_all_parameter_tensors_loaded(self.model, loaded_backbone, label="Bailing-v3 backbone")

        tts_weights = [(name, value) for name, value in weights if not name.startswith("model.")]
        loaded_tts = AutoWeightsLoader(self).load_weights(tts_weights)
        params = dict(self.named_parameters(remove_duplicate=False))
        for module_name in ("flowloss", "linear_proj_audio", "stop_head", "spk_head"):
            _warn_missing_prefix(
                module_name,
                params,
                loaded_tts,
                prefix=f"{module_name}.",
            )
        self._warmup_cfm_graphs()
        return {f"model.{name}" for name in loaded_backbone} | loaded_tts


__all__ = ["Ming3TTSLLMModel"]
