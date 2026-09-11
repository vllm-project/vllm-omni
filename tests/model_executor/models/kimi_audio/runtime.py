# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU fixtures for native registry construction without starting an engine."""

from contextlib import nullcontext
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from vllm.config import VllmConfig
from vllm.model_executor.model_loader import utils as loader_utils

from vllm_omni.model_executor.models.registry import OmniModelRegistry


@dataclass
class CPUModelRuntime:
    """Use native config copying; omit engine validation and hardware setup."""

    model_config: SimpleNamespace
    quant_config: Any = None
    cache_config: Any = None
    parallel_config: Any = None
    device_config: Any = None
    load_config: Any = None
    additional_config: dict = field(default_factory=dict)

    with_hf_config = VllmConfig.with_hf_config

    def __post_init__(self):
        defaults = dict(
            registry=OmniModelRegistry,
            model="kimi-audio-cpu-fixture",
            model_arch="KimiAudioForConditionalGeneration",
            model_impl="vllm",
            convert_type="none",
            runner_type="generate",
            trust_remote_code=False,
            is_multimodal_model=False,
            get_model_arch_config=lambda: None,
            _get_transformers_backend_cls=lambda: "TransformersForCausalLM",
        )
        for name, value in defaults.items():
            self.model_config.__dict__.setdefault(name, value)


@pytest.fixture
def registered_model_runtime(monkeypatch):
    # Keep init_vllm_registered_model, with_hf_config, architecture resolution,
    # lazy registry imports and initialize_model. Only the surrounding engine
    # context and reload metadata collection are outside this CPU boundary.
    monkeypatch.setattr(loader_utils, "set_current_vllm_config", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(loader_utils, "record_metadata_for_reloading", lambda model: None)
    return CPUModelRuntime
