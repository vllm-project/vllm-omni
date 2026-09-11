# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU fixtures for native registry construction without starting an engine."""

from contextlib import nullcontext
from dataclasses import dataclass, field
from types import MethodType, SimpleNamespace
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
    scheduler_config: Any = None
    compilation_config: Any = None
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
        if not hasattr(self.model_config, "multimodal_config"):
            self.model_config.multimodal_config = SimpleNamespace(skip_mm_profiling=False)
            self.model_config.get_multimodal_config = MethodType(
                lambda config: config.multimodal_config, self.model_config
            )


@pytest.fixture
def registered_model_runtime(monkeypatch):
    # Keep init_vllm_registered_model, with_hf_config, architecture resolution,
    # lazy registry imports and initialize_model. Only the surrounding engine
    # context and reload metadata collection are outside this CPU boundary.
    monkeypatch.setattr(loader_utils, "set_current_vllm_config", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(loader_utils, "record_metadata_for_reloading", lambda model: None)
    return CPUModelRuntime


@pytest.fixture(autouse=True)
def cpu_pp_group(monkeypatch):
    """CPU tests select rank ownership without constructing distributed GPUs."""
    import vllm.distributed as distributed
    import vllm.distributed.parallel_state as parallel_state
    import vllm.model_executor.offloader as offloader

    from vllm_omni.model_executor.models.kimi_audio import kimi_audio, kimi_audio_ar_stage

    group = SimpleNamespace(world_size=1, rank_in_group=0, is_first_rank=True, is_last_rank=True)
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    for module in (distributed, parallel_state, kimi_audio, kimi_audio_ar_stage):
        monkeypatch.setattr(module, "get_pp_group", lambda: group)
    monkeypatch.setattr(offloader, "get_offloader", lambda: SimpleNamespace(wrap_modules=list))
    return group


@pytest.fixture
def kimi_mm_processor():
    """Native processor and context with a local tokenizer and model config."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PretrainedConfig, PreTrainedTokenizerFast
    from vllm.multimodal.processing import InputProcessingContext

    from vllm_omni.model_executor.models.kimi_audio.kimi_audio import KimiAudioForConditionalGeneration

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")), unk_token="[UNK]"
    )

    def make(*, offset, use_whisper=True, stage="kimi_audio_ar", audio_limit=8):
        mm_config = SimpleNamespace(
            enable_mm_embeds=False,
            mm_hasher_algorithm="sha256",
            get_limit_per_prompt=lambda modality: audio_limit,
            mm_processor_cache_gb=0.02,
            limit_per_prompt={},
        )
        config = SimpleNamespace(
            model="kimi-audio-cpu-fixture",
            model_stage=stage,
            multimodal_config=mm_config,
            hf_config=PretrainedConfig(kimia_token_offset=offset, use_whisper_feature=use_whisper),
            get_multimodal_config=lambda: mm_config,
        )
        factories = KimiAudioForConditionalGeneration._processor_factory
        info = factories.info(InputProcessingContext(config, tokenizer))
        return factories.processor(info, factories.dummy_inputs(info))

    return make
