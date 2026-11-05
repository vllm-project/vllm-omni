# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Project production Qwen3-Omni deploy settings without loading the 30B checkpoint.

Resolves ``qwen3_omni_moe.yaml`` into typed stages, then uses the same builders
the engine uses (``build_engine_args_dict_from_omni_stage_config`` and
``extract_stage_metadata_from_omni_stage_config``). ``VllmConfig`` is created
from those args against a local Llama stand-in so the scheduler, cache, and
sampling objects exist without a GPU or a Qwen weight download.
"""

import copy
from dataclasses import fields

import pytest
from transformers import LlamaConfig, Qwen3OmniMoeConfig
from vllm import SamplingParams

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.omni_config import (
    VllmOmniARStageConfig,
    VllmOmniConfig,
    VllmOmniGenerationStageConfig,
)
from vllm_omni.config.pipeline_registry import resolve_pipeline_config
from vllm_omni.config.stage_config import load_deploy_config
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.engine.stage_init_utils import (
    build_engine_args_dict_from_omni_stage_config,
    extract_stage_metadata_from_omni_stage_config,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Compared after SamplingParams normalization. ``stop_token_ids`` is a pipeline
# constraint and is not part of that check.
_PROJECTED_SAMPLING_KEYS = (
    "temperature",
    "max_tokens",
    "top_k",
    "top_p",
    "repetition_penalty",
    "seed",
    "ignore_eos",
    "detokenize",
)

# The stand-in is a local Llama config. Drop fields that would bind the Qwen
# architecture or its CUDA graph settings; the compared deploy values stay.
_STAND_IN_DROPPED_ARGS = (
    "model_arch",
    "hf_config_name",
    "hf_overrides",
    "hf_config_path",
    "tokenizer",
    "compilation_config",
    "speculative_config",
    "kernel_config",
    "profiler_config",
)


def _production_stages(monkeypatch: pytest.MonkeyPatch):
    """Resolve the packaged production deploy as CUDA would, with no checkpoint."""
    monkeypatch.setattr(current_omni_platform, "device_name", "cuda")
    pipeline = resolve_pipeline_config("qwen3_omni_moe", Qwen3OmniMoeConfig(enable_audio_output=True))
    assert pipeline is not None
    deploy = load_deploy_config(get_deploy_config_path("qwen3_omni_moe.yaml"))
    config = VllmOmniConfig.from_pipeline_config(
        pipeline,
        user_deploy_config=copy.deepcopy(deploy),
    )
    return config.stage_configs


def _stand_in_model_dir(tmp_path, stage) -> str:
    """Local config long enough for this stage's batch limit, with no weights."""
    batched = stage.scheduler_config.max_num_batched_tokens or 2048
    max_len = batched
    stage_max_model_len = stage.scheduler_config.max_model_len
    if stage_max_model_len is not None and stage_max_model_len > 0:
        max_len = max(max_len, stage_max_model_len)
    if stage.scheduler_config.enable_chunked_prefill is False:
        max_len = min(max_len, batched)
    model_dir = tmp_path / f"stage-{stage.stage_id}"
    LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=64,
        max_position_embeddings=int(max_len),
    ).save_pretrained(model_dir)
    return str(model_dir)


def _project_vllm_config(stage, model_dir: str):
    engine_args = build_engine_args_dict_from_omni_stage_config(stage, model=model_dir)
    accepted = {field.name for field in fields(OmniEngineArgs)}
    kwargs = {name: value for name, value in engine_args.items() if name in accepted}
    for name in _STAND_IN_DROPPED_ARGS:
        kwargs.pop(name, None)
    kwargs["model"] = model_dir
    kwargs["skip_tokenizer_init"] = True
    return OmniEngineArgs(**kwargs).create_engine_config()


def test_production_deploy_projects_into_vllm_config_and_sampling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Resolved stage settings reach ``VllmConfig`` and normalized sampling params."""
    resolved_stages = _production_stages(monkeypatch)
    assert len(resolved_stages) == 3
    thinker, talker, code2wav = resolved_stages
    assert isinstance(thinker, VllmOmniARStageConfig)
    assert isinstance(talker, VllmOmniARStageConfig)
    assert isinstance(code2wav, VllmOmniGenerationStageConfig)
    assert [stage.stage_id for stage in resolved_stages] == [0, 1, 2]
    assert [stage.model_stage for stage in resolved_stages] == ["thinker", "talker", "code2wav"]
    assert code2wav.final_output_type == "audio"
    assert thinker.runtime_config.devices == "0"
    assert talker.runtime_config.devices == code2wav.runtime_config.devices == "1"
    assert talker.connector_config.input_connectors == {"from_stage_0": "connector_of_shared_memory"}
    assert code2wav.connector_config.input_connectors == {"from_stage_1": "connector_of_shared_memory"}

    for stage in resolved_stages:
        vllm_config = _project_vllm_config(stage, _stand_in_model_dir(tmp_path, stage))
        assert vllm_config.scheduler_config.max_num_seqs == stage.scheduler_config.max_num_seqs
        assert vllm_config.scheduler_config.max_num_batched_tokens == stage.scheduler_config.max_num_batched_tokens
        assert vllm_config.cache_config.gpu_memory_utilization == stage.cache_config.gpu_memory_utilization
        assert vllm_config.model_config.enforce_eager is stage.model_config.enforce_eager
        assert vllm_config.cache_config.enable_prefix_caching is stage.cache_config.enable_prefix_caching
        assert vllm_config.model_config.trust_remote_code is stage.model_config.trust_remote_code
        if stage.scheduler_config.enable_chunked_prefill is not None:
            assert vllm_config.scheduler_config.enable_chunked_prefill is stage.scheduler_config.enable_chunked_prefill
        if stage.scheduler_config.async_scheduling is not None:
            assert vllm_config.scheduler_config.async_scheduling is stage.scheduler_config.async_scheduling

        runtime_params = extract_stage_metadata_from_omni_stage_config(stage).default_sampling_params
        raw = stage.model_config.default_sampling_params or {}
        expected = {name: raw[name] for name in _PROJECTED_SAMPLING_KEYS if name in raw}
        assert expected
        normalized = SamplingParams(**expected)
        assert all(getattr(runtime_params, name) == getattr(normalized, name) for name in expected)
