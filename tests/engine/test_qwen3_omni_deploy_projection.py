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
from vllm_omni.config.stage_config import (
    PIPELINE_WIDE_ENGINE_FIELDS,
    load_deploy_config,
    resolve_deploy_yaml,
)
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
# ``compilation_config`` is in this list, so this CPU test does not cover
# production cudagraphs. That path stays in ``test_video_to_audio``.
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

# Builder outputs that are not ``OmniEngineArgs`` fields. ``scheduler_cls`` is
# chosen by stage topology and is not a per-stage engine argument.
_NON_ENGINE_ARGS = ("scheduler_cls",)
_ALLOWED_DROPPED_ARGS = frozenset(_STAND_IN_DROPPED_ARGS) | frozenset(_NON_ENGINE_ARGS)

# Scheduler, cache, and model fields projected into ``VllmConfig``. The tuple
# is the table order, so a new key is added in one place.
_ENGINE_VALUE_ATTRS = {
    "max_num_seqs": ("scheduler_config", "max_num_seqs"),
    "max_num_batched_tokens": ("scheduler_config", "max_num_batched_tokens"),
    "gpu_memory_utilization": ("cache_config", "gpu_memory_utilization"),
    "enforce_eager": ("model_config", "enforce_eager"),
    "enable_prefix_caching": ("cache_config", "enable_prefix_caching"),
    "trust_remote_code": ("model_config", "trust_remote_code"),
    "enable_chunked_prefill": ("scheduler_config", "enable_chunked_prefill"),
    "async_scheduling": ("scheduler_config", "async_scheduling"),
}
_DEPLOY_ENGINE_KEYS = tuple(_ENGINE_VALUE_ATTRS)
# Only keys the assertions read. Other stage YAML stays out of ``explicit``.
_EXPLICIT_STAGE_KEYS = frozenset(_DEPLOY_ENGINE_KEYS) | {"default_sampling_params"}


def _production_resolution(monkeypatch: pytest.MonkeyPatch):
    """Resolve the packaged production deploy as CUDA would, with no checkpoint.

    ``raw`` is the deploy dict before dataclass defaults are applied.
    """
    monkeypatch.setattr(current_omni_platform, "device_name", "cuda")
    pipeline = resolve_pipeline_config("qwen3_omni_moe", Qwen3OmniMoeConfig(enable_audio_output=True))
    assert pipeline is not None
    deploy_path = get_deploy_config_path("qwen3_omni_moe.yaml")
    raw = resolve_deploy_yaml(deploy_path)
    deploy = load_deploy_config(deploy_path)
    config = VllmOmniConfig.from_pipeline_config(
        pipeline,
        user_deploy_config=copy.deepcopy(deploy),
    )
    return raw, config.stage_configs


def _explicit_stage_settings(raw_deploy: dict, stage_id: int, platform: str = "cuda") -> dict:
    """YAML values the projection asserts read. Omitted keys are absent.

    Stage entries, that platform's overlay, and pipeline-wide keys are
    consulted, but only ``_EXPLICIT_STAGE_KEYS`` are kept. Dataclass
    defaults never appear here.
    """
    by_id = {stage["stage_id"]: stage for stage in raw_deploy.get("stages") or []}
    explicit: dict = {}

    def _keep(source: dict) -> None:
        for key, value in source.items():
            if key in _EXPLICIT_STAGE_KEYS:
                explicit[key] = value

    def _take(mapping: dict) -> None:
        _keep(mapping)
        runtime = mapping.get("runtime") or {}
        if isinstance(runtime, dict):
            _keep(runtime)
        for nested_name in ("engine_args", "engine_extras"):
            nested = mapping.get(nested_name) or {}
            if isinstance(nested, dict):
                _keep(nested)

    _take(by_id[stage_id])
    platform_section = (raw_deploy.get("platforms") or {}).get(platform) or {}
    for overlay in platform_section.get("stages") or []:
        if overlay.get("stage_id") == stage_id:
            _take(overlay)
    for name in PIPELINE_WIDE_ENGINE_FIELDS:
        if name in _EXPLICIT_STAGE_KEYS and name in raw_deploy and name not in explicit:
            explicit[name] = raw_deploy[name]
    return explicit


def _engine_value(config, name: str):
    """Read one projected field from a resolved stage or a ``VllmConfig``."""
    group, attr = _ENGINE_VALUE_ATTRS[name]
    return getattr(getattr(config, group), attr)


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
    # Keys removed because OmniEngineArgs rejects them, or because the stand-in
    # cannot consume them, must be on the allowlist. A renamed field fails here
    # instead of disappearing from the projected config.
    dropped = set(engine_args) - set(kwargs)
    unexpected = sorted(dropped - _ALLOWED_DROPPED_ARGS)
    assert not unexpected, f"stage {stage.stage_id} silently dropped engine args {unexpected}"
    kwargs["model"] = model_dir
    kwargs["skip_tokenizer_init"] = True
    return OmniEngineArgs(**kwargs).create_engine_config()


def test_production_deploy_projects_into_vllm_config_and_sampling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Resolved stage settings reach ``VllmConfig`` and normalized sampling params."""
    raw_deploy, resolved_stages = _production_resolution(monkeypatch)
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
        explicit = _explicit_stage_settings(raw_deploy, stage.stage_id)
        vllm_config = _project_vllm_config(stage, _stand_in_model_dir(tmp_path, stage))
        for name in _DEPLOY_ENGINE_KEYS:
            stage_value = _engine_value(stage, name)
            vllm_value = _engine_value(vllm_config, name)
            if name == "enforce_eager":
                # Unset on the CUDA production thinker/talker. ``ModelConfig``
                # stores False; the stage may still be None. Do not require ``is``.
                assert bool(vllm_value) == bool(stage_value)
            elif stage_value is not None:
                assert vllm_value == stage_value
            if name not in explicit:
                continue
            yaml_value = explicit[name]
            if name == "enforce_eager":
                assert bool(vllm_value) == bool(yaml_value)
                if stage_value is not None:
                    assert bool(stage_value) == bool(yaml_value)
            else:
                assert stage_value == yaml_value
                assert vllm_value == yaml_value

        runtime_params = extract_stage_metadata_from_omni_stage_config(stage).default_sampling_params
        deploy_sampling = explicit.get("default_sampling_params") or {}
        expected = {name: deploy_sampling[name] for name in _PROJECTED_SAMPLING_KEYS if name in deploy_sampling}
        assert expected
        assert expected.items() <= (stage.model_config.default_sampling_params or {}).items()
        normalized = SamplingParams(**expected)
        assert all(getattr(runtime_params, name) == getattr(normalized, name) for name in expected)
