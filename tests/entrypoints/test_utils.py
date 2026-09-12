# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unit tests for vllm_omni.entrypoints.utils module."""

import logging
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pytest_mock import MockerFixture
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.config.resolver import (
    OmniConfigResolution,
    _convert_dataclasses_to_dict,
    _filter_dict_like_object,
    resolve_omni_config,
)
from vllm_omni.config.stage_config import PipelineConfig
from vllm_omni.config.yaml_util import create_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.stage_init_utils import get_stage_connector_spec, load_omni_transfer_config_for_model
from vllm_omni.entrypoints.utils import (
    _try_resolve_omni_model_type,
    coerce_param_message_types,
    filter_dataclass_kwargs,
    load_and_resolve_stage_configs,
    resolve_model_config_path,
)
from vllm_omni.model_executor.models.qwen3_omni.pipeline import QWEN3_OMNI_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestFilterDictLikeObject:
    """Test suite for _filter_dict_like_object function."""

    def test_simple_dict(self):
        """Test filtering a simple dictionary with no callables."""
        input_dict = {"key1": "value1", "key2": 42, "key3": [1, 2, 3]}
        result = _filter_dict_like_object(input_dict)

        assert result == input_dict
        assert isinstance(result, dict)

    def test_dict_with_nested_values(self):
        """Test filtering dict with nested dict and list values."""
        input_dict = {
            "level1": {
                "level2": {"key": "value"},
                "list": [1, 2, 3],
            },
            "simple": "string",
        }

        result = _filter_dict_like_object(input_dict)

        # Nested dicts and lists should be recursively processed
        assert result["simple"] == "string"
        assert isinstance(result["level1"], dict)

    def test_dict_with_dataclass_values(self):
        """Test filtering dict containing dataclass values."""

        @dataclass
        class TestDataclass:
            field1: str
            field2: int

        obj = TestDataclass(field1="test", field2=42)
        input_dict = {"data": obj, "normal": "value"}

        result = _filter_dict_like_object(input_dict)

        # Dataclass should be converted to dict by recursive _convert_dataclasses_to_dict
        assert "data" in result
        assert "normal" in result
        assert result["normal"] == "value"

    def test_dict_with_counter_values(self):
        """Test filtering dict containing Counter objects."""
        counter_obj = Counter({"a": 1, "b": 2})
        input_dict = {"counter": counter_obj, "normal": "value"}

        result = _filter_dict_like_object(input_dict)

        # Counter should be converted to regular dict
        assert "counter" in result
        assert "normal" in result
        assert result["normal"] == "value"

    def test_empty_dict(self):
        """Test filtering an empty dictionary."""
        result = _filter_dict_like_object({})
        assert result == {}
        assert isinstance(result, dict)

    def test_dict_with_set_values(self):
        """Test filtering dict with set values."""
        input_dict = {"set_key": {1, 2, 3}, "normal": "value"}

        result = _filter_dict_like_object(input_dict)

        assert "set_key" in result
        assert "normal" in result
        # Set should be converted to list by _convert_dataclasses_to_dict
        assert result["normal"] == "value"

    def test_dict_with_none_values(self):
        """Test filtering dict with None values."""
        input_dict = {"key1": None, "key2": "value", "key3": 0}

        result = _filter_dict_like_object(input_dict)

        assert result == input_dict

    def test_dict_with_mixed_types(self):
        """Test filtering dict with mixed value types."""
        input_dict = {
            "string": "hello",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
            "list": [1, 2, 3],
            "tuple": (1, 2, 3),
            "set": {1, 2, 3},
            "dict": {"nested": "value"},
        }

        result = _filter_dict_like_object(input_dict)

        assert "string" in result
        assert "int" in result
        assert "float" in result
        assert "bool" in result
        assert "none" in result
        assert "list" in result
        assert "tuple" in result
        assert "set" in result
        assert "dict" in result

    def test_dict_preserves_key_types(self):
        """Test that dict key types are preserved."""
        input_dict = {
            "string_key": "value1",
            42: "value2",
            (1, 2): "value3",  # tuple as key
        }

        result = _filter_dict_like_object(input_dict)

        # Keys should remain the same
        assert "string_key" in result
        assert 42 in result
        assert (1, 2) in result

    def test_dict_with_recursive_structure(self, mocker: MockerFixture):
        """Test filtering dict with recursive/complex nested structure."""
        input_dict = {
            "level1": {
                "level2": {
                    "level3": {"key": "value"},
                    "callable": lambda x: x,
                }
            },
            "normal": "value",
        }

        mocker.patch("vllm_omni.config.resolver.logger")
        result = _filter_dict_like_object(input_dict)

        # Normal key should exist
        assert "normal" in result
        # Level1 should exist
        assert "level1" in result

    def test_integration_with_convert_dataclasses(self, mocker: MockerFixture):
        """Test that _filter_dict_like_object integrates properly with _convert_dataclasses_to_dict."""

        @dataclass
        class Config:
            name: str
            count: int

        input_dict = {
            "config": Config(name="test", count=5),
            "func": lambda x: x,
            "normal": "value",
        }

        mocker.patch("vllm_omni.config.resolver.logger")
        result = _filter_dict_like_object(input_dict)

        # Callable should be filtered
        assert "func" not in result
        # Config should be converted to dict
        assert "config" in result
        assert "normal" in result


class TestConvertDataclassesToDict:
    """Test suite for _convert_dataclasses_to_dict function."""

    def test_uses_filter_dict_like_object(self, mocker: MockerFixture):
        """Test that _convert_dataclasses_to_dict uses _filter_dict_like_object for dicts."""
        input_dict = {
            "normal": "value",
            "callable": lambda x: x,
        }

        mocker.patch("vllm_omni.config.resolver.logger")
        result = _convert_dataclasses_to_dict(input_dict)

        # Callable should be filtered out by _filter_dict_like_object
        assert "normal" in result
        assert "callable" not in result


class TestFilterDataclassKwargs:
    """Test basic functionality of filter_dataclass_kwargs."""

    def test_simple_filtering(self):
        """Test basic dataclass kwargs filtering."""

        @dataclass
        class SimpleConfig:
            name: str
            count: int

        kwargs = {"name": "test", "count": 42, "invalid": "should_be_removed"}
        result = filter_dataclass_kwargs(SimpleConfig, kwargs)

        assert "name" in result
        assert "count" in result
        assert "invalid" not in result

    def test_invalid_dataclass_raises_error(self):
        """Test that non-dataclass raises ValueError."""
        with pytest.raises(ValueError, match="is not a dataclass"):
            filter_dataclass_kwargs(dict, {})

    def test_invalid_kwargs_type_raises_error(self):
        """Test that non-dict kwargs raises ValueError."""

        @dataclass
        class SimpleConfig:
            name: str

        with pytest.raises(ValueError, match="kwargs must be a dictionary"):
            filter_dataclass_kwargs(SimpleConfig, "invalid")

    def test_filters_omni_engine_args_unknown_fields(self, caplog):
        """Test that OmniEngineArgs kwargs are filtered to valid fields only,
        and that the WARNING contract fires for every drop — the affordance
        that lets a ``--stage-overrides`` typo (``{"0":{"kv_cache_dtpye":...}}``)
        surface in the log rather than being silently misapplied downstream.
        """
        kwargs = {
            "model": "dummy",
            "stage_id": 1,
            "engine_output_type": "image",
            "unknown_field": "drop_me",
        }

        with caplog.at_level(logging.WARNING, logger="vllm_omni.entrypoints.utils"):
            result = filter_dataclass_kwargs(OmniEngineArgs, kwargs)

        assert "model" in result
        assert "stage_id" in result
        assert "engine_output_type" in result
        assert "unknown_field" not in result
        assert any(rec.levelno == logging.WARNING and "unknown_field" in rec.message for rec in caplog.records), (
            f"expected WARNING naming 'unknown_field'; got {[r.message for r in caplog.records]}"
        )

    def test_filters_omni_diffusion_config_union_dataclass(self):
        """Test that OmniDiffusionConfig filters nested dataclass in Union fields."""
        kwargs = {
            "model": "dummy",
            "cache_config": {
                "rel_l1_thresh": 0.3,
                "extra_param": "should_drop",
            },
            "unknown_top": "drop_me",
        }

        result = filter_dataclass_kwargs(OmniDiffusionConfig, kwargs)

        assert "model" in result
        assert "cache_config" in result
        assert "unknown_top" not in result
        assert result["cache_config"]["rel_l1_thresh"] == 0.3
        assert "extra_param" not in result["cache_config"]


class TestResolveModelConfigPath:
    """Test suite for resolve_model_config_path function with diffusers format models."""

    def test_glm_image_diffusers_format_resolution(self, mocker: MockerFixture):
        """Test GlmImagePipeline diffusers class resolves to glm_image config."""
        mocker.patch(
            "vllm_omni.entrypoints.utils.file_or_path_exists",
            return_value=True,
        )
        mocker.patch(
            "vllm_omni.entrypoints.utils._try_get_class_name_from_diffusers_config",
            return_value="GlmImagePipeline",
        )
        mocker.patch(
            "vllm_omni.entrypoints.utils.current_omni_platform.get_default_stage_config_path",
            return_value="vllm_omni/model_executor/stage_configs",
        )

        original_exists = os.path.exists

        def mock_exists(path):
            if "glm_image.yaml" in str(path):
                return True
            return original_exists(path)

        mocker.patch("os.path.exists", side_effect=mock_exists)

        result = resolve_model_config_path("zai-org/GLM-Image")

        assert result is not None
        assert "glm_image.yaml" in result

    def test_uses_registry_default_deploy_when_hf_model_type_differs(
        self,
        mocker: MockerFixture,
        tmp_path,
    ):
        from vllm_omni.config.stage_config import _DEPLOY_DIR as _REAL_DEPLOY_DIR
        from vllm_omni.entrypoints import utils as utils_mod

        hf_config = SimpleNamespace(
            model_type="minicpmo",
            architectures=["MiniCPMO"],
            version="4.5",
        )
        model = "local/minicpmo-registry-default-test"
        mocker.patch(
            "vllm_omni.entrypoints.utils.get_config",
            return_value=hf_config,
        )
        mocker.patch(
            "vllm_omni.config.config_factory.get_config",
            return_value=hf_config,
        )
        # A bare deploy/<hf_model_type>.yaml must not win over the registered
        # pipeline default when HF model_type is not an OMNI_PIPELINES key.
        (tmp_path / "minicpmo.yaml").write_text("stages: []\n", encoding="utf-8")
        (tmp_path / "minicpmo_4_5.yaml").write_bytes((_REAL_DEPLOY_DIR / "minicpmo_4_5.yaml").read_bytes())
        mocker.patch.object(utils_mod, "_DEPLOY_DIR", tmp_path)

        result = resolve_model_config_path(model)

        assert result is not None
        assert Path(result).as_posix().endswith("minicpmo_4_5.yaml")
        assert Path(result).name == "minicpmo_4_5.yaml"
        transfer_config = load_omni_transfer_config_for_model(model, result)
        assert transfer_config is not None

        sender = get_stage_connector_spec(
            omni_transfer_config=transfer_config,
            stage_id=1,
            async_chunk=True,
        )
        receiver = get_stage_connector_spec(
            omni_transfer_config=transfer_config,
            stage_id=2,
            async_chunk=True,
        )
        assert sender["extra"]["role"] == "sender"
        assert receiver["extra"]["role"] == "receiver"
        assert sender["extra"]["connector_get_sleep_s"] == 0.01
        assert sender["extra"]["connector_get_max_wait_first_chunk"] == 3000
        assert sender["extra"]["connector_get_max_wait"] == 300

    def test_object_storage_uri_resolves_via_materialized_configs(
        self,
        mocker: MockerFixture,
        tmp_path,
    ):
        """An object-storage URI must not reach HF helpers; resolution reads the
        locally materialized config files instead."""
        hf_config = SimpleNamespace(
            model_type="qwen3_omni_moe",
            architectures=["Qwen3OmniMoeForConditionalGeneration"],
        )
        uri = "s3://qwen3-tts-models/Qwen3-Omni-30B-A3B-Instruct"
        materialize = mocker.patch(
            "vllm_omni.entrypoints.utils.materialize_object_storage_configs",
            return_value=str(tmp_path),
        )
        get_config = mocker.patch(
            "vllm_omni.entrypoints.utils.get_config",
            return_value=hf_config,
        )

        result = resolve_model_config_path(uri)

        materialize.assert_called_once_with(uri)
        get_config.assert_called_once()
        assert get_config.call_args[0][0] == str(tmp_path)
        assert result is not None
        assert Path(result).as_posix().endswith("qwen3_omni_moe.yaml")


class TestTryResolveOmniModelType:
    """Name matching must only scan the last path component."""

    def test_uri_bucket_name_cannot_hijack_pipeline(self):
        # "qwen3-tts-models" normalizes to "qwen3ttsmodels", which contains the
        # registered key "qwen3tts" — only the basename may participate, so a
        # misleadingly named bucket selects nothing.
        assert _try_resolve_omni_model_type("s3://qwen3-tts-models/plain-checkpoint") is None

    def test_basename_still_matches_pipeline(self):
        assert _try_resolve_omni_model_type("gs://any-bucket/Fun-CosyVoice3-0.5B-2512") == "cosyvoice3"


class TestLoadAndResolveStageConfigs:
    def test_load_and_resolve_with_kwargs(self, mocker: MockerFixture):
        """Ensure that kwargs survive default stage creation."""
        engine_backend = "vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine"
        kwargs = {"dtype": torch.float32, "engine_backend": engine_backend}
        kwargs["revision"] = "pinned-revision"
        mocker.patch("vllm_omni.entrypoints.utils.resolve_model_config_path", return_value=None)
        mocker.patch("vllm_omni.entrypoints.utils.load_stage_configs_from_model", return_value=([], None))

        config_path, stage_configs, _ = load_and_resolve_stage_configs(
            model="black-forest-labs/FLUX.2-klein-4B",
            kwargs=kwargs,
            trust_remote_code=False,
            default_stage_cfg_factory=lambda: AsyncOmniEngine._create_default_diffusion_stage_cfg(kwargs),
        )
        assert config_path is None
        assert len(stage_configs) == 1
        assert "dtype" in stage_configs[0]["engine_args"]
        assert stage_configs[0]["engine_args"]["engine_backend"] == engine_backend
        assert stage_configs[0]["engine_args"]["revision"] == "pinned-revision"


class TestResolveOmniConfig:
    def test_bare_deploy_name_returns_packaged_resolved_path(self, tmp_path, mocker: MockerFixture):
        deploy_name = "qwen3_omni_moe.yaml"
        deploy_path = tmp_path / deploy_name
        deploy_path.write_text(
            "duplex_session:\n  server_vad_model_path: /models/silero_vad.onnx\nstages: []\n",
            encoding="utf-8",
        )
        mocker.patch("vllm_omni.config.config_factory._DEPLOY_DIR", tmp_path)
        mocker.patch("vllm_omni.config.omni_config._DEPLOY_DIR", tmp_path)
        mocker.patch(
            "vllm_omni.config.config_factory.StageConfigFactory.get_pipeline_config",
            return_value=QWEN3_OMNI_PIPELINE,
        )
        mocker.patch("vllm_omni.config.omni_config.read_checkpoint_quantization_config", return_value=None)

        resolved = resolve_omni_config(
            "dummy-model",
            trust_remote_code=False,
            deploy_config_path=deploy_name,
            cli_overrides={},
            stage_overrides=None,
            strategy_config_path=None,
        )

        assert resolved.config_path == str(deploy_path)

    def test_stage_lookup_error_lists_resolved_ids(self):
        resolved = OmniConfigResolution(
            config_path=None,
            stage_configs=(SimpleNamespace(stage_id=1), SimpleNamespace(stage_id=3)),
        )

        with pytest.raises(KeyError, match=r"no stage 2; resolved stages: \[1, 3\]"):
            resolved.stage_by_id(2)

    def test_load_and_resolve_with_kwargs(self, mocker: MockerFixture):
        """Ensure that generic diffusion overrides survive resolution."""
        engine_backend = "vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine"
        mocker.patch(
            "vllm_omni.config.resolver.StageConfigFactory.create_from_model",
            return_value=None,
        )
        mocker.patch(
            "vllm_omni.config.resolver._resolve_generic_diffusion_model_class",
            return_value=(True, "FluxPipeline"),
        )
        kwargs = {
            "dtype": torch.float32,
            "engine_backend": engine_backend,
            "revision": "pinned-revision",
        }
        resolved = resolve_omni_config(
            "black-forest-labs/FLUX.2-klein-4B",
            trust_remote_code=False,
            deploy_config_path=None,
            cli_overrides=kwargs,
            stage_overrides=None,
            strategy_config_path=None,
        )
        assert resolved.config_path is None
        assert len(resolved.stage_configs) == 1
        engine_args = resolved.stage_configs[0]["engine_args"]
        assert "dtype" in engine_args
        assert engine_args["engine_backend"] == engine_backend
        assert engine_args["revision"] == "pinned-revision"

    def test_generic_diffusion_uses_registered_model_metadata(self, mocker: MockerFixture):
        mocker.patch(
            "vllm_omni.config.resolver.StageConfigFactory.create_from_model",
            return_value=None,
        )
        resolve_model_class = mocker.patch(
            "vllm_omni.config.resolver.resolve_model_class_name",
            return_value="WanImageToVideoPipeline",
        )
        mocker.patch(
            "vllm_omni.config.resolver.DiffusionModelRegistry.get_supported_archs",
            return_value={"WanImageToVideoPipeline"},
        )

        resolved = resolve_omni_config(
            "/models/Wan2.2-I2V",
            trust_remote_code=False,
            deploy_config_path=None,
            cli_overrides={
                "diffusion_load_format": "diffusers",
                "revision": "pinned-revision",
            },
            stage_overrides=None,
            strategy_config_path=None,
        )

        stage = resolved.stage_configs[0]
        resolve_model_class.assert_called_once_with(
            "/models/Wan2.2-I2V",
            "diffusers",
            "pinned-revision",
        )
        assert stage.engine_args.model_class_name == "WanImageToVideoPipeline"
        assert stage.final_output_type == "video"

    def test_registered_pipeline_uses_structured_metadata_and_preserves_override_trust(self, mocker: MockerFixture):
        endpoint_restriction = SimpleNamespace(name="chat")
        structured_config = SimpleNamespace(
            orchestrator_config=SimpleNamespace(deploy_config_path="/resolved/deploy.yaml"),
            pipeline_config=SimpleNamespace(endpoint_restrictions=(endpoint_restriction,)),
        )
        runtime_stage = create_config(
            {
                "stage_id": 1,
                "runtime": {"devices": "1,2,3", "num_replicas": 3},
                "engine_args": {"model": "dummy-model"},
            }
        )
        legacy_stage = SimpleNamespace(to_omegaconf=lambda: runtime_stage)
        create_structured = mocker.patch(
            "vllm_omni.config.resolver.StageConfigFactory.create_from_model",
            return_value=structured_config,
        )
        create_legacy = mocker.patch(
            "vllm_omni.config.resolver.StageConfigFactory._resolve_legacy_from_registry",
            return_value=SimpleNamespace(
                stage_configs=[legacy_stage],
                pipeline_config=structured_config.pipeline_config,
                omni_lb_policy="round_robin",
            ),
        )
        strategy_specs = {"stage_1": {"dp": 3}}
        load_strategy = mocker.patch(
            "vllm_omni.config.resolver._load_strategy_specs",
            return_value=strategy_specs,
        )

        resolved = resolve_omni_config(
            "dummy-model",
            trust_remote_code=None,
            deploy_config_path="deploy.yaml",
            cli_overrides={"dtype": "bfloat16", "trust_remote_code": True},
            stage_overrides={"1": {"tensor_parallel_size": 2}},
            strategy_config_path="strategy.yaml",
        )

        expected_overrides = {
            "dtype": "bfloat16",
            "trust_remote_code": True,
            "stage_1_tensor_parallel_size": 2,
        }
        create_structured.assert_called_once_with(
            "dummy-model",
            trust_remote_code=None,
            cli_overrides=expected_overrides,
            deploy_config_path="deploy.yaml",
        )
        create_legacy.assert_called_once_with(
            structured_config.pipeline_config,
            expected_overrides,
            "/resolved/deploy.yaml",
            strategy_specs=strategy_specs,
        )
        load_strategy.assert_called_once_with("strategy.yaml")
        assert resolved.config_path == "/resolved/deploy.yaml"
        assert resolved.pipeline_config is structured_config.pipeline_config
        assert resolved.omni_lb_policy == "round_robin"
        assert resolved.endpoint_restrictions == (endpoint_restriction,)
        assert resolved.stage_configs == (runtime_stage,)

    def test_registered_resolution_exposes_forced_aligner_topology(self, mocker: MockerFixture):
        pipeline = OMNI_PIPELINES["qwen3_tts"]
        assert isinstance(pipeline, PipelineConfig)
        structured_config = SimpleNamespace(
            orchestrator_config=SimpleNamespace(deploy_config_path=None),
            pipeline_config=pipeline,
        )
        mocker.patch(
            "vllm_omni.config.resolver.StageConfigFactory.create_from_model",
            return_value=structured_config,
        )

        resolved = resolve_omni_config(
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            trust_remote_code=False,
            deploy_config_path=None,
            cli_overrides={"forced_aligner": "/models/Qwen3-ForcedAligner-0.6B"},
            stage_overrides=None,
            strategy_config_path=None,
        )

        assert resolved.pipeline_config is not None
        pipeline_stage_ids = [stage.stage_id for stage in resolved.pipeline_config.stages]
        runtime_stage_ids = [stage.stage_id for stage in resolved.stage_configs]
        assert pipeline_stage_ids == runtime_stage_ids
        assert resolved.pipeline_config.stages[-1].model_stage == "forced_aligner"
        assert len(resolved.pipeline_config.stages) == len(pipeline.stages) + 1


class TestCumulativeStreamingCoercion:
    @pytest.mark.parametrize("skip_clone", [True, False])
    def test_cumulative_default_becomes_delta_if_stream(self, skip_clone):
        """Ensure cumulative messages are coercible to delta if streaming."""
        sp = SamplingParams(output_kind=RequestOutputKind.CUMULATIVE)
        sp.skip_clone = skip_clone
        result = coerce_param_message_types([sp], is_streaming=True)[0]
        assert isinstance(result, SamplingParams)
        assert result.output_kind == RequestOutputKind.DELTA
        assert (skip_clone and sp is result) or (not skip_clone and sp is not result)

    @pytest.mark.parametrize("skip_clone", [True, False])
    def test_cumulative_default_becomes_final_only_if_not_stream(self, skip_clone):
        """Ensure cumulative messages are coercible to final only if not streaming."""
        sp = SamplingParams(output_kind=RequestOutputKind.CUMULATIVE)
        sp.skip_clone = skip_clone
        result = coerce_param_message_types([sp], is_streaming=False)[0]
        assert isinstance(result, SamplingParams)
        assert result.output_kind == RequestOutputKind.FINAL_ONLY
        assert (skip_clone and sp is result) or (not skip_clone and sp is not result)

    @pytest.mark.parametrize("is_streaming", [True, False])
    @pytest.mark.parametrize("output_kind", [RequestOutputKind.DELTA, RequestOutputKind.FINAL_ONLY])
    def test_non_cumulative_are_coerced(self, output_kind, is_streaming):
        """Ensure non-cumulative params are coerced to the target type."""
        sp = SamplingParams(output_kind=output_kind)
        expected = RequestOutputKind.DELTA if is_streaming else RequestOutputKind.FINAL_ONLY
        result = coerce_param_message_types([sp], is_streaming=is_streaming)[0]
        assert isinstance(result, SamplingParams)
        assert result.output_kind == expected

    def test_coercion_applies_to_all_stages(self):
        """Ensure all stages are coerced to DELTA for streaming."""
        sp0 = SamplingParams(output_kind=RequestOutputKind.CUMULATIVE)
        sp1 = SamplingParams(output_kind=RequestOutputKind.CUMULATIVE)
        result = coerce_param_message_types([sp0, sp1], is_streaming=True)
        assert all([isinstance(r, SamplingParams) for r in result])
        assert result[0].output_kind == RequestOutputKind.DELTA
        assert result[1].output_kind == RequestOutputKind.DELTA
