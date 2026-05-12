"""Unit tests for vllm_omni.entrypoints.utils module."""

import os
from collections import Counter
from dataclasses import dataclass

import pytest
import torch
from pytest_mock import MockerFixture
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.config.yaml_util import create_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.entrypoints.utils import (
    _convert_dataclasses_to_dict,
    _filter_dict_like_object,
    coerce_param_message_types,
    filter_dataclass_kwargs,
    filter_stages,
    load_and_resolve_stage_configs,
    load_stage_configs_from_yaml,
    resolve_model_config_path,
)

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

        mocker.patch("vllm_omni.entrypoints.utils.logger")
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

        mocker.patch("vllm_omni.entrypoints.utils.logger")
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

        mocker.patch("vllm_omni.entrypoints.utils.logger")
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

    def test_filters_omni_engine_args_unknown_fields(self):
        """Test that OmniEngineArgs kwargs are filtered to valid fields only."""
        kwargs = {
            "model": "dummy",
            "stage_id": 1,
            "engine_output_type": "image",
            "unknown_field": "drop_me",
        }

        result = filter_dataclass_kwargs(OmniEngineArgs, kwargs)

        assert "model" in result
        assert "stage_id" in result
        assert "engine_output_type" in result
        assert "unknown_field" not in result

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

    def test_voxcpm_transformers_format_resolution(self, mocker: MockerFixture):
        """Test VoxCPM transformers config resolves to the voxcpm stage config."""
        mocker.patch(
            "vllm_omni.entrypoints.utils.get_config",
            side_effect=ValueError("missing transformers config"),
        )
        mocker.patch(
            "vllm_omni.entrypoints.utils.file_or_path_exists",
            side_effect=lambda _model, filename, revision=None: filename == "config.json",
        )
        mocker.patch(
            "vllm_omni.entrypoints.utils.get_hf_file_to_dict",
            return_value={"model_type": "voxcpm"},
        )
        mocker.patch(
            "vllm_omni.entrypoints.utils.current_omni_platform.get_default_stage_config_path",
            return_value="vllm_omni/deploy",
        )

        original_exists = os.path.exists

        def mock_exists(path):
            if "voxcpm.yaml" in str(path):
                return True
            return original_exists(path)

        mocker.patch("os.path.exists", side_effect=mock_exists)

        result = resolve_model_config_path("OpenBMB/VoxCPM1.5")

        assert result is not None
        assert "voxcpm.yaml" in result


class TestLoadAndResolveStageConfigs:
    def test_load_and_resolve_with_kwargs(self):
        """Ensure that dtype survives default stage creation."""
        kwargs = {"dtype": torch.float32}
        config_path, stage_configs = load_and_resolve_stage_configs(
            model="black-forest-labs/FLUX.2-klein-4B",
            stage_configs_path=None,
            kwargs=kwargs,
            default_stage_cfg_factory=lambda: AsyncOmniEngine._create_default_diffusion_stage_cfg(kwargs),
        )
        assert config_path is None
        assert len(stage_configs) == 1
        assert "dtype" in stage_configs[0]["engine_args"]

    def test_stage_configs_path_promotes_new_deploy_yaml_without_expanding_replicas(
        self, tmp_path, mocker: MockerFixture
    ):
        deploy_path = tmp_path / "qwen3_multi.yaml"
        deploy_path.write_text(
            'stages:\n  - stage_id: 0\n    devices: "0"\n  - stage_id: 1\n    devices: "1,2,3"\n    num_replicas: 3\n',
            encoding="utf-8",
        )

        returned_stage_configs = [
            create_config({"stage_id": 0, "runtime": {"devices": "0"}, "engine_args": {"model": "dummy"}}),
            create_config(
                {
                    "stage_id": 1,
                    "runtime": {"devices": "1,2,3", "num_replicas": 3},
                    "engine_args": {"model": "dummy"},
                }
            ),
        ]
        load_stage_configs = mocker.patch(
            "vllm_omni.entrypoints.utils.load_stage_configs_from_model",
            return_value=returned_stage_configs,
        )

        config_path, stage_configs = load_and_resolve_stage_configs(
            model="dummy-model",
            stage_configs_path=str(deploy_path),
            kwargs={},
        )

        load_stage_configs.assert_called_once_with(
            "dummy-model",
            base_engine_args={},
            deploy_config_path=str(deploy_path),
            stage_overrides=None,
        )
        assert config_path == str(deploy_path)
        assert len(stage_configs) == 2
        assert stage_configs[1].runtime.num_replicas == 3
        assert stage_configs[1].runtime.devices == "1,2,3"

    def test_filter_stages_selects_mode_stages_without_mutating_stage_config(self, tmp_path):
        config_path = tmp_path / "deploy.yaml"
        config_path.write_text(
            """modes:
  - mode: text-to-text
    stages: [0]
  - mode: text-to-image
    stages: [0, 1]
""",
            encoding="utf-8",
        )
        stages = [
            create_config(
                {
                    "stage_id": 0,
                    "runtime": {"requires_multimodal_data": True},
                    "final_output": False,
                    "final_output_type": None,
                }
            ),
            create_config(
                {
                    "stage_id": 1,
                    "runtime": {"requires_multimodal_data": True},
                    "final_output": True,
                    "final_output_type": "image",
                }
            ),
        ]

        filtered = filter_stages(str(config_path), stages, {"mode": "text-to-text"})

        assert len(filtered) == 1
        assert filtered[0].stage_id == 0
        assert filtered[0].runtime.requires_multimodal_data is True
        assert filtered[0].final_output is False
        assert filtered[0].final_output_type is None


class TestLoadStageConfigsFromYaml:
    """Regression tests for stage-config loading and merging."""

    def test_deep_merges_stage_engine_args(self, mocker: MockerFixture):
        yaml_config = create_config(
            {
                "async_chunk": True,
                "stage_args": [
                    {
                        "stage_id": 0,
                        "runtime": {"device": 0},
                        "engine_args": {
                            "parallel_config": {"tensor_parallel_size": 4},
                        },
                    }
                ],
            }
        )
        mocker.patch(
            "vllm_omni.entrypoints.utils.load_yaml_config",
            return_value=yaml_config,
        )

        stages = load_stage_configs_from_yaml(
            "fake.yaml",
            base_engine_args={
                "parallel_config": {
                    "tensor_parallel_size": 1,
                    "pipeline_parallel_size": 2,
                },
                "model": "base-model",
            },
        )

        merged_engine_args = stages[0]["engine_args"]
        assert merged_engine_args["parallel_config"]["tensor_parallel_size"] == 4
        assert merged_engine_args["parallel_config"]["pipeline_parallel_size"] == 2
        assert merged_engine_args["model"] == "base-model"
        assert merged_engine_args["async_chunk"] is True

    def test_merges_nested_stage_engine_args(self, mocker: MockerFixture):
        yaml_config = create_config(
            {
                "stage_args": [
                    {
                        "stage_id": 0,
                        "engine_args": {
                            "nested": {"override": 2},
                        },
                    }
                ],
            }
        )
        mocker.patch(
            "vllm_omni.entrypoints.utils.load_yaml_config",
            return_value=yaml_config,
        )

        stages = load_stage_configs_from_yaml(
            "fake.yaml",
            base_engine_args={"nested": {"base": 1}},
        )

        assert stages[0]["engine_args"]["nested"]["base"] == 1
        assert stages[0]["engine_args"]["nested"]["override"] == 2


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
