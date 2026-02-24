"""Unit tests for vllm_omni.entrypoints.utils module."""

from collections import Counter
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.entrypoints.utils import (
    _convert_dataclasses_to_dict,
    _filter_dict_like_object,
    filter_dataclass_kwargs,
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

    def test_dict_with_recursive_structure(self):
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

        with patch("vllm_omni.entrypoints.utils.logger"):
            result = _filter_dict_like_object(input_dict)

            # Normal key should exist
            assert "normal" in result
            # Level1 should exist
            assert "level1" in result

    def test_integration_with_convert_dataclasses(self):
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

        with patch("vllm_omni.entrypoints.utils.logger"):
            result = _filter_dict_like_object(input_dict)

            # Callable should be filtered
            assert "func" not in result
            # Config should be converted to dict
            assert "config" in result
            assert "normal" in result


class TestConvertDataclassesToDict:
    """Test suite for _convert_dataclasses_to_dict function."""

    def test_uses_filter_dict_like_object(self):
        """Test that _convert_dataclasses_to_dict uses _filter_dict_like_object for dicts."""
        input_dict = {
            "normal": "value",
            "callable": lambda x: x,
        }

        with patch("vllm_omni.entrypoints.utils.logger"):
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
