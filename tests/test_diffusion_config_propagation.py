# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests that parallel_config survives the create_default_diffusion roundtrip.

Regression tests for https://github.com/vllm-project/vllm-omni/issues/1862
"""

from collections.abc import Mapping
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from vllm_omni.config.stage_config import StageConfigFactory
from vllm_omni.diffusion.data import (
    DiffusionParallelConfig,
    OmniDiffusionConfig,
    _nested_parallel_config_to_plain_dict,
    build_parallel_config_dict_from_engine_args,
)
from vllm_omni.diffusion.model_metadata import QWEN_IMAGE_EDIT_PLUS_MAX_INPUT_IMAGES

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _roundtrip_diffusion_config(**kwargs) -> OmniDiffusionConfig:
    """Simulate the real path: create_default_diffusion → OmniDiffusionConfig.

    Does NOT manually reconstruct parallel_config — relies on
    OmniDiffusionConfig.__post_init__ to handle the dict, just like
    the production code path does.
    """
    stages = StageConfigFactory.create_default_diffusion(kwargs)
    engine_args = dict(stages[0]["engine_args"])
    return OmniDiffusionConfig.from_kwargs(**engine_args)


class TestParallelConfigPropagation:
    """Core regression tests: parallel_config must survive serialization."""

    def test_tp2_roundtrip(self):
        pc = DiffusionParallelConfig(tensor_parallel_size=2)
        od = _roundtrip_diffusion_config(model="test-model", parallel_config=pc)
        assert od.parallel_config.tensor_parallel_size == 2
        assert od.parallel_config.world_size == 2

    def test_tp4_devices_and_config(self):
        pc = DiffusionParallelConfig(tensor_parallel_size=4)
        stages = StageConfigFactory.create_default_diffusion({"parallel_config": pc, "model": "x"})
        assert stages[0]["runtime"]["devices"] == "0,1,2,3"

        # Let __post_init__ reconstruct from dict (real code path)
        ea = dict(stages[0]["engine_args"])
        od = OmniDiffusionConfig.from_kwargs(**ea)
        assert od.parallel_config.tensor_parallel_size == 4
        assert od.parallel_config.world_size == 4

    def test_sp_config_roundtrip(self):
        pc = DiffusionParallelConfig(
            tensor_parallel_size=2,
            ulysses_degree=2,
            ring_degree=1,
        )
        od = _roundtrip_diffusion_config(model="x", parallel_config=pc)
        assert od.parallel_config.ulysses_degree == 2
        assert od.parallel_config.ring_degree == 1

    def test_cfg_parallel_roundtrip(self):
        pc = DiffusionParallelConfig(cfg_parallel_size=2)
        od = _roundtrip_diffusion_config(model="x", parallel_config=pc)
        assert od.parallel_config.cfg_parallel_size == 2
        assert od.parallel_config.world_size == 2

    def test_top_level_cfg_parallel_into_engine_args_dict(self):
        """Multi-stage paths pass ``cfg_parallel_size`` at engine_args top-level."""
        engine_args = {"cfg_parallel_size": 2}
        merged = build_parallel_config_dict_from_engine_args(engine_args)
        assert merged.get("cfg_parallel_size") == 2
        dpc = DiffusionParallelConfig.from_dict(merged)
        assert dpc.cfg_parallel_size == 2
        assert dpc.world_size == 2

    def test_no_parallel_config_defaults_to_tp1(self):
        od = _roundtrip_diffusion_config(model="x")
        assert od.parallel_config.tensor_parallel_size == 1
        assert od.parallel_config.world_size == 1

    def test_num_gpus_derived_from_world_size(self):
        pc = DiffusionParallelConfig(tensor_parallel_size=2)
        od = _roundtrip_diffusion_config(model="x", parallel_config=pc)
        assert od.num_gpus == 2


class TestParallelConfigHelpers:
    """Cover parallel merge helpers defined in ``vllm_omni/diffusion/data.py`` (CPU-only).

    These functions merge nested ``engine_args.parallel_config`` with overlapping top-level
    diffusion parallel keys before ``OmniDiffusionConfig.from_kwargs`` builds
    ``DiffusionParallelConfig``. **Multi-stage** flows are the sharp edge: YAML/OmegaConf often supplies
    a ``parallel_config`` subtree while CLI or stage merges add the same fields at ``engine_args`` top
    level—we assert coercion, YAML-null stripping, and nested-vs-top-level precedence.

    Symbols: ``_nested_parallel_config_to_plain_dict``, ``build_parallel_config_dict_from_engine_args``
    """

    def test_nested_parallel_config_dictconfig_to_plain_dict(self):
        """OmegaConf DictConfig → plain dict (deploy/YAML merged engine_args)."""
        cfg = OmegaConf.create({"tensor_parallel_size": 2, "pipeline_parallel_size": 1})
        out = _nested_parallel_config_to_plain_dict(cfg)
        assert isinstance(out, dict)
        assert type(out) is dict  # plain dict, not OmegaConf container
        assert out["tensor_parallel_size"] == 2
        assert out["pipeline_parallel_size"] == 1

    def test_nested_parallel_config_dataclass_like_asdict(self):
        pc = DiffusionParallelConfig(tensor_parallel_size=4)
        out = _nested_parallel_config_to_plain_dict(pc)
        assert isinstance(out, dict)
        assert out["tensor_parallel_size"] == 4

    def test_nested_parallel_config_simple_namespace(self):
        ns = SimpleNamespace(tensor_parallel_size=2, cfg_parallel_size=1)
        out = _nested_parallel_config_to_plain_dict(ns)
        assert out["tensor_parallel_size"] == 2
        assert out["cfg_parallel_size"] == 1

    def test_build_merge_top_level_when_missing_nested(self):
        """Top-level parallel CLI knobs fill absent keys inside merged parallel dict."""
        merged = build_parallel_config_dict_from_engine_args(
            {
                "parallel_config": {},
                "cfg_parallel_size": 2,
                "tensor_parallel_size": 2,
            }
        )
        assert merged["cfg_parallel_size"] == 2
        assert merged["tensor_parallel_size"] == 2

    @pytest.mark.parametrize(
        "nested_parallel",
        [
            pytest.param({"cfg_parallel_size": 1, "tensor_parallel_size": 1}, id="plain_dict"),
            pytest.param(OmegaConf.create({"cfg_parallel_size": 1, "tensor_parallel_size": 1}), id="dictconfig"),
        ],
    )
    def test_build_nested_yaml_wins_over_top_level_when_key_present(self, nested_parallel):
        """Nested deploy subtree wins for keys it defines; CLI only fills absent slots."""
        merged = build_parallel_config_dict_from_engine_args(
            {
                "parallel_config": nested_parallel,
                "cfg_parallel_size": 2,
                "tensor_parallel_size": 4,
            }
        )
        assert merged["cfg_parallel_size"] == 1
        assert merged["tensor_parallel_size"] == 1

    @pytest.mark.parametrize(
        ("field", "nested", "override", "expect"),
        [
            ("tensor_parallel_size", {"tensor_parallel_size": None}, 2, 2),
            ("cfg_parallel_size", {"cfg_parallel_size": None}, 2, 2),
        ],
    )
    def test_build_strip_none_so_top_level_applies(self, field, nested, override, expect):
        """YAML null for a declared parallel field drops the key so top-level knobs can populate it."""
        engine_args = {"parallel_config": nested, field: override}
        merged = build_parallel_config_dict_from_engine_args(engine_args)
        assert merged[field] == expect

    def test_build_dictconfig_nested_with_null_placeholder_plus_top_level(self):
        """Nested DictConfig with YAML nulls: strip Nones then apply top-level overrides."""
        nested = OmegaConf.create({"tensor_parallel_size": None, "cfg_parallel_size": None})
        merged = build_parallel_config_dict_from_engine_args(
            {
                "parallel_config": nested,
                "tensor_parallel_size": 2,
                "cfg_parallel_size": 2,
            }
        )
        assert merged["tensor_parallel_size"] == 2
        assert merged["cfg_parallel_size"] == 2


class TestCreateDefaultDiffusion:
    """Verify engine_args structure from create_default_diffusion."""

    def test_parallel_config_serialized_as_dict(self):
        """The key fix: parallel_config must appear in engine_args as a dict."""
        pc = DiffusionParallelConfig(tensor_parallel_size=2)
        stages = StageConfigFactory.create_default_diffusion({"model": "x", "parallel_config": pc})
        ea = stages[0]["engine_args"]
        assert "parallel_config" in ea
        assert isinstance(ea["parallel_config"], Mapping)
        assert ea["parallel_config"]["tensor_parallel_size"] == 2

    def test_dtype_serialized_as_string(self):
        stages = StageConfigFactory.create_default_diffusion({"dtype": torch.float16, "model": "x"})
        assert stages[0]["engine_args"]["dtype"] == "torch.float16"

    def test_cache_backend_defaults_to_none(self):
        stages = StageConfigFactory.create_default_diffusion({"model": "x"})
        assert stages[0]["engine_args"]["cache_backend"] == "none"

    def test_single_gpu_default_devices(self):
        stages = StageConfigFactory.create_default_diffusion({"model": "x"})
        assert stages[0]["runtime"]["devices"] == "0"

    def test_extra_kwargs_forwarded(self):
        stages = StageConfigFactory.create_default_diffusion(
            {"model": "x", "enforce_eager": True, "lora_path": "/tmp/lora"}
        )
        ea = stages[0]["engine_args"]
        assert ea["enforce_eager"] is True
        assert ea["lora_path"] == "/tmp/lora"


def test_qwen_image_edit_plus_sets_generic_multimodal_limit():
    od_config = OmniDiffusionConfig(model="Qwen/Qwen-Image-Edit-2511", model_class_name="QwenImageEditPlusPipeline")

    od_config.update_multimodal_support()

    assert od_config.supports_multimodal_inputs is True
    assert od_config.max_multimodal_image_inputs == QWEN_IMAGE_EDIT_PLUS_MAX_INPUT_IMAGES
