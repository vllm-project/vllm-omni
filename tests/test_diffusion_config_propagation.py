# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests that parallel_config survives the create_default_diffusion roundtrip.

Regression tests for https://github.com/vllm-project/vllm-omni/issues/1862
"""

from collections.abc import Mapping

import pytest
import torch

from vllm_omni.config.stage_config import StageConfigFactory
from vllm_omni.diffusion.data import (
    DiffusionParallelConfig,
    OmniDiffusionConfig,
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

    def test_no_parallel_config_defaults_to_tp1(self):
        od = _roundtrip_diffusion_config(model="x")
        assert od.parallel_config.tensor_parallel_size == 1
        assert od.parallel_config.world_size == 1

    def test_num_gpus_derived_from_world_size(self):
        pc = DiffusionParallelConfig(tensor_parallel_size=2)
        od = _roundtrip_diffusion_config(model="x", parallel_config=pc)
        assert od.num_gpus == 2


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

    assert od_config.max_multimodal_image_inputs == QWEN_IMAGE_EDIT_PLUS_MAX_INPUT_IMAGES
    assert od_config.max_multimodal_image_inputs > 1
