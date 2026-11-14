# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.models.diffusers_adapter import DiffusersAdapterPipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def adapter():
    config = OmniDiffusionConfig(
        model="test/model",
        model_class_name="DiffusersAdapterPipeline",
        dtype=torch.float16,
        diffusion_load_format="diffusers",
        parallel_config=DiffusionParallelConfig(cfg_parallel_size=1, sequence_parallel_size=1),
        cache_backend="none",
    )
    return DiffusersAdapterPipeline(od_config=config, device=torch.device("cpu"))


class StructureOnlyPipeline:
    def __call__(self, prompt, num_inference_steps=1):
        raise AssertionError("Attachment must not execute the pipeline")


@pytest.mark.parametrize("slicing,tiling", [(False, False), (True, False), (False, True), (True, True)])
def test_attach_initializes_adapter_without_loading_or_moving_weights(adapter, mocker, slicing, tiling):
    adapter.od_config.vae_use_slicing = slicing
    adapter.od_config.vae_use_tiling = tiling
    pipeline = StructureOnlyPipeline()
    enable_slicing = mocker.patch.object(pipeline, "enable_vae_slicing", create=True)
    enable_tiling = mocker.patch.object(pipeline, "enable_vae_tiling", create=True)
    forbidden = [
        mocker.patch.object(pipeline, name, create=True, side_effect=AssertionError(name))
        for name in ("to", "from_pretrained", "enable_model_cpu_offload", "enable_sequential_cpu_offload")
    ]
    load = mocker.patch.object(adapter, "load_weights", side_effect=AssertionError("load_weights"))
    pretrained = mocker.patch.object(
        adapter, "_load_pipeline_from_pretrained", side_effect=AssertionError("pretrained")
    )
    utils = mocker.Mock()
    get_utils = mocker.patch(
        "vllm_omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter.get_pipeline_utils_for_config",
        return_value=utils,
    )
    attention = mocker.patch.object(adapter, "_set_attention_backend")

    adapter.attach_structure_only_pipeline(pipeline)

    assert adapter._pipeline is pipeline
    assert adapter._pipeline_utils is utils
    assert adapter._accept_call_kwargs == {"prompt", "num_inference_steps"}
    get_utils.assert_called_once_with(adapter.od_config)
    utils.apply_post_load_updates.assert_called_once_with(pipeline, adapter.od_config)
    assert enable_slicing.call_count == int(slicing)
    assert enable_tiling.call_count == int(tiling)
    attention.assert_called_once_with()
    for method in [*forbidden, load, pretrained]:
        method.assert_not_called()


@pytest.mark.parametrize(
    "method,option", [("enable_vae_slicing", "vae_use_slicing"), ("enable_vae_tiling", "vae_use_tiling")]
)
def test_attach_optional_vae_failure_keeps_attention_setup(adapter, mocker, caplog, method, option):
    setattr(adapter.od_config, option, True)
    pipeline = StructureOnlyPipeline()
    mocker.patch.object(pipeline, method, create=True, side_effect=NotImplementedError("unsupported"))
    attention = mocker.patch.object(adapter, "_set_attention_backend")

    adapter.attach_structure_only_pipeline(pipeline)

    assert "unsupported" in caplog.text
    attention.assert_called_once_with()
