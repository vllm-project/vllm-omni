# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pipeline boundary tests using real tiny weights; requires the vLLM Python runtime."""

import json
from dataclasses import asdict

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.diffusion.models.lingbot_vla_v2.modeling_lingbot_vla_v2 import LingBotVLAV2ForActionPrediction

from .conftest import observation, tiny_backbone, tiny_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("model_class_name", [None, "LingBotVLAV2Pipeline"])
def test_lerobot_checkpoint_discovery(tmp_path, model_class_name):
    pytest.importorskip("vllm")
    from vllm_omni.diffusion.data import OmniDiffusionConfig, resolve_model_class_name

    (tmp_path / "config.json").write_text(json.dumps({"type": "lingbot_vla_v2"}))
    assert resolve_model_class_name(str(tmp_path)) == "LingBotVLAV2Pipeline"
    config = OmniDiffusionConfig(model=str(tmp_path), model_class_name=model_class_name)
    config.enrich_config()
    assert config.model_class_name == "LingBotVLAV2Pipeline"


def test_registered_pipeline_contract():
    pytest.importorskip("vllm")
    from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
    from vllm_omni.diffusion.registry import (
        _DIFFUSION_MODELS,
        _DIFFUSION_POST_PROCESS_FUNCS,
        _NO_CACHE_ACCELERATION,
    )

    assert _DIFFUSION_MODELS["LingBotVLAV2Pipeline"] == (
        "lingbot_vla_v2",
        "pipeline_lingbot_vla_v2",
        "LingBotVLAV2Pipeline",
    )
    assert _DIFFUSION_POST_PROCESS_FUNCS["LingBotVLAV2Pipeline"] == "get_lingbot_vla_v2_post_process_func"
    assert "LingBotVLAV2Pipeline" in _NO_CACHE_ACCELERATION
    topology = OMNI_PIPELINES["lingbot_vla_v2"]
    assert topology.model_arch == "LingBotVLAV2Pipeline"
    assert len(topology.stages) == 1 and topology.stages[0].final_output_type == "action"


def test_pipeline_checkpoint_to_actions(assets, model, monkeypatch):
    pytest.importorskip("vllm")
    from vllm.config.load import LoadConfig

    from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
    from vllm_omni.diffusion.models.lingbot_vla_v2.pipeline_lingbot_vla_v2 import LingBotVLAV2Pipeline
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    cfg = tiny_config()
    backbone = assets / "backbone"
    tiny_backbone().save_pretrained(backbone)
    cfg.tokenizer_path = str(backbone)
    (assets / "config.json").write_text(json.dumps(asdict(cfg)))
    save_file({"model." + k: v.clone() for k, v in model.state_dict().items()}, assets / "model.safetensors")
    monkeypatch.setattr("vllm_omni.diffusion.distributed.utils.get_local_device", lambda: torch.device("cpu"))
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    runtime = OmniDiffusionConfig(model=str(assets), dtype=torch.float32, num_gpus=1)
    with torch.inference_mode():
        pipeline = LingBotVLAV2Pipeline(od_config=runtime)
    assert not next(pipeline.model.parameters()).is_inference()
    loader = DiffusersPipelineLoader(LoadConfig(), runtime)
    loader.counter_before_loading_weights = 0.0
    loader.load_weights(pipeline)
    params = OmniDiffusionSamplingParams(num_inference_steps=2, extra_args={"robot_obs": observation()})
    params.generator = torch.Generator().manual_seed(18)
    request = OmniDiffusionRequest(prompt="pick up the cube", sampling_params=params, request_id="real")
    result = pipeline(DiffusionRequestBatch(requests=[request]))
    assert result.output["actions"].shape == (4, 7)


@pytest.mark.parametrize(
    ("streamed_weights", "unreported_names"),
    [((("model.state_proj.weight", torch.zeros(1)),), set()), ((), {"model.state_proj.weight"})],
    ids=["streamed", "unreported"],
)
def test_loader_rejects_streamed_or_unreported_weights(
    model: LingBotVLAV2ForActionPrediction,
    monkeypatch: pytest.MonkeyPatch,
    streamed_weights: tuple[tuple[str, torch.Tensor], ...],
    unreported_names: set[str],
) -> None:
    pytest.importorskip("vllm")
    from vllm.config.load import LoadConfig

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
    from vllm_omni.diffusion.models.lingbot_vla_v2.pipeline_lingbot_vla_v2 import LingBotVLAV2Pipeline

    pipeline = LingBotVLAV2Pipeline.__new__(LingBotVLAV2Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.model = model
    pipeline.loaded_weight_names = {f"model.{name}" for name in model.state_dict()} - unreported_names
    loader = DiffusersPipelineLoader(LoadConfig(), OmniDiffusionConfig(model="", dtype=torch.float32))
    loader.counter_before_loading_weights = 0.0
    monkeypatch.setattr(loader, "get_all_weights", lambda pipeline: iter(streamed_weights))

    with pytest.raises(ValueError, match="model.state_proj.weight"):
        loader.load_weights(pipeline)


def test_missing_observation_is_not_a_one_step_warmup():
    pytest.importorskip("vllm")
    from vllm_omni.diffusion.models.lingbot_vla_v2.pipeline_lingbot_vla_v2 import LingBotVLAV2Pipeline
    from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID, OmniDiffusionRequest
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    pipeline = LingBotVLAV2Pipeline.__new__(LingBotVLAV2Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.config = tiny_config()
    params = OmniDiffusionSamplingParams(num_inference_steps=1)
    request = OmniDiffusionRequest(prompt="dummy run", sampling_params=params, request_id="real-client")
    with pytest.raises(ValueError, match="robot_obs"):
        pipeline(DiffusionRequestBatch(requests=[request]))
    dummy = OmniDiffusionRequest(prompt="dummy run", sampling_params=params, request_id=DUMMY_DIFFUSION_REQUEST_ID)
    assert pipeline(DiffusionRequestBatch(requests=[dummy])).output["actions"].shape == (4, 7)
    with pytest.raises(ValueError, match="one observation"):
        pipeline(DiffusionRequestBatch(requests=[dummy, request]))
