# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Falcon Perception pipeline topology and registry wiring."""

import importlib
from pathlib import Path

import pytest
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

from vllm_omni.config.pipeline_registry import OMNI_PIPELINES, resolve_pipeline_config
from vllm_omni.config.stage_config import (
    StageExecutionType,
    load_deploy_config,
    merge_pipeline_deploy,
)
from vllm_omni.model_executor.models.falcon_perception.falcon_perception_thinker import (
    FalconPerceptionBackbone,
)
from vllm_omni.model_executor.models.falcon_perception.pipeline import (
    FALCON_PERCEPTION_PIPELINE,
)
from vllm_omni.model_executor.models.registry import _OMNI_MODELS

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DEPLOY_DIR = Path(__file__).resolve().parents[4] / "vllm_omni" / "deploy"


def test_pipelines_validate_and_are_registered():
    assert FALCON_PERCEPTION_PIPELINE.validate() == []
    assert OMNI_PIPELINES["falcon_perception"] is FALCON_PERCEPTION_PIPELINE
    assert resolve_pipeline_config("falcon_perception") is FALCON_PERCEPTION_PIPELINE


def test_two_stage_topology():
    stages = FALCON_PERCEPTION_PIPELINE.stages
    assert len(stages) == 2

    thinker, segmentation = stages
    assert thinker.model_stage == "thinker"
    assert thinker.execution_type is StageExecutionType.LLM_AR
    assert thinker.owns_tokenizer and thinker.requires_multimodal_data
    # "latent" so the hidden-state trajectory is shipped to the mask head;
    # the user-visible output is still text.
    assert thinker.engine_output_type == "latent"
    assert thinker.final_output_type == "text"

    assert segmentation.model_stage == "segmentation"
    # The mask head is not autoregressive — it runs once, after decode.
    assert segmentation.execution_type is StageExecutionType.LLM_GENERATION
    assert segmentation.input_sources == (0,)
    # AnyUp guides upsampling with the original pixels, so stage 1 needs the image.
    assert segmentation.requires_multimodal_data


def test_stage_one_performance_knobs_are_model_local():
    deploy = load_deploy_config(_DEPLOY_DIR / "falcon_perception.yaml")
    stages = merge_pipeline_deploy(FALCON_PERCEPTION_PIPELINE, deploy)

    assert stages[1].yaml_engine_args["hf_overrides"] == {"hr_cache_mb": 12288, "compile_anyup": True}
    assert "env" not in stages[1].yaml_runtime


def test_falcon_profile_is_non_eager():
    deploy = load_deploy_config(_DEPLOY_DIR / "falcon_perception.yaml")
    stages = merge_pipeline_deploy(FALCON_PERCEPTION_PIPELINE, deploy)

    assert all(stage.yaml_engine_args["enforce_eager"] is False for stage in stages)
    assert stages[0].yaml_engine_args["compilation_config"] == {"cudagraph_mode": "FULL_DECODE_ONLY"}
    assert stages[1].yaml_engine_args["compilation_config"] == {"cudagraph_mode": "NONE"}


def test_a100_profile_uses_measured_scheduler_limits():
    deploy = load_deploy_config(_DEPLOY_DIR / "falcon_perception.yaml")
    stages = merge_pipeline_deploy(FALCON_PERCEPTION_PIPELINE, deploy)

    assert [stage.yaml_engine_args["max_num_seqs"] for stage in stages] == [4, 4]
    assert stages[0].yaml_engine_args["max_num_batched_tokens"] == 16384
    assert stages[0].yaml_engine_args["gpu_memory_utilization"] == 0.66
    assert stages[1].yaml_engine_args["gpu_memory_utilization"] == 0.10
    assert stages[0].yaml_engine_args["enable_prefix_caching"] is False


def test_thinker_backbone_opts_into_torch_compile():
    assert TorchCompileWithNoGuardsWrapper in FalconPerceptionBackbone.__bases__


def test_stage_bridge_function_resolves():
    """The pipeline references the processor by dotted path; it must exist."""
    dotted = FALCON_PERCEPTION_PIPELINE.stages[1].sync_process_input_func
    assert dotted, "stage bridge hook is not configured"
    module_path, func_name = dotted.rsplit(".", 1)
    module = importlib.import_module(module_path)
    assert callable(getattr(module, func_name))


def test_no_bare_bridge_function_shadows_the_token_only_hook():
    """``_select_processor_funcs`` always prefers ``*_token_only`` in sync mode.

    A bare ``thinker2segmentation`` alongside it would silently never run, which
    the omni model guide calls out explicitly.
    """
    module = importlib.import_module("vllm_omni.model_executor.stage_input_processors.falcon_perception")
    assert not hasattr(module, "thinker2segmentation")


def test_greedy_sampling_and_both_reference_stop_tokens():
    constraints = FALCON_PERCEPTION_PIPELINE.stages[0].sampling_constraints
    # The reference stops on EOS *and* <|end_of_query|>; dropping the second
    # makes generation run to max_tokens.
    assert constraints["stop_token_ids"] == [11, 263]


def test_registry_maps_the_published_architecture_and_both_stages():
    # config.json ships architectures: ["FalconPerceptionForSegmentation"].
    assert "FalconPerceptionForSegmentation" in _OMNI_MODELS
    for arch in ("FalconPerceptionThinker", "FalconPerceptionSegmentation"):
        assert arch in _OMNI_MODELS
        folder, module, cls = _OMNI_MODELS[arch]
        assert folder == "falcon_perception"
        assert cls == arch


def test_no_single_stage_pipeline_is_registered():
    """Detection-only serving is deliberately absent.

    A single-stage Falcon pipeline would have ``final_stage_id == 0``, so the AR
    runner never collects ``postprocess`` output, the geometry feedback loop
    never fires, and the model re-emits ``<|size|>`` forever — it runs and emits
    plausible-looking garbage rather than failing. Re-adding one requires
    generalizing the shared payload gate first.
    """
    assert "falcon_perception_thinker_only" not in OMNI_PIPELINES
