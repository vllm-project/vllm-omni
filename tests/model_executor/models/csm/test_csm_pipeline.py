# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the CSM pipeline declaration, its registry entries and its deploy config."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from transformers import CsmConfig
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.config import get_hf_text_config
from vllm.transformers_utils.model_arch_config_convertor import ModelArchConfigConvertorBase

from tests.model_executor.models.csm.test_csm_talker import _build_talker, _config
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.config.stage_config import StageExecutionType, load_deploy_config, merge_pipeline_deploy
from vllm_omni.model_executor.models.csm.pipeline import CSM_PIPELINE
from vllm_omni.model_executor.models.registry import _OMNI_MODELS
from vllm_omni.model_executor.stage_input_processors.csm import talker2code2wav_async_chunk

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]

DEPLOY_DIR = Path(__file__).parents[4] / "vllm_omni" / "deploy"


@pytest.fixture(scope="module")
def deploy():
    return load_deploy_config(DEPLOY_DIR / CSM_PIPELINE.default_deploy_config_name)


def test_pipeline_is_registered_under_the_checkpoint_model_type() -> None:
    assert OMNI_PIPELINES[CsmConfig.model_type] is CSM_PIPELINE
    assert CSM_PIPELINE.model_type == "csm"
    assert [stage.model_stage for stage in CSM_PIPELINE.stages] == ["csm_talker", "csm_code2wav"]
    assert (DEPLOY_DIR / CSM_PIPELINE.default_deploy_config_name).is_file()


def test_stage_architectures_resolve_to_the_csm_classes() -> None:
    for stage, class_name in zip(CSM_PIPELINE.stages, ["CsmTalkerForConditionalGeneration", "CsmCode2Wav"]):
        folder, module_name, registered = _OMNI_MODELS[stage.model_arch or CSM_PIPELINE.model_arch]
        module = importlib.import_module(f"vllm_omni.model_executor.models.{folder}.{module_name}")
        assert registered == class_name
        assert getattr(module, registered).__name__ == class_name


def test_talker_stage_declarations(monkeypatch: pytest.MonkeyPatch) -> None:
    talker_stage = CSM_PIPELINE.stages[0]
    assert talker_stage.execution_type is StageExecutionType.LLM_AR
    assert talker_stage.owns_tokenizer
    assert talker_stage.supports_native_mrv2_data_plane
    # A replayed prompt cannot rebuild the frames behind the generated tokens.
    assert talker_stage.recompute_preemption == "fail"
    assert talker_stage.sampling_constraints["detokenize"] is False
    assert "min_tokens" not in talker_stage.sampling_constraints

    # The pipeline cannot read the checkpoint config, so its literal is tied to the config default here.
    talker = _build_talker(_config(), monkeypatch)
    assert talker_stage.sampling_constraints["stop_token_ids"] == [talker.stop_token_id]
    assert talker_stage.sampling_constraints["stop_token_ids"] == [CsmConfig().codebook_pad_token_id]


def test_codec_stage_declarations() -> None:
    codec_stage = CSM_PIPELINE.stages[1]
    assert codec_stage.execution_type is StageExecutionType.LLM_GENERATION
    assert codec_stage.input_sources == (0,)
    assert codec_stage.final_output and codec_stage.final_output_type == "audio"
    assert codec_stage.supports_native_mrv2_data_plane
    # Requests waiting for their next chunk keep their decoder state.
    assert codec_stage.retains_state_across_chunks


def test_stages_are_connected_by_chunks_only() -> None:
    talker_stage, codec_stage = CSM_PIPELINE.stages
    module_name, _, function_name = talker_stage.async_chunk_process_next_stage_input_func.rpartition(".")
    assert getattr(importlib.import_module(module_name), function_name) is talker2code2wav_async_chunk
    assert talker_stage.custom_process_next_stage_input_func is None
    assert codec_stage.sync_process_input_func is None
    assert not codec_stage.requires_full_payload_input


def test_deploy_config_matches_the_design_decisions(deploy) -> None:
    talker, codec = deploy.stages
    assert deploy.async_chunk is True
    assert talker.engine_extras["enable_prefix_caching"] is False
    assert talker.async_scheduling is True
    assert talker.max_model_len == CsmConfig().max_position_embeddings
    assert talker.default_sampling_params == {"temperature": 0.9, "top_k": 50, "max_tokens": 125}
    assert talker.subtalker_sampling_params == {"do_sample": True, "temperature": 0.9, "top_k": 50, "top_p": 1.0}
    # The talker replays CUDA graphs: backbone decode steps, and the frame step the model captures itself.
    assert talker.enforce_eager is False and talker.compilation_config == {"cudagraph_mode": "FULL_DECODE_ONLY"}
    assert codec.enforce_eager is True
    # A waiting request keeps its codec slot, so the codec stage needs one per running talker request.
    assert codec.max_num_seqs >= talker.max_num_seqs
    # The longest request is 32 codes for every position the talker can generate.
    assert codec.max_model_len >= CsmConfig().num_codebooks * talker.max_model_len

    (connector,) = deploy.connectors.values()
    assert connector["extra"]["codec_left_context_frames"] == 0
    assert connector["extra"]["initial_codec_chunk_frames"] <= connector["extra"]["codec_chunk_frames"]


def test_merged_stage_arguments(deploy) -> None:
    talker, codec = merge_pipeline_deploy(CSM_PIPELINE, deploy)

    assert talker.yaml_engine_args["model_arch"] == "CsmTalkerForConditionalGeneration"
    assert talker.yaml_engine_args["async_chunk"] is True
    assert talker.yaml_engine_args["use_v2_model_runner"] is True
    assert talker.yaml_engine_args["recompute_preemption"] == "fail"
    assert talker.yaml_engine_args["subtalker_sampling_params"]["top_k"] == 50
    assert talker.yaml_engine_args["custom_process_next_stage_input_func"].endswith("csm.talker2code2wav_async_chunk")
    sampling = talker.yaml_extras["default_sampling_params"]
    assert sampling["stop_token_ids"] == [2050] and sampling["detokenize"] is False
    assert "min_tokens" not in sampling

    assert codec.yaml_engine_args["model_arch"] == "CsmCode2Wav"
    assert codec.yaml_engine_args["retains_state_across_chunks"] is True
    assert codec.yaml_engine_args["async_chunk"] is True


def test_stage_sampling_params_pass_the_engine_stop_token_check(deploy) -> None:
    """vLLM rejects every request of a stage whose stop token ids lie outside the model's vocabulary."""
    talker, _ = merge_pipeline_deploy(CSM_PIPELINE, deploy)
    params = SamplingParams(**talker.yaml_extras["default_sampling_params"])
    # What ModelConfig.get_vocab_size() resolves to for this model type, without a model directory.
    hf_config = CsmConfig()
    convertor = ModelArchConfigConvertorBase(hf_config, get_hf_text_config(hf_config))
    model_config = SimpleNamespace(get_vocab_size=convertor.get_vocab_size)

    params._validate_stop_token_ids(model_config)
    one_past = SamplingParams(stop_token_ids=[convertor.get_vocab_size()])
    with pytest.raises(VLLMValidationError, match="out-of-vocab"):
        one_past._validate_stop_token_ids(model_config)
