# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from pathlib import Path

import pytest
import yaml
from vllm.utils.import_utils import resolve_obj_by_qualname

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.config.stage_config import (
    PipelineConfig,
    load_deploy_config,
    merge_pipeline_deploy,
)
from vllm_omni.model_executor.models.chatterbox.pipeline import CHATTERBOX_ORIGINAL_PIPELINE, CHATTERBOX_TURBO_PIPELINE
from vllm_omni.model_executor.stage_input_processors.chatterbox import t3_to_s3gen, t3_to_s3gen_async_chunk
from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(scope="module")
def deploy() -> dict:
    return yaml.safe_load(Path(get_deploy_config_path("chatterbox_turbo.yaml")).read_text())


def test_each_variant_has_its_own_pipeline() -> None:
    assert OMNI_PIPELINES["chatterbox_turbo"] is CHATTERBOX_TURBO_PIPELINE
    assert OMNI_PIPELINES["chatterbox"] is CHATTERBOX_ORIGINAL_PIPELINE


def test_a_config_less_checkout_is_matched_by_its_directory_name(tmp_path: Path) -> None:
    turbo, original = tmp_path / "chatterbox-turbo", tmp_path / "chatterbox"
    turbo.mkdir()
    original.mkdir()
    try:
        assert StageConfigFactory.try_infer_model_type(str(turbo), trust_remote_code=False) == "chatterbox_turbo"
        assert StageConfigFactory.try_infer_model_type(str(original), trust_remote_code=False) == "chatterbox"
    finally:
        StageConfigFactory.get_hf_config.cache_clear()
        StageConfigFactory.try_infer_model_type.cache_clear()


def test_stage_0_stops_on_the_stop_token_and_stage_1_holds_no_slot_between_chunks() -> None:
    talker, decoder = CHATTERBOX_TURBO_PIPELINE.stages
    assert talker.sampling_constraints["stop_token_ids"] == [ChatterboxConfig().stop_speech_token]
    # On Model Runner V2 both stages use its native data plane; without the
    # declaration a stage falls back to the scheduler-side transport.
    assert talker.supports_native_mrv2_data_plane and decoder.supports_native_mrv2_data_plane
    # Read by the generation scheduler from this field. False: a stream holds
    # a decoder slot for the step that decodes its chunk, not for its
    # lifetime, so more streams than slots can be live at once.
    assert decoder.retains_state_across_chunks is False


@pytest.mark.parametrize(
    ("filename", "pipeline"),
    [("chatterbox_turbo.yaml", CHATTERBOX_TURBO_PIPELINE), ("chatterbox.yaml", CHATTERBOX_ORIGINAL_PIPELINE)],
    ids=["turbo", "original"],
)
@pytest.mark.parametrize("async_chunk", [True, False], ids=["async_chunk", "whole utterance"])
def test_each_mode_gets_its_own_stage_handoff(filename: str, pipeline: PipelineConfig, async_chunk: bool) -> None:
    """Streaming hands stage 0's tokens over chunk by chunk, the other mode the finished utterance once.

    In each mode exactly one processor runs: the chunk processor in stage 0,
    or the whole-utterance one in front of stage 1. Both would send stage 1
    every utterance twice; neither would leave it waiting forever.
    """
    deploy = load_deploy_config(Path(get_deploy_config_path(filename)))
    assert deploy.async_chunk is True
    deploy.async_chunk = async_chunk

    talker, decoder = merge_pipeline_deploy(pipeline, deploy)

    assert [talker.yaml_engine_args["async_chunk"], decoder.yaml_engine_args["async_chunk"]] == [async_chunk] * 2
    handoff = (
        talker.yaml_engine_args.get("custom_process_next_stage_input_func"),
        decoder.custom_process_input_func,
    )
    if async_chunk:
        assert handoff[1] is None and resolve_obj_by_qualname(handoff[0]) is t3_to_s3gen_async_chunk
    else:
        assert handoff[0] is None and resolve_obj_by_qualname(handoff[1]) is t3_to_s3gen


def test_the_deploy_files_context_and_budgets_fit_the_models_needs(deploy: dict) -> None:
    config = ChatterboxConfig()
    extra = deploy["connectors"]["connector_of_shared_memory"]["extra"]
    talker, decoder = deploy["stages"]

    # The chunk processor reads the first three and stage 1 the fourth; none
    # has a default. The flow's lookahead is not among them: the processor
    # takes it from the model config, where the decoder takes it.
    assert set(extra) == {
        "codec_chunk_frames",
        "codec_max_chunk_frames",
        "codec_stream_scale_factor",
        "codec_left_context_frames",
    }
    # The output cap is the model's, which the API bounds requests by.
    assert talker["default_sampling_params"]["max_tokens"] == config.max_new_tokens
    # Stage 0's context leaves room for text once the speaker slot, the
    # longest reference, the start token and the output cap are taken.
    assert talker["max_model_len"] - (1 + config.cond_prompt_len + 1 + config.max_new_tokens) > 0
    # Stage 1 is handed the whole utterance so far, for every request of a
    # step; a budget short of that gives a new request part of its tokens.
    # A request may raise max_tokens up to what stage 0's context allows, so
    # that context, not the default cap, bounds an utterance. Stage 1 checks
    # its budget against its own max_model_len, which must therefore be stage 0's.
    assert decoder["max_model_len"] == talker["max_model_len"]
    # The prompt is placeholder ids; a prefix cache would match any two
    # requests of equal length regardless of voice or text.
    assert talker["enable_prefix_caching"] is False
    # HiFT is not half-precision safe.
    assert decoder["dtype"] == "float32"
    # Prompts are token ids and nothing is detokenized. A tokenizer would
    # also have vLLM look in the repo for the config.json it does not ship.
    assert talker["skip_tokenizer_init"] is True and decoder["skip_tokenizer_init"] is True
