# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import importlib
from pathlib import Path

import pytest
import yaml

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.config.stage_config import StageExecutionType
from vllm_omni.model_executor.models.chatterbox.pipeline import CHATTERBOX_TURBO_PIPELINE
from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(scope="module")
def deploy() -> dict:
    return yaml.safe_load(Path(get_deploy_config_path("chatterbox_turbo.yaml")).read_text())


def test_pipeline_is_registered_under_the_turbo_key_only_001() -> None:
    assert OMNI_PIPELINES["chatterbox_turbo"] is CHATTERBOX_TURBO_PIPELINE
    # A bare "chatterbox" key would also claim ResembleAI/chatterbox, the
    # Original model, through the basename fallback.
    assert "chatterbox" not in OMNI_PIPELINES


def test_a_config_less_checkout_is_matched_by_its_directory_name_001(tmp_path: Path) -> None:
    turbo, original = tmp_path / "chatterbox-turbo", tmp_path / "chatterbox"
    turbo.mkdir()
    original.mkdir()
    try:
        assert StageConfigFactory.try_infer_model_type(str(turbo), trust_remote_code=False) == "chatterbox_turbo"
        assert StageConfigFactory.try_infer_model_type(str(original), trust_remote_code=False) is None
    finally:
        StageConfigFactory.get_hf_config.cache_clear()
        StageConfigFactory.try_infer_model_type.cache_clear()


def test_stage_topology_001() -> None:
    config = ChatterboxConfig()
    talker, decoder = CHATTERBOX_TURBO_PIPELINE.stages
    assert CHATTERBOX_TURBO_PIPELINE.model_arch == "ChatterboxForConditionalGeneration"
    assert (talker.model_stage, talker.execution_type) == ("chatterbox_t3", StageExecutionType.LLM_AR)
    assert (decoder.model_stage, decoder.execution_type) == ("chatterbox_s3gen", StageExecutionType.LLM_GENERATION)
    assert talker.sampling_constraints == {"stop_token_ids": [config.stop_speech_token], "detokenize": False}
    assert decoder.final_output and decoder.final_output_type == "audio"
    for path in (talker.async_chunk_process_next_stage_input_func, decoder.sync_process_input_func):
        module, _, function = path.rpartition(".")
        assert callable(getattr(importlib.import_module(module), function))


def test_deploy_file_agrees_with_the_model_constants_001(deploy: dict) -> None:
    config = ChatterboxConfig()
    extra = deploy["connectors"]["connector_of_shared_memory"]["extra"]
    talker, decoder = deploy["stages"]

    assert deploy["pipeline"] == "chatterbox_turbo"
    assert deploy["async_chunk"] is True
    assert extra["codec_vocab_size"] == config.speech_token_limit
    assert extra["codec_pre_lookahead_frames"] == config.pre_lookahead_len
    assert talker["default_sampling_params"] == {
        "temperature": 0.8,
        "top_k": 1000,
        "top_p": 0.95,
        "repetition_penalty": 1.2,
        "max_tokens": config.max_new_tokens,
    }
    # What stage 0's context leaves for text once the speaker slot, the
    # longest reference, the start token and the output cap are taken. The
    # deploy file's comment states this number; longer text has less room
    # for output.
    assert talker["max_model_len"] - (1 + config.cond_prompt_len + 1 + config.max_new_tokens) == 671
    # Stage 1 is handed the whole utterance so far, for every request of a
    # step; a budget short of that gives a new request part of its tokens.
    # A request may raise max_tokens up to what stage 0's context allows, so
    # that context, not the default cap, bounds an utterance. Stage 1 checks
    # its budget against its own max_model_len, which must therefore be stage 0's.
    assert decoder["max_model_len"] == talker["max_model_len"]
    assert decoder["max_num_batched_tokens"] >= decoder["max_num_seqs"] * decoder["max_model_len"]
    # The prompt is placeholder ids; a prefix cache would match any two
    # requests of equal length regardless of voice or text.
    assert talker["enable_prefix_caching"] is False
    assert decoder["dtype"] == "float32"
    # Prompts are token ids and nothing is detokenized. A tokenizer would
    # also have vLLM look in the repo for the config.json it does not ship.
    assert talker["skip_tokenizer_init"] is True and decoder["skip_tokenizer_init"] is True
