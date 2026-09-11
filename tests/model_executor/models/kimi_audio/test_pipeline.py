# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Resolve the real deploy configuration and execute its admission/bridge hooks."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams

from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy
from vllm_omni.model_executor.models.kimi_audio.audio_processing import prepare_kimi_audio_inputs
from vllm_omni.model_executor.models.kimi_audio.kimi_audio import KimiAudioForConditionalGeneration
from vllm_omni.model_executor.models.kimi_audio.pipeline import KIMI_AUDIO_PIPELINE
from vllm_omni.model_executor.models.kimi_audio.prompt import KimiAudioPromptBuilder, KimiAudioSpecialTokens
from vllm_omni.model_executor.models.registry import _OMNI_MODELS

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
ROOT = Path(__file__).resolve().parents[4]
REFERENCE = json.loads((Path(__file__).parent / "fixtures/prompt_reference.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("output_type, final_stages", [("text", [0]), ("both", [0, 1])])
def test_deploy_admission_and_stage_bridge(output_type, final_stages):
    deploy = load_deploy_config(ROOT / "vllm_omni/deploy/kimi_audio.yaml")
    stages = merge_pipeline_deploy(KIMI_AUDIO_PIPELINE, deploy)
    builder = KimiAudioPromptBuilder(
        REFERENCE["text_tokens"].__getitem__,
        KimiAudioSpecialTokens.from_vocab(REFERENCE["special_tokens"]),
        **REFERENCE["input_config"],
    )
    prompt = prepare_kimi_audio_inputs(
        [{"role": "user", "message_type": "text", "content": "你好"}], builder, output_type=output_type
    )
    selected = [stage.stage_id for stage in stages if stage.final_output_type in prompt["modalities"]]
    assert selected == final_stages
    for stage in stages:
        package, module, cls = _OMNI_MODELS[stage.yaml_engine_args["model_arch"]]
        assert (
            getattr(importlib.import_module(f"vllm_omni.model_executor.models.{package}.{module}"), cls)
            is KimiAudioForConditionalGeneration
        )

    sampling = SamplingParams(**stages[0].yaml_extras["default_sampling_params"])
    path = stages[0].yaml_extras["prompt_transform_func"]
    module, name = path.rsplit(".", 1)
    admitted = getattr(importlib.import_module(module), name)(prompt, [sampling])
    assert admitted["model_intermediate_buffer"]["kimi_audio_request_validated"] is True
    assert sampling.stop_token_ids == [builder.tokens.msg_end]
    assert "kimi_audio" in sampling.extra_args

    if output_type == "both":
        module, name = stages[1].custom_process_input_func.rsplit(".", 1)
        bridge = getattr(importlib.import_module(module), name)
        source = SimpleNamespace(
            finished=True,
            outputs=[
                SimpleNamespace(
                    finish_reason="stop",
                    multimodal_output={
                        "codes": {"audio": torch.tensor([builder.audio_token_offset, builder.tokens.media_end])}
                    },
                )
            ],
        )
        assert bridge([source], prompt)[0]["prompt_token_ids"] == [0]
        assert stages[1].yaml_engine_args["skip_tokenizer_init"] is True


def test_pipeline_rejects_unimplemented_async_transport():
    deploy = load_deploy_config(ROOT / "vllm_omni/deploy/kimi_audio.yaml")
    deploy.async_chunk = True
    with pytest.raises(ValueError, match="async.chunk"):
        merge_pipeline_deploy(KIMI_AUDIO_PIPELINE, deploy)


def test_unified_entry_rejects_unknown_stage():
    config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(), model_stage="unknown"))
    with pytest.raises(ValueError, match="Unsupported Kimi-Audio model_stage"):
        KimiAudioForConditionalGeneration(vllm_config=config)
