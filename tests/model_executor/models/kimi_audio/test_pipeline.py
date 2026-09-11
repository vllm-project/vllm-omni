# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Deployment and code-transfer contracts; no engine or acoustic inference."""

import importlib
import json
from collections import defaultdict
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
@pytest.mark.parametrize("deploy_name", ["kimi_audio.yaml", "kimi_audio_async_chunk.yaml"])
def test_deploy_admission_and_stage_bridge(output_type, final_stages, deploy_name):
    deploy = load_deploy_config(ROOT / "vllm_omni/deploy" / deploy_name)
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
    sampling.seed = 42
    path = stages[0].yaml_extras["prompt_transform_func"]
    module, name = path.rsplit(".", 1)
    admitted = getattr(importlib.import_module(module), name)(prompt, [sampling])
    assert admitted["model_intermediate_buffer"]["kimi_audio_request_validated"] is True
    assert sampling.stop_token_ids == [builder.tokens.msg_end]
    assert "kimi_audio" in sampling.extra_args

    if output_type == "both" and not deploy.async_chunk:
        module, name = stages[1].custom_process_input_func.rsplit(".", 1)
        bridge = getattr(importlib.import_module(module), name)
        source = SimpleNamespace(
            finished=True,
            outputs=[
                SimpleNamespace(
                    finish_reason="stop",
                    multimodal_output={
                        "codes": {"audio": torch.tensor([builder.audio_token_offset + 7, builder.tokens.media_end])}
                    },
                )
            ],
        )
        converted = bridge([source], admitted)[0]
        assert converted["prompt_token_ids"] == [7]
        assert converted["model_intermediate_buffer"]["codes"]["audio"] == [7]
        assert converted["model_intermediate_buffer"]["meta"] == {"finished": True, "audio_seed": 42}
        source.outputs[0].multimodal_output["codes"]["audio"] = torch.tensor([builder.tokens.media_end])
        empty = bridge([source], admitted)[0]
        assert empty["prompt_token_ids"] == [0]  # Scheduler placeholder, not a synthesized code.
        assert empty["model_intermediate_buffer"]["codes"]["audio"] == []
        assert stages[1].yaml_engine_args["skip_tokenizer_init"] is True

    if output_type == "both" and deploy.async_chunk:
        path = stages[0].yaml_engine_args["custom_process_next_stage_input_func"]
        module, name = path.rsplit(".", 1)
        bridge = getattr(importlib.import_module(module), name)
        connector = stages[0].yaml_extras["output_connectors"]["to_stage_1"]
        assert connector == stages[1].yaml_extras["input_connectors"]["from_stage_0"]
        assert connector in deploy.connectors
        assert stages[1].custom_process_input_func is None
        assert stages[1].yaml_engine_args["retains_state_across_chunks"] is True

        # Only the transfer adapter's buffers are supplied, as in the Qwen3-TTS
        # component test. This does not simulate transport, scheduling or decode.
        transfer = SimpleNamespace(
            request_payload={},
            code_prompt_token_ids=defaultdict(list),
            record_send_failure=lambda request_id, reason: pytest.fail(reason),
        )
        request = SimpleNamespace(
            request_id="internal-request",
            external_req_id="request",
            model_intermediate_buffer=admitted["model_intermediate_buffer"],
            sampling_params=sampling,
            is_finished=lambda: False,
        )
        for code in range(60):
            chunk = bridge(
                transfer, {"codes": {"audio": torch.tensor([builder.audio_token_offset + code])}}, request
            )
            if code == 30:
                assert chunk.codes.audio.tolist() == [[i] for i in range(30)]
                assert not chunk.meta.stream_finished.item()
                assert chunk.meta.chunk_seq == 0
            else:
                assert chunk is None
        final = bridge(
            transfer, {"codes": {"audio": torch.tensor([builder.tokens.media_end])}}, request, is_finished=True
        )
        assert final.codes.audio.tolist() == [[i] for i in range(30, 60)]
        assert final.meta.stream_finished.item()
        assert final.meta.chunk_seq == 1
        assert final.meta.audio_seed == sampling.seed
        assert transfer.code_prompt_token_ids[request.external_req_id] == []
