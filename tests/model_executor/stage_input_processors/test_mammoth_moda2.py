# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU regressions for MammothModa2's completed-AR to diffusion bridge."""

from dataclasses import dataclass

import pytest
import torch

from vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit import (
    MammothModa2DiTPipeline,
)
from vllm_omni.engine.serialization import (
    deserialize_additional_information,
    serialize_additional_information,
)
from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClient
from vllm_omni.model_executor.models.mammoth_moda2.pipeline import MAMMOTH_MODA2_PIPELINE
from vllm_omni.model_executor.stage_input_processors.mammoth_moda2 import ar2diffusion

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _CompletionOutputStub:
    cumulative_token_ids: list[int]
    multimodal_output: dict[str, torch.Tensor]


@dataclass
class _AROutputStub:
    request_id: str
    prompt_token_ids: list[int]
    outputs: list[_CompletionOutputStub]


@dataclass
class _TextConfigStub:
    gen_vocab_start_index: int


@dataclass
class _MammothConfigStub:
    llm_config: _TextConfigStub
    image_token_id: int
    video_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int


def _source_output(*, include_latent: bool = True) -> _AROutputStub:
    multimodal_output = {"latent": torch.arange(32, dtype=torch.float32).reshape(4, 8)} if include_latent else {}
    completion = _CompletionOutputStub(
        cumulative_token_ids=[100, 101, 102],
        multimodal_output=multimodal_output,
    )
    return _AROutputStub(
        request_id="req-7",
        prompt_token_ids=[10, 11],
        outputs=[completion],
    )


def test_ar2diffusion_builds_one_prompt_with_raw_ar_conditions() -> None:
    result = ar2diffusion(
        [_source_output()],
        {"prompt": "a cat", "mm_processor_kwargs": {"target_h": 512, "target_w": 768}},
    )
    assert not isinstance(result, list)
    assert result["prompt"] == ""
    assert result["height"] == 512
    assert result["width"] == 768
    info = result["additional_information"]
    assert info["full_token_ids"] == [10, 11, 100, 101]
    assert info["answer_start_index"] == 2
    torch.testing.assert_close(
        info["full_hidden_states"],
        torch.arange(32, dtype=torch.float32).reshape(4, 8),
    )
    assert info["full_hidden_states"].is_contiguous()


def test_ar2diffusion_uses_prompt_dimension_fallbacks() -> None:
    result = ar2diffusion(
        [_source_output()],
        {"additional_information": {"image_height": [256], "image_width": [384]}},
    )
    assert (result["height"], result["width"]) == (256, 384)


def test_ar2diffusion_preserves_request_level_sampling_fallbacks() -> None:
    result = ar2diffusion(
        [_source_output()],
        {
            "additional_information": {
                "text_guidance_scale": [1.5],
                "num_inference_steps": [3],
                "cfg_range": [0.25, 0.75],
            }
        },
    )

    info = result["additional_information"]
    assert info["text_guidance_scale"] == [1.5]
    assert info["num_inference_steps"] == [3]
    assert info["cfg_range"] == [0.25, 0.75]


def test_ar2diffusion_unwraps_the_orchestrator_prompt_list() -> None:
    result = ar2diffusion(
        [_source_output()],
        [{"mm_processor_kwargs": {"target_h": 640, "target_w": 960}}],
    )
    assert (result["height"], result["width"]) == (640, 960)


def test_ar2diffusion_rejects_multiple_source_requests() -> None:
    with pytest.raises(ValueError, match="exactly one AR output"):
        ar2diffusion([_source_output(), _source_output()], {})


def test_ar2diffusion_reports_missing_latent_with_request_id() -> None:
    with pytest.raises(ValueError, match="req-7"):
        ar2diffusion([_source_output(include_latent=False)], {})


def test_ar2diffusion_rejects_hidden_state_length_mismatch() -> None:
    source = _source_output()
    source.outputs[0].multimodal_output["latent"] = torch.zeros(3, 8)
    with pytest.raises(ValueError, match="Hidden states length mismatch"):
        ar2diffusion([source], {})


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_ar2diffusion_preserves_low_precision_through_engine_core_payload(dtype: torch.dtype) -> None:
    source = _source_output()
    hidden_states = torch.arange(32, dtype=dtype).reshape(4, 8)
    source.outputs[0].multimodal_output["latent"] = hidden_states
    diffusion_input = ar2diffusion([source], {})

    wire_payload = serialize_additional_information(diffusion_input["additional_information"])
    assert wire_payload is not None
    restored = deserialize_additional_information(wire_payload)
    restored_hidden_states = restored["full_hidden_states"]

    assert isinstance(restored_hidden_states, torch.Tensor)
    assert restored_hidden_states.dtype == dtype
    assert restored_hidden_states.is_contiguous()
    assert torch.equal(restored_hidden_states, hidden_states)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_dit_condition_split_preserves_compact_transfer_dtype(dtype: torch.dtype) -> None:
    pipeline = object.__new__(MammothModa2DiTPipeline)
    object.__setattr__(
        pipeline,
        "config",
        _MammothConfigStub(
            llm_config=_TextConfigStub(gen_vocab_start_index=100),
            image_token_id=20,
            video_token_id=21,
            vision_start_token_id=22,
            vision_end_token_id=23,
        ),
    )
    hidden_states = torch.arange(20, dtype=dtype).reshape(5, 4)

    text_cond, image_cond = pipeline._split_ar_conditions(
        full_hidden_states=hidden_states,
        full_token_ids=[7, 20, 8, 101, 102],
        answer_start_index=3,
    )

    assert text_cond.dtype == dtype
    assert image_cond.dtype == dtype
    assert text_cond.is_contiguous()
    assert image_cond.is_contiguous()
    assert torch.equal(text_cond, hidden_states[[0, 2]])
    assert torch.equal(image_cond, hidden_states[[3, 4]])


def test_mammoth_pipeline_uses_standard_completed_ar_forwarding() -> None:
    stage0, stage1 = MAMMOTH_MODA2_PIPELINE.stages

    assert stage0.custom_process_next_stage_input_func is None
    assert stage1.custom_process_input_func.endswith(".ar2diffusion")
    assert stage1.sync_process_input_func is None
    assert stage1.requires_full_payload_input is False


def test_stage_client_forwards_completed_ar_output_to_mammoth_adapter() -> None:
    source = _source_output()
    client = object.__new__(StageEngineCoreClient)
    client.custom_process_input_func = ar2diffusion
    client.requires_multimodal_data = False

    diffusion_input = client.process_engine_inputs(
        [source],
        {"mm_processor_kwargs": {"target_h": 512, "target_w": 768}},
    )

    assert diffusion_input["height"] == 512
    assert diffusion_input["width"] == 768
    info = diffusion_input["additional_information"]
    assert info["full_token_ids"] == [10, 11, 100, 101]
    assert info["answer_start_index"] == 2
    assert torch.equal(info["full_hidden_states"], source.outputs[0].multimodal_output["latent"])
