# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU regression tests for MammothModa2's completed-AR to DiT bridge."""

from types import SimpleNamespace

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
from vllm_omni.model_executor.stage_input_processors.mammoth_moda2 import ar2dit

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


_PROMPT_TOKEN_IDS = [7, 8, 9]
_GENERATED_TOKEN_IDS = [101, 102, 103]


def _hidden_states(rows: int = 5, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return torch.arange(rows * 4, dtype=dtype).reshape(rows, 4)


def _ar_output(hidden_states: torch.Tensor | None = None) -> SimpleNamespace:
    multimodal_output = {} if hidden_states is None else {"latent": hidden_states}
    return SimpleNamespace(
        request_id="mammoth-request",
        prompt_token_ids=list(_PROMPT_TOKEN_IDS),
        outputs=[
            SimpleNamespace(
                cumulative_token_ids=list(_GENERATED_TOKEN_IDS),
                multimodal_output=multimodal_output,
            )
        ],
    )


def _prompt(
    *,
    target_h: int | None = None,
    target_w: int | None = None,
) -> dict[str, object]:
    mm_processor_kwargs: dict[str, int] = {}
    if target_h is not None:
        mm_processor_kwargs["target_h"] = target_h
    if target_w is not None:
        mm_processor_kwargs["target_w"] = target_w
    return {
        "additional_information": {
            "image_height": [384],
            "image_width": [640],
            "text_guidance_scale": [4.5],
            "cfg_range": [0.1, 0.9],
            "num_inference_steps": [20],
        },
        "mm_processor_kwargs": mm_processor_kwargs,
    }


def _assert_complete_dit_input(dit_inputs: list[dict[str, object]], hidden_states: torch.Tensor) -> None:
    assert len(dit_inputs) == 1
    dit_input = dit_inputs[0]
    assert dit_input["prompt_token_ids"] == [0]

    additional_information = dit_input["additional_information"]
    assert isinstance(additional_information, dict)
    full_hidden_states = additional_information["full_hidden_states"]
    assert isinstance(full_hidden_states, torch.Tensor)
    assert full_hidden_states.dtype == torch.bfloat16
    assert full_hidden_states.is_contiguous()
    assert torch.equal(full_hidden_states, hidden_states)
    assert additional_information["full_token_ids"] == [7, 8, 9, 101, 102]
    assert additional_information["answer_start_index"] == [3]
    assert additional_information["image_height"] == [384]
    assert additional_information["image_width"] == [640]
    assert additional_information["text_guidance_scale"] == [4.5]
    assert additional_information["cfg_range"] == [0.1, 0.9]
    assert additional_information["num_inference_steps"] == [20]


def test_ar2dit_builds_one_complete_dit_input() -> None:
    hidden_states = _hidden_states()

    dit_inputs = ar2dit([_ar_output(hidden_states)], _prompt())

    _assert_complete_dit_input(dit_inputs, hidden_states)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_ar2dit_preserves_low_precision_through_engine_core_payload(dtype: torch.dtype) -> None:
    hidden_states = _hidden_states(dtype=dtype)
    dit_input = ar2dit([_ar_output(hidden_states)], _prompt())[0]

    wire_payload = serialize_additional_information(dit_input["additional_information"])
    assert wire_payload is not None
    restored = deserialize_additional_information(wire_payload)
    restored_hidden_states = restored["full_hidden_states"]

    assert isinstance(restored_hidden_states, torch.Tensor)
    assert restored_hidden_states.dtype == dtype
    assert restored_hidden_states.is_contiguous()
    assert torch.equal(restored_hidden_states, hidden_states)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_dit_condition_split_preserves_compact_transfer_dtype(dtype: torch.dtype) -> None:
    # Construct only the config read by _split_ar_conditions.  This keeps the
    # test CPU-only and avoids loading the DiT/VAE weights.
    pipeline = object.__new__(MammothModa2DiTPipeline)
    object.__setattr__(
        pipeline,
        "config",
        SimpleNamespace(
            llm_config=SimpleNamespace(gen_vocab_start_index=100),
            image_token_id=20,
            video_token_id=21,
            vision_start_token_id=22,
            vision_end_token_id=23,
        ),
    )
    hidden_states = _hidden_states(dtype=dtype)

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


def test_ar2dit_prefers_processor_image_dimensions() -> None:
    hidden_states = _hidden_states()

    dit_inputs = ar2dit([_ar_output(hidden_states)], _prompt(target_h=512, target_w=768))

    additional_information = dit_inputs[0]["additional_information"]
    assert additional_information["image_height"] == [512]
    assert additional_information["image_width"] == [768]


def test_ar2dit_rejects_missing_latent_output() -> None:
    with pytest.raises(ValueError, match="missing latent multimodal output"):
        ar2dit([_ar_output()], _prompt())


def test_ar2dit_rejects_hidden_token_row_mismatch() -> None:
    with pytest.raises(AssertionError, match="Hidden states length mismatch"):
        ar2dit([_ar_output(_hidden_states(rows=4))], _prompt())


def test_mammoth_pipeline_uses_standard_completed_ar_forwarding() -> None:
    stage0, stage1 = MAMMOTH_MODA2_PIPELINE.stages

    assert stage0.custom_process_next_stage_input_func is None
    assert stage1.custom_process_input_func.endswith(".ar2dit")
    assert stage1.sync_process_input_func is None
    assert stage1.requires_full_payload_input is False


def test_stage_client_forwards_completed_ar_output_to_mammoth_adapter() -> None:
    hidden_states = _hidden_states()
    client = object.__new__(StageEngineCoreClient)
    client.custom_process_input_func = ar2dit
    client.requires_multimodal_data = False

    dit_inputs = client.process_engine_inputs(
        [_ar_output(hidden_states)],
        _prompt(),
    )

    _assert_complete_dit_input(dit_inputs, hidden_states)
