# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for the MammothModa2 AR -> DiT request-end full payload path."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.mammoth_moda2.pipeline import MAMMOTH_MODA2_PIPELINE
from vllm_omni.model_executor.stage_input_processors.mammoth_moda2 import (
    ar2dit,
    ar2dit_full_payload,
    ar2dit_token_only,
)
from vllm_omni.worker.omni_connector_model_runner_mixin import (
    OmniConnectorModelRunnerMixin,
    should_accumulate_full_payload_output,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _accumulate_hidden_slices(slices: list[torch.Tensor]) -> torch.Tensor:
    """Feed per-step hidden slices through the real full-payload accumulator.

    Mirrors the runner: one pooler payload per step for the same request id,
    concatenated at request end by ``_materialize_full_payload_entry``.
    """
    runner = object.__new__(OmniConnectorModelRunnerMixin)
    runner._custom_process_func = ar2dit_full_payload
    runner._pending_full_payload_send = {}
    request = SimpleNamespace(output_token_ids=[])
    for hidden_slice in slices:
        OmniConnectorModelRunnerMixin.accumulate_full_payload_output(runner, "r1", {"hidden": hidden_slice}, request)
    output, _ = OmniConnectorModelRunnerMixin._materialize_full_payload_entry(runner._pending_full_payload_send["r1"])
    return output["hidden"]


def test_ar2dit_full_payload_materializes_cpu_float_payload_once():
    """Device-resident slices -> accumulator -> one CPU/float32 payload at request end."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt_token_ids = [1, 2, 3]
    output_token_ids = [10, 11, 12, 13, 14, 15, 16]  # final look-ahead token has no hidden state
    prefill = torch.arange(18, dtype=torch.bfloat16, device=device).reshape(3, 6)
    decode = torch.arange(18, 54, dtype=torch.bfloat16, device=device).reshape(6, 6)

    # Prefill + decode slices accumulate on device (the accumulator concatenates
    # per-step chunks); the single materialization happens in ar2dit_full_payload
    # at request end.
    hidden = _accumulate_hidden_slices([prefill, decode])
    assert hidden.device.type == device
    assert hidden.shape == (len(prompt_token_ids) + len(output_token_ids) - 1, 6)

    request = SimpleNamespace(
        request_id="r1",
        prompt_token_ids=prompt_token_ids,
        output_token_ids=output_token_ids,
        additional_information_cpu={"image_height": [512], "image_width": [768]},
    )
    payload = ar2dit_full_payload(transfer_manager=None, pooling_output={"hidden": hidden}, request=request)

    assert payload is not None
    full_hidden = payload["full_hidden_states"]
    assert full_hidden.device.type == "cpu"
    assert full_hidden.dtype == torch.float32
    assert full_hidden.shape == (len(prompt_token_ids) + len(output_token_ids) - 1, 6)
    assert full_hidden.is_contiguous()
    assert torch.equal(full_hidden, torch.cat([prefill, decode], dim=0).float().cpu())

    assert payload["full_token_ids"] == prompt_token_ids + output_token_ids[:-1]
    assert payload["answer_start_index"] == [len(prompt_token_ids)]
    assert payload["image_height"] == [512]
    assert payload["image_width"] == [768]
    assert payload["text_guidance_scale"] == [9.0]
    assert payload["cfg_range"] == [0.0, 1.0]
    assert payload["num_inference_steps"] == [50]


def test_ar2dit_full_payload_matches_legacy_ar2dit_schema():
    """The request-end payload must match the legacy ar2dit bridge schema/values."""
    prompt_token_ids = [1, 2, 3]
    gen_token_ids = [10, 11, 12, 13]
    full_hidden = torch.arange(36, dtype=torch.bfloat16).reshape(6, 6).float()
    addi_info = {
        "image_height": [512],
        "image_width": [768],
        "text_guidance_scale": [4.0],
        "cfg_range": [0.0, 1.0],
        "num_inference_steps": [25],
    }

    legacy = ar2dit(
        [
            SimpleNamespace(
                prompt_token_ids=prompt_token_ids,
                outputs=[
                    SimpleNamespace(
                        cumulative_token_ids=gen_token_ids,
                        multimodal_output={"latent": full_hidden},
                    )
                ],
            )
        ],
        prompts=[{"additional_information": addi_info}],
    )[0]
    request_end = ar2dit_full_payload(
        transfer_manager=None,
        pooling_output={"hidden": full_hidden},
        request=SimpleNamespace(
            request_id="r1",
            prompt_token_ids=prompt_token_ids,
            output_token_ids=gen_token_ids,
            additional_information_cpu=addi_info,
        ),
    )

    assert request_end is not None
    assert set(request_end) == set(legacy["additional_information"])
    for key, value in legacy["additional_information"].items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(request_end[key], value)
        else:
            assert request_end[key] == value


def test_ar2dit_full_payload_rejects_hidden_token_id_mismatch():
    request = SimpleNamespace(
        request_id="r1",
        prompt_token_ids=[1, 2, 3],
        output_token_ids=[10, 11, 12, 13],  # expects 6 hidden rows
        additional_information_cpu=None,
    )
    with pytest.raises(ValueError, match="length mismatch"):
        ar2dit_full_payload(
            transfer_manager=None,
            pooling_output={"hidden": torch.zeros((5, 6))},
            request=request,
        )


def test_ar2dit_full_payload_synthesizes_t2i_ids_from_placeholder_outputs():
    prompt_token_ids = [1, 2, 3]
    ar_width = 2
    ar_height = 2
    generated_hidden_len = ar_height * (ar_width + 1)
    visual_start = 152072
    eol_token_id = 152064
    request = SimpleNamespace(
        request_id="r1",
        prompt_token_ids=prompt_token_ids,
        # The async AR path may still expose placeholders here at request end.
        # The final look-ahead token has no hidden state and is dropped.
        output_token_ids=[-1] * (generated_hidden_len + 1),
        additional_information_cpu={
            "omni_task": ["t2i"],
            "ar_width": [ar_width],
            "ar_height": [ar_height],
            "eol_token_id": [eol_token_id],
            "visual_token_start_id": [visual_start],
            "visual_token_end_id": [168456],
        },
    )

    payload = ar2dit_full_payload(
        transfer_manager=None,
        pooling_output={"hidden": torch.zeros((len(prompt_token_ids) + generated_hidden_len, 6))},
        request=request,
    )

    assert payload is not None
    assert payload["full_token_ids"] == prompt_token_ids + [
        visual_start,
        visual_start,
        eol_token_id,
        visual_start,
        visual_start,
        eol_token_id,
    ]


def test_ar2dit_full_payload_fills_partial_t2i_placeholders():
    visual_start = 152072
    eol_token_id = 152064
    request = SimpleNamespace(
        request_id="r1",
        prompt_token_ids=[1, 2, 3],
        output_token_ids=[152100, -1, eol_token_id, -1, 152101, -1, 168456],
        additional_information_cpu={
            "omni_task": ["t2i"],
            "ar_width": [2],
            "ar_height": [2],
            "eol_token_id": [eol_token_id],
            "visual_token_start_id": [visual_start],
        },
    )

    payload = ar2dit_full_payload(
        transfer_manager=None,
        pooling_output={"hidden": torch.zeros((9, 6))},
        request=request,
    )

    assert payload is not None
    assert payload["full_token_ids"] == [
        1,
        2,
        3,
        152100,
        visual_start,
        eol_token_id,
        visual_start,
        152101,
        eol_token_id,
    ]


def test_ar2dit_full_payload_rejects_t2i_grid_hidden_mismatch():
    request = SimpleNamespace(
        request_id="r1",
        prompt_token_ids=[1, 2, 3],
        output_token_ids=[-1] * 7,
        additional_information_cpu={
            "omni_task": ["t2i"],
            "ar_width": [2],
            "ar_height": [3],
            "eol_token_id": [152064],
            "visual_token_start_id": [152072],
        },
    )

    with pytest.raises(ValueError, match="expected 9 from AR grid 2x3, got 6"):
        ar2dit_full_payload(
            transfer_manager=None,
            pooling_output={"hidden": torch.zeros((9, 6))},
            request=request,
        )


def test_ar2dit_full_payload_does_not_synthesize_non_t2i_placeholders():
    request = SimpleNamespace(
        request_id="r1",
        prompt_token_ids=[1, 2, 3],
        output_token_ids=[-1, -1, -1, -1],
        additional_information_cpu={"omni_task": ["chat"]},
    )
    with pytest.raises(ValueError, match="unresolved output token placeholders"):
        ar2dit_full_payload(
            transfer_manager=None,
            pooling_output={"hidden": torch.zeros((6, 6))},
            request=request,
        )


def test_ar2dit_token_only_returns_placeholder_for_finished_outputs():
    finished = SimpleNamespace(finished=True)
    running = SimpleNamespace(finished=False)

    placeholders = ar2dit_token_only([running, finished])

    assert len(placeholders) == 1
    assert placeholders[0]["prompt_token_ids"] == [0]
    assert placeholders[0]["additional_information"] is None


def test_mammoth_moda2_pipeline_uses_request_end_full_payload():
    stage0, stage1 = MAMMOTH_MODA2_PIPELINE.stages

    assert stage0.custom_process_next_stage_input_func.endswith("ar2dit_full_payload")
    assert stage1.requires_full_payload_input is True
    assert stage1.sync_process_input_func.endswith("ar2dit_token_only")

    # The AR stage must qualify as a producer for the request-end accumulator.
    model_config = SimpleNamespace(
        async_chunk=False,
        final_output=False,
        model_stage="ar",
        custom_process_next_stage_input_func=stage0.custom_process_next_stage_input_func,
    )
    assert should_accumulate_full_payload_output(model_config, ar2dit_full_payload)
