# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for the MammothModa2 AR -> DiT request-end full payload path."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.mammoth_moda2.conditioning import (
    conditioning_spec_from_config,
    select_ar_conditions,
)
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


def _transfer_manager() -> SimpleNamespace:
    """Minimal runner shape matching the full-payload callback owner."""
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                llm_config=SimpleNamespace(gen_vocab_start_index=10),
                image_token_id=4,
                video_token_id=5,
                vision_start_token_id=6,
                vision_end_token_id=7,
            )
        )
    )


def test_select_ar_conditions_matches_mammoth_token_categories():
    config = _transfer_manager().model_config.hf_config
    hidden = torch.arange(30, dtype=torch.bfloat16).reshape(5, 6)
    text_cond, image_cond = select_ar_conditions(
        hidden,
        # The image placeholder in the question is discarded; only generated
        # vocabulary ids after answer_start_index become image conditions.
        [1, config.image_token_id, 2, 10, 11],
        answer_start_index=3,
        spec=conditioning_spec_from_config(config),
    )

    assert torch.equal(text_cond, hidden[[0, 2]])
    assert torch.equal(image_cond, hidden[[3, 4]])


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


def test_ar2dit_full_payload_selects_bf16_conditions_before_one_time_d2h():
    """Device-resident slices -> selected BF16 conditions -> one request-end D2H."""
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
    payload = ar2dit_full_payload(
        transfer_manager=_transfer_manager(),
        pooling_output={"hidden": hidden},
        request=request,
    )

    assert payload is not None
    text_cond = payload["text_prompt_embeds"]
    image_cond = payload["image_prompt_embeds"]
    assert text_cond.device.type == "cpu"
    assert image_cond.device.type == "cpu"
    assert text_cond.dtype == torch.bfloat16
    assert image_cond.dtype == torch.bfloat16
    assert text_cond.shape == (len(prompt_token_ids), 6)
    assert image_cond.shape == (len(output_token_ids) - 1, 6)
    assert text_cond.is_contiguous()
    assert image_cond.is_contiguous()
    assert torch.equal(text_cond, prefill.cpu())
    assert torch.equal(image_cond, decode.cpu())

    assert payload["full_token_ids"] == prompt_token_ids + output_token_ids[:-1]
    assert payload["answer_start_index"] == [len(prompt_token_ids)]
    assert payload["image_height"] == [512]
    assert payload["image_width"] == [768]
    assert payload["text_guidance_scale"] == [9.0]
    assert payload["cfg_range"] == [0.0, 1.0]
    assert payload["num_inference_steps"] == [50]


def test_ar2dit_full_payload_emits_direct_dit_condition_schema():
    """The request-end payload uses the DiT's direct-condition input branch."""
    prompt_token_ids = [1, 2, 3]
    gen_token_ids = [10, 11, 12, 13]
    full_hidden = torch.arange(36, dtype=torch.bfloat16).reshape(6, 6)
    addi_info = {
        "image_height": [512],
        "image_width": [768],
        "text_guidance_scale": [4.0],
        "cfg_range": [0.0, 1.0],
        "num_inference_steps": [25],
    }

    request_end = ar2dit_full_payload(
        transfer_manager=_transfer_manager(),
        pooling_output={"hidden": full_hidden},
        request=SimpleNamespace(
            request_id="r1",
            prompt_token_ids=prompt_token_ids,
            output_token_ids=gen_token_ids,
            additional_information_cpu=addi_info,
        ),
    )

    assert request_end is not None
    assert "full_hidden_states" not in request_end
    assert torch.equal(request_end["text_prompt_embeds"], full_hidden[:3])
    assert torch.equal(request_end["image_prompt_embeds"], full_hidden[3:])
    assert request_end["full_token_ids"] == prompt_token_ids + gen_token_ids[:-1]
    assert request_end["answer_start_index"] == [len(prompt_token_ids)]
    assert request_end["image_height"] == [512]
    assert request_end["image_width"] == [768]
    assert request_end["text_guidance_scale"] == [4.0]
    assert request_end["cfg_range"] == [0.0, 1.0]
    assert request_end["num_inference_steps"] == [25]

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
    )[0]["additional_information"]
    assert torch.equal(request_end["text_prompt_embeds"].float(), legacy["full_hidden_states"][:3])
    assert torch.equal(request_end["image_prompt_embeds"].float(), legacy["full_hidden_states"][3:])


def test_ar2dit_full_payload_rejects_hidden_token_id_mismatch():
    request = SimpleNamespace(
        request_id="r1",
        prompt_token_ids=[1, 2, 3],
        output_token_ids=[10, 11, 12, 13],  # expects 6 hidden rows
        additional_information_cpu=None,
    )
    with pytest.raises(ValueError, match="length mismatch"):
        ar2dit_full_payload(
            transfer_manager=_transfer_manager(),
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
        transfer_manager=_transfer_manager(),
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
        transfer_manager=_transfer_manager(),
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
            transfer_manager=_transfer_manager(),
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
            transfer_manager=_transfer_manager(),
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
