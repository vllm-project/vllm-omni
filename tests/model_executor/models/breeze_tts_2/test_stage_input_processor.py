from types import SimpleNamespace

import torch

from vllm_omni.model_executor.stage_input_processors.breeze_tts_2 import (
    talker2codec,
    talker2codec_full_payload,
)


def test_talker2codec_flattens_codebook_major():
    codes = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int16)
    output = SimpleNamespace(multimodal_output={"codes": {"audio": codes}})

    prompts = talker2codec([output])

    assert prompts[0]["prompt_token_ids"] == [1, 3, 5, 2, 4, 6]
    assert torch.equal(prompts[0]["multi_modal_data"]["codes"]["audio"], codes.transpose(0, 1))


def test_talker2codec_keeps_empty_request_terminal():
    output = SimpleNamespace(multimodal_output={"codes": {}})

    prompts = talker2codec([output])

    assert prompts == [{"prompt_token_ids": []}]


def test_talker2codec_accepts_engine_core_output_wrapper():
    codes = torch.tensor([[7, 8]], dtype=torch.int16)
    output = SimpleNamespace(
        outputs=[SimpleNamespace(multimodal_output={"codes": {"audio": codes}})]
    )

    prompts = talker2codec([output])

    assert prompts[0]["prompt_token_ids"] == [7, 8]


def test_talker2codec_unwraps_per_request_audio_container():
    codes = torch.tensor([[7, 8], [9, 10]], dtype=torch.int16)
    output = SimpleNamespace(multimodal_output={"codes": {"audio": [codes]}})

    prompts = talker2codec([output])

    assert prompts[0]["prompt_token_ids"] == [7, 9, 8, 10]


def test_talker2codec_accepts_flat_full_payload():
    output = {"codes.audio": torch.tensor([1, 3, 2, 4], dtype=torch.int16)}

    prompts = talker2codec([output])

    assert prompts[0]["prompt_token_ids"] == [1, 3, 2, 4]
    assert torch.equal(
        prompts[0]["multi_modal_data"]["codes"]["audio"],
        torch.tensor([1, 3, 2, 4], dtype=torch.int16),
    )


def test_talker2codec_full_payload_transposes_accumulated_frames():
    payload = talker2codec_full_payload(
        transfer_manager=None,
        pooling_output={"codes.audio": torch.tensor([[1, 2], [3, 4], [5, 6]])},
        request=None,
    )

    assert payload is not None
    assert torch.equal(
        payload.codes.audio,
        torch.tensor([1, 3, 5, 2, 4, 6]),
    )


def test_talker2codec_full_payload_accepts_runner_pooling_output():
    payload = talker2codec_full_payload(
        pooling_output={"codes.audio": torch.tensor([[9, 8]], dtype=torch.int16)},
        request=None,
    )

    assert payload is not None
    assert torch.equal(payload.codes.audio, torch.tensor([9, 8]))
