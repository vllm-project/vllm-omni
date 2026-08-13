# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MiniMax H3's disaggregated text-encoder contract."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
    _load_audio,
    resolve_minimax_h3_diffusion_model_path,
)
from vllm_omni.diffusion.models.minimax_h3.presentation import (
    minimax_h3_ref2va_presentation,
    minimax_h3_ref2va_video_presentation,
)
from vllm_omni.model_executor.models.minimax_h3.checkpoint import (
    resolve_minimax_h3_model_root,
)
from vllm_omni.model_executor.models.minimax_h3.conditioning import (
    MINIMAX_H3_CONDITION_LABELS_KEY,
    MINIMAX_H3_PRESENTATION_TASK_KEY,
    MINIMAX_H3_TEXT_HIDDEN_SIZE,
)
from vllm_omni.model_executor.models.minimax_h3.pipeline import MINIMAX_H3_PIPELINE
from vllm_omni.model_executor.models.minimax_h3.text_encoder import (
    MiniMaxH3MultiModalProcessor,
    _build_minimax_h3_presentation,
)
from vllm_omni.model_executor.stage_input_processors.minimax_h3 import (
    _audio_items,
    prepare_text_encoder_prompt,
    text_encoder2diffusion,
    text_encoder2diffusion_full_payload,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _SegmentTokenizer:
    _special_ids = {
        "<|vision_start|>": 1,
        "<|vision_end|>": 2,
        "<|image_pad|>": 3,
        "<|video_pad|>": 4,
    }

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [100 + len(text), 1000 + sum(text.encode())]}

    def convert_tokens_to_ids(self, token):
        return self._special_ids[token]


def test_h3_processor_reprocesses_media_instead_of_using_partial_sender_cache(mocker):
    processor = object.__new__(MiniMaxH3MultiModalProcessor)
    sentinel = ([1, 2, 3], object(), True)
    apply_processor = mocker.patch.object(
        MiniMaxH3MultiModalProcessor,
        "_apply_hf_processor",
        return_value=sentinel,
    )
    inputs = object()
    timing_ctx = object()

    result = processor._cached_apply_hf_processor(inputs, timing_ctx)

    assert result is sentinel
    apply_processor.assert_called_once_with(inputs, timing_ctx)


@pytest.mark.parametrize(
    ("value", "expected_count"),
    [
        ((torch.zeros(16), 16_000), 1),
        ([np.zeros(16), 16_000], 1),
        ([(torch.zeros(16), 16_000), (torch.ones(16), 24_000)], 2),
        (["first.wav", "second.wav"], 2),
    ],
)
def test_audio_items_preserves_waveform_pairs(value, expected_count):
    assert len(_audio_items(value)) == expected_count


def test_fused_audio_loader_accepts_list_waveform_pair():
    waveform, sample_rate = _load_audio([[0.0, 0.5, -0.5], 16_000])
    assert sample_rate == 16_000
    torch.testing.assert_close(waveform, torch.tensor([0.0, 0.5, -0.5]))


def test_prepare_ref2va_keeps_original_text_and_exact_condition_order():
    prompt = {
        "prompt": "hello",
        "multi_modal_data": {
            "image": Image.new("RGB", (256, 256)),
            "audio": [np.zeros(16), 16_000],
        },
    }
    sampling = SimpleNamespace(
        height=256,
        width=448,
        extra_args={"task": "ref2va"},
    )

    transformed = prepare_text_encoder_prompt(prompt, [sampling])

    assert transformed["prompt"] == "hello"
    assert len(transformed["multi_modal_data"]["image"]) == 1
    assert "audio" not in transformed["multi_modal_data"]
    assert transformed["mm_processor_kwargs"][MINIMAX_H3_PRESENTATION_TASK_KEY] == "ref2va"
    assert transformed["mm_processor_kwargs"][MINIMAX_H3_CONDITION_LABELS_KEY] == [
        ("image", 1),
        ("audio", 1),
    ]


def test_ref2va_one_image_tokens_and_tags_match_fused_presentation():
    tokenizer = _SegmentTokenizer()
    labels = [("image", 1), ("audio", 1)]
    image_grid = torch.tensor([[1, 4, 4]])

    actual = _build_minimax_h3_presentation(
        tokenizer,
        prompt="hello",
        task="ref2va",
        condition_labels=labels,
        image_grid_thw=image_grid,
        video_grid_thw=None,
        video_timestamps=None,
        merge_size=2,
    )
    expected = minimax_h3_ref2va_presentation(
        tokenizer,
        prompt="hello",
        condition_labels=labels,
        image_token_count=[4],
    )

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_ref2va_video_tokens_and_tags_match_fused_without_outer_markers():
    tokenizer = _SegmentTokenizer()
    labels = [("audio", 1), ("video", 1)]
    video_grid = torch.tensor([[2, 4, 4]])
    timestamps = [[0.2, 0.4]]

    actual = _build_minimax_h3_presentation(
        tokenizer,
        prompt="hello",
        task="ref2va",
        condition_labels=labels,
        image_grid_thw=None,
        video_grid_thw=video_grid,
        video_timestamps=timestamps,
        merge_size=2,
    )
    expected = minimax_h3_ref2va_video_presentation(
        tokenizer,
        prompt="hello",
        condition_labels=labels,
        image_token_count=None,
        video_block_token_counts=[[4, 4]],
        video_block_timestamps=timestamps,
    )

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    assert int((actual[0] == tokenizer._special_ids["<|vision_start|>"]).sum()) == 2
    assert int((actual[0] == tokenizer._special_ids["<|vision_end|>"]).sum()) == 2


def test_checkpoint_resolver_selects_local_partition(tmp_path):
    root = tmp_path / "MiniMax-H3"
    (root / "FL2VA" / "text_encoder").mkdir(parents=True)
    (root / "Ref2VA" / "text_encoder").mkdir(parents=True)

    assert resolve_minimax_h3_model_root(str(root), None, "fl2va") == str(root / "FL2VA" / "text_encoder")
    assert resolve_minimax_h3_model_root(str(root), None, "ref2va") == str(root / "Ref2VA" / "text_encoder")
    assert resolve_minimax_h3_model_root(str(root / "Ref2VA"), None, None) == str(root / "Ref2VA" / "text_encoder")
    assert resolve_minimax_h3_model_root(str(root / "FL2VA"), None, "ref2va") == str(root / "Ref2VA" / "text_encoder")


def test_checkpoint_resolver_rejects_unknown_task(tmp_path):
    with pytest.raises(ValueError, match="task_type must be one of"):
        resolve_minimax_h3_model_root(str(tmp_path), None, "unknown")


def test_diffusion_resolver_selects_startup_partition(tmp_path):
    root = tmp_path / "MiniMax-H3"
    fl2va = root / "FL2VA"
    ref2va = root / "Ref2VA"
    fl2va.mkdir(parents=True)
    ref2va.mkdir()
    (fl2va / "model_index.json").write_text("{}")
    (ref2va / "model_index.json").write_text("{}")

    assert resolve_minimax_h3_diffusion_model_path(str(root), None, "fl2va") == str(fl2va)
    assert resolve_minimax_h3_diffusion_model_path(str(root), None, "ref2va") == str(ref2va)
    assert resolve_minimax_h3_diffusion_model_path(str(ref2va), None, None) == str(ref2va)


def _stage0_payload(tokens=4):
    hidden = torch.randn(tokens, MINIMAX_H3_TEXT_HIDDEN_SIZE)
    tags = torch.zeros(tokens, 1, dtype=torch.long)
    return {"encoder_hidden_states": hidden, "token_tags": tags}


def _source_outputs(multimodal_output):
    completion = SimpleNamespace(multimodal_output=multimodal_output)
    return [SimpleNamespace(outputs=[completion])]


def test_full_payload_hook_emits_the_diffusion_ready_structure():
    payload = _stage0_payload()

    result = text_encoder2diffusion_full_payload(pooling_output=payload)

    # The DiT worker merges this verbatim, so it must already be unpacked.
    assert set(result) == {"text_encoder_output"}
    conditioning = result["text_encoder_output"]
    assert torch.equal(conditioning["hidden_states"], payload["encoder_hidden_states"])
    assert conditioning["token_tags"].shape == (4,)


def test_full_payload_hook_tolerates_a_connector_less_stage():
    assert text_encoder2diffusion_full_payload(pooling_output=None) is None
    assert text_encoder2diffusion_full_payload(pooling_output={"encoder_hidden_states": None}) is None


def test_full_payload_hook_rejects_mismatched_conditioning():
    payload = _stage0_payload()
    payload["token_tags"] = torch.zeros(9, 1, dtype=torch.long)

    with pytest.raises(RuntimeError, match="align"):
        text_encoder2diffusion_full_payload(pooling_output=payload)


def test_inline_conditioning_still_reaches_the_diffusion_prompt():
    prompt = text_encoder2diffusion(_source_outputs(_stage0_payload()), {"prompt": "a cat"})

    conditioning = prompt["additional_information"]["text_encoder_output"]
    assert conditioning["hidden_states"].shape == (4, MINIMAX_H3_TEXT_HIDDEN_SIZE)
    assert conditioning["token_tags"].shape == (4,)


@pytest.mark.parametrize("multimodal_output", [None, {}, {"encoder_hidden_states": torch.zeros(1)}])
def test_prompt_passes_through_when_conditioning_travels_over_the_connector(multimodal_output):
    prompt = text_encoder2diffusion(_source_outputs(multimodal_output), {"prompt": "a cat"})

    assert prompt == {"prompt": "a cat"}


def test_stage0_declares_the_producer_hook_the_connector_path_needs():
    stage0, stage1 = MINIMAX_H3_PIPELINE.stages

    assert stage0.custom_process_next_stage_input_func.endswith(".text_encoder2diffusion_full_payload")
    assert stage1.stage_input_payload_keys == ("text_encoder_output",)
