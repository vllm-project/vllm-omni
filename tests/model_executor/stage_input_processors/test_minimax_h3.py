# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for MiniMax H3's disaggregated text-encoder contract."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image

from vllm_omni.data_entry_keys import flatten_payload
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import resolve_minimax_h3_diffusion_model_path
from vllm_omni.engine.serialization import (
    deserialize_additional_information,
    serialize_additional_information,
)
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.models.minimax_h3.checkpoint import (
    resolve_minimax_h3_encoder_model_root,
    resolve_minimax_h3_partition,
)
from vllm_omni.model_executor.models.minimax_h3.conditioning import (
    MINIMAX_H3_CONDITION_LABELS_KEY,
    MINIMAX_H3_ENCODER_REQUEST_KEY,
    MINIMAX_H3_PRESENTATION_TASK_KEY,
    MiniMaxH3EncoderConditioning,
)
from vllm_omni.model_executor.models.minimax_h3.encoder_processing import _audio_items, _load_audio
from vllm_omni.model_executor.models.minimax_h3.preprocessing import (
    minimax_h3_ref2va_presentation,
    minimax_h3_ref2va_video_presentation,
)
from vllm_omni.model_executor.models.minimax_h3.text_encoder import (
    MiniMaxH3MultiModalProcessor,
    _build_minimax_h3_presentation,
)
from vllm_omni.model_executor.stage_input_processors.minimax_h3 import (
    _diffusion_sampling_params,
    encoder2diffusion,
    prepare_encoder_prompt,
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


def test_h3_processor_reprocesses_media_instead_of_using_partial_sender_cache(monkeypatch):
    processor = object.__new__(MiniMaxH3MultiModalProcessor)
    sentinel = ([1, 2, 3], object(), True)
    apply_processor = Mock(return_value=sentinel)
    monkeypatch.setattr(
        MiniMaxH3MultiModalProcessor,
        "_apply_hf_processor",
        apply_processor,
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


def test_h3_selects_the_single_explicit_diffusion_stage_params() -> None:
    stage_zero = SimpleNamespace(extra_args={"task": "t2va"})
    diffusion = OmniDiffusionSamplingParams(extra_args={"task": "ref2va"})

    assert _diffusion_sampling_params([stage_zero, diffusion]) is diffusion


def test_h3_rejects_missing_or_ambiguous_diffusion_stage_params() -> None:
    with pytest.raises(RuntimeError, match="exactly one OmniDiffusionSamplingParams"):
        _diffusion_sampling_params([SimpleNamespace(extra_args={"task": "t2va"})])
    with pytest.raises(RuntimeError, match="got 2"):
        _diffusion_sampling_params([OmniDiffusionSamplingParams(), OmniDiffusionSamplingParams()])


def test_fused_audio_loader_accepts_list_waveform_pair():
    waveform, sample_rate = _load_audio([[0.0, 0.5, -0.5], 16_000])
    assert sample_rate == 16_000
    torch.testing.assert_close(waveform, torch.tensor([0.0, 0.5, -0.5]))


def test_prepare_ref2va_keeps_original_text_and_exact_condition_order():
    prompt = {
        "prompt": "hello",
        "additional_information": {"global_request_id": ["request-1"]},
        "model_intermediate_buffer": {"private": "preserved"},
        "multi_modal_data": {
            "image": Image.new("RGB", (256, 256)),
            "audio": [np.zeros(32_000), 16_000],
        },
    }
    sampling = OmniDiffusionSamplingParams(
        height=256,
        width=448,
        extra_args={"task": "ref2va"},
    )

    transformed = prepare_encoder_prompt(prompt, [sampling])

    assert transformed["prompt"] == "hello"
    assert len(transformed["multi_modal_data"]["image"]) == 1
    assert "audio" not in transformed["multi_modal_data"]
    assert transformed["mm_processor_kwargs"][MINIMAX_H3_PRESENTATION_TASK_KEY] == "ref2va"
    assert transformed["mm_processor_kwargs"][MINIMAX_H3_CONDITION_LABELS_KEY] == [
        ("image", 1),
        ("audio", 1),
    ]
    assert transformed["model_intermediate_buffer"] == {"private": "preserved"}
    runner_info = transformed["additional_information"]
    for _ in range(2):
        wire = serialize_additional_information(runner_info)
        runner_info = deserialize_additional_information(wire)
    assert runner_info["global_request_id"] == ["request-1"]
    request_metadata = runner_info["meta"][MINIMAX_H3_ENCODER_REQUEST_KEY]
    assert request_metadata["task"] == "ref2va"
    assert isinstance(
        runner_info["hidden_states"]["layers"][0],
        torch.Tensor,
    )
    assert isinstance(
        runner_info["hidden_states"]["layers"][1],
        torch.Tensor,
    )
    from vllm_omni.model_executor.models.minimax_h3.encoder import MiniMaxH3Encoder

    media = MiniMaxH3Encoder._media_input(runner_info)
    assert media.task == "ref2va"
    assert len(media.images) == 1
    assert len(media.audios) == 1


def _mock_ref2va_video_with_audio(monkeypatch, *, duration_seconds: float) -> None:
    from vllm_omni.model_executor.models.minimax_h3 import encoder_processing

    sample_rate = 16_000
    frames = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    monkeypatch.setattr(
        encoder_processing,
        "prepare_reference_videos",
        lambda *args, **kwargs: [
            {
                "prepared_path": "prepared.mp4",
                "original_path": "original.mp4",
                "input_has_audio": True,
                "duration_seconds": duration_seconds,
                "audio_duration_seconds": duration_seconds,
            }
        ],
    )
    monkeypatch.setattr(encoder_processing, "load_video_frames", lambda path: frames)
    monkeypatch.setattr(
        encoder_processing,
        "sample_reference_video_frames",
        lambda *args, **kwargs: {"block_timestamps": [0.0], "frames": frames[:1]},
    )
    monkeypatch.setattr(
        encoder_processing,
        "load_video_audio",
        lambda *args, **kwargs: (torch.zeros(round(duration_seconds * sample_rate)), sample_rate),
    )


def test_prepare_ref2va_rejects_short_embedded_video_audio(monkeypatch):
    _mock_ref2va_video_with_audio(monkeypatch, duration_seconds=1.0)
    sampling = OmniDiffusionSamplingParams(
        height=256,
        width=448,
        num_frames=96,
        extra_args={"task": "ref2va"},
    )
    prompt = {"prompt": "hello", "multi_modal_data": {"video": "original.mp4"}}

    with pytest.raises(OmniClientError, match=r"duration must be in \[2, 15\] seconds, got 1\.000"):
        prepare_encoder_prompt(prompt, [sampling])


def test_prepare_ref2va_rejects_combined_embedded_and_standalone_audio_duration(monkeypatch):
    duration_seconds = 8.0
    sample_rate = 16_000
    _mock_ref2va_video_with_audio(monkeypatch, duration_seconds=duration_seconds)
    sampling = OmniDiffusionSamplingParams(
        height=256,
        width=448,
        num_frames=240,
        extra_args={"task": "ref2va"},
    )
    prompt = {
        "prompt": "hello",
        "multi_modal_data": {
            "video": "original.mp4",
            "audio": (torch.zeros(int(duration_seconds * sample_rate)), sample_rate),
        },
    }

    with pytest.raises(OmniClientError, match="at most 15 seconds in total"):
        prepare_encoder_prompt(prompt, [sampling])


def _encoder_output() -> dict:
    return MiniMaxH3EncoderConditioning(
        hidden_states=torch.randn(3, 5120, dtype=torch.bfloat16),
        token_tags=torch.tensor([1, 0, 1], dtype=torch.int64),
        task="t2va",
        height=256,
        width=448,
        num_frames=17,
        latent_t=5,
        audio_t=10,
    ).to_omni_payload()


def test_encoder2diffusion_reuses_encoder_handoff() -> None:
    prompt = {
        "prompt": "test prompt",
        "multi_modal_data": {"image": object()},
        "additional_information": {
            "private": "preserved",
            "meta": {MINIMAX_H3_ENCODER_REQUEST_KEY: {"task": "t2va"}},
            "hidden_states": {"layers": {0: torch.zeros(1)}},
        },
        "model_intermediate_buffer": {"private": "encoder-only"},
    }
    source = SimpleNamespace(
        finished=True,
        request_id="request-1",
        outputs=[SimpleNamespace(multimodal_output=flatten_payload(_encoder_output()))],
    )

    result = encoder2diffusion([source], prompt)

    assert result["prompt"] == "test prompt"
    assert result["multi_modal_data"] is None
    assert "model_intermediate_buffer" not in result
    additional_information = result["additional_information"]
    assert additional_information["private"] == "preserved"
    assert "hidden_states" not in additional_information
    assert "meta" not in additional_information
    parsed = MiniMaxH3EncoderConditioning.from_omni_payload(additional_information["encoder_output"])
    assert parsed.task == "t2va"


def test_encoder2diffusion_waits_for_one_finished_source() -> None:
    assert encoder2diffusion([SimpleNamespace(finished=False)], {"prompt": "hello"}) is None
    with pytest.raises(RuntimeError, match="exactly one encoder source"):
        encoder2diffusion([SimpleNamespace(), SimpleNamespace()], {"prompt": "hello"})


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

    assert resolve_minimax_h3_encoder_model_root(str(root), None, "fl2va") == str(root / "FL2VA" / "text_encoder")
    assert resolve_minimax_h3_encoder_model_root(str(root), None, "ref2va") == str(root / "Ref2VA" / "text_encoder")
    assert resolve_minimax_h3_encoder_model_root(str(root), None, "combined") == str(root / "FL2VA" / "text_encoder")
    assert resolve_minimax_h3_encoder_model_root(str(root / "Ref2VA"), None, None) == str(
        root / "Ref2VA" / "text_encoder"
    )
    assert resolve_minimax_h3_encoder_model_root(str(root / "FL2VA"), None, "ref2va") == str(
        root / "Ref2VA" / "text_encoder"
    )


def test_checkpoint_resolver_rejects_unknown_task(tmp_path):
    with pytest.raises(ValueError, match="task_type must be one of"):
        resolve_minimax_h3_encoder_model_root(str(tmp_path), None, "unknown")


def test_partition_resolver_preserves_consumer_auto_default(tmp_path):
    root = tmp_path / "MiniMax-H3"
    ref2va = root / "Ref2VA"
    ref2va.mkdir(parents=True)

    assert resolve_minimax_h3_partition(str(root), "auto", auto_partition="fl2va") == "fl2va"
    assert resolve_minimax_h3_partition(str(root), "auto", auto_partition="combined") == "combined"
    assert resolve_minimax_h3_partition(str(ref2va), "auto", auto_partition="combined") == "ref2va"


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
    assert resolve_minimax_h3_diffusion_model_path(str(root), None, None) == str(fl2va)
    assert resolve_minimax_h3_diffusion_model_path(str(root), None, "combined") == str(root)
    assert resolve_minimax_h3_diffusion_model_path(str(ref2va), None, None) == str(ref2va)


def test_diffusion_resolver_normalizes_partial_partition_directory(tmp_path):
    root = tmp_path / "MiniMax-H3"
    ref2va = root / "Ref2VA"
    (ref2va / "text_encoder").mkdir(parents=True)

    assert resolve_minimax_h3_diffusion_model_path(str(ref2va), None, "ref2va") == str(ref2va)
