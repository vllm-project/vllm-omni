# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for MiniMax H3's disaggregated text-encoder contract."""

import os
import shutil
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image

from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
    _load_audio,
    resolve_minimax_h3_diffusion_model_path,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.models.minimax_h3.checkpoint import (
    resolve_minimax_h3_model_root,
    resolve_minimax_h3_partition,
)
from vllm_omni.model_executor.models.minimax_h3.conditioning import (
    MINIMAX_H3_CONDITION_LABELS_KEY,
    MINIMAX_H3_PRESENTATION_TASK_KEY,
)
from vllm_omni.model_executor.models.minimax_h3.preprocessing import (
    minimax_h3_ref2va_presentation,
    minimax_h3_ref2va_video_presentation,
)
from vllm_omni.model_executor.models.minimax_h3.reference_video import (
    MINIMAX_H3_PREPARED_REFERENCE_VIDEOS_KEY,
    deserialize_prepared_reference_videos,
)
from vllm_omni.model_executor.models.minimax_h3.text_encoder import (
    MiniMaxH3MultiModalProcessor,
    _build_minimax_h3_presentation,
)
from vllm_omni.model_executor.stage_input_processors.minimax_h3 import (
    _audio_items,
    _diffusion_sampling_params,
    prepare_text_encoder_prompt,
    text_encoder2diffusion,
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
        "multi_modal_data": {
            "image": Image.new("RGB", (256, 256)),
            "audio": [np.zeros(16), 16_000],
        },
    }
    sampling = OmniDiffusionSamplingParams(
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


def test_text_encoder_prompt_rejects_injected_prepared_video_descriptor():
    prompt = {
        "prompt": "hello",
        "additional_information": {"meta": {MINIMAX_H3_PREPARED_REFERENCE_VIDEOS_KEY: '{"artifact_dir":"/tmp/user"}'}},
    }
    sampling = OmniDiffusionSamplingParams(height=256, width=448, extra_args={"task": "t2va"})

    transformed = prepare_text_encoder_prompt(prompt, [sampling])

    assert MINIMAX_H3_PREPARED_REFERENCE_VIDEOS_KEY not in transformed["additional_information"]["meta"]


def test_prepare_ref2va_video_uses_shared_frame_sampler_once(monkeypatch):
    prepare_videos = Mock()

    def prepare(_videos, *, target_frame_count, workdir, start_time_seconds):
        prepare_videos(_videos, target_frame_count, workdir, start_time_seconds)
        prepared_path = os.path.join(workdir, "prepared.mp4")
        open(prepared_path, "wb").close()
        return [
            {
                "original_path": "/tmp/input.mp4",
                "prepared_path": prepared_path,
                "input_has_audio": True,
                "width": 448,
                "height": 256,
                "start_time_seconds": 0.0,
                "duration_seconds": 5.2,
                "audio_duration_seconds": 5.2,
            }
        ]

    sample_frames = Mock(return_value={"frames": [np.zeros((4, 4, 3), dtype=np.uint8)]})
    monkeypatch.setattr("vllm_omni.model_executor.stage_input_processors.minimax_h3.prepare_reference_videos", prepare)
    monkeypatch.setattr(
        "vllm_omni.model_executor.stage_input_processors.minimax_h3.sample_reference_video_frames",
        sample_frames,
    )
    prompt = {
        "prompt": "hello",
        "multi_modal_data": {"video": "/tmp/input.mp4"},
    }
    sampling = OmniDiffusionSamplingParams(height=256, width=448, num_frames=124, extra_args={"task": "ref2va"})

    transformed = prepare_text_encoder_prompt(prompt, [sampling])

    prepare_videos.assert_called_once()
    _, target_frame_count, artifact_dir, start_time_seconds = prepare_videos.call_args.args
    assert target_frame_count == 124
    assert start_time_seconds is None
    descriptor = transformed["additional_information"]["meta"][MINIMAX_H3_PREPARED_REFERENCE_VIDEOS_KEY]
    described_dir, described_videos = deserialize_prepared_reference_videos(descriptor)
    assert described_dir == artifact_dir
    assert os.path.isfile(described_videos[0]["prepared_path"])
    sample_frames.assert_called_once_with(described_videos[0]["prepared_path"])
    assert prompt["multi_modal_data"] == {"video": "/tmp/input.mp4"}
    assert transformed["mm_processor_kwargs"][MINIMAX_H3_CONDITION_LABELS_KEY] == [
        ("audio", 1),
        ("video", 1),
    ]
    shutil.rmtree(artifact_dir)


def test_stage_wire_rejects_multiple_text_encoder_sources():
    with pytest.raises(RuntimeError, match="exactly one text-encoder source"):
        text_encoder2diffusion([SimpleNamespace(), SimpleNamespace()], prompt={"prompt": "hello"})


def test_stage_wire_rejects_request_id_mismatch():
    source = SimpleNamespace(request_id="other", outputs=[SimpleNamespace()])
    prompt = {
        "prompt": "hello",
        "additional_information": {"global_request_id": ["req-1"]},
    }

    with pytest.raises(RuntimeError, match="request ID does not match"):
        text_encoder2diffusion([source], prompt=prompt)


def test_stage_wire_rejects_multiple_completions():
    source = SimpleNamespace(request_id="req-1", outputs=[SimpleNamespace(), SimpleNamespace()])

    with pytest.raises(RuntimeError, match="exactly one completion"):
        text_encoder2diffusion([source], prompt={"prompt": "hello"})


def _source_output(payload: dict) -> SimpleNamespace:
    completion = SimpleNamespace(multimodal_output=payload)
    return SimpleNamespace(request_id="request-1", outputs=[completion])


def _diffusion_prompt() -> dict:
    return {
        "request_id": "request-1",
        "prompt": "test prompt",
        "additional_information": {},
    }


def test_text_encoder2diffusion_reads_hidden_states_output_and_token_role_ids() -> None:
    hidden = torch.randn(4, 5120)
    token_role_ids = torch.tensor([[1], [1], [0], [0]])

    result = text_encoder2diffusion(
        [
            _source_output(
                {
                    "hidden_states": {"output": hidden},
                    "meta": {"token_role_ids": token_role_ids},
                }
            )
        ],
        _diffusion_prompt(),
    )

    conditioning = result["additional_information"]["text_encoder_output"]
    torch.testing.assert_close(conditioning["hidden_states"], hidden)
    torch.testing.assert_close(conditioning["token_tags"], token_role_ids.squeeze(-1))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"meta": {"token_role_ids": torch.ones(4, 1)}}, "no hidden_states payload"),
        ({"hidden_states": {}, "meta": {"token_role_ids": torch.ones(4, 1)}}, "no hidden_states.output tensor"),
        ({"hidden_states": {"output": torch.randn(4, 5120)}}, "no conditioning metadata"),
        ({"hidden_states": {"output": torch.randn(4, 5120)}, "meta": {}}, "no token_role_ids tensor"),
        (
            {
                "hidden_states": {"output": torch.randn(4, 5120)},
                "meta": {"token_role_ids": torch.ones(4)},
            },
            r"must have shape \[tokens, 1\]",
        ),
    ],
)
def test_text_encoder2diffusion_rejects_invalid_conditioning_payload(payload: dict, message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        text_encoder2diffusion([_source_output(payload)], _diffusion_prompt())


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
