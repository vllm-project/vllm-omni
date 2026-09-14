# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for MiniMax H3's disaggregated encoder contract."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image

from vllm_omni.data_entry_keys import flatten_payload, to_struct
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
    MINIMAX_H3_ENCODER_LAYOUT_KEY,
    MINIMAX_H3_ENCODER_REQUEST_KEY,
    MINIMAX_H3_PRESENTATION_TASK_KEY,
    MINIMAX_H3_TEXT_CONDITIONING_SCHEMA,
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
    encoder2diffusion_full_payload,
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


@pytest.mark.parametrize("standalone_count", [1, 2])
def test_prepare_ref2va_uses_separate_embedded_and_standalone_audio_budgets(monkeypatch, standalone_count):
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
            "audio": [(torch.zeros(int(duration_seconds * sample_rate)), sample_rate) for _ in range(standalone_count)],
        },
    }

    if standalone_count == 2:
        with pytest.raises(OmniClientError, match="at most 15 seconds in total"):
            prepare_encoder_prompt(prompt, [sampling])
    else:
        transformed = prepare_encoder_prompt(prompt, [sampling])
        from vllm_omni.model_executor.models.minimax_h3.encoder import MiniMaxH3Encoder

        media = MiniMaxH3Encoder._media_input(transformed["additional_information"])
        assert len(media.video_audios) == 1 and len(media.audios) == 1
        assert media.video_audios[0][0].shape[-1] == 8 * sample_rate
        assert media.audios[0][0].shape[-1] == 8 * sample_rate


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


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("hidden_width", r"hidden_states must have shape \[tokens, 5120\]"),
        ("hidden_dtype", "hidden_states must have dtype torch.bfloat16"),
        ("hidden_noncontiguous", "hidden_states must use contiguous strided layout"),
        ("hidden_non_strided", "hidden_states must use contiguous strided layout"),
        ("token_dtype", "token_tags must have dtype torch.int64"),
        ("token_layout", "token_tags must use contiguous strided layout"),
        ("token_count", "token_tags must align with hidden_states"),
        ("token_value", "token_tags must contain only 0 and 1"),
    ],
)
@pytest.mark.parametrize("via_connector", [False, True])
def test_encoder2diffusion_rejects_h3_v1_contract_mismatch(case: str, message: str, via_connector: bool) -> None:
    payload = _encoder_output()
    payload["hidden_states"]["output"] = torch.zeros(4, 5120, dtype=torch.bfloat16)
    payload["meta"]["token_role_ids"] = torch.tensor([[1], [1], [0], [0]], dtype=torch.int64)

    if case == "hidden_width":
        payload["hidden_states"]["output"] = torch.zeros(4, 5119, dtype=torch.bfloat16)
    elif case == "hidden_dtype":
        payload["hidden_states"]["output"] = torch.zeros(4, 5120, dtype=torch.float32)
    elif case == "hidden_noncontiguous":
        payload["hidden_states"]["output"] = torch.empty(5120, 4, dtype=torch.bfloat16).t()
    elif case == "hidden_non_strided":
        payload["hidden_states"]["output"] = torch.empty(
            (4, 5120),
            dtype=torch.bfloat16,
            layout=torch.sparse_coo,
        )
    elif case == "token_dtype":
        payload["meta"]["token_role_ids"] = torch.tensor([[1], [1], [0], [0]], dtype=torch.int32)
    elif case == "token_layout":
        payload["meta"]["token_role_ids"] = torch.tensor(
            [[1], [0], [1], [0], [1], [0], [1], [0]],
            dtype=torch.int64,
        )[::2]
    elif case == "token_count":
        payload["meta"]["token_role_ids"] = torch.tensor([[1], [0], [0]], dtype=torch.int64)
    elif case == "token_value":
        payload["meta"]["token_role_ids"] = torch.tensor([[1], [1], [2], [0]], dtype=torch.int64)
    else:  # pragma: no cover - parameterization is exhaustive
        raise AssertionError(f"unknown test case: {case}")

    with pytest.raises(RuntimeError, match=message) as exc_info:
        if via_connector:
            encoder2diffusion_full_payload(pooling_output=payload)
        else:
            encoder2diffusion([_source_output(payload)], {"prompt": "hello"})

    assert MINIMAX_H3_TEXT_CONDITIONING_SCHEMA in str(exc_info.value)


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


def _source_output(payload) -> SimpleNamespace:
    return SimpleNamespace(
        finished=True,
        request_id="request-1",
        outputs=[SimpleNamespace(multimodal_output=payload)],
    )


def _full_encoder_output() -> dict:
    return MiniMaxH3EncoderConditioning(
        hidden_states=torch.randn(4, 5120, dtype=torch.bfloat16),
        token_tags=torch.tensor([1, 1, 0, 0], dtype=torch.int64),
        task="ref2va",
        height=256,
        width=448,
        num_frames=17,
        latent_t=5,
        audio_t=10,
        visual_condition=torch.arange(12 * 96, dtype=torch.float32).reshape(12, 96),
        visual_condition_shapes=((1, 4, 4), (2, 4, 4)),
        audio_condition=torch.arange(10 * 32, dtype=torch.float32).reshape(10, 32),
        audio_condition_lengths=(3, 2),
        ref_blocks=(
            {"kind": "image", "latent_t": 1, "latent_h": 4, "latent_w": 4},
            {"kind": "video_audio", "ref_audio_t": 3, "latent_t": 2, "latent_h": 4, "latent_w": 4},
            {"kind": "audio", "ref_audio_t": 2},
        ),
        keyframe_frame_indices=(0, 16),
    ).to_omni_payload()


def _runner_payload(payload: dict) -> dict:
    # Encoder.make_omni_output emits dotted private layout metadata. The runner
    # selects each request's media tensors and flattens its nested categories.
    payload = dict(payload)
    payload[f"kv_metadata.{MINIMAX_H3_ENCODER_LAYOUT_KEY}"] = payload.pop("kv_metadata")[MINIMAX_H3_ENCODER_LAYOUT_KEY]
    payload["meta"] = {"token_role_ids": payload["meta"]["token_role_ids"].reshape(-1, 1)}
    return flatten_payload(payload)


@pytest.mark.parametrize("representation", ["nested", "struct", "flat", "runner"])
@pytest.mark.parametrize("with_media", [False, True])
@pytest.mark.parametrize("via_connector", [False, True])
def test_encoder_handoff_preserves_all_components(representation, with_media, via_connector):
    payload = _full_encoder_output() if with_media else _encoder_output()
    inputs = {
        "nested": payload,
        "struct": to_struct(payload),
        "flat": flatten_payload(payload),
        "runner": _runner_payload(payload),
    }
    if via_connector:
        result = encoder2diffusion_full_payload(pooling_output=inputs[representation], request_id="request-1")
        assert set(result) == {"encoder_output"}
    else:
        result = encoder2diffusion([_source_output(inputs[representation])], {"prompt": "hello"})[
            "additional_information"
        ]
    output = result["encoder_output"]
    for group in ("hidden_states", "embed", "meta", "kv_metadata"):
        for key, tensor in payload[group].items():
            torch.testing.assert_close(output[group][key], tensor, rtol=0, atol=0)
    # Nonempty tensors must not be stripped, copied to CPU, or coerced to a
    # different dtype by the adapter. Empty FP32 optional slots remain on wire.
    assert output["hidden_states"]["output"] is payload["hidden_states"]["output"]
    if with_media:
        assert output["embed"]["embedding"] is payload["embed"]["embedding"]
        assert output["embed"]["speech_feat"] is payload["embed"]["speech_feat"]
    else:
        for tensor in output["embed"].values():
            assert tensor.shape == (0,)
            assert tensor.dtype == torch.float32
    for _ in range(2):
        result = deserialize_additional_information(serialize_additional_information(result))
    restored = MiniMaxH3EncoderConditioning.from_omni_payload(result["encoder_output"])
    original = MiniMaxH3EncoderConditioning.from_omni_payload(payload)
    for name in (
        "task",
        "height",
        "width",
        "num_frames",
        "latent_t",
        "audio_t",
        "visual_condition_shapes",
        "audio_condition_lengths",
        "ref_blocks",
        "keyframe_frame_indices",
    ):
        assert getattr(restored, name) == getattr(original, name)
    for group, fields in output.items():
        for key, tensor in fields.items():
            torch.testing.assert_close(result["encoder_output"][group][key], tensor, rtol=0, atol=0)


def test_full_payload_hook_skips_only_absent_output():
    assert encoder2diffusion_full_payload(pooling_output=None) is None


@pytest.mark.parametrize("via_connector", [False, True])
@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("not_mapping", "no conditioning payload"),
        ("text_only", "requires text, visual and audio tensors"),
        ("missing_hidden", "requires text, visual and audio tensors"),
        ("missing_tags", "requires text, visual and audio tensors"),
        ("missing_layout", "requires private layout metadata"),
        ("truncated_layout", "header is truncated"),
        ("layout_dtype", "one-dimensional integer tensor"),
        ("schema", "unsupported MiniMax H3 encoder wire schema"),
        ("visual_shape", "visual condition must be FP32"),
        ("audio_dtype", "audio condition must be FP32"),
        ("empty_visual", "empty visual condition slot"),
        ("empty_audio", "empty audio condition slot"),
    ],
)
def test_encoder_handoff_rejects_invalid_full_payload(case, message, via_connector):
    payload = _full_encoder_output()
    layout = payload["kv_metadata"][MINIMAX_H3_ENCODER_LAYOUT_KEY]
    if case == "not_mapping":
        payload = []
    elif case == "text_only":
        payload.pop("embed")
    elif case == "missing_hidden":
        payload.pop("hidden_states")
    elif case == "missing_tags":
        payload["meta"] = {}
    elif case == "missing_layout":
        payload.pop("kv_metadata")
    elif case == "truncated_layout":
        payload["kv_metadata"][MINIMAX_H3_ENCODER_LAYOUT_KEY] = layout[:5]
    elif case == "layout_dtype":
        payload["kv_metadata"][MINIMAX_H3_ENCODER_LAYOUT_KEY] = layout.float()
    elif case == "schema":
        layout[1] = 99
    elif case == "visual_shape":
        payload["embed"]["embedding"] = torch.zeros(11, 96)
    elif case == "audio_dtype":
        payload["embed"]["speech_feat"] = payload["embed"]["speech_feat"].to(torch.bfloat16)
    elif case == "empty_visual":
        payload = _encoder_output()
        payload["embed"]["embedding"] = torch.empty(0, 96)
    elif case == "empty_audio":
        payload = _encoder_output()
        payload["embed"]["speech_feat"] = torch.empty(0, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match=message):
        if via_connector:
            encoder2diffusion_full_payload(pooling_output=payload)
        else:
            encoder2diffusion([_source_output(payload)], {"prompt": "hello"})


@pytest.mark.parametrize("inline", [False, True])
def test_encoder_handoff_cleans_media_without_mutating_prompt_or_stripping_outputs(inline):
    incoming = _full_encoder_output()
    retained_hidden = torch.ones(1)
    media = torch.zeros(2)
    prompt = {
        "prompt": "hello",
        "negative_prompt": "blur",
        "multi_modal_data": {"image": object(), "audio": object(), "video": object()},
        "model_intermediate_buffer": {"media": media},
        "additional_information": {
            "global_request_id": ["request-1"],
            "private": "preserved",
            "hidden_states": {"layers": {0: media}, "output": retained_hidden},
            "meta": {MINIMAX_H3_ENCODER_REQUEST_KEY: {"task": "ref2va"}, "private": 42},
            "encoder_output": incoming,
        },
    }
    source = _source_output(_runner_payload(incoming) if inline else None)
    if not inline:
        del source.outputs[0].multimodal_output
    result = encoder2diffusion([source], [prompt])
    info = result["additional_information"]
    assert result["multi_modal_data"] is None
    assert "model_intermediate_buffer" not in result
    assert result["negative_prompt"] == "blur"
    assert info["global_request_id"] == ["request-1"]
    assert info["private"] == "preserved"
    assert info["meta"] == {"private": 42}
    assert set(info["hidden_states"]) == {"output"}
    assert info["hidden_states"]["output"] is retained_hidden
    assert info["encoder_output"]["embed"]["embedding"] is incoming["embed"]["embedding"]
    assert info["encoder_output"]["embed"]["speech_feat"] is incoming["embed"]["speech_feat"]
    if not inline:
        assert info["encoder_output"] is incoming
    assert prompt["multi_modal_data"] is not None
    assert prompt["model_intermediate_buffer"]["media"] is media
    assert prompt["additional_information"]["hidden_states"]["layers"][0] is media
    assert MINIMAX_H3_ENCODER_REQUEST_KEY in prompt["additional_information"]["meta"]


def test_connector_handoff_without_inline_output_cleans_encoder_only_information():
    prompt = {
        "prompt": "hello",
        "multi_modal_data": {"image": object()},
        "model_intermediate_buffer": {"private": object()},
        "additional_information": {
            "hidden_states": {"layers": {0: torch.zeros(1)}},
            "meta": {MINIMAX_H3_ENCODER_REQUEST_KEY: {"task": "t2va"}},
        },
    }
    result = encoder2diffusion([_source_output(None)], prompt)
    assert result == {"prompt": "hello", "multi_modal_data": None, "additional_information": {}}


@pytest.mark.parametrize("payload", [None, _encoder_output()])
@pytest.mark.parametrize("request_id", ["other", ["other"], ("other",)])
def test_stage_wire_rejects_request_id_mismatch(payload, request_id):
    prompt = {"prompt": "hello", "additional_information": {"global_request_id": request_id}}
    with pytest.raises(RuntimeError, match="request ID does not match"):
        encoder2diffusion([_source_output(payload)], prompt)


@pytest.mark.parametrize("outputs", [None, [], (), [SimpleNamespace(), SimpleNamespace()]])
def test_stage_wire_rejects_invalid_completion_count(outputs):
    with pytest.raises(RuntimeError, match="exactly one completion"):
        encoder2diffusion([SimpleNamespace(outputs=outputs)], {"prompt": "hello"})


def test_stage_wire_empty_sources_and_invalid_prompt():
    assert encoder2diffusion([], {"prompt": "hello"}) is None
    with pytest.raises(TypeError, match="invalid MiniMax H3 prompt type"):
        encoder2diffusion([_source_output(None)], object())
