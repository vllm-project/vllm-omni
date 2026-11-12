# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax-H3 keep_steps: CodeVideoModel-style early-only video conditioning."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3
from vllm_omni.diffusion.models.minimax_h3.denoise_loop import (
    MiniMaxH3DenoiseBranch,
    minimax_h3_denoise_loop,
    minimax_h3_switch_denoise_branch,
)
from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
    minimax_h3_packed_sequence_ref2va_blocks,
)
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
    MINIMAX_H3_FPS,
    MINIMAX_H3_SHAPE_PLANNER,
    MiniMaxH3Pipeline,
)
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.models.minimax_h3.conditioning import (
    MiniMaxH3EncoderConditioning,
    MiniMaxH3EncoderMediaConditioning,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

LATENT_H = 4
LATENT_W = 6
IMAGE_SHAPE = (1, LATENT_H, LATENT_W)  # 1 * 2 * 3 = 6 rows
VIDEO_SHAPE = (2, LATENT_H, LATENT_W)  # 2 * 2 * 3 = 12 rows
BASE_IMAGE_ROWS = 6
NUM_FRAMES = 22
# 107 = 17n+5 frame grid; 107/24s ~ 4.46s clears the 4s minimum output the
# real request path validates (tests below that never pass through it use
# NUM_FRAMES to keep the toy layouts small).
REQUEST_PATH_NUM_FRAMES = 107
SIGMAS_VIDEO = [1.0, 0.6, 0.3, 0.0]
SIGMAS_AUDIO = [1.0, 0.5, 0.25, 0.0]


def _ref_blocks():
    return [
        {"kind": "image", "latent_h": LATENT_H, "latent_w": LATENT_W},
        {"kind": "video", "ref_audio_t": 0, "latent_t": 2, "latent_h": LATENT_H, "latent_w": LATENT_W},
    ]


def _full_packed():
    return minimax_h3_packed_sequence_ref2va_blocks(
        text_len=2,
        latent_t=2,
        latent_h=LATENT_H,
        latent_w=LATENT_W,
        audio_t=3,
        ref_blocks=_ref_blocks(),
    )


def _base_packed():
    return minimax_h3_packed_sequence_ref2va_blocks(
        text_len=3,
        latent_t=2,
        latent_h=LATENT_H,
        latent_w=LATENT_W,
        audio_t=3,
        ref_blocks=[{"kind": "image", "latent_h": LATENT_H, "latent_w": LATENT_W}],
    )


def _branch(packed, text_len):
    tags = packed["token_tags"].clone()
    tags[packed["text_pos"]] = torch.full((text_len,), 7, dtype=torch.long)
    return MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.zeros(text_len, 4),
        token_tags=tags,
        device=torch.device("cpu"),
    )


def _full_branch():
    return _branch(_full_packed(), 2)


def _base_branch():
    return _branch(_base_packed(), 3)


def _visual_rows():
    return torch.arange(18 * 96, dtype=torch.float32).reshape(18, 96) / 100


def _base_cond_rows():
    return torch.arange(BASE_IMAGE_ROWS * 96, dtype=torch.float32).reshape(BASE_IMAGE_ROWS, 96) / 50


class _RecordingModel:
    """Deterministic DiT stand-in recording the packed length of each call."""

    def __init__(self):
        self.seq_lens = []

    def __call__(self, **forward_kwargs):
        self.seq_lens.append(int(forward_kwargs["x"].shape[1]))
        img_positions = forward_kwargs["img_pos_info"]["position_ids"].long()
        audio_positions = forward_kwargs["audio_pos_info"]["position_ids"].long()
        video_velocity = forward_kwargs["x"][0, img_positions] * 0.5 + 0.25
        audio_velocity = forward_kwargs["audio_x"][0, audio_positions] * 0.25 + 0.125
        return video_velocity, audio_velocity


# ---------------------------------------------------------------------------
# Switch helper
# ---------------------------------------------------------------------------


def test_switch_moves_target_rows_and_re_pins_anchor():
    full, base = _full_branch(), _base_branch()
    rows = torch.randn(full.img_pos.shape[0], 96)
    anchor = _base_cond_rows()

    switched, update, cond_anchor = minimax_h3_switch_denoise_branch(
        base=base,
        video_rows=rows,
        update=full.update_mask_dev,
        base_cond_rows=anchor,
        device=torch.device("cpu"),
    )

    assert update is base.update_mask_dev
    torch.testing.assert_close(cond_anchor, anchor)
    assert switched.shape == (base.img_pos.shape[0], 96)
    torch.testing.assert_close(switched[base.update_mask_dev], rows[full.update_mask_dev])
    torch.testing.assert_close(switched[~base.update_mask_dev], anchor)
    # The running rows must not alias the full-layout tensor.
    assert switched.data_ptr() != rows.data_ptr()


def test_switch_rejects_target_row_count_mismatch():
    full, base = _full_branch(), _base_branch()
    short_update = full.update_mask_dev.clone()
    # ref2va packs condition rows before targets, so index 0 is already False
    # and clearing it would not change the count; drop a trailing target row.
    short_update[-1] = False
    with pytest.raises(ValueError, match="target row count"):
        minimax_h3_switch_denoise_branch(
            base=base,
            video_rows=torch.zeros(full.img_pos.shape[0], 96),
            update=short_update,
            base_cond_rows=_base_cond_rows(),
            device=torch.device("cpu"),
        )


# ---------------------------------------------------------------------------
# _build_denoise_inputs
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline():
    instance = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(instance)
    instance.device = torch.device("cpu")
    return instance


def _inputs_kwargs():
    return dict(
        task="ref2va",
        text_embeddings=torch.zeros(2, 4),
        text_tags=torch.ones(2, dtype=torch.long),
        seed=42,
        latent_t=2,
        latent_h=LATENT_H,
        latent_w=LATENT_W,
        audio_t=3,
        num_frames=NUM_FRAMES,
        num_steps=3,
        video_shift=12.0,
        audio_shift=3.0,
        base_schedule=None,
        visual_condition=_visual_rows(),
        visual_condition_shape=None,
        audio_condition=None,
        ref_audio_t=None,
        ref_blocks=_ref_blocks(),
        visual_condition_shapes=[IMAGE_SHAPE, VIDEO_SHAPE],
    )


def _keep_steps_kwargs():
    return {
        **_inputs_kwargs(),
        "keep_steps": 2,
        "base_text_embeddings": torch.zeros(3, 4),
        "base_text_tags": torch.ones(3, dtype=torch.long),
    }


def test_default_inputs_build_no_base_branch(pipeline):
    inputs = pipeline._build_denoise_inputs(**_inputs_kwargs())
    assert inputs["keep_steps"] is None
    assert inputs["base_branch"] is None
    assert inputs["base_cond_rows"] is None


def test_keep_steps_builds_matching_base_layout(pipeline):
    inputs = pipeline._build_denoise_inputs(**_keep_steps_kwargs())
    branch, base = inputs["branch"], inputs["base_branch"]
    assert inputs["keep_steps"] == 2
    assert int(base.update_mask.sum()) == int(branch.update_mask.sum())
    assert int(base.audio_pos.shape[0]) == int(branch.audio_pos.shape[0])
    # Both toy layouts round up to the same 64-row alignment bucket, so the
    # padded seq_len cannot witness the dropped silent video reference; the
    # real document length and the img-row count can.
    assert base.used_len < branch.used_len
    assert int(branch.img_pos.shape[0]) - int(base.img_pos.shape[0]) == (
        VIDEO_SHAPE[0] * (VIDEO_SHAPE[1] // 2) * (VIDEO_SHAPE[2] // 2)
    )
    assert tuple(inputs["base_cond_rows"].shape) == (BASE_IMAGE_ROWS, 96)
    # The base layout's non-update image rows are exactly the image condition
    # rows: audio references and guides are rejected, so the anchor prefix
    # covers precisely the image conditions.
    assert int((~base.update_mask).sum()) == BASE_IMAGE_ROWS
    assert int((~branch.update_mask).sum()) == BASE_IMAGE_ROWS + 12


def test_base_cond_rows_equal_image_only_anchor(pipeline):
    """The base anchor is the full request's anchor with the video rows dropped.

    The noise draw depends on the request's condition count (each condition
    draws [1, 24, target_t + num_conditions, h, w] under its own seed), so an
    anchor built from a standalone image-only presentation draws a different
    RNG stream than the full request. The production switch keeps the running
    rows on the original full anchor and only re-pins the condition rows to
    its image-row prefix, so the exact expectation is that prefix of the full
    anchor — asserted against the row count the base layout derives
    independently.
    """
    inputs = pipeline._build_denoise_inputs(**_keep_steps_kwargs())
    branch, base = inputs["branch"], inputs["base_branch"]
    assert int((~base.update_mask).sum()) == BASE_IMAGE_ROWS
    torch.testing.assert_close(
        inputs["base_cond_rows"],
        inputs["cond_anchor"][:BASE_IMAGE_ROWS],
    )


@pytest.mark.parametrize(
    "override,match",
    [
        ({"task": "t2va"}, "requires task ref2va"),
        ({"guide_blocks": [{"kind": "image", "frame_index": 0, "latent_h": 4, "latent_w": 6}]}, "timeline guides"),
        ({"base_text_embeddings": None}, "requires base text conditioning"),
        ({"base_text_tags": None}, "requires base text conditioning"),
        # The shapes keep describing both retained condition rows, so the
        # noise-aug row check passes and the keep_steps guard is what rejects
        # the missing video reference.
        (
            {"ref_blocks": [_ref_blocks()[0]], "visual_condition_shapes": [IMAGE_SHAPE, VIDEO_SHAPE]},
            "silent video reference",
        ),
        # The packer requires images before the video block; the encoder and
        # the API layer normalize upload order into exactly this order.
        (
            {"ref_blocks": [_ref_blocks()[1], _ref_blocks()[0]], "visual_condition_shapes": [IMAGE_SHAPE, VIDEO_SHAPE]},
            "silent video reference",
        ),
        ({"visual_condition": None, "visual_condition_shapes": None}, "silent video reference"),
    ],
)
def test_keep_steps_build_rejects_incompatible_layouts(pipeline, override, match):
    with pytest.raises(ValueError, match=match):
        pipeline._build_denoise_inputs(**{**_keep_steps_kwargs(), **override})


# ---------------------------------------------------------------------------
# Denoise loop
# ---------------------------------------------------------------------------


def _run_loop(
    model,
    *,
    positive,
    keyframe,
    initial_rows,
    base=None,
    keep_steps=None,
    sigmas=None,
    audio_sigmas=None,
    on_step=None,
    initial_audio_rows=None,
):
    return minimax_h3_denoise_loop(
        model=model,
        positive=positive,
        initial_video_rows=initial_rows,
        initial_audio_rows=(
            torch.zeros(positive.audio_pos.shape[0], 32)
            if initial_audio_rows is None
            else initial_audio_rows
        ),
        keyframe_cond_rows=keyframe,
        sigmas_video=SIGMAS_VIDEO if sigmas is None else sigmas,
        sigmas_audio=SIGMAS_AUDIO if audio_sigmas is None else audio_sigmas,
        device=torch.device("cpu"),
        base=base,
        keep_steps=keep_steps,
        base_cond_rows=_base_cond_rows() if base is not None else None,
        update_progress=on_step,
    )


def test_keep_steps_loop_decomposes_bit_exact():
    full, base = _full_branch(), _base_branch()
    full_anchor = torch.randn((~full.update_mask).sum(), 96)
    initial = torch.randn(full.img_pos.shape[0], 96)
    base_anchor = _base_cond_rows()

    switched_model = _RecordingModel()
    switched_rows, switched_audio = _run_loop(
        switched_model,
        positive=full,
        keyframe=full_anchor,
        initial_rows=initial,
        base=base,
        keep_steps=2,
    )

    # Phase check: steps before the switch use the full layout, later steps the
    # shorter image-only layout.
    assert len(switched_model.seq_lens) == 3
    assert switched_model.seq_lens[0] == switched_model.seq_lens[1] == full.seq_len
    assert switched_model.seq_lens[2] == base.seq_len

    captured = {}

    def capture(step, video_rows, audio_rows):
        captured[step] = (video_rows.clone(), audio_rows.clone())

    prefix_model = _RecordingModel()
    prefix_rows, prefix_audio = _run_loop(
        prefix_model,
        positive=full,
        keyframe=full_anchor,
        initial_rows=initial,
        sigmas=SIGMAS_VIDEO[:3],
        audio_sigmas=SIGMAS_AUDIO[:3],
        on_step=capture,
    )
    manual_rows, _, manual_anchor = minimax_h3_switch_denoise_branch(
        base=base,
        video_rows=prefix_rows,
        update=full.update_mask_dev,
        base_cond_rows=base_anchor,
        device=torch.device("cpu"),
    )
    tail_model = _RecordingModel()
    manual_final, manual_audio = _run_loop(
        tail_model,
        positive=base,
        keyframe=manual_anchor,
        initial_rows=manual_rows,
        # The switch only moves video rows; the manual tail must carry the
        # prefix run's audio state, exactly as the in-loop switch does.
        initial_audio_rows=prefix_audio,
        sigmas=SIGMAS_VIDEO[2:],
        audio_sigmas=SIGMAS_AUDIO[2:],
    )

    assert torch.equal(switched_rows, manual_final)
    assert torch.equal(switched_audio, manual_audio)
    assert torch.equal(captured[1][0], prefix_rows)


def test_default_loop_never_calls_the_switch(monkeypatch: pytest.MonkeyPatch):
    def forbidden(**kwargs):
        raise AssertionError("the default path must never switch branches")

    monkeypatch.setattr(
        "vllm_omni.diffusion.models.minimax_h3.denoise_loop.minimax_h3_switch_denoise_branch", forbidden
    )
    full = _full_branch()
    anchor = torch.randn((~full.update_mask).sum(), 96)
    rows, audio = _run_loop(
        _RecordingModel(),
        positive=full,
        keyframe=anchor,
        initial_rows=torch.randn(full.img_pos.shape[0], 96),
    )
    assert rows.shape[0] == full.img_pos.shape[0]
    assert audio.shape[0] == full.audio_pos.shape[0]


def test_keep_steps_loop_rejects_invalid_switch_point():
    full, base = _full_branch(), _base_branch()
    for keep_steps in (-1, 0, 3, "2"):
        with pytest.raises(ValueError, match="keep_steps"):
            _run_loop(
                _RecordingModel(),
                positive=full,
                keyframe=torch.randn((~full.update_mask).sum(), 96),
                initial_rows=torch.randn(full.img_pos.shape[0], 96),
                base=base,
                keep_steps=keep_steps,
            )


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


def _validation_pipeline():
    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.partition = "combined"
    pipeline.supported_tasks = frozenset({"t2va", "fl2va", "ref2va"})
    pipeline.default_video_shift = 12.0
    pipeline.default_audio_shift = 3.0
    pipeline.device = torch.device("cpu")
    pipeline.od_config = SimpleNamespace()
    pipeline._base_schedule_by_partition = {"t2va": None, "fl2va": None, "ref2va": None}
    pipeline.load_text_encoder = False
    pipeline.load_vae_encoder = False
    pipeline._quality_policy = Mock()
    pipeline._quality_policy.resolve.return_value = SimpleNamespace(cache_dit=None)
    pipeline._cache_dit_runtime = SimpleNamespace(prepare=lambda spec: None)
    pipeline._turbo_lora_specs = {}
    pipeline._native_lora_adapter_ids = set()
    pipeline._lora_sigma_schedules = {}
    pipeline.transformer = None
    return pipeline


def _keep_steps_conditioning(**overrides):
    kwargs = dict(
        hidden_states=torch.ones(2, 8, dtype=torch.bfloat16),
        token_tags=torch.ones(2, dtype=torch.long),
        task="ref2va",
        height=64,
        width=64,
        num_frames=NUM_FRAMES,
        latent_t=MINIMAX_H3_SHAPE_PLANNER.video_latent_t(NUM_FRAMES),
        audio_t=MINIMAX_H3_SHAPE_PLANNER.audio_latent_t(NUM_FRAMES / MINIMAX_H3_FPS),
        ref_blocks=(
            {"kind": "image", "latent_h": LATENT_H, "latent_w": LATENT_W},
            {"kind": "video", "ref_audio_t": 0, "latent_t": 2, "latent_h": LATENT_H, "latent_w": LATENT_W},
        ),
        visual_condition_shapes=(IMAGE_SHAPE, VIDEO_SHAPE),
        base_hidden_states=torch.ones(3, 8, dtype=torch.bfloat16),
        base_token_tags=torch.ones(3, dtype=torch.long),
    )
    return MiniMaxH3EncoderConditioning(**{**kwargs, **overrides})


def _sampling(**extra_args):
    return OmniDiffusionSamplingParams(
        quality="lossless",
        fps=24,
        num_inference_steps=4,
        extra_args={"task": "ref2va", **extra_args},
    )


def test_keep_steps_context_carries_base_presentation():
    pipeline = _validation_pipeline()
    context = pipeline._prepare_encoder_conditioning_inputs(_keep_steps_conditioning(), _sampling(keep_steps=2))
    assert context["keep_steps"] == 2
    assert context["base_text_embeddings"] is not None
    assert context["base_text_tags"] is not None
    assert context["base_text_embeddings"].dtype == torch.bfloat16
    assert context["base_text_tags"].dtype == torch.long


def test_keep_steps_equal_to_step_count_degrades_to_full_video():
    pipeline = _validation_pipeline()
    context = pipeline._prepare_encoder_conditioning_inputs(_keep_steps_conditioning(), _sampling(keep_steps=4))
    assert context["keep_steps"] is None
    assert context["base_text_embeddings"] is None
    assert context["base_text_tags"] is None


def test_degenerate_keep_steps_skips_switch_specific_restrictions():
    """``== num_steps`` normalizes away before the switch-only checks, so a
    t2va request carrying the no-op value runs as plain t2va instead of
    being rejected (the original degeneration semantics)."""
    pipeline = _validation_pipeline()
    context = pipeline._prepare_encoder_conditioning_inputs(
        _keep_steps_conditioning(task="t2va"), _sampling(task="t2va", keep_steps=4)
    )
    assert context["keep_steps"] is None


def test_keep_steps_rejects_pinned_distilled_schedule():
    pipeline = _validation_pipeline()
    pipeline._base_schedule_by_partition["ref2va"] = SimpleNamespace(
        num_inference_steps=4, base_schedule=(1.0, 0.6, 0.0)
    )
    with pytest.raises(OmniClientError, match="pinned distilled sigma schedules"):
        pipeline._prepare_encoder_conditioning_inputs(_keep_steps_conditioning(), _sampling(keep_steps=2))


@pytest.mark.parametrize(
    "conditioning_overrides,extra_args,match",
    [
        ({"task": "t2va"}, {"task": "t2va", "keep_steps": 2}, "requires task ref2va"),
        (
            {"audio_condition": torch.ones(4, 32), "audio_condition_lengths": (2,)},
            {"keep_steps": 2},
            "audio references",
        ),
        (
            {"audio_condition": torch.ones(4, 32), "audio_condition_lengths": (2,)},
            {"keep_steps": 2, "audio_mode": "lock_source"},
            "audio references",
        ),
        (dict(), {"keep_steps": 2, "pad_seq_len": 64}, "pad_seq_len"),
        (
            {"guide_blocks": ({"kind": "image", "frame_index": 0, "latent_h": 4, "latent_w": 6},)},
            {"keep_steps": 2},
            "timeline guides",
        ),
        ({"base_hidden_states": None, "base_token_tags": None}, {"keep_steps": 2}, "image-only"),
        (dict(), {"keep_steps": "2"}, "must be an integer"),
        # bool is an int subclass; type() rejects it on purpose.
        (dict(), {"keep_steps": True}, "must be an integer"),
        (dict(), {"keep_steps": 0}, "must be an integer"),
        (dict(), {"keep_steps": 5}, "must be an integer"),
        (dict(), {"keep_steps": -1}, "must be an integer"),
    ],
)
def test_keep_steps_request_validation_rejects_unsupported_combinations(
    conditioning_overrides, extra_args, match
):
    pipeline = _validation_pipeline()
    conditioning = _keep_steps_conditioning(**conditioning_overrides)
    with pytest.raises(OmniClientError, match=match):
        pipeline._prepare_encoder_conditioning_inputs(conditioning, _sampling(**extra_args))


def test_keep_steps_rejects_latent_offload_features(monkeypatch: pytest.MonkeyPatch):
    pipeline = _validation_pipeline()
    pipeline.od_config.kv_offload_config = "tier2"
    with pytest.raises(OmniClientError, match="reference-KV offload"):
        pipeline._prepare_encoder_conditioning_inputs(_keep_steps_conditioning(), _sampling(keep_steps=2))

    pipeline = _validation_pipeline()
    resolver = Mock()
    monkeypatch.setattr(pipeline, "_resolve_latent_continuation", resolver)
    with pytest.raises(OmniClientError, match="continue_from"):
        pipeline._prepare_encoder_conditioning_inputs(
            _keep_steps_conditioning(), _sampling(keep_steps=2, continue_from="job-1")
        )
    # The rejection fires in the shared helper before continuation work, so
    # the resolver must not have run; if the check ever moves back behind it,
    # this assertion fails and keeps the front-loading honest.
    resolver.assert_not_called()

    pipeline = _validation_pipeline()
    monkeypatch.setattr(pipeline, "_resolve_latent_refine", lambda extra: SimpleNamespace())
    with pytest.raises(OmniClientError, match="latent refine/upscale"):
        pipeline._prepare_encoder_conditioning_inputs(_keep_steps_conditioning(), _sampling(keep_steps=2))

    pipeline = _validation_pipeline()
    monkeypatch.setattr(
        pipeline_minimax_h3,
        "require_request_attention_schedule_fits",
        lambda sampling, schedule, num_steps: SimpleNamespace(),
    )
    with pytest.raises(OmniClientError, match="attention_schedule"):
        pipeline._prepare_encoder_conditioning_inputs(_keep_steps_conditioning(), _sampling(keep_steps=2))


# ---------------------------------------------------------------------------
# Encoder wire
# ---------------------------------------------------------------------------


def _wire_conditioning():
    return MiniMaxH3EncoderConditioning(
        hidden_states=torch.ones(2, 5120, dtype=torch.bfloat16),
        token_tags=torch.tensor([0, 1]),
        task="ref2va",
        height=768,
        width=1344,
        num_frames=124,
        latent_t=37,
        audio_t=207,
        visual_condition=torch.arange(96, dtype=torch.float32).reshape(1, 96),
        visual_condition_shapes=((1, 2, 2),),
        audio_condition=torch.ones(160, 32),
        audio_condition_lengths=(80,),
        ref_blocks=(
            {"kind": "video_audio", "ref_audio_t": 80, "latent_t": 1, "latent_h": 2, "latent_w": 2},
        ),
        keyframe_frame_indices=(0, -1),
        base_hidden_states=torch.arange(6, dtype=torch.float32).reshape(3, 2),
        base_token_tags=torch.tensor([0, 0, 1]),
    )


def test_wire_payload_round_trips_base_presentation():
    conditioning = _wire_conditioning()
    payload = conditioning.to_omni_payload()
    assert torch.equal(payload["base_hidden_states"], conditioning.base_hidden_states)
    assert torch.equal(payload["base_token_tags"], conditioning.base_token_tags)

    restored = MiniMaxH3EncoderConditioning.from_omni_payload(payload)
    torch.testing.assert_close(restored.base_hidden_states, conditioning.base_hidden_states)
    torch.testing.assert_close(restored.base_token_tags, conditioning.base_token_tags)


def test_legacy_wire_payload_decodes_base_fields_to_none():
    payload = _wire_conditioning().to_omni_payload()
    legacy = {key: value for key, value in payload.items() if not key.startswith("base_")}
    restored = MiniMaxH3EncoderConditioning.from_omni_payload(legacy)
    assert restored.base_hidden_states is None
    assert restored.base_token_tags is None


# ---------------------------------------------------------------------------
# Local encoder second pass
# ---------------------------------------------------------------------------


@dataclass
class _Prepared:
    prompt: str
    images: list
    qwen_videos: list
    video_timestamps: list
    condition_labels: list
    media: Any


def _local_pipeline(monkeypatch: pytest.MonkeyPatch):
    """Patch a bare pipeline so ``_prepare_local_conditioning`` runs without models."""
    pipeline = _validation_pipeline()
    prepared = _Prepared(
        prompt="p",
        images=[object()],
        qwen_videos=[object()],
        video_timestamps=[[0.0]],
        condition_labels=[("image", 1), ("video", 1)],
        media=SimpleNamespace(num_frames=NUM_FRAMES, height=64, width=64),
    )
    media_conditioning = MiniMaxH3EncoderMediaConditioning(
        task="ref2va",
        height=64,
        width=64,
        num_frames=NUM_FRAMES,
        latent_t=MINIMAX_H3_SHAPE_PLANNER.video_latent_t(NUM_FRAMES),
        audio_t=MINIMAX_H3_SHAPE_PLANNER.audio_latent_t(NUM_FRAMES / MINIMAX_H3_FPS),
        visual_condition=None,
        visual_condition_shapes=(),
        audio_condition=None,
        audio_condition_lengths=(),
    )
    monkeypatch.setattr(pipeline, "_extract_prompt", lambda raw: ("p", {}))
    monkeypatch.setattr(pipeline, "_active_turbo_spec", lambda sampling: None)
    monkeypatch.setattr(pipeline, "_has_active_native_lora", lambda sampling: False)
    monkeypatch.setattr(pipeline, "_resolve_task", lambda task, multi_modal_data=None, **kwargs: "ref2va")
    monkeypatch.setattr(pipeline, "_extract_text_conditioning", lambda raw: None)
    monkeypatch.setattr(pipeline, "_encode_local_media", lambda media: media_conditioning)
    monkeypatch.setattr(pipeline_minimax_h3, "prepare_encoder_inputs", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(pipeline_minimax_h3, "resolve_continuation", lambda extra_args, **kwargs: None)
    # The conditioning contract validates the real 5120-wide Qwen3-VL hidden
    # size (MiniMaxH3TextConditioning.from_payload), so the mocks must produce
    # it: narrower tensors would trip that validation instead of exercising
    # the behaviour under test.
    encoded = [
        (torch.ones(2, 5120, dtype=torch.bfloat16), torch.ones(2, dtype=torch.long)),
        (torch.full((3, 5120), 2.0, dtype=torch.bfloat16), torch.ones(3, dtype=torch.long)),
    ]
    encode_prompt = Mock(side_effect=encoded)
    monkeypatch.setattr(pipeline, "encode_prompt", encode_prompt)
    # Mock turns a list side_effect into an iterator, so the expected values
    # travel back separately for the assertions.
    return pipeline, prepared, encode_prompt, encoded


def _local_sampling(**extra_args):
    return OmniDiffusionSamplingParams(
        quality="lossless",
        fps=24,
        num_inference_steps=4,
        extra_args={"task": "ref2va", **extra_args},
    )


def test_local_conditioning_encodes_image_only_presentation(monkeypatch: pytest.MonkeyPatch):
    pipeline, prepared, encode_prompt, encoded = _local_pipeline(monkeypatch)
    conditioning, window = pipeline._prepare_local_conditioning({"prompt": "p"}, _local_sampling(keep_steps=2))

    assert window is None
    assert encode_prompt.call_count == 2
    torch.testing.assert_close(conditioning.hidden_states, encoded[0][0])
    torch.testing.assert_close(conditioning.base_hidden_states, encoded[1][0])
    torch.testing.assert_close(conditioning.base_token_tags, encoded[1][1])
    base_prepared = encode_prompt.call_args_list[1].args[0]
    assert base_prepared is not prepared
    assert base_prepared.qwen_videos == []
    assert base_prepared.video_timestamps == []
    assert base_prepared.condition_labels == [("image", 1)]
    assert base_prepared.prompt == prepared.prompt


def test_degenerate_keep_steps_skips_the_second_encoder_pass(monkeypatch: pytest.MonkeyPatch):
    # keep_steps == num_inference_steps degrades to full-video conditioning
    # during normalization, so the image-only presentation is never encoded.
    pipeline, _, encode_prompt, _ = _local_pipeline(monkeypatch)
    conditioning, _ = pipeline._prepare_local_conditioning({"prompt": "p"}, _local_sampling(keep_steps=4))
    assert encode_prompt.call_count == 1
    assert conditioning.base_hidden_states is None
    assert conditioning.base_token_tags is None


def test_invalid_keep_steps_fails_before_any_encoder_pass(monkeypatch: pytest.MonkeyPatch):
    pipeline, _, encode_prompt, _ = _local_pipeline(monkeypatch)
    with pytest.raises(OmniClientError, match="must be an integer"):
        pipeline._prepare_local_conditioning({"prompt": "p"}, _local_sampling(keep_steps=9))
    assert encode_prompt.call_count == 0


@pytest.mark.parametrize(
    "extra_args,config_patch,active_resolver,expected_match",
    [
        ({"keep_steps": 2}, {"kv_offload_config": "tier2"}, None, "reference-KV offload"),
        ({"keep_steps": 2, "pad_seq_len": 64}, dict(), None, "pad_seq_len"),
        ({"keep_steps": 2}, {"step_execution": True}, None, "step execution"),
        (
            {"keep_steps": 2, "latent_refine": {"steps": 4}},
            dict(),
            "refine",
            "latent refine/upscale",
        ),
        (
            {"keep_steps": 2, "latent_upscale": {"latent_height": 48, "latent_width": 48}},
            dict(),
            "upscale",
            "latent refine/upscale",
        ),
    ],
)
def test_config_incompatible_keep_steps_fails_before_any_encoder_pass(
    monkeypatch, extra_args, config_patch, active_resolver, expected_match
):
    """Restrictions decidable from the config alone reject before any Qwen pass."""
    pipeline, _, encode_prompt, _ = _local_pipeline(monkeypatch)
    for key, value in config_patch.items():
        setattr(pipeline.od_config, key, value)
    # Only the resolver under test returns a resolved spec — the helper's
    # `or` chain would otherwise short-circuit on the other one and the case
    # would not prove its own rejection path.
    refine = Mock(return_value=SimpleNamespace(strength=0.2) if active_resolver == "refine" else None)
    upscale = Mock(
        return_value=SimpleNamespace(latent_height=1, latent_width=1) if active_resolver == "upscale" else None
    )
    monkeypatch.setattr(pipeline, "_resolve_latent_refine", refine)
    monkeypatch.setattr(pipeline, "_resolve_latent_upscale", upscale)
    with pytest.raises(OmniClientError, match=expected_match):
        pipeline._prepare_local_conditioning({"prompt": "p"}, _local_sampling(**extra_args))
    assert encode_prompt.call_count == 0
    if active_resolver == "refine":
        refine.assert_called_once()
    if active_resolver == "upscale":
        upscale.assert_called_once()


def test_local_continue_from_fails_before_any_encoder_pass(monkeypatch: pytest.MonkeyPatch):
    """The local path rejects continue_from before the second Qwen pass too."""
    pipeline, _, encode_prompt, _ = _local_pipeline(monkeypatch)
    resolve_continuation = Mock(return_value=None)
    monkeypatch.setattr(pipeline_minimax_h3, "resolve_continuation", resolve_continuation)
    with pytest.raises(OmniClientError, match="continue_from"):
        pipeline._prepare_local_conditioning({"prompt": "p"}, _local_sampling(keep_steps=2, continue_from="job-1"))
    assert encode_prompt.call_count == 0


def test_local_long_video_continuation_fails_before_any_encoder_pass(monkeypatch: pytest.MonkeyPatch):
    """keep_steps plus a long-video continuation rejects with zero Qwen passes.

    request_context={} is what makes _prepare_local_conditioning resolve the
    continuation at all (it only calls resolve_continuation when prompts or a
    ref2va request_context is present, matching _prepare_request_inputs); the
    stub then mirrors the real resolver's long-video return (upstream
    contract: (277, 22)) so the helper's continuation rejection fires.
    """
    pipeline, _, encode_prompt, _ = _local_pipeline(monkeypatch)
    resolve_continuation = Mock(return_value=(277, 22))
    monkeypatch.setattr(pipeline_minimax_h3, "resolve_continuation", resolve_continuation)
    with pytest.raises(OmniClientError, match="does not support continuation"):
        pipeline._prepare_local_conditioning(
            {"prompt": "p"}, _local_sampling(keep_steps=2, long_video=True), request_context={}
        )
    resolve_continuation.assert_called_once()
    assert encode_prompt.call_count == 0


def test_local_attention_schedule_fails_before_any_encoder_pass(monkeypatch: pytest.MonkeyPatch):
    """keep_steps plus an attention schedule rejects with zero Qwen passes."""
    pipeline, _, encode_prompt, _ = _local_pipeline(monkeypatch)
    monkeypatch.setattr(
        pipeline_minimax_h3,
        "require_request_attention_schedule_fits",
        lambda sampling, schedule, num_steps: SimpleNamespace(),
    )
    with pytest.raises(OmniClientError, match="attention_schedule"):
        pipeline._prepare_local_conditioning({"prompt": "p"}, _local_sampling(keep_steps=2))
    assert encode_prompt.call_count == 0


def test_local_turbo_and_native_lora_fail_before_any_encoder_pass(monkeypatch: pytest.MonkeyPatch):
    """keep_steps plus an active turbo or native LoRA adapter rejects with zero Qwen passes.

    The artifact validators run before the keep_steps helper on this path, so
    they are stubbed too: the turbo case's empty spec would otherwise crash
    _validate_turbo_sampling on its attributes, and the native case would hit
    the validator's task-mismatch error before the rejection under test.
    """
    pipeline, _, encode_prompt, _ = _local_pipeline(monkeypatch)
    monkeypatch.setattr(pipeline, "_validate_turbo_sampling", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "_validate_native_sampling", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "_active_turbo_spec", lambda sampling: SimpleNamespace())
    with pytest.raises(OmniClientError, match="turbo or native LoRA"):
        pipeline._prepare_local_conditioning({"prompt": "p"}, _local_sampling(keep_steps=2))
    assert encode_prompt.call_count == 0

    pipeline, _, encode_prompt, _ = _local_pipeline(monkeypatch)
    monkeypatch.setattr(pipeline, "_validate_turbo_sampling", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "_validate_native_sampling", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "_has_active_native_lora", lambda sampling: True)
    with pytest.raises(OmniClientError, match="turbo or native LoRA"):
        pipeline._prepare_local_conditioning({"prompt": "p"}, _local_sampling(keep_steps=2))
    assert encode_prompt.call_count == 0


def test_plain_requests_never_touch_the_keep_steps_resolver(monkeypatch: pytest.MonkeyPatch):
    """Requests without keep_steps keep their original code path untouched."""
    pipeline, _, encode_prompt, _ = _local_pipeline(monkeypatch)
    resolver = Mock()
    monkeypatch.setattr(pipeline, "_resolve_sigma_positions", resolver)
    conditioning, _ = pipeline._prepare_local_conditioning({"prompt": "p"}, _local_sampling())
    resolver.assert_not_called()
    assert encode_prompt.call_count == 1
    assert conditioning.base_hidden_states is None


# ---------------------------------------------------------------------------
# Reference-order normalization through the real request path
# ---------------------------------------------------------------------------


def test_request_path_normalizes_video_first_references(monkeypatch: pytest.MonkeyPatch):
    """Video uploaded before the image still packs image labels first.

    This is the invariant the switch relies on: the image rows must be the
    prefix of the full layout's condition rows. The loader layer is stubbed;
    everything else (reference assembly, label ordering) is the real code.
    """
    import numpy as np
    from PIL import Image

    from vllm_omni.model_executor.models.minimax_h3 import encoder_processing

    monkeypatch.setattr(
        encoder_processing, "load_minimax_h3_images", lambda values: [Image.new("RGB", (64, 64))]
    )
    monkeypatch.setattr(
        encoder_processing,
        "prepare_reference_videos",
        lambda videos, **kwargs: [
            {
                "original_path": "v.mp4",
                "prepared_path": "v.mp4",
                "input_has_audio": False,
                "duration_seconds": 5.0,
                "audio_duration_seconds": None,
                "start_time_seconds": 0.0,
            }
        ],
    )
    monkeypatch.setattr(
        encoder_processing, "load_video_frames", lambda path: np.zeros((4, 8, 8, 3), dtype=np.uint8)
    )
    monkeypatch.setattr(
        encoder_processing,
        "sample_reference_video_frames",
        lambda path, decoded_frames: {
            "block_timestamps": [0.0, 1.0],
            "frames": [np.zeros((8, 8, 3), dtype=np.uint8)] * 2,
        },
    )
    prepared = encoder_processing.prepare_encoder_inputs(
        {"prompt": "p", "multi_modal_data": {"video": [b"v"], "image": [b"i"]}},
        OmniDiffusionSamplingParams(
            quality="lossless",
            fps=24,
            num_inference_steps=4,
            height=64,
            width=64,
            num_frames=REQUEST_PATH_NUM_FRAMES,
            extra_args={"task": "ref2va"},
        ),
        task="ref2va",
    )
    assert prepared.condition_labels == [("image", 1), ("video", 1)]
    assert len(prepared.images) == 1
    assert len(prepared.qwen_videos) == 1


# ---------------------------------------------------------------------------
# Separate-encoder-stage intake
# ---------------------------------------------------------------------------


def test_encoder_stage_rejects_keep_steps_before_encoder_preparation(monkeypatch: pytest.MonkeyPatch):
    """Split-stage clients get the 400 before any encoder-stage work runs."""
    from vllm_omni.model_executor.stage_input_processors import minimax_h3 as processor

    def forbidden(*args, **kwargs):
        raise AssertionError("keep_steps must be rejected before any encoder-stage preparation")

    # If the intake check is bypassed or reordered, the media-preparation
    # call fires and this bomb fails the test.
    monkeypatch.setattr(processor, "prepare_encoder_inputs", forbidden)
    with pytest.raises(OmniClientError, match="separate encoder stage") as exc_info:
        processor.prepare_encoder_prompt({"prompt": "p"}, [_local_sampling(keep_steps=2)])
    assert exc_info.value.status_code == 400


def test_encoder_stage_passes_degenerate_keep_steps(monkeypatch: pytest.MonkeyPatch):
    """``keep_steps == num_inference_steps`` is a no-op and must pass intake.

    Split-stage deployments keep the combined deployment's contract: the
    degenerate value means "keep the video for every step", so no encoder
    support is needed and the diffusion stage normalizes it away later.
    """
    from vllm_omni.model_executor.stage_input_processors import minimax_h3 as processor

    prepared = _Prepared(
        prompt="p",
        images=[],
        qwen_videos=[],
        video_timestamps=[],
        condition_labels=[],
        media=SimpleNamespace(
            task="ref2va",
            num_frames=NUM_FRAMES,
            to_mm_tensors=lambda: [],
            to_metadata=lambda: {"task": "ref2va"},
        ),
    )
    monkeypatch.setattr(processor, "prepare_encoder_inputs", lambda *args, **kwargs: prepared)
    transformed = processor.prepare_encoder_prompt({"prompt": "p"}, [_local_sampling(keep_steps=4)])
    assert transformed["prompt"] == "p"


def test_pinned_schedule_step_counts_diverge_between_intake_and_diffusion(monkeypatch: pytest.MonkeyPatch):
    """The documented split-stage divergence, pinned as behaviour.

    On a checkpoint whose distilled schedule pins 4 steps, ``keep_steps=4``
    with ``num_inference_steps`` unspecified is a no-op for the diffusion
    stage — the schedule resolver returns the pinned count and the degenerate
    value normalizes away — while the split-stage intake, which only sees the
    default step count of 50, conservatively rejects the same request before
    the encoder stage runs. An explicit matching step count passes intake.

    The intake rejection is the current conservative limitation, not a
    desired endpoint: intake cannot see the pinned step count (see the
    encoder-stage TODO in stage_input_processors/minimax_h3.py). If intake
    later learns it, the rejection assertion here should flip to a
    passthrough.
    """
    from vllm_omni.model_executor.stage_input_processors import minimax_h3 as processor
    from vllm_omni.model_executor.stage_input_processors.minimax_h3 import prepare_encoder_prompt

    # Diffusion side: the pinned schedule's own count (4) normalizes
    # keep_steps=4 away before the pinned-schedule rejection is reached.
    pipeline = _validation_pipeline()
    pipeline._base_schedule_by_partition["ref2va"] = SimpleNamespace(
        num_inference_steps=4, base_schedule=(1.0, 0.6, 0.3, 0.0)
    )
    # num_inference_steps is left unspecified so the case matches the
    # documented divergence scenario exactly: the resolver returns the
    # pinned schedule's own count (4).
    sampling = OmniDiffusionSamplingParams(
        quality="lossless", fps=24, extra_args={"task": "ref2va", "keep_steps": 4}
    )
    context = pipeline._prepare_encoder_conditioning_inputs(_keep_steps_conditioning(), sampling)
    assert context["keep_steps"] is None
    assert context["base_text_embeddings"] is None

    # Intake side: only the default step count (50) is visible, so the same
    # request looks like a switcher and is rejected at intake.
    prepared = SimpleNamespace(
        prompt="p",
        images=[],
        qwen_videos=[],
        video_timestamps=[],
        condition_labels=[],
        media=SimpleNamespace(task="ref2va", to_mm_tensors=lambda: [], to_metadata=lambda: {}),
    )
    prepare_encoder_inputs = Mock(return_value=prepared)
    monkeypatch.setattr(processor, "prepare_encoder_inputs", prepare_encoder_inputs)
    intake_sampling = OmniDiffusionSamplingParams(
        quality="lossless", fps=24, extra_args={"task": "ref2va", "keep_steps": 4}
    )
    with pytest.raises(OmniClientError, match="separate encoder stage"):
        prepare_encoder_prompt({"prompt": "p"}, [intake_sampling])
    prepare_encoder_inputs.assert_not_called()

    # Contrast: an explicit matching step count passes the intake gate.
    explicit = OmniDiffusionSamplingParams(
        quality="lossless",
        fps=24,
        num_inference_steps=4,
        extra_args={"task": "ref2va", "keep_steps": 4},
    )
    transformed = prepare_encoder_prompt({"prompt": "p"}, [explicit])
    prepare_encoder_inputs.assert_called_once()
    assert transformed["prompt"] == "p"
