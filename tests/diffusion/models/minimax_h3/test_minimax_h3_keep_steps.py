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
    short_update[0] = False
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
    assert base.seq_len < branch.seq_len
    assert tuple(inputs["base_cond_rows"].shape) == (BASE_IMAGE_ROWS, 96)


def test_base_cond_rows_equal_image_only_anchor(pipeline):
    """The sliced prefix of the full anchor is an image-only anchor, bit for bit."""
    inputs = pipeline._build_denoise_inputs(**_keep_steps_kwargs())
    image_only = pipeline._build_denoise_inputs(
        **{
            **_inputs_kwargs(),
            "visual_condition": _visual_rows()[:BASE_IMAGE_ROWS],
            "ref_blocks": [_ref_blocks()[0]],
            "visual_condition_shapes": [IMAGE_SHAPE],
        }
    )
    assert torch.equal(inputs["base_cond_rows"], image_only["cond_anchor"])


@pytest.mark.parametrize(
    "override,match",
    [
        ({"task": "t2va"}, "requires task ref2va"),
        ({"guide_blocks": [{"kind": "image", "frame_index": 0, "latent_h": 4, "latent_w": 6}]}, "timeline guides"),
        ({"base_text_embeddings": None}, "requires base text conditioning"),
        ({"base_text_tags": None}, "requires base text conditioning"),
        ({"ref_blocks": [_ref_blocks()[0]], "visual_condition_shapes": [IMAGE_SHAPE]}, "silent video reference"),
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
):
    return minimax_h3_denoise_loop(
        model=model,
        positive=positive,
        initial_video_rows=initial_rows,
        initial_audio_rows=torch.zeros(positive.audio_pos.shape[0], 32),
        keyframe_cond_rows=keyframe,
        sigmas_video=SIGMAS_VIDEO if sigmas is None else sigmas,
        sigmas_audio=SIGMAS_AUDIO if audio_sigmas is None else audio_sigmas,
        device=torch.device("cpu"),
        base=base,
        keep_steps=keep_steps,
        base_cond_rows=_base_cond_rows() if base is not None else None,
        on_step=on_step,
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
    for keep_steps in (-1, 3, "2"):
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


@pytest.mark.parametrize(
    "conditioning_overrides,extra_args,match",
    [
        ({"task": "t2va"}, {"task": "t2va", "keep_steps": 2}, "requires task ref2va"),
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
    monkeypatch.setattr(
        pipeline,
        "_resolve_latent_continuation",
        lambda extra, conditioning, *, inherit_canvas: (conditioning, SimpleNamespace()),
    )
    with pytest.raises(OmniClientError, match="continue_from"):
        pipeline._prepare_encoder_conditioning_inputs(_keep_steps_conditioning(), _sampling(keep_steps=2))

    pipeline = _validation_pipeline()
    monkeypatch.setattr(pipeline, "_resolve_latent_refine", lambda extra: SimpleNamespace())
    with pytest.raises(OmniClientError, match="latent refine/upscale"):
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


def test_local_conditioning_encodes_image_only_presentation(monkeypatch: pytest.MonkeyPatch):
    pipeline = _validation_pipeline()
    prepared = _Prepared(
        prompt="p",
        images=[object()],
        qwen_videos=[object()],
        video_timestamps=[[0.0]],
        condition_labels=[("image", 1), ("video", 1)],
        media=SimpleNamespace(num_frames=NUM_FRAMES),
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
    monkeypatch.setattr(pipeline, "_resolve_task", lambda task, **kwargs: "ref2va")
    monkeypatch.setattr(pipeline, "_extract_text_conditioning", lambda raw: None)
    monkeypatch.setattr(pipeline, "_encode_local_media", lambda media: media_conditioning)
    monkeypatch.setattr(pipeline_minimax_h3, "prepare_encoder_inputs", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(pipeline_minimax_h3, "resolve_continuation", lambda extra_args, **kwargs: None)
    encoded = [
        (torch.ones(2, 4, dtype=torch.bfloat16), torch.ones(2, dtype=torch.long)),
        (torch.full((3, 4), 2.0, dtype=torch.bfloat16), torch.ones(3, dtype=torch.long)),
    ]
    encode_prompt = Mock(side_effect=encoded)
    monkeypatch.setattr(pipeline, "encode_prompt", encode_prompt)

    sampling = OmniDiffusionSamplingParams(
        quality="lossless",
        fps=24,
        num_inference_steps=4,
        extra_args={"task": "ref2va", "keep_steps": 2},
    )
    conditioning, window = pipeline._prepare_local_conditioning({"prompt": "p"}, sampling)

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
