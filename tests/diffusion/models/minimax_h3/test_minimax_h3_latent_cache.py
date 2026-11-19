# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cross-request continuation: a hidden latent prefix that every other request feature ignores."""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from vllm_omni.diffusion.models.minimax_h3.continuation import (
    ContinuationWindow,
    _audio_t,
    _video_t,
    diffuse_continuation,
    latent_guide_args,
)
from vllm_omni.diffusion.models.minimax_h3.latent_cache import (
    CachedClip,
    LatentClipCache,
    latent_cache_clips,
    plan_latent_continuation,
    trim_decoded_overlap,
)
from vllm_omni.diffusion.models.minimax_h3.packed_tokens import (
    minimax_h3_pack_audio_latent,
    minimax_h3_patchify_video_latent,
)
from vllm_omni.errors import OmniClientError
from vllm_omni.model_executor.models.minimax_h3.continuation_request import (
    ContinuationRequest,
    continuation_output_frames,
    resolve_continuation_request,
    window_frame_index,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

LATENT_H, LATENT_W = 4, 6


def _clip(num_frames: int, *, start: int = 0, fill: float = 0.0) -> CachedClip:
    latent_t = _video_t(num_frames)
    audio_t = _audio_t(start + num_frames) - _audio_t(start)
    video = fill + torch.arange(latent_t, dtype=torch.float32).view(1, 1, latent_t, 1, 1)
    audio = fill + torch.arange(audio_t, dtype=torch.float32).view(1, 1, audio_t)
    return CachedClip(
        video=video.expand(1, 24, latent_t, LATENT_H, LATENT_W).contiguous(),
        audio=audio.expand(2, 32, audio_t).contiguous(),
        start_frame=start,
        num_frames=num_frames,
    )


# ---------------------------------------------------------------- request contract


def test_lru_refreshes_on_get_and_evicts_the_oldest():
    cache = LatentClipCache(capacity=2)
    first, second, third = (_clip(22, fill=i) for i in range(3))
    cache.put("a", first)
    cache.put("b", second)
    assert cache.get("a") is first
    cache.put("c", third)
    assert cache.get("b") is None and cache.get("a") is first and cache.get("c") is third


def test_zero_capacity_disables_the_cache():
    cache = LatentClipCache(capacity=0)
    cache.put("a", _clip(22))
    assert cache.get("a") is None and len(cache) == 0


def test_cache_size_comes_from_additional_config():
    assert latent_cache_clips(None) == latent_cache_clips({}) == 16
    assert latent_cache_clips({"latent_cache_clips": 0}) == 0
    for value in (-1, "8", True, 2.5):
        with pytest.raises(ValueError, match="latent_cache_clips"):
            latent_cache_clips({"latent_cache_clips": value})


def test_request_contract():
    assert resolve_continuation_request({}) is None
    assert resolve_continuation_request({"continue_from": "v"}) == ContinuationRequest("v", 22)
    assert resolve_continuation_request({"continue_from": "v", "continue_overlap_frames": 39}).overlap_frames == 39
    for extra in (
        {"continue_from": ""},
        {"continue_from": 3},
        {"continue_from": "v", "continue_overlap_frames": 20},
        {"continue_from": "v", "continue_overlap_frames": True},
    ):
        with pytest.raises(OmniClientError):
            resolve_continuation_request(extra)
    # New frames after a 17n+5 overlap come in whole 17-frame groups.
    assert [continuation_output_frames(frames) for frames in (5, 107, 124, 141, 345)] == [17, 102, 119, 136, 340]
    assert [window_frame_index(index, 22) for index in (0, 36, -1, -22)] == [22, 58, -1, -22]


@pytest.mark.parametrize("source_start", [0, 102])
def test_window_continues_the_global_timeline_from_the_cached_tail(source_start):
    clip = _clip(124, start=source_start)
    plan = plan_latent_continuation(
        clip, ContinuationRequest("a", 22), num_frames=141, latent_h=LATENT_H, latent_w=LATENT_W
    )
    start = clip.end_frame - 22
    assert (plan.prefix.start_frame, plan.prefix.overlap_frames) == (start, 22)
    torch.testing.assert_close(plan.prefix.video, clip.video[:, :, -7:])
    overlap_audio_t = _audio_t(clip.end_frame) - _audio_t(start)
    torch.testing.assert_close(plan.prefix.audio, clip.audio[..., -overlap_audio_t:])
    assert plan.overlap_audio_t == overlap_audio_t
    assert plan.audio_t == _audio_t(start + 141) - _audio_t(start)


@pytest.mark.parametrize(
    "clip,num_frames,overlap,latent_hw,match",
    [
        (_clip(124), 141, 22, (LATENT_H, LATENT_W + 2), "same canvas"),
        (_clip(22), 141, 39, (LATENT_H, LATENT_W), "exceeds the source video"),
        (_clip(124), 22, 22, (LATENT_H, LATENT_W), "must exceed"),
        (_clip(124, start=7100), 141, 22, (LATENT_H, LATENT_W), "limited to 300 seconds"),
    ],
)
def test_invalid_continuation_windows(clip, num_frames, overlap, latent_hw, match):
    with pytest.raises(OmniClientError, match=match):
        plan_latent_continuation(
            clip, ContinuationRequest("a", overlap), num_frames=num_frames, latent_h=latent_hw[0], latent_w=latent_hw[1]
        )


def test_trim_drops_the_hidden_overlap_frames_and_audio():
    video = torch.arange(141, dtype=torch.float32).view(1, 1, 141, 1, 1).expand(1, 3, 141, 2, 2)
    audio = torch.arange(1000, dtype=torch.float32).view(1, 1, -1).expand(1, 2, -1)
    trimmed_video, trimmed_audio = trim_decoded_overlap(video, audio, frames=22, audio_fraction=0.25)
    assert trimmed_video.shape[2] == 119 and trimmed_video[0, 0, 0, 0, 0] == 22
    assert trimmed_audio.shape[-1] == 750 and trimmed_audio[0, 0, 0] == 250


# ---------------------------------------------------------------- windows


def _prefix(clip, overlap=22):
    return plan_latent_continuation(
        clip, ContinuationRequest("a", overlap), num_frames=141, latent_h=LATENT_H, latent_w=LATENT_W
    ).prefix


def _sampler(calls):
    def sample(**kw):
        calls.append(kw)
        video = torch.full((1, 24, kw["latent_t"], kw["latent_h"], kw["latent_w"]), 100.0 + len(calls))
        audio = torch.full((2, 32, kw["audio_t"]), 100.0 + len(calls))
        return video, audio

    return sample


def test_one_window_behind_a_prefix_keeps_the_real_tail():
    clip = _clip(124, fill=7.0)
    prefix = _prefix(clip)
    window = ContinuationWindow(prefix.start_frame, prefix.start_frame + 141, 22)
    audio_t = window.audio_end - window.audio_start
    existing = torch.zeros(3, 96)  # e.g. an FL2VA keyframe or a reference image
    kwargs = dict(
        num_frames=141,
        latent_t=_video_t(141),
        latent_h=LATENT_H,
        latent_w=LATENT_W,
        audio_t=audio_t,
        visual_condition=existing,
        visual_condition_shapes=[(1, LATENT_H, LATENT_W)],
    )
    calls: list[dict] = []
    video, audio = diffuse_continuation(_sampler(calls), kwargs, window_frames=141, overlap_frames=22, prefix=prefix)
    (call,) = calls
    assert call["temporal_offset"] == pytest.approx(prefix.start_frame * 40 / 24)
    guide_rows = minimax_h3_patchify_video_latent(prefix.video, patch_size=(1, 2, 2))
    # The prefix is a frame-0 AV timeline guide, packed ahead of every reference.
    assert call["guide_blocks"] == [
        {
            "kind": "video_audio",
            "frame_index": 0,
            "ref_audio_t": prefix.audio.shape[-1],
            "latent_t": 7,
            "latent_h": LATENT_H,
            "latent_w": LATENT_W,
        }
    ]
    assert not call.get("ref_blocks")
    torch.testing.assert_close(call["visual_condition"], torch.cat((guide_rows, existing)))
    assert call["visual_condition_shapes"] == [(7, LATENT_H, LATENT_W), (1, LATENT_H, LATENT_W)]
    torch.testing.assert_close(call["audio_condition"], minimax_h3_pack_audio_latent(prefix.audio))
    # The real tail stays in front of the new suffix, as decode context.
    torch.testing.assert_close(video[:, :, :7], prefix.video)
    assert torch.all(video[:, :, 7:] == 101) and video.shape[2] == _video_t(141)
    torch.testing.assert_close(audio[..., : prefix.audio.shape[-1]], prefix.audio)
    assert audio.shape[-1] == audio_t


def test_long_video_windows_start_behind_the_prefix_on_the_global_clock():
    clip = _clip(124, start=102)
    prefix = _prefix(clip)
    total = 22 + 17 * 40  # 702 frames: three 277-frame windows overlapping by 22
    origin = prefix.start_frame
    audio_t = _audio_t(origin + total) - _audio_t(origin)
    locked = torch.arange(2 * audio_t * 32, dtype=torch.float32).reshape(2 * audio_t, 32)
    kwargs = dict(
        num_frames=total,
        latent_t=_video_t(total),
        latent_h=LATENT_H,
        latent_w=LATENT_W,
        audio_t=audio_t,
        locked_audio_rows=locked,
    )
    calls: list[dict] = []
    video, audio = diffuse_continuation(_sampler(calls), kwargs, window_frames=277, overlap_frames=22, prefix=prefix)
    assert len(calls) == 3
    # The prefix guides the first window; later windows chain through main's latent_guide.
    assert calls[0]["guide_blocks"][0]["frame_index"] == 0 and not calls[0].get("ref_blocks")
    assert [call["ref_blocks"][-1]["kind"] for call in calls[1:]] == ["latent_guide", "latent_guide"]
    starts = [origin, origin + 255, origin + 510]
    assert [call["temporal_offset"] for call in calls] == pytest.approx([start * 40 / 24 for start in starts])
    first_audio = _audio_t(starts[0] + 277) - _audio_t(starts[0])
    torch.testing.assert_close(
        calls[0]["locked_audio_rows"], locked.reshape(2, audio_t, 32)[:, :first_audio].reshape(-1, 32)
    )
    assert video.shape[2] == _video_t(total) and audio.shape[-1] == audio_t


# ---------------------------------------------------------------- real packing


def _bare_pipeline():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    return pipeline


@pytest.mark.parametrize("task", ["ref2va", "t2va", "fl2va"])
def test_hidden_prefix_packs_with_references_user_guides_and_keyframes(task):
    clip = _clip(124)
    prefix = _prefix(clip)
    window = ContinuationWindow(prefix.start_frame, prefix.start_frame + 141, 22)
    patches = (LATENT_H // 2) * (LATENT_W // 2)
    # A user timeline guide (image at output frame 36 -> window frame 58) and, per task,
    # a reference image or an FL2VA keyframe guide; all address the returned frames.
    guide_blocks = [{"kind": "image", "frame_index": 58, "latent_h": LATENT_H, "latent_w": LATENT_W}]
    shapes = [(1, LATENT_H, LATENT_W)]
    ref_blocks = None
    if task == "ref2va":
        ref_blocks = [{"kind": "image", "latent_h": LATENT_H, "latent_w": LATENT_W}]
        shapes.append((1, LATENT_H, LATENT_W))
    elif task == "fl2va":
        guide_blocks.append({"kind": "image", "frame_index": 22, "latent_h": LATENT_H, "latent_w": LATENT_W})
        shapes.append((1, LATENT_H, LATENT_W))
    kwargs = dict(
        task=task,
        text_embeddings=torch.zeros(3, 4),
        text_tags=torch.ones(3, dtype=torch.long),
        seed=3,
        num_frames=141,
        latent_t=_video_t(141),
        latent_h=LATENT_H,
        latent_w=LATENT_W,
        audio_t=window.audio_end - window.audio_start,
        num_steps=2,
        video_shift=12.0,
        audio_shift=3.0,
        base_schedule=None,
        visual_condition=torch.zeros(len(shapes) * patches, 96),
        visual_condition_shape=None,
        audio_condition=None,
        ref_audio_t=None,
        ref_blocks=ref_blocks,
        visual_condition_shapes=shapes,
        audio_condition_lengths=None,
        guide_blocks=guide_blocks,
        temporal_offset=window.start * 40 / 24,
    )
    kwargs.update(latent_guide_args(kwargs, prefix.video, prefix.audio, timeline_guide=True))
    inputs = _bare_pipeline()._build_denoise_inputs(**kwargs)
    target_rows = _video_t(141) * patches + 2 * kwargs["audio_t"]
    condition_rows = (len(shapes) + 7) * patches + 2 * prefix.audio.shape[-1]
    assert inputs["branch"].used_len == 3 + condition_rows + target_rows


# ---------------------------------------------------------------- encoder mapping


def test_encoder_samples_the_returned_clip_behind_the_hidden_overlap():
    from vllm_omni.model_executor.models.minimax_h3.encoder_processing import (
        _resolve_task,
        continuation_inherits_source_canvas,
        resolve_minimax_h3_shape,
    )

    sampling = SimpleNamespace(extra_args={"continue_from": "v"}, num_frames=124, fps=24, height=768, width=1344)
    assert resolve_minimax_h3_shape("ref2va", sampling, None)[2:4] == (141, _video_t(141))
    assert resolve_minimax_h3_shape("ref2va", sampling, None, continuation_window=False)[2] == 124
    # Task inference is the request's own; a continuation changes nothing about it.
    assert _resolve_task({"continue_from": "v"}, {}) == _resolve_task({}, {}) == "t2va"
    assert _resolve_task({"continue_from": "v"}, {"image": "first.png"}) == "fl2va"
    # A T2VA continuation keeps its source's canvas, so it needs no aspect ratio of its own.
    unsized = SimpleNamespace(extra_args={"continue_from": "v"}, num_frames=124, fps=24, height=None, width=None)
    assert resolve_minimax_h3_shape("t2va", unsized, None)[:2] == (768, 1344)
    with pytest.raises(OmniClientError, match="explicit aspect_ratio"):
        resolve_minimax_h3_shape(
            "t2va", SimpleNamespace(extra_args={}, num_frames=124, fps=24, height=None, width=None), None
        )
    # A lone dimension is neither a size nor an inherited canvas for T2VA; other
    # tasks plan their canvas from the aspect ratio or image, as without continue_from.
    for height, width in ((768, None), (None, 1344)):
        half = SimpleNamespace(extra_args={"continue_from": "v"}, num_frames=124, fps=24, height=height, width=width)
        assert not continuation_inherits_source_canvas("t2va", half)
        with pytest.raises(OmniClientError, match="both height and width"):
            resolve_minimax_h3_shape("t2va", half, None)
        assert resolve_minimax_h3_shape("ref2va", half, None)[:2] == (768, 1344)


def test_edit_masks_written_for_the_returned_clip_map_onto_the_window():
    from vllm_omni.model_executor.models.minimax_h3.encoder_processing import (
        _continuation_audio_edit_mask,
        _continuation_video_edit_mask,
    )

    kept_second_half = torch.ones(124, 2, 3)
    kept_second_half[62:] = 0.0  # frame-space mask: keep the second half of the requested clip
    for value in (kept_second_half, 0.0):
        mask = _continuation_video_edit_mask(
            value, requested_frames=124, output_frames=119, overlap=22, latent_h=LATENT_H, latent_w=LATENT_W
        )
        assert mask.shape == (_video_t(141), LATENT_H, LATENT_W)
        assert torch.all(mask[:7] == 1.0)  # the hidden overlap is regenerated behind the guide
        assert torch.all(mask[-1] == 0.0)
        if isinstance(value, float):
            assert torch.all(mask[7:] == 0.0)
        else:
            assert torch.all(mask[8:20] == 1.0) and torch.all(mask[-15:] == 0.0)
    audio = _continuation_audio_edit_mask(0.0, requested_audio_t=207, output_audio_t=198, overlap_audio_t=37)
    assert audio.shape == (2, 235) and torch.all(audio[:, :37] == 1) and torch.all(audio[:, 37:] == 0)


def test_timeline_guides_address_the_returned_frames():
    from vllm_omni.model_executor.models.minimax_h3.encoder_processing import prepare_encoder_inputs

    image = Image.new("RGB", (64, 48), "red")
    prepared = prepare_encoder_inputs(
        {
            "prompt": "the scene continues",
            "multi_modal_data": {
                "timeline_guides": [{"frame_index": 36, "image": image}, {"frame_index": -1, "image": image}]
            },
        },
        SimpleNamespace(extra_args={"continue_from": "v"}, num_frames=124, fps=24, height=64, width=96),
    )
    assert prepared.media.num_frames == 141 and prepared.media.task == "t2va"
    assert [guide.start for guide in prepared.media.timeline_guides] == [58, 140]


# ---------------------------------------------------------------- pipeline


def _conditioning(task, num_frames, **media):
    from vllm_omni.model_executor.models.minimax_h3.conditioning import (
        MiniMaxH3EncoderConditioning,
        MiniMaxH3EncoderMediaConditioning,
        MiniMaxH3TextConditioning,
    )

    media = MiniMaxH3EncoderMediaConditioning(
        task=task,
        height=LATENT_H * 16,
        width=LATENT_W * 16,
        num_frames=num_frames,
        latent_t=_video_t(num_frames),
        audio_t=_audio_t(num_frames),
        **media,
    )
    text = MiniMaxH3TextConditioning(torch.zeros(3, 5120, dtype=torch.bfloat16), torch.ones(3, dtype=torch.int64))
    return MiniMaxH3EncoderConditioning.from_components(text, media)


def _pipeline(*, load_vae_decoder=True):
    pipeline = _bare_pipeline()
    pipeline.default_video_shift = 12.0
    pipeline.default_audio_shift = 3.0
    pipeline.od_config = SimpleNamespace(step_execution=False)
    pipeline.load_vae_decoder = load_vae_decoder
    pipeline._fasth3 = pipeline._fasth3_checkpoint = pipeline._vdn = None
    pipeline._active_turbo_spec = lambda _sampling: None
    pipeline._has_active_native_lora = lambda _sampling: False
    pipeline._resolve_task = lambda task, **_kwargs: task
    pipeline._resolve_sigma_positions = lambda _task, _sampling: (None, 3)
    pipeline._quality_policy = SimpleNamespace(resolve=lambda **_kwargs: SimpleNamespace(cache_dit=None))
    pipeline._cache_dit_runtime = SimpleNamespace(prepare=lambda _plan: None)
    pipeline._latent_clip_cache = LatentClipCache(capacity=8)
    pipeline._offload_model_cpu_stage_output = lambda value: value
    pipeline._release_stage_cache = lambda: None
    calls, mp4 = [], []

    def diffuse(**kwargs):
        calls.append(kwargs)
        video = 100.0 * len(calls) + torch.arange(kwargs["latent_t"], dtype=torch.float32).view(1, 1, -1, 1, 1)
        audio = 100.0 * len(calls) + torch.arange(kwargs["audio_t"], dtype=torch.float32).view(1, 1, -1)
        return (
            video.expand(1, 24, kwargs["latent_t"], kwargs["latent_h"], kwargs["latent_w"]).contiguous(),
            audio.expand(2, 32, kwargs["audio_t"]).contiguous(),
        )

    def decode(video_latent, audio_latent, *, height, width):
        frames = (video_latent.shape[2] - 2) // 5 * 17 + 5
        video = torch.arange(frames, dtype=torch.float32).view(1, 1, -1, 1, 1) / frames
        audio = torch.arange(audio_latent.shape[-1] * 800, dtype=torch.float32).view(1, 1, -1)
        return video.expand(1, 3, frames, height, width).contiguous(), audio.expand(1, 2, -1).contiguous()

    def decode_to_mp4(video_latent, audio_latent, *, trim, **_kwargs):
        mp4.append(trim)
        return b"mp4"

    pipeline.diffuse, pipeline.decode, pipeline.decode_to_mp4 = diffuse, decode, decode_to_mp4
    return pipeline, calls, mp4


def _run(pipeline, conditioning, extra, *, num_outputs=1):
    sampling = SimpleNamespace(extra_args=extra, quality=None, seed=7, num_outputs_per_prompt=num_outputs)
    pipeline._prepare_request_inputs = lambda _prompt, _sampling: pipeline._prepare_encoder_conditioning_inputs(
        conditioning, _sampling
    )
    return pipeline.forward(SimpleNamespace(prompts=["prompt"], sampling_params=sampling)).output


def test_chained_requests_combine_continuation_with_every_task_and_lock_source():
    pipeline, calls, _ = _pipeline()
    first, _ = _run(pipeline, _conditioning("t2va", 124), {"latent_cache_key": "a"})
    assert first.shape[1] == 124 and len(calls[0].get("ref_blocks") or []) == 0
    source = pipeline._latent_clip_cache.get("a")

    # Ref2VA continuation: the cached tail is a frame-0 timeline guide, like every task.
    video, audio = _run(pipeline, _conditioning("ref2va", 141), {"latent_cache_key": "b", "continue_from": "a"})
    assert calls[1]["guide_blocks"][0]["kind"] == "video_audio" and not calls[1]["ref_blocks"]
    torch.testing.assert_close(
        calls[1]["visual_condition"], minimax_h3_patchify_video_latent(source.video[:, :, -7:], patch_size=(1, 2, 2))
    )
    assert video.shape[1] == 119 and audio.shape[-1] == (calls[1]["audio_t"] - (_audio_t(124) - _audio_t(102))) * 800
    second = pipeline._latent_clip_cache.get("b")
    assert (second.start_frame, second.num_frames) == (102, 141)
    torch.testing.assert_close(second.video[:, :, :7], source.video[:, :, -7:])

    # T2VA continuation with driving audio: the tail is a frame-0 guide and the
    # locked track starts with the tail audio; this window's global audio length
    # (179) differs from the encoder's standalone one (178).
    drive = torch.full((2 * 200, 32), 5.0)
    conditioning = _conditioning("t2va", 107, audio_condition=drive, audio_condition_lengths=(200,))
    video, _ = _run(
        pipeline, conditioning, {"latent_cache_key": "c", "continue_from": "b", "audio_mode": "lock_source"}
    )
    call = calls[2]
    assert call["guide_blocks"][0] == {
        "kind": "video_audio",
        "frame_index": 0,
        "ref_audio_t": 37,
        "latent_t": 7,
        "latent_h": LATENT_H,
        "latent_w": LATENT_W,
    }
    assert call["audio_t"] == 179 and video.shape[1] == 107 - 22
    locked = call["locked_audio_rows"].reshape(2, 179, 32)
    torch.testing.assert_close(locked[:, :37], second.audio[..., -37:].permute(0, 2, 1))
    assert torch.all(locked[:, 37:] == 5.0)

    # FL2VA continuation: the user's first-frame keyframe lands on the first returned frame.
    keyframe = torch.zeros((LATENT_H // 2) * (LATENT_W // 2), 96)
    conditioning = _conditioning(
        "fl2va",
        141,
        visual_condition=keyframe,
        visual_condition_shapes=((1, LATENT_H, LATENT_W),),
        keyframe_frame_indices=(0,),
    )
    _run(pipeline, conditioning, {"continue_from": "c"})
    assert [(block["kind"], block["frame_index"]) for block in calls[3]["guide_blocks"]] == [
        ("video_audio", 0),
        ("image", 22),
    ]


def test_continuation_rejects_latent_refine():
    pipeline, calls, _ = _pipeline()
    _run(pipeline, _conditioning("t2va", 124), {"latent_cache_key": "a"})
    with pytest.raises(OmniClientError, match="latent_refine does not support continue_from"):
        _run(pipeline, _conditioning("ref2va", 141), {"continue_from": "a", "latent_refine": 0.4})
    assert len(calls) == 1  # rejected before any denoising


def test_refined_source_caches_the_latents_it_decodes():
    pipeline, calls, _ = _pipeline()
    decoded = []
    decode = pipeline.decode

    def capture(video_latent, audio_latent, **kwargs):
        decoded.append((video_latent.clone(), audio_latent.clone()))
        return decode(video_latent, audio_latent, **kwargs)

    pipeline.decode = capture
    _run(pipeline, _conditioning("t2va", 124), {"latent_cache_key": "a", "latent_refine": 0.4})
    assert len(calls) == 2 and calls[1]["refine"] is not None  # base pass, then the refine pass
    cached = pipeline._latent_clip_cache.get("a")
    torch.testing.assert_close(cached.video, decoded[0][0].cpu())
    torch.testing.assert_close(cached.audio, decoded[0][1].cpu())


def test_unsized_t2va_continuation_keeps_a_portrait_source_canvas():
    pipeline, calls, _ = _pipeline()
    # A portrait source: its latent canvas is the transpose of the encoder's provisional 16:9 one.
    portrait = CachedClip(
        video=torch.zeros(1, 24, _video_t(124), LATENT_W, LATENT_H),
        audio=torch.zeros(2, 32, _audio_t(124)),
        start_frame=0,
        num_frames=124,
    )
    pipeline._latent_clip_cache.put("portrait", portrait)
    video, _ = _run(pipeline, _conditioning("t2va", 141), {"continue_from": "portrait"})
    assert (calls[0]["latent_h"], calls[0]["latent_w"]) == (LATENT_W, LATENT_H)
    assert video.shape[-3:-1] == (LATENT_W * 16, LATENT_H * 16)  # [batch, frames, height, width, channels]
    # An explicit aspect ratio is the caller's canvas: it must match the source like any request.
    with pytest.raises(OmniClientError, match="same canvas"):
        _run(pipeline, _conditioning("t2va", 141), {"continue_from": "portrait", "aspect_ratio": "16:9"})
    # Guide frames were encoded on the provisional canvas, so they cannot follow the swap.
    guided = _conditioning(
        "t2va",
        141,
        guide_blocks=({"kind": "image", "frame_index": 30, "latent_h": LATENT_H, "latent_w": LATENT_W},),
        guide_visual_condition=torch.zeros((LATENT_H // 2) * (LATENT_W // 2), 96),
    )
    with pytest.raises(OmniClientError, match="timeline guides with continue_from"):
        _run(pipeline, guided, {"continue_from": "portrait"})
    assert len(calls) == 1


def test_unknown_source_is_a_client_error():
    pipeline, _, _ = _pipeline()
    with pytest.raises(OmniClientError, match="no cached latents"):
        _run(pipeline, _conditioning("ref2va", 141), {"continue_from": "missing"})


def test_preencoded_multi_output_and_decoder_stage_outputs_are_trimmed():
    pipeline, calls, mp4 = _pipeline()
    _run(pipeline, _conditioning("t2va", 124), {"latent_cache_key": "a"})
    _run(
        pipeline,
        _conditioning("ref2va", 141),
        {"latent_cache_key": "b", "continue_from": "a", "preencode_mp4": True},
        num_outputs=2,
    )
    window = ContinuationWindow(102, 243, 22)
    expected = {"frames": 22, "audio_fraction": window.overlap_audio_t / (window.audio_end - window.audio_start)}
    assert mp4 == [expected, expected] and len(calls) == 3
    # Only the first output is addressable as this job, so only it is cached.
    assert pipeline._latent_clip_cache.get("b").video[0, 0, 7, 0, 0] == 200 + 7

    staged, _, _ = _pipeline(load_vae_decoder=False)
    staged._latent_clip_cache = pipeline._latent_clip_cache
    staged._prepare_request_inputs = lambda _p, _s: staged._prepare_encoder_conditioning_inputs(
        _conditioning("ref2va", 141), _s
    )
    output = staged.forward(
        SimpleNamespace(
            prompts=["p"],
            sampling_params=SimpleNamespace(
                extra_args={"continue_from": "b"}, quality=None, seed=7, num_outputs_per_prompt=1
            ),
        )
    )
    assert output.output["metadata"]["minimax_h3_decode"]["trim"]["frames"] == 22


def test_step_post_decode_keeps_the_real_tail_caches_and_trims():
    pipeline, _, _ = _pipeline()
    pipeline._latent_clip_cache.put("a", _clip(124, fill=7.0))
    context = pipeline._prepare_encoder_conditioning_inputs(
        _conditioning("ref2va", 141),
        SimpleNamespace(
            extra_args={"continue_from": "a", "latent_cache_key": "b"}, quality=None, seed=1, num_outputs_per_prompt=1
        ),
    )
    window_video = torch.full((1, 24, _video_t(141), LATENT_H, LATENT_W), 50.0)
    window_audio = torch.full((2, 32, context["audio_t"]), 50.0)
    pipeline._unpack_denoised_rows = lambda *args, **kwargs: (window_video, window_audio)
    shape = {
        key: context[key]
        for key in (
            "height",
            "width",
            "latent_t",
            "latent_h",
            "latent_w",
            "audio_t",
            "latent_continuation",
            "latent_cache_key",
            "num_frames",
        )
    }
    state = SimpleNamespace(
        extra={"minimax_h3_shape": shape, "minimax_h3_branch": None, "minimax_h3_audio_rows": None}, latents=None
    )
    pipeline.od_config = SimpleNamespace(step_execution=True)
    video, _ = pipeline.post_decode(state).output
    assert video.shape[1] == 119
    cached = pipeline._latent_clip_cache.get("b")
    torch.testing.assert_close(cached.video[:, :, :7], _clip(124, fill=7.0).video[:, :, -7:])
    assert torch.all(cached.video[:, :, 7:] == 50.0)


def test_edit_media_for_the_returned_clip_gets_a_hidden_prefix(monkeypatch, tmp_path):
    from vllm_omni.model_executor.models.minimax_h3 import encoder_processing as processing

    calls = {}
    frames = torch.arange(119, dtype=torch.uint8).view(119, 1, 1, 1).expand(119, 64, 96, 3).numpy()

    def prepare_video(value, **kwargs):
        calls["prepare"] = kwargs
        return {"original_path": value, "prepared_path": str(tmp_path / "prepared.mp4"), "input_has_audio": True}

    monkeypatch.setattr(processing, "prepare_edit_video", prepare_video)
    monkeypatch.setattr(processing, "load_video_frames", lambda _path: frames)
    monkeypatch.setattr(processing, "load_video_audio", lambda _value, **kw: (torch.ones(2, 32_000 * 5), 32_000))
    prepared = processing.prepare_encoder_inputs(
        {
            "prompt": "replace the sky",
            "multi_modal_data": {"source_video": "source.mp4", "video_noise_mask": 0.0, "audio_noise_mask": 0.0},
        },
        SimpleNamespace(
            extra_args={"task": "t2va", "aspect_ratio": "16:9", "continue_from": "v"},
            num_frames=124,
            fps=24,
            height=64,
            width=96,
        ),
    )
    media = prepared.media
    assert calls["prepare"]["target_frame_count"] == 119 and media.num_frames == 141
    assert (
        media.video_edit.shape[0] == 141
        and torch.all(media.video_edit[:22] == 0)
        and media.video_edit[22, 0, 0, 0] == 0
    )
    assert media.video_edit[23, 0, 0, 0] == 1
    waveform, rate = media.audio_edit
    silence = round(22 / 24 * rate)
    assert torch.all(waveform[..., :silence] == 0) and torch.all(waveform[..., silence:] == 1)
    assert waveform.shape[-1] == silence + round(119 / 24 * rate)
    assert torch.all(media.video_edit_mask[:7] == 1) and torch.all(media.video_edit_mask[7:] == 0)
    assert torch.all(media.audio_edit_mask[:, :37] == 1) and torch.all(media.audio_edit_mask[:, 37:] == 0)
