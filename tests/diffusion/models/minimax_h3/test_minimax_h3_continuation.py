# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3.continuation import (
    _FIRST_WINDOW_CONDITION_KEYS,
    LatentPrefix,
    continuation_uses_timeline_guides,
    diffuse_continuation,
    later_windows_drop_text_images,
    plan_continuation_windows,
    resolve_continuation,
)
from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence_ref2va_blocks
from vllm_omni.errors import OmniClientError
from vllm_omni.model_executor.models.minimax_h3.long_video import (
    MINIMAX_H3_CONTINUATION_TASKS,
    check_continuation_conditions,
    max_output_seconds,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("vary_prompts", [False, True])
@pytest.mark.parametrize("total,window", [(1807, 277), (1807, 243), (124, 277)])
def test_continuation_retains_prefix_and_global_stereo_timeline(total, window, vary_prompts):
    audio_t = round(total * 40 / 24)
    source = torch.arange(2 * audio_t * 32).reshape(2, audio_t, 32).float()
    calls = []
    plan = plan_continuation_windows(total, window, 22)
    texts = [(torch.full((3 + i, 8), float(i)), torch.ones(3 + i, dtype=torch.long)) for i in range(len(plan))]

    def sample(**kw):
        calls.append(kw)
        t = kw["latent_t"]
        if vary_prompts:
            assert kw["text_embeddings"] is texts[len(calls) - 1][0]
            assert kw["text_tags"] is texts[len(calls) - 1][1]
            assert kw["media_time_origin"] == len(texts[-1][0])
        # Different constants expose replacement of an old prefix or an
        # accidental append of the hidden overlap.
        video = torch.full((1, 24, t, 2, 2), float(len(calls)))
        audio = kw["locked_audio_rows"].reshape(2, kw["audio_t"], 32).permute(0, 2, 1)
        if len(calls) > 1:
            assert torch.all(kw["visual_condition"] == len(calls) - 1)
            assert kw["ref_blocks"][-1]["kind"] == "latent_guide"
        return video, audio

    kwargs = dict(
        num_frames=total,
        latent_t=(total - 5) // 17 * 5 + 2,
        latent_h=2,
        latent_w=2,
        audio_t=audio_t,
        locked_audio_rows=source.reshape(-1, 32),
    )
    video, audio = diffuse_continuation(
        sample, kwargs, window_frames=window, overlap_frames=22, text_conditioning=texts if vary_prompts else None
    )
    torch.testing.assert_close(audio, source.permute(0, 2, 1))
    plan = plan_continuation_windows(total, window, 22)
    assert [call["temporal_offset"] for call in calls] == pytest.approx([part.start * 40 / 24 for part in plan])
    offset = 0
    for i, part in enumerate(plan):
        new_frames = part.end - part.start - part.overlap
        count = (new_frames - 5) // 17 * 5 + 2 if i == 0 else new_frames // 17 * 5
        assert torch.all(video[:, :, offset : offset + count] == i + 1)
        offset += count
    assert offset == video.shape[2] == kwargs["latent_t"]


@pytest.mark.parametrize("temporal_offset", [0.0, 425.0, 221 * 40 / 24])
def test_latent_guide_shares_target_clock_and_remains_condition_only(temporal_offset):
    packed = minimax_h3_packed_sequence_ref2va_blocks(
        temporal_offset=temporal_offset,
        text_len=3,
        latent_t=12,
        latent_h=4,
        latent_w=4,
        audio_t=68,
        ref_blocks=[
            {"kind": "image", "latent_h": 4, "latent_w": 4},
            {"kind": "latent_guide", "latent_t": 7, "latent_h": 4, "latent_w": 4, "ref_audio_t": 37},
        ],
    )
    pos = packed["img_position_ids"]
    visual = packed["img_pos"]
    target = visual[packed["update_mask"]]
    guide = visual[~packed["update_mask"]][4:]
    torch.testing.assert_close(pos[guide], pos[target[:28]])
    audio = packed["audio_pos"]
    guide_audio = audio[~packed["audio_update_mask"]].reshape(2, 37)
    target_audio = audio[packed["audio_update_mask"]].reshape(2, 68)
    torch.testing.assert_close(pos[guide_audio], pos[target_audio[:, :37]])
    assert pos[target[0], 0] == pytest.approx(4 + temporal_offset)
    torch.testing.assert_close(pos[:3, 0], torch.arange(3, dtype=torch.float64))
    assert torch.all(pos[3:7, 0] == 3)  # static reference image stays fixed


@pytest.mark.parametrize(
    "extra,step",
    [
        ({"long_video_mode": "bad"}, False),
        ({"long_video_mode": "continuation"}, True),
        ({"long_video_mode": "continuation", "continuation_overlap_frames": 23}, False),
        ({"long_video_mode": "continuation", "continuation_window_frames": True}, False),
        ({"long_video_mode": "continuation", "continuation_overlap_frames": 277}, False),
    ],
)
def test_invalid_continuation_options(extra, step):
    with pytest.raises(OmniClientError):
        resolve_continuation(extra, task="ref2va", step_execution=step)


def test_global_offset_moves_temporal_references_but_not_images_or_padding():
    kwargs = dict(
        text_len=3,
        latent_t=12,
        latent_h=4,
        latent_w=4,
        audio_t=68,
        ref_blocks=[
            {"kind": "image", "latent_h": 4, "latent_w": 4},
            {"kind": "audio", "ref_audio_t": 3},
            {"kind": "video_audio", "ref_audio_t": 2, "latent_t": 2, "latent_h": 4, "latent_w": 4},
            {"kind": "image", "latent_h": 4, "latent_w": 4},
            {"kind": "latent_guide", "ref_audio_t": 37, "latent_t": 7, "latent_h": 4, "latent_w": 4},
        ],
    )
    base = minimax_h3_packed_sequence_ref2va_blocks(**kwargs)
    shifted = minimax_h3_packed_sequence_ref2va_blocks(**kwargs, temporal_offset=425.0)
    moving = base["image_mask"] | base["audio_mask"]
    moving[3:7] = False  # first static image
    moving[25:29] = False  # static image after temporal references
    delta = shifted["img_position_ids"] - base["img_position_ids"]
    torch.testing.assert_close(delta[:, 0], moving.double() * 425)
    assert torch.count_nonzero(delta[:, 1:]) == 0
    for key, value in base.items():
        if isinstance(value, torch.Tensor) and key != "img_position_ids":
            torch.testing.assert_close(shifted[key], value)


def test_global_offset_moves_image_timeline_guides_with_the_window():
    kwargs = dict(
        text_len=3,
        latent_t=12,
        latent_h=4,
        latent_w=4,
        audio_t=68,
        guide_blocks=[
            {"kind": "video_audio", "frame_index": 0, "ref_audio_t": 37, "latent_t": 7, "latent_h": 4, "latent_w": 4},
            {"kind": "image", "frame_index": 82, "latent_h": 4, "latent_w": 4},
        ],
        ref_blocks=[{"kind": "image", "latent_h": 4, "latent_w": 4}],
    )
    base = minimax_h3_packed_sequence_ref2va_blocks(**kwargs)
    shifted = minimax_h3_packed_sequence_ref2va_blocks(**kwargs, temporal_offset=425.0)
    delta = (shifted["img_position_ids"] - base["img_position_ids"])[:, 0]
    guide_rows = slice(3, 3 + 7 * 4 + 2 * 37 + 4)
    torch.testing.assert_close(delta[guide_rows], torch.full_like(delta[guide_rows], 425))
    assert torch.count_nonzero(delta[guide_rows.stop : guide_rows.stop + 4]) == 0  # static image reference


def test_long_ref2va_defaults_to_continuation_with_explicit_full_opt_out():
    assert resolve_continuation({"long_video": True}, task="ref2va", step_execution=False) == (277, 22)
    assert resolve_continuation({}, task="ref2va", step_execution=False) is None
    assert (
        resolve_continuation({"long_video": True, "long_video_mode": "full"}, task="ref2va", step_execution=False)
        is None
    )


@pytest.mark.parametrize("task", ["t2va", "fl2va"])
def test_t2va_and_fl2va_continuation_is_opt_in(task):
    # long_video alone keeps whole-target sampling for these tasks, as before.
    assert resolve_continuation({"long_video": True}, task=task, step_execution=False) is None
    assert max_output_seconds({"long_video": True}, task) == 30.0
    extra = {"long_video": True, "long_video_mode": "continuation"}
    # FL2VA windows default shorter: the first frame's look does not survive a 277-frame join with Turbo.
    assert resolve_continuation(extra, task=task, step_execution=False) == ({"t2va": 277, "fl2va": 141}[task], 22)
    assert resolve_continuation({**extra, "continuation_window_frames": 209}, task=task, step_execution=False) == (
        209,
        22,
    )
    assert max_output_seconds(extra, task) == 300.0
    with pytest.raises(OmniClientError, match="request execution"):
        resolve_continuation(extra, task=task, step_execution=True)


def test_continuation_rejects_unknown_tasks_and_late_keyframes():
    with pytest.raises(OmniClientError, match="continuation supports"):
        resolve_continuation({"long_video_mode": "continuation"}, task="t2a", step_execution=False)
    check_continuation_conditions("t2va", ())
    check_continuation_conditions("fl2va", (0,))
    check_continuation_conditions("ref2va", (0, 123))
    for indices in [(123,), (0, 123)]:
        with pytest.raises(OmniClientError, match="first frame only"):
            check_continuation_conditions("fl2va", indices)


@pytest.mark.parametrize("with_prefix", [False, True])
def test_timeline_guides_carry_every_tail_and_keep_keyframes_in_the_first_window(with_prefix):
    total, window, overlap = 379, 141, 22
    overlap_v = 7
    plan = plan_continuation_windows(
        total, window, overlap, first_overlap=overlap if with_prefix else 0, origin=119 if with_prefix else 0
    )
    keyframe = torch.full((4, 96), -1.0)
    calls = []

    def sample(**kw):
        calls.append(kw)
        video = torch.full((1, 24, kw["latent_t"], 2, 2), float(len(calls)))
        audio = torch.full((2, 32, kw["audio_t"]), float(len(calls)))
        return video, audio

    kwargs = dict(
        task="fl2va",
        num_frames=total,
        latent_t=(total - 5) // 17 * 5 + 2,
        latent_h=2,
        latent_w=2,
        audio_t=round(total * 40 / 24),
        visual_condition=keyframe,
        visual_condition_shape=(1, 2, 2),
        visual_condition_shapes=None,
        audio_condition=None,
        audio_condition_lengths=None,
        keyframe_frame_indices=[0],
        guide_blocks=None,
        # FL2VA conditioning lists the keyframe image as a block too.
        ref_blocks=[{"kind": "image", "latent_h": 2, "latent_w": 2}],
        locked_audio_rows=None,
    )
    prefix = None
    if with_prefix:
        prefix = LatentPrefix(
            video=torch.zeros(1, 24, overlap_v, 2, 2),
            audio=torch.zeros(2, 32, plan[0].overlap_audio_t),
            start_frame=119,
            overlap_frames=overlap,
        )
    video, audio = diffuse_continuation(
        sample, kwargs, window_frames=window, overlap_frames=overlap, prefix=prefix, timeline_guides=True
    )

    assert len(calls) == len(plan) == 3
    first = calls[0]
    # The request's own keyframe stays in its first window ...
    assert first["keyframe_frame_indices"] == [0]
    if with_prefix:
        # ... behind the hidden prefix, which packs ahead of it.
        assert first["guide_blocks"][0]["kind"] == "video_audio"
        assert torch.all(first["visual_condition"][:overlap_v] == 0)
        assert torch.all(first["visual_condition"][overlap_v:] == -1)
    else:
        assert first["guide_blocks"] is None
        assert first["visual_condition"] is keyframe
    for index, call in enumerate(calls[1:], start=2):
        # ... and later windows carry only their predecessor's tail, as a frame-0 AV timeline guide.
        overlap_audio = plan[index - 1].overlap_audio_t
        (block,) = call["guide_blocks"]
        assert block == {
            "kind": "video_audio",
            "frame_index": 0,
            "ref_audio_t": overlap_audio,
            "latent_t": overlap_v,
            "latent_h": 2,
            "latent_w": 2,
        }
        assert call["keyframe_frame_indices"] is None
        assert call["visual_condition_shape"] is None
        assert call["visual_condition_shapes"] == [(overlap_v, 2, 2)]
        assert call["audio_condition_lengths"] == [overlap_audio]
        assert call["ref_blocks"] is None
        assert torch.all(call["visual_condition"] == index - 1)
        assert torch.all(call["audio_condition"] == index - 1)
        assert call["temporal_offset"] == pytest.approx(plan[index - 1].start * 40 / 24)
    assert video.shape[2] == kwargs["latent_t"] and audio.shape[-1] == kwargs["audio_t"]


def test_ref2va_windows_keep_the_latent_guide_reference():
    calls = []

    def sample(**kw):
        calls.append(kw)
        return torch.zeros(1, 24, kw["latent_t"], 2, 2), torch.zeros(2, 32, kw["audio_t"])

    reference = torch.ones(4, 96)
    kwargs = dict(
        num_frames=260,
        latent_t=77,
        latent_h=2,
        latent_w=2,
        audio_t=433,
        visual_condition=reference,
        visual_condition_shapes=[(1, 2, 2)],
        ref_blocks=[{"kind": "image", "latent_h": 2, "latent_w": 2}],
    )
    diffuse_continuation(sample, kwargs, window_frames=141, overlap_frames=22)
    assert len(calls) == 2
    assert calls[1]["ref_blocks"][-1]["kind"] == "latent_guide"
    assert calls[1].get("guide_blocks") is None
    assert torch.all(calls[1]["visual_condition"][:4] == 1)  # the static reference stays in every window


def test_different_chunk_prompt_lengths_keep_the_same_media_clock():
    kwargs = dict(
        latent_t=7,
        latent_h=4,
        latent_w=4,
        audio_t=37,
        ref_blocks=[{"kind": "image", "latent_h": 4, "latent_w": 4}],
        media_time_origin=20,
    )
    first = minimax_h3_packed_sequence_ref2va_blocks(text_len=3, **kwargs)
    second = minimax_h3_packed_sequence_ref2va_blocks(text_len=15, temporal_offset=425, **kwargs)
    for key, mask in (("img_pos", "update_mask"), ("audio_pos", "audio_update_mask")):
        a = first["img_position_ids"][first[key][first[mask]]]
        b = second["img_position_ids"][second[key][second[mask]]]
        torch.testing.assert_close(b[:, 0] - a[:, 0], torch.full_like(a[:, 0], 425))
        torch.testing.assert_close(b[:, 1:], a[:, 1:])


@pytest.mark.parametrize("prompt_count", [None, 7, 6])
def test_request_encodes_only_window_prompts_and_prepares_references_once(monkeypatch, prompt_count):
    from PIL import Image

    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as module
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.model_executor.models.minimax_h3.conditioning import MiniMaxH3EncoderMediaConditioning

    pipeline = object.__new__(module.MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.load_text_encoder = True
    pipeline.device = torch.device("cpu")
    pipeline.od_config = SimpleNamespace(step_execution=False)
    pipeline._active_turbo_spec = lambda sampling: None
    pipeline._active_native_spec = lambda sampling: None
    pipeline._resolve_task = lambda *args, **kwargs: "ref2va"
    encoded, prepared_inputs, encoded_media = [], [], []
    prepare = module.prepare_encoder_inputs

    def prepare_once(*args, **kwargs):
        result = prepare(*args, **kwargs)
        prepared_inputs.append(result)
        return result

    def encode_text(prepared):
        encoded.append(prepared)
        return torch.full((3, 5120), len(encoded), dtype=torch.bfloat16), torch.zeros(3, dtype=torch.long)

    def encode_media(media):
        encoded_media.append(media)
        return MiniMaxH3EncoderMediaConditioning(
            task=media.task,
            height=media.height,
            width=media.width,
            num_frames=media.num_frames,
            latent_t=media.latent_t,
            audio_t=media.audio_t,
        )

    monkeypatch.setattr(module, "prepare_encoder_inputs", prepare_once)
    pipeline.encode_prompt = encode_text
    pipeline._encode_local_media = encode_media
    pipeline._prepare_encoder_conditioning_inputs = lambda conditioning, sampling, **_: {
        "text_embeddings": conditioning.hidden_states,
        "continuation": (277, 22),
        "num_frames": conditioning.num_frames,
        "task": conditioning.task,
    }
    extra = {"task": "ref2va", "long_video": True, "duration": 75}
    if prompt_count is not None:
        extra["continuation_prompts"] = [f"window {i}" for i in range(prompt_count)]
    sampling = OmniDiffusionSamplingParams(width=64, height=64, fps=24, extra_args=extra)
    request = {"prompt": "main prompt", "multi_modal_data": {"image": Image.new("RGB", (256, 256))}}
    if prompt_count == 6:
        with pytest.raises(OmniClientError, match="exactly 7"):
            pipeline._prepare_request_inputs(request, sampling)
        assert not encoded and not encoded_media
        return
    context = pipeline._prepare_request_inputs(request, sampling)
    assert [item.prompt for item in encoded] == extra.get("continuation_prompts", ["main prompt"])
    assert len(prepared_inputs) == len(encoded_media) == 1
    assert all(item.media is prepared_inputs[0].media for item in encoded)
    if prompt_count is not None:
        texts = context["continuation_text_conditioning"]
        assert len(texts) == 7
        assert texts[0][0] is context["text_embeddings"]
        assert [int(hidden[0, 0]) for hidden, _ in texts] == list(range(1, 8))
    else:
        assert "continuation_text_conditioning" not in context


@pytest.mark.parametrize(
    "task,expected",
    [("ref2va", False), ("t2va", True), ("fl2va", True)],
)
def test_which_tasks_carry_the_tail_as_a_timeline_guide(task, expected):
    # The pipeline passes this to diffuse_continuation, so the loop's two tail
    # carriers are selected here: a frame-0 timeline guide, or Ref2VA's
    # latent_guide reference block.
    assert continuation_uses_timeline_guides(task) is expected


def test_only_ref2va_keeps_the_latent_guide_carrier():
    # Every task the loop accepts is covered by the parametrization above, so a
    # newly admitted task cannot silently default to one carrier.
    assert set(MINIMAX_H3_CONTINUATION_TASKS) == {"ref2va", "t2va", "fl2va"}
    carriers = {task: continuation_uses_timeline_guides(task) for task in MINIMAX_H3_CONTINUATION_TASKS}
    assert [task for task, guided in carriers.items() if not guided] == ["ref2va"]


def test_timeline_guide_windows_keep_their_own_text_origin():
    # The first T2VA/FL2VA window has no guide and is sampled in the task's native
    # layout, where media follows that window's text. A shared origin could only
    # reach the later windows, so none of them takes one.
    kwargs = dict(num_frames=379, latent_t=112, latent_h=2, latent_w=2, audio_t=632, task="t2va")
    texts = [(torch.zeros(length, 4), torch.ones(length)) for length in (5, 9, 7)]
    calls = []

    def sample(**kw):
        calls.append(kw)
        return torch.zeros(1, 24, kw["latent_t"], 2, 2), torch.zeros(2, 32, kw["audio_t"])

    diffuse_continuation(
        sample, kwargs, window_frames=141, overlap_frames=22, text_conditioning=texts, timeline_guides=True
    )
    assert [call["text_embeddings"].shape[0] for call in calls] == [5, 9, 7]
    assert all("media_time_origin" not in call for call in calls)


def test_every_denoise_input_is_carried_or_dropped_after_the_first_window():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _MINIMAX_H3_DENOISE_INPUT_KEYS

    # What a later T2VA/FL2VA window still takes from the request.
    carried = {
        "task",
        "text_embeddings",
        "text_tags",
        "seed",
        "latent_t",
        "latent_h",
        "latent_w",
        "audio_t",
        "num_frames",
        "num_steps",
        "video_shift",
        "audio_shift",
        "base_schedule",
        "pad_seq_len",
        "locked_audio_rows",
    }
    # Continuation refuses control and latent-mask editing, so these are always empty in the loop.
    refused = {"control_rows", "control_context_scale"} | {
        f"{modality}_edit_{part}_rows" for modality in ("video", "audio") for part in ("clean", "mask", "restore_mask")
    }
    first_window = set(_FIRST_WINDOW_CONDITION_KEYS)
    assert not first_window & carried and not first_window & refused
    # A new denoise input must be classified here before the loop can carry or drop it.
    assert set(_MINIMAX_H3_DENOISE_INPUT_KEYS) == carried | refused | first_window


def _local_pipeline(monkeypatch, task, *, load_text_encoder=True):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as module
    from vllm_omni.model_executor.models.minimax_h3.conditioning import MiniMaxH3EncoderMediaConditioning

    pipeline = object.__new__(module.MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.load_text_encoder = load_text_encoder
    pipeline.load_vae_encoder = True
    pipeline.device = torch.device("cpu")
    pipeline.od_config = SimpleNamespace(step_execution=False)
    pipeline._fasth3 = pipeline._vdn = None
    pipeline._active_turbo_spec = lambda sampling: None
    pipeline._has_active_native_lora = lambda sampling: False
    pipeline._resolve_task = lambda *args, **kwargs: task
    pipeline._extract_text_conditioning = lambda prompt: (
        None if load_text_encoder else SimpleNamespace(hidden_states=torch.zeros(3, 5120), token_tags=torch.zeros(3))
    )
    encoded, encoded_media = [], []

    def encode_text(prepared):
        encoded.append(prepared)
        return torch.full((3, 5120), len(encoded), dtype=torch.bfloat16), torch.zeros(3, dtype=torch.long)

    def encode_media(media):
        encoded_media.append(media)
        return MiniMaxH3EncoderMediaConditioning(
            task=media.task,
            height=media.height,
            width=media.width,
            num_frames=media.num_frames,
            latent_t=media.latent_t,
            audio_t=media.audio_t,
        )

    pipeline.encode_prompt = encode_text
    pipeline._encode_local_media = encode_media
    pipeline._prepare_encoder_conditioning_inputs = lambda conditioning, sampling, **_: {
        "text_embeddings": conditioning.hidden_states,
        "continuation": resolve_continuation(sampling.extra_args, task=conditioning.task, step_execution=False),
        "num_frames": conditioning.num_frames,
        "task": conditioning.task,
    }
    return pipeline, encoded, encoded_media


def _local_request(task, *, images=1, **extra):
    from PIL import Image

    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    extra = {"task": task, "long_video": True, "long_video_mode": "continuation", "duration": 30, **extra}
    if not images:
        extra["aspect_ratio"] = "1:1"
    sampling = OmniDiffusionSamplingParams(width=64, height=64, fps=24, extra_args=extra)
    request = {"prompt": "main prompt"}
    if images:
        request["multi_modal_data"] = {"image": [Image.new("RGB", (256, 256)) for _ in range(images)]}
    return request, sampling


def test_only_fl2va_encodes_text_again_for_later_windows():
    assert [task for task in MINIMAX_H3_CONTINUATION_TASKS if later_windows_drop_text_images(task)] == ["fl2va"]


@pytest.mark.parametrize("prompts", [None, ["first", "second", "third", "fourth", "fifth", "sixth"]])
def test_fl2va_later_windows_take_text_without_the_first_frame(monkeypatch, prompts):
    pipeline, encoded, encoded_media = _local_pipeline(monkeypatch, "fl2va")
    request, sampling = _local_request("fl2va", **({"continuation_prompts": prompts} if prompts else {}))
    context = pipeline._prepare_request_inputs(request, sampling)

    texts = context["continuation_text_conditioning"]
    # 30 s at the FL2VA default of 141-frame windows is six windows.
    assert len(texts) == 6 and texts[0][0] is context["text_embeddings"]
    with_image = [item for item in encoded if item.images]
    without = [item for item in encoded if not item.images]
    # The first window keeps the FL2VA presentation with its picture ...
    assert len(with_image) == 1 and with_image[0].media.task == "fl2va" and with_image[0].condition_labels
    # ... and every later one the text-only presentation a continue_from request without an image gets.
    assert all(item.media.task == "t2va" and not item.condition_labels for item in without)
    if prompts is None:
        # One text-only encode serves all later windows, ahead of the request's own encode.
        assert [item.prompt for item in encoded] == ["main prompt", "main prompt"]
        assert not encoded[0].images and encoded[1].images
        assert all(text is texts[1] for text in texts[2:])
        assert int(texts[1][0][0, 0]) == 1 and int(texts[0][0][0, 0]) == 2
    else:
        assert [item.prompt for item in encoded] == prompts
        assert [int(hidden[0, 0]) for hidden, _ in texts] == [1, 2, 3, 4, 5, 6]
    # The media is encoded once, as FL2VA.
    assert len(encoded_media) == 1 and encoded_media[0].task == "fl2va"


def test_t2va_and_single_window_fl2va_encode_their_prompt_once(monkeypatch):
    pipeline, encoded, _ = _local_pipeline(monkeypatch, "t2va")
    context = pipeline._prepare_request_inputs(*_local_request("t2va", images=0))
    assert len(encoded) == 1 and "continuation_text_conditioning" not in context

    pipeline, encoded, _ = _local_pipeline(monkeypatch, "fl2va")
    context = pipeline._prepare_request_inputs(*_local_request("fl2va", duration=5))
    assert len(encoded) == 1 and encoded[0].images and "continuation_text_conditioning" not in context


@pytest.mark.parametrize("task,images,windows", [("t2va", 0, 3), ("fl2va", 1, 6)])
def test_window_prompts_are_refused_under_distributed_text_offload(monkeypatch, task, images, windows):
    prompts = [f"prompt {index}" for index in range(windows)]
    # Text-only prompt encodes in a row are not safe on that backend, so nothing is encoded.
    pipeline, encoded, encoded_media = _local_pipeline(monkeypatch, task)
    pipeline._text_encoder_uses_distributed_layerwise_offload = lambda: True
    with pytest.raises(OmniClientError, match="distributed layerwise offload; send a single prompt"):
        pipeline._prepare_request_inputs(*_local_request(task, images=images, continuation_prompts=prompts))
    assert not encoded and not encoded_media
    # A single prompt for every window stays available there, and so does a one-window request.
    pipeline._prepare_request_inputs(*_local_request(task, images=images))
    assert len(encoded) == (1 if task == "t2va" else 2)
    pipeline._prepare_request_inputs(*_local_request(task, images=images, duration=5, continuation_prompts=["only"]))
    # Without that offload the prompts are encoded, one per window.
    pipeline, encoded, _ = _local_pipeline(monkeypatch, task)
    pipeline._prepare_request_inputs(*_local_request(task, images=images, continuation_prompts=prompts))
    assert [item.prompt for item in encoded] == prompts


def test_distributed_text_offload_is_read_from_the_offload_policy():
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    def refuses(*, text_encoder=True, **config):
        pipeline = object.__new__(MiniMaxH3Pipeline)
        torch.nn.Module.__init__(pipeline)
        if text_encoder:
            pipeline.text_encoder = torch.nn.Identity()
        pipeline.od_config = OmniDiffusionConfig(**config)
        return pipeline._text_encoder_uses_distributed_layerwise_offload()

    assert refuses(enable_distributed_layerwise_offload=True)
    assert not refuses()
    assert not refuses(enable_layerwise_offload=True)
    assert not refuses(enable_distributed_layerwise_offload=True, text_encoder=False)
    # The compact setting picks the distributed backend through a component's weight transfer.
    allgather = {"mode": "layer", "layer_options": {"dit": {"weight_transfer": "allgather"}}}
    assert refuses(diffusion_offload_config={**allgather, "components": ["dit", "text_encoder"]})
    assert not refuses(diffusion_offload_config={**allgather, "components": ["dit"]})
    assert not refuses(diffusion_offload_config={"mode": "layer", "components": ["dit", "text_encoder"]})


def test_fl2va_windows_need_a_local_text_encoder(monkeypatch):
    pipeline, encoded, _ = _local_pipeline(monkeypatch, "fl2va", load_text_encoder=False)
    with pytest.raises(OmniClientError, match="requires a local text encoder"):
        pipeline._prepare_request_inputs(*_local_request("fl2va"))
    assert not encoded
    # One window has no later window to encode for.
    pipeline, _, _ = _local_pipeline(monkeypatch, "fl2va", load_text_encoder=False)
    pipeline._prepare_request_inputs(*_local_request("fl2va", duration=5))


@pytest.mark.parametrize(
    "task,images,extra,match",
    [
        ("t2va", 0, {"continuation_window_frames": 140}, "17n\\+5 frame grid"),
        ("fl2va", 1, {"continuation_overlap_frames": 23}, "17n\\+5 frame grid"),
        ("fl2va", 2, {"frame_indices": [0, -1]}, "first frame only"),
        ("fl2va", 1, {"frame_indices": [-1]}, "first frame only"),
    ],
)
def test_t2va_and_fl2va_options_are_refused_before_any_encoder_pass(monkeypatch, task, images, extra, match):
    pipeline, encoded, encoded_media = _local_pipeline(monkeypatch, task)
    with pytest.raises(OmniClientError, match=match):
        pipeline._prepare_request_inputs(*_local_request(task, images=images, **extra))
    assert not encoded and not encoded_media
