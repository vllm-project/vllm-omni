# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3.continuation import (
    diffuse_continuation,
    plan_continuation_windows,
    resolve_continuation,
)
from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence_ref2va_blocks
from vllm_omni.errors import OmniClientError

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


def test_long_ref2va_defaults_to_continuation_with_explicit_full_opt_out():
    assert resolve_continuation({"long_video": True}, task="ref2va", step_execution=False) == (277, 22)
    assert resolve_continuation({}, task="ref2va", step_execution=False) is None
    assert (
        resolve_continuation({"long_video": True, "long_video_mode": "full"}, task="ref2va", step_execution=False)
        is None
    )


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
    pipeline.od_config = None
    pipeline._active_turbo_spec = lambda sampling: None
    pipeline._has_active_native_lora = lambda sampling: False
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
    pipeline._prepare_encoder_conditioning_inputs = lambda conditioning, sampling: {
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
