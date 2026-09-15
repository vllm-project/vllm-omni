# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.diffusion.models.minimax_h3.latent_upscaler import (
    MiniMaxH3LatentResizer3D,
    MiniMaxH3LatentUpscaler,
    MiniMaxH3LatentUpscalerArch,
    MiniMaxH3LatentUpscalerError,
    detect_minimax_h3_upscaler_arch,
    load_minimax_h3_latent_upscaler,
    parse_minimax_h3_latent_upscale_request,
    resolve_minimax_h3_latent_upscale_target,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

# A released checkpoint is 512 channels and 12+12 blocks; these tiny shapes keep
# the same layout so the parameter names, and therefore the strict load, match.
TINY_ARCH = MiniMaxH3LatentUpscalerArch(
    in_channels=24,
    channels=32,
    in_blocks=3,
    out_blocks=3,
    temporal_every=2,
    temporal_kernel=5,
)


class TemporallyLocalResizer(torch.nn.Module):
    """A resizer whose every output frame depends on that frame alone.

    The real network cannot stand in here: its GroupNorms pool statistics over
    the whole clip, so no amount of overlap makes a chunk see what a single
    pass sees. Isolating the chunk arithmetic from that is the point -- the
    index bookkeeping and the overlap ramps are what this module owns.
    """

    temporal_kernel = 5

    def forward(self, latent, *, scale, target_size):
        return torch.nn.functional.interpolate(latent * scale, size=target_size, mode="nearest")


def build_resizer(arch: MiniMaxH3LatentUpscalerArch) -> MiniMaxH3LatentResizer3D:
    torch.manual_seed(0)
    return MiniMaxH3LatentResizer3D(arch).double().eval().requires_grad_(False)


# The real statistics differ per channel and neither centre nor scale is
# trivial, which is what makes the normalization observable in the tests below.
LATENTS_MEAN = tuple(0.1 * index - 1.0 for index in range(24))
LATENTS_STD = tuple(0.5 + 0.1 * index for index in range(24))
NORM = {"latents_mean": LATENTS_MEAN, "latents_std": LATENTS_STD}


def build_upscaler(arch: MiniMaxH3LatentUpscalerArch, **kwargs) -> MiniMaxH3LatentUpscaler:
    return MiniMaxH3LatentUpscaler(
        build_resizer(arch),
        device=torch.device("cpu"),
        dtype=torch.float64,
        **NORM,
        **kwargs,
    )


def test_arch_detection_round_trips_through_a_checkpoint(tmp_path):
    state_dict = {name: tensor.float().contiguous() for name, tensor in build_resizer(TINY_ARCH).state_dict().items()}
    save_file(state_dict, tmp_path / "upscaler.safetensors")

    assert detect_minimax_h3_upscaler_arch(state_dict) == TINY_ARCH
    # A directory with one checkpoint resolves, and the strict load proves the
    # module layout reproduces the checkpoint's parameter names exactly.
    upscaler = load_minimax_h3_latent_upscaler(tmp_path, device=torch.device("cpu"), dtype=torch.float32, **NORM)
    assert upscaler.resizer.arch == TINY_ARCH
    assert upscaler.chunk_overlap == TINY_ARCH.temporal_kernel


def test_a_checkpoint_that_is_not_an_h3_upscaler_is_rejected():
    with pytest.raises(MiniMaxH3LatentUpscalerError, match="rank-5"):
        detect_minimax_h3_upscaler_arch({"conv_in.weight": torch.zeros(32, 24, 3, 3)})
    with pytest.raises(MiniMaxH3LatentUpscalerError, match="attention"):
        detect_minimax_h3_upscaler_arch(
            {
                "conv_in.weight": torch.zeros(32, 24, 3, 3, 3),
                "in_blocks.0.attn.q.weight": torch.zeros(32, 32, 1, 1, 1),
            }
        )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        # Every mode names the same 960x544 -> 1920x1088 upscale.
        ({"scale": 2.0}, (68, 120, 2.0)),
        ({"height": 1088, "width": 1920}, (68, 120, 2.0)),
        ({"megapixels": 2.0}, (68, 120, 2.0)),
        # A target that misses the 32px grid snaps onto it, and the scale hint
        # follows the size that is actually produced.
        ({"scale": 1.5}, (52, 90, 1.5147058823529411)),
    ],
)
def test_target_resolution_covers_every_sizing_mode(kwargs, expected):
    target = resolve_minimax_h3_latent_upscale_target(latent_height=34, latent_width=60, **kwargs)
    assert (target.latent_height, target.latent_width, target.scale) == expected
    assert (target.height, target.width) == (target.latent_height * 16, target.latent_width * 16)


def test_target_resolution_rejects_unserviceable_requests():
    with pytest.raises(MiniMaxH3LatentUpscalerError, match="only upscales"):
        resolve_minimax_h3_latent_upscale_target(latent_height=34, latent_width=60, scale=0.5)
    with pytest.raises(MiniMaxH3LatentUpscalerError, match="exactly one of"):
        resolve_minimax_h3_latent_upscale_target(latent_height=34, latent_width=60, scale=2.0, megapixels=1.0)
    with pytest.raises(MiniMaxH3LatentUpscalerError, match="exactly one of"):
        resolve_minimax_h3_latent_upscale_target(latent_height=34, latent_width=60)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (2, {"scale": 2.0}),
        (1.5, {"scale": 1.5}),
        (None, None),
        (False, None),
        ({"width": 2688, "height": 1536}, {"width": 2688, "height": 1536}),
        ({"megapixels": 4, "align": 64}, {"megapixels": 4.0, "align": 64}),
    ],
)
def test_request_parsing(value, expected):
    assert parse_minimax_h3_latent_upscale_request(value) == expected


def test_request_parsing_rejects_unknown_keys():
    with pytest.raises(MiniMaxH3LatentUpscalerError, match="unknown latent_upscale keys"):
        parse_minimax_h3_latent_upscale_request({"factor": 2})
    with pytest.raises(MiniMaxH3LatentUpscalerError, match="not true"):
        parse_minimax_h3_latent_upscale_request(True)


def test_upscale_resizes_only_the_spatial_axes():
    upscaler = build_upscaler(TINY_ARCH, chunk_frames=0)
    latent = torch.randn(1, 24, 7, 6, 8, dtype=torch.float64)
    target = resolve_minimax_h3_latent_upscale_target(latent_height=6, latent_width=8, scale=2.0)

    upscaled = upscaler.upscale(latent, target)

    assert upscaled.shape == (1, 24, 7, 12, 16)
    assert upscaled.dtype == latent.dtype


def test_upscale_to_the_source_size_returns_the_latent_untouched():
    upscaler = build_upscaler(TINY_ARCH, chunk_frames=0)
    latent = torch.randn(1, 24, 7, 6, 8, dtype=torch.float64)
    target = resolve_minimax_h3_latent_upscale_target(latent_height=6, latent_width=8, scale=1.0)

    assert upscaler.upscale(latent, target) is latent


@pytest.mark.parametrize("frames", [20, 21])
@pytest.mark.parametrize(("chunk_frames", "chunk_overlap"), [(4, 5), (8, 2), (7, 7)])
def test_temporal_chunking_rebuilds_the_whole_clip(frames, chunk_frames, chunk_overlap):
    """Chunking must cover every frame exactly once, with no seam and no gap.

    Against a temporally local resizer the chunked pass is the single pass, so
    any error in the segment bounds or in the overlap ramps -- a dropped frame,
    a double-counted one, a ramp that does not sum to one -- shows up as a
    difference here.
    """
    common = {"device": torch.device("cpu"), "dtype": torch.float64, **NORM}
    resizer = TemporallyLocalResizer()
    full = MiniMaxH3LatentUpscaler(resizer, chunk_frames=0, **common)
    chunked = MiniMaxH3LatentUpscaler(resizer, chunk_frames=chunk_frames, chunk_overlap=chunk_overlap, **common)
    latent = torch.randn(1, 24, frames, 6, 8, dtype=torch.float64)
    target = resolve_minimax_h3_latent_upscale_target(latent_height=6, latent_width=8, scale=2.0)

    torch.testing.assert_close(
        chunked.upscale(latent, target),
        full.upscale(latent, target),
        rtol=0,
        atol=1e-12,
    )


def test_temporal_chunking_ramps_stay_positive_on_every_frame():
    """A frame the ramps zero out everywhere would divide by the clamp floor."""
    upscaler = MiniMaxH3LatentUpscaler(
        TemporallyLocalResizer(),
        device=torch.device("cpu"),
        dtype=torch.float64,
        **NORM,
        chunk_frames=4,
        chunk_overlap=3,
    )
    frames = 20
    weights = torch.zeros(frames, dtype=torch.float64)
    for start in range(0, frames, upscaler.chunk_frames):
        core_end = min(frames, start + upscaler.chunk_frames)
        blend_start = max(0, start - upscaler.chunk_overlap)
        blend_end = min(frames, core_end + upscaler.chunk_overlap)
        weight = upscaler._blend_weights(
            blend_start,
            blend_end,
            start,
            core_end,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        weights[blend_start:blend_end] += weight.flatten()

    assert (weights > 0).all()


def test_weights_park_in_host_memory_between_requests():
    upscaler = build_upscaler(TINY_ARCH, chunk_frames=0)
    assert not upscaler.resident
    assert all(parameter.device.type == "cpu" for parameter in upscaler.parameters())


def build_pipeline(additional_config=None, upscaler=None):
    """A pipeline stub carrying only what the latent-upscale stage reads."""
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = SimpleNamespace(additional_config=additional_config or {})
    pipeline.latent_upscaler = upscaler
    return pipeline


def test_a_request_opts_into_latent_upscaling():
    pipeline = build_pipeline(upscaler=build_upscaler(TINY_ARCH, chunk_frames=0))

    assert pipeline._resolve_latent_upscale({}, latent_h=34, latent_w=60) is None

    target = pipeline._resolve_latent_upscale({"latent_upscale": 2.0}, latent_h=34, latent_w=60)
    assert (target.latent_height, target.latent_width) == (68, 120)
    assert (target.height, target.width) == (1088, 1920)


def test_a_deployment_default_applies_to_every_request_and_a_request_can_decline():
    pipeline = build_pipeline(
        additional_config={"latent_upscale": {"scale": 2.0}},
        upscaler=build_upscaler(TINY_ARCH, chunk_frames=0),
    )

    assert pipeline._resolve_latent_upscale({}, latent_h=34, latent_w=60).scale == 2.0
    assert pipeline._resolve_latent_upscale({"latent_upscale": False}, latent_h=34, latent_w=60) is None
    assert pipeline._resolve_latent_upscale({"latent_upscale": 1.0}, latent_h=34, latent_w=60) is None


def test_upscaling_without_a_checkpoint_is_a_client_error():
    from vllm_omni.errors import OmniClientError

    pipeline = build_pipeline()

    with pytest.raises(OmniClientError, match="latent_upscaler_path"):
        pipeline._resolve_latent_upscale({"latent_upscale": 2.0}, latent_h=34, latent_w=60)
    with pytest.raises(OmniClientError, match="only upscales"):
        build_pipeline(upscaler=build_upscaler(TINY_ARCH, chunk_frames=0))._resolve_latent_upscale(
            {"latent_upscale": 0.5},
            latent_h=34,
            latent_w=60,
        )


def test_the_decode_canvas_follows_the_upscaled_latent():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _minimax_h3_output_canvas

    shape = {"height": 544, "width": 960}
    assert _minimax_h3_output_canvas(shape, None) == (544, 960)

    target = resolve_minimax_h3_latent_upscale_target(latent_height=34, latent_width=60, scale=2.0)
    assert _minimax_h3_output_canvas(shape, target) == (1088, 1920)


def test_the_upscale_stage_is_skipped_when_no_target_is_resolved():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = build_pipeline()
    latent = torch.randn(1, 24, 7, 34, 60)

    assert pipeline._upscaled_latent(latent, None) is latent

    pipeline.latent_upscaler = build_upscaler(TINY_ARCH, chunk_frames=0)
    target = resolve_minimax_h3_latent_upscale_target(latent_height=34, latent_width=60, scale=2.0)
    assert MiniMaxH3Pipeline._upscaled_latent(pipeline, latent, target).shape == (1, 24, 7, 68, 120)


# --------------------------------------------------------------------------
# Second denoise pass (hi-res refine)
# --------------------------------------------------------------------------

REFINE_SHAPE = {"latent_t": 2, "latent_h": 4, "latent_w": 6, "audio_t": 3}


def build_denoise_inputs(pipeline, *, init_latents=None, refine=None, num_steps=11, **overrides):
    kwargs = {
        "task": "t2va",
        "text_embeddings": torch.zeros(9, 8, dtype=torch.float32),
        "text_tags": torch.zeros(9, dtype=torch.int64),
        "seed": 8,
        "num_frames": 5,
        "num_steps": num_steps,
        "video_shift": 12.0,
        "audio_shift": 3.0,
        "base_schedule": None,
        "visual_condition": None,
        "visual_condition_shape": None,
        "audio_condition": None,
        "ref_audio_t": None,
        **REFINE_SHAPE,
        **overrides,
    }
    return pipeline._build_denoise_inputs(init_latents=init_latents, refine=refine, **kwargs)


@pytest.mark.parametrize(
    ("strength", "num_points", "expected_start"),
    [
        (1.0, 51, 0),  # the whole schedule
        (0.4, 51, 30),  # 20 of 50 steps
        (0.5, 11, 5),
        (0.01, 51, 49),  # never fewer than one step
    ],
)
def test_refine_strength_selects_a_schedule_position(strength, num_points, expected_start):
    from vllm_omni.diffusion.models.minimax_h3.latent_upscaler import MiniMaxH3LatentRefineSpec

    assert MiniMaxH3LatentRefineSpec(strength=strength).start_index(num_points) == expected_start


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.4, 0.4), (1, 1.0), ({"strength": 0.25}, 0.25), (None, None), (False, None), ({}, None)],
)
def test_refine_request_parsing(value, expected):
    from vllm_omni.diffusion.models.minimax_h3.latent_upscaler import parse_minimax_h3_latent_refine_request

    spec = parse_minimax_h3_latent_refine_request(value)
    assert (spec.strength if spec is not None else None) == expected


@pytest.mark.parametrize("value", [0.0, -0.5, 1.5, True, {"amount": 0.4}, "half"])
def test_refine_request_parsing_rejects_bad_values(value):
    from vllm_omni.diffusion.models.minimax_h3.latent_upscaler import parse_minimax_h3_latent_refine_request

    with pytest.raises(MiniMaxH3LatentUpscalerError):
        parse_minimax_h3_latent_refine_request(value)


def test_renoising_lands_on_the_flow_the_euler_step_walks_back_down():
    """``x_s = (1 - s) * x0 + s * noise`` is what the scheduler inverts."""
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3.packed_tokens import (
        minimax_h3_pack_audio_latent,
        minimax_h3_patchify_video_latent,
    )

    video_latent = torch.randn(1, 24, 2, 4, 6)
    audio_latent = torch.randn(2, 32, 3)
    noise_video = torch.randn(2 * 2 * 3, 96)
    noise_audio = torch.randn(6, 32)

    video_rows, audio_rows = MiniMaxH3Pipeline._renoise_rows(
        (video_latent, audio_latent),
        noise_video=noise_video,
        noise_audio=noise_audio,
        sigma_video=0.25,
        sigma_audio=0.75,
    )

    video_prior = minimax_h3_patchify_video_latent(video_latent, patch_size=(1, 2, 2))
    audio_prior = minimax_h3_pack_audio_latent(audio_latent)
    torch.testing.assert_close(video_rows, 0.75 * video_prior + 0.25 * noise_video)
    torch.testing.assert_close(audio_rows, 0.25 * audio_prior + 0.75 * noise_audio)

    # At sigma 0 the pass would resume from the latents themselves.
    video_rows, audio_rows = MiniMaxH3Pipeline._renoise_rows(
        (video_latent, audio_latent),
        noise_video=noise_video,
        noise_audio=noise_audio,
        sigma_video=0.0,
        sigma_audio=0.0,
    )
    torch.testing.assert_close(video_rows, video_prior)
    torch.testing.assert_close(audio_rows, audio_prior)


def test_renoising_rejects_latents_that_do_not_fit_the_target_layout():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    with pytest.raises(ValueError, match="refine video rows"):
        MiniMaxH3Pipeline._renoise_rows(
            (torch.randn(1, 24, 2, 4, 6), torch.randn(2, 32, 3)),
            noise_video=torch.randn(2 * 4 * 6, 96),
            noise_audio=torch.randn(6, 32),
            sigma_video=0.5,
            sigma_audio=0.5,
        )


def test_a_refine_pass_resumes_the_schedule_it_would_have_been_on():
    """The second pass must start where the first pass stood at that index.

    Video and audio are shifted apart, so resuming correctly means truncating
    both schedules at the same *index* and re-noising each modality to its own
    sigma there -- not to a shared one.
    """
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3.latent_upscaler import MiniMaxH3LatentRefineSpec

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")

    full = build_denoise_inputs(pipeline)
    video_latent = torch.randn(1, 24, 2, 4, 6)
    audio_latent = torch.randn(2, 32, 3)
    refined = build_denoise_inputs(
        pipeline,
        init_latents=(video_latent, audio_latent),
        refine=MiniMaxH3LatentRefineSpec(strength=0.5),
    )

    start = MiniMaxH3LatentRefineSpec(strength=0.5).start_index(len(full["sigmas_video"]))
    assert refined["sigmas_video"] == full["sigmas_video"][start:]
    assert refined["sigmas_audio"] == full["sigmas_audio"][start:]
    assert refined["sigmas_video"][0] != refined["sigmas_audio"][0]

    expected_video, expected_audio = MiniMaxH3Pipeline._renoise_rows(
        (video_latent, audio_latent),
        noise_video=full["video_rows"],
        noise_audio=full["audio_rows"],
        sigma_video=refined["sigmas_video"][0],
        sigma_audio=refined["sigmas_audio"][0],
    )
    torch.testing.assert_close(refined["video_rows"], expected_video)
    torch.testing.assert_close(refined["audio_rows"], expected_audio)


def test_a_full_strength_refine_replays_the_whole_schedule_from_noise():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3.latent_upscaler import MiniMaxH3LatentRefineSpec

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")

    full = build_denoise_inputs(pipeline)
    refined = build_denoise_inputs(
        pipeline,
        init_latents=(torch.randn(1, 24, 2, 4, 6), torch.randn(2, 32, 3)),
        refine=MiniMaxH3LatentRefineSpec(strength=1.0),
    )

    assert refined["sigmas_video"] == full["sigmas_video"]
    # sigma 1.0 is pure noise, so the prior washes out entirely.
    torch.testing.assert_close(refined["video_rows"], full["video_rows"])


def test_init_latents_without_a_refine_spec_is_a_programming_error():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")

    with pytest.raises(ValueError, match="refine spec"):
        build_denoise_inputs(pipeline, init_latents=(torch.randn(1, 24, 2, 4, 6), torch.randn(2, 32, 3)))


def test_a_request_opts_into_a_refine_pass():
    pipeline = build_pipeline(additional_config={"latent_refine": 0.3})

    assert pipeline._resolve_latent_refine({}).strength == 0.3
    assert pipeline._resolve_latent_refine({"latent_refine": 0.6}).strength == 0.6
    assert pipeline._resolve_latent_refine({"latent_refine": False}) is None
    assert build_pipeline()._resolve_latent_refine({}) is None


def test_a_refine_pass_runs_at_the_upscaled_size():
    from vllm_omni.diffusion.models.minimax_h3.latent_upscaler import MiniMaxH3LatentRefineSpec

    recorded = {}

    def fake_diffuse(**kwargs):
        recorded.update(kwargs)
        return torch.zeros(1), torch.zeros(1)

    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as mod

    pipeline = build_pipeline()
    pipeline.diffuse = fake_diffuse
    target = resolve_minimax_h3_latent_upscale_target(latent_height=34, latent_width=60, scale=2.0)
    context = {
        **{key: None for key in mod._MINIMAX_H3_DENOISE_INPUT_KEYS},
        "task": "t2va",
        "latent_refine": MiniMaxH3LatentRefineSpec(strength=0.4),
        "latent_upscale": target,
    }

    pipeline._refined_latents(torch.zeros(1), torch.zeros(1), context=context, seed=7)

    assert (recorded["latent_h"], recorded["latent_w"]) == (68, 120)
    assert recorded["refine"].strength == 0.4
    assert recorded["seed"] == 7
    assert recorded["init_latents"] is not None


def test_a_pinned_pad_seq_len_does_not_follow_the_pass_that_outgrew_it():
    """pad_seq_len sizes the first pass; the refine pass packs far more rows.

    Carrying the pin over trips the layout's ``seq_len >= used`` check, and the
    error names neither the refine pass nor the size that outgrew it.
    """
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as mod
    from vllm_omni.diffusion.models.minimax_h3.latent_upscaler import MiniMaxH3LatentRefineSpec

    recorded = {}
    pipeline = build_pipeline()
    pipeline.diffuse = lambda **kwargs: (recorded.update(kwargs), (torch.zeros(1), torch.zeros(1)))[1]
    context = {
        **{key: None for key in mod._MINIMAX_H3_DENOISE_INPUT_KEYS},
        "task": "t2va",
        "pad_seq_len": 4096,
        "latent_refine": MiniMaxH3LatentRefineSpec(strength=0.4),
        "latent_upscale": resolve_minimax_h3_latent_upscale_target(latent_height=34, latent_width=60, scale=2.0),
    }

    pipeline._refined_latents(torch.zeros(1), torch.zeros(1), context=context, seed=7)

    assert recorded["pad_seq_len"] is None

    # Without an upscale the layout is unchanged, so the pin still applies.
    recorded.clear()
    context["latent_upscale"] = None
    pipeline._refined_latents(torch.zeros(1), torch.zeros(1), context=context, seed=7)
    assert recorded["pad_seq_len"] == 4096


def test_the_hi_res_route_runs_a_shortened_pass_at_the_new_size():
    """diffuse() twice: a cheap first pass, then a partial one after the upscale.

    Against a stand-in DiT this covers what unit-testing the pieces cannot --
    that the packed layout, the branch masks and the row bookkeeping all agree
    at a resolution the first pass never used.
    """
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3.latent_upscaler import MiniMaxH3LatentRefineSpec

    from .test_minimax_h3_step_execution import _SegmentMeanModel

    calls = []

    class CountingModel(_SegmentMeanModel):
        def __call__(self, **kwargs):
            calls.append(kwargs["unique_timesteps"])
            return super().__call__(**kwargs)

    model = CountingModel()
    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.transformer = model
    pipeline._transformer_for_task = lambda task: model

    kwargs = {
        "task": "t2va",
        "text_embeddings": torch.zeros(9, 8, dtype=torch.float32),
        "text_tags": torch.zeros(9, dtype=torch.int64),
        "seed": 8,
        "num_frames": 5,
        "num_steps": 11,
        "video_shift": 12.0,
        "audio_shift": 3.0,
        "base_schedule": None,
        "visual_condition": None,
        "visual_condition_shape": None,
        "audio_condition": None,
        "ref_audio_t": None,
        **REFINE_SHAPE,
    }

    video_latent, audio_latent = pipeline.diffuse(**kwargs)
    assert video_latent.shape == (1, 24, 2, 4, 6)
    first_pass_steps = len(calls)
    assert first_pass_steps == 10

    upscaled = torch.nn.functional.interpolate(video_latent, size=(2, 8, 12), mode="nearest")
    refined_video, refined_audio = pipeline.diffuse(
        **{**kwargs, "latent_h": 8, "latent_w": 12},
        init_latents=(upscaled, audio_latent),
        refine=MiniMaxH3LatentRefineSpec(strength=0.5),
    )

    assert refined_video.shape == (1, 24, 2, 8, 12)
    assert refined_audio.shape == audio_latent.shape
    # Half the schedule, so half the forwards -- at the expensive size.
    assert len(calls) - first_pass_steps == 5
    assert torch.isfinite(refined_video).all()


def test_refining_fl2va_re_encodes_the_keyframe_at_the_new_size():
    """FL2VA condition rows are sized by the output latent, so they must move.

    The packed layout derives the keyframe's row count from ``latent_h`` and
    ``latent_w``; reusing the first pass's condition at a larger output size
    would build a sequence whose condition block no longer matches its rows.
    """
    from PIL import Image

    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as mod
    from vllm_omni.diffusion.models.minimax_h3.latent_upscaler import MiniMaxH3LatentRefineSpec

    from .test_minimax_h3_step_execution import _SegmentMeanModel

    model = _SegmentMeanModel()
    pipeline = build_pipeline()
    pipeline.device = torch.device("cpu")
    pipeline.transformer = model
    pipeline._transformer_for_task = lambda task: model

    encoded_sizes: list[tuple[int, int]] = []

    def fake_encode(images, prepared_videos, *, video_count):
        encoded_sizes.extend((image.width, image.height) for image in images)
        shapes = [(1, image.height // 16, image.width // 16) for image in images]
        rows = sum(t * (h // 2) * (w // 2) for t, h, w in shapes)
        return torch.zeros(rows, 96, dtype=torch.float32), shapes

    pipeline._encode_visual_conditions = fake_encode

    # The first pass ran at 64x96 latent cells 4x6; the refine runs at 8x12.
    context = {
        **{key: None for key in mod._MINIMAX_H3_DENOISE_INPUT_KEYS},
        "task": "fl2va",
        "text_embeddings": torch.zeros(9, 8, dtype=torch.float32),
        "text_tags": torch.zeros(9, dtype=torch.int64),
        "seed": 8,
        "latent_t": 2,
        "latent_h": 4,
        "latent_w": 6,
        "audio_t": 3,
        "num_frames": 5,
        "num_steps": 11,
        "video_shift": 12.0,
        "audio_shift": 3.0,
        "keyframe_frame_indices": [0],
        "latent_refine": MiniMaxH3LatentRefineSpec(strength=0.5),
        "latent_upscale": resolve_minimax_h3_latent_upscale_target(latent_height=4, latent_width=6, scale=2.0),
        "keyframe_images": [Image.new("RGB", (512, 512))],
    }

    for seed in (8, 9):
        video_latent, audio_latent = MiniMaxH3Pipeline._refined_latents(
            pipeline,
            torch.randn(1, 24, 2, 8, 12),
            torch.randn(2, 32, 3),
            context=context,
            seed=seed,
        )
        assert video_latent.shape == (1, 24, 2, 8, 12)
        assert audio_latent.shape == (2, 32, 3)

    # 4x6 latent cells at 2x is 192x128 pixels, and the encode -- which
    # broadcasts across the DiT group -- is shared by every output.
    assert encoded_sizes == [(192, 128)]


def test_refining_ref2va_keeps_its_references_at_their_own_size():
    """REF2VA references carry their own latent shape, so only the video moves.

    Unlike FL2VA there is nothing to re-encode, but the packed layout still has
    to place fixed-size reference blocks beside a video block that grew.
    """
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3.latent_upscaler import MiniMaxH3LatentRefineSpec

    from .test_minimax_h3_step_execution import _SegmentMeanModel

    model = _SegmentMeanModel()
    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.transformer = model
    pipeline._transformer_for_task = lambda task: model

    ref_blocks = [
        {"kind": "image", "latent_h": 4, "latent_w": 6},
        {"kind": "audio", "ref_audio_t": 3},
    ]
    kwargs = {
        "task": "ref2va",
        "text_embeddings": torch.zeros(9, 8, dtype=torch.float32),
        "text_tags": torch.zeros(9, dtype=torch.int64),
        "seed": 8,
        "latent_t": 2,
        "latent_h": 8,
        "latent_w": 12,
        "audio_t": 3,
        "num_frames": 5,
        "num_steps": 11,
        "video_shift": 12.0,
        "audio_shift": 3.0,
        "base_schedule": None,
        "visual_condition": torch.zeros(1 * 2 * 3, 96, dtype=torch.float32),
        "visual_condition_shape": (1, 4, 6),
        "visual_condition_shapes": [(1, 4, 6)],
        "audio_condition": torch.zeros(3 * 2, 32, dtype=torch.float32),
        "audio_condition_lengths": [3],
        "ref_audio_t": 3,
        "ref_blocks": ref_blocks,
    }

    video_latent, audio_latent = pipeline.diffuse(
        **kwargs,
        init_latents=(torch.randn(1, 24, 2, 8, 12), torch.randn(2, 32, 3)),
        refine=MiniMaxH3LatentRefineSpec(strength=0.5),
    )

    assert video_latent.shape == (1, 24, 2, 8, 12)
    assert audio_latent.shape == (2, 32, 3)
    assert torch.isfinite(video_latent).all()


def test_the_network_is_fed_one_normalization_below_the_pipeline_latent():
    """The checkpoint's space, not the pipeline's -- and the round trip undoes it.

    A vLLM-Omni H3 latent is already normalized, but the released upscaler was
    trained on that tensor normalized *again* by the VAE's per-channel
    statistics. Feeding it the pipeline latent directly inflates the result
    about 5x, which decodes as a magenta grid at one tile per latent cell, so
    this pins both halves of the conversion.
    """
    seen = {}

    class RecordingResizer(torch.nn.Module):
        temporal_kernel = 5

        def forward(self, latent, *, scale, target_size):
            seen["input"] = latent.clone()
            return torch.nn.functional.interpolate(latent, size=target_size, mode="nearest")

    upscaler = MiniMaxH3LatentUpscaler(
        RecordingResizer(),
        device=torch.device("cpu"),
        dtype=torch.float64,
        **NORM,
        chunk_frames=0,
    )
    latent = torch.randn(1, 24, 3, 6, 8, dtype=torch.float64)
    target = resolve_minimax_h3_latent_upscale_target(latent_height=6, latent_width=8, scale=2.0)

    upscaled = upscaler.upscale(latent, target)

    mean = torch.tensor(LATENTS_MEAN, dtype=torch.float64).view(1, -1, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, dtype=torch.float64).view(1, -1, 1, 1, 1)
    # In: the pipeline latent normalized again.
    torch.testing.assert_close(seen["input"], (latent - mean) / std)
    # Out: back in the pipeline's space. Nearest-neighbour resizing is exact, so
    # the round trip reproduces the source values rather than merely their scale.
    torch.testing.assert_close(
        upscaled,
        torch.nn.functional.interpolate(latent, size=(3, 12, 16), mode="nearest"),
    )


def test_latent_statistics_must_be_usable():
    resizer = build_resizer(TINY_ARCH)
    common = {"device": torch.device("cpu"), "dtype": torch.float64}
    with pytest.raises(MiniMaxH3LatentUpscalerError, match="length mismatch"):
        MiniMaxH3LatentUpscaler(resizer, latents_mean=(0.0,), latents_std=(1.0, 1.0), **common)
    with pytest.raises(MiniMaxH3LatentUpscalerError, match="must not contain zeros"):
        MiniMaxH3LatentUpscaler(resizer, latents_mean=(0.0,), latents_std=(0.0,), **common)


def test_a_latent_with_the_wrong_channel_count_is_rejected():
    upscaler = build_upscaler(TINY_ARCH, chunk_frames=0)
    target = resolve_minimax_h3_latent_upscale_target(latent_height=6, latent_width=8, scale=2.0)

    with pytest.raises(MiniMaxH3LatentUpscalerError, match="latent channels"):
        upscaler.upscale(torch.randn(1, 16, 3, 6, 8, dtype=torch.float64), target)
