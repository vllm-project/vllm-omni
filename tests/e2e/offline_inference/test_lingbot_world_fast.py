# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""L2 offline smoke for the Lingbot World Fast pipeline."""

from __future__ import annotations

import pytest
import torch

from tests.diffusion.models.lingbot_world_fast.conftest import (
    make_dummy_camera_inputs,
    make_dummy_image,
    make_stubbed_pipeline,
)
from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.lingbot_world_fast.pipeline_lingbot_world_fast import (
    get_lingbot_world_fast_post_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


# ``torch.amp.autocast("cuda", …)`` inside the pipeline requires CUDA at import
# time on hosts where PyTorch is compiled without CUDA support.
if not torch.cuda.is_available():
    pytest.skip(
        'Lingbot World Fast pipeline requires CUDA (torch.amp.autocast("cuda", …))',
        allow_module_level=True,
    )

# Default ``num_frames`` argument; the pipeline floors to ``25`` internally on
# a fresh call (smallest length that maps to a non-empty latent).
_FRESH_NUM_FRAMES = 25
_EXTENSION_NUM_FRAMES = 24

_DIM = 16
_NUM_HEADS = 4
_NUM_LAYERS = 2
_HEAD_DIM = _DIM // _NUM_HEADS


def _build_request(
    *,
    image,
    camera,
    session_id: str,
    num_frames: int,
    prompt: str = "walk forward",
) -> OmniDiffusionRequest:
    multi_modal_data: dict = {"camera": camera}
    if image is not None:
        multi_modal_data["image"] = image
    return OmniDiffusionRequest(
        prompts=[{"prompt": prompt, "multi_modal_data": multi_modal_data}],
        sampling_params=OmniDiffusionSamplingParams(
            height=None,
            width=None,
            num_frames=num_frames,
            seed=42,
            extra_args={"session_id": session_id},
        ),
        request_ids=[f"req-{session_id}"],
    )


@pytest.fixture
def stubbed_pipeline(monkeypatch):
    """Build a stub-backed pipeline and shrink CONFIG['max_area'] for speed."""
    pipeline = make_stubbed_pipeline(
        dim=_DIM,
        num_heads=_NUM_HEADS,
        num_layers=_NUM_LAYERS,
        target_dtype=torch.float32,
    )
    yield pipeline


@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards={"cuda": 1})
def test_session_lifecycle_fresh_then_extension(stubbed_pipeline) -> None:
    """Drive a fresh + extension pair through the pipeline and assert that
    ``LingbotWorldFastState`` advances exactly as the chunk arithmetic prescribes."""
    pipeline = stubbed_pipeline
    session_id = "session-l2-offline"

    # --- Fresh call ---------------------------------------------------------
    camera_fresh = make_dummy_camera_inputs(num_frames=_FRESH_NUM_FRAMES)
    image = make_dummy_image()
    req_fresh = _build_request(
        image=image,
        camera=camera_fresh,
        session_id=session_id,
        num_frames=_FRESH_NUM_FRAMES,
    )

    out_fresh = pipeline.forward(req_fresh)

    assert isinstance(out_fresh, DiffusionOutput)
    assert out_fresh.output is not None
    assert torch.isfinite(out_fresh.output).all(), "Fresh-call video contains NaN/Inf."

    state = pipeline.state
    assert state.is_initialized is True
    assert state.session_id == session_id
    assert state.current_lat_f > 0
    assert state.kv_cache is not None
    assert state.crossattn_cache is not None
    assert state.last_decoded_latent is not None

    fresh_lat_f = state.current_lat_f
    fresh_kv_size = state.kv_cache[0].shape[2]
    frame_seqlen = state.frame_seqlen
    assert frame_seqlen == state.lat_h * state.lat_w // 4
    assert fresh_kv_size == frame_seqlen * fresh_lat_f

    # The spatial dims must come from the input image on the fresh call so the
    # extension branch can later reuse them — make sure they were captured.
    assert state.h is not None and state.w is not None
    assert state.lat_h is not None and state.lat_w is not None

    # --- Extension call -----------------------------------------------------
    camera_ext = make_dummy_camera_inputs(num_frames=_EXTENSION_NUM_FRAMES)
    req_ext = _build_request(
        image=None,
        camera=camera_ext,
        session_id=session_id,
        num_frames=_EXTENSION_NUM_FRAMES,
    )

    out_ext = pipeline.forward(req_ext)

    assert isinstance(out_ext, DiffusionOutput)
    assert out_ext.output is not None
    assert torch.isfinite(out_ext.output).all(), "Extension-call video contains NaN/Inf."

    assert state.session_id == session_id, "Same session_id must not trigger a reset."
    assert state.is_initialized is True
    assert state.current_lat_f > fresh_lat_f, "current_lat_f must advance on extension."
    ext_lat_f = state.current_lat_f - fresh_lat_f
    assert ext_lat_f > 0

    # ``extend_kv_caches`` allocates a fresh tensor of size old + frame_seqlen *
    # new_lat_f and concatenates; assert the trailing slice grew by exactly
    # ``frame_seqlen * ext_lat_f`` for every layer.
    for layer_idx, layer in enumerate(state.kv_cache):
        assert layer.shape == (
            2,
            1,
            fresh_kv_size + frame_seqlen * ext_lat_f,
            _NUM_HEADS,
            _HEAD_DIM,
        ), f"layer {layer_idx} KV cache did not grow by exactly frame_seqlen * ext_lat_f"


@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards={"cuda": 1})
def test_different_session_id_resets_state(stubbed_pipeline) -> None:
    """A new ``session_id`` must hard-reset the cached spatial dims and KV cache"""
    pipeline = stubbed_pipeline
    image = make_dummy_image()

    req_a = _build_request(
        image=image,
        camera=make_dummy_camera_inputs(num_frames=_FRESH_NUM_FRAMES),
        session_id="session-a",
        num_frames=_FRESH_NUM_FRAMES,
    )
    pipeline.forward(req_a)

    assert pipeline.state.session_id == "session-a"
    lat_f_after_a = pipeline.state.current_lat_f
    assert lat_f_after_a > 0

    # A different session_id on the next call must drop the prior KV cache —
    # ``current_lat_f`` resets to ``new_lat_f`` of the second call, not to
    # ``lat_f_after_a + new_lat_f``.
    req_b = _build_request(
        image=make_dummy_image(),
        camera=make_dummy_camera_inputs(num_frames=_FRESH_NUM_FRAMES),
        session_id="session-b",
        num_frames=_FRESH_NUM_FRAMES,
    )
    out_b = pipeline.forward(req_b)
    assert torch.isfinite(out_b.output).all()

    assert pipeline.state.session_id == "session-b"
    assert pipeline.state.current_lat_f == lat_f_after_a, (
        "Stub fresh-call advances by the same fresh new_lat_f, so the reset must "
        "have wiped the prior cumulative count rather than added to it."
    )


@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards={"cuda": 1})
def test_post_process_shapes_videos_for_external_output(stubbed_pipeline) -> None:
    """The model-specific post-process flips ``[C, F, H, W]`` to ``[F, H, W, C]``;
    that's what diffusion engine + serving code downstream expects."""
    pipeline = stubbed_pipeline
    req = _build_request(
        image=make_dummy_image(),
        camera=make_dummy_camera_inputs(num_frames=_FRESH_NUM_FRAMES),
        session_id="session-postprocess",
        num_frames=_FRESH_NUM_FRAMES,
    )

    out = pipeline.forward(req)
    post = get_lingbot_world_fast_post_process_func(pipeline.od_config)
    framed = post(out.output)

    # [C, F, H, W] → [F, H, W, C]
    assert framed.ndim == 4
    assert framed.shape[-1] == out.output.shape[0]
    assert framed.shape[0] == out.output.shape[1]
