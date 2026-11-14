# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from fractions import Fraction
from typing import cast

import pytest
import torch

from vllm_omni.diffusion.postprocess.frame_timing import FrameInterpolationPlan
from vllm_omni.diffusion.postprocess.rife_interpolator import FrameInterpolator, Model, resample_video_tensor

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class LinearModel:
    def __init__(self) -> None:
        self.timesteps: list[float] = []

    def device(self) -> torch.device:
        return torch.device("cpu")

    def inference(
        self, img0: torch.Tensor, img1: torch.Tensor, scale: float = 1.0, timestep: float = 0.5
    ) -> torch.Tensor:
        self.timesteps.append(timestep)
        return img0 * (1 - timestep) + img1 * timestep


@pytest.fixture
def linear_model(monkeypatch: pytest.MonkeyPatch) -> LinearModel:
    model = LinearModel()
    monkeypatch.setattr(FrameInterpolator, "_ensure_model_loaded", lambda *args, **kwargs: cast(Model, model))
    return model


@pytest.mark.parametrize("source_frames,output_frames", [(192, 720), (362, 736), (3, 3), (2, 4096)])
def test_timeline_matches_rational_clock(source_frames: int, output_frames: int) -> None:
    plan = FrameInterpolationPlan(source_frames, Fraction(30000, 1001), 48, Fraction(output_frames, 48))
    assert plan.output_frames == output_frames
    assert plan.source_duration == Fraction(source_frames * 1001, 30000)
    assert plan.output_duration * plan.output_fps == output_frames
    for slot in range(output_frames):
        position = min(Fraction(slot * source_frames, output_frames), Fraction(source_frames - 1))
        left, right, timestep = plan.source_position(slot)
        assert left == position.numerator // position.denominator
        assert right == min(left + (position.denominator != 1), source_frames - 1)
        assert timestep == float(position - left)


@pytest.mark.parametrize(
    "args",
    [
        (1, 24, 48, 1),
        (2, 0, 48, 1),
        (2, 24, -1, 1),
        (2, 24, 48, 0),
        (2, 24, 24, Fraction(1, 7)),
        (4, 24, 2, 1),
        (2, 24, 4097, 1),
        (True, 24, 48, 1),
        (2, 24.0, 48, 1),
        (2, 24, True, 1),
    ],
)
def test_invalid_clock_is_rejected(args: tuple) -> None:
    with pytest.raises((ValueError, TypeError)):
        FrameInterpolationPlan(*args)


def test_slot_bounds() -> None:
    plan = FrameInterpolationPlan(2, 24, 5, 1)
    for slot in (-1, 5, True, 0.5):
        with pytest.raises((ValueError, TypeError, IndexError)):
            plan.source_position(slot)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.uint8])
def test_fractional_sampling_and_tail_hold(linear_model: LinearModel, dtype: torch.dtype) -> None:
    values = torch.tensor([0.0, 1.0]) if dtype != torch.uint8 else torch.tensor([0, 255])
    video = values.to(dtype).reshape(1, 1, 2, 1, 1).expand(2, 3, 2, 2, 3)
    plan = FrameInterpolationPlan(2, 24, 5, 1)
    output = resample_video_tensor(video, plan)
    expected = torch.tensor([0.0, 0.4, 0.8, 1.0, 1.0])
    if dtype == torch.uint8:
        expected = (expected * 255).round()
    assert output.shape == (2, 3, 5, 2, 3)
    assert output.dtype == dtype
    assert torch.equal(output[0, 0, :, 0, 0], expected.to(dtype))
    assert linear_model.timesteps == [0.4, 0.8]
    assert torch.equal(video[:, :, -1], output[:, :, -1])
    output.zero_()
    assert video.max() > 0


def test_invalid_input_and_budget_do_not_load_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_load(*args, **kwargs):
        pytest.fail("invalid requests must fail before loading RIFE")

    monkeypatch.setattr(FrameInterpolator, "_ensure_model_loaded", unexpected_load)
    plan = FrameInterpolationPlan(2, 24, 5, 1)
    valid = torch.zeros(1, 3, 2, 2, 2)
    invalid = [
        valid[0],
        valid[:, :1],
        valid[:, :, :1],
        valid[:, :, :, :0],
        valid.to(torch.int32),
        valid - 1,
        valid + float("nan"),
        valid + float("inf"),
    ]
    for video in invalid:
        with pytest.raises((ValueError, TypeError)):
            FrameInterpolator().resample_tensor(video, plan)
    with pytest.raises(ValueError):
        FrameInterpolator().resample_tensor(valid, plan, max_output_bytes=239)
    for scale in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            FrameInterpolator().resample_tensor(valid, plan, scale=scale)


def test_output_budget_boundary(linear_model: LinearModel) -> None:
    output = FrameInterpolator().resample_tensor(
        torch.zeros(1, 3, 2, 2, 2), FrameInterpolationPlan(2, 24, 5, 1), max_output_bytes=240
    )
    assert output.shape == (1, 3, 5, 2, 2)


def test_cancellation_propagates_between_slots(linear_model: LinearModel) -> None:
    checks = 0

    def check_cancel() -> None:
        nonlocal checks
        checks += 1
        if linear_model.timesteps:
            raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError, match="cancelled"):
        FrameInterpolator().resample_tensor(
            torch.zeros(1, 3, 2, 2, 2), FrameInterpolationPlan(2, 24, 5, 1), check_cancel=check_cancel
        )
    assert linear_model.timesteps == [0.4]
    assert checks >= 2


def test_model_failure_propagates(linear_model: LinearModel, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(linear_model, "inference", fail)
    with pytest.raises(RuntimeError, match="inference failed"):
        resample_video_tensor(torch.zeros(1, 3, 2, 2, 2), FrameInterpolationPlan(2, 24, 5, 1))


def test_real_rife_fractional_timesteps(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import av

    torch.manual_seed(7)
    model = Model().eval()
    monkeypatch.setattr(FrameInterpolator, "_ensure_model_loaded", lambda *args, **kwargs: model)
    timesteps: list[float] = []

    def record_timestep(module, args):
        timesteps.append(float(args[0][0, 14, 0, 0]))

    handles = [
        block.register_forward_pre_hook(record_timestep)
        for block in (model.flownet.block0, model.flownet.block1, model.flownet.block2, model.flownet.block3)
    ]
    video = torch.rand(1, 3, 2, 32, 32)
    try:
        output = FrameInterpolator().resample_tensor(video, FrameInterpolationPlan(2, 24, 5, 1))
    finally:
        for handle in handles:
            handle.remove()
    assert timesteps == pytest.approx([0.4] * 4 + [0.8] * 4)
    assert output.shape == (1, 3, 5, 32, 32)
    assert torch.isfinite(output).all()
    assert torch.equal(output[:, :, 0], video[:, :, 0])
    assert torch.equal(output[:, :, -1], video[:, :, -1])
    path = tmp_path / "resampled.mp4"
    plan = FrameInterpolationPlan(2, 24, 5, 1)
    rgb = (output[0].permute(1, 2, 3, 0) * 255).round().to(torch.uint8).numpy()
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=plan.output_fps)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for index, array in enumerate(rgb):
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            frame.pts, frame.time_base = index, 1 / plan.output_fps
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(path)) as container:
        frames = list(container.decode(video=0))
        assert len(frames) == plan.output_frames
        assert [frame.pts * frame.time_base for frame in frames] == [Fraction(j, 5) for j in range(5)]
        assert container.streams.video[0].duration * container.streams.video[0].time_base == plan.output_duration


def test_legacy_exponent_path(linear_model: LinearModel) -> None:
    video = torch.tensor([0.0, 1.0]).reshape(1, 1, 2, 1, 1).expand(1, 3, 2, 2, 2)
    result, multiplier = FrameInterpolator().interpolate_tensor(video, exp=2)
    assert multiplier == 4
    assert torch.equal(result[0, 0, :, 0, 0], torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
