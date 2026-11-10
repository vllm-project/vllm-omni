# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import numpy as np
import pytest
import torch

import vllm_omni.diffusion.diffusion_engine as diffusion_engine_module
import vllm_omni.diffusion.ipc as diffusion_ipc
import vllm_omni.diffusion.postprocess.media as media_postprocess
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig, VideoOutputTransportConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.ipc import pack_diffusion_output_shm, unpack_diffusion_output_shm
from vllm_omni.diffusion.media import (
    DiffusionMediaOutput,
    FloatVideoConsumer,
    VideoMediaOutput,
    VideoTensorEncoding,
    VideoTensorLayout,
    VideoTensorSpec,
    VideoTransportConstraints,
    VideoValueRange,
)
from vllm_omni.diffusion.postprocess.device_reduction import prepare_diffusion_media_for_transport
from vllm_omni.diffusion.postprocess.media import finalize_diffusion_media
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _video(
    tensor: torch.Tensor,
    *,
    encoding: VideoTensorEncoding,
    value_range: VideoValueRange,
    consumers: frozenset[FloatVideoConsumer] = frozenset(),
) -> VideoMediaOutput:
    layout = VideoTensorLayout.BTHWC if encoding is VideoTensorEncoding.UINT8_FRAMES else VideoTensorLayout.BCTHW
    return VideoMediaOutput(
        tensor=tensor,
        spec=VideoTensorSpec(
            layout=layout,
            encoding=encoding,
            value_range=value_range,
        ),
        constraints=VideoTransportConstraints(pending_float_consumers=consumers),
    )


def _prepared_media(video: VideoMediaOutput) -> DiffusionMediaOutput:
    return DiffusionMediaOutput(video=video, prepared_for_transport=True)


def _od_config(*, enabled: bool = False) -> OmniDiffusionConfig:
    config = OmniDiffusionConfig(
        model=None,
        video_output_transport=VideoOutputTransportConfig(enable_device_postprocess=enabled),
    )
    config.model_class_name = "WanPipeline"
    return config


def test_finalizer_rejects_media_that_skipped_runner_preparation() -> None:
    media = DiffusionMediaOutput(
        video=_video(
            torch.randn(1, 3, 2, 4, 5),
            encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
            value_range=VideoValueRange.NEGATIVE_ONE_TO_ONE,
        )
    )

    with pytest.raises(ValueError, match="before transport preparation"):
        finalize_diffusion_media(media, sampling_params=OmniDiffusionSamplingParams(output_type="np"))


def test_uint8_media_finalization_preserves_frames() -> None:
    frames = torch.randint(0, 256, (1, 4, 8, 8, 3), dtype=torch.uint8)

    output = finalize_diffusion_media(
        _prepared_media(
            _video(
                frames,
                encoding=VideoTensorEncoding.UINT8_FRAMES,
                value_range=VideoValueRange.ZERO_TO_255,
            )
        ),
        sampling_params=OmniDiffusionSamplingParams(output_type="np"),
    )

    np.testing.assert_array_equal(output["payload"]["video"], frames.numpy())
    assert output["metadata"] == {}


def test_float_media_finalization_uses_the_declared_range() -> None:
    video = torch.tensor([[[[[-1.0, 0.0, 1.0]]]]]).expand(1, 3, 1, 1, 3).contiguous()

    output = finalize_diffusion_media(
        _prepared_media(
            _video(
                video,
                encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
                value_range=VideoValueRange.NEGATIVE_ONE_TO_ONE,
            )
        ),
        sampling_params=OmniDiffusionSamplingParams(output_type="np"),
    )

    expected = (video.permute(0, 2, 3, 4, 1).numpy() * 0.5 + 0.5).clip(0.0, 1.0)
    np.testing.assert_array_equal(output["payload"]["video"], expected)


def test_zero_to_one_float_media_is_not_denormalized() -> None:
    video = torch.tensor([[[[[0.0, 0.5, 1.0]]]]]).expand(1, 3, 1, 1, 3).contiguous()
    media = _prepared_media(
        _video(
            video,
            encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
            value_range=VideoValueRange.ZERO_TO_ONE,
        )
    )

    output = finalize_diffusion_media(media, sampling_params=OmniDiffusionSamplingParams(output_type="np"))

    expected = video.permute(0, 2, 3, 4, 1).numpy()
    np.testing.assert_array_equal(output["payload"]["video"], expected)


def test_presentation_is_resolved_outside_the_tensor_descriptor() -> None:
    video = torch.zeros(1, 3, 2, 4, 5)
    media = _prepared_media(
        _video(
            video,
            encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
            value_range=VideoValueRange.NEGATIVE_ONE_TO_ONE,
        )
    )

    output = finalize_diffusion_media(media, sampling_params=OmniDiffusionSamplingParams(output_type="pil"))

    assert "presentation" not in media.video.__dataclass_fields__
    assert len(output["payload"]["video"]) == 1
    assert len(output["payload"]["video"][0]) == 2


def test_frame_interpolation_consumes_its_constraint(monkeypatch: pytest.MonkeyPatch) -> None:
    video = torch.randn(1, 3, 2, 4, 5)
    interpolated = torch.randn(1, 3, 3, 4, 5)
    monkeypatch.setattr(
        media_postprocess,
        "interpolate_video_tensor",
        lambda *args, **kwargs: (interpolated, 2.0),
    )

    output = finalize_diffusion_media(
        _prepared_media(
            _video(
                video,
                encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
                value_range=VideoValueRange.NEGATIVE_ONE_TO_ONE,
                consumers=frozenset({FloatVideoConsumer.FRAME_INTERPOLATION}),
            )
        ),
        sampling_params=OmniDiffusionSamplingParams(output_type="np", enable_frame_interpolation=True),
    )

    assert output["payload"]["video"].shape == (1, 3, 4, 5, 3)
    assert output["metadata"] == {"video": {"video_fps_multiplier": 2.0}}


def test_uint8_media_rejects_a_pending_float_consumer() -> None:
    media = _prepared_media(
        _video(
            torch.zeros(1, 2, 4, 5, 3, dtype=torch.uint8),
            encoding=VideoTensorEncoding.UINT8_FRAMES,
            value_range=VideoValueRange.ZERO_TO_255,
            consumers=frozenset({FloatVideoConsumer.FRAME_INTERPOLATION}),
        )
    )

    with pytest.raises(ValueError, match="cannot have pending float consumers"):
        media.validate()


def test_unconsumed_float_constraint_fails_before_public_formatting() -> None:
    media = _prepared_media(
        _video(
            torch.randn(1, 3, 2, 4, 5),
            encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
            value_range=VideoValueRange.NEGATIVE_ONE_TO_ONE,
            consumers=frozenset({FloatVideoConsumer.VIDEO_GUARDRAILS}),
        )
    )

    with pytest.raises(ValueError, match="pending float consumers"):
        finalize_diffusion_media(media, sampling_params=OmniDiffusionSamplingParams(output_type="np"))


def test_engine_routes_typed_media_exclusively(monkeypatch: pytest.MonkeyPatch) -> None:
    media = _prepared_media(
        _video(
            torch.zeros(1, 1, 2, 2, 3, dtype=torch.uint8),
            encoding=VideoTensorEncoding.UINT8_FRAMES,
            value_range=VideoValueRange.ZERO_TO_255,
        )
    )
    request = OmniDiffusionRequest(
        prompt="test",
        sampling_params=OmniDiffusionSamplingParams(output_type="np"),
        request_id="request-0",
    )
    engine = object.__new__(DiffusionEngine)
    engine.od_config = _od_config()
    engine.post_process_func = lambda _: pytest.fail("legacy postprocessor was called")
    engine._post_process_accepts_sampling_params = False

    finalized = {"payload": {"video": np.zeros((1, 1, 2, 2, 3), dtype=np.uint8)}, "metadata": {}}

    def _format_outputs(
        *,
        request: object,
        od_config: object,
        diffusion_output: object,
        output_data: object,
        postprocess_output: object,
    ) -> list[object]:
        del request, od_config, diffusion_output, output_data
        return [postprocess_output]

    monkeypatch.setattr(diffusion_engine_module, "finalize_diffusion_media", lambda *args, **kwargs: finalized)
    monkeypatch.setattr(diffusion_engine_module, "normalize_diffusion_postprocess_output", lambda value: value)
    monkeypatch.setattr(diffusion_engine_module, "format_diffusion_outputs", _format_outputs)

    result = engine.postprocess_output(request, DiffusionOutput(media=media))

    assert result == [finalized]


def test_typed_media_crosses_runner_ipc_and_engine_without_legacy_postprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampling_params = OmniDiffusionSamplingParams(output_type="np", num_outputs_per_prompt=1)
    raw_media = DiffusionMediaOutput(
        video=_video(
            torch.linspace(-1.0, 1.0, 1 * 3 * 2 * 4 * 5).reshape(1, 3, 2, 4, 5),
            encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
            value_range=VideoValueRange.NEGATIVE_ONE_TO_ONE,
        )
    )
    prepared = prepare_diffusion_media_for_transport(
        raw_media,
        od_config=_od_config(enabled=True),
        sampling_params=sampling_params,
    )
    diffusion_output = DiffusionOutput(media=prepared)
    monkeypatch.setattr(diffusion_ipc, "_SHM_TENSOR_THRESHOLD", 1)
    pack_diffusion_output_shm(diffusion_output)
    unpack_diffusion_output_shm(diffusion_output)

    engine = object.__new__(DiffusionEngine)
    engine.od_config = _od_config()
    engine.post_process_func = lambda _: pytest.fail("legacy postprocessor was called")
    engine._post_process_accepts_sampling_params = False
    request = OmniDiffusionRequest(prompt="test", sampling_params=sampling_params, request_id="request-0")

    outputs = engine.postprocess_output(request, diffusion_output)

    assert len(outputs) == 1
    assert outputs[0].images[0].dtype == np.uint8
    assert outputs[0].images[0].shape == (1, 2, 4, 5, 3)


def test_diffusion_output_rejects_typed_and_legacy_output_together() -> None:
    media = _prepared_media(
        _video(
            torch.zeros(1, 1, 2, 2, 3, dtype=torch.uint8),
            encoding=VideoTensorEncoding.UINT8_FRAMES,
            value_range=VideoValueRange.ZERO_TO_255,
        )
    )
    with pytest.raises(ValueError, match="both media and legacy output"):
        DiffusionOutput(output=torch.zeros(1), media=media)


def test_diffusion_output_rejects_postprocessor_on_typed_media() -> None:
    media = _prepared_media(
        _video(
            torch.zeros(1, 1, 2, 2, 3, dtype=torch.uint8),
            encoding=VideoTensorEncoding.UINT8_FRAMES,
            value_range=VideoValueRange.ZERO_TO_255,
        )
    )

    with pytest.raises(ValueError, match="model-specific post_process_func"):
        DiffusionOutput(media=media, post_process_func=lambda value: value)


def test_diffusion_output_preserves_the_legacy_positional_constructor() -> None:
    # ``media`` was originally declared right after ``output``, which broke
    # ``DiffusionOutput(output, trajectory_timesteps)`` by binding the second
    # argument to media. It must stay last so out-of-tree positional callers keep
    # the pre-existing field order.
    output_tensor = torch.randn(1)
    timesteps = torch.tensor([1, 2, 3])

    output = DiffusionOutput(output_tensor, timesteps)

    assert output.output is output_tensor
    assert output.trajectory_timesteps is timesteps
    assert output.media is None


def test_frame_interpolation_uses_the_declared_range_negative_one_to_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # RIFE infers [0,1] vs [-1,1] from the tensor min/max, so an all-nonnegative
    # [-1,1] clip is misread as [0,1] and interpolated in the wrong space. The
    # interpolator must receive the unit-range form per the declared spec.
    video = torch.full((1, 3, 2, 3, 3), 0.25)
    media = DiffusionMediaOutput(
        video=VideoMediaOutput(
            tensor=video,
            spec=VideoTensorSpec(
                layout=VideoTensorLayout.BCTHW,
                encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
                value_range=VideoValueRange.NEGATIVE_ONE_TO_ONE,
            ),
            constraints=VideoTransportConstraints(
                pending_float_consumers=frozenset({FloatVideoConsumer.FRAME_INTERPOLATION})
            ),
        ),
        prepared_for_transport=True,
    )

    seen: list[torch.Tensor] = []

    def _square_interpolator(tensor: torch.Tensor, **_kwargs: object) -> tuple[torch.Tensor, float]:
        seen.append(tensor)
        return tensor * tensor, 2.0

    monkeypatch.setattr(media_postprocess, "interpolate_video_tensor", _square_interpolator)

    output = finalize_diffusion_media(
        media,
        sampling_params=OmniDiffusionSamplingParams(output_type="np", enable_frame_interpolation=True),
    )

    # The interpolator is fed the unit-range form 0.25→0.625, not raw 0.25.
    assert len(seen) == 1
    assert seen[0].shape == video.shape
    assert float(seen[0][0, 0, 0, 0, 0]) == pytest.approx(0.625, abs=1e-6)
    # Squaring gives 0.390625; restored to [-1,1] (-0.21875) by the fix, then the
    # existing denormalize maps it back to [0,1] as 0.390625 in the public frames.
    frames = output["payload"]["video"]
    assert np.allclose(np.asarray(frames), 0.390625, atol=1e-6)


def test_frame_interpolation_uses_the_declared_range_zero_to_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With a declared [0,1] range, the all-0.25 clip is already in unit space, so
    # squaring gives 0.390625 and there is no restore step.
    video = torch.full((1, 3, 2, 3, 3), 0.25)
    media = DiffusionMediaOutput(
        video=VideoMediaOutput(
            tensor=video,
            spec=VideoTensorSpec(
                layout=VideoTensorLayout.BCTHW,
                encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
                value_range=VideoValueRange.ZERO_TO_ONE,
            ),
            constraints=VideoTransportConstraints(
                pending_float_consumers=frozenset({FloatVideoConsumer.FRAME_INTERPOLATION})
            ),
        ),
        prepared_for_transport=True,
    )

    def _square_interpolator(tensor: torch.Tensor, **_kwargs: object) -> tuple[torch.Tensor, float]:
        return tensor * tensor, 2.0

    monkeypatch.setattr(media_postprocess, "interpolate_video_tensor", _square_interpolator)

    output = finalize_diffusion_media(
        media,
        sampling_params=OmniDiffusionSamplingParams(output_type="np", enable_frame_interpolation=True),
    )

    # 0.25 is already [0,1]; squaring gives 0.0625, and no denormalize is applied.
    frames = output["payload"]["video"]
    assert np.allclose(np.asarray(frames), 0.0625, atol=1e-6)


def test_frame_interpolation_clamps_zero_to_one_overshoot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A declared [0,1] clip whose sample is almost entirely inside [0,1] — but
    # with one bf16 value overshooting 1 — is still in unit space, so it must be
    # clamped to [0,1] before RIFE and interpolated as [0,1]. Without the clamp
    # the interpolator classifies it as [-1,1] (amax > 1) and returns the wrong
    # space, which is then written back with no *2-1 restore.
    video = torch.full((1, 3, 2, 3, 3), 0.25, dtype=torch.bfloat16)
    video[0, 0, 0, 0, 0] = 1.25  # single overshoot > 1 -> amax > 1
    media = DiffusionMediaOutput(
        video=VideoMediaOutput(
            tensor=video,
            spec=VideoTensorSpec(
                layout=VideoTensorLayout.BCTHW,
                encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
                value_range=VideoValueRange.ZERO_TO_ONE,
            ),
            constraints=VideoTransportConstraints(
                pending_float_consumers=frozenset({FloatVideoConsumer.FRAME_INTERPOLATION})
            ),
        ),
        prepared_for_transport=True,
    )

    seen: list[torch.Tensor] = []

    def _square_interpolator(tensor: torch.Tensor, **_kwargs: object) -> tuple[torch.Tensor, float]:
        seen.append(tensor)
        return tensor.float().square(), 2.0

    monkeypatch.setattr(media_postprocess, "interpolate_video_tensor", _square_interpolator)

    output = finalize_diffusion_media(
        media,
        sampling_params=OmniDiffusionSamplingParams(output_type="np", enable_frame_interpolation=True),
    )

    # The interpolator is fed the clamped bf16 unit-range tensor (max now 1.0),
    # not the raw 1.25 overshoot that would be misread as [-1,1].
    assert len(seen) == 1
    assert seen[0].dtype is torch.bfloat16
    assert float(seen[0].min()) == pytest.approx(0.25, abs=1e-6)
    assert float(seen[0].max()) == pytest.approx(1.0, abs=1e-6)
    # Squaring the clamped values gives 0.0625 at the 0.25 samples and 1.0 at the
    # overshoot; the overall shape is unchanged (no *2-1 for ZERO_TO_ONE).
    frames = output["payload"]["video"]
    assert np.allclose(np.asarray(frames).reshape(-1)[1:], 0.0625, atol=1e-6)
    assert np.isclose(float(np.asarray(frames).reshape(-1)[0]), 1.0, atol=1e-6)


def test_packing_failure_unlinks_created_segments_and_leaves_output_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Packing a late field can fail after an SHM segment is already created;
    # the segment must be unlinked and ``output`` left unmutated so a partial
    # object is never enqueued.
    prepared = prepare_diffusion_media_for_transport(
        DiffusionMediaOutput(
            video=_video(
                torch.linspace(-1.0, 1.0, 2 * 3 * 2 * 4 * 5).reshape(2, 3, 2, 4, 5),
                encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
                value_range=VideoValueRange.NEGATIVE_ONE_TO_ONE,
            )
        ),
        od_config=_od_config(enabled=False),
        sampling_params=OmniDiffusionSamplingParams(output_type="np"),
    )
    output = DiffusionOutput(media=prepared, trajectory_latents=torch.randn(30))
    original_media = prepared
    original_latents = output.trajectory_latents

    real_array_to_shm = diffusion_ipc._array_to_shm
    created_handles: list[dict[str, object]] = []

    def _fail_on_second(array: np.ndarray) -> dict[str, object]:
        if created_handles:
            raise RuntimeError("injected packing failure")
        handle = real_array_to_shm(array)
        created_handles.append(handle)
        return handle

    monkeypatch.setattr(diffusion_ipc, "_SHM_TENSOR_THRESHOLD", 1)
    monkeypatch.setattr(diffusion_ipc, "_array_to_shm", _fail_on_second)

    with pytest.raises(RuntimeError, match="injected packing failure"):
        pack_diffusion_output_shm(output)

    assert created_handles, "no SHM segment was created before the injected failure"
    # Output was not mutated despite the media segment being created first.
    assert output.media is original_media
    assert isinstance(output.media, DiffusionMediaOutput)
    assert output.trajectory_latents is original_latents

    from multiprocessing import shared_memory

    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=str(created_handles[0]["name"]))


def test_second_batch_entry_failure_leaves_the_whole_batch_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multiprocessing import shared_memory

    from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput

    first_media = _prepared_media(
        _video(
            torch.zeros(1, 3, 2, 4, 5),
            encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
            value_range=VideoValueRange.NEGATIVE_ONE_TO_ONE,
        )
    )
    second_media = _prepared_media(
        _video(
            torch.ones(1, 3, 2, 4, 5),
            encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
            value_range=VideoValueRange.NEGATIVE_ONE_TO_ONE,
        )
    )
    first = DiffusionOutput(media=first_media)
    second = DiffusionOutput(media=second_media)
    batch = BatchRunnerOutput(
        runner_outputs=[
            RunnerOutput(request_id="first", result=first),
            RunnerOutput(request_id="second", result=second),
        ]
    )

    real_array_to_shm = diffusion_ipc._array_to_shm
    created_handles: list[dict[str, object]] = []

    def _fail_on_second(array: np.ndarray) -> dict[str, object]:
        if created_handles:
            raise RuntimeError("second batch entry failed")
        handle = real_array_to_shm(array)
        created_handles.append(handle)
        return handle

    monkeypatch.setattr(diffusion_ipc, "_SHM_TENSOR_THRESHOLD", 1)
    monkeypatch.setattr(diffusion_ipc, "_array_to_shm", _fail_on_second)

    with pytest.raises(RuntimeError, match="second batch entry failed"):
        pack_diffusion_output_shm(batch)

    # No per-entry assignment is committed until every entry has packed.
    assert first.media is first_media
    assert second.media is second_media
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=str(created_handles[0]["name"]))
