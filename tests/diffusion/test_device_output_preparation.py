# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch

import vllm_omni.diffusion.ipc as diffusion_ipc
import vllm_omni.diffusion.postprocess.device_reduction as device_reduction
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig, VideoOutputTransportConfig
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
    ensure_request_owned_tensor,
    slice_diffusion_media_output,
)
from vllm_omni.diffusion.postprocess.device_reduction import prepare_diffusion_media_for_transport
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _config(*, enabled: bool) -> OmniDiffusionConfig:
    return OmniDiffusionConfig(
        model=None,
        video_output_transport=VideoOutputTransportConfig(enable_device_postprocess=enabled),
    )


def _sampling(*, output_type: str | None = "np", interpolate: bool = False) -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        output_type=output_type,
        enable_frame_interpolation=interpolate,
    )


def _media(
    tensor: torch.Tensor,
    *,
    value_range: VideoValueRange = VideoValueRange.NEGATIVE_ONE_TO_ONE,
    consumers: frozenset[FloatVideoConsumer] = frozenset(),
) -> DiffusionMediaOutput:
    return DiffusionMediaOutput(
        video=VideoMediaOutput(
            tensor=tensor,
            spec=VideoTensorSpec(
                layout=VideoTensorLayout.BCTHW,
                encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
                value_range=value_range,
            ),
            constraints=VideoTransportConstraints(pending_float_consumers=consumers),
        )
    )


def test_runtime_prepares_request_local_video_as_uint8() -> None:
    video = torch.linspace(-1.0, 1.0, 2 * 3 * 2 * 4 * 5).reshape(2, 3, 2, 4, 5)

    prepared = prepare_diffusion_media_for_transport(
        _media(video),
        od_config=_config(enabled=True),
        sampling_params=_sampling(),
    )

    assert prepared.prepared_for_transport is True
    assert prepared.video.tensor.dtype == torch.uint8
    assert prepared.video.tensor.shape == (2, 2, 4, 5, 3)
    assert prepared.video.spec == VideoTensorSpec(
        layout=VideoTensorLayout.BTHWC,
        encoding=VideoTensorEncoding.UINT8_FRAMES,
        value_range=VideoValueRange.ZERO_TO_255,
    )
    assert prepared.video.constraints.pending_float_consumers == frozenset()


def test_disabled_runtime_preserves_float_but_marks_media_prepared() -> None:
    video = torch.randn(1, 3, 2, 4, 5)

    prepared = prepare_diffusion_media_for_transport(
        _media(video),
        od_config=_config(enabled=False),
        sampling_params=_sampling(),
    )

    assert prepared.prepared_for_transport is True
    assert prepared.video.tensor is video
    assert prepared.video.spec.encoding is VideoTensorEncoding.NORMALIZED_FLOAT


def test_device_oom_falls_back_to_request_owned_float_media(monkeypatch: pytest.MonkeyPatch) -> None:
    source = torch.randn(2, 3, 2, 4, 5)
    sliced = slice_diffusion_media_output(_media(source), 0, 1)

    def raise_oom(video: torch.Tensor, *, do_denormalize: bool) -> torch.Tensor:
        raise torch.OutOfMemoryError("injected")

    monkeypatch.setattr(device_reduction, "reduce_video_to_uint8_frames", raise_oom)

    prepared = prepare_diffusion_media_for_transport(
        sliced,
        od_config=_config(enabled=True),
        sampling_params=_sampling(),
    )

    assert prepared.prepared_for_transport is True
    assert prepared.video.spec.encoding is VideoTensorEncoding.NORMALIZED_FLOAT
    assert prepared.video.tensor.dtype == source.dtype
    assert prepared.video.tensor.shape == (1, 3, 2, 4, 5)
    assert prepared.video.tensor.is_contiguous()
    assert prepared.video.tensor._base is None


def test_device_non_oom_error_is_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_runtime_error(video: torch.Tensor, *, do_denormalize: bool) -> torch.Tensor:
        raise RuntimeError("injected")

    monkeypatch.setattr(device_reduction, "reduce_video_to_uint8_frames", raise_runtime_error)

    with pytest.raises(RuntimeError, match="injected"):
        prepare_diffusion_media_for_transport(
            _media(torch.randn(1, 3, 2, 4, 5)),
            od_config=_config(enabled=True),
            sampling_params=_sampling(),
        )


def test_declared_and_request_float_consumers_are_unioned() -> None:
    video = torch.randn(1, 3, 2, 4, 5)
    media = _media(
        video,
        consumers=frozenset({FloatVideoConsumer.VIDEO_GUARDRAILS}),
    )

    prepared = prepare_diffusion_media_for_transport(
        media,
        od_config=_config(enabled=True),
        sampling_params=_sampling(interpolate=True),
    )

    assert prepared.video.tensor is video
    assert prepared.video.constraints.pending_float_consumers == frozenset(
        {
            FloatVideoConsumer.FRAME_INTERPOLATION,
            FloatVideoConsumer.VIDEO_GUARDRAILS,
        }
    )


def test_non_numpy_presentation_preserves_float() -> None:
    prepared = prepare_diffusion_media_for_transport(
        _media(torch.randn(1, 3, 2, 4, 5)),
        od_config=_config(enabled=True),
        sampling_params=_sampling(output_type="pil"),
    )

    assert prepared.prepared_for_transport is True
    assert prepared.video.spec.encoding is VideoTensorEncoding.NORMALIZED_FLOAT


def test_zero_to_one_video_is_not_denormalized() -> None:
    video = torch.tensor([[[[[0.0, 0.5, 1.0]]]]]).expand(1, 3, 1, 1, 3)

    prepared = prepare_diffusion_media_for_transport(
        _media(video, value_range=VideoValueRange.ZERO_TO_ONE),
        od_config=_config(enabled=True),
        sampling_params=_sampling(),
    )

    expected = np.rint(video.permute(0, 2, 3, 4, 1).numpy() * 255.0).astype(np.uint8)
    np.testing.assert_array_equal(prepared.video.tensor.numpy(), expected)


def test_invalid_contract_raises_even_when_feature_is_off() -> None:
    media = _media(torch.randn(3, 2, 4, 5))

    with pytest.raises(ValueError, match="rank 5"):
        prepare_diffusion_media_for_transport(
            media,
            od_config=_config(enabled=False),
            sampling_params=_sampling(),
        )


@pytest.mark.parametrize(
    ("tensor", "spec", "message"),
    [
        (
            torch.randn(1, 2, 4, 5, 3),
            VideoTensorSpec(
                layout=VideoTensorLayout.BTHWC,
                encoding=VideoTensorEncoding.UINT8_FRAMES,
                value_range=VideoValueRange.ZERO_TO_255,
            ),
            "torch.uint8",
        ),
        (
            torch.zeros(1, 2, 4, 5, 3, dtype=torch.uint8),
            VideoTensorSpec(
                layout=VideoTensorLayout.BTHWC,
                encoding=VideoTensorEncoding.UINT8_FRAMES,
                value_range=VideoValueRange.ZERO_TO_ONE,
            ),
            "incompatible value range",
        ),
        (
            torch.randn(1, 4, 2, 4, 5),
            VideoTensorSpec(
                layout=VideoTensorLayout.BCTHW,
                encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
                value_range=VideoValueRange.NEGATIVE_ONE_TO_ONE,
            ),
            "3 channels",
        ),
    ],
)
def test_invalid_spec_tensor_combinations_are_contract_errors(
    tensor: torch.Tensor,
    spec: VideoTensorSpec,
    message: str,
) -> None:
    media = DiffusionMediaOutput(video=VideoMediaOutput(tensor=tensor, spec=spec))

    with pytest.raises(ValueError, match=message):
        media.validate()


def test_pipeline_cannot_emit_unprepared_uint8_media() -> None:
    media = DiffusionMediaOutput(
        video=VideoMediaOutput(
            tensor=torch.zeros(1, 2, 4, 5, 3, dtype=torch.uint8),
            spec=VideoTensorSpec(
                layout=VideoTensorLayout.BTHWC,
                encoding=VideoTensorEncoding.UINT8_FRAMES,
                value_range=VideoValueRange.ZERO_TO_255,
            ),
        )
    )

    with pytest.raises(ValueError, match="must emit unprepared media as NORMALIZED_FLOAT"):
        prepare_diffusion_media_for_transport(
            media,
            od_config=_config(enabled=True),
            sampling_params=_sampling(),
        )


def test_descriptors_are_immutable() -> None:
    media = _media(torch.randn(1, 3, 2, 4, 5))

    with pytest.raises(FrozenInstanceError):
        media.prepared_for_transport = True  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        media.video.spec.layout = VideoTensorLayout.BTHWC  # type: ignore[misc]


def test_float_fallback_detaches_a_request_view_before_transport() -> None:
    source = torch.randn(4, 3, 2, 4, 5)
    sliced = slice_diffusion_media_output(_media(source), 1, 3)

    prepared = prepare_diffusion_media_for_transport(
        sliced,
        od_config=_config(enabled=False),
        sampling_params=_sampling(),
    )

    tensor = prepared.video.tensor
    assert tensor.shape == (2, 3, 2, 4, 5)
    assert tensor.is_contiguous()
    assert tensor._base is None
    assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
    assert tensor.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()


def test_full_batch_slice_does_not_duplicate_owned_storage() -> None:
    source = torch.randn(2, 3, 2, 4, 5)

    sliced = slice_diffusion_media_output(_media(source), 0, 2)

    assert sliced.video.tensor.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()


def test_diffusion_output_to_cpu_moves_prepared_typed_media() -> None:
    prepared = prepare_diffusion_media_for_transport(
        _media(torch.randn(1, 3, 2, 4, 5)),
        od_config=_config(enabled=False),
        sampling_params=_sampling(),
    )
    output = DiffusionOutput(media=prepared, to_cpu=True)

    assert output.media is not None
    assert output.media.video.tensor.device.type == "cpu"


def test_diffusion_output_to_cpu_rejects_unprepared_media() -> None:
    with pytest.raises(ValueError, match="prepared before to_cpu"):
        DiffusionOutput(media=_media(torch.randn(1, 3, 2, 4, 5)), to_cpu=True)


def test_prepared_media_round_trips_through_typed_ipc(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = prepare_diffusion_media_for_transport(
        _media(torch.randn(1, 3, 2, 256, 256)),
        od_config=_config(enabled=True),
        sampling_params=_sampling(),
    )
    output = DiffusionOutput(media=expected)

    monkeypatch.setattr(diffusion_ipc, "_SHM_TENSOR_THRESHOLD", 1)
    pack_diffusion_output_shm(output)

    assert isinstance(output.media, dict)
    assert output.media["__type__"] == "diffusion_media_v1"
    assert output.media["video"]["tensor"]["__tensor_shm__"] is True

    unpack_diffusion_output_shm(output)

    assert isinstance(output.media, DiffusionMediaOutput)
    assert output.media.prepared_for_transport is True
    assert output.media.video.spec == expected.video.spec
    torch.testing.assert_close(output.media.video.tensor, expected.video.tensor)


def test_float_media_constraints_round_trip_through_typed_ipc(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = prepare_diffusion_media_for_transport(
        _media(torch.randn(1, 3, 2, 256, 256, dtype=torch.bfloat16)),
        od_config=_config(enabled=True),
        sampling_params=_sampling(interpolate=True),
    )
    output = DiffusionOutput(media=expected)

    monkeypatch.setattr(diffusion_ipc, "_SHM_TENSOR_THRESHOLD", 1)
    pack_diffusion_output_shm(output)
    unpack_diffusion_output_shm(output)

    assert isinstance(output.media, DiffusionMediaOutput)
    assert output.media.video.tensor.dtype is torch.bfloat16
    assert output.media.video.constraints.pending_float_consumers == frozenset({FloatVideoConsumer.FRAME_INTERPOLATION})
    torch.testing.assert_close(output.media.video.tensor, expected.video.tensor)


def test_ipc_rejects_media_that_skipped_runner_preparation() -> None:
    output = DiffusionOutput(media=_media(torch.randn(1, 3, 2, 4, 5)))

    with pytest.raises(ValueError, match="must be prepared"):
        pack_diffusion_output_shm(output)


def test_ipc_rejects_unknown_media_schema_version() -> None:
    prepared = prepare_diffusion_media_for_transport(
        _media(torch.randn(1, 3, 2, 4, 5)),
        od_config=_config(enabled=False),
        sampling_params=_sampling(),
    )
    output = DiffusionOutput(media=prepared)
    pack_diffusion_output_shm(output)
    assert isinstance(output.media, dict)
    output.media["schema_version"] = 2

    with pytest.raises(ValueError, match="schema version"):
        unpack_diffusion_output_shm(output)


def test_prepare_rejects_media_a_pipeline_already_prepared(monkeypatch: pytest.MonkeyPatch) -> None:
    # A pipeline must emit unprepared media so the runner applies request policy
    # (e.g. frame interpolation). Accepting a forged prepared uint8 payload would
    # skip the interpolation constraint; interpolator never runs and gets dropped.
    forged = DiffusionMediaOutput(
        video=VideoMediaOutput(
            tensor=torch.zeros(1, 2, 4, 5, 3, dtype=torch.uint8),
            spec=VideoTensorSpec(
                layout=VideoTensorLayout.BTHWC,
                encoding=VideoTensorEncoding.UINT8_FRAMES,
                value_range=VideoValueRange.ZERO_TO_255,
            ),
        ),
        prepared_for_transport=True,
    )

    with pytest.raises(ValueError, match="already prepared"):
        prepare_diffusion_media_for_transport(
            forged,
            od_config=_config(enabled=True),
            sampling_params=_sampling(interpolate=True),
        )


def test_prepare_rejects_prepared_media_despite_pending_interpolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The forged payload is the exact case the reviewer described: device
    # postprocessing enabled plus frame_interpolation, but the media already being
    # prepared. It must be rejected, not silently returned with interpolation dropped.
    forged = DiffusionMediaOutput(
        video=VideoMediaOutput(
            tensor=torch.zeros(1, 3, 2, 2, 3, dtype=torch.float32),
            spec=VideoTensorSpec(
                layout=VideoTensorLayout.BCTHW,
                encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
                value_range=VideoValueRange.NEGATIVE_ONE_TO_ONE,
            ),
        ),
        prepared_for_transport=True,
    )

    with pytest.raises(ValueError, match="already prepared"):
        prepare_diffusion_media_for_transport(
            forged,
            od_config=_config(enabled=True),
            sampling_params=_sampling(interpolate=True),
        )


def test_request_owned_tensor_rejects_a_detached_storage_view() -> None:
    # ``tensor._base is None`` is not proof of ownership: a detached slice keeps a
    # nonzero storage offset and the whole source storage. Serializing it would
    # leak every other request's pixels, so it must be cloned before transport.
    source = torch.randn(4, 3, 2, 4, 5)
    detached_slice = source[1:2].detach()
    assert detached_slice.is_contiguous()
    assert detached_slice._base is None
    assert detached_slice.storage_offset() != 0

    owned = ensure_request_owned_tensor(detached_slice)

    assert owned.shape == detached_slice.shape
    assert owned.storage_offset() == 0
    assert owned.untyped_storage().nbytes() == owned.numel() * owned.element_size()
    assert owned.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    torch.testing.assert_close(owned, detached_slice)


def test_prepared_media_rejects_non_compact_request_storage() -> None:
    # A zero-offset view of a bigger buffer is contiguous but not compact, so
    # prepared transport must refuse it just as it refuses a storage view.
    source = torch.randn(2, 3, 2, 4, 5)
    zero_offset_view = source[0:1]
    media = DiffusionMediaOutput(
        video=VideoMediaOutput(
            tensor=zero_offset_view,
            spec=VideoTensorSpec(
                layout=VideoTensorLayout.BCTHW,
                encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
                value_range=VideoValueRange.NEGATIVE_ONE_TO_ONE,
            ),
        ),
        prepared_for_transport=True,
    )

    with pytest.raises(ValueError, match="request-local storage"):
        media.validate()
