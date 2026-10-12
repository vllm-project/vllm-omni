# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""WAN reference coverage for the typed pre-D2H media contract."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig, VideoOutputTransportConfig
from vllm_omni.diffusion.ipc import pack_diffusion_output_shm, unpack_diffusion_output_shm
from vllm_omni.diffusion.media import (
    DiffusionMediaOutput,
    VideoMediaOutput,
    VideoTensorEncoding,
    VideoTensorLayout,
    VideoTensorSpec,
    VideoValueRange,
)
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import get_wan22_post_process_func
from vllm_omni.diffusion.postprocess.device_reduction import prepare_diffusion_media_for_transport
from vllm_omni.diffusion.postprocess.media import finalize_diffusion_media
from vllm_omni.entrypoints.openai.video_api_utils import _coerce_video_to_uint8_frames
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
]

_GPU = torch.accelerator.is_available() if hasattr(torch, "accelerator") else torch.cuda.is_available()
requires_gpu = pytest.mark.skipif(not _GPU, reason="device reduction is a GPU path")


def _decoded_wan_video(tensor: torch.Tensor) -> DiffusionMediaOutput:
    return DiffusionMediaOutput(
        video=VideoMediaOutput(
            tensor=tensor,
            spec=VideoTensorSpec(
                layout=VideoTensorLayout.BCTHW,
                encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
                value_range=VideoValueRange.NEGATIVE_ONE_TO_ONE,
            ),
        )
    )


def _config(*, enabled: bool) -> OmniDiffusionConfig:
    return OmniDiffusionConfig(
        model=None,
        video_output_transport=VideoOutputTransportConfig(enable_device_postprocess=enabled),
    )


def _encoder_frames(video_payload: np.ndarray) -> list[np.ndarray]:
    return [_coerce_video_to_uint8_frames(video_payload[i]) for i in range(video_payload.shape[0])]


@requires_gpu
def test_wan_media_reduction_matches_float32_reference_and_bounds_native_path() -> None:
    torch.manual_seed(0)
    vae_out = torch.rand(2, 3, 8, 64, 96, device="cuda:0", dtype=torch.bfloat16) * 2.4 - 1.2
    sampling = OmniDiffusionSamplingParams(output_type=None, enable_frame_interpolation=False)

    legacy_postprocess = get_wan22_post_process_func(_config(enabled=False))
    widened_out = legacy_postprocess(vae_out.float(), output_type="np", sampling_params=sampling)
    widened_frames = _encoder_frames(widened_out["payload"]["video"])
    native_out = legacy_postprocess(vae_out, output_type="np", sampling_params=sampling)
    native_frames = _encoder_frames(native_out["payload"]["video"])

    prepared = prepare_diffusion_media_for_transport(
        _decoded_wan_video(vae_out),
        od_config=_config(enabled=True),
        sampling_params=sampling,
    )
    reduced_out = finalize_diffusion_media(prepared, sampling_params=sampling)
    assert prepared.video.spec.encoding is VideoTensorEncoding.UINT8_FRAMES
    assert reduced_out["payload"]["video"].dtype == np.uint8
    reduced_frames = _encoder_frames(reduced_out["payload"]["video"])

    assert len(widened_frames) == len(reduced_frames) == 2
    for expected, native, produced in zip(widened_frames, native_frames, reduced_frames, strict=True):
        assert produced.shape == expected.shape == (8, 64, 96, 3)
        np.testing.assert_array_equal(produced, expected)
        deviation = np.abs(produced.astype(np.int16) - native.astype(np.int16))
        assert deviation.max() <= 1


@requires_gpu
def test_wan_uint8_media_finalization_does_not_rescale() -> None:
    frames = torch.randint(0, 256, (1, 4, 16, 16, 3), dtype=torch.uint8, device="cuda:0")
    media = DiffusionMediaOutput(
        video=VideoMediaOutput(
            tensor=frames,
            spec=VideoTensorSpec(
                layout=VideoTensorLayout.BTHWC,
                encoding=VideoTensorEncoding.UINT8_FRAMES,
                value_range=VideoValueRange.ZERO_TO_255,
            ),
        ),
        prepared_for_transport=True,
    )

    out = finalize_diffusion_media(media, sampling_params=OmniDiffusionSamplingParams(output_type="np"))

    np.testing.assert_array_equal(out["payload"]["video"], frames.cpu().numpy())


@requires_gpu
def test_diffusion_output_to_cpu_moves_wan_media_off_device() -> None:
    prepared = prepare_diffusion_media_for_transport(
        _decoded_wan_video(torch.randn(1, 3, 2, 4, 5, device="cuda:0")),
        od_config=_config(enabled=False),
        sampling_params=OmniDiffusionSamplingParams(output_type="np"),
    )
    output = DiffusionOutput(media=prepared, to_cpu=True)

    assert output.media is not None
    assert output.media.video.tensor.device.type == "cpu"


@requires_gpu
def test_small_typed_media_is_on_cpu_before_ipc() -> None:
    sampling = OmniDiffusionSamplingParams(output_type="np")
    prepared = prepare_diffusion_media_for_transport(
        _decoded_wan_video(torch.randn(1, 3, 2, 4, 5, device="cuda:0")),
        od_config=_config(enabled=True),
        sampling_params=sampling,
    )
    output = DiffusionOutput(media=prepared)

    pack_diffusion_output_shm(output)

    assert isinstance(output.media, dict)
    packed_tensor = output.media["video"]["tensor"]
    assert isinstance(packed_tensor, torch.Tensor)
    assert packed_tensor.device.type == "cpu"
    unpack_diffusion_output_shm(output)


@requires_gpu
def test_wan_float_media_finalization_matches_the_legacy_path() -> None:
    video = torch.rand(1, 3, 4, 16, 16, device="cuda:0", dtype=torch.bfloat16) * 2 - 1
    sampling = OmniDiffusionSamplingParams(output_type="np")
    prepared = prepare_diffusion_media_for_transport(
        _decoded_wan_video(video),
        od_config=_config(enabled=False),
        sampling_params=sampling,
    )

    produced = finalize_diffusion_media(prepared, sampling_params=sampling)
    expected = get_wan22_post_process_func(_config(enabled=False))(
        video,
        output_type="np",
        sampling_params=sampling,
    )

    assert prepared.video.spec.encoding is VideoTensorEncoding.NORMALIZED_FLOAT
    np.testing.assert_array_equal(produced["payload"]["video"], expected["payload"]["video"])


@requires_gpu
def test_oom_fallback_media_packs_without_device_widening(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: when the fp32 reduction OOMs, the fallback keeps the bf16 float
    # payload. Async IPC must copy it to the host in bf16 and widen on the CPU;
    # a second device-side fp32 conversion would OOM again and drop the payload.
    from vllm_omni.diffusion.postprocess import device_reduction

    # Large enough (> 1 MB in bf16) that packing routes the tensor through the
    # shared-memory D2H path rather than the small-tensor CPU copy.
    video = torch.rand(1, 3, 16, 128, 128, device="cuda:0", dtype=torch.bfloat16) * 2 - 1

    def _raise_oom(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise torch.OutOfMemoryError("forced fp32 reduction OOM")

    monkeypatch.setattr(device_reduction, "reduce_video_to_uint8_frames", _raise_oom)

    prepared = prepare_diffusion_media_for_transport(
        _decoded_wan_video(video),
        od_config=_config(enabled=True),
        sampling_params=OmniDiffusionSamplingParams(output_type="np"),
    )
    assert prepared.video.spec.encoding is VideoTensorEncoding.NORMALIZED_FLOAT
    assert prepared.video.tensor.dtype == torch.bfloat16

    output = DiffusionOutput(media=prepared)

    pinned_dtypes: list[torch.dtype] = []
    real_empty = torch.empty

    def _spy_empty(*args: object, **kwargs: object) -> torch.Tensor:
        tensor = real_empty(*args, **kwargs)
        if kwargs.get("pin_memory"):
            pinned_dtypes.append(tensor.dtype)
        return tensor

    monkeypatch.setattr(torch, "empty", _spy_empty)

    d2h_stream = torch.Stream(device=torch.device("cuda", 0))
    pack_diffusion_output_shm(output, d2h_stream=d2h_stream)
    d2h_stream.synchronize()

    assert pinned_dtypes == [torch.bfloat16], (
        "async packing must D2H in the original dtype and widen on the CPU, "
        "never widen bf16 to fp32 on the accelerator before the copy"
    )

    unpack_diffusion_output_shm(output)
    assert isinstance(output.media, DiffusionMediaOutput)
    assert output.media.video.tensor.dtype == torch.bfloat16
    torch.testing.assert_close(output.media.video.tensor, video.cpu())


@requires_gpu
def test_oom_fallback_recovers_the_allocator(monkeypatch: pytest.MonkeyPatch) -> None:
    # A real float32 OOM leaves the CUDA caching allocator in a failed state;
    # the fallback must empty the cache and drain the device before marking the
    # float representation transport-ready, so the worker D2H of the same tensor
    # can actually run. Use a genuine allocation failure (per-process cap) rather
    # than injecting the exception, which never poisons the allocator.
    from vllm_omni.platforms import current_omni_platform

    video = torch.empty(1, 3, 16, 128, 128, device="cuda:0", dtype=torch.bfloat16)
    video.uniform_(-1.0, 1.0)
    expected = video.cpu()
    d2h_stream = torch.Stream(device=video.device)
    base_bytes = video.numel() * video.element_size()
    _free, total = current_omni_platform.get_device_memory()

    # Cap the process so the fp32 conversion (2x base bytes) cannot fit, while
    # retaining the original bf16 tensor. This forces a real allocator OOM at
    # the .to(fp32), rather than injecting an exception from a monkeypatch.
    torch.cuda.set_per_process_memory_fraction(min(0.99, (1.2 * base_bytes) / total))
    real_empty_cache = current_omni_platform.empty_cache
    real_synchronize = current_omni_platform.synchronize
    try:
        empty_calls: list[int] = []
        sync_calls: list[bool] = []

        def _empty_cache() -> None:
            real_empty_cache()
            empty_calls.append(torch.accelerator.current_device_index())

        def _synchronize() -> None:
            real_synchronize()
            sync_calls.append(True)

        monkeypatch.setattr(current_omni_platform, "empty_cache", _empty_cache)
        monkeypatch.setattr(current_omni_platform, "synchronize", _synchronize)

        prepared = prepare_diffusion_media_for_transport(
            _decoded_wan_video(video),
            od_config=_config(enabled=True),
            sampling_params=OmniDiffusionSamplingParams(output_type="np"),
        )

        # The fallback ran past a genuine OOM and kept the bf16 float payload.
        assert prepared.video.spec.encoding is VideoTensorEncoding.NORMALIZED_FLOAT
        assert prepared.video.tensor.dtype == torch.bfloat16
        assert empty_calls, "fallback must empty the CUDA allocator cache after a real OOM"
        assert sync_calls, "fallback must drain the device after a real OOM"

        # Exercise the next production step: async D2H/SHM packing of the same
        # tensor must succeed after allocator recovery and preserve its values.
        output = DiffusionOutput(media=prepared)
        pack_diffusion_output_shm(output, d2h_stream=d2h_stream)
        d2h_stream.synchronize()
        unpack_diffusion_output_shm(output)
        assert isinstance(output.media, DiffusionMediaOutput)
        torch.testing.assert_close(output.media.video.tensor, expected)
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)
        real_empty_cache()
        real_synchronize()


@requires_gpu
def test_oom_fallback_releases_late_uint8_traceback_before_clone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A final uint8 allocation OOM must release float intermediates first."""
    from vllm_omni.diffusion.postprocess import device_reduction
    from vllm_omni.platforms import current_omni_platform

    # A split request is a non-compact view, so float fallback must allocate a
    # request-owned clone after reduction fails.
    backing = torch.empty(2, 3, 16, 128, 128, device="cuda:0", dtype=torch.bfloat16)
    backing.uniform_(-1.0, 1.0)
    video = backing[1:2].detach()
    expected = video.cpu()
    _, total = current_omni_platform.get_device_memory()
    real_to = torch.Tensor.to
    real_empty_cache = current_omni_platform.empty_cache
    real_synchronize = current_omni_platform.synchronize
    real_ensure_request_owned = device_reduction.ensure_request_owned_tensor
    uint8_attempts: list[int] = []
    fallback_allocations: list[int] = []

    def _record_fallback_allocation(tensor: torch.Tensor) -> torch.Tensor:
        fallback_allocations.append(torch.accelerator.memory_allocated(tensor.device))
        return real_ensure_request_owned(tensor)

    def _fail_real_uint8_allocation(
        tensor: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        dtype = kwargs.get("dtype")
        if dtype is None and args and isinstance(args[0], torch.dtype):
            dtype = args[0]
        if tensor.is_cuda and dtype is torch.uint8:
            # Earlier float32 work runs without a cap. Immediately before the
            # real final uint8 allocation, cap the allocator below that request
            # but above the post-unwind request-owned bf16 clone requirement.
            real_empty_cache()
            allocated = torch.accelerator.memory_allocated(tensor.device)
            uint8_bytes = tensor.numel()
            limit_bytes = allocated + max(1, uint8_bytes // 2)
            torch.cuda.set_per_process_memory_fraction(min(0.99, limit_bytes / total))
            uint8_attempts.append(allocated)
        return real_to(tensor, *args, **kwargs)

    monkeypatch.setattr(device_reduction, "ensure_request_owned_tensor", _record_fallback_allocation)
    monkeypatch.setattr(torch.Tensor, "to", _fail_real_uint8_allocation)
    try:
        prepared = prepare_diffusion_media_for_transport(
            _decoded_wan_video(video),
            od_config=_config(enabled=True),
            sampling_params=OmniDiffusionSamplingParams(output_type="np"),
        )

        assert uint8_attempts, "the real final .to(uint8) allocation was not attempted"
        assert fallback_allocations, "float fallback did not request compact storage"
        assert fallback_allocations[0] < uint8_attempts[0], (
            "the failed reduction's float intermediates remained live during fallback"
        )
        assert prepared.video.spec.encoding is VideoTensorEncoding.NORMALIZED_FLOAT
        assert prepared.video.tensor.dtype is torch.bfloat16
        assert prepared.video.tensor.storage_offset() == 0
        assert prepared.video.tensor.untyped_storage().data_ptr() != video.untyped_storage().data_ptr()
        torch.testing.assert_close(prepared.video.tensor.cpu(), expected)
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)
        real_empty_cache()
        real_synchronize()
