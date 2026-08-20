# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for OmniPlatform.record_device_event implementations."""

import pytest

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestOmniPlatformRecordDeviceEventInterface:
    """Test that the base OmniPlatform returns None (safe no-op fallback)."""

    def test_base_class_returns_none(self):
        from vllm_omni.platforms.interface import OmniPlatform

        assert OmniPlatform.record_device_event() is None


class TestCudaOmniPlatformRecordDeviceEvent:
    """Test CudaOmniPlatform.record_device_event with mocked torch.Event."""

    def test_records_event_successfully(self, mocker):
        from vllm_omni.platforms.cuda.platform import CudaOmniPlatform

        mock_event = mocker.MagicMock()
        mocker.patch("torch.Event", return_value=mock_event)

        result = CudaOmniPlatform.record_device_event()
        assert result is mock_event
        mock_event.record.assert_called_once()

    def test_returns_none_on_failure(self, mocker):
        from vllm_omni.platforms.cuda.platform import CudaOmniPlatform

        mocker.patch("torch.Event", side_effect=RuntimeError("no CUDA"))

        result = CudaOmniPlatform.record_device_event()
        assert result is None


class TestXPUOmniPlatformRecordDeviceEvent:
    """XPU must record a device-agnostic ``torch.Event``, not ``torch.xpu.Event``.

    The async diffusion output thread gates its D2H copy with
    ``torch.Stream.wait_event`` on a generic ``torch.Stream``, and that binding
    silently drops the dependency for a ``torch.xpu.Event`` — the copy then races
    the still-running forward and the host reads a partially written tensor
    (garbage rows at the bottom of the image).
    """

    def test_records_device_agnostic_torch_event(self, mocker):
        # vLLM's XPUPlatform imports vllm_xpu_kernels at module scope, and that
        # package is absent on non-XPU images.
        pytest.importorskip("vllm_xpu_kernels")

        from vllm_omni.platforms.xpu.platform import XPUOmniPlatform

        mock_event = mocker.MagicMock()
        mocker.patch("torch.Event", return_value=mock_event)
        xpu_event = mocker.patch("torch.xpu.Event")

        result = XPUOmniPlatform.record_device_event()

        assert result is mock_event
        mock_event.record.assert_called_once()
        xpu_event.assert_not_called()

    @hardware_test(res={"xpu": "B60"})
    def test_recorded_event_orders_side_stream_d2h(self):
        """On-device: the event must make a side-stream D2H wait for compute."""
        import torch

        if not torch.xpu.is_available():
            pytest.skip("XPU not available")

        from vllm_omni.platforms.xpu.platform import XPUOmniPlatform

        device = torch.device("xpu", 0)
        # Normalized so the matmul chain stays finite; the ``* 0.0`` term keeps
        # every element exactly 7.0 while still depending on the whole chain.
        base = torch.randn(4096, 4096, device=device) / 64.0
        chained = base.clone()
        for _ in range(40):
            chained = chained @ base
        tensor = torch.full_like(chained, 7.0) + chained * 0.0

        event = XPUOmniPlatform.record_device_event()
        assert event is not None

        # Mirrors WorkerProc._async_output_loop.
        d2h_stream = torch.Stream(device=device)
        d2h_stream.wait_event(event)
        previous = torch.accelerator.current_stream()
        torch.accelerator.set_stream(d2h_stream)
        try:
            host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
            host.copy_(tensor, non_blocking=True)
        finally:
            torch.accelerator.set_stream(previous)
        d2h_stream.synchronize()

        copied_early = int((host != 7.0).sum())
        assert copied_early == 0, f"{copied_early} elements copied before compute finished"


class TestNPUOmniPlatformRecordDeviceEvent:
    """Test NPUOmniPlatform.record_device_event with mocked torch.npu."""

    def test_returns_none_on_failure(self, mocker):
        """When torch.npu.current_stream().synchronize() fails, return None."""
        try:
            from vllm_omni.platforms.npu.platform import NPUOmniPlatform
        except ModuleNotFoundError:
            pytest.skip("vllm_ascend not available")

        mock_torch = mocker.MagicMock()
        mock_torch.npu.current_stream.return_value.synchronize.side_effect = RuntimeError("no NPU")
        mocker.patch("vllm_omni.platforms.npu.platform.torch", mock_torch)

        result = NPUOmniPlatform.record_device_event()
        assert result is None

    def test_synchronizes_stream_then_records_event(self, mocker):
        """NPU should sync stream first, then create and record an Event."""
        try:
            from vllm_omni.platforms.npu.platform import NPUOmniPlatform
        except ModuleNotFoundError:
            pytest.skip("vllm_ascend not available")

        mock_event = mocker.MagicMock()
        mock_stream = mocker.MagicMock()
        mock_torch = mocker.MagicMock()
        mock_torch.npu.current_stream.return_value = mock_stream
        mock_torch.npu.Event.return_value = mock_event
        mocker.patch("vllm_omni.platforms.npu.platform.torch", mock_torch)

        result = NPUOmniPlatform.record_device_event()

        # Stream should be synced first (HCCL ordering)
        mock_stream.synchronize.assert_called_once()
        # Then event should be created and recorded
        mock_torch.npu.Event.assert_called_once()
        mock_event.record.assert_called_once()
        assert result is mock_event
