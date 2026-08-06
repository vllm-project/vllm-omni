# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for OmniPlatform.record_device_event implementations."""

import pytest

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
