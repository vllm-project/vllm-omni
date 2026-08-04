# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for IPC async D2H stream packing."""

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.ipc import (
    _pack_diffusion_fields,
    _pack_tensor_if_large,
    _pack_value_if_large,
    pack_diffusion_output_shm,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestPackTensorIfLargeWithD2hStream:
    """Test _pack_tensor_if_large with mock d2h_stream parameter."""

    def test_small_tensor_not_packed_with_stream(self, mocker):
        """Small tensors pass through even when d2h_stream is provided."""
        small_tensor = torch.zeros(10)  # way below threshold
        mock_stream = mocker.MagicMock()
        result = _pack_tensor_if_large(small_tensor, d2h_stream=mock_stream)
        # Small tensor should be returned as-is (not a SHM handle dict)
        assert isinstance(result, torch.Tensor)

    def test_large_tensor_packed_without_stream(self, mocker):
        """Large tensor is packed via SHM when no d2h_stream."""
        large_tensor = torch.randn(500_000)  # > 1MB for float32
        result = _pack_tensor_if_large(large_tensor, d2h_stream=None)
        assert isinstance(result, dict)
        assert result.get("__tensor_shm__") is True

    def test_pack_value_if_large_passes_stream_to_tensor_pack(self, mocker):
        """_pack_value_if_large propagates d2h_stream to _pack_tensor_if_large."""
        mock_pack = mocker.patch(
            "vllm_omni.diffusion.ipc._pack_tensor_if_large",
            return_value="packed",
        )
        tensor = torch.zeros(1)
        d2h = mocker.MagicMock()
        _pack_value_if_large(tensor, d2h_stream=d2h)
        mock_pack.assert_called_once_with(tensor, d2h_stream=d2h)

    def test_pack_value_if_large_passes_stream_through_dict(self, mocker):
        """d2h_stream propagates through dict container packing."""
        mock_pack = mocker.patch(
            "vllm_omni.diffusion.ipc._pack_tensor_if_large",
            return_value="packed",
        )
        d2h = mocker.MagicMock()
        data = {"image": torch.zeros(1), "video": torch.zeros(1)}
        _pack_value_if_large(data, d2h_stream=d2h)
        assert mock_pack.call_count == 2
        for call in mock_pack.call_args_list:
            assert call.kwargs.get("d2h_stream") is d2h or call.args[1] is d2h

    def test_pack_value_if_large_passes_stream_through_list(self, mocker):
        """d2h_stream propagates through list container packing."""
        mock_pack = mocker.patch(
            "vllm_omni.diffusion.ipc._pack_tensor_if_large",
            return_value="packed",
        )
        d2h = mocker.MagicMock()
        data = [torch.zeros(1), torch.zeros(1)]
        _pack_value_if_large(data, d2h_stream=d2h)
        assert mock_pack.call_count == 2

    def test_pack_value_if_large_passes_stream_through_tuple(self, mocker):
        """d2h_stream propagates through tuple container packing."""
        mock_pack = mocker.patch(
            "vllm_omni.diffusion.ipc._pack_tensor_if_large",
            return_value="packed",
        )
        d2h = mocker.MagicMock()
        data = (torch.zeros(1), torch.zeros(1))
        result = _pack_value_if_large(data, d2h_stream=d2h)
        assert isinstance(result, tuple)
        assert mock_pack.call_count == 2


class TestPackDiffusionFieldsWithD2hStream:
    """Test _pack_diffusion_fields propagates d2h_stream."""

    def test_output_field_packed_with_stream(self, mocker):
        mock_pack_value = mocker.patch(
            "vllm_omni.diffusion.ipc._pack_value_if_large",
            return_value="packed",
        )
        d2h = mocker.MagicMock()
        out = DiffusionOutput(output=torch.zeros(1))
        _pack_diffusion_fields(out, d2h_stream=d2h)
        mock_pack_value.assert_called_once_with(torch.zeros(1), d2h_stream=d2h)

    def test_trajectory_fields_packed_with_stream(self, mocker):
        mock_pack_tensor = mocker.patch(
            "vllm_omni.diffusion.ipc._pack_tensor_if_large",
            return_value="packed",
        )
        d2h = mocker.MagicMock()
        out = DiffusionOutput(
            trajectory_latents=torch.zeros(1),
            trajectory_log_probs=torch.zeros(1),
        )
        _pack_diffusion_fields(out, d2h_stream=d2h)
        # Should pack trajectory_latents and trajectory_log_probs
        assert mock_pack_tensor.call_count == 2


class TestPackDiffusionOutputShmWithD2hStream:
    """Test pack_diffusion_output_shm top-level dispatch with d2h_stream."""

    def test_diffusion_output_path_with_stream(self, mocker):
        mock_pack_fields = mocker.patch(
            "vllm_omni.diffusion.ipc._pack_diffusion_fields",
        )
        d2h = mocker.MagicMock()
        out = DiffusionOutput()
        pack_diffusion_output_shm(out, d2h_stream=d2h)
        mock_pack_fields.assert_called_once_with(out, d2h_stream=d2h)

    def test_rpc_envelope_path_with_stream(self, mocker):
        mock_pack_fields = mocker.patch(
            "vllm_omni.diffusion.ipc._pack_diffusion_fields",
        )
        d2h = mocker.MagicMock()
        result = DiffusionOutput()
        envelope = {"type": "diffusion_rpc_result", "result": result}
        pack_diffusion_output_shm(envelope, d2h_stream=d2h)
        # envelope["result"] is replaced in-place by pack_diffusion_output_shm;
        # assert with the original reference, not the replaced one.
        mock_pack_fields.assert_called_once_with(
            result,
            d2h_stream=d2h,
        )

    def test_runner_output_path_with_stream(self, mocker):
        mock_pack_fields = mocker.patch(
            "vllm_omni.diffusion.ipc._pack_diffusion_fields",
        )
        d2h = mocker.MagicMock()

        from vllm_omni.diffusion.worker.utils import RunnerOutput

        result = DiffusionOutput()
        runner_out = RunnerOutput(
            request_id="req1",
            result=result,
        )
        pack_diffusion_output_shm(runner_out, d2h_stream=d2h)
        # runner_out.result is replaced in-place by pack_diffusion_output_shm;
        # assert with the original reference, not the replaced one.
        mock_pack_fields.assert_called_once_with(
            result,
            d2h_stream=d2h,
        )
