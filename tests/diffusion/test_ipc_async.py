# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for IPC async D2H stream packing."""

from types import SimpleNamespace
from unittest.mock import ANY

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.ipc import (
    _pack_diffusion_fields,
    _pack_diffusion_media,
    _pack_tensor_if_large,
    _pack_value_if_large,
    pack_diffusion_output_shm,
)
from vllm_omni.platforms.interface import OmniPlatform

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
        mock_pack.assert_called_once_with(tensor, d2h_stream=d2h, created=None)

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
        mock_pack_value.assert_called_once_with(torch.zeros(1), d2h_stream=d2h, created=None)

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


def _fake_device_tensor(mocker, host: torch.Tensor):
    """A sub-threshold tensor that reports a non-CPU device.

    ``.detach().cpu().contiguous()`` returns *host*, so the test can tell
    whether packing moved the tensor to CPU without accelerator hardware.
    """
    fake = mocker.MagicMock(spec=torch.Tensor)
    fake.device = torch.device("meta")
    fake.nelement.return_value = host.nelement()
    fake.element_size.return_value = host.element_size()
    fake.untyped_storage.return_value.nbytes.return_value = host.untyped_storage().nbytes()
    fake.detach.return_value.cpu.return_value.contiguous.return_value = host
    return fake


@pytest.fixture
def set_device_tensor_ipc(monkeypatch):
    """Make ``current_omni_platform.supports_device_tensor_ipc()`` return *flag*.

    ``ipc.py`` imports ``current_omni_platform`` lazily from
    ``vllm_omni.platforms``, whose module ``__getattr__`` returns
    ``_current_omni_platform`` once it is set.
    """

    def _set(flag: bool) -> None:
        platform = SimpleNamespace(supports_device_tensor_ipc=lambda: flag)
        monkeypatch.setattr("vllm_omni.platforms._current_omni_platform", platform)

    return _set


class TestSmallDeviceTensorIpcFlag:
    """``OmniPlatform.supports_device_tensor_ipc()`` gates the CPU move of sub-threshold device tensors."""

    def test_flag_defaults_to_true(self):
        assert OmniPlatform.supports_device_tensor_ipc() is True

    @pytest.mark.parametrize("flag", [True, False])
    def test_small_cpu_tensor_returned_unchanged(self, set_device_tensor_ipc, flag):
        set_device_tensor_ipc(flag)
        small_tensor = torch.zeros(10)
        assert _pack_tensor_if_large(small_tensor) is small_tensor

    def test_small_device_tensor_kept_on_device_when_flag_true(self, mocker, set_device_tensor_ipc):
        set_device_tensor_ipc(True)
        fake = _fake_device_tensor(mocker, torch.arange(10, dtype=torch.float32))
        result = _pack_tensor_if_large(fake)
        assert result is fake
        assert result.device.type == "meta"
        fake.detach.assert_not_called()

    def test_small_device_tensor_moved_to_cpu_when_flag_false(self, mocker, set_device_tensor_ipc):
        set_device_tensor_ipc(False)
        host = torch.arange(10, dtype=torch.float32)
        fake = _fake_device_tensor(mocker, host)
        result = _pack_tensor_if_large(fake)
        assert result is host
        assert result.device.type == "cpu"
        assert result.is_contiguous()

    def test_legacy_output_and_trajectory_fields_moved_to_cpu_when_flag_false(self, mocker, set_device_tensor_ipc):
        set_device_tensor_ipc(False)
        video_host, audio_host = torch.zeros(8), torch.ones(4)
        latents_host, timesteps_host = torch.zeros(2), torch.ones(2)
        out = DiffusionOutput(
            output=(_fake_device_tensor(mocker, video_host), _fake_device_tensor(mocker, audio_host)),
            trajectory_latents=_fake_device_tensor(mocker, latents_host),
            trajectory_timesteps=_fake_device_tensor(mocker, timesteps_host),
        )
        _pack_diffusion_fields(out)
        assert out.output[0] is video_host
        assert out.output[1] is audio_host
        assert out.trajectory_latents is latents_host
        assert out.trajectory_timesteps is timesteps_host

    @pytest.mark.parametrize("flag", [True, False])
    def test_diffusion_media_tensor_moved_to_cpu(self, mocker, set_device_tensor_ipc, flag):
        set_device_tensor_ipc(flag)
        host = torch.zeros(6)
        media = mocker.MagicMock()
        media.prepared_for_transport = True
        media.video.schema_version = 1
        media.video.tensor = _fake_device_tensor(mocker, host)
        packed = _pack_diffusion_media(media, d2h_stream=None)
        assert packed["video"]["tensor"] is host


class TestPackDiffusionOutputShmWithD2hStream:
    """Test pack_diffusion_output_shm top-level dispatch with d2h_stream."""

    def test_diffusion_output_path_with_stream(self, mocker):
        mock_pack_fields = mocker.patch(
            "vllm_omni.diffusion.ipc._pack_diffusion_fields",
        )
        d2h = mocker.MagicMock()
        out = DiffusionOutput()
        pack_diffusion_output_shm(out, d2h_stream=d2h)
        mock_pack_fields.assert_called_once_with(
            out,
            d2h_stream=d2h,
            created=ANY,
            pending_updates=ANY,
        )

    def test_rpc_envelope_path_with_stream(self, mocker):
        mock_pack_fields = mocker.patch(
            "vllm_omni.diffusion.ipc._pack_diffusion_fields",
        )
        d2h = mocker.MagicMock()
        result = DiffusionOutput()
        envelope = {"type": "diffusion_rpc_result", "result": result}
        pack_diffusion_output_shm(envelope, d2h_stream=d2h)
        mock_pack_fields.assert_called_once_with(
            result,
            d2h_stream=d2h,
            created=ANY,
            pending_updates=ANY,
        )

    @pytest.mark.parametrize("wrapped_in_rpc_envelope", [False, True])
    def test_dp_tagged_output_path_with_stream(self, mocker, wrapped_in_rpc_envelope):
        mock_pack_fields = mocker.patch(
            "vllm_omni.diffusion.ipc._pack_diffusion_fields",
        )
        d2h = mocker.MagicMock()
        result = DiffusionOutput()
        tagged = {"dp_rank": 1, "output": result}
        value = {"type": "diffusion_rpc_result", "result": tagged} if wrapped_in_rpc_envelope else tagged
        pack_diffusion_output_shm(value, d2h_stream=d2h)
        mock_pack_fields.assert_called_once_with(
            result,
            d2h_stream=d2h,
            created=ANY,
            pending_updates=ANY,
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
        mock_pack_fields.assert_called_once_with(
            result,
            d2h_stream=d2h,
            created=ANY,
            pending_updates=ANY,
        )
