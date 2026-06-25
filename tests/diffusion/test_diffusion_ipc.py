# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.ipc import (
    _SHM_TENSOR_THRESHOLD,
    DIFFUSION_RPC_RESULT_ENVELOPE,
    _pack_value_if_large,
    _unpack_if_shm_handle,
    pack_diffusion_output_shm,
    unpack_diffusion_output_shm,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _large_numel(dtype: torch.dtype) -> int:
    return (_SHM_TENSOR_THRESHOLD // torch.empty((), dtype=dtype).element_size()) + 1


def _cleanup_shm_handle(value: object) -> None:
    if isinstance(value, dict) and value.get("__tensor_shm__"):
        with contextlib.suppress(FileNotFoundError):
            _unpack_if_shm_handle(value)


def test_diffusion_output_dict_tensors_round_trip_through_shm() -> None:
    image = torch.arange(300_000, dtype=torch.float32)
    video = torch.arange(300_000, dtype=torch.float32) * 2
    output = DiffusionOutput(output={"image": image, "video": video, "metadata": {"keep": "inline"}})

    pack_diffusion_output_shm(output)

    assert output.output["image"]["__tensor_shm__"] is True
    assert output.output["video"]["__tensor_shm__"] is True
    assert output.output["metadata"] == {"keep": "inline"}

    unpack_diffusion_output_shm(output)

    torch.testing.assert_close(output.output["image"], image)
    torch.testing.assert_close(output.output["video"], video)
    assert output.output["metadata"] == {"keep": "inline"}


def test_diffusion_output_tuple_tensors_round_trip_through_shm() -> None:
    # LTX2 / DreamID return (video, audio) tuples as DiffusionOutput.output.
    video = torch.arange(300_000, dtype=torch.float32)
    audio = torch.arange(300_000, dtype=torch.float32) * 3
    output = DiffusionOutput(output=(video, audio))

    pack_diffusion_output_shm(output)

    assert isinstance(output.output, tuple)
    assert output.output[0]["__tensor_shm__"] is True
    assert output.output[1]["__tensor_shm__"] is True

    unpack_diffusion_output_shm(output)

    assert isinstance(output.output, tuple)
    torch.testing.assert_close(output.output[0], video)
    torch.testing.assert_close(output.output[1], audio)


def test_diffusion_output_list_tensors_round_trip_through_shm() -> None:
    frames = [torch.arange(300_000, dtype=torch.float32), torch.arange(300_000, dtype=torch.float32) + 1]
    output = DiffusionOutput(output=list(frames))

    pack_diffusion_output_shm(output)

    assert isinstance(output.output, list)
    assert all(isinstance(item, dict) and item["__tensor_shm__"] is True for item in output.output)

    unpack_diffusion_output_shm(output)

    assert isinstance(output.output, list)
    torch.testing.assert_close(output.output[0], frames[0])
    torch.testing.assert_close(output.output[1], frames[1])


def test_rpc_result_envelope_diffusion_output_round_trips_through_shm() -> None:
    tensor = torch.arange(300_000, dtype=torch.float32)
    envelope = {
        "type": DIFFUSION_RPC_RESULT_ENVELOPE,
        "result": DiffusionOutput(output=tensor),
        "rank_statuses": [{"rank": 0, "ok": True}],
    }

    packed = pack_diffusion_output_shm(envelope)

    assert packed is envelope
    result = packed["result"]
    assert isinstance(result, DiffusionOutput)
    assert result.output["__tensor_shm__"] is True
    assert packed["rank_statuses"] == [{"rank": 0, "ok": True}]

    unpacked = unpack_diffusion_output_shm(packed)

    assert unpacked is envelope
    result = unpacked["result"]
    assert isinstance(result, DiffusionOutput)
    torch.testing.assert_close(result.output, tensor)
    assert unpacked["rank_statuses"] == [{"rank": 0, "ok": True}]


def test_pack_value_keeps_tensor_at_threshold_inline() -> None:
    tensor = torch.arange(
        _SHM_TENSOR_THRESHOLD // torch.empty((), dtype=torch.float32).element_size(),
        dtype=torch.float32,
    )

    packed = _pack_value_if_large(tensor)

    assert packed is tensor


def test_pack_value_packs_large_tensor_and_round_trips() -> None:
    tensor = torch.arange(_large_numel(torch.float32), dtype=torch.float32)
    packed = _pack_value_if_large(tensor)

    try:
        assert isinstance(packed, dict)
        assert packed["__tensor_shm__"] is True
        assert packed["shape"] == [tensor.numel()]
        assert packed["torch_dtype"] == "torch.float32"

        unpacked = _unpack_if_shm_handle(packed)
        assert isinstance(unpacked, torch.Tensor)
        torch.testing.assert_close(unpacked, tensor)
    finally:
        _cleanup_shm_handle(packed)


def test_pack_value_recurses_nested_dicts_and_lists_without_mutating_inline_values() -> None:
    large = torch.arange(_large_numel(torch.float32), dtype=torch.float32)
    small = torch.arange(8, dtype=torch.float32)
    list_tensor = torch.arange(_large_numel(torch.float32), dtype=torch.float32)
    payload = {
        "media": {
            "large": large,
            "small": small,
        },
        "list_value": [list_tensor],
        "metadata": {"prompt": "keep inline"},
    }

    packed = _pack_value_if_large(payload)

    try:
        assert packed is not payload
        assert packed["media"] is not payload["media"]
        assert packed["media"]["large"]["__tensor_shm__"] is True
        assert packed["media"]["small"] is small
        # Lists are recursed too: the large tensor inside is packed and a new
        # list is returned, while the input payload is left untouched.
        assert packed["list_value"] is not payload["list_value"]
        assert packed["list_value"][0]["__tensor_shm__"] is True
        assert payload["list_value"][0] is list_tensor
        assert packed["metadata"] == {"prompt": "keep inline"}

        torch.testing.assert_close(_unpack_if_shm_handle(packed["media"]["large"]), large)
        torch.testing.assert_close(_unpack_if_shm_handle(packed["list_value"][0]), list_tensor)
    finally:
        if isinstance(packed, dict):
            _cleanup_shm_handle(packed.get("media", {}).get("large"))
            list_value = packed.get("list_value")
            if isinstance(list_value, list) and list_value:
                _cleanup_shm_handle(list_value[0])


def test_pack_value_preserves_dtype_shape_and_values_for_bfloat16() -> None:
    tensor = torch.arange(_large_numel(torch.bfloat16), dtype=torch.float32).to(torch.bfloat16).reshape(1, -1)
    packed = _pack_value_if_large(tensor)

    try:
        assert isinstance(packed, dict)
        assert packed["__tensor_shm__"] is True
        assert packed["shape"] == list(tensor.shape)
        assert packed["torch_dtype"] == "torch.bfloat16"
        assert packed["numpy_dtype"] == "float32"

        unpacked = _unpack_if_shm_handle(packed)
        assert isinstance(unpacked, torch.Tensor)
        assert unpacked.dtype == torch.bfloat16
        torch.testing.assert_close(unpacked, tensor)
    finally:
        _cleanup_shm_handle(packed)


def test_pack_value_packs_non_contiguous_large_tensor_values() -> None:
    tensor = torch.arange(_large_numel(torch.float32) * 2, dtype=torch.float32).reshape(-1, 2)[:, 0]
    assert not tensor.is_contiguous()

    packed = _pack_value_if_large(tensor)

    try:
        assert isinstance(packed, dict)
        assert packed["__tensor_shm__"] is True
        assert packed["shape"] == list(tensor.shape)

        unpacked = _unpack_if_shm_handle(packed)
        assert isinstance(unpacked, torch.Tensor)
        torch.testing.assert_close(unpacked, tensor)
    finally:
        _cleanup_shm_handle(packed)
