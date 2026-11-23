# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Optional FRACTAL_NZ storage for finalized NPU INT8 linear weights.

Quantization retains its existing logical [K, N] weights and scales. Resident
weights are cast after device placement; offload owners use the packing and
view helpers to preserve NZ across their own storage transitions.
"""

import os
from collections.abc import Sequence, Set

import torch
from torch import nn

from vllm_omni.platforms import current_omni_platform

_NZ_FORMAT = 29
_ROW_BLOCK = 16
_COL_BLOCK = 32
_verified_view_devices: set[torch.device] = set()


def configured_weight_nz_mode() -> str:
    return os.getenv("VLLM_OMNI_NPU_INT8_WEIGHT_NZ_MODE", "off").strip().lower()


def get_weight_nz_mode() -> str | None:
    """Return the explicit layout mode on NPU, or None when disabled."""
    if not current_omni_platform.is_npu():
        return None
    mode = configured_weight_nz_mode()
    if mode == "off":
        return None
    if mode not in ("cast", "reactor"):
        raise ValueError("VLLM_OMNI_NPU_INT8_WEIGHT_NZ_MODE must be 'off', 'cast' or 'reactor'")
    return mode


def int8_weight_names(model: nn.Module) -> set[str]:
    """Select finalized weights for online and checkpoint NPU INT8 linear layers."""
    from vllm_omni.quantization.int8_config import NPUInt8LinearMethod

    return {
        f"{name}.weight" if name else "weight"
        for name, layer in model.named_modules()
        if isinstance(getattr(layer, "quant_method", None), NPUInt8LinearMethod)
        and layer._parameters.get("weight") is not None
    }


def cast_weight_to_nz(weight: torch.Tensor, *, resident: bool = False) -> torch.Tensor:
    """Create NZ storage for a resident weight or a non-owning offload view."""
    import torch_npu

    if torch_npu.get_npu_format(weight) == _NZ_FORMAT:
        return weight
    if resident and torch_npu.npu.get_device_name(weight.device).startswith("Ascend950"):
        # Ascend 950 supports this explicit cast with ALLOW_INTERNAL_FORMAT disabled.
        result = torch_npu.npu_format_cast(weight, _NZ_FORMAT)
        torch_npu.npu.current_stream(weight.device).synchronize()
    else:
        result = torch_npu.empty_with_format(
            tuple(weight.shape), dtype=weight.dtype, device=weight.device, acl_format=_NZ_FORMAT
        )
        result.copy_(weight)
    if torch_npu.get_npu_format(result) != _NZ_FORMAT:
        raise RuntimeError("INT8 weight conversion did not return FRACTAL_NZ")
    return result


def prepare_resident_weights(model: nn.Module, excluded_parameter_ids: Set[int] = frozenset()) -> None:
    """Cast finalized resident weights, leaving offloader-owned parameters alone."""
    for name in int8_weight_names(model):
        weight = model.get_parameter(name)
        if weight.device.type == "npu" and id(weight) not in excluded_parameter_ids:
            weight.data = cast_weight_to_nz(weight.data, resident=True)


def nz_storage_numel(shape: Sequence[int]) -> int:
    """Physical INT8 NZ footprint, including padding of both logical axes."""
    rows, cols = shape
    return ((rows + 15) // _ROW_BLOCK) * ((cols + 31) // _COL_BLOCK) * _ROW_BLOCK * _COL_BLOCK


def pack_nz_weight(weight: torch.Tensor) -> torch.Tensor:
    """Pack a CPU logical matrix into INT8 FRACTAL_NZ byte order."""
    rows, cols = weight.shape
    row_blocks = (rows + 15) // _ROW_BLOCK
    col_blocks = (cols + 31) // _COL_BLOCK
    if rows % _ROW_BLOCK or cols % _COL_BLOCK:
        padded = weight.new_zeros((row_blocks * _ROW_BLOCK, col_blocks * _COL_BLOCK))
        padded[:rows, :cols].copy_(weight)
        weight = padded
    return weight.reshape(row_blocks, _ROW_BLOCK, col_blocks, _COL_BLOCK).permute(2, 0, 1, 3).contiguous().view(-1)


def unpack_nz_weight(storage: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    """Restore ordinary CPU weights from NZ transport bytes, removing padding."""
    rows, cols = shape
    row_blocks = (rows + 15) // _ROW_BLOCK
    col_blocks = (cols + 31) // _COL_BLOCK
    padded = (
        storage.reshape(col_blocks, row_blocks, _ROW_BLOCK, _COL_BLOCK)
        .permute(1, 2, 0, 3)
        .reshape(row_blocks * _ROW_BLOCK, col_blocks * _COL_BLOCK)
    )
    return padded[:rows, :cols].contiguous()


def nz_view(buffer: torch.Tensor, offset: int, shape: Sequence[int]) -> torch.Tensor:
    """Bind existing NZ bytes without a device conversion or allocation.

    The returned tensor does not own its storage. Its offload owner must keep
    the backing buffer alive until all users of the view have finished.
    """
    import torch_npu

    rows, cols = shape
    metadata = {
        "size": (rows, cols),
        "stride": (cols, 1),
        "nbytes": nz_storage_numel(shape) * buffer.element_size(),
        "dtype": buffer.dtype,
        "data_ptr": buffer.data_ptr() + offset * buffer.element_size(),
        "storage_offset": 0,
        "device": buffer.device,
        "npu_format": _NZ_FORMAT,
        "layout": torch.strided,
        "requires_grad": False,
    }
    storage = torch_npu._C._construct_storage_from_data_pointer(
        metadata["data_ptr"], metadata["device"], metadata["nbytes"]
    )
    view = torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(metadata, storage)
    # Validate the private descriptor API on the first real upload per device.
    if buffer.device not in _verified_view_devices:
        if torch_npu.get_npu_format(view) != _NZ_FORMAT:
            raise RuntimeError("INT8 NZ reactor could not bind a FRACTAL_NZ view; select NZ mode 'cast'")
        _verified_view_devices.add(buffer.device)
    return view


def check_reactor_support() -> None:
    """Check the runtime prerequisites before packing weights on the host."""
    import torch_npu

    get_option = getattr(torch_npu._C, "_npu_getOption", None)
    if get_option is None or get_option("ALLOW_INTERNAL_FORMAT") != b"enable":
        raise RuntimeError("INT8 NZ reactor requires torch.npu.config.allow_internal_format=True")
    if not all(
        callable(getattr(torch_npu._C, name, None))
        for name in ("_construct_storage_from_data_pointer", "_construct_NPU_Tensor_From_Storage_And_Metadata")
    ):
        raise RuntimeError("INT8 NZ reactor is unavailable on this torch_npu; select NZ mode 'cast'")
