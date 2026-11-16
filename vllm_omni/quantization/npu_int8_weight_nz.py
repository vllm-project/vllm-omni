# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Optional FRACTAL_NZ storage for finalized NPU INT8 linear weights.

Quantization retains its existing logical [K, N] weights and scales. Resident
weights are cast after device placement; offload owners use the packing and
view helpers to preserve NZ across their own storage transitions.
"""

from collections.abc import Sequence, Set
from functools import cache

import torch
from torch import nn

from vllm_omni.platforms import current_omni_platform

_NZ_FORMAT = 29
_ROW_BLOCK = 16
_COL_BLOCK = 32


def get_weight_nz_mode() -> str | None:
    """Resolve the opt-in once per storage owner, without enabling offload."""
    from vllm_omni.diffusion import envs

    if not envs.VLLM_OMNI_NPU_INT8_WEIGHT_NZ or not current_omni_platform.is_npu():
        return None
    mode = envs.VLLM_OMNI_NPU_INT8_WEIGHT_NZ_MODE.strip().lower()
    if mode not in ("cast", "reactor"):
        raise ValueError("VLLM_OMNI_NPU_INT8_WEIGHT_NZ_MODE must be 'cast' or 'reactor'")
    return mode


def int8_weight_names(model: nn.Module) -> set[str]:
    """Select weights owned by NPU INT8 methods, including the online subclass."""
    from vllm_omni.quantization.int8_config import NPUInt8LinearMethod

    return {
        f"{name}.weight" if name else "weight"
        for name, layer in model.named_modules()
        if isinstance(getattr(layer, "quant_method", None), NPUInt8LinearMethod)
        and layer._parameters.get("weight") is not None
    }


def cast_weight_to_nz(weight: torch.Tensor) -> torch.Tensor:
    """Create NZ storage even when the source is a non-owning offload view."""
    import torch_npu

    if torch_npu.get_npu_format(weight) == _NZ_FORMAT:
        return weight
    result = torch_npu.empty_with_format(
        tuple(weight.shape), dtype=weight.dtype, device=weight.device, acl_format=_NZ_FORMAT
    )
    result.copy_(weight)
    if torch_npu.get_npu_format(result) != _NZ_FORMAT:
        raise RuntimeError("INT8 weight NZ requires torch.npu.config.allow_internal_format=True")
    return result


def prepare_resident_weights(model: nn.Module, excluded_parameter_ids: Set[int] = frozenset()) -> None:
    """Cast finalized resident weights, leaving offloader-owned parameters alone."""
    if get_weight_nz_mode() is None:
        return
    for name in int8_weight_names(model):
        weight = model.get_parameter(name)
        if id(weight) not in excluded_parameter_ids and weight.device.type == "npu":
            weight.data = cast_weight_to_nz(weight.data)


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
    return torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(metadata, storage)


@cache
def check_reactor_support() -> None:
    """Check the private storage API once before preparing reactor host bytes."""
    import torch_npu

    try:
        buffer = torch.zeros(_ROW_BLOCK * _COL_BLOCK, dtype=torch.int8, device="npu")
        if torch_npu.get_npu_format(nz_view(buffer, 0, (_ROW_BLOCK, _COL_BLOCK))) == _NZ_FORMAT:
            return
    except (AttributeError, RuntimeError) as exc:
        raise RuntimeError("INT8 NZ reactor is unavailable on this torch_npu; select NZ mode 'cast'") from exc
    raise RuntimeError("INT8 NZ reactor requires internal-format support; select NZ mode 'cast'")
