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
    """Read the requested layout policy; each storage owner resolves auto."""
    from vllm_omni.diffusion import envs

    if not current_omni_platform.is_npu():
        return None
    mode = envs.VLLM_OMNI_NPU_INT8_WEIGHT_NZ_MODE.strip().lower()
    if mode == "off":
        return None
    if mode not in ("auto", "cast", "reactor"):
        raise ValueError("VLLM_OMNI_NPU_INT8_WEIGHT_NZ_MODE must be 'auto', 'off', 'cast' or 'reactor'")
    return mode


def _internal_format_enabled() -> bool:
    import torch_npu

    get_option = getattr(getattr(torch_npu, "_C", None), "_npu_getOption", None)
    return get_option is not None and get_option("ALLOW_INTERNAL_FORMAT") == b"enable"


def _auto_cast_available(device: torch.device) -> bool:
    import torch_npu

    # A2/A3 accept the finalized [K, N] right operand in FRACTAL_NZ.
    soc = torch_npu.npu.get_device_name(device).lower()
    return (
        soc.startswith(("ascend910b", "ascend910c", "ascend910_93"))
        and _internal_format_enabled()
        and callable(getattr(torch_npu, "empty_with_format", None))
        and callable(getattr(torch_npu, "get_npu_format", None))
    )


def auto_reactor_available() -> bool:
    """Limit automatic use of the borrowed-storage API to the validated runtime."""
    import torch_npu

    return (
        torch_npu.npu.get_device_name() == "Ascend910B4-1"
        and getattr(torch_npu, "__version__", "").partition("+")[0] == "2.10.0.post4"
        and _internal_format_enabled()
        and _reactor_descriptor_supported(torch_npu.npu.current_device())
    )


def int8_weight_names(model: nn.Module, *, online_only: bool = False) -> set[str]:
    """Select finalized NPU INT8 linear weights, optionally only online weights."""
    from vllm_omni.quantization.int8_config import NPUInt8LinearMethod, NPUInt8OnlineLinearMethod

    method = NPUInt8OnlineLinearMethod if online_only else NPUInt8LinearMethod
    return {
        f"{name}.weight" if name else "weight"
        for name, layer in model.named_modules()
        if isinstance(getattr(layer, "quant_method", None), method) and layer._parameters.get("weight") is not None
    }


class _NZFormatUnavailableError(RuntimeError):
    """The allocator returned ordinary storage after a request for NZ."""


def cast_weight_to_nz(weight: torch.Tensor) -> torch.Tensor:
    """Create NZ storage even when the source is a non-owning offload view."""
    import torch_npu

    if torch_npu.get_npu_format(weight) == _NZ_FORMAT:
        return weight
    result = torch_npu.empty_with_format(
        tuple(weight.shape), dtype=weight.dtype, device=weight.device, acl_format=_NZ_FORMAT
    )
    if torch_npu.get_npu_format(result) != _NZ_FORMAT:
        raise _NZFormatUnavailableError("INT8 weight NZ requires torch.npu.config.allow_internal_format=True")
    result.copy_(weight)
    return result


def prepare_resident_weights(model: nn.Module, excluded_parameter_ids: Set[int] = frozenset()) -> None:
    """Cast finalized resident weights, leaving offloader-owned parameters alone."""
    mode = get_weight_nz_mode()
    if mode is None:
        return
    # Query each device at most once per model load, after finding an eligible weight.
    auto_cast_available = cache(_auto_cast_available)
    for name in int8_weight_names(model):
        weight = model.get_parameter(name)
        if id(weight) in excluded_parameter_ids or weight.device.type != "npu":
            continue
        if mode == "auto" and not auto_cast_available(weight.device):
            continue
        try:
            weight.data = cast_weight_to_nz(weight.data)
        except _NZFormatUnavailableError:
            if mode != "auto":
                raise


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
def _reactor_descriptor_supported(device_index: int) -> bool:
    import torch_npu

    if not all(
        callable(getattr(torch_npu._C, name, None))
        for name in ("_construct_storage_from_data_pointer", "_construct_NPU_Tensor_From_Storage_And_Metadata")
    ) or not callable(getattr(torch_npu, "get_npu_format", None)):
        return False
    buffer = torch.zeros(_ROW_BLOCK * _COL_BLOCK, dtype=torch.int8, device=torch.device("npu", device_index))
    try:
        return torch_npu.get_npu_format(nz_view(buffer, 0, (_ROW_BLOCK, _COL_BLOCK))) == _NZ_FORMAT
    except (NotImplementedError, TypeError):
        return False


def check_reactor_support() -> None:
    """An explicit reactor request fails if this device cannot build NZ views."""
    import torch_npu

    if not _internal_format_enabled():
        raise RuntimeError("INT8 NZ reactor requires torch.npu.config.allow_internal_format=True")
    if not _reactor_descriptor_supported(torch_npu.npu.current_device()):
        raise RuntimeError("INT8 NZ reactor is unavailable on this torch_npu; select NZ mode 'cast'")
