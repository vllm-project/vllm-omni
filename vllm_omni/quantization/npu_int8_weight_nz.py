# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Optional FRACTAL_NZ storage for finalized NPU INT8 linear weights.

Quantization retains its existing logical [K, N] weights and scales. Resident
weights are cast after device placement; offload owners use the packing and
view helpers to preserve NZ across their own storage transitions.
"""

from collections.abc import Iterable, Sequence, Set
from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig


_NZ_FORMAT = 29
_ROW_BLOCK = 16
_COL_BLOCK = 32
_verified_view_devices: set[torch.device] = set()
_verified_a5_reactor_devices: set[torch.device] = set()


def stage_uses_weight_nz(config: "QuantizationConfig | None") -> bool:
    from vllm_omni.quantization.component_config import ComponentQuantizationConfig
    from vllm_omni.quantization.int8_config import DiffusionInt8Config

    configs = (
        (*config.component_configs.values(), config.default_config)
        if isinstance(config, ComponentQuantizationConfig)
        else (config,)
    )
    return any(isinstance(item, DiffusionInt8Config) and item.weight_nz_mode != "off" for item in configs)


def int8_weight_modes(model: nn.Module) -> dict[str, str]:
    """Read each finalized NPU INT8 layer's resolved component config."""
    if not current_omni_platform.is_npu():
        return {}
    from vllm_omni.quantization.int8_config import NPUInt8LinearMethod

    return {
        (f"{name}.weight" if name else "weight"): method.quant_config.weight_nz_mode
        for name, layer in model.named_modules()
        if isinstance(method := getattr(layer, "quant_method", None), NPUInt8LinearMethod)
        and method.quant_config.weight_nz_mode != "off"
        and layer._parameters.get("weight") is not None
    }


def cast_weight_to_nz(weight: torch.Tensor, *, resident: bool = False) -> torch.Tensor:
    """Create NZ storage for a resident weight or a non-owning offload view."""
    import torch_npu

    if torch_npu.get_npu_format(weight) == _NZ_FORMAT:
        return weight
    if torch_npu.npu.get_device_name(weight.device).startswith("Ascend950"):
        # A DLO weight can be a view at a nonzero offset into its transfer slot.
        # The A5 cast requires owned ND storage for such inputs.
        if not resident:
            weight = weight.clone(memory_format=torch.contiguous_format)
        result = torch_npu.npu_format_cast(weight, _NZ_FORMAT)
        if resident:
            torch_npu.npu.current_stream(weight.device).synchronize()
    else:
        result = torch_npu.empty_with_format(
            tuple(weight.shape), dtype=weight.dtype, device=weight.device, acl_format=_NZ_FORMAT
        )
        result.copy_(weight)
    if torch_npu.get_npu_format(result) != _NZ_FORMAT:
        raise RuntimeError("INT8 weight conversion did not return FRACTAL_NZ")
    return result


def prepare_resident_weights(
    model: nn.Module, names: Iterable[str], excluded_parameter_ids: Set[int] = frozenset()
) -> None:
    """Cast finalized resident weights, leaving offloader-owned parameters alone."""
    for name in names:
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


def _probe_a5_reactor(device: torch.device) -> None:
    """Verify a borrowed, padded NZ weight with the actual INT8 matmul once per device."""
    import torch_npu

    if device in _verified_a5_reactor_devices:
        return

    rows, cols = 17, 33
    offset = 256
    weight = (torch.arange(rows * cols, dtype=torch.int32) % 251 - 125).to(torch.int8).reshape(rows, cols)
    packed = pack_nz_weight(weight)
    slot = torch.empty(offset + packed.numel(), device=device, dtype=torch.int8)
    slot[offset:].copy_(packed)
    try:
        view = nz_view(slot, offset, weight.shape)
        if torch_npu.get_npu_format(view) != _NZ_FORMAT or view.data_ptr() != slot.data_ptr() + offset:
            raise RuntimeError("the weight does not borrow the expected FRACTAL_NZ storage")
        if not torch.equal(view.cpu(), weight):
            raise RuntimeError("the NZ view changed the logical INT8 weight")

        activation = (torch.arange(16 * rows, dtype=torch.int32) % 253 - 126).to(torch.int8).reshape(16, rows)
        activation = activation.to(device)
        weight_scale = torch.full((cols,), 0.00625, device=device, dtype=torch.float32)
        token_scale = torch.full((16,), 0.0078125, device=device, dtype=torch.float32)
        options = dict(pertoken_scale=token_scale, output_dtype=torch.bfloat16)
        reference = torch_npu.npu_quant_matmul(activation, weight.to(device), weight_scale, **options)
        actual = torch_npu.npu_quant_matmul(activation, view, weight_scale, **options)
        if not torch.equal(actual.cpu(), reference.cpu()):
            raise RuntimeError("the NZ matmul differs from the ND matmul")
    except RuntimeError as exc:
        raise RuntimeError("INT8 NZ reactor failed the Ascend 950 borrowed-storage probe") from exc

    _verified_a5_reactor_devices.add(device)


def check_reactor_support() -> None:
    """Check the runtime prerequisites before packing weights on the host."""
    import torch_npu

    get_option = getattr(torch_npu._C, "_npu_getOption", None)
    option = get_option("ALLOW_INTERNAL_FORMAT") if get_option is not None else None
    a5_probe = option == b"disable" and torch_npu.npu.get_device_name().startswith("Ascend950")
    if option != b"enable" and not a5_probe:
        raise RuntimeError("INT8 NZ reactor requires torch.npu.config.allow_internal_format=True")
    if not all(
        callable(getattr(torch_npu._C, name, None))
        for name in ("_construct_storage_from_data_pointer", "_construct_NPU_Tensor_From_Storage_And_Metadata")
    ):
        raise RuntimeError("INT8 NZ reactor is unavailable on this torch_npu; select NZ mode 'cast'")
    if a5_probe:
        _probe_a5_reactor(torch.device("npu", torch_npu.npu.current_device()))
