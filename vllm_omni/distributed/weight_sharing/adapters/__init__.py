# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-model adapters that bind the CUDA IPC backend to concrete model backbones.

Each adapter defines the shared backbone boundary ("fusion ABI"), the canonical
tensor name space used in the manifest, and the provider/consumer entry points.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING

from ..contracts import CompatibilityError, TensorSpec, WeightSpec

if TYPE_CHECKING:
    import torch


def build_weight_spec(
    model_id: str,
    checkpoint_revision: str,
    bindings: Mapping[str, "torch.Tensor"],
    *,
    layout: str,
) -> WeightSpec:
    """Describe selected tensors in name order without allocating device memory.

    Adapters select the bindings; the caller supplies the final layout contract.
    Only dtype, shape and stride are read, so meta tensors are supported too.
    """
    if not layout:
        raise CompatibilityError("weight sharing requires a nonempty layout contract")
    if not bindings:
        raise CompatibilityError("no shared weight bindings were found")
    return WeightSpec(
        model_id=model_id,
        checkpoint_revision=checkpoint_revision,
        layout=layout,
        tensors=tuple(
            TensorSpec(
                name=name,
                dtype=str(tensor.dtype).removeprefix("torch."),
                shape=tuple(tensor.shape),
                stride=tuple(int(s) for s in tensor.stride()),
                alias_of=None,
            )
            for name, tensor in sorted(bindings.items())
        ),
    )
