# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LoRA support for Stable Audio 3.

PORT_FROM: stable_audio_3/models/lora/ (3 files, ~1000 lines total)
  - model.py:   LoRAParametrization, has_lora, enable_lora, disable_lora,
                set_lora_strength, filter_lora_layers (470 lines)
  - loader.py:  load_lora_from_safetensors + state_dict converter (111 lines)
  - utils.py:   helpers — module discovery, name remapping (419 lines)

Approach: upstream uses torch.nn.utils.parametrize.register_parametrization
to inject LoRA into nn.Linear layers in-place. The wrapped Linear's
.weight attribute is replaced by W + scaling * B @ A.

This is a different LoRA style than vllm-omni's BaseLinearLayerWithLoRA
wrapper (in vllm_omni/diffusion/lora/). We port upstream's approach in v1;
a follow-up PR can swap to vllm-omni style if needed for serving-side
LoRA hot-swap.

Public API matches upstream so the DiT's existing `from .lora import ...`
statements work after porting.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


class LoRAParametrization(nn.Module):
    """LoRA parametrization: W → W + (alpha/r) * B @ A.

    PORT_FROM: stable_audio_3/models/lora/model.py LoRAParametrization
    Registered onto an existing nn.Linear via torch.nn.utils.parametrize.register_parametrization.
    """

    def __init__(
        self,
        fan_in: int,
        fan_out: int,
        rank: int = 4,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = False,
    ) -> None:
        super().__init__()
        # PORT_FROM: model.py LoRAParametrization.__init__
        raise NotImplementedError

    def forward(self, original_weight: torch.Tensor) -> torch.Tensor:
        # PORT_FROM: model.py LoRAParametrization.forward — applies the LoRA delta
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Module-level helpers (PORT_FROM: model.py top-level functions)
# ---------------------------------------------------------------------------


def has_lora(module: nn.Module) -> bool:
    """Does this module (or any descendant) have a LoRA parametrization?

    PORT_FROM: model.py has_lora
    """
    raise NotImplementedError


def enable_lora(module: nn.Module) -> None:
    """Enable LoRA on all parametrized layers in the module tree.

    PORT_FROM: model.py enable_lora
    """
    raise NotImplementedError


def disable_lora(module: nn.Module) -> None:
    """Disable LoRA on all parametrized layers.

    PORT_FROM: model.py disable_lora
    """
    raise NotImplementedError


def set_lora_strength(module: nn.Module, strength: float) -> None:
    """Set LoRA alpha/scale dynamically — used for runtime adjustment.

    Per issue #3787: 'LoRA adapters are stackable and runtime-adjustable,
    which may want a serving-side knob later.' This is the hook.

    PORT_FROM: model.py set_lora_strength
    """
    raise NotImplementedError


def filter_lora_layers(module: nn.Module) -> Iterable[nn.Module]:
    """Yield all LoRA-parametrized layers in the module tree.

    PORT_FROM: model.py filter_lora_layers
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Loader (PORT_FROM: stable_audio_3/models/lora/loader.py — 111 lines)
# ---------------------------------------------------------------------------


def load_lora_from_safetensors(
    model: nn.Module,
    safetensors_path: str,
    strict: bool = False,
) -> set[str]:
    """Load a LoRA checkpoint (.safetensors) into the model.

    PORT_FROM: loader.py load_lora_from_safetensors
    Returns the set of parameter names that were loaded.
    """
    raise NotImplementedError
