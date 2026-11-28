# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stream DiT submodules through the existing DLO storage and event protocol.

AdaLN -> attention -> MLP is one circular weight ring. Token refiners and
encoders retain block granularity. All groups share the DLO two-slot buffers,
including padded AllGather shards, mmap staging, skipped-hook recovery and
failure-safe cleanup. Reference KV staging remains independently owned.
"""

from __future__ import annotations

from collections.abc import Sequence

from torch import nn
from vllm.logger import init_logger

from .component_utils import move_non_block_state_to_device
from .config import DIT_COMPONENT
from .distributed_layerwise_backend import (
    DistributedLayerwiseOffloadBackend,
    DistributedLayerwiseOffloadHook,
)

logger = init_logger(__name__)

# The hook's current/next ownership, stream events and restore path must be
# identical at block and submodule granularity.
SubmoduleOffloadHook = DistributedLayerwiseOffloadHook


def _get_submodules(block: nn.Module) -> list[nn.Module]:
    modules = [getattr(block, name, None) for name in ("adaln_proj", "attn", "mlp")]
    if not all(isinstance(module, nn.Module) for module in modules):
        return []
    return modules


class SubmoduleOffloadBackend(DistributedLayerwiseOffloadBackend):
    """Use DLO transport with a finer weight unit for compatible DiT stacks."""

    def _install_hook_group(
        self,
        blocks: Sequence[nn.Module] | nn.ModuleList,
        component: str,
        *,
        use_dit_mmap: bool = False,
    ) -> list[DistributedLayerwiseOffloadHook]:
        block_list = list(blocks)
        submodules = [_get_submodules(block) for block in block_list]
        if component != DIT_COMPONENT or not block_list or not all(submodules):
            return super()._install_hook_group(block_list, component, use_dit_mmap=use_dit_mmap)

        units = [module for group in submodules for module in group]
        if len({id(module) for module in units}) != len(units):
            raise ValueError("Submodule offload cannot stream shared submodule instances twice")
        for block, group in zip(block_list, submodules, strict=True):
            # Norms and any other state outside streamed leaves stay resident.
            move_non_block_state_to_device(block, (group,), self.device)
        hooks = super()._install_hook_group(units, component, use_dit_mmap=use_dit_mmap)
        logger.info("Submodule offload: split %d DiT blocks into %d weight units", len(block_list), len(units))
        return hooks
