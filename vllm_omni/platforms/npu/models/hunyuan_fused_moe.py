# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import vllm.forward_context as _vllm_fc
from vllm.config import VllmConfig
from vllm.distributed import get_ep_group
from vllm.distributed.parallel_state import (
    init_model_parallel_group as vllm_init_model_parallel_group,
)
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE
from vllm_ascend.ops.fused_moe.moe_comm_method import _MoECommMethods
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

from vllm_omni.diffusion.distributed.parallel_state import (
    get_expert_parallel_group_ranks,
    get_world_group,
)
from vllm_omni.diffusion.forward_context import get_forward_context as omni_get_ctx


def _ensure_forward_context_attr(name: str, annotation: Any, default: Any) -> None:
    if name not in _vllm_fc.ForwardContext.__annotations__:
        _vllm_fc.ForwardContext.__annotations__[name] = annotation
    if not hasattr(_vllm_fc.ForwardContext, name):
        setattr(_vllm_fc.ForwardContext, name, default)


def _set_hunyuan_fused_moe_forward_context(num_tokens: int) -> None:
    if not _vllm_fc.is_forward_context_available():
        return

    forward_context = _vllm_fc.get_forward_context()
    forward_context.num_tokens = num_tokens
    forward_context.moe_comm_type = _select_moe_comm_method(vllm_config=omni_get_ctx().vllm_config)
    forward_context.moe_comm_method = _MoECommMethods.get(forward_context.moe_comm_type)
    forward_context.flash_comm_v1_enabled = False


def _init_mc2_group_for_diffusion(
    world_size: int,
    mc2_group_size: int,
    backend: str,
    local_rank: int,
    group_ranks: list[list[int]] | None = None,
) -> None:
    import vllm_ascend.distributed.parallel_state as vllm_ascend_parallel_state

    if getattr(vllm_ascend_parallel_state, "_MC2", None) is not None:
        return
    if group_ranks is None:
        if world_size % mc2_group_size != 0:
            raise ValueError(f"world_size ({world_size}) must be divisible by mc2_group_size ({mc2_group_size})")
        all_ranks = torch.arange(world_size).reshape(-1, mc2_group_size)
        group_ranks = [x.tolist() for x in all_ranks.unbind(0)]
    elif any(len(ranks) != mc2_group_size for ranks in group_ranks):
        raise ValueError(
            f"All MC2 groups must have mc2_group_size ({mc2_group_size}) ranks, but got group_ranks={group_ranks}"
        )
    vllm_ascend_parallel_state._MC2 = vllm_init_model_parallel_group(
        group_ranks,
        local_rank,
        backend,
        group_name="mc2",
    )


def _sync_ascend_ep_group_for_diffusion() -> None:
    """Make vllm-ascend use vLLM's EP group for expert placement.

    vllm-omni maps diffusion SP to vLLM PCP and CFG to vLLM DP, so vLLM's
    FusedMoE computes EP from TP * PCP * DP. vllm-ascend keeps its own
    parallel-state module, so mirror the vLLM EP group there without changing
    FusedMoE's tp_size/pcp_size/dp_size inputs.
    """
    import vllm_ascend.distributed.parallel_state as vllm_ascend_parallel_state

    omni_ep_group = get_ep_group()
    ascend_ep_group = getattr(vllm_ascend_parallel_state, "_EP", None)
    if ascend_ep_group is not omni_ep_group:
        vllm_ascend_parallel_state._EP = omni_ep_group


def _select_moe_comm_method(vllm_config: VllmConfig) -> MoECommType | None:
    soc_version = get_ascend_device_type()
    if not vllm_config.parallel_config.enable_expert_parallel or get_ep_group().world_size == 1:
        moe_comm_type = MoECommType.ALLGATHER
    elif soc_version in {AscendDeviceType.A2}:
        moe_comm_type = MoECommType.ALLGATHER
    elif soc_version in {AscendDeviceType.A3}:
        moe_comm_type = MoECommType.ALLTOALL
    elif soc_version in {AscendDeviceType._310P}:
        moe_comm_type = MoECommType.ALLGATHER
    elif soc_version in {AscendDeviceType.A5}:
        moe_comm_type = MoECommType.ALLTOALL
    else:
        raise ValueError(f"Unsupported soc_version: {soc_version}")
    return moe_comm_type


def prepare_hunyuan_fused_moe_runtime() -> None:
    vllm_config = omni_get_ctx().vllm_config
    if vllm_config.parallel_config.enable_expert_parallel:
        world_size = torch.distributed.get_world_size()
        backend = torch.distributed.get_backend(get_world_group().device_group)
        local_rank = get_world_group().local_rank
        mc2_group_size = get_ep_group().world_size
        _sync_ascend_ep_group_for_diffusion()
        _init_mc2_group_for_diffusion(
            world_size=world_size,
            mc2_group_size=mc2_group_size,
            backend=backend,
            local_rank=local_rank,
            group_ranks=get_expert_parallel_group_ranks(),
        )

    moe_comm_type = _select_moe_comm_method(vllm_config=vllm_config)
    _ensure_forward_context_attr("num_tokens", int | None, None)
    _ensure_forward_context_attr("in_profile_run", bool, False)
    _ensure_forward_context_attr("moe_comm_type", MoECommType | None, moe_comm_type)
    _ensure_forward_context_attr("moe_comm_method", Any, _MoECommMethods.get(moe_comm_type))
    _ensure_forward_context_attr("flash_comm_v1_enabled", bool, False)


# NOTE: vLLM v0.20.0 folded SharedFusedMoE into FusedMoE, and vllm-ascend in turn
# removed AscendSharedFusedMoE — the shared-experts / gate / multistream-overlap
# paths now live directly on AscendFusedMoE and are activated by passing
# shared_experts= as a kwarg.
class AscendHunyuanFusedMoE(AscendFusedMoE):
    def __init__(self, *, prefix: str = "", **kwargs: Any) -> None:
        super().__init__(prefix=prefix, **kwargs)
        self._prefix = prefix

    def forward(self, hidden_states: Any, router_logits: Any) -> Any:
        _set_hunyuan_fused_moe_forward_context(hidden_states.shape[0])
        return super().forward(hidden_states, router_logits)

    def __del__(self):
        import vllm_ascend.distributed.parallel_state as vllm_ascend_parallel_state

        if vllm_ascend_parallel_state._MC2:
            vllm_ascend_parallel_state._MC2.destroy()
        vllm_ascend_parallel_state._MC2 = None
