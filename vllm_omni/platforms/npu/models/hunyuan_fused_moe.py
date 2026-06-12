# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import vllm.forward_context as _vllm_fc
from vllm.config import VllmConfig
from vllm.distributed import get_ep_group
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import (
    init_model_parallel_group as vllm_init_model_parallel_group,
)
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE
from vllm_ascend.ops.fused_moe.moe_comm_method import _MoECommMethods
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

from vllm_omni.diffusion.distributed.parallel_state import (
    get_data_parallel_world_size,
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
    expert_parallel_size: int,
    backend: str,
    local_rank: int,
) -> None:
    import vllm_ascend.distributed.parallel_state as vllm_ascend_parallel_state

    if getattr(vllm_ascend_parallel_state, "_MC2", None) is not None:
        return
    if world_size % expert_parallel_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by expert_parallel_size ({expert_parallel_size})"
        )
    all_ranks = torch.arange(world_size).reshape(-1, expert_parallel_size)
    group_ranks = all_ranks.unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    vllm_ascend_parallel_state._MC2 = vllm_init_model_parallel_group(
        group_ranks,
        local_rank,
        backend,
        group_name="mc2",
    )


def _sync_ascend_ep_group_for_diffusion() -> None:
    """Make vllm-ascend use Omni's diffusion EP group for expert placement.

    HunyuanImage3 diffusion EP includes the SP dimension. vllm-ascend keeps its
    own parallel-state module, so without this sync its FusedMoE may size local
    experts from the TP-only EP group while token dispatch uses the diffusion
    MC2 group.
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
    world_size = torch.distributed.get_world_size()
    vllm_config = omni_get_ctx().vllm_config
    tp_size = get_tensor_model_parallel_world_size()
    dp_size = get_data_parallel_world_size()
    if vllm_config.parallel_config.enable_expert_parallel:
        expert_parallel_size = get_ep_group().world_size
    else:
        expert_parallel_size = dp_size * tp_size
    backend = torch.distributed.get_backend(get_world_group().device_group)
    local_rank = get_world_group().local_rank
    if vllm_config.parallel_config.enable_expert_parallel:
        _sync_ascend_ep_group_for_diffusion()
    _init_mc2_group_for_diffusion(
        world_size=world_size,
        expert_parallel_size=expert_parallel_size,
        backend=backend,
        local_rank=local_rank,
    )

    moe_comm_type = _select_moe_comm_method(vllm_config=vllm_config)
    _ensure_forward_context_attr("num_tokens", int | None, None)
    _ensure_forward_context_attr("in_profile_run", bool, False)
    _ensure_forward_context_attr("moe_comm_type", MoECommType | None, moe_comm_type)
    _ensure_forward_context_attr("moe_comm_method", Any, _MoECommMethods.get(moe_comm_type))
    _ensure_forward_context_attr("flash_comm_v1_enabled", bool, False)


def _set_effective_moe_tp_size(kwargs: dict[str, Any]) -> None:
    vllm_config = omni_get_ctx().vllm_config
    if not vllm_config.parallel_config.enable_expert_parallel:
        return

    effective_ep_size = get_ep_group().world_size
    dp_size = kwargs.get("dp_size", get_data_parallel_world_size())
    pcp_size = kwargs.get("pcp_size", 1)
    flatten_size = dp_size * pcp_size
    if effective_ep_size % flatten_size != 0:
        raise ValueError(
            f"effective_ep_size ({effective_ep_size}) must be divisible by dp_size * pcp_size ({flatten_size})"
        )
    # vLLM MoE folds DP/PCP into TP while constructing the EP layout.
    kwargs["tp_size"] = effective_ep_size // flatten_size


def _sync_effective_moe_ep_rank(layer: Any) -> None:
    vllm_config = omni_get_ctx().vllm_config
    if not vllm_config.parallel_config.enable_expert_parallel:
        return

    ep_group = get_ep_group()
    moe_parallel_config = getattr(layer, "moe_parallel_config", None)
    if moe_parallel_config is None:
        return

    current_ep_size = getattr(layer, "ep_size", None)
    current_ep_rank = getattr(layer, "ep_rank", None)
    if current_ep_size == ep_group.world_size and current_ep_rank == ep_group.rank_in_group:
        return

    moe_parallel_config.ep_size = ep_group.world_size
    moe_parallel_config.ep_rank = ep_group.rank_in_group
    moe_parallel_config.tp_size = 1
    moe_parallel_config.tp_rank = 0
    moe_parallel_config.use_ep = True

    moe_config = getattr(layer, "moe_config", None)
    if moe_config is not None:
        moe_config.moe_parallel_config = moe_parallel_config
        moe_config.num_local_experts = layer.global_num_experts // ep_group.world_size
        if hasattr(moe_config, "ep_group"):
            moe_config.ep_group = ep_group

    expert_map_manager = getattr(layer, "expert_map_manager", None)
    if expert_map_manager is not None:
        expert_map_manager.update(
            moe_parallel_config,
            global_num_experts=layer.global_num_experts,
        )
        layer.local_num_experts = expert_map_manager.local_num_experts
        layer.expert_placement_strategy = expert_map_manager.placement_strategy
        layer._expert_map = expert_map_manager.expert_map
        layer.expert_mask = expert_map_manager.expert_mask
        if moe_config is not None:
            moe_config.num_local_experts = layer.local_num_experts


# NOTE: vLLM v0.20.0 folded SharedFusedMoE into FusedMoE, and vllm-ascend in turn
# removed AscendSharedFusedMoE — the shared-experts / gate / multistream-overlap
# paths now live directly on AscendFusedMoE and are activated by passing
# shared_experts= as a kwarg.
class AscendHunyuanFusedMoE(AscendFusedMoE):
    def __init__(self, *, prefix: str = "", **kwargs: Any) -> None:
        kwargs = dict(kwargs)
        _set_effective_moe_tp_size(kwargs)
        super().__init__(prefix=prefix, **kwargs)
        _sync_effective_moe_ep_rank(self)
        self._prefix = prefix

    def forward(self, hidden_states: Any, router_logits: Any) -> Any:
        _set_hunyuan_fused_moe_forward_context(hidden_states.shape[0])
        return super().forward(hidden_states, router_logits)

    def __del__(self):
        import vllm_ascend.distributed.parallel_state as vllm_ascend_parallel_state

        if vllm_ascend_parallel_state._MC2:
            vllm_ascend_parallel_state._MC2.destroy()
        vllm_ascend_parallel_state._MC2 = None
