# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import vllm.distributed.parallel_state as vllm_parallel_state
import vllm.forward_context as _vllm_fc
from vllm.config import VllmConfig
from vllm.distributed import get_dp_group, get_ep_group
from vllm.distributed.parallel_state import (
    init_model_parallel_group as vllm_init_model_parallel_group,
)
from vllm_ascend.ascend_forward_context import MoECommType
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


def _ensure_ascend_moe_customop_registered() -> None:
    """Register vllm-ascend's out-of-tree MoE layer implementations here.

    vllm-ascend registers its OOT layers from ``register_ascend_customop``,
    which only ever runs inside their own ``NPUWorker.__init__``. The diffusion
    path runs on omni's DiffusionWorker, so the OOT registry stays empty and
    ``PluggableLayer.__new__`` falls back to the upstream classes. FusedMoE then
    breaks outright: vllm-ascend's ``patch_fused_moe`` passes
    ``n_shared_experts`` through ``routed_experts_args``, which only
    ``AscendRoutedExperts`` accepts.

    Only the MoE pair is registered, not the full ``REGISTERED_ASCEND_OPS``
    table. Most of that table assumes a text-generation ``ModelConfig``:
    ``AscendRotaryEmbedding.__init__`` reads ``model_config.use_mla`` and
    ``hf_text_config.qk_rope_head_dim``, neither of which exists on the
    diffusion config shim. ``AscendRoutedExperts``/``AscendMoERunner`` only need
    ``parallel_config``, which the shim does provide.
    """
    from vllm.model_executor.custom_op import CustomOp, op_registry_oot
    from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
    from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts

    ascend_moe_ops: dict[str, type] = {
        "MoERunner": AscendMoERunner,
        "RoutedExperts": AscendRoutedExperts,
    }
    for name, op_cls in ascend_moe_ops.items():
        # register_oot asserts on duplicates, so skip names vllm-ascend already
        # claimed. Its own registration is authoritative when it has run.
        if name in op_registry_oot:
            continue
        CustomOp.register_oot(_decorated_op_cls=op_cls, name=name)


def _init_mc2_group_for_diffusion(
    backend: str,
    local_rank: int,
    group_ranks: list[list[int]],
) -> None:
    import vllm_ascend.distributed.parallel_state as vllm_ascend_parallel_state

    if getattr(vllm_ascend_parallel_state, "_MC2", None) is not None:
        return
    vllm_ascend_parallel_state._MC2 = vllm_init_model_parallel_group(
        group_ranks,
        local_rank,
        backend,
        group_name="mc2",
    )


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


def prepare_fused_moe_runtime() -> None:
    vllm_config = omni_get_ctx().vllm_config

    _ensure_ascend_moe_customop_registered()

    backend = torch.distributed.get_backend(get_world_group().device_group)
    local_rank = get_world_group().local_rank
    _init_mc2_group_for_diffusion(
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
    _ensure_forward_context_attr("max_tokens_across_dp", int | None, None)
    _ensure_forward_context_attr("max_tokens_across_pcp", int | None, None)


def reset_fused_moe_forward_context() -> None:
    try:
        forward_context = _vllm_fc.get_forward_context()
    except AssertionError:
        return

    # MoE input token counts can change between transformer forwards, for
    # example when text tokens are present only in some forwards. Clear cached
    # DP/PCP padding bounds so the first MoE layer recomputes them.
    forward_context.max_tokens_across_dp = None
    forward_context.max_tokens_across_pcp = None


def _set_max_tokens(forward_context: Any, hidden_states: torch.Tensor) -> None:
    if (
        getattr(forward_context, "max_tokens_across_dp", None) is not None
        or getattr(forward_context, "max_tokens_across_pcp", None) is not None
    ):
        return

    num_tokens_before_pcp = hidden_states.shape[0]
    dp_metadata = getattr(forward_context, "dp_metadata", None)
    if dp_metadata is not None:
        max_tokens_across_dp = int(dp_metadata.num_tokens_across_dp_cpu.max().item())
        forward_context.max_tokens_across_dp = max_tokens_across_dp
        # DP prepare pads each rank to max_tokens_across_dp, then gathers
        # all DP ranks before the PCP gather.
        num_tokens_before_pcp = max_tokens_across_dp * get_dp_group().world_size

    pcp_group = getattr(vllm_parallel_state, "_PCP", None)
    if pcp_group is None or getattr(pcp_group, "world_size", 1) <= 1:
        return

    gathered_num_tokens: list[int | None] = [None] * pcp_group.world_size
    torch.distributed.all_gather_object(
        gathered_num_tokens,
        int(num_tokens_before_pcp),
        group=pcp_group.cpu_group,
    )
    if any(num_tokens is None for num_tokens in gathered_num_tokens):
        raise RuntimeError(f"Failed to gather MoE PCP token counts: {gathered_num_tokens}")
    forward_context.max_tokens_across_pcp = max(gathered_num_tokens)


def fused_moe_forward_context_pre_hook(
    module: Any,
    args: Any,
    kwargs: Any,
) -> None:
    forward_context = _vllm_fc.get_forward_context()

    hidden_states = kwargs.get("hidden_states")
    if hidden_states is None and args:
        hidden_states = args[0]
    if hidden_states is not None:
        _set_max_tokens(forward_context, hidden_states)

    forward_context.moe_comm_type = _select_moe_comm_method(vllm_config=omni_get_ctx().vllm_config)
    forward_context.moe_comm_method = _MoECommMethods.get(forward_context.moe_comm_type)
