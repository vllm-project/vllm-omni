# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import vllm.forward_context as _vllm_fc
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.utils.import_utils import resolve_obj_by_qualname

from vllm_omni.platforms import current_omni_platform


def _set_forward_context_num_tokens(num_tokens: int) -> None:
    """Set num_tokens on the vLLM ForwardContext for MoE routing.

    After the rebase to vLLM 0.18.0, FusedMoE expects
    ForwardContext.num_tokens to be set. Without it, MoE expert
    routing may produce incorrect results (silent correctness bug).
    """
    if not _vllm_fc.is_forward_context_available():
        return
    forward_context = _vllm_fc.get_forward_context()
    forward_context.num_tokens = num_tokens
    if not hasattr(forward_context, "in_profile_run"):
        forward_context.in_profile_run = False


class HunyuanFusedMoEDefault(FusedMoE):
    def __init__(self, *, prefix: str = "", **kwargs: Any) -> None:
        super().__init__(prefix=prefix, **kwargs)
        self._prefix = prefix
        self._init_hook_handle = self.register_forward_pre_hook(self._initialize_kernel_hook, with_kwargs=True)

    def _initialize_kernel_hook(self, module: Any, args: Any, kwargs: Any) -> None:
        if self.quant_method:
            self.quant_method.process_weights_after_loading(self)
        self._init_hook_handle.remove()

    def forward(self, hidden_states: Any, router_logits: Any) -> Any:
        _set_forward_context_num_tokens(hidden_states.shape[0])
        return super().forward(hidden_states, router_logits)


class HunyuanFusedMoE:
    def __new__(cls, *, prefix: str = "", **kwargs: Any) -> Any:
        op_name = "hunyuan_fused_moe"
        current_omni_platform.prepare_diffusion_op_runtime(op_name)
        impl = resolve_obj_by_qualname(
            current_omni_platform.get_diffusion_model_impl_qualname(op_name),
        )
        return impl(prefix=prefix, **kwargs)

    @classmethod
    def make_expert_params_mapping(
        cls,
        model: Any,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        num_experts: int,
        num_redundant_experts: int = 0,
    ) -> list[tuple[str, str, int, str]]:
        impl = resolve_obj_by_qualname(
            current_omni_platform.get_diffusion_model_impl_qualname("hunyuan_fused_moe"),
        )
        return impl.make_expert_params_mapping(
            model,
            ckpt_gate_proj_name=ckpt_gate_proj_name,
            ckpt_down_proj_name=ckpt_down_proj_name,
            ckpt_up_proj_name=ckpt_up_proj_name,
            num_experts=num_experts,
            num_redundant_experts=num_redundant_experts,
        )
