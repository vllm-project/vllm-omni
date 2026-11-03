# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import vllm.distributed.parallel_state as _vllm_ps
import vllm.forward_context as _vllm_fc

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


def _set_forward_context_dp_metadata(num_tokens: int) -> None:
    """Populate vLLM MoE DP token metadata for diffusion-only DP mappings."""
    if not _vllm_fc.is_forward_context_available():
        return
    forward_context = _vllm_fc.get_forward_context()
    if getattr(forward_context, "dp_metadata", None) is not None:
        return

    dp_group = getattr(_vllm_ps, "_DP", None)
    if dp_group is None or getattr(dp_group, "world_size", 1) <= 1:
        return

    gathered_num_tokens: list[int | None] = [None] * dp_group.world_size
    torch.distributed.all_gather_object(
        gathered_num_tokens,
        int(num_tokens),
        group=dp_group.cpu_group,
    )
    if any(count is None for count in gathered_num_tokens):
        raise RuntimeError(f"Failed to gather MoE DP token counts: {gathered_num_tokens}")

    forward_context.dp_metadata = _vllm_fc.DPMetadata(torch.tensor(gathered_num_tokens, dtype=torch.int64))


class FusedMoE:
    """Adapter that configures the upstream FusedMoE ``MoERunner`` for diffusion models.

    Upstream commit dc68bd8c41 refactored FusedMoE from a class (``nn.Module``)
    into a factory function that returns a ``MoERunner`` instance, whose expert
    weights live in a ``routed_experts`` submodule
    (``...experts.routed_experts.w13_weight`` / ``...w2_weight``).

    This adapter builds that runner, installs the diffusion forward-context
    setup and additional platform hooks, and returns the runner *directly* from
    ``__new__`` so the parent MoE block registers it as a real ``nn.Module``
    submodule.

    Returning the runner (rather than wrapping it in a plain object that holds it
    in an attribute) is required for correctness: a non-Module wrapper hides the
    runner's parameters from ``named_parameters()``, so ``load_weights`` cannot
    find ``...experts.routed_experts.w13_weight`` and raises ``KeyError`` during
    weight loading.
    """

    def __new__(cls, *, prefix: str = "", **kwargs: Any) -> Any:
        current_omni_platform.prepare_diffusion_op_runtime("fused_moe")

        # Current vLLM FusedMoE handles output reduction internally.
        kwargs.pop("reduce_results", None)
        # FusedMoE is now a factory function; call it to get a MoERunner.
        from vllm.model_executor.layers.fused_moe import FusedMoEFactory as _FusedMoE

        moe_runner = _FusedMoE(prefix=prefix, **kwargs)

        # Set ForwardContext.num_tokens before each forward. After the rebase to
        # vLLM 0.18.0 FusedMoE requires this; without it MoE routing is silently
        # incorrect. Previously done in the wrapper's forward(); now a pre-hook
        # on the runner so we can return the runner directly.
        def _num_tokens_pre_hook(module: Any, args: Any, kwargs: Any) -> None:
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and args:
                hidden_states = args[0]
            if hidden_states is not None:
                _set_forward_context_num_tokens(hidden_states.shape[0])
                _set_forward_context_dp_metadata(hidden_states.shape[0])

        moe_runner.register_forward_pre_hook(_num_tokens_pre_hook, with_kwargs=True)
        current_omni_platform.register_additional_diffusion_fused_moe_hooks(moe_runner)

        return moe_runner

    @staticmethod
    def make_expert_params_mapping(
        model: Any,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        num_experts: int,
        num_redundant_experts: int = 0,
    ) -> list[tuple[str, str, int, str]]:
        """Delegate to the upstream standalone function.

        Upstream vLLM refactored ``FusedMoE`` from a class (which had
        ``make_expert_params_mapping`` as a classmethod) to a factory
        function.  The method was moved to a standalone function
        ``fused_moe_make_expert_params_mapping`` in
        ``vllm.model_executor.layers.fused_moe``.
        """
        from vllm.model_executor.layers.fused_moe import (
            fused_moe_make_expert_params_mapping,
        )

        return fused_moe_make_expert_params_mapping(
            model,
            ckpt_gate_proj_name=ckpt_gate_proj_name,
            ckpt_down_proj_name=ckpt_down_proj_name,
            ckpt_up_proj_name=ckpt_up_proj_name,
            num_experts=num_experts,
            num_redundant_experts=num_redundant_experts,
        )
