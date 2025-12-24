import sys

from vllm.inputs.data import TokensPrompt as _OriginalTokensPrompt
from vllm.model_executor.layers.rotary_embedding import (
    MRotaryEmbedding as _OriginalMRotaryEmbedding,
)
from vllm.v1.engine import EngineCoreOutput as _OriginalEngineCoreOutput
from vllm.v1.engine import EngineCoreOutputs as _OriginalEngineCoreOutputs
from vllm.v1.engine import EngineCoreRequest as _OriginalEngineCoreRequest
from vllm.v1.request import Request as _OriginalRequest

import vllm_omni.logger  # noqa: F401
from vllm_omni.engine import OmniEngineCoreOutput, OmniEngineCoreOutputs, OmniEngineCoreRequest
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.layers.mrope import MRotaryEmbedding
from vllm_omni.request import OmniRequest


# Patch for vllm-ascend prefetch functions bug fix
# Issue: The original functions access forward_context attributes like
# prefetch_mlp_gate_up_proj, prefetch_mlp_down_proj, layer_idx without checking
# if they exist, which causes AttributeError when prefetch_mlp_enabled is not set.
# TODO: Remove this patch after upgrading to vllm-ascend v0.13.0 or later.
# This issue has been fixed in https://github.com/vllm-project/vllm-ascend/pull/5035
def _patch_vllm_ascend_prefetch():
    try:
        import vllm_ascend.ops.register_custom_ops as register_custom_ops
        import vllm_ascend.envs as envs_ascend
        from vllm.forward_context import get_forward_context
        import torch
        import torch_npu
    except ImportError:
        return

    def _maybe_prefetch_mlp_gate_up_proj_impl_patched(
            x_dependency: torch.Tensor, prefix: str) -> None:
        try:
            forward_context = get_forward_context()
        except AssertionError:
            return

        if not getattr(forward_context, 'prefetch_mlp_enabled', False):
            return
        model_instance = forward_context.model_instance
        prefetch_stream = forward_context.prefetch_stream
        layer_idx = int(prefix.split('.')[2])

        # start point of gate_up_proj weight prefetch
        if prefix.split('.')[-2] == "self_attn":
            forward_context.prefetch_mlp_gate_up_proj = True
        if getattr(forward_context, 'prefetch_mlp_gate_up_proj', False):
            prefetch_stream.wait_stream(torch.npu.current_stream())

            with torch.npu.stream(prefetch_stream):
                mlp_gate_up_prefetch_size = envs_ascend.VLLM_ASCEND_MLP_GATE_UP_PREFETCH_SIZE
                torch_npu.npu_prefetch(
                    model_instance.model.layers[layer_idx].mlp.gate_up_proj.weight,
                    x_dependency, mlp_gate_up_prefetch_size)
        return

    def _maybe_prefetch_mlp_down_proj_impl_patched(
            x_dependency: torch.Tensor) -> None:
        try:
            forward_context = get_forward_context()
        except AssertionError:
            return

        if not getattr(forward_context, 'prefetch_mlp_enabled', False):
            return
        forward_context.prefetch_mlp_down_proj = True
        model_instance = forward_context.model_instance
        prefetch_stream = forward_context.prefetch_stream
        layer_idx = getattr(forward_context, 'layer_idx', None)
        if layer_idx is None:
            return

        # start point of down_proj weight prefetch
        prefetch_stream.wait_stream(torch.npu.current_stream())

        with torch.npu.stream(prefetch_stream):
            mlp_down_prefetch_size = envs_ascend.VLLM_ASCEND_MLP_DOWN_PREFETCH_SIZE
            torch_npu.npu_prefetch(
                model_instance.model.layers[layer_idx].mlp.down_proj.weight,
                x_dependency, mlp_down_prefetch_size)
        forward_context.layer_idx += 1
        return

    def _maybe_wait_prefetch_done_impl_patched(x: torch.Tensor) -> None:
        try:
            forward_context = get_forward_context()
        except AssertionError:
            return

        if not getattr(forward_context, 'prefetch_mlp_enabled', False):
            return
        if getattr(forward_context, 'prefetch_mlp_gate_up_proj', False) or \
                getattr(forward_context, 'prefetch_mlp_down_proj', False):
            prefetch_stream = forward_context.prefetch_stream
            # wait until prefetch done
            torch.npu.current_stream().wait_stream(prefetch_stream)
            forward_context.prefetch_mlp_gate_up_proj = False
            forward_context.prefetch_mlp_down_proj = False
        return

    # Apply the patches
    register_custom_ops._maybe_prefetch_mlp_gate_up_proj_impl = \
        _maybe_prefetch_mlp_gate_up_proj_impl_patched
    register_custom_ops._maybe_prefetch_mlp_down_proj_impl = \
        _maybe_prefetch_mlp_down_proj_impl_patched
    register_custom_ops._maybe_wait_prefetch_done_impl = \
        _maybe_wait_prefetch_done_impl_patched


_patch_vllm_ascend_prefetch()


# Patch for vllm-ascend AscendMRotaryEmbedding to use vllm-omni's MRotaryEmbedding
# Issue: AscendMRotaryEmbedding inherits from vllm's original MRotaryEmbedding,
# but vllm-omni replaces MRotaryEmbedding in modules. This causes the inheritance
# chain to break when AscendMRotaryEmbedding is instantiated.
def _patch_vllm_ascend_mrope():
    try:
        import vllm_ascend.ops.rotary_embedding as ascend_rotary
        from vllm_ascend.ops.rotary_embedding import AscendMRotaryEmbedding
    except ImportError:
        return

    # Update AscendMRotaryEmbedding's base class to use vllm-omni's MRotaryEmbedding
    AscendMRotaryEmbedding.__bases__ = (MRotaryEmbedding,)

    # Also update the module reference
    ascend_rotary.MRotaryEmbedding = MRotaryEmbedding


_patch_vllm_ascend_mrope()

for module_name, module in sys.modules.items():
    # only do patch on module of vllm, pass others
    if "vllm" not in module_name:
        continue
    if hasattr(module, "EngineCoreOutput") and module.EngineCoreOutput == _OriginalEngineCoreOutput:
        module.EngineCoreOutput = OmniEngineCoreOutput
    if hasattr(module, "EngineCoreOutputs") and module.EngineCoreOutputs == _OriginalEngineCoreOutputs:
        module.EngineCoreOutputs = OmniEngineCoreOutputs
    if hasattr(module, "TokensPrompt") and module.TokensPrompt == _OriginalTokensPrompt:
        module.TokensPrompt = OmniTokensPrompt
    if hasattr(module, "MRotaryEmbedding") and module.MRotaryEmbedding == _OriginalMRotaryEmbedding:
        module.MRotaryEmbedding = MRotaryEmbedding
    if hasattr(module, "Request") and module.Request == _OriginalRequest:
        module.Request = OmniRequest
    if hasattr(module, "EngineCoreRequest") and module.EngineCoreRequest == _OriginalEngineCoreRequest:
        module.EngineCoreRequest = OmniEngineCoreRequest
