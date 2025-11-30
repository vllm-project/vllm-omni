import sys

from vllm.inputs.data import TokensPrompt as _OriginalTokensPrompt
from vllm.model_executor.layers.rotary_embedding import (
    MRotaryEmbedding as _OriginalMRotaryEmbedding,
)
from vllm.v1.engine import EngineCoreOutput as _OriginalEngineCoreOutput
from vllm.v1.engine import EngineCoreOutputs as _OriginalEngineCoreOutputs
from vllm.v1.engine import EngineCoreRequest as _OriginalEngineCoreRequest
from vllm.v1.request import Request as _OriginalRequest

from vllm_omni.engine import OmniEngineCoreOutput, OmniEngineCoreOutputs, OmniEngineCoreRequest
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.layers.mrope import MRotaryEmbedding
from vllm_omni.request import OmniRequest


# This must be done early to ensure it's applied before Scheduler class imports check_stop
def _patch_check_stop():
    """Patch check_stop function to handle empty output_token_ids safely.

    This function replaces vLLM's check_stop with a safe version that checks
    if output_token_ids is empty before accessing [-1]. This prevents IndexError
    during prefill phase or when output_token_ids is empty for any reason.


    The patch is applied at module import time to ensure it takes effect before
    any scheduler classes import and bind check_stop to a local variable.
    """
    try:
        import vllm.v1.core.sched.utils as sched_utils

        original_check_stop = sched_utils.check_stop

        def safe_check_stop(request, max_model_len, pooler_output=None):
            """Safe version of check_stop that handles empty output_token_ids.

            Args:
                request: Request object containing output_token_ids
                max_model_len: Maximum model length
                pooler_output: Optional pooler output tensor

            Returns:
                bool: True if request should stop, False otherwise

            Note:
                If output_token_ids is empty (e.g., during prefill phase),
                we return False (don't stop) since there are no tokens to check.
                This prevents IndexError when accessing output_token_ids[-1].
            """
            if not request.output_token_ids:
                return False

            return original_check_stop(request, max_model_len, pooler_output)

        # Replace check_stop in the utils module
        sched_utils.check_stop = safe_check_stop

        # Also patch in scheduler module if it's already imported
        if "vllm.v1.core.sched.scheduler" in sys.modules:
            sched_module = sys.modules["vllm.v1.core.sched.scheduler"]
            if hasattr(sched_module, "check_stop"):
                sched_module.check_stop = safe_check_stop
    except (ImportError, AttributeError):
        # If vllm is not available or check_stop doesn't exist, skip patching
        pass


# Apply check_stop patch early
_patch_check_stop()


for module_name, module in sys.modules.items():
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
