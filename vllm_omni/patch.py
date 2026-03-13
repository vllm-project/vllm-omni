import sys
from functools import cached_property as _cached_property

from aenum import extend_enum
from vllm.config import ModelConfig as _ModelConfig
from vllm.v1.engine import EngineCoreRequestType as _ECRType

# Add UPDATE request type for streaming additional_information updates.
# This follows the same fire-and-forget pattern as ABORT.
if not hasattr(_ECRType, "UPDATE"):
    extend_enum(_ECRType, "UPDATE", b"\x05")

from vllm.inputs.data import TokensPrompt as _OriginalTokensPrompt
from vllm.model_executor.layers.rotary_embedding import (
    MRotaryEmbedding as _OriginalMRotaryEmbedding,
)
from vllm.v1.engine import EngineCoreOutput as _OriginalEngineCoreOutput
from vllm.v1.engine import EngineCoreOutputs as _OriginalEngineCoreOutputs
from vllm.v1.engine import EngineCoreRequest as _OriginalEngineCoreRequest
from vllm.v1.request import Request as _OriginalRequest
from vllm.v1.request import RequestStatus

import vllm_omni.logger  # noqa: F401
from vllm_omni.engine import OmniEngineCoreOutput, OmniEngineCoreOutputs, OmniEngineCoreRequest
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.layers.rotary_embedding import OmniMRotaryEmbedding
from vllm_omni.request import OmniRequest

# =============================================================================
# Patch GlmImageTextConfig to expose mrope_section in rope_parameters
# =============================================================================
# GLM-Image uses M-RoPE with mrope_section: [8, 12, 12], but transformers'
# implementation doesn't expose it in rope_parameters. vLLM's uses_mrope
# detection relies on "mrope_section" being present in rope_parameters.
# This patch ensures proper M-RoPE detection for GLM-Image.
try:
    from transformers.models.glm_image.configuration_glm_image import GlmImageTextConfig

    _original_glm_image_text_config_init = GlmImageTextConfig.__init__

    def _patched_glm_image_text_config_init(self, *args, **kwargs):
        _original_glm_image_text_config_init(self, *args, **kwargs)
        # Ensure rope_parameters exists and contains mrope_section
        if self.rope_parameters is None:
            self.rope_parameters = {}
        if isinstance(self.rope_parameters, dict) and "mrope_section" not in self.rope_parameters:
            # GLM-Image uses mrope_section: [8, 12, 12] for T/H/W dimensions
            self.rope_parameters["mrope_section"] = [8, 12, 12]

    GlmImageTextConfig.__init__ = _patched_glm_image_text_config_init
except ImportError:
    # GlmImageTextConfig not available, skip patching
    pass


# Patch ModelConfig.is_mm_prefix_lm to include Bagel (bidirectional attention
# for multimodal prefix positions, same as Gemma3/Molmo2/PaliGemma).
_orig_is_mm_prefix_lm = _ModelConfig.__dict__["is_mm_prefix_lm"].func


@_cached_property
def _patched_is_mm_prefix_lm(self) -> bool:
    return _orig_is_mm_prefix_lm(self) or getattr(self.hf_config, "model_type", None) == "bagel"


_patched_is_mm_prefix_lm.__set_name__(_ModelConfig, "is_mm_prefix_lm")
_ModelConfig.is_mm_prefix_lm = _patched_is_mm_prefix_lm

# Extend RequestStatus enum with omni-specific statuses
if not hasattr(RequestStatus, "WAITING_FOR_CHUNK"):
    # The value - 1 is intentionally chosen to ensure it is treated
    # as a non-finished state and remains compatible with existing comparisons.
    extend_enum(RequestStatus, "WAITING_FOR_CHUNK", -1)


# ---------------------------------------------------------------------------
# Patch EngineCore._handle_client_request to support UPDATE messages.
# UPDATE carries (request_id, update_dict) and routes to the scheduler's
# update_request_additional_info() method.  Fire-and-forget, same as ABORT.
# ---------------------------------------------------------------------------
from vllm.v1.engine.core import EngineCoreProc as _EngineCoreProc

_original_handle_client_request = _EngineCoreProc._handle_client_request


def _patched_handle_client_request(self, request_type, request):
    if request_type == _ECRType.UPDATE:
        try:
            req_id, update_dict = request
        except (TypeError, ValueError):
            return
        if hasattr(self.scheduler, "update_request_additional_info"):
            self.scheduler.update_request_additional_info(req_id, update_dict)
        return
    return _original_handle_client_request(self, request_type, request)


_EngineCoreProc._handle_client_request = _patched_handle_client_request

# Also patch DPEngineCoreProc if it exists
try:
    from vllm.v1.engine.core import DPEngineCoreProc as _DPEngineCoreProc
    _DPEngineCoreProc._handle_client_request = _patched_handle_client_request
except ImportError:
    pass

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
        module.MRotaryEmbedding = OmniMRotaryEmbedding
    if hasattr(module, "Request") and module.Request == _OriginalRequest:
        module.Request = OmniRequest
    if hasattr(module, "EngineCoreRequest") and module.EngineCoreRequest == _OriginalEngineCoreRequest:
        module.EngineCoreRequest = OmniEngineCoreRequest
