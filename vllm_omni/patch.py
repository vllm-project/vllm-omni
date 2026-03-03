import sys

from aenum import extend_enum
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

# Extend RequestStatus enum with omni-specific statuses
if not hasattr(RequestStatus, "WAITING_FOR_CHUNK"):
    # The value - 1 is intentionally chosen to ensure it is treated
    # as a non-finished state and remains compatible with existing comparisons.
    extend_enum(RequestStatus, "WAITING_FOR_CHUNK", -1)

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
