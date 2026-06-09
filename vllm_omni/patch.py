import logging
import sys
from functools import cached_property

from aenum import extend_enum
from vllm.config import ModelConfig as _OriginalModelConfig
from vllm.inputs import TokensPrompt as _OriginalTokensPrompt
from vllm.model_executor.layers.rotary_embedding import (
    MRotaryEmbedding as _OriginalMRotaryEmbedding,
)
from vllm.v1.engine import EngineCoreOutput as _OriginalEngineCoreOutput
from vllm.v1.engine import EngineCoreOutputs as _OriginalEngineCoreOutputs
from vllm.v1.engine import EngineCoreRequest as _OriginalEngineCoreRequest
from vllm.v1.request import Request as _OriginalRequest
from vllm.v1.request import RequestStatus
from vllm.v1.request import StreamingUpdate as _OriginalStreamingUpdate

import vllm_omni.logger  # noqa: F401
from vllm_omni.engine import OmniEngineCoreOutput, OmniEngineCoreOutputs, OmniEngineCoreRequest
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.layers.rotary_embedding import OmniMRotaryEmbedding
from vllm_omni.request import OmniRequest, OmniStreamingUpdate

# =============================================================================
# Patch ModelConfig.is_mm_prefix_lm to support omni-specific models
# =============================================================================
# WHY: HunyuanImage-3.0 requires bidirectional attention for image tokens
# (cond_token_attn_type: "joint_full" in config.json). vLLM gates this on
# is_mm_prefix_lm, which checks an internal MM_PREFIX_LM_MODELS list that
# does not include "hunyuan_image_3_moe" (the upstream HF model_type).
#
# WHY NOT model-level: is_mm_prefix_lm is checked in vLLM core (scheduler,
# attention backend selection) before model code runs — no model-level hook.
#
# SCOPE: Only affects model_type in _OMNI_MM_PREFIX_LM_MODELS (currently
# just "hunyuan_image_3_moe"). All other models fall through to the
# original vLLM implementation unchanged.
#
# FRAGILITY: Relies on is_mm_prefix_lm being a cached_property on
# ModelConfig. The __dict__ access + __set_name__ dance works around a
# pydantic dataclass issue in vllm 0.19.0+. If vLLM changes
# is_mm_prefix_lm to a regular method or removes it, this will break.
#
# TODO: Upstream a configurable MM_PREFIX_LM_MODELS or a model_config flag
# so this patch can be removed.
_OMNI_MM_PREFIX_LM_MODELS = ("hunyuan_image_3_moe",)
# Access via __dict__ to avoid triggering cached_property.__get__ which fails
# with "Cannot use cached_property instance without calling __set_name__" in
# pydantic dataclasses (vllm 0.19.0+).
_cp = _OriginalModelConfig.__dict__["is_mm_prefix_lm"]
_original_is_mm_prefix_lm = _cp.func if hasattr(_cp, "func") else _cp.fget


def _patched_is_mm_prefix_lm(self):
    if _original_is_mm_prefix_lm(self):
        return True
    model_type = getattr(self.hf_config, "model_type", "")
    return model_type in _OMNI_MM_PREFIX_LM_MODELS


_patched_cp = cached_property(_patched_is_mm_prefix_lm)
_patched_cp.__set_name__(_OriginalModelConfig, "is_mm_prefix_lm")
_OriginalModelConfig.is_mm_prefix_lm = _patched_cp

# Sanity check: verify the patch is active. If vLLM changes the descriptor
# type or __set_name__ semantics, this will fail loudly at import time
# rather than silently falling back to unpatched behavior.
_installed = _OriginalModelConfig.__dict__.get("is_mm_prefix_lm")
assert _installed is _patched_cp, (
    "is_mm_prefix_lm patch failed to install — bidirectional attention "
    "for HunyuanImage3 will not work. Check vLLM ModelConfig changes."
)

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

if not hasattr(RequestStatus, "WAITING_FOR_INPUT"):
    # Full-payload stage handoff uses a distinct waiting state so the
    # scheduler can restore the request once non-stage-0 inputs arrive.
    extend_enum(RequestStatus, "WAITING_FOR_INPUT", -2)

# Snapshot sys.modules: `hasattr` below can trigger lazy submodule imports
# (e.g. transformers' `_LazyModule.__getattr__`), which mutate sys.modules
# during iteration and raise `dictionary changed size during iteration`.
for module_name, module in list(sys.modules.items()):
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
    if hasattr(module, "StreamingUpdate") and module.StreamingUpdate == _OriginalStreamingUpdate:
        module.StreamingUpdate = OmniStreamingUpdate
    if hasattr(module, "EngineCoreRequest") and module.EngineCoreRequest == _OriginalEngineCoreRequest:
        module.EngineCoreRequest = OmniEngineCoreRequest

# =============================================================================
# Patch unregister_vllm_metrics to skip vllm:omni_* collectors
# =============================================================================
# WHY: Upstream unregister_vllm_metrics() uses `"vllm" in collector._name` to
# strip collectors before each new PrometheusStatLogger registers, which is
# how it avoids "Duplicated timeseries" when the same process spawns multiple
# engines / orchestrators.  But its substring match also wipes our vllm:omni_*
# families, so we must replace it with a scoped version that keeps ours.
#
# REMOVAL: Remove this patch once upstream vLLM adds
# _STAT_LOGGER_METRIC_NAMES to vllm.v1.metrics.prometheus and scopes
# unregister_vllm_metrics() to that set.  Track:
# https://github.com/vllm-project/vllm/pull/42331
import vllm.v1.metrics.prometheus as _vllm_prometheus  # noqa: E402

_logger = logging.getLogger(__name__)


def _scoped_unregister_vllm_metrics():
    """Drop upstream vllm:* collectors but preserve vllm:omni_* (ours)."""
    from prometheus_client import REGISTRY

    for collector in list(REGISTRY._collector_to_names):
        name = getattr(collector, "_name", "")
        if "vllm" not in name:
            continue
        if name.startswith("vllm:omni_") or name.startswith("vllm_omni"):
            continue
        REGISTRY.unregister(collector)


_vllm_prometheus.unregister_vllm_metrics = _scoped_unregister_vllm_metrics
_logger.warning(
    "Monkey-patched unregister_vllm_metrics() to scope drops to non-omni vllm:* collectors. "
    "Remove this patch once vLLM adds _STAT_LOGGER_METRIC_NAMES."
)


# Patch: add qwen3_omni_moe to vllm's chat template fallback registry.
# Qwen/Qwen3-Omni-30B-A3B-Instruct stores its chat_template in a standalone
# chat_template.json (not in tokenizer_config.json).  transformers < 5.9.0
# does not load this file, so the tokenizer has no chat_template attribute.
# vllm's resolve_chat_template falls back to MODEL_TYPE_TO_CHAT_TEMPLATE
# which has "qwen" but not "qwen3_omni_moe".  Register the same fallback.
def _patch_chat_template_registry():
    try:
        from vllm.transformers_utils.chat_templates.registry import (
            _MODEL_TYPE_TO_CHAT_TEMPLATE_FALLBACK,
            _get_qwen_chat_template_fallback,
        )

        if "qwen3_omni_moe" not in _MODEL_TYPE_TO_CHAT_TEMPLATE_FALLBACK:
            _MODEL_TYPE_TO_CHAT_TEMPLATE_FALLBACK["qwen3_omni_moe"] = _get_qwen_chat_template_fallback
    except ImportError:
        pass


_patch_chat_template_registry()


def _patch_scaled_mm_fp8_contiguous_activation():
    """Support batched diffusion activations on the ModelOpt FP8 (ScaledMM) path.

    The FP8 ScaledMM linear flattens its activation with ``x.view(-1, ...)``, which
    needs a contiguous tensor. Under step-execution batching (``--max-num-seqs > 1``)
    the sequence-packed diffusion activations can be non-contiguous, so we make the
    activation contiguous before the GEMM (no-op when it already is). Mixed FP8/NVFP4
    routes through the CUTLASS NVFP4 path and is unaffected.
    """
    try:
        from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
            ScaledMMLinearKernel,
        )
    except ImportError:
        return

    _original_apply_weights = ScaledMMLinearKernel.apply_weights

    def _patched_apply_weights(self, layer, x, bias=None):
        if not x.is_contiguous():
            x = x.contiguous()
        return _original_apply_weights(self, layer, x, bias)

    ScaledMMLinearKernel.apply_weights = _patched_apply_weights


_patch_scaled_mm_fp8_contiguous_activation()


def _patch_flashinfer_fp8_scaled_mm_output_shape():
    """Restore the N-D output shape for the FlashInfer FP8 ScaledMM kernel.

    ``FlashInferFP8ScaledMMLinearKernel.apply_scaled_mm`` returns the raw 2-D
    GEMM result and ignores ``output_shape``, unlike the CUTLASS / PyTorch
    ScaledMM kernels which reshape to it. A 3-D activation ``(B, S, D)`` thus
    collapses to ``(B*S, D)``, breaking diffusion DiTs that reshape the linear
    output by absolute dim (e.g. Wan2.2 ``qkv.unflatten(2, ...)``) with
    ``IndexError: Dimension out of range``. It only bites >2-D inputs, so LLM
    (token-flattened, 2-D) paths are unaffected.

    Carried here because the upstream fix may not have landed yet; this override
    becomes a harmless no-op once vLLM honors ``output_shape`` itself.
    """
    try:
        from vllm.model_executor.kernels.linear.scaled_mm.flashinfer import (
            FlashInferFP8ScaledMMLinearKernel,
        )
    except ImportError:
        return

    _original_apply_scaled_mm = FlashInferFP8ScaledMMLinearKernel.apply_scaled_mm
    if getattr(_original_apply_scaled_mm, "_omni_output_shape_patched", False):
        return

    def _patched_apply_scaled_mm(self, *, A, B, out_dtype, As, Bs, bias, output_shape):  # noqa: N803
        out = _original_apply_scaled_mm(
            self, A=A, B=B, out_dtype=out_dtype, As=As, Bs=Bs, bias=bias, output_shape=output_shape
        )
        if tuple(out.shape) != tuple(output_shape):
            out = out.view(*output_shape)
        return out

    _patched_apply_scaled_mm._omni_output_shape_patched = True
    FlashInferFP8ScaledMMLinearKernel.apply_scaled_mm = _patched_apply_scaled_mm


_patch_flashinfer_fp8_scaled_mm_output_shape()


def _patch_fp8_use_quack_fused_bias():
    try:
        from vllm_omni.quantization.quack_fp8 import install_quack_fp8_patch

        install_quack_fp8_patch()
    except Exception:  # noqa: BLE001
        pass


_patch_fp8_use_quack_fused_bias()
