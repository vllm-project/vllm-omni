import logging
import os
import sys
from functools import cached_property

import torch
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

_PATCH_LOGGER = logging.getLogger("vllm_omni.patch")

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
# Patch ModelOptNvFp4LinearMethod.process_weights_after_loading to clamp NaN
# bytes in per-block FP8 weight_scale tensors at load time.
# =============================================================================
# WHY: ModelOpt 0.44's float32 -> torch.float8_e4m3fn cast of per-block weight
# scales occasionally emits NaN bytes (E4M3 encoding 0x7F / 0xFF) when the
# pre-cast scale rounds above the FP8 E4M3 max of 448 after the global-scale
# division. Any single NaN byte in weight_scale propagates through the
# FlashInfer FP4 GEMM and the served model collapses to `!!!!`. NVFP4 W4A4
# Qwen3-Omni checkpoints published by the community — exported with stock
# ModelOpt 0.44 — currently fail this way the moment they're served.
#
# WHY here (vllm-omni, not upstream): the root cause sits in ModelOpt; the
# analogous fix in vLLM's own PWAL is also pending. Both are planned as
# follow-up PRs. Until those land in vllm-omni's pinned vLLM, this load-time
# clamp keeps the user-visible failure from looking like a vllm-omni
# inference bug. Newly calibrated, clean checkpoints pay no runtime cost
# (the clamp is a no-op when no NaN bytes are present).
#
# SCOPE: ModelOptNvFp4LinearMethod (W4A4 NVFP4 Linear) only. NvFp4FusedMoE /
# NvFp4W4A16 / CompressedTensors / Quark NVFP4 paths are not covered.
#
# SELF-EXTINGUISH: `_already_patched_upstream` heuristically detects when
# vLLM's own PWAL contains an in-place `masked_fill_` against `weight_scale`
# / `isnan` — the structure the upstream fix is expected to take when it is
# filed (planned as a follow-up PR after this one merges). Once vllm-omni's
# vllm pin moves to a release with that upstream fix, the override is
# skipped at import and this block can be deleted. NOTE: the heuristic only
# matches "vLLM PWAL clamps NaN in-place with `masked_fill_`"; if the
# upstream fix lands as `nan_to_num_` or as a clamp before the FP32→FP8
# cast, the check won't fire and this override stays active. The override
# is idempotent so the overlap is a warning log, not a correctness issue —
# but the heuristic should be revisited when the upstream PR is filed.
#
# ORDERING: the clamp must run BEFORE the original PWAL. The non-Blackwell
# Marlin fallback (sm_<100) casts weight_scale FP8 -> bf16/fp16 and permutes
# inside its kernel PWAL; an after-PWAL clamp would either trip the
# byte-view shape assertion (FP8 1B/elem -> bf16 2B/elem) or operate on
# already-transformed bytes where NaN has propagated through permute. On
# Blackwell (Cutlass/TRTLLM/cuDNN) the kernel PWAL only swizzles/shuffles
# FP8 bytes, so clamp-before is equivalent to clamp-after there.


def _clamp_nvfp4_weight_scale_nans(layer) -> int:
    """Scan ``layer.weight_scale`` for NaN bytes and clamp them to the FP8
    E4M3 max byte (0x7E) in place. Returns the number of bytes clamped.

    Scoped to NVFP4 ``float8_e4m3fn`` weight_scale only — a future MXFP4
    weight-scale clamp would need its own helper (MXFP4 uses
    ``float8_e8m0fnu`` for scales, and the byte encoding of NaN differs).

    Exposed at module scope so tests can exercise the clamp directly
    without monkey-patching + reloading the parent ``vllm_omni.patch``
    module.
    """
    # Defensive: a stray subclass without weight_scale would crash inside
    # the override; just bail out.
    weight_scale = getattr(layer, "weight_scale", None)
    if weight_scale is None:
        return 0
    # Re-entry safety: this clamp expects raw on-disk FP8 weight_scale (1
    # byte/elem). If we are ever invoked after the original PWAL ran (e.g.
    # a caller re-issues process_weights_after_loading on the same layer),
    # weight_scale may have been cast to bf16/fp16 by the Marlin path, in
    # which case the uint8 byte view below would have a different shape
    # than the NaN mask. Bail out cleanly rather than trip the assert.
    if weight_scale.dtype != torch.float8_e4m3fn:
        return 0
    # `.view(torch.uint8)` below requires a contiguous tensor and raises on a
    # non-contiguous one. A checkpoint whose per-block scale tensor isn't
    # contiguous would otherwise crash here. Materialize a contiguous copy and
    # write it back to the Parameter so the in-place byte clamp lands on the
    # tensor the kernel will actually read (a bare `.contiguous()` copy would
    # be discarded and the original left un-clamped).
    weight_scale_data = weight_scale.data
    if not weight_scale_data.is_contiguous():
        weight_scale_data = weight_scale_data.contiguous()
        weight_scale.data = weight_scale_data
    nan_mask = torch.isnan(weight_scale_data)
    if not nan_mask.any():
        return 0
    # PyTorch does not implement masked_fill_ for float8_e4m3fn, so view
    # the storage as uint8 and write the FP8 max byte (0x7E) directly.
    # float8_e4m3fn is 1 byte per element so the byte view is shape-
    # for-shape with the original tensor; pin that here so a future
    # packed scale format would fail loudly rather than silently corrupt
    # unrelated bytes.
    byte_view = weight_scale_data.view(torch.uint8)
    if byte_view.shape != nan_mask.shape:
        raise RuntimeError(
            f"NVFP4 weight_scale uint8 view shape {byte_view.shape} != NaN mask shape "
            f"{nan_mask.shape}; per-block scale layout changed — recheck the clamp."
        )
    fp8_max_byte = (
        torch.tensor(torch.finfo(torch.float8_e4m3fn).max, dtype=torch.float8_e4m3fn).view(torch.uint8).item()
    )
    byte_view.masked_fill_(nan_mask, fp8_max_byte)
    n = int(nan_mask.sum().item())
    _PATCH_LOGGER.warning("Clamped %d NaN entries in NVFP4 per-block weight_scale.", n)
    return n


# Module-level defaults so downstream code (and tests) can import these names
# without guarding for the import-failure / escape-hatch branches below.
# `_already_patched_upstream` = upstream PWAL contains its own NaN clamp.
# `_clamp_installed`         = our wrapper was installed on the upstream class.
# These are independent: the env-var escape hatch and the import-failure path
# both leave the wrapper uninstalled WITHOUT upstream being patched, so the
# right check for "we own NaN-clamp behavior" is `_clamp_installed`.
_already_patched_upstream = False
_clamp_installed = False

try:
    # Escape hatch — set VLLM_OMNI_SKIP_NVFP4_NAN_CLAMP=1 to skip installing
    # the patch (e.g. to confirm a `!!!!` failure is the NaN-byte case).
    # The escape-hatch deliberately raises ImportError so the not-installed
    # warning below logs through the same path a real ImportError would.
    # Use the repo-wide bool-env idiom so values like `0`, `false`, `no`,
    # `off` correctly mean "do not skip" rather than tripping naive
    # truthiness on the non-empty string.
    if os.environ.get("VLLM_OMNI_SKIP_NVFP4_NAN_CLAMP", "").lower() in ("1", "true", "yes", "on"):
        raise ImportError("VLLM_OMNI_SKIP_NVFP4_NAN_CLAMP is set; skipping NaN-clamp install")
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4LinearMethod as _OriginalModelOptNvFp4LinearMethod,
    )
except ImportError as _nan_clamp_import_err:
    _PATCH_LOGGER.warning(
        "NVFP4 weight_scale NaN-clamp patch could NOT install: %s. NVFP4 W4A4 "
        "checkpoints with NaN bytes in per-block weight_scale will serve `!!!!`.",
        _nan_clamp_import_err,
    )
else:
    _current_nvfp4_pwal = _OriginalModelOptNvFp4LinearMethod.process_weights_after_loading
    # Reload idempotency: on a module reload (importlib.reload in a test, or a
    # second import path) the class attribute already holds OUR wrapper, so
    # capturing it as the "original" and re-wrapping would nest the clamp and
    # run it twice per load. Recover the genuine upstream method from the
    # sentinel we stamp below; the heuristic + a fresh single-level wrapper are
    # then computed against the real upstream, not against our own wrapper.
    _original_nvfp4_pwal = getattr(_current_nvfp4_pwal, "_vllm_omni_wrapped_pwal", _current_nvfp4_pwal)
    _upstream_pwal_names = set(_original_nvfp4_pwal.__code__.co_names or ())
    # Require ALL three names — `weight_scale` alone is too loose because the
    # current upstream PWAL already references `weight_scale_2` (close prefix
    # collisions aside, future PWALs may legitimately reference `weight_scale`
    # without a NaN clamp). `masked_fill_` + `isnan` + `weight_scale` together
    # are the structural signature of an in-place NaN clamp on weight_scale.
    # KNOWN BLIND SPOTS (false-negatives, safe direction — we install a
    # redundant but idempotent clamp): upstream using
    # `getattr(layer, "weight_scale")` (string lands in co_consts, not
    # co_names), or factoring the clamp into a helper called from PWAL
    # (none of the three names appear in the top-level co_names). Revisit
    # this heuristic when the upstream PR is actually filed.
    _already_patched_upstream = all(n in _upstream_pwal_names for n in ("masked_fill_", "isnan", "weight_scale"))

    def _patched_nvfp4_pwal(self, layer, *args, **kwargs):
        # Clamp BEFORE the original PWAL — see ORDERING note above.
        _clamp_nvfp4_weight_scale_nans(layer)
        _original_nvfp4_pwal(self, layer, *args, **kwargs)

    # Sentinel so a later reload of this module recovers the genuine upstream
    # method above instead of wrapping this wrapper (see reload-idempotency note).
    _patched_nvfp4_pwal._vllm_omni_wrapped_pwal = _original_nvfp4_pwal

    if not _already_patched_upstream:
        _OriginalModelOptNvFp4LinearMethod.process_weights_after_loading = _patched_nvfp4_pwal
        # Fail loudly if install dropped silently (e.g. another plugin
        # patched the same class) rather than degrade to `!!!!` at decode.
        # Use raise, not assert: asserts are compiled out under `python -O` /
        # PYTHONOPTIMIZE, which would silently disable this guard in exactly
        # the optimized runs where a conflicting plugin must still be caught.
        if _OriginalModelOptNvFp4LinearMethod.process_weights_after_loading is not _patched_nvfp4_pwal:
            raise RuntimeError("NVFP4 weight_scale NaN-clamp install failed — check for conflicting plugins.")
        _clamp_installed = True

    _PATCH_LOGGER.info(
        "NVFP4 W4A4 weight_scale NaN-clamp: %s.",
        "skipped (upstream already patched)" if _already_patched_upstream else "installed",
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
            CHAT_TEMPLATES_DIR,
        )

        if "qwen3_omni_moe" not in _MODEL_TYPE_TO_CHAT_TEMPLATE_FALLBACK:
            _MODEL_TYPE_TO_CHAT_TEMPLATE_FALLBACK["qwen3_omni_moe"] = (
                lambda _: CHAT_TEMPLATES_DIR / "template_chatml.jinja"
            )
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
