# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A16 NVFP4 config for ``nvfp4_blockwise_mixed_v1`` checkpoints.

The checkpoint is weight-only NVFP4 (packed FP4 weights + FP8 block
scales + FP32 global scale, **no** activation ``input_scale``) and quantizes
only selected MLP projections. Serving it FP4-resident therefore
requires:

1. the **W4A16** ModelOpt path (``ModelOptNvFp4W4A16LinearMethod``, pinned to the
   Marlin FP4 kernel) rather than the default W4A4 path, which would demand an
   ``input_scale`` the artifact lacks and would quantize activations; and
2. **target-inclusion** layer selection: vLLM's ``ModelOptNvFp4Config`` is
   exclusion-based ("quantize every Linear except ``exclude_modules``"), which
   for a selective checkpoint would need every non-target enumerated. We invert
   it: quantize **only** ``...mlp`` / ``...mlp_moe_gen``
   ``.{gate,up,down}_proj``, leaving attention, embeddings, ``lm_head``,
   norms, projections, audio and action modules BF16
   (``UnquantizedLinearMethod``).

The transformer's own ``config.json`` carries ``quant_recipe`` at top level (not
in a nested ``quantization_config`` block), so model construction resolves
this config via :func:`maybe_build_nvfp4_blockwise_config` at construction time.
"""

import re

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

logger = init_logger(__name__)

RECIPE = "nvfp4_blockwise_mixed_v1"
# Target modules end in ``.mlp`` / ``.mlp_moe_gen``
# ``.{gate,up,down}_proj``. Attention uses
# ``to_q``/``to_k``/``to_v``/``to_out``/``add_*_proj``; small projections,
# embeddings, and heads stay unquantized.
_TARGET_RE = re.compile(r"\.(mlp|mlp_moe_gen)\.(gate_proj|up_proj|down_proj)$")


def is_target_prefix(prefix: str) -> bool:
    """True iff *prefix* is one of the quantized MLP projections (pure)."""
    return bool(_TARGET_RE.search(prefix))


def _load_base_config_cls():
    """Import the vLLM ModelOpt NVFP4 config lazily (only when serving NVFP4)."""
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config

    return ModelOptNvFp4Config


def build_nvfp4_blockwise_w4a16_config() -> QuantizationConfig:
    """Build the W4A16 target-inclusion NVFP4 config for the blockwise recipe."""
    base_cls = _load_base_config_cls()

    class Nvfp4BlockwiseW4A16Config(base_cls):
        """W4A16 NVFP4 with target-inclusion selection (only MLP projections).

        ``get_name()`` stays ``"modelopt_fp4"`` (inherited) so downstream
        ModelOpt-NVFP4 handling keeps working; only layer selection is inverted.
        """

        def is_target_module(self, prefix: str) -> bool:
            return is_target_prefix(prefix)

        def is_layer_excluded(self, prefix: str, *args, **kwargs) -> bool:
            # Invert vLLM's exclusion default: quantize ONLY the MLP targets;
            # every other Linear is excluded -> UnquantizedLinearMethod (BF16).
            # *args/**kwargs tolerate the base signature (some vLLM builds pass
            # exclude_modules as a 2nd positional) so selection can't silently
            # mismatch across versions.
            return not self.is_target_module(prefix)

    cfg = Nvfp4BlockwiseW4A16Config(
        quant_method="W4A16_NVFP4",
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
        group_size=16,
    )
    cfg.vllm_omni_quant_recipe = RECIPE
    # Fail fast if this vLLM build does not resolve the weight-only W4A16 method:
    # a stock/older vLLM could otherwise silently attach the W4A4 method, which
    # needs an activation input_scale this artifact does not carry.
    method_name = getattr(cfg.LinearMethodCls, "__name__", "")
    if method_name != "ModelOptNvFp4W4A16LinearMethod":
        raise RuntimeError(
            f"nvfp4_blockwise requires vLLM's ModelOptNvFp4W4A16LinearMethod but this build "
            f"resolved {method_name!r}. Serve with the version-matched vllm-omni image "
            "(vLLM 0.24.0 with W4A16_NVFP4 support)."
        )
    logger.info(
        "Built NVFP4 blockwise W4A16 config (recipe=%s): target-inclusion on "
        "'.mlp|.mlp_moe_gen.{gate,up,down}_proj', method=%s",
        RECIPE,
        cfg.LinearMethodCls.__name__,
    )
    return cfg


def _is_generic_nvfp4(quant_config: QuantizationConfig) -> bool:
    getter = getattr(quant_config, "get_name", None)
    return callable(getter) and getter() in {"fp4", "nvfp4", "modelopt_fp4"}


def is_nvfp4_blockwise_w4a16_config(quant_config: QuantizationConfig | None) -> bool:
    return (
        quant_config is not None
        and getattr(quant_config, "vllm_omni_quant_recipe", None) == RECIPE
        and getattr(getattr(quant_config, "LinearMethodCls", None), "__name__", "") == "ModelOptNvFp4W4A16LinearMethod"
    )


def maybe_build_nvfp4_blockwise_config(
    quant_recipe: str | None,
    active_quant_config: QuantizationConfig | None = None,
) -> QuantizationConfig | None:
    """Resolve the NVFP4 blockwise W4A16 config from a transformer's recipe.

    Rules (mirroring the intent of ``resolve_quant_config_from_disk``):
      - ``quant_recipe`` is not the blockwise recipe: return *active* unchanged.
      - an explicit, non-generic active config is present: respect it (do not
        silently override an operator's choice).
      - otherwise: build and return the W4A16 target-inclusion config.
    """
    if quant_recipe != RECIPE:
        return active_quant_config
    if active_quant_config is not None and not _is_generic_nvfp4(active_quant_config):
        logger.info(
            "nvfp4_blockwise recipe present but a specific quant_config (%s) is active; keeping it.",
            getattr(active_quant_config, "get_name", lambda: active_quant_config)(),
        )
        return active_quant_config
    return build_nvfp4_blockwise_w4a16_config()
