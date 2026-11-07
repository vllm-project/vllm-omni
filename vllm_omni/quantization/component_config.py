# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-component quantization routing for multi-stage models.

Routes get_quant_method() to different configs based on longest-prefix match:
    {"transformer": fp8_config, "vae": None}
    "transformer.blocks.0.attn.to_q" -> fp8_config
    "vae.encoder.conv_in"            -> None

Component prefixes may use ``fnmatch`` wildcards to target a subtree of
repeated layers, e.g. ``model.layers.*.mlp.experts`` selects the MoE experts
of every decoder layer while leaving attention/norm layers unquantized.
"""

from __future__ import annotations

import fnmatch
from typing import TYPE_CHECKING, Any

import torch
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizeMethodBase,
    )
    from vllm.model_executor.models.utils import (
        WeightsMapper,
    )


# These pre-quantized formats require serialized scale or correction tensors
# that the vision and audio encoder checkpoints do not provide.
PRE_QUANTIZED_METHODS: frozenset[str] = frozenset(
    {"modelopt", "modelopt_fp4", "modelopt_mxfp8", "modelopt_mixed", "svdquant"}
)


def resolve_component_quant_config(
    quant_config: QuantizationConfig | None,
    component: str,
) -> QuantizationConfig | None:
    """Resolve one pipeline component from a global or component config.

    A plain config is global and therefore applies unchanged to every
    quantization-aware component. Only ``ComponentQuantizationConfig`` narrows
    the scope through its explicit prefix map.
    """
    if isinstance(quant_config, ComponentQuantizationConfig):
        return quant_config.resolve(component)
    return quant_config


def _is_wildcard(pattern: str) -> bool:
    """Return True if *pattern* contains fnmatch wildcard metacharacters."""
    return any(c in pattern for c in "*?[")


def resolve_encoder_quant_config(
    quant_config: QuantizationConfig | None,
) -> QuantizationConfig | None:
    """Resolve quantization config for vision / audio encoders.

    Returns *None* for pre-quantized methods so that FP8 kernels are never
    applied to BF16 encoder weights (which lack scale tensors).  All other
    configs — including ``ComponentQuantizationConfig`` and ``None`` — are
    returned as-is so the caller can handle them.
    """
    if (
        quant_config is not None
        and not isinstance(quant_config, ComponentQuantizationConfig)
        and quant_config.get_name() in PRE_QUANTIZED_METHODS
    ):
        return None
    return quant_config


def safe_quant_config(
    quant_config: QuantizationConfig | None,
) -> QuantizationConfig | None:
    """Return *quant_config* only if it is safe for norm/modulation layers.

    Norm and modulation layers (LayerNorm, RMSNorm, AdaLayerNorm, img_mod,
    txt_mod, etc.) produce precision-sensitive shift/scale/gate values and
    should not receive FP8 quant configs (see #2728).  Pre-quantized methods
    like INC/AutoRound W4A16 need the config propagated so packed weights
    load correctly.

    This is the inverse of :func:`resolve_encoder_quant_config`: that function
    strips pre-quantized configs from encoders, while this one strips
    *every config except* pre-quantized configs from norm/mod layers.
    """
    if quant_config is None:
        return None
    from vllm.model_executor.layers.quantization.inc import INCConfig

    if isinstance(quant_config, INCConfig):
        return quant_config
    return None


class ComponentQuantizationConfig(QuantizationConfig):
    """Routes quantization to different configs by layer prefix."""

    def __init__(
        self,
        component_configs: dict[str, QuantizationConfig | None],
        default_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self._components = component_configs
        self._default = default_config
        self._sorted_prefixes = sorted(self._components.keys(), key=len, reverse=True)

    def resolve(self, prefix: str) -> QuantizationConfig | None:
        """Find the config for a given layer prefix (longest-prefix match).

        Component keys are matched two ways, in order of specificity:

        1. ``fnmatch`` wildcard match against the full prefix (keys containing
           ``*``/``?``/``[``), e.g. ``model.layers.*.mlp.experts`` matches
           ``model.layers.0.mlp.experts``.
        2. Plain ``startswith`` prefix match, e.g. ``language_model`` matches
           ``language_model.model.layers.0``.

        Keys are tried longest-first so the most specific component wins.

        Note: vLLM may remap quantization prefixes vs model definition
        prefixes (e.g. via WeightsMapper). :meth:`apply_vllm_mapper` rewrites
        the component keys to the runtime structure; unmatched layers fall
        through to the default config.
        """
        for comp_prefix in self._sorted_prefixes:
            if _is_wildcard(comp_prefix):
                if fnmatch.fnmatchcase(prefix, comp_prefix) or fnmatch.fnmatchcase(prefix, comp_prefix + ".*"):
                    return self._components[comp_prefix]
            elif prefix.startswith(comp_prefix):
                return self._components[comp_prefix]
        return self._default

    def get_name(self) -> str:
        return "component"

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        config = self.resolve(prefix)
        if config is not None:
            return config.get_quant_method(layer, prefix)

        # No component config applies to this layer (matched a null component
        # or fell through to a null default). Returning ``None`` is only safe
        # for layers whose ``__init__`` tolerates it (embeddings, FusedMoE).
        # ``LinearBase`` raises "All linear layers should support quant method."
        # on ``None``, so hand it an explicit unquantized method — mirroring how
        # vLLM's own Fp8Config handles layers listed in ``ignored_layers``.
        from vllm.model_executor.layers.linear import (  # noqa: PLC0415
            LinearBase,
            UnquantizedLinearMethod,
        )

        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        return None

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        """Remap component prefixes from checkpoint names to vLLM runtime names.

        Checkpoint ``quantization_config`` keys reference the HF module tree
        (e.g. ``model.layers.*.mlp.experts``), but at runtime a model's
        ``hf_to_vllm_mapper`` may strip/rename prefixes (e.g. ``model.`` ->
        ``""`` so layers are built as ``layers.0.mlp.experts``). Without this
        rewrite the component keys would never match and every layer would
        fall through to the default config.

        Delegates to nested configs so their own layer lists (e.g. INC
        ``block_name_to_quantize``) are remapped too.
        """
        prefix_map = getattr(hf_to_vllm_mapper, "orig_to_new_prefix", None) or {}
        if prefix_map:
            sorted_orig = sorted(prefix_map, key=len, reverse=True)
            remapped: dict[str, QuantizationConfig | None] = {}
            for comp_prefix, cfg in self._components.items():
                new_prefix = comp_prefix
                for orig in sorted_orig:
                    if comp_prefix.startswith(orig):
                        new_prefix = (prefix_map[orig] or "") + comp_prefix[len(orig) :]
                        break
                remapped[new_prefix] = cfg
            self._components = remapped
            self._sorted_prefixes = sorted(self._components.keys(), key=len, reverse=True)

        for cfg in self._components.values():
            if cfg is not None:
                cfg.apply_vllm_mapper(hf_to_vllm_mapper)
        if self._default is not None:
            self._default.apply_vllm_mapper(hf_to_vllm_mapper)

    def get_min_capability(self) -> int:
        """Return the minimum capability across all component configs."""
        caps = [c.get_min_capability() for c in self._components.values() if c is not None]
        if self._default is not None:
            caps.append(self._default.get_min_capability())
        return min(caps) if caps else 0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ComponentQuantizationConfig:
        raise NotImplementedError("Use build_quant_config() instead")

    def get_config_filenames(self) -> list[str]:
        return []

    @property
    def component_configs(self) -> dict[str, QuantizationConfig | None]:
        return self._components

    @property
    def default_config(self) -> QuantizationConfig | None:
        return self._default
