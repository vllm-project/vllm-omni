# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-component quantization routing for multi-stage models.

Routes get_quant_method() to different configs based on longest-prefix match:
    {"transformer": fp8_config, "vae": None}
    "transformer.blocks.0.attn.to_q" -> fp8_config
    "vae.encoder.conv_in"            -> None

A key may also be a shell-style pattern, which expresses precision per layer role
rather than per component::

    {"transformer": mxfp8_config,
     "transformer.*.mlp": nvfp4_config,     # every MLP in the stack
     "transformer.*.attn": mxfp8_config,
     "refiner": None}

A pattern is matched against the layer prefix and each of its dotted ancestors
(``transformer.blocks.0.mlp.fc1`` is tried as itself, then ``...mlp``, then
``...blocks.0``, and so on), so ``transformer.*.mlp`` covers every layer under an
MLP. Patterns are checked before plain prefixes because a pattern refines a
component: if any pattern matches, the deepest matching pattern wins (patterns are one
set, so a plain key never out-ranks a matching role pattern — otherwise a broad
``transformer`` key, which covers every path below it, would make per-role keys
unreachable); plain prefixes are consulted only when no pattern matches, longest first.

``*`` follows fnmatch and therefore spans ``.``: ``*.mlp`` matches the ancestor
``transformer.blocks.0.mlp``, and ``transformer.*`` matches every path below
``transformer``. Write the trailing segment you mean (``*.mlp``, ``*.attn``) rather
than relying on ``*`` stopping at a dot.
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
        # A key with a wildcard is a layer-role pattern and refines a plain component
        # prefix, so patterns are consulted first. Longest pattern wins so a narrower
        # role beats a broader one, and the order is deterministic for equal lengths.
        self._patterns = {k: v for k, v in component_configs.items() if any(c in k for c in "*?[")}
        self._sorted_patterns = sorted(self._patterns.keys(), key=len, reverse=True)
        self._sorted_prefixes = sorted((k for k in self._components if k not in self._patterns), key=len, reverse=True)

    def matches(self, prefix: str) -> bool:
        """Whether *prefix* is explicitly covered by a key (not by the default).

        Lets a caller distinguish "this layer role was named in the config" from
        "this layer merely fell through to the default", which matters when the
        default must not silently override a model's own default arm.
        """
        return self._match(prefix) is not None

    def _match(self, prefix: str) -> tuple[str, QuantizationConfig | None] | None:
        """Return the (key, config) that explicitly covers *prefix*, else None."""
        # Deepest ancestor first (segs[0] is the prefix itself), then longest pattern.
        ancestors = [".".join(prefix.split(".")[:i]) for i in range(len(prefix.split(".")), 0, -1)]
        best: tuple[tuple[int, int], str, QuantizationConfig | None] | None = None
        for depth, ancestor in enumerate(ancestors):
            for pattern in self._sorted_patterns:
                if fnmatch.fnmatchcase(ancestor, pattern):
                    key = (len(ancestors) - depth, len(pattern))
                    if best is None or key > best[0]:
                        best = (key, pattern, self._patterns[pattern])
            if best is not None and best[0][0] == len(ancestors) - depth:
                break  # no shallower ancestor can beat a deeper match
        if best is not None:
            return (best[1], best[2])
        for comp_prefix in self._sorted_prefixes:
            if prefix.startswith(comp_prefix):
                return (comp_prefix, self._components[comp_prefix])
        return None

    def resolve(self, prefix: str) -> QuantizationConfig | None:
        """Find the config for a given layer prefix.

        Layer-role patterns are tried first (longest pattern wins), then plain
        component prefixes (longest prefix wins), then the default.

        Note: vLLM may remap quantization prefixes vs model definition
        prefixes (e.g. via WeightsMapper). If prefixes don't match after
        remapping, layers may fall through to the default config.
        """
        matched = self._match(prefix)
        return matched[1] if matched is not None else self._default

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        """Apply a weight mapper to every routed quantization config."""
        for quant_config in self._components.values():
            if quant_config is not None:
                quant_config.apply_vllm_mapper(hf_to_vllm_mapper)
        if self._default is not None:
            self._default.apply_vllm_mapper(hf_to_vllm_mapper)

    def get_name(self) -> str:
        return "component"

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        config = self.resolve(prefix)
        if config is None:
            return None
        return config.get_quant_method(layer, prefix)

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

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
