# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extended INC/AutoRound config for multi-stage omni models."""

from __future__ import annotations

from os.path import commonprefix
from typing import TYPE_CHECKING, Any

from vllm.model_executor.layers.quantization.inc import INCConfig
from vllm.model_executor.models.utils import WeightsMapper

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )

_REGEX_SPECIAL_CHARS = frozenset(r"*+?^$()[]{}|\\")


def _stage_prefix(prefix_map: dict[str, str | None]) -> str:
    """Derive the container/stage prefix from mapper source keys."""
    cp = commonprefix(list(prefix_map.keys()))
    dot = cp.rfind(".")
    return cp[: dot + 1] if dot >= 0 else ""


def _map_with_stage_prefix(
    items: list[str],
    prefix_map: dict[str, str | None],
    stage: str,
) -> list[str]:
    """Apply *prefix_map* to each item and prepend *stage* to mapped items."""
    sorted_keys = sorted(prefix_map, key=len, reverse=True)
    result: list[str] = []
    for item in items:
        new_item = item
        for orig in sorted_keys:
            if item.startswith(orig):
                new_val = prefix_map[orig] or ""
                new_item = stage + new_val + item[len(orig) :]
                break
        result.append(new_item)
    return result


class OmniINCConfig(INCConfig):
    """INCConfig extended with multi-stage prefix remapping."""

    # ------------------------------------------------------------------
    # Core integration: called by vLLM's configure_quant_config()
    # ------------------------------------------------------------------

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        """Remap HF checkpoint names to vLLM runtime prefixes."""
        prefix_map = getattr(hf_to_vllm_mapper, "orig_to_new_prefix", None) or {}
        stage = _stage_prefix(prefix_map) if prefix_map else ""

        # -- Normalize CSV string -----------------------------------------
        if isinstance(self.block_name_to_quantize, str):
            self.block_name_to_quantize = [b.strip() for b in self.block_name_to_quantize.split(",") if b.strip()]

        # -- block_name_to_quantize ----------------------------------------
        if self.block_name_to_quantize is not None:
            if prefix_map and stage:
                self.block_name_to_quantize = _map_with_stage_prefix(
                    self.block_name_to_quantize,
                    prefix_map,
                    stage,
                )
            else:
                self.block_name_to_quantize = hf_to_vllm_mapper.apply_list(self.block_name_to_quantize)

        # -- extra_config --------------------------------------------------
        if self.extra_config is not None and prefix_map:
            new_extra: dict[str, Any] = {}
            sorted_keys = sorted(prefix_map, key=len, reverse=True)

            # Build escaped-dot map for regex pattern keys
            escaped_map: dict[str, str] = {}
            for orig, new in prefix_map.items():
                escaped_map[orig.replace(".", r"\.")] = (new or "").replace(".", r"\.")
            escaped_sorted = sorted(escaped_map, key=len, reverse=True)

            for key, val in self.extra_config.items():
                is_regex = any(c in _REGEX_SPECIAL_CHARS for c in key)
                if is_regex:
                    # Regex keys: escaped-dot substring replacement.
                    # re.search matches anywhere so no stage prefix needed.
                    new_key = key
                    for esc_orig in escaped_sorted:
                        if esc_orig in new_key:
                            new_key = new_key.replace(
                                esc_orig,
                                escaped_map[esc_orig],
                                1,
                            )
                            break
                else:
                    # Plain keys: prefix replacement + stage prefix.
                    new_key = key
                    for orig in sorted_keys:
                        if key.startswith(orig):
                            new_val = prefix_map[orig] or ""
                            new_key = stage + new_val + key[len(orig) :]
                            break
                new_extra[new_key] = val
            self.extra_config = new_extra
        elif self.extra_config is not None:
            self.extra_config = hf_to_vllm_mapper.apply_dict(self.extra_config)

    # ------------------------------------------------------------------
    # Upgrading a vanilla INCConfig created by vLLM
    # ------------------------------------------------------------------

    @classmethod
    def from_inc_config(cls, inc: INCConfig) -> OmniINCConfig:
        """Promote a vanilla :class:`INCConfig` to :class:`OmniINCConfig`.

        Copies all attributes so that the new instance is a drop-in
        replacement.
        """
        omni = object.__new__(cls)
        omni.__dict__.update(inc.__dict__)
        return omni

    @classmethod
    def maybe_upgrade(cls, quant_config: QuantizationConfig | None) -> QuantizationConfig | None:
        """Upgrade *quant_config* to :class:`OmniINCConfig` if applicable.

        Returns the original config unchanged when it is not an INC
        config or is already an :class:`OmniINCConfig`.
        """
        if quant_config is None:
            return None
        if isinstance(quant_config, cls):
            return quant_config
        if isinstance(quant_config, INCConfig):
            return cls.from_inc_config(quant_config)
        return quant_config
