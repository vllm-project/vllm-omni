# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Validated, framework-independent Lychee-FD checkpoint contract.

The old integration inferred branch layout and special-token semantics in the
model forward path.  The native port validates them once, before model or
session state is allocated, so a malformed or mismatched checkpoint fails
with an actionable error.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum


class LycheeConfigError(ValueError):
    """The checkpoint does not satisfy the Lychee-FD model contract."""


class LycheeDialogueState(str, Enum):
    """Model-owned dialogue state exposed through Unified Duplex events."""

    LISTENING = "listening"
    SPEAKING = "speaking"
    BACKCHANNEL = "backchannel"


def _read(config: object, key: str, default: object = None) -> object:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _required(config: object, key: str, *, context: str) -> object:
    value = _read(config, key)
    if value is None:
        raise LycheeConfigError(f"Missing {context}.{key}")
    return value


def _required_int(config: object, key: str, *, context: str) -> int:
    value = _required(config, key, context=context)
    if isinstance(value, bool):
        raise LycheeConfigError(f"{context}.{key} must be an integer, got bool")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise LycheeConfigError(f"{context}.{key} must be an integer, got {value!r}") from exc
    return result


@dataclass(frozen=True, slots=True)
class LycheeBranchLayout:
    """Layer topology shared by weight loading and the MRV2 model state."""

    main_layers: int
    stoken_layers: int
    control_layers: int
    merge_layers: int
    control_branch_index: int
    hidden_size: int
    vocab_size: int

    @property
    def total_layers(self) -> int:
        return self.main_layers + self.stoken_layers + self.control_layers + self.merge_layers

    @classmethod
    def from_config(cls, config: object) -> LycheeBranchLayout:
        sections = {
            "text_config": _required(config, "text_config", context="config"),
            "stoken_layer_config": _required(config, "stoken_layer_config", context="config"),
            "control_layer_config": _required(config, "control_layer_config", context="config"),
            "merge_layer_config": _required(config, "merge_layer_config", context="config"),
        }
        layer_counts = {
            name: _required_int(section, "num_hidden_layers", context=name) for name, section in sections.items()
        }
        if any(count <= 0 for count in layer_counts.values()):
            raise LycheeConfigError(f"Every Lychee-FD branch must contain layers: {layer_counts}")

        hidden_sizes = {_required_int(section, "hidden_size", context=name) for name, section in sections.items()}
        if len(hidden_sizes) != 1:
            raise LycheeConfigError(f"All Lychee-FD branches must share hidden_size, got {sorted(hidden_sizes)}")

        vocab_sizes = {_required_int(section, "vocab_size", context=name) for name, section in sections.items()}
        if len(vocab_sizes) != 1:
            raise LycheeConfigError(f"All Lychee-FD branches must share vocab_size, got {sorted(vocab_sizes)}")

        main_layers = layer_counts["text_config"]
        branch_from_top = _required_int(config, "control_branch_layer", context="config")
        control_branch_index = main_layers - branch_from_top
        if not 0 <= control_branch_index < main_layers:
            raise LycheeConfigError(
                "config.control_branch_layer must select a main-decoder boundary; "
                f"got {branch_from_top} for {main_layers} main layers"
            )

        return cls(
            main_layers=main_layers,
            stoken_layers=layer_counts["stoken_layer_config"],
            control_layers=layer_counts["control_layer_config"],
            merge_layers=layer_counts["merge_layer_config"],
            control_branch_index=control_branch_index,
            hidden_size=hidden_sizes.pop(),
            vocab_size=vocab_sizes.pop(),
        )


@dataclass(frozen=True, slots=True)
class LycheeControlTokenIds:
    """Checkpoint-provided control and speech-token vocabulary boundaries."""

    start_speaking: int
    start_listening: int
    keep_speaking: int
    keep_listening: int
    start_backchannel: int
    keep_backchannel: int
    end_backchannel: int
    stoken_min: int
    stoken_max: int
    control_min: int
    control_max: int

    @classmethod
    def from_config(cls, config: object, *, vocab_size: int) -> LycheeControlTokenIds:
        values = cls(
            start_speaking=_required_int(config, "start_speaking_token_id", context="config"),
            start_listening=_required_int(config, "start_listening_token_id", context="config"),
            keep_speaking=_required_int(config, "keep_speaking_token_id", context="config"),
            keep_listening=_required_int(config, "keep_listening_token_id", context="config"),
            start_backchannel=_required_int(config, "start_bc_token_id", context="config"),
            keep_backchannel=_required_int(config, "keep_bc_token_id", context="config"),
            end_backchannel=_required_int(config, "end_bc_token_id", context="config"),
            stoken_min=_required_int(config, "stoken_token_ids_min", context="config"),
            stoken_max=_required_int(config, "stoken_token_ids_max", context="config"),
            control_min=_required_int(config, "control_token_ids_min", context="config"),
            control_max=_required_int(config, "control_token_ids_max", context="config"),
        )
        if not 0 <= values.stoken_min < values.stoken_max <= vocab_size:
            raise LycheeConfigError(
                f"Invalid speech-token range [{values.stoken_min}, {values.stoken_max}) for vocab_size={vocab_size}"
            )
        if not 0 <= values.control_min < values.control_max <= vocab_size:
            raise LycheeConfigError(
                f"Invalid control-token range [{values.control_min}, {values.control_max}) for vocab_size={vocab_size}"
            )
        for name in (
            "start_speaking",
            "start_listening",
            "keep_speaking",
            "keep_listening",
            "start_backchannel",
            "keep_backchannel",
            "end_backchannel",
        ):
            token_id = getattr(values, name)
            if not 0 <= token_id < vocab_size:
                raise LycheeConfigError(f"{name}={token_id} is outside vocab_size={vocab_size}")
        return values

    def next_state(self, current: LycheeDialogueState, token_id: int) -> LycheeDialogueState:
        """Apply one control-head token without inventing client-side policy."""

        if token_id == self.start_speaking or token_id == self.keep_speaking:
            return LycheeDialogueState.SPEAKING
        if token_id == self.start_listening or token_id == self.keep_listening or token_id == self.end_backchannel:
            return LycheeDialogueState.LISTENING
        if token_id == self.start_backchannel or token_id == self.keep_backchannel:
            return LycheeDialogueState.BACKCHANNEL
        return current


@dataclass(frozen=True, slots=True)
class LycheeFDContract:
    """Normalized values used by the native model, plugin, and deploy checks."""

    layout: LycheeBranchLayout
    tokens: LycheeControlTokenIds
    input_sample_rate_hz: int
    output_sample_rate_hz: int
    inference_window_ms: int
    stoken_delay: int

    @classmethod
    def from_config(
        cls,
        config: object,
        *,
        input_sample_rate_hz: int = 16_000,
        output_sample_rate_hz: int = 24_000,
        inference_window_ms: int = 400,
    ) -> LycheeFDContract:
        model_type = _required(config, "model_type", context="config")
        if model_type != "step_audio_2_full_duplex":
            raise LycheeConfigError(f"Lychee-FD requires model_type='step_audio_2_full_duplex', got {model_type!r}")
        layout = LycheeBranchLayout.from_config(config)
        tokens = LycheeControlTokenIds.from_config(config, vocab_size=layout.vocab_size)
        stoken_delay = _required_int(config, "stoken_delay_num", context="config")
        if stoken_delay < 0:
            raise LycheeConfigError(f"config.stoken_delay_num must be non-negative, got {stoken_delay}")
        if input_sample_rate_hz <= 0 or output_sample_rate_hz <= 0 or inference_window_ms <= 0:
            raise LycheeConfigError("Sample rates and inference_window_ms must be positive")
        return cls(
            layout=layout,
            tokens=tokens,
            input_sample_rate_hz=input_sample_rate_hz,
            output_sample_rate_hz=output_sample_rate_hz,
            inference_window_ms=inference_window_ms,
            stoken_delay=stoken_delay,
        )


__all__ = [
    "LycheeBranchLayout",
    "LycheeConfigError",
    "LycheeControlTokenIds",
    "LycheeDialogueState",
    "LycheeFDContract",
]
