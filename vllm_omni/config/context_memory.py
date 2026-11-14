# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Experimental, deployment-owned limits for text memory extraction."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DuplexAuxiliaryBudgetConfig:
    """Explicit shared allowance; mixed work has no implicit deployment budget."""

    max_input_tokens: int
    max_output_tokens: int
    window_s: float
    cooldown_s: float
    idle_s: float

    def __post_init__(self) -> None:
        for name in ("max_input_tokens", "max_output_tokens"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 2**31 - 1:
                raise ValueError("invalid_context_auxiliary_budget_" + name)
        for name in ("window_s", "cooldown_s", "idle_s"):
            value = getattr(self, name)
            if type(value) not in (int, float):
                raise ValueError("invalid_context_auxiliary_budget_" + name)
            try:
                valid = math.isfinite(value) and value >= 0
            except OverflowError:
                valid = False
            if not valid or (name == "window_s" and value == 0):
                raise ValueError("invalid_context_auxiliary_budget_" + name)


@dataclass(frozen=True)
class DuplexContextMemoryConfig:
    """Opt-in extraction limits; absent configuration keeps extraction off."""

    max_sources: int = 8
    max_chars: int = 512
    input_token_limit: int = 2048
    output_token_limit: int = 512
    timeout_s: float = 10.0
    compute_input_tokens: int = 4096
    compute_output_tokens: int = 1024
    compute_window_s: float = 60.0
    cooldown_s: float = 5.0
    idle_s: float = 1.0

    def __post_init__(self) -> None:
        for name, maximum in (
            ("max_sources", 8),
            ("max_chars", 8192),
            ("input_token_limit", 2048),
            ("output_token_limit", 512),
            ("compute_input_tokens", 2**31 - 1),
            ("compute_output_tokens", 2**31 - 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError("invalid_context_memory_" + name)
        for name in ("timeout_s", "compute_window_s", "cooldown_s", "idle_s"):
            value = getattr(self, name)
            if type(value) not in (int, float):
                raise ValueError("invalid_context_memory_" + name)
            try:
                valid = math.isfinite(value) and value >= 0
            except OverflowError:
                valid = False
            if not valid or (name in {"timeout_s", "compute_window_s"} and value == 0):
                raise ValueError("invalid_context_memory_" + name)
        if self.timeout_s > 30 or self.compute_output_tokens < self.output_token_limit:
            raise ValueError("invalid_context_memory_request_allowance")
