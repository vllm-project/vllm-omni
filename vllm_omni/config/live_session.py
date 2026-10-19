# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Configuration for OpenAI Live sessions (``/v1/live/sessions``).

Two halves: ``LiveSessionConfig`` is a ``PipelineConfig`` attribute that
describes how a native full-duplex model runs on a Live session;
``LiveSessionDeployConfig`` is the optional ``live_session_config`` deploy-YAML
object that enables the endpoint for a deployment.

Kept free of engine and API-server imports: ``stage_config`` imports it, and
pipeline modules construct ``LiveSessionConfig`` at import time.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any


@dataclass(frozen=True)
class UnsupportedFeatures:
    """Live client events and session-config paths a model rejects.

    ``events`` are Live client event types; ``session_configs`` are dotted
    paths into ``session.start`` / ``session.update``. The server-wide
    non-goals are rejected before this list is consulted and need not be
    listed.
    """

    events: tuple[str, ...] = ()
    session_configs: tuple[str, ...] = ()


@dataclass(frozen=True)
class LiveSessionConfig:
    """Per-model capabilities for ``/v1/live/sessions``."""

    # Dotted path (``module:Class`` or ``module.Class``) of the model's
    # ``LiveSessionProcessor``; resolved lazily by the API server so pipeline
    # modules stay import-light.
    live_session_processor: str
    # Stage id -> ``module:Class`` of a vLLM ``LogitsProcessor``, appended to
    # that stage's ``logits_processors`` engine arg on Live deployments.
    logits_processor: Mapping[int, str] = field(default_factory=dict)
    unsupported_features: UnsupportedFeatures = field(default_factory=UnsupportedFeatures)

    def get_validation_errors(self) -> list[str]:
        errors: list[str] = []
        if not self.live_session_processor:
            errors.append("live_session_config.live_session_processor is required")
        for stage_id, fqcn in self.logits_processor.items():
            if not isinstance(stage_id, int) or stage_id < 0:
                errors.append(f"live_session_config.logits_processor key {stage_id!r} must be a stage id")
            if ":" not in fqcn:
                errors.append(f"live_session_config.logits_processor[{stage_id}] must be 'module:Class', got {fqcn!r}")
        return errors


@dataclass(frozen=True)
class LiveSessionDeployConfig:
    """The ``live_session_config`` deploy-YAML object.

    Its presence enables ``/v1/live/sessions`` for the deployment.
    """

    # Wall-clock limit on a session's total duration; fills ``expires_at``.
    session_lifetime_s: int = 1800
    # Concurrent Live sessions on this API server; None is unbounded.
    max_sessions: int | None = None

    def __post_init__(self) -> None:
        if self.session_lifetime_s <= 0:
            raise ValueError("live_session_config.session_lifetime_s must be positive")
        if self.max_sessions is not None and self.max_sessions <= 0:
            raise ValueError("live_session_config.max_sessions must be positive or null")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> LiveSessionDeployConfig:
        raw = dict(raw or {})
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown live_session_config keys: {sorted(unknown)}")
        return cls(**raw)
