# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections.abc import Mapping
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


def _stage_value(stage: Any, key: str, default: Any = None) -> Any:
    if isinstance(stage, Mapping):
        return stage.get(key, default)
    return getattr(stage, key, default)


def should_enable_duplex_endpoint(
    stage_configs: list | None,
    *,
    config_path: str | None = None,
) -> bool:
    """Enable the realtime session handler for explicitly configured deployments."""
    has_typed_session_mode = False
    if stage_configs:
        for stage in stage_configs:
            model_config = _stage_value(stage, "model_config")
            session_mode = _stage_value(model_config, "session_mode")
            if session_mode is not None:
                has_typed_session_mode = True
            else:
                session_mode = _stage_value(stage, "session_mode")
            if session_mode == "duplex":
                return True
    if config_path:
        try:
            from vllm_omni.config.stage_config import resolve_deploy_yaml

            raw_config = resolve_deploy_yaml(config_path)
            # Follow the upstream Server-VAD contract: a turn-based model may
            # opt into the Realtime handler with an explicit duplex_session.
            # A raw session_mode must not override an already-resolved typed
            # mode, but duplex_session is an independent endpoint opt-in.
            if isinstance(raw_config.get("duplex_session"), dict):
                return True
            if not has_typed_session_mode and raw_config.get("session_mode") == "duplex":
                return True
        except Exception as exc:
            logger.warning("Failed to inspect realtime session configuration from %s: %s", config_path, exc)
    return False


__all__ = ["should_enable_duplex_endpoint"]
