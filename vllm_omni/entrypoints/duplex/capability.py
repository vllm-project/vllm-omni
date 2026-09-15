# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from vllm.logger import init_logger

logger = init_logger(__name__)


def should_enable_duplex_endpoint(
    stage_configs: list | None,
    *,
    config_path: str | None = None,
) -> bool:
    """Enable the realtime session handler for explicitly configured deployments."""
    if stage_configs:
        for stage in stage_configs:
            session_mode = (
                stage.get("session_mode") if isinstance(stage, dict) else getattr(stage, "session_mode", None)
            )
            if session_mode == "duplex":
                return True
    if config_path:
        try:
            from vllm_omni.config.stage_config import resolve_deploy_yaml

            raw_config = resolve_deploy_yaml(config_path)
            if raw_config.get("session_mode") == "duplex" or isinstance(raw_config.get("duplex_session"), dict):
                return True
        except Exception as exc:
            logger.warning("Failed to inspect realtime session configuration from %s: %s", config_path, exc)
    return False


__all__ = ["should_enable_duplex_endpoint"]
