# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger

logger = init_logger(__name__)


def should_enable_duplex_endpoint(
    stage_configs: list | None,
    *,
    config_path: str | None = None,
) -> bool:
    """Enable duplex routes only for deployments that explicitly opt in."""
    if stage_configs:
        for stage in stage_configs:
            session_mode = (
                stage.get("session_mode") if isinstance(stage, dict) else getattr(stage, "session_mode", None)
            )
            if session_mode == "duplex":
                return True
    if config_path:
        try:
            from omegaconf import OmegaConf

            raw_config = OmegaConf.load(config_path)
            session_mode = raw_config.get("session_mode") if hasattr(raw_config, "get") else None
            if session_mode == "duplex":
                return True
        except Exception as exc:
            logger.warning("Failed to inspect duplex session_mode from %s: %s", config_path, exc)
    return False


__all__ = ["should_enable_duplex_endpoint"]
