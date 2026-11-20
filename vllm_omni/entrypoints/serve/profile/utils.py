# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Utilities for Omni profile control routes."""


def _should_enable_profiler_endpoints(stage_configs: list | None) -> bool:
    """Check whether any typed or legacy stage enables a profiler."""
    if not stage_configs:
        return False
    for stage in stage_configs:
        profiler_config = (
            stage.get("profiler_config") if isinstance(stage, dict) else getattr(stage, "profiler_config", None)
        )
        if profiler_config is None:
            engine_args = stage.get("engine_args") if isinstance(stage, dict) else getattr(stage, "engine_args", None)
            profiler_config = (
                engine_args.get("profiler_config")
                if isinstance(engine_args, dict)
                else getattr(engine_args, "profiler_config", None)
            )
        if profiler_config is not None:
            profiler = (
                profiler_config.get("profiler")
                if isinstance(profiler_config, dict)
                else getattr(profiler_config, "profiler", None)
            )
            if profiler is not None:
                return True
    return False
