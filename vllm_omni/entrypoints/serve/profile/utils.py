# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Utilities for Omni profile control routes."""


def _should_enable_profiler_endpoints(stage_configs: list | None) -> bool:
    """Check if any stage has profiler_config set in its engine_args."""
    if not stage_configs:
        return False
    for stage in stage_configs:
        engine_args = stage.get("engine_args") if isinstance(stage, dict) else getattr(stage, "engine_args", None)
        if engine_args is None:
            continue
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
