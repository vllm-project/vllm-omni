"""Deploy-time runtime state shared across process boundaries.

``load_deploy_config`` records facts here while the deploy YAML is being
resolved; readers that run in spawned stage workers -- where no
``vllm_config`` is available yet -- query it afterwards. The state is
deliberately tiny and model-agnostic: it describes what the deploy layer
resolved, not what any particular model does with it.
"""

from typing import Any

# Stage 1's multi-frame decode is configured through its ``speculative_config``
# in the deploy YAML (method ``ngram`` with ``num_speculative_tokens == K - 1``).
# The parse records the resulting frame count here because the reader
# (``ascend_warmup_patch._kstep_armed``) can run in a spawned stage worker that
# has no vllm_config to read it from. It only answers "armed or not" -- 1 means
# not armed.
_resolved_talker_frames: int = 1


def talker_frames_per_step() -> int:
    """Codec frames one stage-1 step produces; 1 when the deploy layer asked for none."""
    return _resolved_talker_frames


def record_talker_frames(stages: list[Any], platforms: dict[str, Any] | None = None) -> None:
    """Record stage 1's frame count (its n-gram ``num_speculative_tokens`` is K - 1).

    The K block sits under ``platforms.npu`` -- the loop is NPU-only -- so the
    NPU overlay is merged here. This runs at deploy-load time, before the
    per-platform merge, and its reader (``ascend_warmup_patch._kstep_armed``)
    runs in a spawned stage worker that may have no vllm_config to read it from.
    """
    global _resolved_talker_frames
    npu_overlay: dict[int, dict[str, Any]] = {}
    for stage_override in ((platforms or {}).get("npu") or {}).get("stages") or []:
        if isinstance(stage_override, dict) and "stage_id" in stage_override:
            npu_overlay[stage_override["stage_id"]] = stage_override
    frames = 1
    for stage in stages:
        if getattr(stage, "stage_id", None) != 1:
            continue
        spec = (getattr(stage, "engine_extras", None) or {}).get("speculative_config")
        if spec is None:
            # Platform overrides follow _extract_platform_overrides: every key
            # but stage_id/devices/env is an engine or stage override.
            spec = (npu_overlay.get(1) or {}).get("speculative_config")
        if isinstance(spec, dict) and spec.get("method") == "ngram":
            num_spec = spec.get("num_speculative_tokens", 0) or 0
            if num_spec > 0:
                frames = int(num_spec) + 1
    _resolved_talker_frames = frames
