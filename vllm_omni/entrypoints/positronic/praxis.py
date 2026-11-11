# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Praxis-specific query parsing and metadata for Positronic sessions."""

from __future__ import annotations

import json
import secrets
from collections.abc import Mapping

import msgspec

from vllm_omni.model_extras.praxis1 import DEFAULT_MAX_NUM_STEPS, PraxisSessionOptions

STACK_KEYS = ("image.wrist", "image.exterior", "robot_state.ee_pose", "grip")


def parse_session(
    query: Mapping[str, str], defaults: Mapping[str, object]
) -> tuple[PraxisSessionOptions, dict[str, object]]:
    """Parse JSON query literals and refuse settings that native execution cannot honor."""
    values: dict[str, object] = {}
    strings = {"rollout_id", "prompt", "context_mode", "control_mode", "carry_mode", "solver", "prompt_style"}
    for key, raw in query.items():
        if key in strings and not raw.startswith('"'):
            values[key] = raw
            continue
        if key == "return_server_timing" and raw.lower() in {"1", "on", "0", "off"}:
            values[key] = raw.lower() in {"1", "on"}
            continue
        try:
            values[key] = json.loads(raw)
        except json.JSONDecodeError:
            values[key] = raw
    names = PraxisSessionOptions.__dataclass_fields__
    extra = {
        "sample_seed",
        "prompt",
        "rollout_id",
        "return_server_timing",
        "max_chunks_per_episode",
        "declare_stop_on_fault",
    }
    fixed = {
        "carry_mode": "clean",
        "solver": "euler",
        "cfg_batched": False,
        "smoothing": None,
        "record_denoised": False,
        "compile": False,
        "skip_video_decoder": True,
        "prompt_style": "caption",
        "pad_start": False,
        "stride": 1,
    }
    unknown = values.keys() - names.keys() - extra - fixed.keys() - {"history_frames", "stack_keys", "action_steps"}
    if unknown:
        raise ValueError(f"Unsupported session parameters: {sorted(unknown)}")
    for key, expected in fixed.items():
        if key in values and (
            values[key] != expected or (isinstance(expected, bool) and type(values[key]) is not bool)
        ):
            raise ValueError(f"{key}={values[key]!r} is not supported; expected {expected!r}")
    options = msgspec.convert(
        {k: values.get(k, defaults.get(k, f.default)) for k, f in names.items()}, type=PraxisSessionOptions
    )
    maximum = defaults.get("max_num_steps", DEFAULT_MAX_NUM_STEPS)
    if type(maximum) is not int or not 1 <= options.num_steps <= maximum:
        raise ValueError(f"num_steps must be in [1, {maximum}]")
    stack_keys = defaults.get("stack_keys", STACK_KEYS)
    if not isinstance(stack_keys, (list, tuple)) or not stack_keys or any(not isinstance(k, str) for k in stack_keys):
        raise ValueError("Configured stack_keys must be a nonempty list of keys")
    _validate_history(values, options, tuple(stack_keys))
    if defaults.get("embodiment") == "yam_ee2" and options.control_mode != "none":
        raise ValueError("YAM control_mode must be none")
    return options, _session_values(values)


def _validate_history(values: Mapping[str, object], options: PraxisSessionOptions, stack_keys: tuple[str, ...]) -> None:
    """Check aliases and client-side stacking against the selected execution mode."""
    for key, expected in (("history_frames", 0 if options.stateless else 24), ("action_steps", options.num_steps)):
        if (
            key in values
            and not (key == "action_steps" and values[key] is None)
            and (type(values[key]) is not int or values[key] != expected)
        ):
            raise ValueError(f"{key} must be {expected}")
    if "stack_keys" in values:
        keys = values["stack_keys"]
        if isinstance(keys, str):
            keys = [k.strip() for k in keys.split(",") if k.strip()]
        if not isinstance(keys, list) or tuple(keys) != stack_keys:
            raise ValueError("stack_keys must match the declared rig")


def _session_values(values: dict[str, object]) -> dict[str, object]:
    """Validate envelope settings and choose the connection's base seed."""
    seed = values.get("sample_seed")
    if seed is None:
        seed = secrets.randbits(31)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("sample_seed must be a nonnegative 63-bit integer")
    for key in ("return_server_timing", "declare_stop_on_fault"):
        if key in values and type(values[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    budget = values.get("max_chunks_per_episode")
    if budget is not None and (type(budget) is not int or budget < 1):
        raise ValueError("max_chunks_per_episode must be a positive integer")
    if not isinstance(values.get("prompt", ""), str):
        raise ValueError("prompt must be a string")
    return {
        **values,
        "sample_seed": seed,
        "seed_source": "session_param" if values.get("sample_seed") is not None else "rotated",
    }


def session_metadata(
    options: PraxisSessionOptions,
    values: Mapping[str, object],
    model: str,
    codecs: list[str],
    policy_config: Mapping | None = None,
) -> dict[str, object]:
    """Declare the standard Positronic rig pipeline without importing model weights."""
    from vllm_omni import __version__

    policy_config = policy_config or {}
    tile_size = policy_config.get("tile_size", [224, 416])
    height, width = policy_config.get("image_resolution", [234, 416])
    fps = policy_config.get("fps", 15)
    restrict: dict[str, object] = {
        "name": "restrict_image_size",
        "args": {"width": width, "height": height},
    }
    seq: list[dict[str, object]] = [{"name": "stop_on_fault"}] if values.get("declare_stop_on_fault", True) else []
    seq.append(restrict)
    if not options.stateless:
        seq.append(
            {
                "name": "temporal_stack",
                "args": {
                    "keys": policy_config.get("stack_keys", list(STACK_KEYS)),
                    "offsets_sec": [0.0 if i == 0 else -i / fps + 0.5 / fps for i in range(24, -1, -1)],
                    "pad_start": False,
                },
            }
        )
    seq.append({"name": "chunked_schedule"})
    return {
        **options.to_dict(),
        "policy": "praxis",
        "code_version": f"vllm-omni-{__version__}",
        "positronic_wire_ref": "d5f2a47",
        "arm": model,
        "checkpoint": model,
        "embodiment": policy_config.get("embodiment", "franka_ee"),
        "chunk_rows": 24,
        "action_fps": fps,
        "sample_seed_base": values["sample_seed"],
        "seed_source": values["seed_source"],
        "prompt": values.get("prompt", ""),
        "prompt_overridable": True,
        "prompt_style": "caption",
        "prompt_rewritten": None,
        "pad_start": False,
        "wire_commands": policy_config.get("wire_commands", ["robot_command", "target_grip"]),
        "action_layout": policy_config.get("action_layout", "ee-10d [xyz3 rot6d6 grip1], rel-to-anchor, grip abs"),
        "context_mode": "stateless" if options.stateless else "observed",
        "carry_mode": "clean",
        "solver": "euler",
        "cfg_batched": False,
        "num_evals": options.num_steps,
        "record_denoised": False,
        "compile": False,
        "skip_video_decoder": True,
        "action_steps": None,
        "sampler_layout": "window_joint",
        "backbone_layout": "chunk_causal",
        "canvas": "tiles",
        "tile_size": tile_size,
        "pose_frame": policy_config.get("pose_frame", "flange"),
        "text_uncond": "blank_caption",
        "wan_pattern": "contiguous-window-144444",
        "local_stack": {"seq": seq},
        "compress_images": True,
        "image_codecs": codecs,
    }
