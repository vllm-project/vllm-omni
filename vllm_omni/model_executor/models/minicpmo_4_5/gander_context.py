# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Gander-owned unit selection and protected-prefix reconstruction policy.

The engine consumes the plan and owns physical requests/KV. No raw KV tensor
is manipulated here. Completed assistant units are retained as teacher-forced
history; unfinished response output is deliberately invalidated on replacement.
"""

from __future__ import annotations

from copy import deepcopy

from vllm_omni.engine.duplex.contracts import DuplexContextPlan, DuplexContextUnit, DuplexInputMode

DEFAULT_MAX_UNITS = 128
DEFAULT_RETAIN_UNITS = 96
MAX_PINNED_UNITS = 16


def metadata(prompt):
    return prompt.get("model_intermediate_buffer", {}).get("duplex", {})


def unit_id(prompt):
    d = metadata(prompt)
    return str(d.get("gander_unit_id") or f"u{d.get('epoch', 0)}-{d.get('seq', 0)}")


def window_config(runtime):
    raw = runtime.get("gander_history", {})
    if not isinstance(raw, dict):
        raise ValueError("gander_history must be an object")
    maximum = raw.get("max_units", DEFAULT_MAX_UNITS)
    retain = raw.get("retain_units", min(DEFAULT_RETAIN_UNITS, maximum - 1) if type(maximum) is int else 0)
    if type(maximum) is not int or not 2 <= maximum <= 128:
        raise ValueError("gander_history.max_units must be between 2 and 128")
    if type(retain) is not int or not 1 <= retain < maximum:
        raise ValueError("gander_history.retain_units must be smaller than max_units")
    return maximum, retain


def describe(prompt):
    d = metadata(prompt)
    p = d.get("payload", {})
    return {
        "unit_id": unit_id(prompt),
        "kind": "event" if p.get("gander_control") else "audio",
        "pinned": bool(d.get("gander_pinned")),
        "completed": bool(d.get("gander_output_ids")),
        "input_tokens": len(prompt.get("prompt_token_ids", [])),
    }


def should_rollover(prompts, runtime):
    maximum, _ = window_config(runtime)
    return len(prompts) >= maximum


def select_units(prompts, runtime, *, compact=False):
    """Keep protected units plus a contiguous recent suffix, in original order."""
    units = [deepcopy(dict(p)) for p in prompts if not metadata(p).get("payload", {}).get("gander_prefix_seed")]
    maximum, retain = window_config(runtime)
    keep = retain if compact or len(units) > maximum else maximum
    if compact and units:
        # Byte/token pressure can arrive before the unit trigger. Retaining
        # the entire shorter journal would rebuild again on every next append.
        keep = min(keep, max(1, len(units) * 3 // 4))
    pins = {unit_id(p) for p in units if metadata(p).get("gander_pinned")}
    if len(pins) > min(MAX_PINNED_UNITS, maximum - 1):
        raise ValueError("too many pinned Gander history units")
    recent = {unit_id(p) for p in units[-max(0, keep - len(pins)) :]} if keep > len(pins) else set()
    return [p for p in units if unit_id(p) in pins | recent]


def make_plan(*, prompts, runtime_config, session_config, request_id, fence, context):
    from .duplex.runtime import build_duplex_data_plane_prompt

    original = list(prompts)
    units = select_units(original, runtime_config)
    edits = context.get("edits", [])
    if not isinstance(edits, list) or len(edits) > 32:
        raise ValueError("context edits must be a list of at most 32 operations")

    def index(key):
        for i, p in enumerate(units):
            if unit_id(p) == key:
                return i
        raise ValueError(f"unknown history unit: {key}")

    for edit in edits:
        if not isinstance(edit, dict):
            raise ValueError("each context edit must be an object")
        op = edit.get("op")
        if op == "insert":
            payload = deepcopy(edit.get("payload"))
            if not isinstance(payload, dict) or not payload.get("gander_control"):
                raise ValueError("historical insertion requires a validated external event")
            identifier = edit.get("unit_id")
            if not isinstance(identifier, str) or not identifier or len(identifier) > 128:
                raise ValueError("insert requires a bounded unit_id")
            if any(unit_id(p) == identifier for p in units):
                raise ValueError("duplicate history unit_id")
            p = {
                "model_intermediate_buffer": {
                    "duplex": {
                        "payload": payload,
                        "gander_unit_id": identifier,
                        "gander_output_ids": [],
                        "turn_id": fence.turn_id,
                    }
                }
            }
            target = len(units) if edit.get("before") is None else index(edit["before"])
            units.insert(target, p)
        elif op in {"delete", "move", "pin", "unpin"}:
            at = index(edit.get("unit_id"))
            p = units[at]
            if op == "delete":
                if metadata(p).get("gander_pinned"):
                    raise ValueError("cannot delete a pinned unit; unpin explicitly first")
                units.pop(at)
            elif op == "move":
                if edit.get("before") == unit_id(p):
                    continue
                units.pop(at)
                target = len(units) if edit.get("before") is None else index(edit["before"])
                units.insert(target, p)
            else:
                metadata(p)["gander_pinned"] = op == "pin"
        else:
            raise ValueError(f"unsupported history edit: {op}")
    ids = [unit_id(p) for p in units]
    if len(set(ids)) != len(ids):
        raise ValueError("history contains duplicate unit identities")
    units = select_units(units, runtime_config, compact=context.get("reason") == "context_rollover")
    retained_ids = tuple(unit_id(p) for p in units)
    # The system/tools/reference/slate prefix is never a removable history unit.
    if not units:
        units = [
            {
                "model_intermediate_buffer": {
                    "duplex": {
                        "gander_unit_id": "prefix-seed",
                        "payload": {"gander_control": True, "gander_prefix_seed": True, "token_ids": []},
                    }
                }
            }
        ]
    rebuilt = []
    for seq, old in enumerate(units, 1):
        d = metadata(old)
        payload = deepcopy(d.get("payload", {}))
        payload["gander_replay"] = True
        payload["force_listen"] = True
        payload["context_version"] = int(runtime_config.get("gander_context_version", 0))
        outputs = list(d.get("gander_output_ids", []))
        if (
            context.get("discard_turn_id") is not None
            and not d.get("recovery_replay")
            and d.get("turn_id") == context["discard_turn_id"]
        ):
            outputs = []
        payload["gander_replay_output_ids"] = outputs
        p = build_duplex_data_plane_prompt(
            request_id=request_id,
            fence=fence,
            session_config=session_config,
            runtime_config=runtime_config,
            seq=seq,
            turn_seq=seq,
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
            payload=payload,
            final=False,
        )
        new = metadata(p)
        new.update(
            gander_unit_id=unit_id(old),
            gander_pinned=bool(d.get("gander_pinned")),
            gander_output_ids=outputs,
            recovery_replay=True,
        )
        rebuilt.append(DuplexContextUnit(unit_id=unit_id(old), prompt=p))
    return DuplexContextPlan(
        units=tuple(rebuilt),
        retained_unit_ids=retained_ids,
        dropped_unit_ids=tuple(unit_id(p) for p in original if unit_id(p) not in retained_ids),
    )
