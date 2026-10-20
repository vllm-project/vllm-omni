# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Strict parsing for the boolean knobs of the MiniCPM-o 4.5 stage config."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def parse_bool_flag(value: Any, key: str) -> bool:
    """Parse a stage-config boolean strictly, so a typo fails at startup.

    ``bool("false")`` is ``True``, which would silently keep a feature on for a
    quoted YAML value or a JSON/CLI stage override -- exactly the case these
    switches exist to triage. Accepts bools, 0/1 and the ``"true"``/``"false"``/
    ``"1"``/``"0"`` strings; anything else raises.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in ("true", "1"):
            return True
        if token in ("false", "0"):
            return False
    raise ValueError(f"{key} must be a boolean (true/false/1/0), got {value!r}")


def read_bool_flag(extra: Mapping[str, Any], key: str, default: bool) -> bool:
    """Read ``extra[key]`` strictly, keeping ``default`` when the key is unset.

    An explicit ``null`` (``key: ~`` in a YAML override) means "not set" and
    falls back to ``default`` rather than flipping the feature off, so dropping
    a key from a stage config never changes behaviour silently.
    """
    value = extra.get(key, default)
    if value is None:
        return default
    return parse_bool_flag(value, key)
