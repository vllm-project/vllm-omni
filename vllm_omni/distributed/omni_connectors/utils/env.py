# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import re
from typing import Any

AUTO_HOST_VALUES = {"", "auto", "*", "0.0.0.0", "::"}
AUTO_DEVICE_VALUES = {"", "auto"}

_UNEXPANDED_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)|%([^%]+)%")


class EnvVarExpansionError(ValueError):
    pass


def _unexpanded_env_vars(value: str) -> list[str]:
    names: list[str] = []
    for match in _UNEXPANDED_ENV_VAR_PATTERN.finditer(value):
        name = next(group for group in match.groups() if group)
        if name not in names:
            names.append(name)
    return names


def expand_env_value(value: Any, *, strict: bool = False, field_name: str = "value") -> Any:
    if not isinstance(value, str):
        return value

    expanded = os.path.expandvars(value)
    if strict:
        unresolved = _unexpanded_env_vars(expanded)
        if unresolved:
            names = ", ".join(unresolved)
            raise EnvVarExpansionError(
                f"{field_name} references unset environment variable(s): {names}. Original value: {value!r}"
            )
    return expanded


def expand_env_int(value: Any, field_name: str) -> int:
    expanded = expand_env_value(value, strict=True, field_name=field_name)
    try:
        return int(expanded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer after environment expansion, got {expanded!r}") from exc
