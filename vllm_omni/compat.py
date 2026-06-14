# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import inspect
from typing import Any


def make_filtered_namedtuple(cls, known_extra_fields: set[str] | None = None, **kwargs: Any):
    fields = set(cls._fields)
    unknown = set(kwargs) - fields - (known_extra_fields or set())
    values = {key: value for key, value in kwargs.items() if key in fields}
    return cls(**values), unknown


def make_filtered_call(cls, known_extra_fields: set[str] | None = None, **kwargs: Any):
    params = set(inspect.signature(cls).parameters)
    unknown = set(kwargs) - params - (known_extra_fields or set())
    values = {key: value for key, value in kwargs.items() if key in params}
    return cls(**values), unknown
