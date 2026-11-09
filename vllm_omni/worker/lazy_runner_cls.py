# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Lazy model-runner class attributes for Omni workers."""

import importlib
from typing import Any


class LazyModelRunnerCls:
    """Class attribute that imports a model runner class on first access.

    The GPU workers choose between the Model Runner V1 and V2 runners at
    ``init_device`` time. Importing both at module import time ties the MRv2
    path to vLLM's Model Runner V1 modules, which upstream vLLM removed
    (vllm-project/vllm#63373). Resolving the runner class lazily keeps
    importing a worker free of the runner it does not use.
    """

    def __init__(self, qualname: str) -> None:
        self.qualname = qualname
        self._cls: type[Any] | None = None

    def __get__(self, obj: Any, owner: type[Any] | None = None) -> type[Any]:
        if self._cls is None:
            module_name, _, attr = self.qualname.rpartition(".")
            self._cls = getattr(importlib.import_module(module_name), attr)
        return self._cls
