# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Lazy model-runner class attributes for Omni workers."""

from vllm.utils.import_utils import resolve_obj_by_qualname


class LazyModelRunnerCls:
    """Class attribute that resolves a model runner class on first access.

    The CUDA workers pick the Model Runner V1 or V2 runner at ``init_device``
    time. Importing both at module import ties the MRv2 path to vLLM's Model
    Runner V1 modules, which upstream vLLM removed (vllm-project/vllm#63373),
    so each runner is imported only when it is used.
    """

    def __init__(self, qualname: str) -> None:
        self.qualname = qualname
        self._cls: type | None = None

    def __get__(self, obj: object, owner: type | None = None) -> type:
        if self._cls is None:
            self._cls = resolve_obj_by_qualname(self.qualname)
        return self._cls
