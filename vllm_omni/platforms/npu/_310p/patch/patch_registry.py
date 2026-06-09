# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Register lazy 310P module patches."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from typing import Any, Protocol

from vllm.logger import init_logger

from vllm_omni.platforms.npu._310p.patch import (
    patch_qwen3_tts_code_predictor,
    patch_qwen3_tts_prompt_builder,
    patch_qwen3_tts_talker,
    patch_qwen3_tts_worker,
)

logger = init_logger(__name__)


class ModulePatch(Protocol):
    TARGET_MODULE: str

    def is_ready(self, module: Any) -> bool: ...

    def apply(self, module: Any) -> None: ...


_PATCHES: dict[str, ModulePatch] = {
    patch_qwen3_tts_code_predictor.TARGET_MODULE: patch_qwen3_tts_code_predictor,
    patch_qwen3_tts_prompt_builder.TARGET_MODULE: patch_qwen3_tts_prompt_builder,
    patch_qwen3_tts_talker.TARGET_MODULE: patch_qwen3_tts_talker,
    patch_qwen3_tts_worker.TARGET_MODULE: patch_qwen3_tts_worker,
}

_HOOK_INSTALLED = False
_PATCHED_MODULES: set[str] = set()


def _patch_loaded_module(fullname: str, module: Any) -> None:
    if fullname in _PATCHED_MODULES:
        return
    patch = _PATCHES.get(fullname)
    if patch is None:
        return
    patch.apply(module)
    _PATCHED_MODULES.add(fullname)
    logger.debug("Applied 310P patch for %s", fullname)


def _module_ready(fullname: str, module: Any) -> bool:
    patch = _PATCHES.get(fullname)
    return patch is not None and patch.is_ready(module)


class _PatchLoader(importlib.abc.Loader):
    def __init__(self, fullname: str, loader: importlib.abc.Loader) -> None:
        self.fullname = fullname
        self.loader = loader

    def create_module(self, spec):
        create_module = getattr(self.loader, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module) -> None:
        self.loader.exec_module(module)
        _patch_loaded_module(self.fullname, module)


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path, target=None):
        if fullname not in _PATCHES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        if not isinstance(spec.loader, _PatchLoader):
            spec.loader = _PatchLoader(fullname, spec.loader)
        return spec


def apply_qwen3_tts_patches() -> None:
    global _HOOK_INSTALLED

    # Platform construction happens while current_omni_platform is still being
    # resolved, so register lazy module patches instead of importing model code.
    if not _HOOK_INSTALLED:
        sys.meta_path.insert(0, _PatchFinder())
        _HOOK_INSTALLED = True

    for fullname in _PATCHES:
        module = sys.modules.get(fullname)
        if module is not None and _module_ready(fullname, module):
            _patch_loaded_module(fullname, module)
