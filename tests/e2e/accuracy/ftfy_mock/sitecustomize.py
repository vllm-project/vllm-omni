"""
This sitecustomize.py is used to ensure the presence of an ftfy implementation.
ftfy is a text encoding sanitizer and it is implicitly required by diffusers' WanImageToVideoPipeline.
And this pipeline is used in several accuracy tests.

A sitecustomize.py is a Python mechanism to inject code into the interpreter at startup.
To use this sitecustomize.py, include its directory in the PYTHONPATH environment variable.

The sitecustomize approach must be used because diffusers' Wan pipelines may be imported in a subprocess.
For example, when launching with vLLM-Omni's diffusers backend in online serving mode.
`subprocess.Popen` (used by OmniServer test helper) does not inherit any mocks from the parent process.
This is the only way to inject mocks into the subprocess.

.. note::
   If installing the real ftfy library, the relevant tests may fail the similarity assertion.
   Because vLLM-Omni doesn't use ftfy to preprocess the text input.
   Hence, we must hack diffusers' Wan pipelines to not use ftfy either.
"""

import importlib.abc
import sys
from types import ModuleType
from typing import Any

_TARGET_MODULE = "diffusers.pipelines.wan.pipeline_wan_i2v"


class _IdentityFtfy:
    @staticmethod
    def fix_text(text: str) -> str:
        return text


def _patch_wan_i2v_module(module: ModuleType) -> None:
    # If real ftfy was imported by diffusers, preserve it.
    if not hasattr(module, "ftfy"):
        module.ftfy = _IdentityFtfy()  # type: ignore[attr-defined]
        print("ftfy (text encoding sanitizer) is not installed. Using mock ftfy implementation (identity function)")
    else:
        print("ftfy (text encoding sanitizer) is installed. Using actual ftfy implementation.")


class _PatchWanI2VLoader(importlib.abc.Loader):
    def __init__(self, wrapped_loader: importlib.abc.Loader) -> None:
        self._wrapped_loader = wrapped_loader

    def create_module(self, spec: Any) -> ModuleType | None:
        create_module = getattr(self._wrapped_loader, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._wrapped_loader.exec_module(module)
        _patch_wan_i2v_module(module)


class _PatchWanI2VFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:
        if fullname != _TARGET_MODULE:
            return None
        # Ask the remaining finders for the real module spec. Skip this finder
        # to avoid infinite recursion.
        for finder in sys.meta_path:
            if finder is self:
                continue
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            spec = find_spec(fullname, path, target)
            if spec is None:
                continue
            if spec.loader is not None and not isinstance(spec.loader, _PatchWanI2VLoader):
                spec.loader = _PatchWanI2VLoader(spec.loader)
            return spec
        return None


def _install_hook() -> None:
    existing_module = sys.modules.get(_TARGET_MODULE)
    if existing_module is not None:
        _patch_wan_i2v_module(existing_module)
        return
    if not any(isinstance(finder, _PatchWanI2VFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _PatchWanI2VFinder())


print("Installing ftfy mock hook", flush=True)
_install_hook()
print("ftfy mock hook installed", flush=True)
