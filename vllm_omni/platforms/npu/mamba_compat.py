# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ascend Mamba allocator compatibility, installed before engine construction."""

from typing import Any


def _accepts_prefill_end(method: Any) -> bool:
    """Whether ``method`` can take vLLM's ``prefill_end`` keyword."""
    if getattr(method, "_omni_accepts_prefill_end", False):
        return True
    import inspect

    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False
    return "prefill_end" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def _install_prefill_end_shim(manager_cls: Any) -> bool:
    """Drop ``prefill_end`` only when ``manager_cls`` would reject it.

    Returns whether a wrapper was installed. Current vLLM-Ascend accepts the
    argument and forwards it; wrapping that method would change the block
    count. The v0.30 signature raises on the keyword, so it still needs the shim.
    """
    current = manager_cls.get_num_blocks_to_allocate
    if _accepts_prefill_end(current):
        return False

    def get_num_blocks_to_allocate(self: Any, *args: Any, prefill_end: int = 0, **kwargs: Any) -> int:
        del prefill_end
        return current(self, *args, **kwargs)

    get_num_blocks_to_allocate._omni_accepts_prefill_end = True  # type: ignore[attr-defined]
    manager_cls.get_num_blocks_to_allocate = get_num_blocks_to_allocate  # type: ignore[method-assign]
    return True


def _install_ascend_mamba_prefill_end_compat() -> None:
    """Let an older Ascend mamba allocator accept vLLM's ``prefill_end``.

    vLLM passes ``prefill_end`` into every cache manager. Ascend's override
    before that argument rejects the call, so the first Drama 3 schedule dies
    before any forward. Builds that already accept the argument are left alone.
    Missing imports are optional for compatibility across Ascend versions.
    Other initialization failures intentionally propagate on NPU, rather than
    silently continuing with a potentially broken allocator. Other platforms
    do not invoke this installer.
    """
    try:
        from vllm_ascend.patch.platform.patch_mamba_manager import AscendMambaManager
    except ImportError:
        return
    _install_prefill_end_shim(AscendMambaManager)
