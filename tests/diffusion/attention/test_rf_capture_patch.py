# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unit tests for the MindIE-SD rf_v2 capture patch.

The patch rewrites a third-party function, so the contract that matters is the
conservative one: when MindIE-SD is absent or its internals do not look like the
version this was written against, leave everything alone and report failure --
never patch half-way, because a wrong permutation would corrupt output silently.
"""

import sys
from types import ModuleType

import pytest

from vllm_omni.diffusion.attention import rf_capture_patch

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]

_RF = "mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2"
_ENTRY = "mindiesd.layers.flash_attn.sparse_flash_attn"


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    monkeypatch.setattr(rf_capture_patch, "_PATCHED", False)
    monkeypatch.setattr(rf_capture_patch, "_FAILED", False)


def _install_fake_mindiesd(monkeypatch, *, with_required: bool):
    """Stub the mindiesd package tree.

    The stub packages carry ``__path__`` and are wired into their parents, so the
    import machinery hands back these modules instead of falling through to a
    real MindIE-SD install (the NPU containers have one).
    """
    tree = {}
    for name in ("mindiesd", "mindiesd.layers", "mindiesd.layers.flash_attn"):
        module = ModuleType(name)
        module.__path__ = []  # marks it as a package
        tree[name] = module

    rf_v2 = ModuleType(_RF)
    entry = ModuleType(_ENTRY)
    if with_required:
        setattr(rf_v2, "_multi_span_plan", lambda *a, **k: ([], set(), set()))
        setattr(rf_v2, "_span_rearranged_indices", lambda *a, **k: None)
        setattr(rf_v2, "get_mask_index", lambda *a, **k: None)
        setattr(rf_v2, "_multi_span_permutation", lambda *a, **k: None)
        setattr(rf_v2, "get_multi_span_blockwise_mask", lambda *a, **k: None)
        setattr(entry, "get_multi_span_blockwise_mask", lambda *a, **k: None)
    tree[_RF] = rf_v2
    tree[_ENTRY] = entry

    for name, module in tree.items():
        monkeypatch.setitem(sys.modules, name, module)
    setattr(tree["mindiesd"], "layers", tree["mindiesd.layers"])
    setattr(tree["mindiesd.layers"], "flash_attn", tree["mindiesd.layers.flash_attn"])
    setattr(tree["mindiesd.layers.flash_attn"], "sparse_flash_attn_rf_v2", rf_v2)
    setattr(tree["mindiesd.layers.flash_attn"], "sparse_flash_attn", entry)
    return rf_v2, entry


def test_missing_mindiesd_skips_cleanly(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name.startswith("mindiesd"):
            raise ImportError("no mindiesd")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    for name in list(sys.modules):
        if name.startswith("mindiesd"):
            monkeypatch.delitem(sys.modules, name, raising=False)

    assert rf_capture_patch.ensure_rf_capture_patch() is False
    assert rf_capture_patch._FAILED is True
    assert rf_capture_patch._PATCHED is False


def test_unexpected_mindiesd_build_is_left_untouched(monkeypatch):
    rf_v2, entry = _install_fake_mindiesd(monkeypatch, with_required=False)
    entry_mask = entry.get_multi_span_blockwise_mask if hasattr(entry, "get_multi_span_blockwise_mask") else None

    assert rf_capture_patch.ensure_rf_capture_patch() is False
    assert rf_capture_patch._FAILED is True
    assert rf_capture_patch._PATCHED is False
    # A build we do not recognise must be left exactly as it was: a half-applied
    # permutation rewrite would corrupt output rather than fail loudly.
    assert not hasattr(rf_v2, "_multi_span_permutation")
    assert getattr(entry, "get_multi_span_blockwise_mask", None) is entry_mask


def test_patch_replaces_both_namespaces_and_is_idempotent(monkeypatch):
    rf_v2, entry = _install_fake_mindiesd(monkeypatch, with_required=True)
    original_perm = rf_v2._multi_span_permutation

    assert rf_capture_patch.ensure_rf_capture_patch() is True
    assert rf_v2._multi_span_permutation is not original_perm
    # The caller binds this name with `from ... import`, so its reference has to
    # move too or the patch silently does nothing.
    assert entry.get_multi_span_blockwise_mask is rf_v2.get_multi_span_blockwise_mask

    patched_perm = rf_v2._multi_span_permutation
    assert rf_capture_patch.ensure_rf_capture_patch() is True
    assert rf_v2._multi_span_permutation is patched_perm
