# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the NVFP4 W4A4 weight_scale NaN clamp installed by
``vllm_omni.patch``.

ModelOpt 0.44's FP32->FP8 E4M3 cast of per-block weight scales can emit
literal NaN bytes (E4M3 encoding 0x7F / 0xFF) when the pre-cast scale
rounds above the FP8 max of 448 after the global-scale division. A
single NaN byte in any per-block weight_scale collapses the FlashInfer
FP4 GEMM output to NaN and the served model emits `!!!!`. On the legacy
pin the patch overrides ``ModelOptNvFp4LinearMethod.process_weights_after_loading``
to scan + clamp those bytes at load time.

vLLM #49381 replaced that class with the generic ``ModelOptLinearMethod``
(and the ``LinearMethodCls`` attributes) and #52501 made a NaN
weight_scale a hard error there ("... was never loaded (still NaN)"), so on
that pin ``vllm_omni.patch`` self-extinguishes: it installs nothing and
upstream's load-time RuntimeError stays observable. These tests cover both
pins:

- the NaN bytes get clamped to the FP8 E4M3 max byte (0x7E) [helper]
- a finite weight_scale is left untouched (no spurious writes) [helper]
- the install asserts itself so silent fallback is impossible [either pin]
- the env-var escape hatch path is reachable [either pin]
- on the redesigned pin nothing is installed on ``ModelOptLinearMethod``
  and upstream's unloaded-scale RuntimeError is still raised [redesigned]

No GPU is required — the tests construct an in-memory FP8 ``weight_scale``
tensor with planted NaN bytes and call either the patched helper or the
upstream PWAL directly.
"""

import importlib

import pytest
import torch

# Importing vllm_omni runs vllm_omni.patch, which installs the override
# we are testing. The import must come before any direct reference to
# the clamp target below.
import vllm_omni  # noqa: F401

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


FP8_NAN_BYTES = (0x7F, 0xFF)
FP8_MAX_BYTE = 0x7E  # encoding of finfo(float8_e4m3fn).max

_MODELOPT_MODULE = "vllm.model_executor.layers.quantization.modelopt"
_LEGACY_NVFP4_LINEAR_METHOD = "ModelOptNvFp4LinearMethod"
_GENERIC_LINEAR_METHOD = "ModelOptLinearMethod"


def _resolve_target(name: str):
    """Return ``vllm.modelopt.<name>`` or None when upstream never had it.

    Imports the module eagerly (a missing module is a hard upstream breakage
    and must fail loudly); only the *attribute* lookup is tolerant, so
    "upstream redesigned the class away" is distinguishable from "the module
    moved" without swallowing real errors.
    """
    modelopt = importlib.import_module(_MODELOPT_MODULE)
    return getattr(modelopt, name, None)


def _resolve_legacy_target():
    return _resolve_target(_LEGACY_NVFP4_LINEAR_METHOD)


def _resolve_generic_target():
    return _resolve_target(_GENERIC_LINEAR_METHOD)


def _pwfn_source(cls) -> str:
    """Source file of ``cls.process_weights_after_loading``."""
    fn = cls.__dict__.get("process_weights_after_loading") or cls.process_weights_after_loading
    return getattr(fn, "__code__", None) and fn.__code__.co_filename or ""


def _last_stdout_line(stdout: str) -> str:
    """Last non-empty stdout line of a subprocess.

    Importing ``vllm_omni`` runs ``vllm_omni.patch``, whose INFO logs land on
    stdout, so a sentinel printed by the child is the *last* line, never the
    first (the previous ``stdout.strip().startswith(...)`` form silently broke
    as soon as the patch module logged anything).
    """
    lines = [line.strip() for line in stdout.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _fp8_tensor_with_nan_bytes(num_blocks: int, nan_indices: list[int]) -> torch.Tensor:
    """Return an FP8 E4M3 tensor where every position is the FP8 max byte
    except ``nan_indices``, which carry NaN bytes (alternating 0x7F/0xFF)."""
    raw = bytearray([FP8_MAX_BYTE] * num_blocks)
    for i, idx in enumerate(nan_indices):
        raw[idx] = FP8_NAN_BYTES[i % len(FP8_NAN_BYTES)]
    # Materialize on a writable storage so masked_fill_ on the uint8 view
    # can write back in-place. ``frombuffer`` produces a read-only view
    # that PyTorch warns about on first use.
    return torch.tensor(list(raw), dtype=torch.uint8).view(torch.float8_e4m3fn)


@pytest.fixture
def clamp_fn():
    """Return the top-level NaN-clamp helper exposed by vllm_omni.patch.

    Skipped when vllm_omni.patch did not expose the helper (e.g. an
    incompatible vllm_omni install).
    """
    try:
        from vllm_omni.patch import _clamp_nvfp4_weight_scale_nans
    except ImportError:
        pytest.skip("vllm_omni.patch._clamp_nvfp4_weight_scale_nans not exposed")
    return _clamp_nvfp4_weight_scale_nans


def test_clamps_nan_bytes_in_weight_scale(clamp_fn):
    weight_scale = _fp8_tensor_with_nan_bytes(num_blocks=8, nan_indices=[2, 5])
    assert torch.isnan(weight_scale).sum().item() == 2

    layer = torch.nn.Module()
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
    n_clamped = clamp_fn(layer)

    assert n_clamped == 2
    assert torch.isnan(layer.weight_scale).sum().item() == 0, (
        "NaN bytes in weight_scale should have been clamped to FP8 E4M3 max"
    )
    # Every byte should now be the FP8 max byte.
    bytes_after = layer.weight_scale.data.view(torch.uint8).tolist()
    assert bytes_after == [FP8_MAX_BYTE] * 8


def test_finite_weight_scale_untouched(clamp_fn):
    # All-finite weight_scale should leave bytes exactly as written.
    raw = bytes([0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x7E])
    weight_scale = torch.frombuffer(raw, dtype=torch.uint8).clone().view(torch.float8_e4m3fn)
    assert torch.isnan(weight_scale).sum().item() == 0
    before = weight_scale.data.view(torch.uint8).clone()

    layer = torch.nn.Module()
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
    n_clamped = clamp_fn(layer)

    assert n_clamped == 0
    after = layer.weight_scale.data.view(torch.uint8)
    assert torch.equal(before, after), "clamp must not modify a finite weight_scale tensor"


def test_clamp_is_noop_when_weight_scale_missing(clamp_fn):
    # Defensive guard: subclasses without weight_scale should not crash.
    layer = torch.nn.Module()
    n_clamped = clamp_fn(layer)
    assert n_clamped == 0


def test_clamps_non_contiguous_weight_scale(clamp_fn):
    # A non-contiguous weight_scale would make `.view(torch.uint8)` raise.
    # The clamp must materialize a contiguous copy, write it back to the
    # Parameter, and still land the in-place clamp where the kernel reads.
    # NaN bytes at even indices so they survive the stride-2 slice below
    # (base[::2] keeps original indices 0, 2, 4, ...).
    base = _fp8_tensor_with_nan_bytes(num_blocks=16, nan_indices=[4, 12])
    # Slice every other element -> stride 2 -> non-contiguous view.
    non_contig = base[::2]
    assert not non_contig.is_contiguous()
    assert torch.isnan(non_contig).sum().item() >= 1

    layer = torch.nn.Module()
    layer.weight_scale = torch.nn.Parameter(non_contig, requires_grad=False)
    n_clamped = clamp_fn(layer)

    assert n_clamped >= 1
    assert layer.weight_scale.is_contiguous(), "weight_scale should be contiguous after clamp"
    assert torch.isnan(layer.weight_scale).sum().item() == 0, (
        "NaN bytes in a non-contiguous weight_scale should have been clamped"
    )


def test_install_is_self_asserting():
    # If the install assert is removed, a future regression could let the
    # override silently drop. There are three legitimate end states:
    #   (a) upstream vLLM already handles NaN weight_scale itself — either its
    #       own in-place clamp, or (on the redesigned generic
    #       ModelOptLinearMethod) a hard rejection — and vllm_omni.patch
    #       intentionally skipped installing the override,
    #   (b) upstream is unpatched (legacy per-format class) and our wrapper is
    #       installed,
    #   (c) the env-var escape hatch is set OR every modelopt target is missing,
    #       so nothing is installed by design — nothing to assert.
    legacy = _resolve_legacy_target()
    if legacy is None and _resolve_generic_target() is None:
        pytest.skip("no ModelOpt linear method exposed by this vLLM build")

    from vllm_omni.patch import _already_patched_upstream, _clamp_installed

    if _already_patched_upstream:
        # Self-extinguish path: presence of our marker here would mean the
        # detection failed and we wrapped a class we must not touch.
        assert not _clamp_installed, "clamp installed although upstream handles NaN weight_scale"
        if legacy is not None:
            name = getattr(legacy.process_weights_after_loading, "__qualname__", "") or getattr(
                legacy.process_weights_after_loading, "__name__", ""
            )
            assert "_patched_nvfp4_pwal" not in name, (
                f"vllm_omni override installed although upstream handles NaN weight_scale "
                f"itself ({name!r}); the self-extinguish detection in vllm_omni.patch needs review"
            )
    elif _clamp_installed:
        assert legacy is not None, "clamp reported installed but the legacy ModelOptNvFp4LinearMethod does not exist"
        current = legacy.process_weights_after_loading
        name = getattr(current, "__qualname__", "") or getattr(current, "__name__", "")
        assert "patched_nvfp4_pwal" in name or "_patched_nvfp4_pwal" in name, (
            f"NVFP4 NaN-clamp PWAL override does not appear installed (saw {name!r}); "
            "another plugin may have re-patched ModelOptNvFp4LinearMethod"
        )
    else:
        pytest.skip("NaN-clamp not installed by design (env-var escape hatch or missing target)")


def test_no_override_on_redesigned_upstream():
    """On the redesigned ModelOpt linear path (#49381) nothing may be installed.

    #52501 made any NaN weight_scale a hard load-time error there, so a clamp
    would mask it and silently serve a partially loaded model. Assert both that
    we installed nothing and that we did not touch the class at all.
    """
    generic = _resolve_generic_target()
    if generic is None:
        pytest.skip("ModelOptLinearMethod not available (legacy pin)")
    if _resolve_legacy_target() is not None:
        pytest.skip("legacy pin: the per-format override path applies instead")

    from vllm_omni.patch import _already_patched_upstream, _clamp_installed

    assert not _clamp_installed, (
        "the NaN clamp must not be installed on the redesigned generic ModelOptLinearMethod: "
        "it would mask upstream's unloaded-weight_scale RuntimeError"
    )
    assert _already_patched_upstream, (
        "vllm_omni.patch must record that upstream itself rejects NaN weight_scale on this pin"
    )
    assert not hasattr(generic.process_weights_after_loading, "_vllm_omni_wrapped_pwal"), (
        "our wrapper is present on ModelOptLinearMethod.process_weights_after_loading"
    )
    # Stronger identity check than the marker: a monkeypatched wrapper would be
    # defined in vllm_omni/patch.py, so the class attribute must still come from
    # vLLM itself.
    assert "vllm_omni" not in _pwfn_source(generic), (
        f"ModelOptLinearMethod.process_weights_after_loading was replaced by {_pwfn_source(generic)}"
    )


def _nvfp4_w4a4_method():
    """Build the generic NVFP4 W4A4 linear method exactly as upstream does."""
    from types import SimpleNamespace

    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptLinearMethod,
        resolve,
    )

    spec, ctx, format_scheme = resolve("NVFP4", SimpleNamespace(group_size=16), "test")
    return ModelOptLinearMethod(spec, ctx, format_scheme)


def test_upstream_nan_sentinel_check_still_reachable():
    """Upstream's unloaded-weight_scale guard must survive our import.

    Drives the *public* entry point (``ModelOptLinearMethod``), not just the
    scheme, so a future re-install of the override on that class would be
    caught: a wrapper that clamped first would swallow the RuntimeError.
    """
    if _resolve_generic_target() is None:
        pytest.skip("ModelOptLinearMethod not available (legacy pin)")

    method = _nvfp4_w4a4_method()
    layer = torch.nn.Module()
    layer.weight_scale = torch.nn.Parameter(
        _fp8_tensor_with_nan_bytes(num_blocks=8, nan_indices=list(range(8))),
        requires_grad=False,
    )

    with pytest.raises(RuntimeError, match="never loaded"):
        method.process_weights_after_loading(layer)


def test_reload_does_not_nest_wrapper():
    """Reloading vllm_omni.patch must not wrap our own wrapper a second time.

    On reload the class attribute already holds our wrapper; capturing it as
    the "original" and re-wrapping would run the clamp twice per load. The
    patch recovers the genuine upstream method via a sentinel, so after a
    reload the installed wrapper's ``_vllm_omni_wrapped_pwal`` must point at
    the real upstream method (which itself carries no sentinel), not at
    another wrapper. Runs in a subprocess: importlib.reload mutates global
    interpreter state we don't want leaking into other tests."""
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import importlib
        import vllm_omni.patch as p
        modelopt = importlib.import_module("vllm.model_executor.layers.quantization.modelopt")
        ModelOptNvFp4LinearMethod = getattr(modelopt, "ModelOptNvFp4LinearMethod", None)
        if ModelOptNvFp4LinearMethod is None:
            print("SKIP: legacy modelopt target not available")
            raise SystemExit(0)
        if not getattr(p, "_clamp_installed", False):
            print("SKIP: clamp not installed (upstream handles NaN, or escape hatch)")
            raise SystemExit(0)
        importlib.reload(p)
        fn = ModelOptNvFp4LinearMethod.process_weights_after_loading
        wrapped = getattr(fn, "_vllm_omni_wrapped_pwal", None)
        # The recovered upstream must NOT itself be one of our wrappers.
        nested = hasattr(wrapped, "_vllm_omni_wrapped_pwal")
        print("NESTED" if nested else "FLAT")
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    # The subprocess also emits vLLM's own INFO log lines on stdout (importing
    # vllm_omni runs vllm_omni.patch), so the sentinel is the LAST stdout line,
    # not the first.
    last_line = _last_stdout_line(result.stdout)
    if last_line.startswith("SKIP:"):
        pytest.skip(last_line)
    assert "FLAT" in result.stdout, (
        f"reload nested the NVFP4 clamp wrapper — clamp would run twice per load.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


@pytest.mark.parametrize(
    ("env_value", "acceptable"),
    [
        # Truthy values activate the escape hatch: the import short-circuits
        # before the upstream-detection runs, so nothing is installed.
        ("1", frozenset({"NOT_INSTALLED"})),
        ("true", frozenset({"NOT_INSTALLED"})),
        # Falsy values mean "do NOT use the escape hatch" — the repo-wide
        # bool-env idiom in vllm_omni.patch guards against a naive
        # `if os.environ.get(...)` wrongly skipping on "0"/"false". The clamp
        # must therefore be in force, which is satisfied EITHER by our wrapper
        # being installed OR by upstream vLLM already carrying its own clamp
        # (self-extinguish, `_already_patched_upstream`), in which case skipping
        # our wrapper is the correct behavior. Accept both so this test keeps
        # passing once the upstream fix lands.
        ("0", frozenset({"INSTALLED", "UPSTREAM_PATCHED"})),
        ("false", frozenset({"INSTALLED", "UPSTREAM_PATCHED"})),
    ],
)
def test_env_var_escape_hatch_disables_install(env_value, acceptable):
    """Spawn a subprocess with VLLM_OMNI_SKIP_NVFP4_NAN_CLAMP set and verify
    the override install state matches the bool-env convention. The patch
    module runs at ``vllm_omni`` import time, so we can't toggle it via
    monkeypatch in the same process.

    The subprocess reports the authoritative state from vllm_omni.patch's
    module flags (the same source of truth as ``test_install_is_self_asserting``)
    rather than introspecting the method qualname — both because that is the
    real contract and because ``"INSTALLED"`` is a substring of
    ``"NOT_INSTALLED"`` and naive substring matching would be vacuous."""
    import os
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import importlib
        import vllm_omni  # noqa: F401
        import vllm_omni.patch as p
        modelopt = importlib.import_module("vllm.model_executor.layers.quantization.modelopt")
        has_target = any(
            getattr(modelopt, name, None) is not None
            for name in ("ModelOptNvFp4LinearMethod", "ModelOptLinearMethod")
        )
        if not has_target:
            print("SKIP: no modelopt linear method target available")
            raise SystemExit(0)
        if getattr(p, "_already_patched_upstream", False):
            # Upstream handles NaN weight_scale itself (its own clamp, or the
            # redesigned generic PWAL's hard rejection); vllm_omni.patch
            # deliberately skipped installing our wrapper (self-extinguish).
            print("UPSTREAM_PATCHED")
        elif getattr(p, "_clamp_installed", False):
            print("INSTALLED")
        else:
            print("NOT_INSTALLED")
        """
    )
    env = os.environ.copy()
    env["VLLM_OMNI_SKIP_NVFP4_NAN_CLAMP"] = env_value
    # Generous timeout: cold CI containers can take 30-50s just to import
    # vllm_omni → vllm → torch, and heavily loaded shared machines have
    # been observed taking >2min for the same import under disk I/O
    # contention. The subprocess does a single import + flag print, so 600s
    # only bounds pathological hangs, not normal runtime.
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600, env=env)
    state = _last_stdout_line(result.stdout)
    if state.startswith("SKIP:"):
        pytest.skip(state)
    # Exact token match on the final stdout line — not a substring test, since
    # "INSTALLED" is a substring of "NOT_INSTALLED".
    assert state in acceptable, (
        f"VLLM_OMNI_SKIP_NVFP4_NAN_CLAMP={env_value!r} expected one of {set(acceptable)!r} "
        f"but got {state!r}:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
