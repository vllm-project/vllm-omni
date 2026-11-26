# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for the NPU native rejection-sampler restore gate.

``vllm_omni/platforms/npu/native_rejection_sampler.py`` decides whether the
Triton rejection-sampler kernels are replaced with vLLM's torch-native
functions. The decision is pure logic - the SoC name reported by torch_npu and
whether the Talker multi-frame decode (the n-gram K-step) is armed - so it is
pinned here on a CPU host.

Both modules are loaded from source with a fake vLLM logger, and the VLLM
sampler import is pointed at a temporary module, so the test never imports a
real sampler module and never touches an Ascend device.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# The restore re-executes vllm's rejection-sampler module from source, so the
# fake leaf module points at a file with the three names the gate re-points.
_SAMPLER_SOURCE = """\
def apply_sampling_constraints(*args, **kwargs):
    return "constraints"


def expand_batch_to_tokens(*args, **kwargs):
    return "expand"


def rejection_sample(*args, **kwargs):
    return "sample"
"""


def _repo_root() -> Path:
    marker = Path("vllm_omni") / "platforms" / "npu"
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).is_dir():
            return parent
    raise FileNotFoundError(f"could not locate repo root containing {marker}")


def _load_source_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fake_vllm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> types.ModuleType:
    """Install a fake ``vllm`` package chain and return the sampler leaf."""
    source = tmp_path / "native_rejection_sampler.py"
    source.write_text(_SAMPLER_SOURCE)

    leaf = types.ModuleType("vllm.v1.sample.rejection_sampler")
    leaf.__file__ = str(source)
    sample_pkg = types.ModuleType("vllm.v1.sample")
    sample_pkg.rejection_sampler = leaf  # type: ignore[attr-defined]
    v1 = types.ModuleType("vllm.v1")
    v1.sample = sample_pkg  # type: ignore[attr-defined]
    logger = types.ModuleType("vllm.logger")
    logger.init_logger = lambda _name: SimpleNamespace(  # type: ignore[attr-defined]
        info=lambda *_args, **_kwargs: None,
        warning=lambda *_args, **_kwargs: None,
    )
    vllm = types.ModuleType("vllm")
    vllm.logger = logger  # type: ignore[attr-defined]
    vllm.v1 = v1  # type: ignore[attr-defined]

    for name, module in (
        ("vllm", vllm),
        ("vllm.logger", logger),
        ("vllm.v1", v1),
        ("vllm.v1.sample", sample_pkg),
        ("vllm.v1.sample.rejection_sampler", leaf),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return leaf


def _load_gate(monkeypatch: pytest.MonkeyPatch):
    """Load the warmup patch and the restore gate from source."""
    root = _repo_root()
    warmup = _load_source_module(
        "_npu_ascend_warmup_patch_under_test",
        root / "vllm_omni" / "platforms" / "npu" / "ascend_warmup_patch.py",
    )
    monkeypatch.setitem(sys.modules, "vllm_omni.platforms.npu.ascend_warmup_patch", warmup)
    gate = _load_source_module(
        "_npu_native_rejection_sampler_under_test",
        root / "vllm_omni" / "platforms" / "npu" / "native_rejection_sampler.py",
    )
    return warmup, gate


def _armed_config(num_speculative_tokens: int = 4, method: str = "ngram"):
    spec = SimpleNamespace(method=method, num_speculative_tokens=num_speculative_tokens)
    return SimpleNamespace(speculative_config=spec)


@pytest.mark.parametrize("soc_name", ["Ascend910_93", "Ascend910C"])
def test_restores_the_native_sampler_on_the_faulting_parts(monkeypatch, tmp_path, soc_name):
    """These parts fault in the Triton kernels, so the restore must happen."""
    leaf = _install_fake_vllm(monkeypatch, tmp_path)
    _warmup, gate = _load_gate(monkeypatch)
    monkeypatch.setattr(gate, "_probe_soc_name", lambda: soc_name)

    gate.restore_native_rejection_sampler()

    assert leaf.apply_sampling_constraints() == "constraints"
    assert leaf.expand_batch_to_tokens() == "expand"
    assert leaf.rejection_sample() == "sample"
    assert gate._RESTORED is True


def test_910b_without_the_kstep_keeps_the_triton_sampler(monkeypatch, tmp_path):
    """A 910B without the Talker K-step keeps its sampler byte-for-byte."""
    leaf = _install_fake_vllm(monkeypatch, tmp_path)
    _warmup, gate = _load_gate(monkeypatch)
    monkeypatch.setattr(gate, "_probe_soc_name", lambda: "Ascend910B3")

    gate.restore_native_rejection_sampler()

    assert not hasattr(leaf, "rejection_sample")
    # A confirmed non-target part is decided once.
    assert gate._RESTORED is True


def test_910b_with_the_kstep_restores_the_native_sampler(monkeypatch, tmp_path):
    """The 910B kernels cannot take the K-step widths, so the restore runs."""
    leaf = _install_fake_vllm(monkeypatch, tmp_path)
    _warmup, gate = _load_gate(monkeypatch)
    monkeypatch.setattr(gate, "_probe_soc_name", lambda: "Ascend910B3")

    gate.restore_native_rejection_sampler(_armed_config())

    assert leaf.rejection_sample() == "sample"


def test_the_gate_probes_the_soc_name_once_per_call(monkeypatch, tmp_path):
    """One probe decides the target check and the latch, so they cannot disagree."""
    _install_fake_vllm(monkeypatch, tmp_path)
    _warmup, gate = _load_gate(monkeypatch)
    probes: list[str] = []

    def fake_probe() -> str:
        probes.append("Ascend910B3")
        return "Ascend910B3"

    monkeypatch.setattr(gate, "_probe_soc_name", fake_probe)

    gate.restore_native_rejection_sampler()

    assert len(probes) == 1


def test_an_unknown_probe_is_not_latched_and_retries(monkeypatch, tmp_path):
    """An empty probe means "unknown", so a later call may still restore."""
    leaf = _install_fake_vllm(monkeypatch, tmp_path)
    _warmup, gate = _load_gate(monkeypatch)
    probes = iter(["", "Ascend910_93"])
    monkeypatch.setattr(gate, "_probe_soc_name", lambda: next(probes, "Ascend910_93"))

    gate.restore_native_rejection_sampler()
    assert gate._RESTORED is False
    assert not hasattr(leaf, "rejection_sample")

    gate.restore_native_rejection_sampler()
    assert leaf.rejection_sample() == "sample"


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (_armed_config(), True),
        (_armed_config(num_speculative_tokens=0), False),
        (_armed_config(method="eagle"), False),
        (SimpleNamespace(speculative_config=None), False),
        (None, False),
    ],
)
def test_the_kstep_gate_reads_the_config_of_the_caller(monkeypatch, tmp_path, config, expected):
    """The runner's own config decides, not the engine context that is already gone."""
    _install_fake_vllm(monkeypatch, tmp_path)
    warmup, _gate = _load_gate(monkeypatch)

    assert warmup._kstep_armed(config) is expected
