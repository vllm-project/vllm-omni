# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from dataclasses import dataclass

import pytest

from vllm_omni.diffusion.model_loader import prewarm

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@dataclass(frozen=True)
class _VirtualMemory:
    available: int


def _make_model_dir(tmp_path, sizes):
    """Create component-like subdirs with safetensors of the given sizes."""
    for i, (sub, size) in enumerate(sizes):
        d = tmp_path / sub
        d.mkdir(exist_ok=True)
        (d / f"model-{i:05d}.safetensors").write_bytes(os.urandom(size))
    (tmp_path / "config.json").write_text("{}")  # non-safetensors must be ignored
    return tmp_path


def test_read_order_and_ram_budget(tmp_path, monkeypatch):
    """The prewarm's two planning inputs in one pass.

    Read order and byte budget: reads are DiT-shards-first (what
    load_weights consumes; other components are read by from_pretrained
    inside the prewarm window), keyed on the MODEL-RELATIVE path so an
    ancestor directory containing 'transformer' cannot poison the ordering;
    the byte budget is shared across readers (a static split strands half
    of it when the DiT lands in one partition) and reading stops within one
    chunk of the cap.

    RAM budget: min(psutil host view, cgroup remaining) — psutil sees the
    HOST in containers (observed: a 126 GiB slice reporting a 1 TiB host)
    while page cache is charged to the cgroup. cgroup v2 and v1 files are
    both parsed, with 'max' / the ~2^63 v1 sentinel meaning unlimited; the
    final budget reserves proportional headroom plus a runtime-growth
    margin, and an unknown RAM figure disables prewarm."""
    base = tmp_path / "transformers-models"  # ancestor name that must not affect the key
    base.mkdir()
    d = _make_model_dir(
        base,
        [("transformer", 1 << 20), ("text_encoder", 1 << 20), ("transformer_2", 1 << 20), ("vae", 1 << 20)],
    )
    files = sorted(
        (os.path.join(r, n) for r, _, ns in os.walk(d) for n in ns if n.endswith(".safetensors")),
        key=lambda p: prewarm._read_order_key(os.path.relpath(p, d)),
    )
    subdirs = [os.path.basename(os.path.dirname(f)) for f in files]
    assert subdirs == ["transformer", "transformer_2", "text_encoder", "vae"]
    parent_files = sorted(
        files,
        key=lambda p: prewarm._parent_read_order_key(os.path.relpath(p, d)),
    )
    parent_subdirs = [os.path.basename(os.path.dirname(f)) for f in parent_files]
    assert parent_subdirs == ["text_encoder", "vae", "transformer", "transformer_2"]

    # shared budget: takes succeed until exhausted, crossing zero is admitted once
    budget = prewarm._Budget(10)
    assert budget.take(6) and budget.take(6)
    assert not budget.take(1)

    # cap-then-stop: never reads past the budget to the tail files; loader
    # handoff stops speculative readers before they touch another chunk.
    cap = (1 << 20) + (1 << 19)
    done_bytes, done_files = prewarm._read_files(files, prewarm._Budget(cap))
    assert done_bytes <= cap + prewarm._CHUNK
    assert done_files < 4
    stop = prewarm.threading.Event()
    stop.set()
    assert prewarm._read_files(files, prewarm._Budget(cap), stop) == (0, 0)

    # psutil host view (faked module)
    import sys as _sys
    import types as _types

    fake = _types.ModuleType("psutil")
    fake.virtual_memory = lambda: _VirtualMemory(available=42 << 30)
    monkeypatch.setitem(_sys.modules, "psutil", fake)
    assert prewarm._host_available_bytes() == 42 << 30

    # cgroup v2 / v1 parsing through redirected paths
    v2l, v2u = tmp_path / "memory.max", tmp_path / "memory.current"
    v1l, v1u = tmp_path / "limit_in_bytes", tmp_path / "usage_in_bytes"
    monkeypatch.setattr(prewarm, "_CGROUP_MEM_PATHS", ((str(v2l), str(v2u)), (str(v1l), str(v1u))))
    v2l.write_text("1073741824\n")
    v2u.write_text("73741824\n")
    assert prewarm._cgroup_available_bytes() == 1073741824 - 73741824
    v2l.write_text("max\n")
    assert prewarm._cgroup_available_bytes() is None  # v2 unlimited, no v1 fallthrough
    v2l.unlink()
    v1l.write_text(str((1 << 63) - 4096))
    v1u.write_text("0")
    assert prewarm._cgroup_available_bytes() is None  # v1 unlimited sentinel
    v1l.write_text("2147483648")
    v1u.write_text("147483648")
    assert prewarm._cgroup_available_bytes() == 2000000000
    v1l.unlink()
    v1u.unlink()
    assert prewarm._cgroup_available_bytes() is None  # nothing present

    # min() merge + headroom math + unknown-RAM gate
    monkeypatch.setattr(prewarm, "_host_available_bytes", lambda: 1000 << 30)
    monkeypatch.setattr(prewarm, "_cgroup_available_bytes", lambda: 100 << 30)
    assert prewarm._available_ram_bytes() == 100 << 30
    monkeypatch.setattr(prewarm, "_cgroup_available_bytes", lambda: None)
    assert prewarm._available_ram_bytes() == 1000 << 30
    rt = prewarm._RUNTIME_HEADROOM
    monkeypatch.setattr(prewarm, "_available_ram_bytes", lambda: 100 << 30)
    assert prewarm._prewarm_budget_bytes(100 << 30) == (100 << 30) - (15 << 30) - rt  # 15% proportional
    assert prewarm._prewarm_budget_bytes(10 << 30) == (100 << 30) - (8 << 30) - rt  # 8 GiB floor
    monkeypatch.setattr(prewarm, "_available_ram_bytes", lambda: None)
    assert prewarm._prewarm_budget_bytes(10 << 30) == 0


def test_resolve_local_dir_uses_revision_and_cache_dir(tmp_path, monkeypatch):
    import huggingface_hub

    calls = []

    def _snapshot_download(model, **kwargs):
        calls.append((model, kwargs))
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _snapshot_download)

    resolved = prewarm._resolve_local_dir("org/model", revision="fixed-revision", cache_dir="/model-cache")

    assert resolved == str(tmp_path)
    assert calls == [
        (
            "org/model",
            {"revision": "fixed-revision", "cache_dir": "/model-cache", "local_files_only": True},
        )
    ]


def test_gating_and_loader_hook(tmp_path, monkeypatch):
    """start_weights_prewarm self-gates (local dirs pass through, an uncached
    repo id resolves to None offline, empty/missing dirs, a sub-threshold
    budget, and a non-zero TP rank all skip; with budget the daemon runs to
    completion), and load_model kicks it off unconditionally."""
    assert prewarm._resolve_local_dir(str(tmp_path)) == str(tmp_path)
    assert prewarm._resolve_local_dir("definitely/not-a-cached-repo") is None
    assert prewarm.start_weights_prewarm("/nonexistent/path") is None
    # A repo id that can never be in the local hub cache: a real model id
    # here would resolve (and start prewarm) on any machine that has
    # downloaded it, making the test machine-dependent.
    assert prewarm.start_weights_prewarm("definitely/not-a-cached-repo") is None

    (tmp_path / "readme.md").write_text("x")
    assert prewarm.start_weights_prewarm(str(tmp_path)) is None  # no safetensors

    d = _make_model_dir(tmp_path, [("transformer", 1 << 20), ("vae", 1 << 19)])
    monkeypatch.setattr(prewarm, "_prewarm_budget_bytes", lambda total: 0)
    assert prewarm.start_weights_prewarm(str(d)) is None  # no budget

    monkeypatch.setattr(prewarm, "_prewarm_budget_bytes", lambda total: 1 << 30)
    monkeypatch.setattr(prewarm, "_MIN_WORTHWHILE", 0)
    monkeypatch.setattr(prewarm, "_is_file_fully_resident", lambda path: True)
    assert prewarm.start_weights_prewarm(str(d)) is None  # resident files skip userspace reads
    monkeypatch.setattr(prewarm, "_is_file_fully_resident", lambda path: False)
    monkeypatch.setattr(prewarm, "_is_prewarm_rank", lambda: False)
    assert prewarm.start_weights_prewarm(str(d)) is None  # non-zero TP rank skips

    monkeypatch.setattr(prewarm, "_is_prewarm_rank", lambda: True)
    t = prewarm.start_weights_prewarm(str(d))
    assert t is not None
    t.join(timeout=10)
    assert not t.is_alive()

    # loader hook: load_model starts the (self-gated) prewarm with the same
    # revision and cache directory used by demand loading, and always cancels
    # it when model construction fails before the native iterator handoff.
    import torch

    from vllm_omni.diffusion.model_loader import diffusers_loader as dl

    class _Handle:
        stopped = False
        joined = False

        def stop(self):
            self.stopped = True

        def join(self, timeout=None):
            assert self.stopped
            self.joined = True

    handle = _Handle()
    calls = []

    def _start(path, *, revision=None, cache_dir=None):
        calls.append((path, revision, cache_dir))
        return handle

    monkeypatch.setattr(dl, "start_weights_prewarm", _start)

    loader = dl.DiffusersPipelineLoader.__new__(dl.DiffusersPipelineLoader)
    loader.quant_config = None
    loader.parallel_config = type("PC", (), {"use_hsdp": False})()
    loader.load_config = type("LC", (), {"download_dir": "/model-cache"})()
    loader.od_config = type(
        "OD",
        (),
        {"model": str(tmp_path), "revision": "fixed-revision", "dtype": torch.float32},
    )()

    sentinel = RuntimeError("stop after prewarm hook")

    def _stop(self, *a, **k):
        raise sentinel

    monkeypatch.setattr(dl.DiffusersPipelineLoader, "_init_from_load_format", _stop)
    with pytest.raises(RuntimeError) as exc_info:
        loader.load_model(load_device="cpu")
    assert exc_info.value is sentinel
    assert calls == [(str(tmp_path), "fixed-revision", "/model-cache")]
    assert handle.stopped
    assert handle.joined
    assert loader._weights_prewarm_handle is None


def test_parent_prewarm_hands_off_before_worker_model_init(monkeypatch):
    class _Handle:
        def __init__(self):
            self.stopped = prewarm.threading.Event()
            self.joined = False

        def stop(self):
            self.stopped.set()

        def join(self, timeout=None):
            assert self.stopped.is_set()
            self.joined = True

    handle = _Handle()
    calls = []

    def _start(model, *, revision=None, cache_dir=None, read_order_key=None):
        calls.append((model, revision, cache_dir, read_order_key))
        return handle

    monkeypatch.setattr(prewarm, "_start_weights_prewarm", _start)

    with prewarm.parent_weights_prewarm(
        "org/model",
        revision="fixed-revision",
        cache_dir="/model-cache",
    ) as handoff:
        assert handoff is not None
        assert not handle.stopped.is_set()
        with prewarm.use_parent_weights_prewarm(handoff):
            assert handle.stopped.is_set()
            assert handle.joined
            # The worker's DiffusersPipelineLoader still calls this interface,
            # but the parent handoff suppresses a duplicate speculative reader.
            assert prewarm.start_weights_prewarm("org/model") is None

    assert calls == [
        ("org/model", "fixed-revision", "/model-cache", prewarm._parent_read_order_key),
    ]


def test_worker_proc_hands_off_parent_prewarm_before_worker_creation(monkeypatch):
    from vllm_omni.diffusion.worker import diffusion_worker as dw

    class _Handle:
        def __init__(self):
            self.stopped = prewarm.threading.Event()
            self.joined = False

        def stop(self):
            self.stopped.set()

        def join(self, timeout=None):
            assert self.stopped.is_set()
            self.joined = True

    handle = _Handle()
    monkeypatch.setattr(prewarm, "_start_weights_prewarm", lambda *args, **kwargs: handle)

    class _MessageQueue:
        def __init__(self, *args, **kwargs):
            pass

        @classmethod
        def create_from_handle(cls, *args, **kwargs):
            return cls()

        def export_handle(self):
            return object()

    monkeypatch.setattr(dw, "MessageQueue", _MessageQueue)
    monkeypatch.setattr(dw.zmq, "Context", lambda **kwargs: object())

    created_worker = object()

    def _create_worker(self, *args, **kwargs):
        assert handle.stopped.is_set()
        assert handle.joined
        assert prewarm.start_weights_prewarm("org/model") is None
        return created_worker

    monkeypatch.setattr(dw.WorkerProc, "_create_worker", _create_worker)
    od_config = type("OD", (), {"master_port": 12345, "step_execution": True})()

    with prewarm.parent_weights_prewarm("org/model") as handoff:
        worker_proc = dw.WorkerProc(
            od_config=od_config,
            gpu_id=0,
            broadcast_handle=object(),
            wake_event=None,
            weights_prewarm_handoff=handoff,
        )

    assert worker_proc.worker is created_worker


def test_parent_prewarm_suppresses_duplicate_worker_scan_when_no_reads_start(monkeypatch):
    calls = []

    def _skip(*args, **kwargs):
        calls.append((args, kwargs))
        return None

    monkeypatch.setattr(prewarm, "_start_weights_prewarm", _skip)

    with prewarm.parent_weights_prewarm("org/model") as handoff:
        assert handoff is not None
        with prewarm.use_parent_weights_prewarm(handoff):
            assert prewarm.start_weights_prewarm("org/model") is None

    assert len(calls) == 1


def test_diffusers_load_hands_off_prewarm_before_demand_loading(monkeypatch):
    import torch

    from vllm_omni.diffusion.model_loader import diffusers_loader as dl

    class _Handle:
        stopped = False
        joined = False

        def stop(self):
            self.stopped = True

        def join(self, timeout=None):
            assert self.stopped
            self.joined = True

        def is_alive(self):
            return not self.joined

    handle = _Handle()
    monkeypatch.setattr(dl, "start_weights_prewarm", lambda *args, **kwargs: handle)

    loader = dl.DiffusersPipelineLoader.__new__(dl.DiffusersPipelineLoader)
    loader.quant_config = None
    loader.parallel_config = type("PC", (), {"use_hsdp": False})()
    loader.load_config = type("LC", (), {"download_dir": None})()
    loader.od_config = type("OD", (), {"model": "org/model", "revision": None, "dtype": torch.float32})()

    class _Model(torch.nn.Module):
        def load_weights(self):
            assert handle.stopped
            assert handle.joined
            assert not handle.is_alive()

    model = _Model()
    monkeypatch.setattr(loader, "_init_from_load_format", lambda *args, **kwargs: model)
    monkeypatch.setattr(loader, "_process_weights_after_loading", lambda *args, **kwargs: None)
    monkeypatch.setattr(loader, "_apply_skip_softmax_calibration", lambda *args, **kwargs: None)

    assert loader.load_model(load_device="cpu", load_format="diffusers") is model
    assert loader._weights_prewarm_handle is None
