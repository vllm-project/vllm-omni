# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import os
from dataclasses import dataclass

import pytest
import torch

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
    # Parent prewarm runs through pipeline construction, so it must not race
    # the text encoder the build is already opening: DiT shards first.
    assert parent_subdirs == ["transformer", "transformer_2", "text_encoder", "vae"]

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
    calls = []

    class _Api:
        def snapshot_download(self, model, **kwargs):
            calls.append((model, kwargs))
            return str(tmp_path)

    monkeypatch.setattr("vllm_omni.transformers_utils.repo_utils.hf_api", lambda: _Api())

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
    monkeypatch.setattr(prewarm, "_is_file_fully_resident", lambda path, **_kwargs: True)
    assert prewarm.start_weights_prewarm(str(d)) is None  # resident files skip userspace reads
    monkeypatch.setattr(prewarm, "_is_file_fully_resident", lambda path, **_kwargs: False)
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

        def is_alive(self):
            return not self.joined

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
        {
            "model": str(tmp_path),
            "revision": "fixed-revision",
            "dtype": torch.float32,
            # Opt-in: prewarm is off unless the flag asks for it.
            "enable_weights_prewarm": True,
        },
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

    loader.od_config.enable_weights_prewarm = False
    calls.clear()
    with pytest.raises(RuntimeError):
        loader.load_model(load_device="cpu")
    assert calls == []
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

        def is_alive(self):
            return not self.joined

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
            assert not handle.stopped.is_set()
            assert not handle.joined
            # Duplicate worker-local scans stay suppressed while parent readers
            # keep running until this context exits.
            assert prewarm.start_weights_prewarm("org/model") is None
        assert not handle.stopped.is_set()

    assert handle.stopped.is_set() and handle.joined
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

        def is_alive(self):
            return not self.joined

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
        assert not handle.stopped.is_set()
        assert not handle.joined
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
    assert handle.stopped.is_set() and handle.joined


def test_parent_prewarm_terminates_stalled_reader_before_releasing_workers(monkeypatch):
    class _StalledWorker:
        def __init__(self):
            self.alive = True
            self.join_timeouts = []
            self.terminated = False

        def start(self):
            pass

        def join(self, timeout=None):
            self.join_timeouts.append(timeout)

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated = True
            self.alive = False

    worker = _StalledWorker()
    handle = prewarm.WeightsPrewarmHandle(worker, prewarm.threading.Event())
    monkeypatch.setattr(prewarm, "_start_weights_prewarm", lambda *args, **kwargs: handle)
    monkeypatch.setattr(prewarm, "_PREWARM_STOP_TIMEOUT_S", 0.01)

    with prewarm.parent_weights_prewarm("org/model") as handoff:
        with prewarm.use_parent_weights_prewarm(handoff):
            assert not worker.terminated
    assert worker.terminated
    assert worker.join_timeouts == [0.01, 0.01]


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
    loader.od_config = type(
        "OD",
        (),
        {"model": "org/model", "revision": None, "dtype": torch.float32, "enable_weights_prewarm": True},
    )()
    # Set in __init__, but this test bypasses it to exercise load_model alone.
    loader.host_weight_plan = None
    loader._hwr_state = None
    loader._force_canonical_load = False

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


def test_yield_file_skips_one_file_and_keeps_reading_others(tmp_path, monkeypatch):
    monkeypatch.setattr(prewarm, "_advise_range", lambda *args, **kwargs: False)
    monkeypatch.setattr(prewarm, "_is_file_fully_resident", lambda path, **_kwargs: False)
    model_dir = _make_model_dir(
        tmp_path,
        [("text_encoder", 1 << 20), ("transformer", 1 << 20)],
    )
    files = sorted(
        (
            os.path.join(root, name)
            for root, _, names in os.walk(model_dir)
            for name in names
            if name.endswith(".safetensors")
        ),
        key=lambda path: prewarm._parent_read_order_key(os.path.relpath(path, model_dir)),
    )
    assert len(files) == 2
    handle = prewarm.WeightsPrewarmHandle(None, prewarm.threading.Event())
    handle.yield_file(files[0])
    done_bytes, done_files = prewarm._read_files(files, prewarm._Budget(4 << 20), handle)
    assert done_bytes == os.path.getsize(files[1])
    assert done_files == 1


def test_residency_helpers(tmp_path, monkeypatch):
    """Pause only when the demanded file still needs the disk; the group
    residency helper requires every file to be resident."""
    model_dir = _make_model_dir(
        tmp_path,
        [("transformer", 1 << 20), ("transformer_2", 1 << 20)],
    )
    files = sorted(
        (
            os.path.join(root, name)
            for root, _, names in os.walk(model_dir)
            for name in names
            if name.endswith(".safetensors")
        ),
        key=lambda path: prewarm._parent_read_order_key(os.path.relpath(path, model_dir)),
    )
    assert len(files) == 2

    monkeypatch.setattr(prewarm, "_is_file_fully_resident", lambda p, **_kwargs: p == files[0])
    assert prewarm.should_pause_for_demand(files[0]) is False
    assert prewarm.should_pause_for_demand(files[1]) is True
    assert prewarm.checkpoint_fully_resident(files) is False

    monkeypatch.setattr(prewarm, "_is_file_fully_resident", lambda p, **_kwargs: True)
    assert prewarm.should_pause_for_demand(files[0]) is False
    assert prewarm.checkpoint_fully_resident(files) is True
    assert prewarm.checkpoint_fully_resident([]) is False


def test_pause_copy_parks_until_resume_without_advising(tmp_path, monkeypatch):
    """A pause must block the reader in place. Hinting and walking the rest of
    the file exits the thread, so resume is a no-op, and the readahead faults
    the pages the residency gate samples."""
    advised = []
    forced = []
    monkeypatch.setattr(prewarm, "_advise_range", lambda fd, offset, length: advised.append((offset, length)) or True)
    monkeypatch.setattr(prewarm, "_range_resident", lambda mapping, offset, length: False)
    monkeypatch.setattr(
        prewarm,
        "_force_read_range",
        lambda handle, offset, length, buf, stop: forced.append(length) or length,
    )
    path = _make_model_dir(tmp_path, [("transformer", 3 << 20)]) / "transformer" / "model-00000.safetensors"
    handle = prewarm.WeightsPrewarmHandle(None, prewarm.threading.Event())
    handle.pause_copy()
    box: dict[str, tuple[int, int]] = {}

    def _run() -> None:
        box["result"] = prewarm._read_files([str(path)], prewarm._Budget(4 << 20), handle)

    worker = prewarm.threading.Thread(target=_run)
    worker.start()
    worker.join(0.2)
    assert worker.is_alive()
    assert advised == [] and forced == [] and "result" not in box

    handle.resume_copy()
    worker.join(2)
    assert not worker.is_alive()
    assert box["result"] == (3 << 20, 1)
    assert forced == [3 << 20]


def test_stop_unblocks_paused_reader(tmp_path, monkeypatch):
    monkeypatch.setattr(prewarm, "_advise_range", lambda *args, **kwargs: False)
    monkeypatch.setattr(prewarm, "_range_resident", lambda mapping, offset, length: False)
    monkeypatch.setattr(prewarm, "_force_read_range", lambda handle, offset, length, buf, stop: length)
    path = _make_model_dir(tmp_path, [("transformer", 3 << 20)]) / "transformer" / "model-00000.safetensors"
    handle = prewarm.WeightsPrewarmHandle(None, prewarm.threading.Event())
    handle.pause_copy()
    worker = prewarm.threading.Thread(
        target=prewarm._read_files,
        args=([str(path)], prewarm._Budget(4 << 20), handle),
    )
    worker.start()
    worker.join(0.1)
    assert worker.is_alive()
    handle.stop()
    worker.join(2)
    assert not worker.is_alive()


def test_strict_residency_probe_includes_chunk_tail():
    page = prewarm.mmap.PAGESIZE
    assert prewarm._residency_probe_offsets(prewarm._CHUNK, strict=False) == [0]
    strict = prewarm._residency_probe_offsets(prewarm._CHUNK, strict=True)
    assert strict[0] == 0 and strict[-1] == prewarm._CHUNK - page
    # A file of exactly one page has no distinct tail probe.
    assert prewarm._residency_probe_offsets(page, strict=True) == [0]
    # Real safetensors sizes are not page-aligned. The tail probe is the page
    # that contains the last byte, not the page before it.
    assert prewarm._residency_probe_offsets(page + 1, strict=True) == [0, page]
    assert prewarm._residency_probe_offsets((2 * page) + 100, strict=True) == [0, 2 * page]


def test_chunked_advise_skips_copy_when_resident(tmp_path, monkeypatch):
    """Hint the current and next chunk; copy only when mincore is negative."""
    advised = []
    forced = []

    monkeypatch.setattr(prewarm, "_advise_range", lambda fd, offset, length: advised.append((offset, length)) or True)
    monkeypatch.setattr(prewarm, "_range_resident", lambda mapping, offset, length: True)
    monkeypatch.setattr(
        prewarm,
        "_force_read_range",
        lambda handle, offset, length, buf, stop: forced.append(length) or length,
    )
    path = _make_model_dir(tmp_path, [("transformer", 3 << 20)]) / "transformer" / "model-00000.safetensors"
    done_bytes, done_files = prewarm._read_files([str(path)], prewarm._Budget(4 << 20))
    assert done_bytes == 3 << 20 and done_files == 1
    assert advised == [(0, 3 << 20)] and forced == []

    advised.clear()
    monkeypatch.setattr(prewarm, "_range_resident", lambda mapping, offset, length: False)
    done_bytes, done_files = prewarm._read_files([str(path)], prewarm._Budget(4 << 20))
    assert done_bytes == 3 << 20 and done_files == 1
    assert advised == [(0, 3 << 20)] and forced == [3 << 20]


@pytest.mark.parametrize(
    ("tp_rank", "tp_world", "local_world", "expected"),
    [
        # Single node (whole group fits here): one prewarmer, as measured.
        (0, 2, 8, True),
        (1, 2, 8, False),
        (0, 2, 2, True),  # the exact configuration the numbers were measured on
        (1, 2, 2, False),
        (0, 8, 8, True),
        (3, 8, 8, False),
        # TP spans nodes: one prewarmer per node instead of one per world.
        (0, 8, 4, True),
        (1, 8, 4, False),
        (4, 8, 4, True),
        (5, 8, 4, False),
        # Ambiguous visibility (one device per process): keep rank 0 only.
        (0, 8, 1, True),
        (4, 8, 1, False),
        # Degenerate group.
        (0, 1, 8, True),
        (5, 1, 8, True),
    ],
)
def test_prewarm_rank_is_per_node(monkeypatch, tp_rank, tp_world, local_world, expected):
    """One prewarming rank per node, not one per world.

    Gating on global TP rank 0 leaves every other node's page cache cold, so a
    multi-node TP group needs a prewarmer per node. The node's rank count comes
    from the visible device count, and anything ambiguous keeps the single-node
    behaviour that was actually measured.
    """
    import vllm.distributed.parallel_state as parallel_state

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", lambda: tp_rank)
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_world_size", lambda: tp_world)
    monkeypatch.setattr(torch.accelerator, "device_count", lambda: local_world)

    assert prewarm._is_prewarm_rank() is expected


def test_weights_prewarm_is_opt_in_by_default(monkeypatch, tmp_path):
    """Prewarm must not start unless it is asked for.

    It pays off when pipeline construction does little checkpoint I/O and costs
    6-9 s when construction reads the same cold disk, so the default has to be the
    safe one and the caller has to opt in.
    """
    import torch

    from vllm_omni.diffusion.model_loader import diffusers_loader as dl

    calls: list[tuple] = []
    monkeypatch.setattr(dl, "start_weights_prewarm", lambda *a, **k: calls.append(a) or None)
    model = torch.nn.Linear(2, 2)
    monkeypatch.setattr(dl.DiffusersPipelineLoader, "_load_model", lambda self, *a, **k: model)

    def _loader(prewarm: bool | None):
        built = dl.DiffusersPipelineLoader.__new__(dl.DiffusersPipelineLoader)
        built.quant_config = None
        built.parallel_config = type("PC", (), {"use_hsdp": False})()
        built.load_config = type("LC", (), {"download_dir": None})()
        od = {"model": str(tmp_path), "dtype": torch.float32}
        if prewarm is not None:
            od["enable_weights_prewarm"] = prewarm
        built.od_config = type("OD", (), od)()
        return built

    # Field absent entirely: absent must mean off, not "some other default".
    assert _loader(None).load_model(load_device="cpu") is model
    assert calls == [], "prewarm must stay off when the flag is unset"

    assert _loader(False).load_model(load_device="cpu") is model
    assert calls == [], "prewarm must stay off when it is explicitly disabled"

    assert _loader(True).load_model(load_device="cpu") is model
    assert len(calls) == 1, "prewarm must start when it is explicitly enabled"
