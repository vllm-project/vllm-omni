# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Preserve the CPU speaker graph's pooling semantics before TRT conversion."""

import os
import threading
from pathlib import Path

import numpy as np
import pytest

from vllm_omni.model_executor.models.cosyvoice3 import speaker_embedding_trt
from vllm_omni.model_executor.models.cosyvoice3.speaker_embedding_trt import _cpu_reference_onnx

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _write_model(path, scale=1.0):
    h = onnx.helper
    graph = h.make_graph(
        [
            h.make_node("Pad", ["x", "pads"], ["padded"], mode="constant"),
            h.make_node("AveragePool", ["padded"], ["pooled"], kernel_shape=[100], strides=[100], ceil_mode=1),
            h.make_node("Mul", ["pooled", "scale"], ["y"]),
        ],
        "speaker_pool",
        [h.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 1, "T"])],
        [h.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 1, "O"])],
        [
            h.make_tensor("pads", onnx.TensorProto.INT64, [6], [0] * 6),
            h.make_tensor("scale", onnx.TensorProto.FLOAT, [1], [scale]),
        ],
    )
    onnx.save(h.make_model(graph, opset_imports=[h.make_opsetid("", 17)], ir_version=10), path)


def _run(path, features, optimization):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = optimization
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    return session.run(None, {"x": features})[0]


@pytest.mark.parametrize("frames", [97, 100, 200, 598])
def test_export_retains_reference_ceil_pool_semantics(tmp_path, monkeypatch, frames):
    monkeypatch.setenv("COSYVOICE3_TRT_CACHE", str(tmp_path / "cache"))
    source = tmp_path / "speaker.onnx"
    _write_model(source)
    features = np.arange(frames, dtype=np.float32).reshape(1, 1, frames)
    reference = _run(source, features, ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
    exported = _cpu_reference_onnx(str(source))
    # No optimizer here: another backend must receive the effective graph,
    # including the reference runtime's treatment of a partial final window.
    actual = _run(exported, features, ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    np.testing.assert_array_equal(actual, reference)


def test_cache_identity_includes_model_content(tmp_path, monkeypatch):
    monkeypatch.setenv("COSYVOICE3_TRT_CACHE", str(tmp_path / "cache"))
    source = tmp_path / "speaker.onnx"
    _write_model(source)
    before = source.stat()
    first = _cpu_reference_onnx(str(source))
    assert _cpu_reference_onnx(str(source)) == first
    _write_model(source, scale=2.0)
    assert source.stat().st_size == before.st_size
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    second = _cpu_reference_onnx(str(source))
    assert second != first
    features = np.ones((1, 1, 97), dtype=np.float32)
    np.testing.assert_array_equal(
        _run(second, features, ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        _run(source, features, ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    )


def test_reference_mode_does_not_reuse_the_default_engine(monkeypatch):
    monkeypatch.setattr(speaker_embedding_trt, "_CAMPPLUS_CACHE", {})
    monkeypatch.setattr(speaker_embedding_trt, "_cpu_reference_onnx", lambda path: path + ".cpu.onnx")
    monkeypatch.setattr(
        speaker_embedding_trt,
        "CampplusTRT",
        lambda path, device, *, allow_tf32: {"path": path, "device": device, "allow_tf32": allow_tf32},
    )
    original = speaker_embedding_trt.get_campplus_trt("speaker.onnx", "cuda:0")
    reference = speaker_embedding_trt.get_campplus_trt("speaker.onnx", "cuda:0", cpu_reference=True)
    assert reference is speaker_embedding_trt.get_campplus_trt("speaker.onnx", "cuda:0", cpu_reference=True)
    assert original is not reference
    assert original["allow_tf32"] is True
    assert reference["allow_tf32"] is False
    assert original["path"] != reference["path"]


def test_plan_publish_replaces_a_private_temp_file(tmp_path, monkeypatch):
    plan = tmp_path / "campplus.plan"
    published = []
    real_replace = os.replace

    def spy_replace(source, destination):
        published.append(Path(source).name)
        real_replace(source, destination)

    monkeypatch.setattr(speaker_embedding_trt.os, "replace", spy_replace)
    speaker_embedding_trt._write_plan_atomic(str(plan), b"engine")

    assert published
    assert published[0] != f"{plan.name}.tmp"
    assert plan.read_bytes() == b"engine"
    assert list(tmp_path.iterdir()) == [plan]


def test_failed_plan_publish_leaves_no_partial_file(tmp_path, monkeypatch):
    plan = tmp_path / "campplus.plan"

    def fail_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr(speaker_embedding_trt.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        speaker_embedding_trt._write_plan_atomic(str(plan), b"engine")
    assert list(tmp_path.iterdir()) == []


def test_concurrent_plan_publish_keeps_one_complete_payload(tmp_path):
    plan = tmp_path / "campplus.plan"
    payloads = [b"A" * 250_000, b"B" * 250_000]

    threads = [
        threading.Thread(target=speaker_embedding_trt._write_plan_atomic, args=(str(plan), payload))
        for payload in payloads
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert plan.read_bytes() in payloads


def test_shared_plan_cache_builds_once(tmp_path, monkeypatch):
    plan = tmp_path / "campplus.plan"
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def slow_convert(onnx_path, plan_path, *, allow_tf32):
        nonlocal calls
        calls += 1
        assert allow_tf32 is False
        entered.set()
        assert release.wait(timeout=5)
        Path(plan_path).write_bytes(b"engine")

    monkeypatch.setattr(speaker_embedding_trt, "_convert_onnx_to_trt", slow_convert)
    errors = []

    def build():
        try:
            speaker_embedding_trt._ensure_plan("speaker.onnx", str(plan), allow_tf32=False)
        except Exception as exc:  # pragma: no cover - assertion helper
            errors.append(exc)

    threads = [threading.Thread(target=build) for _ in range(2)]
    for thread in threads:
        thread.start()
    assert entered.wait(timeout=5)
    release.set()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert errors == []
    assert calls == 1
    assert plan.read_bytes() == b"engine"


def test_failed_plan_build_can_be_retried(tmp_path, monkeypatch):
    plan = tmp_path / "campplus.plan"
    attempts = 0

    def flaky_convert(onnx_path, plan_path, *, allow_tf32):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("build failed")
        Path(plan_path).write_bytes(b"ok")

    monkeypatch.setattr(speaker_embedding_trt, "_convert_onnx_to_trt", flaky_convert)
    with pytest.raises(RuntimeError, match="build failed"):
        speaker_embedding_trt._ensure_plan("speaker.onnx", str(plan), allow_tf32=True)
    assert not plan.exists()

    speaker_embedding_trt._ensure_plan("speaker.onnx", str(plan), allow_tf32=True)
    assert plan.read_bytes() == b"ok"
    assert attempts == 2


@pytest.mark.parametrize("artifact", ["plan", "onnx"])
@pytest.mark.parametrize("mask,expected_mode", [(0o022, 0o644), (0o002, 0o664), (0o077, 0o600)])
def test_published_cache_permissions_follow_normal_file_creation(tmp_path, monkeypatch, artifact, mask, expected_mode):
    """Exercise shared and private permissions independently of the runner's umask."""
    monkeypatch.setenv("COSYVOICE3_TRT_CACHE", str(tmp_path / "cache"))
    previous_mask = os.umask(mask)
    try:
        if artifact == "plan":
            target = tmp_path / "campplus.plan"
            speaker_embedding_trt._write_plan_atomic(str(target), b"engine")
        else:
            source = tmp_path / "speaker.onnx"
            _write_model(source)
            target = Path(_cpu_reference_onnx(str(source)))
        assert target.stat().st_mode & 0o777 == expected_mode
    finally:
        os.umask(previous_mask)


@pytest.mark.parametrize("artifact", ["plan", "onnx"])
def test_complete_cache_hit_does_not_require_a_writable_lock(tmp_path, monkeypatch, artifact):
    monkeypatch.setenv("COSYVOICE3_TRT_CACHE", str(tmp_path / "cache"))
    source = tmp_path / "speaker.onnx"
    _write_model(source)
    if artifact == "onnx":
        target = Path(_cpu_reference_onnx(str(source)))
    else:
        target = tmp_path / "campplus.plan"
        target.write_bytes(b"complete engine")

    def fail_lock(*args, **kwargs):
        raise PermissionError("cache mounted read-only")

    monkeypatch.setattr(speaker_embedding_trt, "FileLock", fail_lock)
    if artifact == "onnx":
        assert _cpu_reference_onnx(str(source)) == str(target)
    else:
        speaker_embedding_trt._ensure_plan(str(source), str(target), allow_tf32=False)
