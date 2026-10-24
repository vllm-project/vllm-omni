# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

from vllm_omni.model_executor.models.cosyvoice3 import flow_estimator_trt

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _temporary_plans(plan_path: Path) -> list[Path]:
    return list(plan_path.parent.glob(f"{plan_path.name}.tmp.*"))


def test_write_plan_cleans_up_after_replace_failure(tmp_path, monkeypatch):
    plan_path = tmp_path / "flow.plan"
    plan_path.write_bytes(b"existing plan")
    replace_error = OSError("replace failed")

    def fail_replace(source, destination):
        raise replace_error

    monkeypatch.setattr(flow_estimator_trt.os, "replace", fail_replace)

    with pytest.raises(OSError) as exc_info:
        flow_estimator_trt._write_plan_atomically(b"new plan", str(plan_path))

    assert exc_info.value is replace_error
    assert plan_path.read_bytes() == b"existing plan"
    assert _temporary_plans(plan_path) == []


def test_write_plan_preserves_replace_error_when_cleanup_fails(tmp_path, monkeypatch):
    plan_path = tmp_path / "flow.plan"
    replace_error = OSError("replace failed")

    def fail_replace(source, destination):
        raise replace_error

    def fail_unlink(path):
        raise PermissionError("cleanup failed")

    monkeypatch.setattr(flow_estimator_trt.os, "replace", fail_replace)
    monkeypatch.setattr(flow_estimator_trt.os, "unlink", fail_unlink)

    with pytest.raises(OSError) as exc_info:
        flow_estimator_trt._write_plan_atomically(b"new plan", str(plan_path))

    assert exc_info.value is replace_error
    assert len(_temporary_plans(plan_path)) == 1


def test_write_plan_cleans_up_after_write_failure(tmp_path, monkeypatch):
    plan_path = tmp_path / "flow.plan"
    write_error = OSError("write failed")
    real_open = open

    class FailingWriter:
        def __init__(self, path, mode):
            self.file = real_open(path, mode)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.file.close()

        def write(self, data):
            self.file.write(data[:1])
            raise write_error

    monkeypatch.setattr(flow_estimator_trt, "open", FailingWriter, raising=False)

    with pytest.raises(OSError) as exc_info:
        flow_estimator_trt._write_plan_atomically(b"new plan", str(plan_path))

    assert exc_info.value is write_error
    assert not plan_path.exists()
    assert _temporary_plans(plan_path) == []


def test_write_plan_does_not_remove_a_colliding_temporary_file(tmp_path, monkeypatch):
    plan_path = tmp_path / "flow.plan"
    token = "0" * 32
    temporary_path = Path(f"{plan_path}.tmp.{flow_estimator_trt.os.getpid()}.{token}")
    temporary_path.write_bytes(b"another writer")
    monkeypatch.setattr(flow_estimator_trt.uuid, "uuid4", lambda: flow_estimator_trt.uuid.UUID(hex=token))

    with pytest.raises(FileExistsError):
        flow_estimator_trt._write_plan_atomically(b"new plan", str(plan_path))

    assert temporary_path.read_bytes() == b"another writer"
    assert not plan_path.exists()


def test_write_plan_supports_concurrent_publication(tmp_path, monkeypatch):
    plan_path = tmp_path / "flow.plan"
    payloads = (b"a" * 4096, b"b" * 4096)
    barrier = threading.Barrier(len(payloads))
    source_paths = []
    source_paths_lock = threading.Lock()
    real_replace = flow_estimator_trt.os.replace

    def synchronized_replace(source, destination):
        with source_paths_lock:
            source_paths.append(Path(source))
        barrier.wait(timeout=5)
        real_replace(source, destination)

    monkeypatch.setattr(flow_estimator_trt.os, "replace", synchronized_replace)

    with ThreadPoolExecutor(max_workers=len(payloads)) as executor:
        futures = [
            executor.submit(flow_estimator_trt._write_plan_atomically, payload, str(plan_path)) for payload in payloads
        ]
        for future in futures:
            future.result(timeout=10)

    assert len(set(source_paths)) == len(payloads)
    assert plan_path.read_bytes() in payloads
    assert _temporary_plans(plan_path) == []


class _FakeDim:
    def __init__(self, value: int = 2):
        self.dim_value: int | None = value
        self.dim_param = ""

    def ClearField(self, name: str):
        assert name == "dim_value"
        self.dim_value = None


class _FakeTensorType:
    def __init__(self, batch: int = 2):
        self.shape = type("Shape", (), {"dim": [_FakeDim(batch)]})()


class _FakeValue:
    def __init__(self, name: str, batch: int = 2):
        self.name = name
        self.type = type("Type", (), {"tensor_type": _FakeTensorType(batch)})()


class _FakeGraph:
    def __init__(self, *, omit: str | None = None):
        names = ["x", "mask", "mu", "t", "spks", "cond"]
        self.input = [_FakeValue(name) for name in names if name != omit]
        self.output = [] if omit == "estimator_out" else [_FakeValue("estimator_out")]
        self.value_info = [_FakeValue("internal_static_annotation")]


class _FakeModel:
    def __init__(self, *, omit: str | None = None):
        self.graph = _FakeGraph(omit=omit)


def test_set_onnx_cfg_batch_dynamic_only_rewrites_graph_io():
    model = _FakeModel()
    internal = model.graph.value_info[0]

    returned = flow_estimator_trt._set_onnx_cfg_batch_dynamic(model)

    assert returned is model
    for value in (*model.graph.input, *model.graph.output):
        batch_dim = value.type.tensor_type.shape.dim[0]
        assert batch_dim.dim_value is None
        assert batch_dim.dim_param == "cfg_batch"
    internal_batch = internal.type.tensor_type.shape.dim[0]
    assert internal_batch.dim_value == 2
    assert internal_batch.dim_param == ""


@pytest.mark.parametrize("missing", ["x", "t", "estimator_out"])
def test_set_onnx_cfg_batch_dynamic_requires_expected_io(missing):
    with pytest.raises(ValueError, match="missing expected graph I/O"):
        flow_estimator_trt._set_onnx_cfg_batch_dynamic(_FakeModel(omit=missing))


def test_fixed_cfg_batch_profile_covers_all_trt_inputs():
    profile = flow_estimator_trt._fixed_cfg_batch_profile()

    assert set(profile) == {"x", "mask", "mu", "t", "spks", "cond"}
    assert profile["x"] == ((2, 80, 4), (2, 80, 500), (2, 80, 3000))
    assert profile["t"] == ((2,), (2,), (2,))
    assert profile["spks"] == ((2, 80), (2, 80), (2, 80))


def test_dynamic_batch_profile_covers_all_trt_inputs():
    profile = flow_estimator_trt._dynamic_batch_profile(16)

    assert set(profile) == {"x", "mask", "mu", "t", "spks", "cond"}
    assert profile["x"] == ((4, 80, 4), (8, 80, 500), (16, 80, 1024))
    assert profile["t"] == ((4,), (8,), (16,))
    assert profile["spks"] == ((4, 80), (8, 80), (16, 80))


@pytest.mark.parametrize("max_cfg_batch", [0, 2, 3])
def test_dynamic_batch_profile_rejects_non_batched_limits(max_cfg_batch):
    with pytest.raises(ValueError, match="at least 4"):
        flow_estimator_trt._dynamic_batch_profile(max_cfg_batch)


class _FakeStream:
    cuda_stream = 1234

    def __init__(self):
        self.synchronize_calls = 0

    def synchronize(self):
        self.synchronize_calls += 1


class _FakeContext:
    def __init__(self, *, switch_ok: bool = True):
        self.switch_ok = switch_ok
        self.profile_switches: list[tuple[int, int]] = []

    def set_optimization_profile_async(self, profile_index, stream_handle):
        self.profile_switches.append((profile_index, stream_handle))
        return self.switch_ok


class _DynamicEngine:
    num_optimization_profiles = 2

    def __init__(self, context):
        self.context = context

    def create_execution_context(self):
        return self.context

    @staticmethod
    def get_tensor_profile_shape(name, profile_index):
        assert name == "x"
        if profile_index == 0:
            return ((2, 80, 4), (2, 80, 500), (2, 80, 3000))
        assert profile_index == 1
        return ((4, 80, 4), (8, 80, 500), (16, 80, 1024))


def test_trt_context_wrapper_switches_profiles_for_cfg_batch(monkeypatch):
    context = _FakeContext()
    engine = _DynamicEngine(context)
    stream = _FakeStream()
    monkeypatch.setattr(flow_estimator_trt.torch.cuda, "Stream", lambda device: stream)

    wrapper = flow_estimator_trt.TrtContextWrapper(
        engine,
        device="cuda:0",
    )

    assert wrapper.supports_estimator_shape(2, 3000)
    assert not wrapper.supports_estimator_shape(2, 3001)
    assert wrapper.supports_estimator_shape(16, 1024)
    assert not wrapper.supports_estimator_shape(16, 1025)

    [ctx2, stream2], engine2 = wrapper.acquire_estimator(2, 3000)
    assert ctx2 is context
    assert stream2 is stream
    assert engine2 is engine
    assert context.profile_switches == []
    wrapper.release_estimator(ctx2, stream2)

    [ctx4, stream4], engine4 = wrapper.acquire_estimator(4, 1024)
    assert ctx4 is context
    assert engine4 is engine
    assert context.profile_switches == [(1, stream.cuda_stream)]
    wrapper.release_estimator(ctx4, stream4)

    [ctx8, stream8], _ = wrapper.acquire_estimator(8, 191)
    assert ctx8 is context
    assert context.profile_switches == [(1, stream.cuda_stream)]
    wrapper.release_estimator(ctx8, stream8)

    [ctx2_again, stream2_again], _ = wrapper.acquire_estimator(2, 3000)
    assert ctx2_again is context
    assert stream2_again is stream
    assert context.profile_switches == [
        (1, stream.cuda_stream),
        (0, stream.cuda_stream),
    ]
    # Profile switches and estimator enqueues share this stream, so CUDA
    # stream ordering is sufficient; no host-side synchronization is required.
    assert stream.synchronize_calls == 0


def test_trt_context_wrapper_rejects_batch_without_dynamic_profiles(monkeypatch):
    class StaticEngine:
        @staticmethod
        def create_execution_context():
            return object()

    monkeypatch.setattr(flow_estimator_trt.torch.cuda, "Stream", lambda device: _FakeStream())
    wrapper = flow_estimator_trt.TrtContextWrapper(StaticEngine(), device="cuda:0")

    assert wrapper.supports_estimator_shape(2, 3000)
    assert not wrapper.supports_estimator_shape(4, 4)
    with pytest.raises(RuntimeError, match="does not support CFG batch 4"):
        wrapper.acquire_estimator(4)


def test_trt_context_wrapper_returns_context_when_profile_switch_fails(monkeypatch):
    context = _FakeContext(switch_ok=False)
    engine = _DynamicEngine(context)
    monkeypatch.setattr(flow_estimator_trt.torch.cuda, "Stream", lambda device: _FakeStream())
    wrapper = flow_estimator_trt.TrtContextWrapper(
        engine,
        device="cuda:0",
    )

    with pytest.raises(RuntimeError, match="failed to select TensorRT optimization profile 1"):
        wrapper.acquire_estimator(4)

    [returned_context, stream], returned_engine = wrapper.acquire_estimator(2)
    assert returned_context is context
    assert returned_engine is engine
    wrapper.release_estimator(returned_context, stream)


def test_profile_switch_exception_returns_context(monkeypatch):
    class RaisingContext(_FakeContext):
        def set_optimization_profile_async(self, profile_index, stream_handle):
            raise RuntimeError("switch failed")

    context = RaisingContext()
    stream = _FakeStream()
    monkeypatch.setattr(flow_estimator_trt.torch.cuda, "Stream", lambda device: stream)
    wrapper = flow_estimator_trt.TrtContextWrapper(
        _DynamicEngine(context),
        device="cuda:0",
    )

    with pytest.raises(RuntimeError, match="switch failed"):
        wrapper.acquire_estimator(4)

    [returned_context, returned_stream], returned_engine = wrapper.acquire_estimator(2)
    assert returned_context is context
    assert returned_stream is stream
    assert returned_engine is wrapper.trt_engine
    wrapper.release_estimator(returned_context, returned_stream)


def test_dynamic_profiles_reject_concurrent_contexts(monkeypatch):
    monkeypatch.setattr(flow_estimator_trt.torch.cuda, "Stream", lambda device: _FakeStream())
    with pytest.raises(ValueError, match="trt_concurrent=1"):
        flow_estimator_trt.TrtContextWrapper(
            _DynamicEngine(_FakeContext()),
            device="cuda:0",
            trt_concurrent=2,
        )


class _BuildEngine:
    def __init__(self, profile_count: int = 1):
        self.context = object()
        self.num_optimization_profiles = profile_count

    def create_execution_context(self):
        return self.context


def _fake_tensorrt_module(engine):
    class Runtime:
        def __init__(self, logger):
            pass

        def deserialize_cuda_engine(self, payload):
            assert payload == b"plan"
            return engine

    module = types.ModuleType("tensorrt")
    setattr(module, "Runtime", Runtime)
    return module


@pytest.mark.parametrize(
    ("onnx_name", "expected_dtype"),
    [
        ("model.fp16.onnx", torch.float16),
        ("model.onnx", torch.float32),
    ],
)
@pytest.mark.parametrize(
    ("max_cfg_batch", "expected_prefix", "dynamic_profiles"),
    [
        (None, "flow_estimator", False),
        (
            16,
            (
                f"flow_estimator_dynamic_v{flow_estimator_trt._DYNAMIC_BATCH_PLAN_VERSION}"
                f"_b16_t{flow_estimator_trt._BATCH_MAX_T}"
            ),
            True,
        ),
    ],
)
def test_build_flow_estimator_uses_one_mode_specific_plan(
    tmp_path,
    monkeypatch,
    onnx_name,
    expected_dtype,
    max_cfg_batch,
    expected_prefix,
    dynamic_profiles,
):
    plan_path = tmp_path / "flow.plan"
    plan_path.write_bytes(b"plan")
    prefixes = []
    engine = _BuildEngine(2 if dynamic_profiles else 1)

    def resolve_plan_path(onnx_path, prefix="flow_estimator"):
        prefixes.append(prefix)
        return str(plan_path)

    monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt_module(engine))
    monkeypatch.setattr(
        flow_estimator_trt,
        "_resolve_plan_path",
        resolve_plan_path,
    )
    monkeypatch.setattr(flow_estimator_trt, "_trt_logger", lambda: object())
    monkeypatch.setattr(flow_estimator_trt.torch.cuda, "Stream", lambda device: object())

    wrapper = flow_estimator_trt.build_flow_estimator_trt(
        str(tmp_path / onnx_name),
        device="cuda:0",
        max_cfg_batch=max_cfg_batch,
    )

    assert prefixes == [expected_prefix]
    assert wrapper.trt_engine is engine
    assert wrapper.io_dtype == expected_dtype
    assert wrapper._dynamic_profiles is dynamic_profiles


def test_build_flow_estimator_builds_dynamic_plan_once(tmp_path, monkeypatch):
    plan_path = tmp_path / "flow.plan"
    calls = []
    engine = _BuildEngine(2)

    monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt_module(engine))
    monkeypatch.setattr(
        flow_estimator_trt,
        "_resolve_plan_path",
        lambda onnx_path, prefix="flow_estimator": str(plan_path),
    )
    monkeypatch.setattr(flow_estimator_trt, "_trt_logger", lambda: object())
    monkeypatch.setattr(flow_estimator_trt.torch.cuda, "Stream", lambda device: object())

    def fake_convert(onnx_path, output_path, strongly_typed, *, max_cfg_batch=None):
        calls.append((onnx_path, output_path, strongly_typed, max_cfg_batch))
        plan_path.write_bytes(b"plan")

    monkeypatch.setattr(flow_estimator_trt, "_convert_onnx_to_trt", fake_convert)

    onnx_path = str(tmp_path / "model.fp16.onnx")
    wrapper = flow_estimator_trt.build_flow_estimator_trt(
        onnx_path,
        device="cuda:0",
        max_cfg_batch=16,
    )

    assert calls == [(onnx_path, str(plan_path), True, 16)]
    assert wrapper.trt_engine is engine
    assert wrapper._dynamic_profiles


def test_context_session_binds_once_for_repeated_steps(monkeypatch):
    import contextlib

    import torch

    class FakeStream:
        cuda_stream = 123

        def __init__(self):
            self.waited_on = []

        def wait_stream(self, other):
            self.waited_on.append(other)

    class FakeContext:
        def __init__(self):
            self.shape_calls = []
            self.address_calls = []
            self.execute_calls = 0

        def set_input_shape(self, name, shape):
            self.shape_calls.append((name, shape))

        def set_tensor_address(self, name, address):
            self.address_calls.append((name, address))

        def execute_async_v3(self, stream):
            self.execute_calls += 1
            return True

    context = FakeContext()

    class FakeEngine:
        def create_execution_context(self):
            return context

        @staticmethod
        def get_tensor_name(index):
            return ("x", "mask", "mu", "t", "spks", "cond", "out")[index]

    estimator_stream = FakeStream()
    caller_stream = FakeStream()
    monkeypatch.setattr(torch.cuda, "Stream", lambda device: estimator_stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args, **kwargs: caller_stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())

    wrapper = flow_estimator_trt.TrtContextWrapper(FakeEngine(), device="cuda:0", io_dtype=torch.float16)
    inputs = (
        torch.randn(2, 80, 4),
        torch.ones(2, 1, 4),
        torch.randn(2, 80, 4),
        torch.randn(2),
        torch.randn(2, 80),
        torch.randn(2, 80, 4),
    )

    with wrapper.estimation_session(*inputs) as session:
        first = session.run(inputs[0], inputs[3])
        static_mask = session._input_buffers[1].clone()

        inputs[0].add_(1)
        inputs[1].add_(2)
        inputs[3].add_(1)
        second = session.run(inputs[0], inputs[3])

        assert torch.equal(session._input_buffers[0], inputs[0].to(torch.float16))
        assert torch.equal(session._input_buffers[3], inputs[3].to(torch.float16))
        # mask/mu/spks/cond are invariant during one Euler solve, so their
        # conversion scratch is populated once and then reused.
        assert torch.equal(session._input_buffers[1], static_mask)

        with pytest.raises(ValueError, match="input shape changed"):
            session.run(torch.randn(2, 80, 5), inputs[3])
        with pytest.raises(ValueError, match="input shape changed"):
            session.run(inputs[0], torch.randn(3))

    assert first.shape == inputs[0].shape
    assert first.dtype == inputs[0].dtype
    assert first.data_ptr() == second.data_ptr()
    assert len(context.shape_calls) == 6
    assert len(context.address_calls) == 7
    assert context.execute_calls == 2
    assert estimator_stream.waited_on == [caller_stream, caller_stream]
    assert caller_stream.waited_on == [estimator_stream, estimator_stream]
    [reused_context, reused_stream], _ = wrapper.acquire_estimator()
    assert reused_context is context
    assert reused_stream is estimator_stream
    wrapper.release_estimator(reused_context, reused_stream)


@pytest.mark.parametrize("caller_dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("n_timesteps", [3, 10])
@pytest.mark.parametrize("requests", [1, 2, 8])
def test_cfm_trt_session_matches_legacy_nonzero_estimator(monkeypatch, caller_dtype, n_timesteps, requests):
    from contextlib import contextmanager, nullcontext

    import torch
    from omegaconf import DictConfig

    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import (
        CausalConditionalCFM,
    )

    # Resolve bindings to live CPU tensors, while exercising the real session,
    # dtype conversions, CFM dispatch and legacy per-step binding path.
    tensors = {}
    data_ptr = torch.Tensor.data_ptr

    def register_tensor(tensor):
        address = data_ptr(tensor)
        tensors[address] = tensor
        return address

    monkeypatch.setattr(torch.Tensor, "data_ptr", register_tensor)
    names = ("x", "mask", "mu", "t", "spks", "cond", "out")

    class FakeStream:
        cuda_stream = 123

        def wait_stream(self, other):
            pass

    class FakeContext:
        def __init__(self):
            self.shapes = {}
            self.bindings = {}
            self.bind_count = 0
            self.calls = []
            self.after_first_call = None

        def set_input_shape(self, name, shape):
            self.shapes[name] = shape

        def set_optimization_profile_async(self, profile_index, stream_handle):
            return True

        def set_tensor_address(self, name, address):
            self.bindings[name] = tensors[address]
            self.bind_count += 1

        def execute_async_v3(self, stream):
            inputs = tuple(self.bindings[name] for name in names[:-1])
            assert all(tensor.dtype == torch.float16 for tensor in inputs)
            for name, tensor in zip(names, inputs):
                assert tuple(tensor.shape) == self.shapes[name]
            self.calls.append(tuple(tensor.clone() for tensor in inputs))
            x, mask, mu, t, spks, cond = inputs
            # Every input affects the result, including the unconditional CFG
            # row, time progression and engine/caller precision boundary.
            out = (0.125 * x + 0.25 * mu + 0.0625 * spks.unsqueeze(-1) + 0.125 * cond + 0.25 * t[:, None, None]) * mask
            self.bindings["out"].copy_(out)
            if len(self.calls) == 1 and self.after_first_call is not None:
                self.after_first_call()
            return True

    class FakeEngine:
        def __init__(self, profiles=1):
            self.context = FakeContext()
            self.num_optimization_profiles = profiles

        def create_execution_context(self):
            return self.context

        @staticmethod
        def get_tensor_name(index):
            return names[index]

        @staticmethod
        def get_tensor_profile_shape(name, profile_index):
            assert name == "x"
            if profile_index == 0:
                return ((2, 80, 4), (2, 80, 500), (2, 80, 3000))
            return ((4, 80, 4), (8, 80, 500), (16, 80, 1024))

    monkeypatch.setattr(torch.cuda, "Stream", lambda device: FakeStream())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args, **kwargs: FakeStream())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())

    session_engine, legacy_engine = FakeEngine(profiles=2), FakeEngine()
    session_pool = flow_estimator_trt.TrtContextWrapper(session_engine, "cpu", io_dtype=torch.float16)
    legacy_pool = flow_estimator_trt.TrtContextWrapper(legacy_engine, "cpu", io_dtype=torch.float16)
    monkeypatch.setattr(legacy_pool, "estimation_session", None)
    entries = exits = 0
    original_session_factory = session_pool.estimation_session

    @contextmanager
    def counted_session(*inputs):
        nonlocal entries, exits
        entries += 1
        try:
            with original_session_factory(*inputs) as session:
                yield session
        finally:
            exits += 1

    monkeypatch.setattr(session_pool, "estimation_session", counted_session)
    params = DictConfig(
        {
            "sigma_min": 1e-6,
            "solver": "euler",
            "t_scheduler": "cosine",
            "training_cfg_rate": 0.2,
            "inference_cfg_rate": 0.7,
        }
    )
    session_cfm = CausalConditionalCFM(80, params, n_spks=1, spk_emb_dim=80, estimator=session_pool)
    legacy_cfm = CausalConditionalCFM(80, params, n_spks=1, spk_emb_dim=80, estimator=legacy_pool)
    dtype = getattr(torch, caller_dtype)
    x = torch.linspace(-0.5, 0.5, requests * 80 * 4, dtype=dtype).reshape(requests, 80, 4)
    mu = torch.linspace(0.1, 0.7, requests * 80 * 4, dtype=dtype).reshape(requests, 80, 4)
    mask = torch.tensor([[[1.0, 1.0, 0.5, 0.0]]], dtype=dtype).expand(requests, 1, 4).clone()
    spks = torch.linspace(-0.3, 0.3, requests * 80, dtype=dtype).reshape(requests, 80)
    cond = torch.linspace(-0.2, 0.4, requests * 80 * 4, dtype=dtype).reshape(requests, 80, 4)
    t_span = torch.linspace(0, 1, n_timesteps + 1, dtype=dtype)
    t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

    original_mu, original_mask, original_spks, original_cond = (tensor.clone() for tensor in (mu, mask, spks, cond))

    def mutate_original_conditioning():
        # The real session must keep the initially staged CFG inputs even if
        # the original conditioning tensors change after the first enqueue.
        for tensor in (mu, mask, spks, cond):
            tensor.add_(2)

    session_engine.context.after_first_call = mutate_original_conditioning
    with torch.inference_mode():
        expected = legacy_cfm.solve_euler(x, t_span, mu, mask, spks, cond)
        actual = session_cfm.solve_euler(x, t_span, mu, mask, spks, cond)

    assert actual.dtype == torch.float32
    assert actual.shape == x.shape
    assert torch.isfinite(actual).all()
    assert not torch.equal(actual, x.float())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert entries == exits == 1
    assert session_pool._pool.qsize() == 1
    assert session_engine.context.bind_count == 7
    assert legacy_engine.context.bind_count == 7 * requests * n_timesteps
    assert len(session_engine.context.calls) == n_timesteps
    assert len(legacy_engine.context.calls) == requests * n_timesteps

    for step, actual_inputs in enumerate(session_engine.context.calls):
        pairs = legacy_engine.context.calls[step * requests : (step + 1) * requests]
        expected_inputs = tuple(
            torch.cat([pair[index][:1] for pair in pairs] + [pair[index][1:] for pair in pairs])
            for index in range(len(names) - 1)
        )
        for actual_input, expected_input in zip(actual_inputs, expected_inputs):
            torch.testing.assert_close(actual_input, expected_input, rtol=0, atol=0)
        x_in, mask_in, mu_in, t_in, spks_in, cond_in = actual_inputs
        assert torch.equal(x_in[:requests], x_in[requests:])
        assert torch.equal(t_in[:requests], t_in[requests:])
        assert torch.equal(mask_in, torch.cat((original_mask, original_mask)).to(torch.float16))
        for cfg_input, source in ((mu_in, original_mu), (spks_in, original_spks), (cond_in, original_cond)):
            assert torch.equal(cfg_input[:requests], source.to(torch.float16))
            assert torch.count_nonzero(cfg_input[requests:]) == 0

    first, last = session_engine.context.calls[0], session_engine.context.calls[-1]
    assert not torch.equal(first[0], last[0])
    assert not torch.equal(first[3], last[3])


@pytest.mark.parametrize(
    ("cfg_batch", "length", "expected_profile"),
    [(2, 4, 0), (4, 4, 1), (16, 1024, 1), (4, 1025, None), (18, 4, None)],
)
def test_dynamic_session_selects_profile_or_preserves_serial_fallback(monkeypatch, cfg_batch, length, expected_profile):
    class Context(_FakeContext):
        def set_input_shape(self, name, shape):
            pass

        def set_tensor_address(self, name, address):
            pass

    class Engine(_DynamicEngine):
        @staticmethod
        def get_tensor_name(index):
            return ("x", "mask", "mu", "t", "spks", "cond", "out")[index]

    context = Context()
    stream = _FakeStream()
    monkeypatch.setattr(torch.cuda, "Stream", lambda device: stream)
    wrapper = flow_estimator_trt.TrtContextWrapper(Engine(context), device="cpu")
    inputs = (
        torch.zeros(cfg_batch, 80, length),
        torch.ones(cfg_batch, 1, length),
        torch.zeros(cfg_batch, 80, length),
        torch.zeros(cfg_batch),
        torch.zeros(cfg_batch, 80),
        torch.zeros(cfg_batch, 80, length),
    )

    with wrapper.estimation_session(*inputs) as session:
        if expected_profile is None:
            assert session is None
            assert wrapper._pool.qsize() == 1
        else:
            assert session is not None
            assert wrapper._pool.qsize() == 0
            assert wrapper._active_profile_by_context[id(context)] == expected_profile

    assert wrapper._pool.qsize() == 1
    expected_switches = [(1, stream.cuda_stream)] if expected_profile == 1 else []
    assert context.profile_switches == expected_switches
