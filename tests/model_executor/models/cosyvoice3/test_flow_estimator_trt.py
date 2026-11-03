# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

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
        first = session.run(*inputs)
        static_mask = session._input_buffers[1].clone()

        inputs[0].add_(1)
        inputs[1].add_(2)
        inputs[3].add_(1)
        second = session.run(*inputs)

        assert torch.equal(session._input_buffers[0], inputs[0].to(torch.float16))
        assert torch.equal(session._input_buffers[3], inputs[3].to(torch.float16))
        # mask/mu/spks/cond are invariant during one Euler solve, so their
        # conversion scratch is populated once and then reused.
        assert torch.equal(session._input_buffers[1], static_mask)

        bad_inputs = (torch.randn(2, 80, 5), *inputs[1:])
        with pytest.raises(ValueError, match="input shape changed"):
            session.run(*bad_inputs)

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


def test_cfm_reuses_one_trt_session_across_euler_steps():
    from contextlib import contextmanager

    import torch
    from omegaconf import DictConfig

    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import (
        CausalConditionalCFM,
    )

    class FakeSession:
        def __init__(self):
            self.calls = 0
            self.static_inputs = []

        def run(self, x, mask_in, mu_in, t, spks_in, cond_in):
            self.calls += 1
            self.static_inputs.append(tuple(tensor.clone() for tensor in (mask_in, mu_in, spks_in, cond_in)))
            if self.calls == 1:
                mask.add_(2)
                mu.add_(2)
                spks.add_(2)
                cond.add_(2)
            return torch.zeros_like(x)

    class FakeEstimatorPool:
        def __init__(self):
            self.session = FakeSession()
            self.entries = 0
            self.exits = 0

        @contextmanager
        def estimation_session(self, *inputs):
            self.entries += 1
            try:
                yield self.session
            finally:
                self.exits += 1

        def acquire_estimator(self):
            pytest.fail("per-step TRT context acquisition must not be used")

    pool = FakeEstimatorPool()
    cfm = CausalConditionalCFM(
        in_channels=80,
        cfm_params=DictConfig(
            {
                "sigma_min": 1e-6,
                "solver": "euler",
                "t_scheduler": "cosine",
                "training_cfg_rate": 0.2,
                "inference_cfg_rate": 0.7,
            }
        ),
        n_spks=1,
        spk_emb_dim=80,
        estimator=pool,
    )
    mu = torch.randn(1, 80, 16)
    mask = torch.ones(1, 1, 16)
    spks = torch.randn(1, 80)
    cond = torch.randn(1, 80, 16)

    out, _ = cfm(mu, mask, n_timesteps=3, spks=spks, cond=cond)

    assert out.shape == mu.shape
    assert pool.entries == 1
    assert pool.exits == 1
    assert pool.session.calls == 3

    first_static = pool.session.static_inputs[0]
    for step_static in pool.session.static_inputs[1:]:
        for first, current in zip(first_static, step_static):
            assert torch.equal(first, current)


@pytest.mark.parametrize("caller_dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("n_timesteps", [3, 10])
def test_cfm_trt_session_matches_legacy_nonzero_estimator(monkeypatch, caller_dtype, n_timesteps):
    from contextlib import nullcontext

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

        def set_input_shape(self, name, shape):
            self.shapes[name] = shape

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
            return True

    class FakeEngine:
        def __init__(self):
            self.context = FakeContext()

        def create_execution_context(self):
            return self.context

        @staticmethod
        def get_tensor_name(index):
            return names[index]

    monkeypatch.setattr(torch.cuda, "Stream", lambda device: FakeStream())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args, **kwargs: FakeStream())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())

    session_engine, legacy_engine = FakeEngine(), FakeEngine()
    session_pool = flow_estimator_trt.TrtContextWrapper(session_engine, "cpu", io_dtype=torch.float16)
    legacy_pool = flow_estimator_trt.TrtContextWrapper(legacy_engine, "cpu", io_dtype=torch.float16)
    monkeypatch.setattr(legacy_pool, "estimation_session", None)
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
    x = torch.linspace(-0.5, 0.5, 80 * 4, dtype=dtype).reshape(1, 80, 4)
    mu = torch.linspace(0.1, 0.7, 80 * 4, dtype=dtype).reshape(1, 80, 4)
    mask = torch.tensor([[[1.0, 1.0, 0.5, 0.0]]], dtype=dtype)
    spks = torch.linspace(-0.3, 0.3, 80, dtype=dtype).reshape(1, 80)
    cond = torch.linspace(-0.2, 0.4, 80 * 4, dtype=dtype).reshape(1, 80, 4)
    t_span = torch.linspace(0, 1, n_timesteps + 1, dtype=dtype)
    t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

    with torch.inference_mode():
        expected = legacy_cfm.solve_euler(x, t_span, mu, mask, spks, cond)
        actual = session_cfm.solve_euler(x, t_span, mu, mask, spks, cond)

    assert actual.dtype == torch.float32
    assert actual.shape == x.shape
    assert torch.isfinite(actual).all()
    assert not torch.equal(actual, x.float())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert session_engine.context.bind_count == 7
    assert legacy_engine.context.bind_count == 7 * n_timesteps
    assert len(session_engine.context.calls) == len(legacy_engine.context.calls) == n_timesteps

    for actual_inputs, expected_inputs in zip(session_engine.context.calls, legacy_engine.context.calls):
        for actual_input, expected_input in zip(actual_inputs, expected_inputs):
            torch.testing.assert_close(actual_input, expected_input, rtol=0, atol=0)
        x_in, mask_in, mu_in, t_in, spks_in, cond_in = actual_inputs
        assert torch.equal(x_in[0], x_in[1])
        assert torch.equal(t_in[0], t_in[1])
        assert torch.equal(mask_in, mask.to(torch.float16).expand_as(mask_in))
        for cfg_input, source in ((mu_in, mu), (spks_in, spks), (cond_in, cond)):
            assert torch.equal(cfg_input[:1], source.to(torch.float16))
            assert torch.count_nonzero(cfg_input[1:]) == 0

    first, last = session_engine.context.calls[0], session_engine.context.calls[-1]
    assert not torch.equal(first[0], last[0])
    assert not torch.equal(first[3], last[3])
