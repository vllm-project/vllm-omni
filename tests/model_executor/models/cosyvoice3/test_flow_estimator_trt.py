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
