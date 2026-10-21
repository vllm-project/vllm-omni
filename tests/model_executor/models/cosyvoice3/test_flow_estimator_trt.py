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


@pytest.mark.parametrize("strongly_typed", [False, True])
@pytest.mark.parametrize("with_attn_mask", [False, True])
def test_chunk_mask_builder_disables_tf32(tmp_path, monkeypatch, strongly_typed, with_attn_mask):
    import sys
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    builder, parser, config = MagicMock(), MagicMock(), MagicMock()
    builder.create_builder_config.return_value = config
    builder.build_serialized_network.return_value = b"engine"
    parser.parse.return_value = True
    flags = SimpleNamespace(FP16="fp16", TF32="tf32")
    trt = SimpleNamespace(
        Builder=MagicMock(return_value=builder),
        OnnxParser=MagicMock(return_value=parser),
        NetworkDefinitionCreationFlag=SimpleNamespace(STRONGLY_TYPED=1),
        BuilderFlag=flags,
        MemoryPoolType=SimpleNamespace(WORKSPACE=0),
    )
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    monkeypatch.setattr(flow_estimator_trt, "_trt_logger", lambda: None)
    graph = tmp_path / "estimator.onnx"
    graph.write_bytes(b"graph")
    flow_estimator_trt._convert_onnx_to_trt(
        str(graph), str(tmp_path / "estimator.plan"), strongly_typed, with_attn_mask
    )
    if with_attn_mask:
        config.clear_flag.assert_called_once_with(flags.TF32)
        config.set_flag.assert_not_called()
    else:
        config.clear_flag.assert_not_called()
        if strongly_typed:
            config.set_flag.assert_not_called()
        else:
            config.set_flag.assert_called_once_with(flags.FP16)


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
            return True

        def set_tensor_address(self, name, address):
            self.address_calls.append((name, address))
            return True

        def execute_async_v3(self, stream):
            self.execute_calls += 1
            return True

    context = FakeContext()

    class FakeEngine:
        def create_execution_context(self):
            return context

        @staticmethod
        def get_tensor_name(index):
            return ("x", "mask", "mu", "t", "spks", "cond", "estimator_out")[index]

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
        supports_attn_mask = False

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
    names = ("x", "mask", "mu", "t", "spks", "cond", "estimator_out")

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
            return True

        def set_tensor_address(self, name, address):
            self.bindings[name] = tensors[address]
            self.bind_count += 1
            return True

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
            self.bindings["estimator_out"].copy_(out)
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
    session_pool = flow_estimator_trt.TrtContextWrapper(
        session_engine, "cpu", io_dtype=torch.float16, input_names=frozenset(names[:-1])
    )
    legacy_pool = flow_estimator_trt.TrtContextWrapper(
        legacy_engine, "cpu", io_dtype=torch.float16, input_names=frozenset(names[:-1])
    )
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


class TestExportCacheKey:
    """Exported ONNX bakes in the checkpoint's weights, so its cache path must
    change with the checkpoint even when the export directory is shared."""

    def test_fingerprint_differs_across_checkpoint_dirs(self, tmp_path):
        a, b = tmp_path / "ckpt_a", tmp_path / "ckpt_b"
        for d in (a, b):
            d.mkdir()
            (d / "flow.pt").write_bytes(b"same bytes")
        assert flow_estimator_trt.flow_checkpoint_fingerprint(str(a)) != flow_estimator_trt.flow_checkpoint_fingerprint(
            str(b)
        )

    def test_fingerprint_changes_when_weights_change_in_place(self, tmp_path):
        d = tmp_path / "ckpt"
        d.mkdir()
        (d / "flow.pt").write_bytes(b"v1")
        before = flow_estimator_trt.flow_checkpoint_fingerprint(str(d))
        assert before == flow_estimator_trt.flow_checkpoint_fingerprint(str(d))
        (d / "flow.pt").write_bytes(b"v2 longer")
        assert flow_estimator_trt.flow_checkpoint_fingerprint(str(d)) != before

    def test_missing_weights_cannot_share_an_export_key(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            flow_estimator_trt.flow_checkpoint_fingerprint(str(tmp_path))

    def test_fingerprint_changes_with_export_version(self, tmp_path, monkeypatch):
        (tmp_path / "flow.pt").write_bytes(b"weights")
        before = flow_estimator_trt.flow_checkpoint_fingerprint(str(tmp_path))
        monkeypatch.setattr(flow_estimator_trt, "_CHUNK_MASK_EXPORT_VERSION", 999)
        assert flow_estimator_trt.flow_checkpoint_fingerprint(str(tmp_path)) != before

    def test_onnx_path_carries_the_key(self, tmp_path):
        keyed = flow_estimator_trt.chunk_mask_estimator_onnx_path(str(tmp_path), fp16=True, cache_key="abc123")
        plain = flow_estimator_trt.chunk_mask_estimator_onnx_path(str(tmp_path), fp16=True, cache_key=None)
        assert keyed != plain
        assert keyed.endswith(".abc123.onnx") and "chunk_mask.autocast_fp16" in keyed
        assert plain.endswith("flow.decoder.estimator.chunk_mask.autocast_fp16.onnx")


class TestFusedAttentionFallback:
    def _patch_builder(self, monkeypatch, fail_fused: bool):
        calls = []

        def fake_build(estimator, onnx_dir, device, *, fp16, cache_key, fused_attention):
            calls.append((fp16, fused_attention))
            if fused_attention and fail_fused:
                raise RuntimeError("UNSUPPORTED_NODE: Attention")
            return f"engine(fused={fused_attention})"

        monkeypatch.setattr(flow_estimator_trt, "_build_chunk_mask_engine", fake_build)
        return calls

    def test_default_uses_fp32_throughout(self, tmp_path, monkeypatch):
        calls = self._patch_builder(monkeypatch, fail_fused=False)
        got = flow_estimator_trt.build_chunk_mask_flow_estimator_trt(object(), str(tmp_path), "cuda")
        assert got == "engine(fused=False)" and calls == [(False, False)]

    def test_explicit_fp16_prefers_fused_attention(self, tmp_path, monkeypatch):
        calls = self._patch_builder(monkeypatch, fail_fused=False)
        got = flow_estimator_trt.build_chunk_mask_flow_estimator_trt(object(), str(tmp_path), "cuda", fp16=True)
        assert got == "engine(fused=True)" and calls == [(True, True)]

    def test_unbuildable_fused_engine_falls_back_to_fp32_attention(self, tmp_path, monkeypatch):
        calls = self._patch_builder(monkeypatch, fail_fused=True)
        got = flow_estimator_trt.build_chunk_mask_flow_estimator_trt(object(), str(tmp_path), "cuda", fp16=True)
        assert got == "engine(fused=False)" and calls == [(True, True), (True, False)]

    def test_fused_and_fp32_attention_exports_do_not_share_a_file(self, tmp_path):
        fused = flow_estimator_trt.chunk_mask_estimator_onnx_path(
            str(tmp_path), fp16=True, cache_key="k", fused_attention=True
        )
        plain = flow_estimator_trt.chunk_mask_estimator_onnx_path(str(tmp_path), fp16=True, cache_key="k")
        assert fused != plain and "fused_attn" in fused


@pytest.fixture
def cached_chunk_builder(tmp_path, monkeypatch):
    """Exercise real cache keys and file locks without a GPU or TRT install."""
    import sys
    from types import SimpleNamespace

    import torch

    exports, builds = [], []

    class Runtime:
        def __init__(self, logger):
            pass

        def deserialize_cuda_engine(self, data):
            return data

    def export(estimator, path, **kwargs):
        exports.append(path)
        Path(path).write_text(str(estimator.weight.item()))

    def convert(onnx_path, plan_path, **kwargs):
        builds.append(plan_path)
        flow_estimator_trt._write_plan_atomically(Path(onnx_path).read_bytes(), plan_path)

    monkeypatch.setenv("COSYVOICE3_TRT_CACHE", str(tmp_path))
    monkeypatch.setitem(sys.modules, "tensorrt", SimpleNamespace(Runtime=Runtime, __version__="test"))
    monkeypatch.setattr(flow_estimator_trt, "_trt_logger", lambda: None)
    monkeypatch.setattr(flow_estimator_trt, "export_chunk_mask_estimator_onnx", export)
    monkeypatch.setattr(flow_estimator_trt, "_convert_onnx_to_trt", convert)
    monkeypatch.setattr(
        flow_estimator_trt,
        "_wrap_engine",
        lambda engine, *args, **kwargs: SimpleNamespace(engine=engine, supports_attn_mask=True),
    )

    def build(checkpoint, value=1.0, **kwargs):
        estimator = torch.nn.Linear(1, 1, bias=False)
        estimator.static_chunk_size = 50
        with torch.no_grad():
            estimator.weight.fill_(value)
        kwargs.setdefault("fp16", False)
        return flow_estimator_trt.build_chunk_mask_flow_estimator_trt(
            estimator,
            str(tmp_path / "exported"),
            "cpu",
            cache_key=flow_estimator_trt.flow_checkpoint_fingerprint(str(checkpoint.parent), checkpoint.name),
            **kwargs,
        )

    return build, exports, builds


def test_chunk_cache_separates_checkpoints_and_reuses_matching_weights(tmp_path, cached_chunk_builder):
    build, exports, builds = cached_chunk_builder
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    a.write_bytes(b"checkpoint A")
    b.write_bytes(b"checkpoint B")

    first = build(a, 1.0)
    second = build(b, 2.0)
    repeated = build(a, 1.0)

    assert first.engine == repeated.engine == b"1.0"
    assert second.engine == b"2.0"
    assert len(exports) == len(builds) == 2
    assert exports[0] != exports[1]


def test_chunk_cache_detects_same_size_and_mtime_checkpoint_replacement(tmp_path, cached_chunk_builder):
    import os

    build, exports, builds = cached_chunk_builder
    checkpoint = tmp_path / "flow.pt"
    checkpoint.write_bytes(b"original")
    original_stat = checkpoint.stat()
    build(checkpoint, 1.0)
    checkpoint.write_bytes(b"replaced")
    os.utime(checkpoint, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    assert build(checkpoint, 2.0).engine == b"2.0"
    assert len(exports) == len(builds) == 2


@pytest.mark.parametrize("change", ["precision", "export_version", "architecture", "profile"])
def test_chunk_cache_invalidates_export_or_plan_configuration(tmp_path, monkeypatch, cached_chunk_builder, change):
    build, exports, builds = cached_chunk_builder
    checkpoint = tmp_path / "flow.pt"
    checkpoint.write_bytes(b"weights")
    build(checkpoint)

    kwargs = {}
    if change == "precision":
        kwargs["fp16"] = True
    elif change == "export_version":
        monkeypatch.setattr(
            flow_estimator_trt, "_CHUNK_MASK_EXPORT_VERSION", flow_estimator_trt._CHUNK_MASK_EXPORT_VERSION + 1
        )
    elif change == "architecture":
        import torch

        monkeypatch.setattr(torch.nn.Linear, "extra_repr", lambda self: "changed architecture")
    else:
        monkeypatch.setattr(flow_estimator_trt, "_MASK_MAX_SHAPE", (2, 1, 4000, 4000))
    build(checkpoint, **kwargs)

    assert len(builds) == 2
    assert len(exports) == (1 if change == "profile" else 2)


def test_concurrent_chunk_builders_publish_one_export_and_plan(tmp_path, monkeypatch, cached_chunk_builder):
    build, exports, builds = cached_chunk_builder
    checkpoint = tmp_path / "flow.pt"
    checkpoint.write_bytes(b"weights")
    original_export = flow_estimator_trt.export_chunk_mask_estimator_onnx
    export_started, second_entered = threading.Event(), threading.Event()
    original_path = flow_estimator_trt._chunk_mask_graph_cache_key

    def export(*args, **kwargs):
        export_started.set()
        assert second_entered.wait(timeout=5)
        return original_export(*args, **kwargs)

    def export_path(*args, **kwargs):
        path = original_path(*args, **kwargs)
        if export_started.is_set():
            second_entered.set()
        return path

    monkeypatch.setattr(flow_estimator_trt, "export_chunk_mask_estimator_onnx", export)
    monkeypatch.setattr(flow_estimator_trt, "_chunk_mask_graph_cache_key", export_path)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(build, checkpoint)
        assert export_started.wait(timeout=5)
        second = executor.submit(build, checkpoint)
        assert first.result(timeout=10).engine == second.result(timeout=10).engine
    assert len(exports) == len(builds) == 1


@pytest.mark.parametrize("stage", ["export_chunk_mask_estimator_onnx", "_convert_onnx_to_trt"])
def test_failed_chunk_build_is_retried(tmp_path, monkeypatch, cached_chunk_builder, stage):
    build, exports, builds = cached_chunk_builder
    checkpoint = tmp_path / "flow.pt"
    checkpoint.write_bytes(b"weights")

    def fail(*args, **kwargs):
        raise RuntimeError("build failed")

    with monkeypatch.context() as patch:
        patch.setattr(flow_estimator_trt, stage, fail)
        with pytest.raises(RuntimeError, match="build failed"):
            build(checkpoint)
    assert build(checkpoint).engine == b"1.0"
    assert len(exports) == len(builds) == 1


@pytest.mark.parametrize("failure", ["export", "replace", None])
def test_onnx_publication_restores_estimator_and_cleans_temporary_files(tmp_path, monkeypatch, failure):
    import sys
    from types import ModuleType

    import torch

    monkeypatch.setitem(sys.modules, "onnx", ModuleType("onnx"))
    estimator = torch.nn.Linear(1, 1).train()
    onnx_path = tmp_path / "estimator.onnx"
    onnx_path.write_bytes(b"previous graph")

    def export(wrapper, inputs, path, **kwargs):
        assert not wrapper.estimator.training
        Path(path).write_bytes(b"new graph")
        if failure == "export":
            raise RuntimeError("export failed")

    def fail_replace(*args):
        raise RuntimeError("replace failed")

    monkeypatch.setattr(torch.onnx, "export", export)
    if failure == "replace":
        monkeypatch.setattr(flow_estimator_trt.os, "replace", fail_replace)
    if failure:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            flow_estimator_trt.export_chunk_mask_estimator_onnx(estimator, str(onnx_path), fp16=False)
    else:
        flow_estimator_trt.export_chunk_mask_estimator_onnx(estimator, str(onnx_path), fp16=False)
    assert estimator.training
    assert onnx_path.read_bytes() == (b"previous graph" if failure else b"new graph")
    assert _temporary_plans(onnx_path) == []


def test_context_creation_failure_raises_runtime_error(monkeypatch):
    from types import SimpleNamespace

    engine = SimpleNamespace(create_execution_context=lambda: None)
    with pytest.raises(RuntimeError, match="failed to create TRT execution context"):
        flow_estimator_trt.TrtContextWrapper(engine, "cpu")


@pytest.mark.parametrize("failure", ["shape", "input", "output", "enqueue", "exception"])
def test_session_failure_returns_context_and_allows_retry(monkeypatch, failure):
    from contextlib import nullcontext
    from types import SimpleNamespace

    import torch

    class Context:
        def set_input_shape(self, name, shape):
            return True

        def set_tensor_address(self, name, address):
            return True

        def execute_async_v3(self, stream):
            return True

    context = Context()
    stream = SimpleNamespace(cuda_stream=0, wait_stream=lambda other: None)
    monkeypatch.setattr(torch.cuda, "Stream", lambda device: stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    wrapper = flow_estimator_trt.TrtContextWrapper(SimpleNamespace(create_execution_context=lambda: context), "cpu")
    inputs = (
        torch.ones(2, 80, 4),
        torch.ones(2, 1, 4),
        torch.ones(2, 80, 4),
        torch.ones(2),
        torch.ones(2, 80),
        torch.ones(2, 80, 4),
    )
    with monkeypatch.context() as patch:
        if failure == "shape":
            patch.setattr(context, "set_input_shape", lambda *args: False)
        elif failure in ("input", "output"):
            target = "x" if failure == "input" else "estimator_out"
            patch.setattr(context, "set_tensor_address", lambda name, address: name != target)
        elif failure == "enqueue":
            patch.setattr(context, "execute_async_v3", lambda *args: False)
        else:

            def fail(*args):
                raise RuntimeError("execution error")

            patch.setattr(context, "execute_async_v3", fail)
        with pytest.raises(RuntimeError):
            with wrapper.estimation_session(*inputs) as session:
                session.run(*inputs)
    assert wrapper._pool.qsize() == 1
    with wrapper.estimation_session(*inputs) as session:
        session.run(*inputs)
    assert wrapper._pool.qsize() == 1


def test_optimized_session_still_enqueues_and_uses_engine_output_dtype(monkeypatch):
    import inspect
    import textwrap
    from contextlib import nullcontext
    from types import SimpleNamespace

    import torch

    source = textwrap.dedent(inspect.getsource(flow_estimator_trt._TrtEstimatorSession.run))
    namespace = dict(vars(flow_estimator_trt))
    exec(compile(source, "<optimized-trt-session>", "exec", optimize=2), namespace)
    monkeypatch.setattr(flow_estimator_trt._TrtEstimatorSession, "run", namespace["run"])
    calls = []

    def enqueue(stream):
        calls.append(stream)
        return True

    context = SimpleNamespace(
        set_input_shape=lambda *args: True,
        set_tensor_address=lambda *args: True,
        execute_async_v3=enqueue,
    )
    stream = SimpleNamespace(cuda_stream=0, wait_stream=lambda other: None)
    monkeypatch.setattr(torch.cuda, "Stream", lambda device: stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    wrapper = flow_estimator_trt.TrtContextWrapper(
        SimpleNamespace(create_execution_context=lambda: context),
        "cpu",
        io_dtype=torch.float16,
        out_dtype=torch.float32,
    )
    inputs = (
        torch.ones(2, 80, 4),
        torch.ones(2, 1, 4),
        torch.ones(2, 80, 4),
        torch.ones(2),
        torch.ones(2, 80),
        torch.ones(2, 80, 4),
    )
    with wrapper.estimation_session(*inputs) as session:
        assert session._engine_output.dtype == torch.float32
        session.run(*inputs)
    assert calls == [0]
