# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for PipelineParallelMixin and related PP communication helpers.

CPU unit tests exercise mixin logic without a process group. Distributed CPU
tests use gloo on CPU tensors; nightly parity tests run the same checks on
real multi-GPU NCCL collectives.
"""

from __future__ import annotations

import os
import socket
from typing import Literal

import pytest
import torch
from vllm.model_executor.models.utils import PPMissingLayer, make_empty_intermediate_tensors_factory, make_layers
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu_worker import AsyncIntermediateTensors

import vllm_omni.diffusion.distributed.pipeline_parallel as pp_module
from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    get_classifier_free_guidance_rank,
    get_pp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.distributed.pipeline_parallel import AsyncLatents, PipelineParallelMixin
from vllm_omni.platforms import current_omni_platform

_L4_TWO_GPU = hardware_marks(res={"cuda": "L4"}, num_cards=2)
_L4_THREE_GPU = hardware_marks(res={"cuda": "L4"}, num_cards=3)
_L4_FOUR_GPU = hardware_marks(res={"cuda": "L4"}, num_cards=4)

DeviceKind = Literal["cpu", "cuda"]
_UNIT_MARKS = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _find_free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return str(s.getsockname()[1])


def update_environment_variables(envs_dict: dict[str, str]) -> None:
    for k, v in envs_dict.items():
        os.environ[k] = v


def _cleanup_distributed() -> None:
    """Destroy omni distributed state and clear test-process leftovers.

    ``destroy_distributed_env`` leaves ``vllm.distributed.parallel_state._PP``
    aliased to the destroyed group. Parent-process baselines must clear that
    dangling reference (and RANK/MASTER_* env) so later unit tests that call
    ``vllm.distributed.initialize_model_parallel`` do not see stale state.
    """
    destroy_distributed_env()
    import vllm.distributed.parallel_state as vllm_parallel_state

    vllm_parallel_state._PP = None
    for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"):
        os.environ.pop(key, None)


# ---------------------------------------------------------------------------
# Shared stubs used by both unit and distributed tests
# ---------------------------------------------------------------------------


class FakeWork:
    """Drop-in for torch.distributed.Work that records whether wait() was called."""

    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


class SimpleScheduler:
    """Minimal diffusion-step scheduler: latents -= 0.1 * noise_pred."""

    def step(self, noise_pred: torch.Tensor, t, latents: torch.Tensor, return_dict: bool = False):
        return (latents - 0.1 * noise_pred,)


class FakeVAE:
    def __init__(self, distributed_enabled: bool = False):
        self.calls = 0
        self.distributed_enabled = distributed_enabled

    def decode(self, z: torch.Tensor):
        """Original decode docstring."""
        self.calls += 1
        return (z + 1,)

    def is_distributed_enabled(self) -> bool:
        return self.distributed_enabled


class MockPipelineParallel(PipelineParallelMixin, CFGParallelMixin):
    """Minimal pipeline used to exercise PipelineParallelMixin.

    Uses vLLM's ``make_layers`` for layer partitioning — the same utility used
    by real DiT models — so the PP layer-split logic is exercised faithfully.

    Each layer's weights are seeded by ``seed + layer_index`` so that layer ``i``
    is initialized identically on every rank regardless of which ranks are active,
    allowing the distributed output to be compared against the single-GPU baseline.

    Args:
        num_layers: Total number of Linear layers.
        dim:        Input / hidden dimension.
        seed:       Base RNG seed; layer ``i`` uses ``seed + i``.
        device:     Target device for layer weights (default: CPU).
        dtype:      Target dtype for layer weights (default: float32).
    """

    def __init__(
        self,
        num_layers: int = 4,
        dim: int = 64,
        seed: int = 42,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.start_layer, self.end_layer, self.layers = make_layers(
            num_layers,
            lambda prefix: torch.nn.Linear(dim, dim, bias=False),
            prefix="layers",
        )

        for i, layer in enumerate(self.layers):
            if not isinstance(layer, PPMissingLayer):
                torch.manual_seed(seed + i)
                torch.nn.init.normal_(layer.weight, mean=0.0, std=0.02)

        self.layers.to(device=device, dtype=dtype)

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(["hidden_states"], dim)
        self.scheduler = SimpleScheduler()

    def predict_noise(self, x=None, intermediate_tensors=None, **_kwargs) -> torch.Tensor | IntermediateTensors:
        """Layer-split forward pass.

        * First PP rank: uses ``x`` from caller kwargs.
        * Later PP ranks: overrides ``x`` with ``intermediate_tensors["hidden_states"]``
          (which transparently waits for the async receive).
        * Non-last PP ranks return ``IntermediateTensors``; the last rank
          returns the plain noise-prediction tensor.
        """
        if intermediate_tensors is not None:
            x = intermediate_tensors["hidden_states"]

        for i in range(self.start_layer, self.end_layer):
            x = self.layers[i](x)

        pp_group = get_pp_group()
        if not pp_group.is_last_rank:
            return IntermediateTensors({"hidden_states": x})
        return x


# ---------------------------------------------------------------------------
# 1.  AsyncLatents – unit tests (no distributed env required)
# ---------------------------------------------------------------------------


class TestAsyncLatents:
    """Verifies the lazy-resolution behaviour of AsyncLatents without a real process group."""

    pytestmark = _UNIT_MARKS

    def _make(self, tensor: torch.Tensor, handles: list | None = None, postproc: list | None = None) -> AsyncLatents:
        return AsyncLatents({"latents": tensor}, handles or [], postproc or [])

    def test_resolve_returns_wrapped_tensor(self):
        t = torch.randn(2, 4)
        al = self._make(t)
        assert al._resolve() is t

    def test_attribute_access_resolves(self):
        t = torch.randn(2, 4)
        al = self._make(t)
        assert al.shape == t.shape
        assert al.dtype == t.dtype

    def test_torch_function_protocol(self):
        """torch ops that receive an AsyncLatents should see the underlying tensor."""
        t = torch.randn(2, 4)
        al = self._make(t)
        mask = torch.ones_like(t)
        result = mask * al  # triggers __torch_function__
        torch.testing.assert_close(result, mask * t)

    def test_torch_function_with_list_arg(self):
        """__torch_function__ must unwrap AsyncLatents inside list/tuple args."""
        t = torch.randn(2, 4)
        al = self._make(t)
        result = torch.cat([al, al], dim=0)
        torch.testing.assert_close(result, torch.cat([t, t], dim=0))

    def test_torch_tensor_conversion(self):
        """torch.as_tensor on an AsyncLatents must share storage with the underlying tensor (no copy)."""
        t = torch.randn(2, 4)
        al = self._make(t)
        result = torch.as_tensor(al)
        assert result.data_ptr() == t.data_ptr(), "torch.as_tensor copied the data instead of sharing storage"

    def test_handles_are_waited_before_resolve(self):
        t = torch.randn(2, 4)
        h1, h2 = FakeWork(), FakeWork()
        al = self._make(t, handles=[h1, h2])
        _ = al.shape  # trigger resolution
        assert h1.waited and h2.waited, "Not all handles were waited on"

    def test_postproc_callbacks_invoked_on_resolve(self):
        t = torch.randn(2, 4)
        log: list[int] = []
        al = self._make(t, postproc=[lambda: log.append(1), lambda: log.append(2)])
        _ = al.shape
        assert log == [1, 2], f"postproc not called in order: {log}"

    def test_idempotent_resolve(self):
        """handle.wait() must not be called twice if _resolve() is called twice."""
        t = torch.randn(2, 4)
        h = FakeWork()
        al = self._make(t, handles=[h])
        _ = al.shape  # first resolve
        h.waited = False  # reset sentinel
        _ = al.dtype  # second resolve
        assert not h.waited, "handle.wait() was called a second time"


# ---------------------------------------------------------------------------
# 2.  _sync_pp_send / diffuse wrapper – unit tests (no distributed env required)
# ---------------------------------------------------------------------------


class TestSyncPPSend:
    """Verifies PipelineParallelMixin's internal PP-send flush."""

    pytestmark = _UNIT_MARKS

    @staticmethod
    def _make_pipeline() -> PipelineParallelMixin:
        # Instantiate a bare mixin — no layers, no distributed env needed.
        # _sync_pp_send only touches _pp_send_work, so this is sufficient.
        class _BarePP(PipelineParallelMixin, CFGParallelMixin):
            pass

        return _BarePP()

    def test_noop_when_work_list_empty(self):
        pipeline = self._make_pipeline()
        pipeline._sync_pp_send()
        assert pipeline._pp_send_work == []

    def test_waits_all_pending_handles(self):
        pipeline = self._make_pipeline()
        works = [FakeWork(), FakeWork(), FakeWork()]
        pipeline._pp_send_work = works
        pipeline._sync_pp_send()
        assert all(w.waited for w in works), "Some handles were not waited on"

    def test_clears_work_list_after_sync(self):
        pipeline = self._make_pipeline()
        pipeline._pp_send_work = [FakeWork()]
        pipeline._sync_pp_send()
        assert pipeline._pp_send_work == []


class TestDiffuseWrapper:
    """Verifies that PipelineParallelMixin flushes pending sends when diffuse() exits."""

    pytestmark = _UNIT_MARKS

    def test_diffuse_flushes_pending_sends_on_success(self):
        work = FakeWork()

        class _DiffusePP(PipelineParallelMixin, CFGParallelMixin):
            def diffuse(self):
                self._pp_send_work = [work]
                return "done"

        pipeline = _DiffusePP()

        assert pipeline.diffuse() == "done"
        assert work.waited
        assert pipeline._pp_send_work == []

    def test_diffuse_flushes_pending_sends_on_exception(self):
        work = FakeWork()

        class _DiffusePP(PipelineParallelMixin, CFGParallelMixin):
            def diffuse(self):
                self._pp_send_work = [work]
                raise RuntimeError("boom")

        pipeline = _DiffusePP()

        with pytest.raises(RuntimeError, match="boom"):
            pipeline.diffuse()
        assert work.waited
        assert pipeline._pp_send_work == []

    def test_diffuse_wrapper_preserves_metadata(self):
        class _DiffusePP(PipelineParallelMixin, CFGParallelMixin):
            def diffuse(self):
                """Original diffuse docstring."""
                return "done"

        assert _DiffusePP.diffuse.__name__ == "diffuse"
        assert _DiffusePP.diffuse.__doc__ == "Original diffuse docstring."


class TestVaeDecodeGuard:
    pytestmark = _UNIT_MARKS

    @staticmethod
    def _make_pipeline(distributed_enabled: bool = False) -> PipelineParallelMixin:
        class _DecodePP(PipelineParallelMixin, CFGParallelMixin):
            def __init__(self):
                self.vae = FakeVAE(distributed_enabled=distributed_enabled)

        return _DecodePP()

    @staticmethod
    def _set_rank(monkeypatch, world_size: int, first_stage: bool) -> None:
        monkeypatch.setattr(pp_module, "get_pipeline_parallel_world_size", lambda: world_size)
        monkeypatch.setattr(pp_module, "is_pipeline_first_stage", lambda: first_stage)

    def test_calls_original_decode_when_pp_disabled(self, monkeypatch):
        self._set_rank(monkeypatch, world_size=1, first_stage=True)
        pipeline = self._make_pipeline()
        z = torch.ones(2, 3)

        output = pipeline.vae.decode(z)[0]

        assert pipeline.vae.calls == 1
        torch.testing.assert_close(output, z + 1)

    def test_calls_original_decode_on_first_stage(self, monkeypatch):
        self._set_rank(monkeypatch, world_size=2, first_stage=True)
        pipeline = self._make_pipeline()
        z = torch.ones(2, 3)

        output = pipeline.vae.decode(z)[0]

        assert pipeline.vae.calls == 1
        torch.testing.assert_close(output, z + 1)

    def test_skips_decode_on_non_first_stage(self, monkeypatch):
        self._set_rank(monkeypatch, world_size=2, first_stage=False)
        pipeline = self._make_pipeline()
        z = torch.ones(2, 3)

        output = pipeline.vae.decode(z)

        assert pipeline.vae.calls == 0
        assert output == (None,)

    def test_calls_original_decode_when_distributed_vae_enabled(self, monkeypatch):
        self._set_rank(monkeypatch, world_size=2, first_stage=False)
        pipeline = self._make_pipeline(distributed_enabled=True)
        z = torch.ones(2, 3)

        output = pipeline.vae.decode(z)[0]

        assert pipeline.vae.calls == 1
        torch.testing.assert_close(output, z + 1)

    def test_decode_wrapper_preserves_metadata(self):
        pipeline = self._make_pipeline()

        assert pipeline.vae.decode.__name__ == "decode"
        assert pipeline.vae.decode.__doc__ == "Original decode docstring."


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
def test_pipeline_parallel_requires_cfg_mixin():
    with pytest.raises(TypeError, match="inherits PipelineParallelMixin but not CFGParallelMixin"):

        class _MissingCFG(PipelineParallelMixin):
            pass


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
def test_pipeline_parallel_requires_mro_before_cfg_mixin():
    with pytest.raises(TypeError, match="must inherit PipelineParallelMixin before CFGParallelMixin"):

        class _WrongOrder(CFGParallelMixin, PipelineParallelMixin):
            pass


# ---------------------------------------------------------------------------
# Distributed test helpers
# ---------------------------------------------------------------------------


def _worker_device(local_rank: int, device_kind: DeviceKind) -> torch.device:
    if device_kind == "cpu":
        return torch.device("cpu")
    return torch.device(f"{current_omni_platform.device_type}:{local_rank}")


def init_dist(
    local_rank: int,
    world_size: int,
    master_port: str,
    device_kind: DeviceKind,
) -> torch.device:
    """Initialise the distributed environment for a spawned worker."""
    device = _worker_device(local_rank, device_kind)
    if device_kind == "cuda":
        current_omni_platform.set_device(device)
    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": master_port,
        }
    )
    backend = "gloo" if device_kind == "cpu" else None
    init_distributed_environment(backend=backend)
    return device


def _mp_backend(device_kind: DeviceKind) -> str | None:
    """Backend for model-parallel process groups.

    ``GroupCoordinator.all_gather`` always uses ``device_group``. CPU tensors
    therefore require gloo device groups; otherwise a CUDA host's default NCCL
    backend breaks PP+CFG CPU tests at CFG all_gather (pp2-cfg2).
    isend/irecv of CPU tensors can already fall back to ``cpu_group``.
    """
    return "gloo" if device_kind == "cpu" else None


def _seed_device(input_seed: int, device: torch.device) -> None:
    torch.manual_seed(input_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(input_seed)


def make_pipeline_and_inputs(
    test_config: dict, dtype: torch.dtype, device: torch.device, do_true_cfg: bool = False
) -> tuple[MockPipelineParallel, dict, dict | None]:
    """Create a MockPipelineParallel and seeded inputs from a test_config dict.

    Must be called after ``initialize_model_parallel`` so that ``make_layers``
    can read the PP group to determine this rank's layer slice.

    Returns ``(pipeline, positive_kwargs, negative_kwargs)``.
    ``negative_kwargs`` is ``None`` when ``do_true_cfg=False``.
    """
    pipeline = MockPipelineParallel(
        num_layers=test_config["num_layers"],
        dim=test_config["dim"],
        seed=test_config["model_seed"],
        device=device,
        dtype=dtype,
    )

    _seed_device(test_config["input_seed"], device)
    pos_x = {"x": torch.randn(test_config["batch_size"], test_config["dim"], dtype=dtype, device=device)}

    negative_kwargs = None
    if do_true_cfg:
        _seed_device(test_config["input_seed"] + 1, device)
        neg_x = torch.randn(test_config["batch_size"], test_config["dim"], dtype=dtype, device=device)
        negative_kwargs = {"x": neg_x}

    return pipeline, pos_x, negative_kwargs


# ---------------------------------------------------------------------------
# 3.  isend_tensor_dict / irecv_tensor_dict
# ---------------------------------------------------------------------------


def isend_irecv_worker(
    local_rank: int,
    world_size: int,
    master_port: str,
    device_kind: DeviceKind,
    result_queue,
):
    device = init_dist(local_rank, world_size, master_port, device_kind)
    initialize_model_parallel(pipeline_parallel_size=world_size, backend=_mp_backend(device_kind))
    pp_group = get_pp_group()

    if pp_group.is_first_rank:
        _seed_device(77, device)
        tensor = torch.randn(3, 5, dtype=torch.float32, device=device)
        handles = pp_group.isend_tensor_dict({"t": tensor})
        for h in handles:
            h.wait()
        result_queue.put(("sent", tensor.cpu()))
    else:
        received = AsyncIntermediateTensors(*pp_group.irecv_tensor_dict())
        result_queue.put(("received", received["t"].cpu()))

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    _cleanup_distributed()


def _run_isend_irecv(pp_size: int, device_kind: DeviceKind, master_port: str) -> None:
    mp_context = torch.multiprocessing.get_context("spawn")
    manager = mp_context.Manager()
    q = manager.Queue()
    torch.multiprocessing.spawn(
        isend_irecv_worker,
        args=(pp_size, master_port, device_kind, q),
        nprocs=pp_size,
    )
    results = {label: tensor for label, tensor in [q.get(), q.get()]}
    torch.testing.assert_close(
        results["received"], results["sent"], rtol=0, atol=0, msg="isend/irecv transferred tensor incorrectly"
    )


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
@pytest.mark.parametrize("pp_size", [2])
def test_isend_irecv_tensor_dict(pp_size: int):
    """isend_tensor_dict / irecv_tensor_dict transfer a tensor dict without loss."""
    _run_isend_irecv(pp_size, device_kind="cpu", master_port=_find_free_port())


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.parallel
@pytest.mark.parametrize("pp_size", [pytest.param(2, marks=_L4_TWO_GPU)])
def test_isend_irecv_tensor_dict_parity(pp_size: int):
    """Nightly: isend/irecv on real multi-GPU NCCL."""
    _run_isend_irecv(pp_size, device_kind="cuda", master_port=_find_free_port())


# ---------------------------------------------------------------------------
# 4.  predict_noise_maybe_with_cfg
# ---------------------------------------------------------------------------

_baseline_cache: dict[tuple, torch.Tensor] = {}


def compute_single_gpu_baseline(
    test_config: dict,
    dtype: torch.dtype,
    do_true_cfg: bool,
    device_kind: DeviceKind,
) -> torch.Tensor:
    """Compute expected single-process output using the same MockPipelineParallel."""
    key = (
        test_config["num_layers"],
        test_config["dim"],
        test_config["batch_size"],
        test_config["model_seed"],
        test_config["input_seed"],
        test_config["cfg_scale"],
        dtype,
        do_true_cfg,
        device_kind,
    )
    if key in _baseline_cache:
        return _baseline_cache[key]

    device = init_dist(0, 1, _find_free_port(), device_kind)
    try:
        initialize_model_parallel(pipeline_parallel_size=1, backend=_mp_backend(device_kind))

        pipeline, positive_kwargs, negative_kwargs = make_pipeline_and_inputs(
            test_config, dtype, device, do_true_cfg=do_true_cfg
        )

        with torch.inference_mode():
            noise_pred = pipeline.predict_noise_maybe_with_cfg(
                do_true_cfg=do_true_cfg,
                true_cfg_scale=test_config["cfg_scale"],
                positive_kwargs=positive_kwargs,
                negative_kwargs=negative_kwargs,
                cfg_normalize=False,
            )
    finally:
        _cleanup_distributed()

    _baseline_cache[key] = noise_pred.cpu()
    return _baseline_cache[key]


def predict_noise_worker(
    local_rank: int,
    world_size: int,
    master_port: str,
    pp_size: int,
    cfg_size: int,
    do_true_cfg: bool,
    dtype: torch.dtype,
    test_config: dict,
    device_kind: DeviceKind,
    result_queue,
):
    """Generic predict-noise worker parameterized by PP and CFG topology."""
    device = init_dist(local_rank, world_size, master_port, device_kind)
    initialize_model_parallel(
        pipeline_parallel_size=pp_size,
        cfg_parallel_size=cfg_size,
        backend=_mp_backend(device_kind),
    )

    pp_group = get_pp_group()
    cfg_rank = get_classifier_free_guidance_rank()

    pipeline, positive_kwargs, negative_kwargs = make_pipeline_and_inputs(
        test_config, dtype, device, do_true_cfg=do_true_cfg
    )

    with torch.inference_mode():
        noise_pred = pipeline.predict_noise_maybe_with_cfg(
            do_true_cfg=do_true_cfg,
            true_cfg_scale=test_config["cfg_scale"],
            positive_kwargs=positive_kwargs,
            negative_kwargs=negative_kwargs,
            cfg_normalize=False,
        )
    pipeline._sync_pp_send()

    if pp_group.is_last_rank:
        assert noise_pred is not None
        if cfg_rank == 0:
            result_queue.put(noise_pred.cpu())
    else:
        assert noise_pred is None

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    _cleanup_distributed()


def _run_predict_noise(
    *,
    pp_size: int,
    cfg_size: int,
    do_true_cfg: bool,
    dtype: torch.dtype,
    num_layers: int,
    input_seed: int,
    rtol: float,
    atol: float,
    device_kind: DeviceKind,
    master_port: str,
) -> None:
    test_config = {
        "num_layers": num_layers,
        "dim": 64,
        "batch_size": 2,
        "cfg_scale": 7.5,
        "model_seed": 42,
        "input_seed": input_seed,
    }

    baseline_out = compute_single_gpu_baseline(test_config, dtype, do_true_cfg, device_kind)

    mp_context = torch.multiprocessing.get_context("spawn")
    manager = mp_context.Manager()
    pp_q = manager.Queue()

    world_size = pp_size * cfg_size
    torch.multiprocessing.spawn(
        predict_noise_worker,
        args=(world_size, master_port, pp_size, cfg_size, do_true_cfg, dtype, test_config, device_kind, pp_q),
        nprocs=world_size,
    )

    pp_out = pp_q.get()
    assert baseline_out.shape == pp_out.shape
    torch.testing.assert_close(
        pp_out,
        baseline_out,
        rtol=rtol,
        atol=atol,
        msg=f"PP={pp_size} cfg={cfg_size} {'with' if do_true_cfg else 'no'} CFG output differs from baseline ({dtype=})",
    )


_CPU_PREDICT_NOISE_CASES = [
    pytest.param(2, 1, False, 4, 100, 1e-5, 1e-5, id="pp2-no_cfg"),
    pytest.param(2, 1, True, 4, 100, 1e-5, 1e-5, id="pp2-seq_cfg"),
    pytest.param(2, 2, True, 4, 100, 1e-5, 1e-5, id="pp2-cfg2"),
    pytest.param(3, 1, False, 6, 100, 1e-5, 1e-5, id="pp3-no_cfg"),
]

_GPU_PREDICT_NOISE_CASES = [
    pytest.param(
        2,
        1,
        False,
        torch.float32,
        4,
        100,
        1e-5,
        1e-5,
        marks=_L4_TWO_GPU,
        id="pp2-no_cfg-float32",
    ),
    pytest.param(
        2,
        1,
        False,
        torch.bfloat16,
        4,
        100,
        1e-2,
        1e-2,
        marks=_L4_TWO_GPU,
        id="pp2-no_cfg-bfloat16",
    ),
    pytest.param(
        2,
        1,
        True,
        torch.bfloat16,
        4,
        100,
        1e-2,
        1e-2,
        marks=_L4_TWO_GPU,
        id="pp2-seq_cfg-bfloat16",
    ),
    pytest.param(
        2,
        2,
        True,
        torch.bfloat16,
        4,
        100,
        1e-2,
        1e-2,
        marks=_L4_FOUR_GPU,
        id="pp2-cfg2-bfloat16",
    ),
    pytest.param(
        3,
        1,
        False,
        torch.bfloat16,
        6,
        100,
        1e-2,
        1e-2,
        marks=_L4_THREE_GPU,
        id="pp3-no_cfg-bfloat16",
    ),
]


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
@pytest.mark.parametrize(
    "pp_size, cfg_size, do_true_cfg, num_layers, input_seed, rtol, atol",
    _CPU_PREDICT_NOISE_CASES,
)
def test_predict_noise(pp_size, cfg_size, do_true_cfg, num_layers, input_seed, rtol, atol):
    """predict_noise_maybe_with_cfg output matches baseline across PP / CFG topologies."""
    _run_predict_noise(
        pp_size=pp_size,
        cfg_size=cfg_size,
        do_true_cfg=do_true_cfg,
        dtype=torch.float32,
        num_layers=num_layers,
        input_seed=input_seed,
        rtol=rtol,
        atol=atol,
        device_kind="cpu",
        master_port=_find_free_port(),
    )


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.parallel
@pytest.mark.parametrize(
    "pp_size, cfg_size, do_true_cfg, dtype, num_layers, input_seed, rtol, atol",
    _GPU_PREDICT_NOISE_CASES,
)
def test_predict_noise_parity(pp_size, cfg_size, do_true_cfg, dtype, num_layers, input_seed, rtol, atol):
    """Nightly: predict_noise parity on real multi-GPU NCCL."""
    _run_predict_noise(
        pp_size=pp_size,
        cfg_size=cfg_size,
        do_true_cfg=do_true_cfg,
        dtype=dtype,
        num_layers=num_layers,
        input_seed=input_seed,
        rtol=rtol,
        atol=atol,
        device_kind="cuda",
        master_port=_find_free_port(),
    )


# ---------------------------------------------------------------------------
# 5.  scheduler_step_maybe_with_cfg
# ---------------------------------------------------------------------------


def compute_scheduler_step_baseline(
    test_config: dict,
    do_true_cfg: bool,
    device_kind: DeviceKind,
) -> torch.Tensor:
    """Single-process reference: predict_noise + scheduler_step."""
    device = init_dist(0, 1, _find_free_port(), device_kind)
    try:
        initialize_model_parallel(pipeline_parallel_size=1, backend=_mp_backend(device_kind))

        pipeline, positive_kwargs, negative_kwargs = make_pipeline_and_inputs(
            test_config, torch.float32, device, do_true_cfg=do_true_cfg
        )
        latents = positive_kwargs["x"]
        t = torch.tensor(500, device=device)

        with torch.inference_mode():
            noise_pred = pipeline.predict_noise_maybe_with_cfg(
                do_true_cfg=do_true_cfg,
                true_cfg_scale=test_config["cfg_scale"],
                positive_kwargs=positive_kwargs,
                negative_kwargs=negative_kwargs,
                cfg_normalize=False,
            )
            result = pipeline.scheduler_step_maybe_with_cfg(
                noise_pred=noise_pred, t=t, latents=latents, do_true_cfg=do_true_cfg
            )
    finally:
        _cleanup_distributed()
    return result.cpu()


def scheduler_step_worker(
    local_rank: int,
    world_size: int,
    master_port: str,
    pp_size: int,
    cfg_size: int,
    do_true_cfg: bool,
    test_config: dict,
    device_kind: DeviceKind,
    result_queue,
):
    device = init_dist(local_rank, world_size, master_port, device_kind)
    initialize_model_parallel(
        pipeline_parallel_size=pp_size,
        cfg_parallel_size=cfg_size,
        backend=_mp_backend(device_kind),
    )

    pp_group = get_pp_group()
    cfg_rank = get_classifier_free_guidance_rank()

    pipeline, positive_kwargs, negative_kwargs = make_pipeline_and_inputs(
        test_config, torch.float32, device, do_true_cfg=do_true_cfg
    )
    latents = positive_kwargs["x"]
    t = torch.tensor(500, device=device)

    with torch.inference_mode():
        noise_pred = pipeline.predict_noise_maybe_with_cfg(
            do_true_cfg=do_true_cfg,
            true_cfg_scale=test_config["cfg_scale"],
            positive_kwargs=positive_kwargs,
            negative_kwargs=negative_kwargs,
            cfg_normalize=False,
        )
        latents = pipeline.scheduler_step_maybe_with_cfg(
            noise_pred=noise_pred, t=t, latents=latents, do_true_cfg=do_true_cfg
        )
    pipeline._sync_pp_send()

    if pp_group.is_first_rank and cfg_rank == 0:
        result_queue.put(latents.contiguous().cpu())

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    _cleanup_distributed()


def _run_scheduler_step(
    *,
    pp_size: int,
    cfg_size: int,
    do_true_cfg: bool,
    input_seed: int,
    device_kind: DeviceKind,
    master_port: str,
) -> None:
    test_config = {
        "num_layers": 4,
        "dim": 64,
        "batch_size": 2,
        "cfg_scale": 7.5,
        "model_seed": 42,
        "input_seed": input_seed,
    }

    baseline = compute_scheduler_step_baseline(test_config, do_true_cfg, device_kind)

    mp_context = torch.multiprocessing.get_context("spawn")
    manager = mp_context.Manager()
    q = manager.Queue()

    world_size = pp_size * cfg_size
    torch.multiprocessing.spawn(
        scheduler_step_worker,
        args=(world_size, master_port, pp_size, cfg_size, do_true_cfg, test_config, device_kind, q),
        nprocs=world_size,
    )

    result = q.get()
    torch.testing.assert_close(
        result,
        baseline,
        rtol=0,
        atol=0,
        msg=f"PP={pp_size} CFG={cfg_size} scheduler step latents on rank 0 do not match single-GPU baseline",
    )


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
@pytest.mark.parametrize(
    "pp_size, cfg_size, do_true_cfg, input_seed",
    [
        pytest.param(2, 1, False, 300, id="pp2-no_cfg"),
        pytest.param(2, 2, True, 600, id="pp2-cfg2-true_cfg"),
    ],
)
def test_scheduler_step(pp_size, cfg_size, do_true_cfg, input_seed):
    """Rank 0 latents after scheduler_step match baseline across PP / CFG topologies."""
    _run_scheduler_step(
        pp_size=pp_size,
        cfg_size=cfg_size,
        do_true_cfg=do_true_cfg,
        input_seed=input_seed,
        device_kind="cpu",
        master_port=_find_free_port(),
    )


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.parallel
@pytest.mark.parametrize(
    "pp_size, cfg_size, do_true_cfg, input_seed",
    [
        pytest.param(2, 1, False, 300, marks=_L4_TWO_GPU, id="pp2-no_cfg"),
        pytest.param(2, 2, True, 600, marks=_L4_FOUR_GPU, id="pp2-cfg2-true_cfg"),
    ],
)
def test_scheduler_step_parity(pp_size, cfg_size, do_true_cfg, input_seed):
    """Nightly: scheduler_step parity on real multi-GPU NCCL."""
    _run_scheduler_step(
        pp_size=pp_size,
        cfg_size=cfg_size,
        do_true_cfg=do_true_cfg,
        input_seed=input_seed,
        device_kind="cuda",
        master_port=_find_free_port(),
    )
