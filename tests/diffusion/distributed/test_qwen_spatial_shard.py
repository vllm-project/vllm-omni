# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import threading
import weakref
from concurrent.futures import Future
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers.models.autoencoders import AutoencoderKLQwenImage

from vllm_omni.diffusion.distributed.autoencoders import wan_spatial_shard
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_qwenimage import DistributedAutoencoderKLQwenImage
from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import DistributedVaeExecutor
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor, _ExecutorShutdownCleaner
from vllm_omni.diffusion.worker.diffusion_worker import WorkerProc

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _model(cls=DistributedAutoencoderKLQwenImage, channels=4, full_topology=False):
    return cls(
        base_dim=4,
        z_dim=2,
        dim_mult=[1, 2, 4, 4] if full_topology else [1, 2],
        num_res_blocks=1,
        temperal_downsample=[False, True, True] if full_topology else [True],
        input_channels=channels,
        latents_mean=[0, 0],
        latents_std=[1, 1],
    ).eval()


def _executor(mode, size=2, group=None):
    def no_tiles(*args, **kwargs):
        raise AssertionError("spatial decode must not use tile executor")

    # Supply Gloo groups directly without initializing the vLLM worker runtime.
    executor = DistributedVaeExecutor.__new__(DistributedVaeExecutor)
    executor.group = group
    executor._error_group = group
    executor._error_status = torch.zeros((), dtype=torch.int32)
    executor.set_parallel_size(size, mode)
    executor.execute = no_tiles
    return executor


@pytest.mark.parametrize("mode", ["spatial_shard_height", "spatial_shard_width"])
def test_uninitialized_spatial_decode_rejected(mode):
    vae = _model()
    vae.distributed_executor = _executor(mode)
    with pytest.raises(RuntimeError, match="initialized"):
        vae.decode(torch.randn(2, 2, 1, 3, 3))


def _worker(rank, path, direction):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{path}", rank=rank, world_size=2, timeout=timedelta(seconds=90)
    )
    try:
        # Include the production spatial/temporal stage layout at small channel width.
        cases = [(3, 2, 4, 1, False), (4, 7, 3, 1, False), (4, 2, 3, 3, False), (4, 2, 1, 1, False), (4, 2, 3, 3, True)]
        for channels, batch, extent, frames, full_topology in cases:
            torch.manual_seed(42)
            reference = _model(AutoencoderKLQwenImage, channels, full_topology)
            vae = _model(channels=channels, full_topology=full_topology)
            vae.load_state_dict(reference.state_dict())
            vae.requires_grad_(False)
            vae.distributed_executor = _executor(f"spatial_shard_{direction}", group=dist.group.WORLD)
            vae.enable_tiling()
            shape = (batch, 2, frames, extent, 3) if direction == "height" else (batch, 2, frames, 3, extent)
            z = torch.randn(shape)
            before = {key: value.clone() for key, value in vae.state_dict().items()}
            with torch.no_grad():
                expected = reference.decode(z).sample
                actual = vae.decode(z).sample
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
                assert vae.state_dict().keys() == before.keys()
                for key, value in vae.state_dict().items():
                    torch.testing.assert_close(value, before[key], rtol=0, atol=0)
                assert not any(p.requires_grad for p in vae.parameters())
                assert not any(m.training for m in vae.modules())
                assert all(v is None for v in vae._feat_map)
                # A second request and direct tiled_decode retain the explicit mode.
                torch.testing.assert_close(vae.tiled_decode(z, return_dict=False)[0], expected, rtol=1e-4, atol=1e-5)
                vae.enable_slicing()
                torch.testing.assert_close(vae.decode(z, return_dict=False)[0], expected, rtol=1e-4, atol=1e-5)
                vae.disable_slicing()
                vae.disable_tiling()
                image = torch.randn(batch, channels, 1, 8, 8)
                reference.encode(image)
                vae.encode(image)
                torch.testing.assert_close(vae.decode(z).sample, expected, rtol=1e-4, atol=1e-5)
        # Configuration locking is independent of tensor shape; check it once
        # per direction on the final, already-patched decoder.
        for bad_mode in ["tile", "spatial_shard_width" if direction == "height" else "spatial_shard_height"]:
            vae.distributed_executor.parallel_mode = bad_mode
            for decode in (vae.decode, vae.tiled_decode):
                with pytest.raises(ValueError, match="fresh VAE"):
                    decode(z)
        vae.distributed_executor.parallel_mode = f"spatial_shard_{direction}"
        vae.distributed_executor.group = object()
        with pytest.raises(ValueError, match="fresh VAE"):
            vae.decode(z)
        vae.distributed_executor.group = dist.group.WORLD
        vae.distributed_executor.parallel_size = 1
        with pytest.raises(ValueError, match="parallel"):
            vae.decode(z)
        vae = _model()
        vae.distributed_executor = _executor(f"spatial_shard_{direction}", size=3, group=dist.group.WORLD)
        with pytest.raises(ValueError, match="parallel"):
            vae.decode(z)
        assert not getattr(vae, "_qwen_spatial_shard_config", None)
        vae.distributed_executor = _executor(f"spatial_shard_{direction}")
        with pytest.raises(ValueError, match="process group"):
            vae.decode(z)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("direction", ["height", "width"])
def test_real_qwen_distributed_decode(tmp_path, direction):
    mp.start_processes(_worker, args=(str(tmp_path / "gloo"), direction), nprocs=2, join=True, start_method="spawn")


def _preparation_failure_worker(rank, path):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{path}", rank=rank, world_size=2, timeout=timedelta(seconds=15)
    )
    try:
        vae = _model()
        vae.distributed_executor = _executor("spatial_shard_height", group=dist.group.WORLD)
        vae.distributed_executor._error_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=15))

        def fail_preparation(module, args):
            if rank == 1:
                raise torch.OutOfMemoryError("injected preparation OOM")

        def forbid_decoder(module, args):
            # Fail immediately if a healthy rank advances into decoder communication.
            raise AssertionError("decoder entered after another rank failed preparation")

        preparation_hook = vae.post_quant_conv.register_forward_pre_hook(fail_preparation)
        decoder_hook = vae.decoder.register_forward_pre_hook(forbid_decoder)
        try:
            with pytest.raises(torch.OutOfMemoryError, match="spatial decode preparation"):
                vae.decode(torch.randn(2, 2, 1, 3, 3))
        finally:
            preparation_hook.remove()
            decoder_hook.remove()
        assert all(value is None for value in vae._feat_map)
        # All ranks have left the failed request together and can decode again.
        with torch.no_grad():
            assert vae.decode(torch.randn(2, 2, 1, 3, 3)).sample.shape == (2, 4, 1, 6, 6)
    finally:
        dist.destroy_process_group()


def test_spatial_preparation_oom_propagates_to_all_ranks(tmp_path):
    mp.start_processes(
        _preparation_failure_worker, args=(str(tmp_path / "gloo"),), nprocs=2, join=True, start_method="spawn"
    )


def _halo_failure_worker(rank, path, ready, start):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{path}", rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    vae = _model()
    vae.distributed_executor = _executor("spatial_shard_height", group=dist.group.WORLD)
    with torch.no_grad():
        vae.decode(torch.randn(1, 2, 1, 3, 3))

    original_ensure_recv_buf = wan_spatial_shard._ensure_recv_buf

    def fail_allocation(buf, reference):
        if rank == 1 and (buf is None or buf.shape != reference.shape):
            raise torch.OutOfMemoryError("injected halo allocation OOM")
        return original_ensure_recv_buf(buf, reference)

    class DecodeWorker:
        def execute_method(self, method, *args, **kwargs):
            # A larger request needs new halo buffers after successful warmup.
            with torch.no_grad():
                return vae.decode(torch.randn(1, 2, 1, 5, 5))

    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(wan_spatial_shard, "_ensure_recv_buf", fail_allocation)
            ready[rank].set()
            assert start.wait(60)
            proc = WorkerProc.__new__(WorkerProc)
            proc.worker = DecodeWorker()
            proc.gpu_id = rank
            proc.od_config = None
            proc.result_mq = object()
            proc._running = True
            proc._return_result = lambda *args, **kwargs: None
            message = {"type": "rpc", "method": "execute_model", "output_rank": 0, "exec_all_ranks": True}

            def recv_message():
                nonlocal message
                if message is not None:
                    result, message = message, None
                    return result
                # A swallowed request error must leave the worker alive long
                # enough for the parent's failure-notification deadline to expire.
                threading.Event().wait(60)
                raise AssertionError("worker continued after spatial decode failure")

            proc.recv_message = recv_message
            proc._worker_busy_loop()
    except BaseException as error:
        # An escaped exception must not look like the worker's fatal exit(1).
        raise SystemExit(2) from error


def test_spatial_halo_oom_closes_worker_group(tmp_path):
    ctx = mp.get_context("spawn")
    ready = [ctx.Event(), ctx.Event()]
    start = ctx.Event()
    processes = [
        ctx.Process(target=_halo_failure_worker, args=(rank, str(tmp_path / "gloo"), ready, start)) for rank in range(2)
    ]
    executor = MultiprocDiffusionExecutor.__new__(MultiprocDiffusionExecutor)
    executor._closed = False
    executor._is_failed = False
    executor._processes = processes
    executor._failure_callbacks = []
    executor._pump_stop = threading.Event()
    executor._futures_lock = threading.RLock()
    pending_request = Future()
    executor._rpc_futures = {0: pending_request}
    executor._output_futures = {}
    executor._batch_split_map = {}
    executor._completed_outputs = {}
    executor._dropped_output_ids = {}
    executor._shutdown_cleaner = _ExecutorShutdownCleaner(processes=processes)
    executor._finalizer = weakref.finalize(executor, executor._shutdown_cleaner)
    failed = threading.Event()
    executor.register_failure_callback(failed.set)
    try:
        for process in processes:
            process.start()
        assert all(event.wait(60) for event in ready), "warmup did not complete"
        executor._start_worker_monitor()
        start.set()
        # Well below the 60s process-group timeout: a rank failure must reach
        # the executor and close its peers, even when rank 1 is not the output rank.
        assert failed.wait(10), "worker group survived a spatial decode failure"
        assert executor.is_dead
        assert all(not process.is_alive() for process in processes)
        assert processes[1].exitcode == 1
        with pytest.raises(RuntimeError, match="shut down"):
            pending_request.result(timeout=1)
        with pytest.raises(RuntimeError, match="closed"):
            executor.collective_rpc("execute_model")
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
        for process in processes:
            if process.pid is not None:
                process.join(5)
        executor.shutdown()


@pytest.mark.parametrize("mode", ["spatial_shard_height", "spatial_shard_width"])
def test_single_rank_full_decode_ignores_tile_threshold(mode):
    torch.set_num_threads(1)
    torch.manual_seed(3)
    vae = _model()
    reference = _model(AutoencoderKLQwenImage)
    reference.load_state_dict(vae.state_dict())
    vae.distributed_executor = _executor(mode, size=1)
    vae.enable_tiling(tile_sample_min_height=2, tile_sample_min_width=2)
    z = torch.randn(2, 2, 2, 3, 3)
    with torch.no_grad():
        expected = reference.decode(z).sample
        torch.testing.assert_close(vae.decode(z).sample, expected)
        torch.testing.assert_close(vae.tiled_decode(z, return_dict=False)[0], expected)
    assert not getattr(vae, "_qwen_spatial_shard_config", None)


def test_cache_cleared_on_decoder_exception():
    vae = _model()
    vae.distributed_executor = _executor("spatial_shard_height", size=1)

    def fail(module, args):
        vae._feat_map[0] = args[0].clone()
        raise RuntimeError("decoder failed")

    handle = vae.decoder.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="decoder failed"):
            vae.decode(torch.randn(2, 2, 1, 3, 3))
    finally:
        handle.remove()
    assert all(v is None for v in vae._feat_map)
    assert vae._conv_idx == [0]
