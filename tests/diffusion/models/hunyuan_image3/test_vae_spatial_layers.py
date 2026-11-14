# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Platform isolation and CPU-checkable semantics of HunyuanImage3 spatial layers."""

from __future__ import annotations

import builtins
import contextlib
import runpy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from vllm_omni.diffusion.distributed.autoencoders import distributed_vae_executor, wan_spatial_shard
from vllm_omni.diffusion.models.hunyuan_image3.layers.native.autoencoder_blocks import AttnBlock as NativeAttnBlock
from vllm_omni.diffusion.models.hunyuan_image3.layers.native.autoencoder_blocks import ResnetBlock as NativeResnetBlock
from vllm_omni.diffusion.models.hunyuan_image3.layers.native.vae_context import VAE_SPATIAL_CONTEXT, VaeSpatialContext
from vllm_omni.diffusion.models.hunyuan_image3.layers.npu import spatial_vae
from vllm_omni.diffusion.models.hunyuan_image3.layers.npu.autoencoder_blocks import AttnBlock, ResnetBlock, VaeConv3d

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_native_dispatch_does_not_import_npu(monkeypatch):
    import vllm_omni.platforms as platforms

    monkeypatch.setattr(
        platforms, "current_omni_platform", SimpleNamespace(is_cuda=lambda: False, is_npu=lambda: False)
    )
    original_import = builtins.__import__

    def isolated_import(name, *args, **kwargs):
        assert ".layers.npu" not in name
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", isolated_import)
    path = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/hunyuan_image3/layers/__init__.py"
    exported = runpy.run_path(str(path))
    assert exported["ResnetBlock"] is NativeResnetBlock
    assert exported["AttnBlock"] is NativeAttnBlock
    assert exported["VaeConv3d"] is nn.Conv3d


def test_npu_dispatch_selects_typed_layers(monkeypatch):
    import vllm_omni.platforms as platforms

    monkeypatch.setattr(platforms, "current_omni_platform", SimpleNamespace(is_cuda=lambda: False, is_npu=lambda: True))
    path = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/hunyuan_image3/layers/__init__.py"
    exported = runpy.run_path(str(path))
    assert exported["ResnetBlock"] is ResnetBlock
    assert exported["AttnBlock"] is AttnBlock
    assert exported["VaeConv3d"] is VaeConv3d
    from vllm_omni.diffusion.models.hunyuan_image3.layers.native import ResBlock

    assert exported["ResBlock"] is ResBlock


@pytest.mark.parametrize("native,accelerated", [(NativeResnetBlock, ResnetBlock), (NativeAttnBlock, AttnBlock)])
def test_checkpoint_keys_and_native_fallback(native, accelerated):
    torch.manual_seed(11)
    reference = native(32, 32) if native is NativeResnetBlock else native(32)
    actual = accelerated(32, 32) if accelerated is ResnetBlock else accelerated(32)
    assert reference.state_dict().keys() == actual.state_dict().keys()
    actual.load_state_dict(reference.state_dict(), strict=True)
    identities = {k: id(v) for k, v in actual.named_parameters()}
    x = torch.randn(1, 32, 2, 4, 4)
    with torch.no_grad():
        torch.testing.assert_close(actual(x), reference(x), rtol=0, atol=0)
    assert identities == {k: id(v) for k, v in actual.named_parameters()}
    assert all("forward" not in m.__dict__ and "attention" not in m.__dict__ for m in actual.modules())


def test_conditional_encode_never_enters_spatial_execution():
    # This gate must run before tensor, device, or process-group inspection.
    assert spatial_vae.spatial_vae_forward(object(), object(), "encode") is None


def test_spatial_context_is_reset_on_failure(monkeypatch):
    monkeypatch.setattr(dist, "get_world_size", lambda group: 1)
    monkeypatch.setattr(dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: contextlib.nullcontext())

    class Failing(nn.Module):
        def forward(self, x):
            assert VAE_SPATIAL_CONTEXT.get() is not None
            raise RuntimeError("expected failure")

    assert VAE_SPATIAL_CONTEXT.get() is None
    with pytest.raises(RuntimeError, match="expected failure"):
        spatial_vae.run_spatial_vae(Failing(), torch.randn(1, 32, 1, 4, 4), object())
    assert VAE_SPATIAL_CONTEXT.get() is None


def _spatial_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        with patch("vllm_omni.platforms.npu.layers.dist_group_norm.dist_group_norm", _cpu_group_norm_operator):
            torch.manual_seed(42)
            reference = nn.Sequential(NativeResnetBlock(32, 32), NativeAttnBlock(32)).eval()
            model = nn.Sequential(ResnetBlock(32, 32), AttnBlock(32)).eval()
            model.load_state_dict(reference.state_dict(), strict=True)
            x = torch.randn(1, 32, 2, 8, 4)
            with torch.no_grad():
                expected = reference(x).chunk(2, dim=3)[rank]
                token = VAE_SPATIAL_CONTEXT.set(VaeSpatialContext(dist.group.WORLD))
                try:
                    actual = model(x.chunk(2, dim=3)[rank].contiguous())
                finally:
                    VAE_SPATIAL_CONTEXT.reset(token)
            torch.testing.assert_close(actual, expected, rtol=3e-05, atol=3e-05)
            assert VAE_SPATIAL_CONTEXT.get() is None
            _check_encoder_decoder_shards(rank)
    finally:
        dist.destroy_process_group()


def test_two_rank_spatial_semantics(tmp_path):
    if not dist.is_gloo_available():
        pytest.skip("Gloo unavailable")
    mp.spawn(_spatial_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)


def _spatial_oom_worker(rank, rendezvous):
    from vllm_omni.diffusion.models.hunyuan_image3.layers.npu.normalization import global_group_norm
    from vllm_omni.platforms.npu.layers import dist_group_norm

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=15))
    try:
        group = dist.group.WORLD
        error_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=15))
        with patch.object(
            distributed_vae_executor,
            "get_world_group",
            return_value=SimpleNamespace(cpu_group=error_group, device_group=group),
        ):
            executor = distributed_vae_executor.DistributedVaeExecutor()
        x = torch.ones(1, 1, 1, 8, 4)
        halo_conv = VaeConv3d(1, 1, 3, padding=1, dtype=torch.bfloat16).eval()
        original_recv_buf = wan_spatial_shard._ensure_recv_buf
        original_cat = torch.cat
        original_empty = torch.empty
        norm = nn.GroupNorm(1, 1)

        class HaloThenNorm(nn.Module):
            def forward(self, x):
                return global_group_norm(halo_conv(x), norm)

        def fail_recv_alloc(*args, **kwargs):
            if rank == 1:
                raise torch.OutOfMemoryError("injected halo allocation failure")
            return original_recv_buf(*args, **kwargs)

        def fail_output_cat(*args, **kwargs):
            if rank == 1:
                raise torch.OutOfMemoryError("injected final output assembly failure")
            return original_cat(*args, **kwargs)

        def fail_gn_gather_alloc(size, *args, **kwargs):
            # (ranks, batch, packed mean/m2/count) in the real NCTHW GN path.
            if rank == 1 and size == (2, 1, 3):
                raise torch.OutOfMemoryError("injected DistGN statistics gather buffer allocation failure")
            return original_empty(size, *args, **kwargs)

        with patch.object(torch, "autocast", return_value=contextlib.nullcontext()), torch.no_grad():
            with patch.object(wan_spatial_shard, "_ensure_recv_buf", fail_recv_alloc):
                with pytest.raises(torch.OutOfMemoryError, match="before first communication"):
                    spatial_vae.run_spatial_vae(halo_conv, x, group, executor=executor)
            # Stub only the NPU kernels. The Halo, DistGN buffer allocation,
            # communication guard and CPU error agreement run through real code.
            with (
                patch.object(dist_group_norm, "_vectorcore_num", return_value=1),
                patch.object(dist_group_norm, "_collect_cf"),
                patch.object(
                    dist_group_norm, "_reduce", return_value=(torch.zeros(1, 1), torch.zeros(1, 1), torch.ones(1))
                ),
                patch.object(torch, "empty", fail_gn_gather_alloc),
                patch.object(dist, "batch_isend_irecv", wraps=dist.batch_isend_irecv) as halo,
                patch.object(dist, "all_gather_into_tensor", wraps=dist.all_gather_into_tensor) as gn_gather,
            ):
                detail = "injected DistGN statistics gather buffer" if rank == 1 else "another rank ran out of memory"
                with pytest.raises(torch.OutOfMemoryError, match=f"after halo exchange failed: {detail}"):
                    spatial_vae.run_spatial_vae(HaloThenNorm(), x, group, executor=executor)
                halo.assert_called_once()
                gn_gather.assert_not_called()
            with patch.object(torch, "cat", fail_output_cat):
                with pytest.raises(torch.OutOfMemoryError, match="after all-gather"):
                    spatial_vae.run_spatial_vae(nn.Identity(), x, group, executor=executor)
            assert VAE_SPATIAL_CONTEXT.get() is None
            actual = spatial_vae.run_spatial_vae(nn.Identity(), x, group, executor=executor)
            torch.testing.assert_close(actual, x.bfloat16())
    finally:
        dist.destroy_process_group()


def test_two_rank_spatial_allocation_oom_preserves_process_group(tmp_path):
    if not dist.is_gloo_available():
        pytest.skip("Gloo unavailable")
    mp.spawn(_spatial_oom_worker, args=((tmp_path / "oom_rendezvous").as_uri(),), nprocs=2, join=True)


@pytest.mark.parametrize("stage", ["encode", "decode"])
def test_spatial_mode_leaves_cpu_on_native_path(stage):
    assert (
        spatial_vae.spatial_vae_forward(
            SimpleNamespace(distributed_executor=SimpleNamespace(parallel_mode="spatial_shard_height")),
            torch.randn(1, 32, 1, 8, 8),
            stage,
        )
        is None
    )


@pytest.mark.parametrize("mode", ["tile", "batch", "spatial_shard_width"])
def test_other_parallel_modes_leave_native_execution(mode):
    vae = SimpleNamespace(distributed_executor=SimpleNamespace(parallel_mode=mode))
    # No tensor or process group is consulted for another execution mode.
    assert spatial_vae.spatial_vae_forward(vae, object(), "decode") is None


def _check_encoder_decoder_shards(rank):
    from vllm_omni.diffusion.models.hunyuan_image3 import autoencoder as ae
    from vllm_omni.diffusion.models.hunyuan_image3.layers.native.normalization import (
        vae_group_norm_for_conv as native_norm,
    )
    from vllm_omni.diffusion.models.hunyuan_image3.layers.npu.normalization import vae_group_norm_for_conv as npu_norm

    x = torch.randn(1, 3, 2, 16, 8)
    z = torch.randn(1, 4, 1, 8, 4)
    native_layers = dict(
        VaeConv3d=nn.Conv3d,
        ResnetBlock=NativeResnetBlock,
        AttnBlock=NativeAttnBlock,
        vae_group_norm_for_conv=native_norm,
    )
    accelerated_layers = dict(
        VaeConv3d=VaeConv3d, ResnetBlock=ResnetBlock, AttnBlock=AttnBlock, vae_group_norm_for_conv=npu_norm
    )
    with patch.multiple(ae, **native_layers), torch.no_grad():
        encoder = ae.Encoder(3, 4, (32, 32), 1, 2, 2).eval()
        decoder = ae.Decoder(4, 3, (32, 32), 1, 2, 2).eval()
        reference_encode = encoder(x)
        reference_decode = decoder(z)
    with patch.multiple(ae, **accelerated_layers), torch.no_grad():
        actual_encoder = ae.Encoder(3, 4, (32, 32), 1, 2, 2).eval()
        actual_decoder = ae.Decoder(4, 3, (32, 32), 1, 2, 2).eval()
        actual_encoder.load_state_dict(encoder.state_dict(), strict=True)
        actual_decoder.load_state_dict(decoder.state_dict(), strict=True)
        token = VAE_SPATIAL_CONTEXT.set(VaeSpatialContext(dist.group.WORLD))
        try:
            actual_encode = actual_encoder(x.chunk(2, dim=3)[rank].contiguous())
            actual_decode = actual_decoder(z.chunk(2, dim=3)[rank].contiguous())
        finally:
            VAE_SPATIAL_CONTEXT.reset(token)
        torch.testing.assert_close(actual_encode, reference_encode.chunk(2, dim=3)[rank], rtol=5e-05, atol=5e-05)
        torch.testing.assert_close(actual_decode, reference_decode.chunk(2, dim=3)[rank], rtol=5e-05, atol=5e-05)


@pytest.mark.parametrize("case", ["group_mismatch", "width_mode", "unaligned", "convolution", "supported"])
def test_spatial_gate_reuses_executor_and_falls_back_before_execution(monkeypatch, case):
    monkeypatch.setattr(dist, "get_world_size", lambda group: 4)
    stride = 2 if case == "convolution" else 1
    network = VaeConv3d(32, 32, 3, stride=stride, padding=1).eval()
    group = object()
    executor = SimpleNamespace(
        group=group,
        parallel_size=2 if case == "group_mismatch" else 4,
        parallel_mode="spatial_shard_width" if case == "width_mode" else "spatial_shard_height",
    )
    vae = SimpleNamespace(distributed_executor=executor, decoder=network)
    x = SimpleNamespace(
        device=SimpleNamespace(type="npu"),
        ndim=5,
        shape=(1, 32, 4, 15 if case == "unaligned" else 64, 64),
        numel=lambda: 1,
    )
    sentinel = torch.zeros(1)
    calls = []

    def execute(actual_network, actual_x, actual_group, *, executor):
        calls.append((actual_network, actual_x, actual_group, executor))
        return sentinel

    monkeypatch.setattr(spatial_vae, "run_spatial_vae", execute)
    with torch.no_grad():
        result = spatial_vae.spatial_vae_forward(vae, x, "decode")
    if case == "supported":
        assert result is sentinel
        assert calls == [(network, x, group, executor)]
    else:
        assert result is None
        assert not calls


def _cpu_group_norm_operator(
    x, groups, weight, bias, eps, activation="swish", device_group=None, layout="NCTHW", **kwargs
):
    assert layout == "NCTHW"
    world = dist.get_world_size(device_group)
    parts = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(parts, x.contiguous(), group=device_group)
    full = torch.cat(parts, dim=3)
    y = torch.nn.functional.group_norm(full, groups, weight, bias, eps)
    if activation == "swish":
        y = torch.nn.functional.silu(y)
    return y.chunk(world, dim=3)[dist.get_rank(device_group)].contiguous()


@pytest.mark.parametrize(
    "stage,distributed,spatial_result",
    [
        ("encode", False, False),
        ("encode", True, False),
        ("decode", False, False),
        ("decode", True, False),
        ("decode", True, True),
    ],
)
def test_distributed_vae_dispatch_reaches_serial_spatial_and_tile_paths(
    monkeypatch, stage, distributed, spatial_result
):
    from vllm_omni.diffusion.distributed.autoencoders import autoencoder_kl_hunyuan as entry

    cls = entry.DistributedAutoencoderKLHunyuan
    vae = cls.__new__(cls)
    nn.Module.__init__(vae)
    vae.is_distributed_enabled = lambda: distributed
    x = torch.zeros(1)
    serial_output = torch.ones(1)
    tile_output = torch.full((1,), 2)
    spatial_output = torch.full((1,), 3)
    calls = []

    def serial(actual, actual_x):
        assert actual is vae and actual_x is x
        calls.append("serial")
        return serial_output

    def spatial(actual, actual_x, actual_stage):
        assert actual is vae and actual_x is x
        assert actual_stage == "decode"
        calls.append("spatial")
        return spatial_output if spatial_result else None

    def tile(actual_x, operator, broadcast_result):
        assert actual_x is x and broadcast_result
        expected = vae.encode_tile_exec if stage == "encode" else vae.tile_exec
        assert operator.exec == expected
        calls.append("tile")
        return tile_output

    vae.distributed_executor = SimpleNamespace(execute=tile)
    monkeypatch.setattr(entry.AutoencoderKLConv3D, "spatial_tiled_" + stage, serial)
    monkeypatch.setattr(entry, "spatial_vae_forward", spatial)
    actual = getattr(vae, "spatial_tiled_" + stage)(x)
    if not distributed:
        assert actual is serial_output
        assert calls == ["serial"]
    elif stage == "encode":
        assert actual is tile_output
        assert calls == ["tile"]
    elif spatial_result:
        assert actual is spatial_output
        assert calls == ["spatial"]
    else:
        assert actual is tile_output
        assert calls == ["spatial", "tile"]


@pytest.mark.parametrize("stage", ["encode", "decode"])
@pytest.mark.parametrize("mode", ["tile", "spatial_shard_height"])
def test_native_spatial_fallback_reports_unsupported_decode(monkeypatch, stage, mode):
    from vllm_omni.diffusion.models.hunyuan_image3.layers.native import spatial_vae as native_spatial

    messages = []
    monkeypatch.setattr(native_spatial.logger, "warning_once", lambda message: messages.append(message))
    vae = SimpleNamespace(distributed_executor=SimpleNamespace(parallel_mode=mode))
    assert native_spatial.spatial_vae_forward(vae, object(), stage) is None
    if stage == "decode" and mode == "spatial_shard_height":
        assert len(messages) == 1
        assert "supported only on Ascend NPU" in messages[0]
    else:
        assert not messages


@pytest.mark.parametrize("reason", ["grad", "compile"])
def test_npu_spatial_fallback_reports_unsupported_execution(monkeypatch, reason):
    messages = []
    monkeypatch.setattr(spatial_vae.logger, "warning_once", lambda message: messages.append(message))
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: reason == "compile")
    vae = SimpleNamespace(distributed_executor=SimpleNamespace(parallel_mode="spatial_shard_height"))
    x = SimpleNamespace(device=SimpleNamespace(type="npu"))
    with torch.set_grad_enabled(reason == "grad"):
        assert spatial_vae.spatial_vae_forward(vae, x, "decode") is None
    assert len(messages) == 1
    assert "requires eager inference without gradients" in messages[0]
