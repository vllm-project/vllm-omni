# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real two-rank coverage for the FLUX sharded single-block projection path.

``tests/diffusion/models/flux/test_flux_transformer_sharded_proj.py`` only
initializes model parallel with ``world_size=1``, so passing
``tensor_parallel_size=2`` there merely selects the sharded branch - the
collectives, the per-rank weight split and the fused ``proj_out`` never run.

These tests spawn ``world_size=2`` processes and check, on *both* ranks:

* every checkpoint key is consumed (``load_weights`` covers the split
  ``proj_out`` -> ``attn_proj``/``mlp_proj`` remap on each rank),
* each rank holds its own half of the sharded weights (and the halves
  concatenate back to the checkpoint tensor),
* the forward output matches the replicated TP=1 baseline, and
* both ranks produce the same output after the ``proj_out`` all-reduce.

CPU tests use gloo collectives; the nightly parity test runs the same checks on
real multi-GPU device collectives.
"""

from __future__ import annotations

import os
import socket
from contextlib import contextmanager
from typing import Literal

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.data import (
    DiffusionParallelConfig,
    OmniDiffusionConfig,
    TransformerConfig,
)

_TWO_CARD = hardware_marks(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"}, num_cards=2)

DeviceKind = Literal["cpu", "device"]

_SHARDED_PROJ_ENV = "VLLM_OMNI_FLUX1_SHARDED_PROJ"

# Small but structurally faithful FLUX config: single-stream blocks only, since
# the sharded path is exclusive to FluxSingleTransformerBlock. attention_head_dim
# must be >= sum(axes_dims_rope) for the RoPE application to be well-formed.
_MODEL_KWARGS = dict(
    in_channels=8,
    num_layers=0,
    num_single_layers=2,
    num_attention_heads=4,
    attention_head_dim=16,
    joint_attention_dim=16,
    pooled_projection_dim=8,
    axes_dims_rope=(4, 6, 6),
    guidance_embeds=False,
)
_MODEL_SEED = 0
_CHECKPOINT_SEED = 7
_INPUT_SEED = 1234
_BATCH_SIZE = 1
_IMAGE_SEQ_LEN = 6
_TEXT_SEQ_LEN = 3


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextmanager
def _sharded_proj_env(sharded: bool):
    """Scope ``VLLM_OMNI_FLUX1_SHARDED_PROJ`` and restore the prior value.

    ``_tp1_baseline`` builds a model in the parent pytest process, so an
    unscoped ``os.environ`` write would leak the flag into every later test.
    """
    previous = os.environ.get(_SHARDED_PROJ_ENV)
    os.environ[_SHARDED_PROJ_ENV] = "1" if sharded else "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_SHARDED_PROJ_ENV, None)
        else:
            os.environ[_SHARDED_PROJ_ENV] = previous


def _build_model(tp_size: int, *, sharded: bool, device: torch.device):
    """Build a FLUX transformer with the sharded path enabled/disabled.

    The env var only has to be set while the module tree is constructed, because
    the sharded branch is resolved at ``FluxSingleTransformerBlock.__init__`` time.
    """
    from vllm_omni.diffusion.models.flux.flux_transformer import FluxTransformer2DModel

    # Real OmniDiffusionConfig/TransformerConfig, not a duck-typed stand-in:
    # TransformerConfig resolves num_layers through its ``params`` dict, which is
    # the attribute path FluxTransformer2DModel.__init__ actually takes.
    od_config = OmniDiffusionConfig(
        tf_model_config=TransformerConfig(params={"num_layers": _MODEL_KWARGS["num_layers"]}),
        parallel_config=DiffusionParallelConfig(tensor_parallel_size=tp_size),
    )
    torch.manual_seed(_MODEL_SEED)
    with _sharded_proj_env(sharded):
        model = FluxTransformer2DModel(od_config=od_config, **_MODEL_KWARGS)
    return model.to(device=device).eval()


def _make_checkpoint(model) -> dict[str, torch.Tensor]:
    """Synthesize a diffusers-style checkpoint for ``model``.

    Emitted with the *unsharded* diffusers names: fused ``to_qkv`` is split back
    into ``to_q``/``to_k``/``to_v``, and ``proj_out``/``proj_mlp`` keep their
    replicated names so ``load_weights`` has to perform the sharded remap.
    """
    torch.manual_seed(_CHECKPOINT_SEED)
    checkpoint: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        weight = torch.randn_like(param, device="cpu") * 0.05
        if ".to_qkv." in name:
            prefix, _, suffix = name.rpartition(".to_qkv.")
            q, k, v = torch.chunk(weight, 3, dim=0)
            checkpoint[f"{prefix}.to_q.{suffix}"] = q
            checkpoint[f"{prefix}.to_k.{suffix}"] = k
            checkpoint[f"{prefix}.to_v.{suffix}"] = v
        else:
            checkpoint[name] = weight
    return checkpoint


def _make_inputs(device: torch.device, dtype: torch.dtype) -> dict:
    torch.manual_seed(_INPUT_SEED)
    kwargs = dict(
        hidden_states=torch.randn(_BATCH_SIZE, _IMAGE_SEQ_LEN, _MODEL_KWARGS["in_channels"], dtype=dtype),
        encoder_hidden_states=torch.randn(
            _BATCH_SIZE, _TEXT_SEQ_LEN, _MODEL_KWARGS["joint_attention_dim"], dtype=dtype
        ),
        pooled_projections=torch.randn(_BATCH_SIZE, _MODEL_KWARGS["pooled_projection_dim"], dtype=dtype),
        timestep=torch.tensor([0.5], dtype=dtype),
        img_ids=torch.zeros(_IMAGE_SEQ_LEN, 3, dtype=dtype),
        txt_ids=torch.zeros(_TEXT_SEQ_LEN, 3, dtype=dtype),
    )
    return {name: tensor.to(device) for name, tensor in kwargs.items()}


def _worker_device(local_rank: int, device_kind: DeviceKind) -> torch.device:
    if device_kind == "cpu":
        return torch.device("cpu")
    from vllm_omni.platforms import current_omni_platform

    return torch.device(f"{current_omni_platform.device_type}:{local_rank}")


def _dist_backend(device_kind: DeviceKind) -> str:
    """Process-group backend: gloo for CPU tensors, the platform's own otherwise.

    ``init_distributed_environment`` defaults to ``"nccl"``, which is wrong on
    non-CUDA accelerators, and passing ``None`` trips ``torch.distributed``.
    """
    if device_kind == "cpu":
        return "gloo"
    from vllm.platforms import current_platform

    return current_platform.dist_backend


def _init_dist(local_rank: int, world_size: int, master_port: int, device_kind: DeviceKind) -> torch.device:
    from vllm.distributed.parallel_state import (
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )

    device = _worker_device(local_rank, device_kind)
    if device_kind != "cpu":
        from vllm_omni.platforms import current_omni_platform

        current_omni_platform.set_device(device)

    for key, value in {
        "RANK": str(local_rank),
        "LOCAL_RANK": str(local_rank),
        "WORLD_SIZE": str(world_size),
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(master_port),
    }.items():
        os.environ[key] = value

    backend = _dist_backend(device_kind)
    init_distributed_environment(
        world_size=world_size,
        rank=local_rank,
        local_rank=local_rank,
        distributed_init_method=f"tcp://127.0.0.1:{master_port}",
        backend=backend,
    )
    initialize_model_parallel(tensor_model_parallel_size=world_size, backend=backend)
    if device_kind == "cpu":
        # Platforms that opt into torch.ops.vllm.* collectives (CUDA/XPU) have no
        # CPU kernel registered for them; take the eager gloo path instead.
        get_tp_group().use_custom_op_call = False
    return device


def _cleanup_dist() -> None:
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    cleanup_dist_env_and_memory()
    for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"):
        os.environ.pop(key, None)


@contextmanager
def _worker_vllm_config(device_kind: DeviceKind):
    """Provide a VllmConfig inside spawned ranks.

    The session-scoped ``default_vllm_config`` fixture only covers the parent
    process; ``initialize_model_parallel`` and CustomOp construction in a spawned
    rank need their own ``set_current_vllm_config`` context.
    """
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.platforms import current_platform

    device = "cpu" if device_kind == "cpu" else current_platform.device_type
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device=device))):
        yield


def _shard_report(model, checkpoint: dict[str, torch.Tensor], rank: int, world_size: int) -> dict:
    """Collect per-rank evidence that proj_out weights are really split."""
    from vllm_omni.diffusion.models.flux.flux_transformer import FluxSingleBlockOutput

    report: dict = {"sharded_blocks": [], "shards": {}}
    for name, block in model.single_transformer_blocks.named_children():
        proj_out = block.proj_out
        report["sharded_blocks"].append(
            (name, bool(block.use_sharded_single_block), isinstance(proj_out, FluxSingleBlockOutput))
        )
        if not isinstance(proj_out, FluxSingleBlockOutput):
            continue
        ckpt_key = f"single_transformer_blocks.{name}.proj_out.weight"
        full = checkpoint[ckpt_key]
        # RowParallelLinear shards the input dim; attn/mlp halves are cut from
        # the checkpoint's [attn_dim : attn_dim + mlp_dim] input columns.
        attn_full = full.narrow(1, 0, proj_out.attn_dim)
        mlp_full = full.narrow(1, proj_out.attn_dim, proj_out.mlp_dim)
        report["shards"][name] = {
            "attn": (
                proj_out.attn_proj.weight.detach().cpu().clone(),
                attn_full.chunk(world_size, dim=1)[rank].cpu().clone(),
            ),
            "mlp": (
                proj_out.mlp_proj.weight.detach().cpu().clone(),
                mlp_full.chunk(world_size, dim=1)[rank].cpu().clone(),
            ),
        }
    return report


def _run_rank(
    local_rank: int,
    world_size: int,
    master_port: int,
    device_kind: DeviceKind,
    sharded: bool,
    checkpoint: dict[str, torch.Tensor],
    result_queue,
) -> None:
    with _worker_vllm_config(device_kind):
        device = _init_dist(local_rank, world_size, master_port, device_kind)
        try:
            model = _build_model(world_size, sharded=sharded, device=device)
            loaded = model.load_weights([(name, weight.clone()) for name, weight in checkpoint.items()])
            report = _shard_report(model, checkpoint, local_rank, world_size)
            with torch.no_grad():
                (output,) = model(**_make_inputs(device, torch.float32), return_dict=False)
            report.update(
                rank=local_rank,
                unloaded=sorted(set(checkpoint) - loaded),
                output=output.float().cpu().clone(),
            )
            result_queue.put(report)
        finally:
            _cleanup_dist()


def _spawn_ranks(
    *,
    world_size: int,
    device_kind: DeviceKind,
    sharded: bool,
    checkpoint: dict[str, torch.Tensor],
) -> list[dict]:
    mp_context = torch.multiprocessing.get_context("spawn")
    queue = mp_context.Manager().Queue()
    torch.multiprocessing.spawn(
        _run_rank,
        args=(world_size, _find_free_port(), device_kind, sharded, checkpoint, queue),
        nprocs=world_size,
    )
    reports = [queue.get() for _ in range(world_size)]
    return sorted(reports, key=lambda report: report["rank"])


def _tp1_baseline(device_kind: DeviceKind) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Replicated TP=1 checkpoint + output, computed in this process."""
    device = _init_dist(0, 1, _find_free_port(), device_kind)
    try:
        model = _build_model(1, sharded=False, device=device)
        checkpoint = _make_checkpoint(model)
        loaded = model.load_weights([(name, weight.clone()) for name, weight in checkpoint.items()])
        unloaded = sorted(set(checkpoint) - loaded)
        assert not unloaded, f"TP=1 baseline left checkpoint keys unloaded: {unloaded}"
        with torch.no_grad():
            (output,) = model(**_make_inputs(device, torch.float32), return_dict=False)
        return checkpoint, output.float().cpu().clone()
    finally:
        _cleanup_dist()


def _tolerance(device_kind: DeviceKind) -> tuple[float, float]:
    # The sharded path reassociates the proj_out sum (per-shard matmul + all-reduce
    # instead of one fused matmul), so parity is float-accurate, not bitwise.
    return (1e-4, 1e-4) if device_kind == "cpu" else (2e-3, 2e-3)


def _check_tp2_sharded(device_kind: DeviceKind) -> None:
    world_size = 2
    checkpoint, baseline = _tp1_baseline(device_kind)
    reports = _spawn_ranks(world_size=world_size, device_kind=device_kind, sharded=True, checkpoint=checkpoint)
    rtol, atol = _tolerance(device_kind)

    for report in reports:
        rank = report["rank"]

        # 1. The sharded branch is actually live on this rank.
        assert report["sharded_blocks"], f"[rank {rank}] no single_transformer_blocks were built"
        for block_name, use_sharded, is_fused_proj_out in report["sharded_blocks"]:
            assert use_sharded, f"[rank {rank}] block {block_name} did not take the sharded path"
            assert is_fused_proj_out, f"[rank {rank}] block {block_name} proj_out is not FluxSingleBlockOutput"

        # 2. Every checkpoint key was consumed on every rank (the split proj_out
        #    remap must not silently drop weights on a non-zero rank).
        assert not report["unloaded"], f"[rank {rank}] unloaded checkpoint keys: {report['unloaded']}"

        # 3. Each rank holds its own shard of the split proj_out weights.
        for block_name, shards in report["shards"].items():
            for half, (actual, expected) in shards.items():
                assert actual.shape == expected.shape, (
                    f"[rank {rank}] block {block_name} {half}_proj shard shape "
                    f"{tuple(actual.shape)} != expected {tuple(expected.shape)}"
                )
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=0,
                    atol=0,
                    msg=f"[rank {rank}] block {block_name} {half}_proj holds the wrong weight shard",
                )

        # 4. The forward output matches the replicated TP=1 baseline.
        torch.testing.assert_close(
            report["output"],
            baseline,
            rtol=rtol,
            atol=atol,
            msg=f"[rank {rank}] sharded TP=2 output diverged from the replicated TP=1 baseline",
        )

    # 5. Both ranks agree after the proj_out all-reduce.
    torch.testing.assert_close(
        reports[0]["output"],
        reports[1]["output"],
        rtol=0,
        atol=0,
        msg="TP=2 ranks disagree on the output; the proj_out all-reduce did not synchronize them",
    )

    # 6. The two ranks' shards are complementary, not duplicated: concatenating
    #    them reconstructs the checkpoint halves.
    for block_name in reports[0]["shards"]:
        proj_out_key = f"single_transformer_blocks.{block_name}.proj_out.weight"
        full = checkpoint[proj_out_key].cpu()
        attn_width = reports[0]["shards"][block_name]["attn"][0].shape[1] * world_size
        rebuilt = torch.cat(
            [
                torch.cat([report["shards"][block_name]["attn"][0] for report in reports], dim=1),
                torch.cat([report["shards"][block_name]["mlp"][0] for report in reports], dim=1),
            ],
            dim=1,
        )
        assert rebuilt.shape == full.shape, (
            f"block {block_name}: rebuilt proj_out shape {tuple(rebuilt.shape)} != {tuple(full.shape)}"
        )
        expected = torch.cat(
            [full.narrow(1, 0, attn_width), full.narrow(1, attn_width, full.shape[1] - attn_width)], dim=1
        )
        torch.testing.assert_close(
            rebuilt,
            expected,
            rtol=0,
            atol=0,
            msg=f"block {block_name}: rank shards do not reconstruct the checkpoint proj_out weight",
        )


def _check_tp2_replicated(device_kind: DeviceKind) -> None:
    """With the env flag off, TP=2 must keep the replicated proj_out path."""
    world_size = 2
    checkpoint, baseline = _tp1_baseline(device_kind)
    reports = _spawn_ranks(world_size=world_size, device_kind=device_kind, sharded=False, checkpoint=checkpoint)
    rtol, atol = _tolerance(device_kind)

    for report in reports:
        rank = report["rank"]
        for block_name, use_sharded, is_fused_proj_out in report["sharded_blocks"]:
            assert not use_sharded, f"[rank {rank}] block {block_name} took the sharded path with the flag off"
            assert not is_fused_proj_out, f"[rank {rank}] block {block_name} built a fused proj_out with the flag off"
        assert not report["shards"], f"[rank {rank}] replicated path reported split shards"
        assert not report["unloaded"], f"[rank {rank}] unloaded checkpoint keys: {report['unloaded']}"
        torch.testing.assert_close(
            report["output"],
            baseline,
            rtol=rtol,
            atol=atol,
            msg=f"[rank {rank}] replicated TP=2 output diverged from the TP=1 baseline",
        )

    torch.testing.assert_close(
        reports[0]["output"],
        reports[1]["output"],
        rtol=0,
        atol=0,
        msg="replicated TP=2 ranks disagree on the output",
    )


# ---------------------------------------------------------------------------
# CPU: two real ranks over gloo collectives (runs in each PR)
# ---------------------------------------------------------------------------


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parallel
@pytest.mark.cpu
def test_tp2_sharded_proj_parity_and_shards():
    """Two real ranks: sharded proj_out loads its own shard and matches TP=1."""
    _check_tp2_sharded(device_kind="cpu")


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parallel
@pytest.mark.cpu
def test_tp2_replicated_proj_when_flag_disabled():
    """Two real ranks: with the flag off, TP=2 keeps replicated projections."""
    _check_tp2_replicated(device_kind="cpu")


# ---------------------------------------------------------------------------
# Nightly: the same checks on real multi-GPU device collectives
# ---------------------------------------------------------------------------


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.parallel
@pytest.mark.parametrize("world_size", [pytest.param(2, marks=_TWO_CARD)])
def test_tp2_sharded_proj_parity_and_shards_gpu(world_size: int):
    """Nightly: sharded proj_out parity on real multi-GPU collectives."""
    assert world_size == 2
    _check_tp2_sharded(device_kind="device")
