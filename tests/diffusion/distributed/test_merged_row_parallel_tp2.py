# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real two-rank coverage for MergedRowParallelLinear in FLUX.1 and FLUX.2-dev.

The CPU block tests in ``tests/diffusion/layers/`` substitute the TP metadata and
the all-reduce. Here two processes run real collectives and the model's own
``load_weights`` on a diffusers-style checkpoint, and check on both ranks that

* every checkpoint key is consumed,
* each rank's merged output projection holds ``cat([W_attn^r, W_mlp^r])`` of the
  checkpoint,
* the output matches the TP=1 baseline, and both ranks agree after the all-reduce.

FLUX.2-dev additionally locks the fused-checkpoint contract: its single-stream
``to_qkv_mlp_proj`` is one tensor on disk and is loaded per segment (``shard_id=None``)
by ``MergedColumnParallelLinear``, so each rank must end up with its own shard of
every q/k/v/gate/up segment and no checkpoint remapping is needed.

Adapted from the two-rank harness in vllm-project/vllm-omni#5518.
"""

from __future__ import annotations

import os
import socket
from contextlib import contextmanager
from typing import Any, Literal

import pytest
import torch

from tests.helpers.distributed import collective_context, configure_cpu_collective_worker, start_collective_workers
from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.data import (
    AttentionConfig,
    AttentionSpec,
    DiffusionParallelConfig,
    OmniDiffusionConfig,
    TransformerConfig,
)

_TWO_CARD = hardware_marks(res={"cuda": ["L4", "B200"], "rocm": "MI325", "xpu": "B60"}, num_cards=2)

DeviceKind = Literal["cpu", "device"]

# Small configs with both block types; attention_head_dim >= sum(axes_dims_rope).
_NUM_HEADS = 4
_HEAD_DIM = 16
_NUM_SINGLE_LAYERS = 2
_MODEL_KWARGS: dict[str, Any] = dict(
    in_channels=8,
    num_layers=1,
    num_single_layers=_NUM_SINGLE_LAYERS,
    num_attention_heads=_NUM_HEADS,
    attention_head_dim=_HEAD_DIM,
    joint_attention_dim=16,
    pooled_projection_dim=8,
    axes_dims_rope=(4, 6, 6),
    guidance_embeds=False,
)
_FLUX2_KWARGS: dict[str, Any] = dict(
    in_channels=8,
    num_layers=1,
    num_single_layers=_NUM_SINGLE_LAYERS,
    num_attention_heads=_NUM_HEADS,
    attention_head_dim=_HEAD_DIM,
    joint_attention_dim=16,
    timestep_guidance_channels=32,
    mlp_ratio=3.0,
    axes_dims_rope=(4, 4, 4, 4),
    guidance_embeds=False,
)
_IMAGE_SEQ_LEN = 6
_TEXT_SEQ_LEN = 3

# FLUX.1 projects ``cat([attn, mlp])`` in ``proj_out``; FLUX.2-dev in ``attn.to_out``.
# mlp_ratio=4 (FLUX.1 default) and 3 (FLUX.2) give the second segment width.
_MERGED_PROJ = {"flux1": "proj_out", "flux2": "attn.to_out"}
_MLP_WIDTH = {"flux1": 4 * _NUM_HEADS * _HEAD_DIM, "flux2": 3 * _NUM_HEADS * _HEAD_DIM}


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _build_model(model_name: str, tp_size: int, device: torch.device):
    from vllm_omni.diffusion.config import set_current_diffusion_config

    # TORCH_SDPA: fp32 parity on CPU and on devices; FlashAttention backends have no
    # CPU kernel and reject fp32.
    kwargs: dict[str, Any] = dict(
        tf_model_config=TransformerConfig(params={"num_layers": _MODEL_KWARGS["num_layers"]}),
        parallel_config=DiffusionParallelConfig(tensor_parallel_size=tp_size),
        diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA")),
    )
    if model_name == "flux2":
        # FLUX.2-dev gates its local single-stream TP on the configured dtype; the
        # parameters themselves stay fp32 so CPU and device parity use one tolerance.
        kwargs["dtype"] = torch.bfloat16
    od_config = OmniDiffusionConfig(**kwargs)
    torch.manual_seed(0)
    with set_current_diffusion_config(od_config):
        if model_name == "flux2":
            from vllm_omni.diffusion.models.flux2.flux2_transformer import Flux2Transformer2DModel

            model = Flux2Transformer2DModel(od_config=od_config, **_FLUX2_KWARGS)
        else:
            from vllm_omni.diffusion.models.flux.flux_transformer import FluxTransformer2DModel

            model = FluxTransformer2DModel(od_config=od_config, **_MODEL_KWARGS)
    return model.to(device=device).eval()


def _make_checkpoint(model) -> dict[str, torch.Tensor]:
    """Diffusers-style checkpoint: unsharded tensors, fused to_qkv split into q/k/v."""
    torch.manual_seed(7)
    checkpoint: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        weight = torch.randn_like(param, device="cpu") * 0.05
        if ".to_qkv." in name or ".add_kv_proj." in name:
            fused = ".to_qkv." if ".to_qkv." in name else ".add_kv_proj."
            names = ("to_q", "to_k", "to_v") if fused == ".to_qkv." else ("add_q_proj", "add_k_proj", "add_v_proj")
            prefix, _, suffix = name.rpartition(fused)
            for part, tensor in zip(names, torch.chunk(weight, 3, dim=0)):
                checkpoint[f"{prefix}.{part}.{suffix}"] = tensor
        else:
            checkpoint[name] = weight
    return checkpoint


def _make_inputs(model_name: str, device: torch.device) -> dict:
    torch.manual_seed(1234)
    config = _FLUX2_KWARGS if model_name == "flux2" else _MODEL_KWARGS
    kwargs = dict(
        hidden_states=torch.randn(1, _IMAGE_SEQ_LEN, config["in_channels"]),
        encoder_hidden_states=torch.randn(1, _TEXT_SEQ_LEN, config["joint_attention_dim"]),
        timestep=torch.tensor([0.5]),
    )
    if model_name == "flux2":
        num_axes = len(_FLUX2_KWARGS["axes_dims_rope"])
        kwargs.update(
            img_ids=torch.zeros(_IMAGE_SEQ_LEN, num_axes),
            txt_ids=torch.zeros(_TEXT_SEQ_LEN, num_axes),
        )
    else:
        kwargs.update(
            pooled_projections=torch.randn(1, _MODEL_KWARGS["pooled_projection_dim"]),
            img_ids=torch.zeros(_IMAGE_SEQ_LEN, 3),
            txt_ids=torch.zeros(_TEXT_SEQ_LEN, 3),
        )
    return {name: tensor.to(device) for name, tensor in kwargs.items()}


def _worker_device(local_rank: int, device_kind: DeviceKind) -> torch.device:
    if device_kind == "cpu":
        return torch.device("cpu")
    from vllm_omni.platforms import current_omni_platform

    return torch.device(f"{current_omni_platform.device_type}:{local_rank}")


def _init_dist(local_rank: int, world_size: int, master_port: int, device_kind: DeviceKind) -> torch.device:
    from vllm.distributed.parallel_state import get_tp_group, init_distributed_environment, initialize_model_parallel
    from vllm.platforms import current_platform

    device = _worker_device(local_rank, device_kind)
    if device_kind != "cpu":
        from vllm_omni.platforms import current_omni_platform

        current_omni_platform.set_device(device)
    os.environ.update(
        RANK=str(local_rank),
        LOCAL_RANK=str(local_rank),
        WORLD_SIZE=str(world_size),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(master_port),
    )
    backend = "gloo" if device_kind == "cpu" else current_platform.dist_backend
    init_distributed_environment(
        world_size=world_size,
        rank=local_rank,
        local_rank=local_rank,
        distributed_init_method=f"tcp://127.0.0.1:{master_port}",
        backend=backend,
    )
    initialize_model_parallel(tensor_model_parallel_size=world_size, backend=backend)
    if device_kind == "cpu":
        # torch.ops.vllm.* collectives have no CPU kernel; take the eager gloo path.
        get_tp_group().use_custom_op_call = False
    return device


def _cleanup_dist() -> None:
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    cleanup_dist_env_and_memory()
    for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"):
        os.environ.pop(key, None)


@contextmanager
def _worker_vllm_config(device_kind: DeviceKind):
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.platforms import current_platform

    device = "cpu" if device_kind == "cpu" else current_platform.device_type
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device=device))):
        yield


# Packed destinations: ``load_weights`` reports either the checkpoint key (FLUX.1) or
# the packed parameter it was folded into (FLUX.2-dev).
_PACKED_DESTINATIONS = (
    (".to_q.", ".to_qkv."),
    (".to_k.", ".to_qkv."),
    (".to_v.", ".to_qkv."),
    (".add_q_proj.", ".add_kv_proj."),
    (".add_k_proj.", ".add_kv_proj."),
    (".add_v_proj.", ".add_kv_proj."),
)


def _unloaded(checkpoint, loaded: set[str]) -> list[str]:
    missing = []
    for name in checkpoint:
        packed = name
        for source, destination in _PACKED_DESTINATIONS:
            if source in name:
                packed = name.replace(source, destination)
                break
        if name not in loaded and packed not in loaded:
            missing.append(name)
    return sorted(missing)


def _submodule(module, path: str):
    for part in path.split("."):
        module = getattr(module, part)
    return module


def _run(model_name: str, world_size: int, rank: int, port: int, device_kind: DeviceKind, checkpoint):
    device = _init_dist(rank, world_size, port, device_kind)
    try:
        model = _build_model(model_name, world_size, device)
        if checkpoint is None:
            checkpoint = _make_checkpoint(model)
        loaded = model.load_weights([(name, weight.clone()) for name, weight in checkpoint.items()])
        with torch.no_grad():
            (output,) = model(**_make_inputs(model_name, device), return_dict=False)
        blocks = dict(model.single_transformer_blocks.named_children())
        merged = {
            name: _submodule(block, _MERGED_PROJ[model_name]).weight.detach().cpu().clone()
            for name, block in blocks.items()
        }
        # FLUX.2-dev also loads the fused q/k/v/gate/up projection per segment.
        fused_in = {}
        if model_name == "flux2":
            fused_in = {
                name: block.attn.to_qkv_mlp_proj.weight.detach().cpu().clone() for name, block in blocks.items()
            }
        return dict(
            rank=rank,
            checkpoint=checkpoint,
            unloaded=_unloaded(checkpoint, loaded),
            output=output.float().cpu().clone(),
            merged=merged,
            fused_in=fused_in,
            tp_sizes=[_submodule(block, _MERGED_PROJ[model_name]).tp_size for block in blocks.values()],
        )
    finally:
        _cleanup_dist()


def _rank_worker(local_rank, model_name, world_size, port, device_kind, checkpoint, queue) -> None:
    if device_kind == "cpu":
        configure_cpu_collective_worker()
    with _worker_vllm_config(device_kind):
        queue.put(_run(model_name, world_size, local_rank, port, device_kind, checkpoint))


def _rank_major(full: torch.Tensor, segments: list[int], world_size: int, rank: int, dim: int) -> torch.Tensor:
    """``cat([W_0^rank, W_1^rank, ...])`` of an unsharded checkpoint tensor."""
    return torch.cat([part.chunk(world_size, dim=dim)[rank] for part in full.split(segments, dim=dim)], dim=dim)


def _check_tp2(model_name: str, device_kind: DeviceKind) -> None:
    world_size = 2
    baseline = _run(model_name, 1, 0, _find_free_port(), device_kind, None)
    assert not baseline["unloaded"], baseline["unloaded"]
    checkpoint = baseline["checkpoint"]
    with collective_context(device_kind).Manager() as manager:
        queue = manager.Queue()
        start_collective_workers(
            _rank_worker,
            args=(model_name, world_size, _find_free_port(), device_kind, checkpoint, queue),
            nprocs=world_size,
            device_kind=device_kind,
        )
        reports = sorted((queue.get() for _ in range(world_size)), key=lambda report: report["rank"])

    rtol, atol = (1e-4, 1e-4) if device_kind == "cpu" else (2e-3, 2e-3)
    inner_dim = _NUM_HEADS * _HEAD_DIM
    mlp_dim = _MLP_WIDTH[model_name]
    proj = _MERGED_PROJ[model_name]
    for report in reports:
        rank = report["rank"]
        assert report["tp_sizes"] == [world_size] * _NUM_SINGLE_LAYERS
        assert not report["unloaded"], f"[rank {rank}] unloaded checkpoint keys: {report['unloaded']}"
        for name, weight in report["merged"].items():
            full = checkpoint[f"single_transformer_blocks.{name}.{proj}.weight"]
            expected = _rank_major(full, [inner_dim, mlp_dim], world_size, rank, dim=1)
            torch.testing.assert_close(weight, expected, rtol=0, atol=0, msg=f"[rank {rank}] block {name} {proj}")
        for name, weight in report["fused_in"].items():
            full = checkpoint[f"single_transformer_blocks.{name}.attn.to_qkv_mlp_proj.weight"]
            # q | k | v | SwiGLU gate | SwiGLU up, each sharded on its own output dim.
            segments = [inner_dim] * 3 + [mlp_dim] * 2
            expected = _rank_major(full, segments, world_size, rank, dim=0)
            torch.testing.assert_close(
                weight, expected, rtol=0, atol=0, msg=f"[rank {rank}] block {name} to_qkv_mlp_proj"
            )
        torch.testing.assert_close(report["output"], baseline["output"], rtol=rtol, atol=atol)
    torch.testing.assert_close(reports[0]["output"], reports[1]["output"], rtol=0, atol=0)


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parallel
@pytest.mark.cpu
@pytest.mark.parametrize("model_name", ["flux1", "flux2"])
def test_flux_merged_row_tp2_parity_cpu(model_name: str):
    _check_tp2(model_name, "cpu")


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.parallel
@pytest.mark.parametrize("model_name", ["flux1", "flux2"])
@pytest.mark.parametrize("world_size", [pytest.param(2, marks=_TWO_CARD)])
def test_flux_merged_row_tp2_parity_device(model_name: str, world_size: int):
    assert world_size == 2
    _check_tp2(model_name, "device")
