# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Same-weight SANA-Video 2.0 dense/SP numerical checks with real collectives."""

from __future__ import annotations

import os
import socket
import time

import pytest
import torch

from tests.helpers.mark import hardware_marks

_SEED = 8006
_CONFIG = dict(
    in_channels=2,
    hidden_size=120,
    depth=32,
    num_heads=20,
    caption_channels=16,
    model_max_length=4,
    linear_head_dim=6,
    softmax_head_dim=12,
    softmax_ratio=0.25,
    mlp_ratio=1.0,
    attn_res_block_size=8,
)
_REL_L2_LIMIT = 1e-5
_ABS_LIMIT = 1e-5


def _model(device: torch.device):
    assert torch.get_float32_matmul_precision() == "highest"
    assert not torch.backends.cuda.matmul.allow_tf32
    from vllm_omni.diffusion.models.sana_video2.transformer_sana_video2 import (
        SanaVideo2TransformerConfig,
        SanaVideo2TransformerModel,
    )

    torch.manual_seed(_SEED)
    model = SanaVideo2TransformerModel(SanaVideo2TransformerConfig(**_CONFIG)).to(device).eval()
    # The release initializes these depth projections at zero. Exercise learned
    # Attention Residuals so a local-depth shortcut cannot pass this check.
    for projection in (model.attn_res.attn_proj, model.attn_res.mlp_proj, model.attn_res.final_proj):
        torch.nn.init.normal_(projection.weight, std=0.1)
    assert len(model.blocks) == 32
    assert model.block_attention_types.count("linear") == 24
    assert model.block_attention_types.count("softmax") == 8
    assert model.blocks[3].attn.heads == 10
    return model


def _inputs(frames: int, height: int, width: int, task: str, seed: int, device: torch.device):
    generator = torch.Generator().manual_seed(seed)
    latent = torch.randn(2, 2, frames, height, width, generator=generator).to(device)
    text = torch.randn(2, 4, 16, generator=generator).to(device)
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]], device=device)
    if task == "t2v":
        timestep = torch.tensor([275.0, 725.0], device=device)
    else:
        # Different per-frame times make a token shard crossing a frame visible.
        timestep = torch.arange(frames, device=device).float().reshape(1, 1, frames, 1, 1)
        timestep = torch.cat((timestep * 127, timestep * 83 + 11), dim=0)
    return latent, timestep, text, mask


def _trajectory(model, task: str, seed: int, device: torch.device):
    from vllm_omni.diffusion.models.sana_video2.sampling import sample_flow_dpm, sample_ltx_euler

    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 2, 3, 2, 2, generator=generator).to(device)
    text = torch.randn(2, 4, 16, generator=generator).to(device)
    text_mask = torch.tensor([[True, True, False, False], [True, True, True, False]], device=device)
    steps = []

    def predict(latents, timestep):
        # Native batch CFG: both branches run together in one model call.
        inputs = torch.cat((latents, latents))
        if task == "t2v":
            time = timestep.expand(2) * 1000
        else:
            time = timestep.expand(2, 1, 3, 1, 1)
        outputs = model(inputs, time, text, text_mask)
        if task == "t2v":
            # DPM-Solver consumes noise. Match the pipeline's flow-to-noise
            # conversion before mixing unconditional and conditional branches.
            sigma = timestep.reshape((1,) * latents.ndim).to(inputs)
            outputs = (1 - sigma) * outputs + inputs
        uncond, cond = outputs.chunk(2)
        return uncond + 8.0 * (cond - uncond)

    def callback(_step, _time, current):
        steps.append(current.cpu().clone())

    sampler = sample_flow_dpm if task == "t2v" else sample_ltx_euler
    sampler(predict, x, 5, 1.5, callback)
    return steps


def _reference(cases, device: torch.device):
    model = _model(device)
    outputs = {}
    # Keep the patch Conv3d in IEEE FP32 as well as the attention layers.
    with torch.no_grad(), torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
        for index, (task, dims) in enumerate(cases):
            outputs[f"forward_{index}"] = model(*_inputs(*dims, task, _SEED + index + 1, device)).cpu()
        outputs["sampling_t2v"] = _trajectory(model, "t2v", _SEED + 101, device)
        outputs["sampling_ti2v"] = _trajectory(model, "ti2v", _SEED + 102, device)
    return model.state_dict(), outputs


def _worker(rank: int, world_size: int, backend: str, mode: str, port: int, directory: str, cases):
    from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_distributed_env,
        get_sp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.platforms import current_omni_platform

    torch.set_num_threads(1)
    local_rank = 0 if backend == "gloo" else rank
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(local_rank),
        WORLD_SIZE=str(world_size),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    device = torch.device("cpu" if backend == "gloo" else f"cuda:{rank}")
    if backend == "nccl":
        current_omni_platform.set_device(device)
    init_distributed_environment(local_rank=local_rank, backend=backend)
    try:
        initialize_model_parallel(
            sequence_parallel_size=world_size,
            ulysses_degree=world_size,
            ring_degree=1,
            backend=backend,
        )
        parallel = DiffusionParallelConfig(
            sequence_parallel_size=world_size,
            ulysses_degree=world_size,
            ring_degree=1,
            ulysses_mode=mode,
        )
        config = OmniDiffusionConfig(model=directory, dtype=torch.float32, parallel_config=parallel)
        model = _model(device)
        model.load_state_dict(torch.load(f"{directory}/weights.pt", map_location=device, weights_only=True))
        sp_group = get_sp_group()
        model.set_sequence_parallel(sp_group)
        results = {}
        with (
            torch.no_grad(),
            torch.backends.cudnn.flags(enabled=True, allow_tf32=False),
            set_forward_context(omni_diffusion_config=config),
        ):
            # All ranks reject the short request before any collective.
            short = _inputs(1, 1, 1, "t2v", _SEED + 99, device)
            try:
                model(*short)
            except ValueError as error:
                assert "tokens" in str(error) and "SP" in str(error)
            else:
                raise AssertionError("N < SP was accepted")
            for index, (task, dims) in enumerate(cases):
                local = _inputs(*dims, task, _SEED + index + 1 + rank * 1000, device)
                replicated = tuple(sp_group.broadcast(value.contiguous()) for value in local)
                results[f"forward_{index}"] = model(*replicated).cpu()
            results["sampling_t2v"] = _trajectory(model, "t2v", _SEED + 101, device)
            results["sampling_ti2v"] = _trajectory(model, "ti2v", _SEED + 102, device)
        torch.save(results, f"{directory}/rank_{rank}.pt")
    finally:
        destroy_distributed_env()


def _port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run(world_size: int, backend: str, mode: str, tmp_path, cases):
    import torch.multiprocessing as mp

    device = torch.device("cpu" if backend == "gloo" else "cuda:0")
    weights, expected = _reference(cases, device)
    torch.save({name: value.cpu() for name, value in weights.items()}, tmp_path / "weights.pt")
    del weights
    if backend == "nccl":
        torch.accelerator.empty_cache()
    context = mp.spawn(
        _worker, args=(world_size, backend, mode, _port(), str(tmp_path), cases), nprocs=world_size, join=False
    )
    try:
        deadline = time.monotonic() + 180
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"SP{world_size} {backend} workers exceeded 180 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)

    for rank in range(world_size):
        actual = torch.load(tmp_path / f"rank_{rank}.pt", map_location="cpu", weights_only=True)
        for name, reference in expected.items():
            observed = actual[name]
            pairs = zip(observed, reference) if isinstance(reference, list) else [(observed, reference)]
            for step, (result, target) in enumerate(pairs):
                delta = (result.double() - target.double()).flatten()
                absolute = delta.abs().max().item()
                relative = (delta.norm() / target.double().flatten().norm().clamp_min(1e-12)).item()
                print(
                    f"SP{world_size} {backend} {mode} rank={rank} {name} step={step} "
                    f"max_abs={absolute:.3e} rel_l2={relative:.3e}"
                )
                torch.testing.assert_close(result, target, rtol=1e-5, atol=_ABS_LIMIT)
                assert relative <= _REL_L2_LIMIT, (
                    f"{name} step={step} rank={rank}: max_abs={absolute:.3e}, rel_l2={relative:.3e}"
                )


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parallel
@pytest.mark.cpu
@pytest.mark.parametrize(
    "world_size,mode,cases",
    [
        (2, "strict", [("t2v", (3, 2, 2)), ("ti2v", (3, 2, 2)), ("t2v", (4, 2, 2))]),
        (4, "advanced_uaa", [("ti2v", (5, 1, 3)), ("t2v", (5, 1, 3)), ("ti2v", (4, 2, 2))]),
    ],
)
def test_gloo_fp32_dense_equivalence(world_size, mode, cases, tmp_path):
    _run(world_size, "gloo", mode, tmp_path, cases)


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.parallel
@pytest.mark.parametrize(
    "world_size,mode",
    [
        pytest.param(2, "strict", marks=hardware_marks(res={"cuda": ["L4", "B200"]}, num_cards=2)),
        pytest.param(2, "advanced_uaa", marks=hardware_marks(res={"cuda": ["L4", "B200"]}, num_cards=2)),
        pytest.param(4, "advanced_uaa", marks=hardware_marks(res={"cuda": ["L4", "B200"]}, num_cards=4)),
        pytest.param(8, "advanced_uaa", marks=hardware_marks(res={"cuda": ["L4", "B200"]}, num_cards=8)),
    ],
)
def test_nccl_fp32_dense_equivalence(world_size, mode, tmp_path):
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"requires {world_size} accessible CUDA devices")
    dims = (5, 2, 2) if mode == "strict" else (5, 1, 3)
    _run(world_size, "nccl", mode, tmp_path, [("ti2v", dims), ("t2v", dims), ("ti2v", (4, 2, 2))])
