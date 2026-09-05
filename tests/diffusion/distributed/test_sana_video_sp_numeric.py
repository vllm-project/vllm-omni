# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real multi-process numeric equivalence gate for SANA-Video sequence parallel.

Runs the native transformer directly (no pipeline, no topology validation) with
real NCCL groups and compares SP>1 output against a same-weight SP=1 baseline.
In fp32 the only difference is reduction reordering, so any halo, gather,
timestep-slice or state-reduction misalignment fails by orders of magnitude.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    get_classifier_free_guidance_rank,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.platforms import current_omni_platform

_TINY_CONFIG = {
    "in_channels": 4,
    "out_channels": 4,
    "num_attention_heads": 2,
    "attention_head_dim": 12,
    "num_layers": 2,
    "num_cross_attention_heads": 2,
    "cross_attention_head_dim": 12,
    "cross_attention_dim": 24,
    "caption_channels": 8,
    "mlp_ratio": 2.0,
    "patch_size": (1, 2, 2),
    "sample_size": 8,
    "rope_max_seq_len": 512,
}

_MODEL_SEED = 42
_INPUT_SEED = 123
_FP32_REL_L2 = 1e-5
# bf16 collectives only reorder low-precision sums; informational sanity bound.
_BF16_REL_L2 = 5e-2


def _make_omni_config(sp_size: int, cfg_size: int, dtype: torch.dtype, tp_size: int = 1) -> OmniDiffusionConfig:
    parallel_config = DiffusionParallelConfig(
        pipeline_parallel_size=1,
        data_parallel_size=1,
        tensor_parallel_size=tp_size,
        sequence_parallel_size=sp_size,
        ulysses_degree=sp_size,
        ring_degree=1,
        cfg_parallel_size=cfg_size,
    )
    return OmniDiffusionConfig(
        model="test",
        dtype=dtype,
        parallel_config=parallel_config,
        # Backend-agnostic gate; SDPA also avoids platform quirks of the
        # default backend on older GPUs.
        diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA")),
    )


def _build_model(device: torch.device, dtype: torch.dtype):
    from vllm_omni.diffusion.models.sana_video import SanaVideoTransformer3DModel

    torch.manual_seed(_MODEL_SEED)
    model = SanaVideoTransformer3DModel(**_TINY_CONFIG).eval()
    for _, param in sorted(model.named_parameters()):
        torch.nn.init.normal_(param, mean=0.0, std=0.02)
    return model.to(device=device, dtype=dtype)


def _make_inputs(task: str, num_frames: int, input_seed: int):
    generator = torch.Generator().manual_seed(input_seed)
    latent = torch.randn(2, 4, num_frames, 8, 8, generator=generator)
    encoder_hidden_states = torch.randn(2, 6, 8, generator=generator)
    encoder_attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])
    if task == "t2v":
        timestep = torch.tensor([500.0, 700.0])
    else:
        timestep = torch.rand(2, 1, num_frames, 4, 4, generator=generator) * 1000.0
    return latent, encoder_hidden_states, encoder_attention_mask, timestep


def _worker(
    local_rank: int,
    world_size: int,
    sp_size: int,
    cfg_size: int,
    tp_size: int,
    dtype: torch.dtype,
    task: str,
    num_frames: int,
    weights_path: str,
    port: int,
    result_queue,
):
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)
    os.environ.update(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(port),
        }
    )

    init_distributed_environment()
    initialize_model_parallel(
        cfg_parallel_size=cfg_size,
        sequence_parallel_size=sp_size,
        ulysses_degree=sp_size,
        tensor_parallel_size=tp_size,
    )
    assert get_sequence_parallel_world_size() == sp_size

    cfg_rank = get_classifier_free_guidance_rank() if cfg_size > 1 else 0
    od_config = _make_omni_config(sp_size, cfg_size, dtype, tp_size)
    with set_current_diffusion_config(od_config):
        model = _build_model(device, dtype)
    # Load the baseline's full weights through the real weight loaders so TP
    # ranks hold proper shards; per-rank random init is not TP-consistent.
    model.load_weights(iter(torch.load(weights_path, map_location="cpu").items()))
    # Each CFG branch gets distinct inputs so a CFG/SP group mix-up cannot
    # cancel out.
    latent, encoder_hidden_states, encoder_attention_mask, timestep = _make_inputs(
        task, num_frames, _INPUT_SEED + cfg_rank
    )

    with torch.no_grad():
        output = model(
            latent.to(device=device, dtype=dtype),
            encoder_hidden_states.to(device=device, dtype=dtype),
            timestep.to(device=device),
            encoder_attention_mask=encoder_attention_mask.to(device=device),
            return_dict=False,
        )[0]

    result_queue.put((cfg_rank, get_sequence_parallel_rank(), output.float().cpu()))
    destroy_distributed_env()


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual - expected).double().norm() / expected.double().norm()).item()


def _spawn(worker_args: tuple, nprocs: int, result_queue) -> list:
    torch.multiprocessing.spawn(_worker, args=(*worker_args, result_queue), nprocs=nprocs)
    return [result_queue.get() for _ in range(nprocs)]


def _run_sp_parity(
    *,
    sp_size: int,
    cfg_size: int,
    dtype: torch.dtype,
    task: str,
    num_frames: int,
    rel_l2_bound: float,
    tp_size: int = 1,
) -> float:
    mp_context = torch.multiprocessing.get_context("spawn")
    manager = mp_context.Manager()

    baselines = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        weights_path = os.path.join(tmpdir, "baseline_weights.pt")
        for cfg_rank in range(cfg_size):
            queue = manager.Queue()
            # A world-1 run sees input seed _INPUT_SEED + 0; shift the seed via
            # the task inputs by spawning one baseline per CFG branch input.
            torch.multiprocessing.spawn(
                _baseline_worker,
                args=(
                    dtype,
                    task,
                    num_frames,
                    _INPUT_SEED + cfg_rank,
                    weights_path if cfg_rank == 0 else "",
                    29531,
                    queue,
                ),
                nprocs=1,
            )
            baselines[cfg_rank] = queue.get()

        queue = manager.Queue()
        world_size = sp_size * cfg_size * tp_size
        results = _spawn(
            (world_size, sp_size, cfg_size, tp_size, dtype, task, num_frames, weights_path, 29532), world_size, queue
        )

    worst = 0.0
    by_cfg: dict[int, list] = {}
    for cfg_rank, sp_rank, output in results:
        by_cfg.setdefault(cfg_rank, []).append((sp_rank, output))
    assert sorted(by_cfg) == list(range(cfg_size))
    for cfg_rank, rank_outputs in by_cfg.items():
        rank_outputs.sort(key=lambda item: item[0])
        first = rank_outputs[0][1]
        for sp_rank, output in rank_outputs[1:]:
            torch.testing.assert_close(
                output, first, rtol=0, atol=0, msg=f"SP ranks disagree (cfg_rank={cfg_rank}, sp_rank={sp_rank})"
            )
        drift = _rel_l2(first, baselines[cfg_rank])
        worst = max(worst, drift)
        assert drift <= rel_l2_bound, (
            f"SP{sp_size} output drifted rel_l2={drift:.3e} from the SP1 baseline "
            f"(bound {rel_l2_bound:.1e}, cfg_rank={cfg_rank}, task={task}, frames={num_frames})"
        )
    return worst


def _baseline_worker(
    local_rank: int,
    dtype: torch.dtype,
    task: str,
    num_frames: int,
    input_seed: int,
    weights_path: str,
    port: int,
    result_queue,
):
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)
    os.environ.update(
        {
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(port),
        }
    )

    init_distributed_environment()
    initialize_model_parallel()

    od_config = _make_omni_config(1, 1, dtype)
    with set_current_diffusion_config(od_config):
        model = _build_model(device, dtype)
    if weights_path:
        torch.save({name: param.detach().cpu() for name, param in model.named_parameters()}, weights_path)
    latent, encoder_hidden_states, encoder_attention_mask, timestep = _make_inputs(task, num_frames, input_seed)

    with torch.no_grad():
        output = model(
            latent.to(device=device, dtype=dtype),
            encoder_hidden_states.to(device=device, dtype=dtype),
            timestep.to(device=device),
            encoder_attention_mask=encoder_attention_mask.to(device=device),
            return_dict=False,
        )[0]

    result_queue.put(output.float().cpu())
    destroy_distributed_env()


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.parallel
@hardware_test(res={"cuda": "L4"}, num_cards=2)
@pytest.mark.parametrize("task", ["t2v", "i2v"])
@pytest.mark.parametrize("num_frames", [21, 5])
def test_sp2_matches_sp1_fp32(task: str, num_frames: int) -> None:
    _run_sp_parity(
        sp_size=2,
        cfg_size=1,
        dtype=torch.float32,
        task=task,
        num_frames=num_frames,
        rel_l2_bound=_FP32_REL_L2,
    )


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.parallel
@hardware_test(res={"cuda": "L4"}, num_cards=4)
@pytest.mark.parametrize("task", ["t2v", "i2v"])
@pytest.mark.parametrize("num_frames", [21, 5])
def test_sp4_matches_sp1_fp32(task: str, num_frames: int) -> None:
    _run_sp_parity(
        sp_size=4,
        cfg_size=1,
        dtype=torch.float32,
        task=task,
        num_frames=num_frames,
        rel_l2_bound=_FP32_REL_L2,
    )


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.parallel
@hardware_test(res={"cuda": "L4"}, num_cards=4)
def test_sp2_cfg2_matches_sp1_fp32() -> None:
    """Four processes: CFG branches with distinct inputs each run their own SP
    group; catches CFG/SP group crossover and deadlocks."""
    _run_sp_parity(
        sp_size=2,
        cfg_size=2,
        dtype=torch.float32,
        task="i2v",
        num_frames=21,
        rel_l2_bound=_FP32_REL_L2,
    )


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.parallel
@hardware_test(res={"cuda": "L4"}, num_cards=4)
def test_tp2_sp2_matches_sp1_fp32() -> None:
    """Four processes: TP shards heads and channels while SP shards frames; the
    combined mesh must still reproduce the serial output."""
    _run_sp_parity(
        sp_size=2,
        cfg_size=1,
        tp_size=2,
        dtype=torch.float32,
        task="t2v",
        num_frames=21,
        rel_l2_bound=_FP32_REL_L2,
    )


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.parallel
@hardware_test(res={"cuda": "L4"}, num_cards=2)
def test_sp2_bf16_drift_informational() -> None:
    """Records the bf16 reduction-reorder drift magnitude; the load-bearing
    numeric gate is the fp32 suite above."""
    drift = _run_sp_parity(
        sp_size=2,
        cfg_size=1,
        dtype=torch.bfloat16,
        task="t2v",
        num_frames=21,
        rel_l2_bound=_BF16_REL_L2,
    )
    print(f"\nSANA-Video SP2 bf16 drift vs SP1: rel_l2={drift:.3e}")
