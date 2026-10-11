# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU checks for HiDream-O1-Image CFG through ``CFGParallelMixin``.

The model call is stubbed, everything else in ``forward`` is real. Sequential
CFG must reproduce the loop the pipeline ran before it used the mixin, and two
CFG ranks, each running its own ``forward``, must agree with that result while
calling the model for one branch only.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.distributed import cfg_parallel as cfg_mod
from vllm_omni.diffusion.models.hidream_o1_image.pipeline_hidream_o1_image import (
    T_EPS,
    HiDreamO1ImagePipeline,
    build_hidream_o1_scheduler,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

HEIGHT = WIDTH = 64
STEPS = 4
SEED = 7


def _x_pred(branch: str, z: torch.Tensor, t_pixeldit: torch.Tensor) -> torch.Tensor:
    gain = 0.25 if branch == "pos" else -0.5
    return (z.float() * gain + t_pixeldit.float()).to(torch.bfloat16)


def _pipeline(guidance_scale: float) -> tuple[HiDreamO1ImagePipeline, list[str]]:
    pipe = HiDreamO1ImagePipeline.__new__(HiDreamO1ImagePipeline)
    nn.Module.__init__(pipe)
    pipe.dtype = torch.bfloat16
    pipe.device = torch.device("cpu")
    calls: list[str] = []

    def forward_once(sample: dict, z_in: torch.Tensor, t_pixeldit: torch.Tensor) -> torch.Tensor:
        calls.append(sample["branch"])
        return _x_pred(sample["branch"], z_in, t_pixeldit)

    pipe._resolve_generation_params = lambda req: ("a cat", HEIGHT, WIDTH, STEPS, SEED, guidance_scale)
    pipe._build_and_validate_sample = lambda prompt, *args: {"branch": "neg" if prompt == " " else "pos"}
    pipe._forward_once = forward_once
    return pipe, calls


def _reference_z(pipe: HiDreamO1ImagePipeline, guidance_scale: float) -> torch.Tensor:
    """The sequential denoising loop as it was written before the mixin."""
    z = pipe._prepare_noise_and_patchify(HEIGHT, WIDTH, SEED, pipe.dtype, pipe.device)
    sched = build_hidream_o1_scheduler(num_inference_steps=STEPS, shift=3.0, device=pipe.device)
    for step_t in sched.timesteps:
        step_t = step_t.float()
        t_pixeldit = 1.0 - step_t / 1000.0
        sigma = (step_t / 1000.0).to(dtype=torch.float32).clamp_min(T_EPS)
        v_guided = (_x_pred("pos", z, t_pixeldit).to(torch.float32) - z.to(torch.float32)) / sigma
        if guidance_scale > 1.0:
            v_uncond = (_x_pred("neg", z, t_pixeldit).to(torch.float32) - z.to(torch.float32)) / sigma
            v_guided = v_uncond + guidance_scale * (v_guided - v_uncond)
        z = sched.step(-v_guided, step_t, z.float(), return_dict=False)[0].to(pipe.dtype)
    return z


class _TwoRankGroup:
    """All-gather between two threads that each play one CFG rank."""

    def __init__(self, rank_of_thread: threading.local) -> None:
        self.rank_of_thread = rank_of_thread
        self.slots: dict[int, torch.Tensor] = {}
        self.filled = threading.Barrier(2, timeout=30)
        self.read = threading.Barrier(2, timeout=30)

    def all_gather(self, input_: torch.Tensor, dim: int = 0, separate_tensors: bool = False) -> list[torch.Tensor]:
        del dim
        assert separate_tensors
        self.slots[self.rank_of_thread.rank] = input_.clone()
        self.filled.wait()
        gathered = [self.slots[rank].clone() for rank in (0, 1)]
        self.read.wait()
        return gathered


def test_static_config_accepts_cfg_parallel() -> None:
    pipe, _ = _pipeline(5.0)
    pipe.od_config = SimpleNamespace(parallel_config=SimpleNamespace(cfg_parallel_size=2))
    pipe._validate_static_config()


@pytest.mark.parametrize(
    ("guidance_scale", "calls_per_step"),
    [(5.0, ["pos", "neg"]), (1.0, ["pos"])],
)
def test_single_rank_matches_the_loop_before_the_mixin(guidance_scale: float, calls_per_step: list[str]) -> None:
    pipe, calls = _pipeline(guidance_scale)

    output = pipe.forward(SimpleNamespace())

    z, height, width = output.output
    assert (height, width) == (HEIGHT, WIDTH)
    assert calls == calls_per_step * STEPS
    assert torch.equal(z, _reference_z(pipe, guidance_scale))


def test_two_cfg_ranks_run_one_branch_each_and_agree(monkeypatch: pytest.MonkeyPatch) -> None:
    guidance_scale = 5.0
    rank_of_thread = threading.local()
    group = _TwoRankGroup(rank_of_thread)
    monkeypatch.setattr(cfg_mod, "_get_cfg_world_size_or_one", lambda: 2)
    monkeypatch.setattr(cfg_mod, "get_classifier_free_guidance_rank", lambda: rank_of_thread.rank)
    monkeypatch.setattr(cfg_mod, "get_cfg_group", lambda: group)

    def run_rank(rank: int) -> tuple[torch.Tensor, list[str]]:
        rank_of_thread.rank = rank
        pipe, calls = _pipeline(guidance_scale)
        return pipe.forward(SimpleNamespace()).output[0], calls

    with ThreadPoolExecutor(max_workers=2) as pool:
        (z0, calls0), (z1, calls1) = (future.result() for future in [pool.submit(run_rank, r) for r in (0, 1)])

    assert calls0 == ["pos"] * STEPS
    assert calls1 == ["neg"] * STEPS
    reference_pipe, _ = _pipeline(guidance_scale)
    reference = _reference_z(reference_pipe, guidance_scale)
    assert torch.equal(z0, reference)
    assert torch.equal(z1, reference)
