# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CFG (Classifier-Free Guidance) parallel functionality.

CPU tests mock the CFG process group and validate mixin logic without GPUs.
Nightly GPU parity tests compare sequential CFG (cfg_parallel_size=1) against
real multi-GPU CFG parallel under fixed seeds.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import torch

from tests.helpers.mark import hardware_marks, hardware_test
from vllm_omni.diffusion.distributed import parallel_state
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin, _dispatch_branches, _wrap
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.platforms import current_omni_platform

_L4_TWO_GPU = hardware_marks(res={"cuda": "L4"}, num_cards=2)
_L4_THREE_GPU = hardware_marks(res={"cuda": "L4"}, num_cards=3)


def _set_random_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _init_model_weights(module: torch.nn.Module, seed: int) -> None:
    _set_random_seeds(seed)
    for param in module.parameters():
        torch.nn.init.normal_(param, mean=0.0, std=0.02)


def _cfg_close_tolerance(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    if dtype == torch.bfloat16:
        return 1e-2, 1e-2
    return 1e-3, 1e-3


def _assert_cfg_outputs_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    dtype: torch.dtype,
    msg: str,
) -> None:
    rtol, atol = _cfg_close_tolerance(dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=msg)


def _update_environment_variables(envs_dict: dict[str, str]) -> None:
    for key, value in envs_dict.items():
        os.environ[key] = value


class SimpleTransformer(torch.nn.Module):
    def __init__(self, in_channels: int = 4, hidden_dim: int = 128, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        self.input_proj = torch.nn.Conv2d(in_channels, hidden_dim, 1)
        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.final_proj = torch.nn.Conv2d(hidden_dim, in_channels, 1)
        self.norm1 = torch.nn.LayerNorm(hidden_dim)
        self.norm2 = torch.nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor]:
        del kwargs
        batch_size, _channels, height, width = x.shape
        x = self.input_proj(x)
        x = x.flatten(2).transpose(1, 2)
        residual = x
        x = self.norm1(x)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        seq_len = height * width
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scale = self.head_dim**-0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        attn_output = self.out_proj(attn_output)
        x = residual + attn_output
        residual = x
        x = self.norm2(x)
        x = residual + x
        x = x.transpose(1, 2).view(batch_size, self.hidden_dim, height, width)
        return (self.final_proj(x),)


class TestCFGPipeline(CFGParallelMixin):
    def __init__(self, in_channels: int = 4, hidden_dim: int = 128, seed: int = 42):
        _set_random_seeds(seed)
        self.transformer = SimpleTransformer(in_channels, hidden_dim)
        _init_model_weights(self.transformer, seed)


class MultiBranchTestPipeline(CFGParallelMixin):
    def __init__(self, in_channels: int = 4, hidden_dim: int = 128, seed: int = 42):
        _set_random_seeds(seed)
        self.transformer = SimpleTransformer(in_channels, hidden_dim)
        _init_model_weights(self.transformer, seed)

    def combine_multi_branch_cfg_noise(self, predictions, true_cfg_scale, cfg_normalize=False):
        if len(predictions) == 4:
            text_scale = true_cfg_scale["text"]
            image_scale = true_cfg_scale["image"]
            vid_ref_scale = true_cfg_scale["vid_ref"]
            pos, neg, vid_neg, audio_neg = predictions
            combined = (
                audio_neg
                + vid_ref_scale * (vid_neg - audio_neg)
                + image_scale * (neg - vid_neg)
                + text_scale * (pos - neg)
            )
        elif len(predictions) == 3:
            text_scale = true_cfg_scale["text"]
            image_scale = true_cfg_scale["image"]
            pos, ref, uncond = predictions
            combined = uncond + image_scale * (ref - uncond) + text_scale * (pos - ref)
        else:
            pos, neg = predictions[0], predictions[1]
            combined = neg + true_cfg_scale * (pos - neg)

        if cfg_normalize:
            combined = self.cfg_normalize_function(pos, combined)
        return combined


class FakeCfgGroup:
    def __init__(self, *, world_size: int, rank_in_group: int, rank_tensors: list[torch.Tensor]):
        self.world_size = world_size
        self.rank_in_group = rank_in_group
        self._rank_tensors = rank_tensors

    def all_gather(
        self,
        input_: torch.Tensor,
        dim: int = 0,
        separate_tensors: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        del dim, input_
        assert separate_tensors
        return [tensor.clone() for tensor in self._rank_tensors]


class FakeMultiBranchCfgGroup:
    def __init__(
        self,
        *,
        world_size: int,
        rank_in_group: int,
        slots_per_rank: list[list[tuple[torch.Tensor, ...]]],
    ):
        self.world_size = world_size
        self.rank_in_group = rank_in_group
        self._slots_per_rank = slots_per_rank
        # Production path re-runs predict_noise, so gathered tensors are value-equal
        # but not identity-equal to the precomputed slot table. Index by call order
        # instead: slot-major then element-major, matching _predict_multi_branch_parallel.
        self._call_idx = 0

    def all_gather(
        self,
        input_: torch.Tensor,
        dim: int = 0,
        separate_tensors: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        del dim, input_
        assert separate_tensors
        n_elems = len(self._slots_per_rank[0][0])
        slot_idx, elem_idx = divmod(self._call_idx, n_elems)
        self._call_idx += 1
        assert slot_idx < len(self._slots_per_rank[self.rank_in_group]), (
            f"all_gather call {self._call_idx} exceeds slot table "
            f"(slot_idx={slot_idx}, n_slots={len(self._slots_per_rank[self.rank_in_group])})"
        )
        return [self._slots_per_rank[rank][slot_idx][elem_idx].clone() for rank in range(self.world_size)]


def _patch_cfg_state(
    monkeypatch: pytest.MonkeyPatch,
    *,
    world_size: int,
    rank: int,
    fake_group: FakeCfgGroup | FakeMultiBranchCfgGroup,
) -> None:
    # Patch both the source module and cfg_parallel's bound imports.
    # cfg_parallel does `from ...parallel_state import get_cfg_group`, so
    # patching parallel_state alone does not affect CFGParallelMixin.
    from vllm_omni.diffusion.distributed import cfg_parallel as cfg_parallel_mod

    for module in (parallel_state, cfg_parallel_mod):
        monkeypatch.setattr(module, "get_classifier_free_guidance_world_size", lambda: world_size)
        monkeypatch.setattr(module, "get_classifier_free_guidance_rank", lambda: rank)
        monkeypatch.setattr(module, "get_cfg_group", lambda: fake_group)


def _make_two_branch_inputs(
    *,
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    input_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _set_random_seeds(input_seed)
    positive = torch.randn(batch_size, channels, height, width, dtype=dtype, device=device)
    _set_random_seeds(input_seed + 1)
    negative = torch.randn(batch_size, channels, height, width, dtype=dtype, device=device)
    return {"x": positive}, {"x": negative}


def _make_multi_branch_inputs(
    *,
    n_branches: int,
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    input_seed: int,
) -> list[dict[str, Any]]:
    branches_kwargs: list[dict[str, Any]] = []
    for branch_idx in range(n_branches):
        _set_random_seeds(input_seed + branch_idx)
        x = torch.randn(batch_size, channels, height, width, dtype=dtype, device=device)
        branches_kwargs.append({"x": x})
    return branches_kwargs


def _multi_branch_cfg_scale(n_branches: int) -> float | dict[str, float]:
    if n_branches == 2:
        return 5.0
    if n_branches == 3:
        return {"text": 5.0, "image": 2.0}
    return {"text": 5.0, "image": 2.0, "vid_ref": 1.5}


def _build_multi_branch_slot_table(
    *,
    n_branches: int,
    cfg_world_size: int,
    branch_preds: list[torch.Tensor | tuple[torch.Tensor, ...]],
) -> list[list[tuple[torch.Tensor, ...]]]:
    assignments = _dispatch_branches(n_branches, cfg_world_size)
    max_per_rank = max(len(branch_ids) for branch_ids in assignments)
    ref_pred = _wrap(branch_preds[0])
    slots_per_rank: list[list[tuple[torch.Tensor, ...]]] = []
    for branch_ids in assignments:
        slots = [_wrap(branch_preds[branch_id]) for branch_id in branch_ids]
        while len(slots) < max_per_rank:
            slots.append(tuple(torch.zeros_like(tensor) for tensor in ref_pred))
        slots_per_rank.append(slots)
    return slots_per_rank


def _run_two_branch_cfg_parallel_mock(
    pipeline: TestCFGPipeline,
    monkeypatch: pytest.MonkeyPatch,
    *,
    cfg_world_size: int,
    positive_kwargs: dict[str, Any],
    negative_kwargs: dict[str, Any],
    true_cfg_scale: float,
    cfg_normalize: bool,
    extra_kwargs: dict[str, Any] | None,
) -> torch.Tensor:
    with torch.no_grad():
        pos_tensor = _wrap(pipeline.predict_noise(**positive_kwargs))[0]
        neg_tensor = _wrap(pipeline.predict_noise(**negative_kwargs))[0]

    outputs: list[torch.Tensor] = []
    for cfg_rank in range(cfg_world_size):
        fake_group = FakeCfgGroup(
            world_size=cfg_world_size,
            rank_in_group=cfg_rank,
            rank_tensors=[pos_tensor, neg_tensor],
        )
        _patch_cfg_state(monkeypatch, world_size=cfg_world_size, rank=cfg_rank, fake_group=fake_group)
        with torch.no_grad():
            output = pipeline.predict_noise_maybe_with_cfg(
                do_true_cfg=True,
                true_cfg_scale=true_cfg_scale,
                positive_kwargs=positive_kwargs,
                negative_kwargs=negative_kwargs,
                cfg_normalize=cfg_normalize,
                kwargs=extra_kwargs,
            )
        assert output is not None
        outputs.append(output)

    for rank_output in outputs[1:]:
        torch.testing.assert_close(
            rank_output,
            outputs[0],
            rtol=0,
            atol=0,
            msg="CFG parallel ranks produced different outputs in mock path",
        )
    return outputs[0]


def _run_multi_branch_cfg_parallel_mock(
    pipeline: MultiBranchTestPipeline,
    monkeypatch: pytest.MonkeyPatch,
    *,
    cfg_world_size: int,
    branches_kwargs: list[dict[str, Any]],
    true_cfg_scale: float | dict[str, float],
    cfg_normalize: bool,
) -> torch.Tensor:
    with torch.no_grad():
        branch_preds = [pipeline.predict_noise(**kwargs) for kwargs in branches_kwargs]
    slots_per_rank = _build_multi_branch_slot_table(
        n_branches=len(branches_kwargs),
        cfg_world_size=cfg_world_size,
        branch_preds=branch_preds,
    )

    outputs: list[torch.Tensor] = []
    for cfg_rank in range(cfg_world_size):
        fake_group = FakeMultiBranchCfgGroup(
            world_size=cfg_world_size,
            rank_in_group=cfg_rank,
            slots_per_rank=slots_per_rank,
        )
        _patch_cfg_state(monkeypatch, world_size=cfg_world_size, rank=cfg_rank, fake_group=fake_group)
        with torch.no_grad():
            output = pipeline.predict_noise_with_multi_branch_cfg(
                do_true_cfg=True,
                true_cfg_scale=true_cfg_scale,
                branches_kwargs=branches_kwargs,
                cfg_normalize=cfg_normalize,
            )
        assert output is not None
        outputs.append(output)

    for rank_output in outputs[1:]:
        torch.testing.assert_close(
            rank_output,
            outputs[0],
            rtol=0,
            atol=0,
            msg="Multi-branch CFG parallel ranks produced different outputs in mock path",
        )
    return outputs[0]


def _build_pipeline_on_device(
    pipeline_cls: type,
    *,
    hidden_dim: int,
    model_seed: int,
    dtype: torch.dtype,
    device: torch.device,
):
    pipeline = pipeline_cls(in_channels=4, hidden_dim=hidden_dim, seed=model_seed)
    pipeline.transformer = pipeline.transformer.to(device=device, dtype=dtype)
    pipeline.transformer.eval()
    return pipeline


# ---------------------------------------------------------------------------
# CPU: mixin logic via mocked CFG groups (no GPU / no torch.distributed spawn)
# ---------------------------------------------------------------------------


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("cfg_normalize", [False, True])
@pytest.mark.parametrize("extra_kwargs", [None, {"step_i": 1}])
def test_predict_noise_with_cfg(
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
    cfg_normalize: bool,
    extra_kwargs: dict[str, Any] | None,
):
    dtype = torch.float32
    device = torch.device("cpu")
    pipeline = _build_pipeline_on_device(TestCFGPipeline, hidden_dim=128, model_seed=42, dtype=dtype, device=device)
    positive_kwargs, negative_kwargs = _make_two_branch_inputs(
        batch_size=batch_size,
        channels=4,
        height=16,
        width=16,
        dtype=dtype,
        device=device,
        input_seed=123,
    )

    _patch_cfg_state(
        monkeypatch,
        world_size=1,
        rank=0,
        fake_group=FakeCfgGroup(world_size=1, rank_in_group=0, rank_tensors=[]),
    )
    with torch.no_grad():
        baseline = pipeline.predict_noise_maybe_with_cfg(
            do_true_cfg=True,
            true_cfg_scale=7.5,
            positive_kwargs=positive_kwargs,
            negative_kwargs=negative_kwargs,
            cfg_normalize=cfg_normalize,
            kwargs=extra_kwargs,
        )

    parallel = _run_two_branch_cfg_parallel_mock(
        pipeline,
        monkeypatch,
        cfg_world_size=2,
        positive_kwargs=positive_kwargs,
        negative_kwargs=negative_kwargs,
        true_cfg_scale=7.5,
        cfg_normalize=cfg_normalize,
        extra_kwargs=extra_kwargs,
    )
    assert baseline is not None
    _assert_cfg_outputs_close(
        parallel,
        baseline,
        dtype=dtype,
        msg=f"mock CFG parallel differs from sequential (cfg_normalize={cfg_normalize})",
    )


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
def test_predict_noise_without_cfg():
    dtype = torch.float32
    device = torch.device("cpu")
    pipeline = _build_pipeline_on_device(TestCFGPipeline, hidden_dim=128, model_seed=42, dtype=dtype, device=device)
    positive_kwargs, _ = _make_two_branch_inputs(
        batch_size=1,
        channels=4,
        height=16,
        width=16,
        dtype=dtype,
        device=device,
        input_seed=123,
    )

    with torch.no_grad():
        noise_pred = pipeline.predict_noise_maybe_with_cfg(
            do_true_cfg=False,
            true_cfg_scale=7.5,
            positive_kwargs=positive_kwargs,
            negative_kwargs=None,
            cfg_normalize=False,
        )

    assert noise_pred is not None
    assert noise_pred.shape == (1, 4, 16, 16)


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
@pytest.mark.parametrize(
    "cfg_parallel_size,n_branches",
    [(2, 2), (2, 3), (3, 3), (2, 4)],
)
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("cfg_normalize", [False, True])
def test_predict_noise_with_multi_branch_cfg(
    monkeypatch: pytest.MonkeyPatch,
    cfg_parallel_size: int,
    n_branches: int,
    batch_size: int,
    cfg_normalize: bool,
):
    dtype = torch.float32
    device = torch.device("cpu")
    pipeline = _build_pipeline_on_device(
        MultiBranchTestPipeline,
        hidden_dim=128,
        model_seed=42,
        dtype=dtype,
        device=device,
    )
    branches_kwargs = _make_multi_branch_inputs(
        n_branches=n_branches,
        batch_size=batch_size,
        channels=4,
        height=16,
        width=16,
        dtype=dtype,
        device=device,
        input_seed=123,
    )
    cfg_scale = _multi_branch_cfg_scale(n_branches)

    _patch_cfg_state(
        monkeypatch,
        world_size=1,
        rank=0,
        fake_group=FakeCfgGroup(world_size=1, rank_in_group=0, rank_tensors=[]),
    )
    with torch.no_grad():
        baseline = pipeline.predict_noise_with_multi_branch_cfg(
            do_true_cfg=True,
            true_cfg_scale=cfg_scale,
            branches_kwargs=branches_kwargs,
            cfg_normalize=cfg_normalize,
        )

    parallel = _run_multi_branch_cfg_parallel_mock(
        pipeline,
        monkeypatch,
        cfg_world_size=cfg_parallel_size,
        branches_kwargs=branches_kwargs,
        true_cfg_scale=cfg_scale,
        cfg_normalize=cfg_normalize,
    )
    assert baseline is not None
    _assert_cfg_outputs_close(
        parallel,
        baseline,
        dtype=dtype,
        msg=(
            f"mock multi-branch CFG parallel differs from sequential "
            f"(n_branches={n_branches}, cfg_parallel_size={cfg_parallel_size})"
        ),
    )


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
def test_multi_branch_without_cfg():
    dtype = torch.float32
    device = torch.device("cpu")
    pipeline = _build_pipeline_on_device(
        MultiBranchTestPipeline,
        hidden_dim=128,
        model_seed=42,
        dtype=dtype,
        device=device,
    )
    branches_kwargs = _make_multi_branch_inputs(
        n_branches=3,
        batch_size=1,
        channels=4,
        height=16,
        width=16,
        dtype=dtype,
        device=device,
        input_seed=123,
    )

    with torch.no_grad():
        noise_pred = pipeline.predict_noise_with_multi_branch_cfg(
            do_true_cfg=False,
            true_cfg_scale=5.0,
            branches_kwargs=branches_kwargs,
            cfg_normalize=False,
        )

    assert noise_pred is not None
    assert noise_pred.shape == (1, 4, 16, 16)


# ---------------------------------------------------------------------------
# GPU nightly: same-seed sequential vs real multi-GPU CFG parallel parity
# ---------------------------------------------------------------------------


def _test_cfg_parallel_worker(
    local_rank: int,
    world_size: int,
    cfg_parallel_size: int,
    dtype: torch.dtype,
    test_config: dict,
    result_queue: torch.multiprocessing.Queue,
):
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)
    _update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29502",
        }
    )

    init_distributed_environment()
    initialize_model_parallel(cfg_parallel_size=cfg_parallel_size)
    assert get_classifier_free_guidance_world_size() == cfg_parallel_size

    pipeline = _build_pipeline_on_device(
        TestCFGPipeline,
        hidden_dim=test_config["hidden_dim"],
        model_seed=test_config["model_seed"],
        dtype=dtype,
        device=device,
    )
    positive_kwargs, negative_kwargs = _make_two_branch_inputs(
        batch_size=test_config["batch_size"],
        channels=test_config["channels"],
        height=test_config["height"],
        width=test_config["width"],
        dtype=dtype,
        device=device,
        input_seed=test_config["input_seed"],
    )

    with torch.no_grad():
        noise_pred = pipeline.predict_noise_maybe_with_cfg(
            do_true_cfg=True,
            true_cfg_scale=test_config["cfg_scale"],
            positive_kwargs=positive_kwargs,
            negative_kwargs=negative_kwargs,
            cfg_normalize=test_config["cfg_normalize"],
            kwargs=test_config["kwargs"],
        )

    assert noise_pred is not None
    result_queue.put((get_classifier_free_guidance_rank(), noise_pred.cpu()))
    destroy_distributed_env()


def _test_cfg_sequential_worker(
    local_rank: int,
    world_size: int,
    dtype: torch.dtype,
    test_config: dict,
    result_queue: torch.multiprocessing.Queue,
):
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)
    _update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29503",
        }
    )

    init_distributed_environment()
    initialize_model_parallel(cfg_parallel_size=1)
    assert get_classifier_free_guidance_world_size() == 1

    pipeline = _build_pipeline_on_device(
        TestCFGPipeline,
        hidden_dim=test_config["hidden_dim"],
        model_seed=test_config["model_seed"],
        dtype=dtype,
        device=device,
    )
    positive_kwargs, negative_kwargs = _make_two_branch_inputs(
        batch_size=test_config["batch_size"],
        channels=test_config["channels"],
        height=test_config["height"],
        width=test_config["width"],
        dtype=dtype,
        device=device,
        input_seed=test_config["input_seed"],
    )

    with torch.no_grad():
        noise_pred = pipeline.predict_noise_maybe_with_cfg(
            do_true_cfg=True,
            true_cfg_scale=test_config["cfg_scale"],
            positive_kwargs=positive_kwargs,
            negative_kwargs=negative_kwargs,
            cfg_normalize=test_config["cfg_normalize"],
            kwargs=test_config["kwargs"],
        )

    assert noise_pred is not None
    result_queue.put(noise_pred.cpu())
    destroy_distributed_env()


def _test_multi_branch_parallel_worker(
    local_rank: int,
    world_size: int,
    cfg_parallel_size: int,
    dtype: torch.dtype,
    test_config: dict,
    result_queue: torch.multiprocessing.Queue,
):
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)
    _update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29504",
        }
    )

    init_distributed_environment()
    initialize_model_parallel(cfg_parallel_size=cfg_parallel_size)
    assert get_classifier_free_guidance_world_size() == cfg_parallel_size

    pipeline = _build_pipeline_on_device(
        MultiBranchTestPipeline,
        hidden_dim=test_config["hidden_dim"],
        model_seed=test_config["model_seed"],
        dtype=dtype,
        device=device,
    )
    branches_kwargs = _make_multi_branch_inputs(
        n_branches=test_config["n_branches"],
        batch_size=test_config["batch_size"],
        channels=test_config["channels"],
        height=test_config["height"],
        width=test_config["width"],
        dtype=dtype,
        device=device,
        input_seed=test_config["input_seed"],
    )

    with torch.no_grad():
        noise_pred = pipeline.predict_noise_with_multi_branch_cfg(
            do_true_cfg=True,
            true_cfg_scale=test_config["cfg_scale"],
            branches_kwargs=branches_kwargs,
            cfg_normalize=test_config["cfg_normalize"],
        )

    assert noise_pred is not None
    result_queue.put((get_classifier_free_guidance_rank(), noise_pred.cpu()))
    destroy_distributed_env()


def _test_multi_branch_sequential_worker(
    local_rank: int,
    world_size: int,
    dtype: torch.dtype,
    test_config: dict,
    result_queue: torch.multiprocessing.Queue,
):
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)
    _update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29505",
        }
    )

    init_distributed_environment()
    initialize_model_parallel(cfg_parallel_size=1)
    assert get_classifier_free_guidance_world_size() == 1

    pipeline = _build_pipeline_on_device(
        MultiBranchTestPipeline,
        hidden_dim=test_config["hidden_dim"],
        model_seed=test_config["model_seed"],
        dtype=dtype,
        device=device,
    )
    branches_kwargs = _make_multi_branch_inputs(
        n_branches=test_config["n_branches"],
        batch_size=test_config["batch_size"],
        channels=test_config["channels"],
        height=test_config["height"],
        width=test_config["width"],
        dtype=dtype,
        device=device,
        input_seed=test_config["input_seed"],
    )

    with torch.no_grad():
        noise_pred = pipeline.predict_noise_with_multi_branch_cfg(
            do_true_cfg=True,
            true_cfg_scale=test_config["cfg_scale"],
            branches_kwargs=branches_kwargs,
            cfg_normalize=test_config["cfg_normalize"],
        )

    assert noise_pred is not None
    result_queue.put(noise_pred.cpu())
    destroy_distributed_env()


def _run_two_branch_gpu_parity(
    *,
    cfg_parallel_size: int,
    dtype: torch.dtype,
    batch_size: int,
    cfg_normalize: bool,
    extra_kwargs: dict[str, Any] | None,
) -> None:
    test_config = {
        "batch_size": batch_size,
        "channels": 4,
        "height": 16,
        "width": 16,
        "hidden_dim": 128,
        "cfg_scale": 7.5,
        "cfg_normalize": cfg_normalize,
        "model_seed": 42,
        "input_seed": 123,
        "kwargs": extra_kwargs,
    }

    mp_context = torch.multiprocessing.get_context("spawn")
    manager = mp_context.Manager()
    baseline_queue = manager.Queue()
    cfg_parallel_queue = manager.Queue()

    torch.multiprocessing.spawn(
        _test_cfg_sequential_worker,
        args=(1, dtype, test_config, baseline_queue),
        nprocs=1,
    )
    torch.multiprocessing.spawn(
        _test_cfg_parallel_worker,
        args=(cfg_parallel_size, cfg_parallel_size, dtype, test_config, cfg_parallel_queue),
        nprocs=cfg_parallel_size,
    )

    baseline_output = baseline_queue.get()
    cfg_parallel_outputs = [cfg_parallel_queue.get() for _ in range(cfg_parallel_size)]
    cfg_parallel_outputs.sort(key=lambda item: item[0])
    cfg_parallel_output = cfg_parallel_outputs[0][1]

    for cfg_rank, rank_output in cfg_parallel_outputs[1:]:
        torch.testing.assert_close(
            rank_output,
            cfg_parallel_output,
            rtol=0,
            atol=0,
            msg=f"CFG parallel ranks differ (rank 0 vs rank {cfg_rank})",
        )

    _assert_cfg_outputs_close(
        cfg_parallel_output,
        baseline_output,
        dtype=dtype,
        msg=(
            f"GPU CFG parallel output differs from sequential CFG "
            f"(cfg_parallel_size={cfg_parallel_size}, cfg_normalize={cfg_normalize})"
        ),
    )


def _run_multi_branch_gpu_parity(
    *,
    cfg_parallel_size: int,
    n_branches: int,
    dtype: torch.dtype,
    batch_size: int,
    cfg_normalize: bool,
) -> None:
    test_config = {
        "batch_size": batch_size,
        "channels": 4,
        "height": 16,
        "width": 16,
        "hidden_dim": 128,
        "cfg_scale": _multi_branch_cfg_scale(n_branches),
        "cfg_normalize": cfg_normalize,
        "model_seed": 42,
        "input_seed": 123,
        "n_branches": n_branches,
    }

    mp_context = torch.multiprocessing.get_context("spawn")
    manager = mp_context.Manager()
    baseline_queue = manager.Queue()
    cfg_parallel_queue = manager.Queue()

    torch.multiprocessing.spawn(
        _test_multi_branch_sequential_worker,
        args=(1, dtype, test_config, baseline_queue),
        nprocs=1,
    )
    torch.multiprocessing.spawn(
        _test_multi_branch_parallel_worker,
        args=(cfg_parallel_size, cfg_parallel_size, dtype, test_config, cfg_parallel_queue),
        nprocs=cfg_parallel_size,
    )

    baseline_output = baseline_queue.get()
    cfg_parallel_outputs = [cfg_parallel_queue.get() for _ in range(cfg_parallel_size)]
    cfg_parallel_outputs.sort(key=lambda item: item[0])
    cfg_parallel_output = cfg_parallel_outputs[0][1]

    for cfg_rank, rank_output in cfg_parallel_outputs[1:]:
        torch.testing.assert_close(
            rank_output,
            cfg_parallel_output,
            rtol=0,
            atol=0,
            msg=f"Multi-branch CFG parallel ranks differ (rank 0 vs rank {cfg_rank})",
        )

    _assert_cfg_outputs_close(
        cfg_parallel_output,
        baseline_output,
        dtype=dtype,
        msg=(
            f"GPU multi-branch CFG parallel differs from sequential "
            f"(n_branches={n_branches}, cfg_parallel_size={cfg_parallel_size})"
        ),
    )


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.parallel
@hardware_test(res={"cuda": "L4"}, num_cards=2)
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("cfg_normalize", [False, True])
@pytest.mark.parametrize("extra_kwargs", [None, {"step_i": 1}])
def test_predict_noise_with_cfg_parity(
    batch_size: int,
    cfg_normalize: bool,
    extra_kwargs: dict[str, Any] | None,
):
    _run_two_branch_gpu_parity(
        cfg_parallel_size=2,
        dtype=torch.bfloat16,
        batch_size=batch_size,
        cfg_normalize=cfg_normalize,
        extra_kwargs=extra_kwargs,
    )


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.parallel
@pytest.mark.parametrize(
    "cfg_parallel_size,n_branches",
    [
        pytest.param(2, 2, marks=_L4_TWO_GPU),
        pytest.param(2, 3, marks=_L4_TWO_GPU),
        pytest.param(3, 3, marks=_L4_THREE_GPU),
        pytest.param(2, 4, marks=_L4_TWO_GPU),
    ],
)
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("cfg_normalize", [False, True])
def test_predict_noise_with_multi_branch_cfg_parity(
    cfg_parallel_size: int,
    n_branches: int,
    batch_size: int,
    cfg_normalize: bool,
):
    _run_multi_branch_gpu_parity(
        cfg_parallel_size=cfg_parallel_size,
        n_branches=n_branches,
        dtype=torch.bfloat16,
        batch_size=batch_size,
        cfg_normalize=cfg_normalize,
    )
