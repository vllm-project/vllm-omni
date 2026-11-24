# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PDD adapters: a Diffusers-layout LoRA plus per-interval output heads fused per denoise step."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.diffusion.models.minimax_h3 import pdd_lora
from vllm_omni.diffusion.models.minimax_h3.lora import (
    _TURBO_EXPECTED_RAW_TARGETS,
    _TURBO_TARGET_DIMS,
    parse_pika_turbo_filename,
)
from vllm_omni.diffusion.models.minimax_h3.pdd_lora import PDDSpec, load_minimax_h3_pdd_lora, pdd_block_plan
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from vllm_omni.lora.request import LoRARequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

GRID, BLOCK, RANK = 8, 2, 1
NAME = "MiniMax-H3-Ref2VA-Acc-4Step.safetensors"


def _adapter_tensors() -> dict[str, torch.Tensor]:
    """The released layout at rank 1, with a short grid so the heads stay small."""
    generator = torch.Generator().manual_seed(0)
    tensors = {}
    for target in sorted(_TURBO_EXPECTED_RAW_TARGETS | pdd_lora._BLOCK_ADALN_TARGETS):
        if target in pdd_lora._BLOCK_ADALN_TARGETS:
            input_dim, output_dim = pdd_lora._TURBO_TIME_EMBED_DIM, pdd_lora._BLOCK_ADALN_OUT
        else:
            input_dim, output_dim = _TURBO_TARGET_DIMS[next(s for s in _TURBO_TARGET_DIMS if target.endswith(s))]
        tensors[f"{target}.lora_down"] = torch.randn(RANK, input_dim, generator=generator)
        tensors[f"{target}.lora_up"] = torch.randn(output_dim, RANK, generator=generator)
    for head, out in (("proj_out", 6), ("audio_proj_out", 4)):
        tensors[f"{head}.weight"] = torch.randn(GRID, out, 5, generator=generator)
        tensors[f"{head}.bias"] = torch.randn(GRID, out, generator=generator)
    return tensors


@pytest.fixture(scope="module")
def adapter(tmp_path_factory):
    path = tmp_path_factory.mktemp("pdd") / NAME
    tensors = _adapter_tensors()
    metadata = {"pdd_num_steps": str(GRID), "pdd_block_size": str(BLOCK), "lora_rank": str(RANK), "lora_alpha": "1.0"}
    save_file(tensors, str(path), metadata=metadata)
    return path, tensors


def _load(path, partition="ref2va", **kwargs):
    return load_minimax_h3_pdd_lora(
        partition=partition,
        lora_request=LoRARequest("pdd", 1, str(path)),
        lora_path=path,
        dtype=torch.float32,
        video_shift=12.0,
        audio_shift=3.0,
        **kwargs,
    )


def test_block_plan_is_the_mean_velocity_over_each_block_of_the_shifted_grid():
    plan = pdd_block_plan(12.0, 32, 4)
    assert plan.shape == (8, 32)
    assert torch.allclose(plan.sum(dim=1), torch.ones(8, dtype=torch.float64))
    # Paper eq. 16: t_n = (u / s) / (1 + (1 / s - 1) u), u = n / N; a block's weights are its interval lengths.
    u = torch.linspace(0.0, 1.0, 33, dtype=torch.float64)
    grid = (u / 12.0) / (1.0 + (1.0 / 12.0 - 1.0) * u)
    lengths = grid.diff()
    assert torch.allclose(plan[2, 8:12], lengths[8:12] / lengths[8:12].sum())
    assert torch.count_nonzero(plan[2]) == 4
    # Block boundaries are the nodes of H3's own shifted schedule at eight steps, so the sampler is unchanged.
    sigma = torch.linspace(1.0, 0.0, 9, dtype=torch.float64)
    assert torch.allclose(1.0 - grid[::4], 12.0 * sigma / (1.0 + 11.0 * sigma))
    # Without a shift every interval is equally long.
    assert torch.allclose(pdd_block_plan(1.0, 8, 2)[0, :2], torch.tensor([0.5, 0.5], dtype=torch.float64))


def test_loader_binds_the_lora_and_fuses_one_head_per_step(adapter):
    path, tensors = adapter
    lora_model, helper, spec = _load(path)
    assert isinstance(spec, PDDSpec)
    assert (spec.task_family, spec.denoise_steps, spec.video_shift, spec.audio_shift) == (
        "ref2v",
        GRID // BLOCK,
        12.0,
        3.0,
    )
    assert spec.supported_tasks == frozenset({"ref2va"})
    assert helper.r == RANK
    # 52 blocks of six projections plus the 50 block AdaLN projections; the output norm is untouched.
    assert len(lora_model.loras) == 52 * 6 + 50
    assert "blocks.0.adaln_proj.linear" in lora_model.loras
    assert "final_layer.adaln_proj.linear" not in lora_model.loras
    # Diffusers stores the MLP's value half first; the native fc1 stores its gate first.
    up = tensors["transformer_blocks.3.ff.net.0.proj.lora_up"]
    value, gate = up.chunk(2, dim=0)
    packed = lora_model.loras["blocks.3.mlp.fc1"]
    assert torch.equal(torch.cat(packed.lora_b), torch.cat((gate, value)))

    for raw, native, shift in (("proj_out", "video_out", 12.0), ("audio_proj_out", "audio_out", 3.0)):
        plan = pdd_block_plan(shift, GRID, BLOCK)
        weight, bias = spec.pdd_heads[f"{native}.weight"], spec.pdd_heads[f"{native}.bias"]
        assert weight.shape == (GRID // BLOCK, *tensors[f"{raw}.weight"].shape[1:])
        for step in range(GRID // BLOCK):
            expected = sum(plan[step, n].item() * tensors[f"{raw}.weight"][n] for n in range(GRID))
            torch.testing.assert_close(weight[step], expected, rtol=1e-5, atol=1e-6)
            expected_bias = sum(plan[step, n].item() * tensors[f"{raw}.bias"][n] for n in range(GRID))
            torch.testing.assert_close(bias[step], expected_bias, rtol=1e-5, atol=1e-6)


def test_loader_leaves_other_artifacts_to_the_next_loader(tmp_path):
    other = tmp_path / "ordinary.safetensors"
    save_file({"x.lora_A.weight": torch.zeros(1, 1)}, str(other))
    assert _load(other) is None
    assert _load(tmp_path / "missing.safetensors") is None


@pytest.mark.parametrize("partition", ["fl2va", "combined"])
def test_loader_refuses_a_dit_the_adapter_was_not_distilled_from(adapter, partition):
    with pytest.raises(ValueError, match="distilled from the Ref2VA DiT.*--task-type ref2va"):
        _load(adapter[0], partition=partition)


def test_loader_refuses_modes_that_cannot_bind_heads_per_request(adapter):
    with pytest.raises(ValueError, match="PDD adapters do not support step execution"):
        _load(adapter[0], unsupported_mode="step execution")


def test_loader_rejects_an_incomplete_adapter(adapter, tmp_path):
    tensors = dict(adapter[1])
    del tensors["transformer_blocks.7.attn.to_q.lora_up"]
    broken = tmp_path / NAME
    save_file(
        tensors,
        str(broken),
        metadata={"pdd_num_steps": "8", "pdd_block_size": "2", "lora_rank": "1", "lora_alpha": "1"},
    )
    with pytest.raises(ValueError, match="Incomplete MiniMax-H3 PDD LoRA pairs"):
        _load(broken)


def test_pipeline_binds_adapter_heads_for_the_request_and_releases_them_after():
    calls: list[object] = []
    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    object.__setattr__(
        pipeline,
        "transformer",
        SimpleNamespace(final_layer=SimpleNamespace(pdd_steps=None, set_fused_pdd_heads=calls.append)),
    )
    heads = {"video_out.weight": torch.zeros(2, 1, 1)}
    spec = PDDSpec("a.safetensors", "ref2v", 2, 12.0, 3.0, 1, 1.0, pdd_heads=heads)
    pipeline._bind_adapter_heads(spec)
    pipeline._bind_adapter_heads(parse_pika_turbo_filename("pika_h3_flex_turbo_4step_ref2va.safetensors"))
    pipeline._bind_adapter_heads(None)
    assert calls == [heads, None, None]
    # A full PDD student owns its heads; nothing is bound over them.
    pipeline.transformer.final_layer.pdd_steps = 32
    pipeline._bind_adapter_heads(spec)
    assert len(calls) == 3


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("pika_h3_flex_turbo_8step_hd_ref2va.safetensors", ("ref2v", 8, 12.0)),
        ("pika_h3_flex_turbo_4step_ref2va_lite_r32.safetensors", ("ref2v", 4, 12.0)),
        ("pika_h3_flex_turbo_2step_draft_ref2va.safetensors", ("ref2v", 2, 12.0)),
        ("pika_h3_flex_turbo_4step_fl2va.safetensors", ("fl2v", 4, 6.0)),
    ],
)
def test_pika_filename_pins_steps_task_and_shift(name, expected):
    spec = parse_pika_turbo_filename(name)
    assert (spec.task_family, spec.denoise_steps, spec.video_shift) == expected
    assert spec.audio_shift == 3.0


@pytest.mark.parametrize(
    "name", ["pika_h3_flex.safetensors", "pika_h3_flex_turbo_4step_t2va.safetensors", "x.safetensors"]
)
def test_other_names_carry_no_pika_contract(name):
    assert parse_pika_turbo_filename(name) is None


def test_final_layer_takes_request_heads_and_gives_the_released_ones_back():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3FinalLayer

    layer = SimpleNamespace(pdd_steps=None, fused_pdd_heads=None, video_out=SimpleNamespace(weight=torch.zeros(1)))
    heads = {"video_out.weight": torch.ones(2, 1, 1, dtype=torch.bfloat16)}
    MiniMaxH3FinalLayer.set_fused_pdd_heads(layer, heads)
    assert layer.fused_pdd_heads["video_out.weight"].dtype == torch.float32
    MiniMaxH3FinalLayer.set_fused_pdd_heads(layer, None)
    assert layer.fused_pdd_heads is None
    layer.pdd_steps = 32
    with pytest.raises(ValueError, match="cannot replace the FastH3 pdd_steps heads"):
        MiniMaxH3FinalLayer.set_fused_pdd_heads(layer, heads)


def test_probing_skips_files_it_cannot_read(adapter, tmp_path):
    (tmp_path / "truncated.safetensors").write_bytes(b"\x10\x00\x00\x00\x00\x00\x00\x00{")
    assert _load(tmp_path) is None
    assert _load(tmp_path / "truncated.safetensors") is None


def _serving_pipeline(**od_config):
    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.partition = "ref2va"
    pipeline.od_config = SimpleNamespace(**{"additional_config": {}, "step_execution": False, **od_config})
    pipeline._turbo_lora_specs = {}
    pipeline._native_lora_specs = {}
    return pipeline


def _pipeline_load(pipeline, path):
    return pipeline._load_diffusion_lora_adapter(
        lora_request=LoRARequest("pdd", 1, str(path)), lora_path=path, dtype=torch.float32
    )


def test_pipeline_holds_a_pdd_adapter_to_its_contract_like_a_turbo_artifact(adapter):
    pipeline = _serving_pipeline()
    lora_model, _ = _pipeline_load(pipeline, adapter[0])
    spec = pipeline._turbo_lora_specs[1]
    assert isinstance(spec, PDDSpec) and (spec.video_shift, spec.audio_shift) == (12.0, 3.0)
    assert "blocks.0.adaln_proj.linear" in lora_model.loras


@pytest.mark.parametrize(
    ("od_config", "mode"),
    [
        ({"step_execution": True}, "step execution"),
        ({"lora_application": "merged"}, "--lora-application merged"),
        ({"cache_backend": "tea_cache"}, "--cache-backend tea_cache"),
    ],
)
def test_pipeline_refuses_a_pdd_adapter_where_its_heads_cannot_be_bound(adapter, od_config, mode):
    pipeline = _serving_pipeline(**od_config)
    with pytest.raises(ValueError, match=f"PDD adapters do not support {mode}"):
        _pipeline_load(pipeline, adapter[0])
    assert pipeline._turbo_lora_specs == {}


@pytest.mark.parametrize(
    ("request_fields", "message"),
    [
        ({"lora_scale": 0.5}, "requires lora scale=1.0"),
        ({"extra_args": {"sampler": "heun"}}, "requires sampler='euler'"),
        ({"extra_args": {"latent_refine": 0.5}}, "partial-schedule latent refinement"),
    ],
)
def test_requests_that_leave_the_fused_schedule_are_rejected(request_fields, message):
    from vllm_omni.errors import OmniClientError

    pipeline = _serving_pipeline()
    spec = PDDSpec(
        "a.safetensors", "ref2v", 8, 12.0, 3.0, 64, 64.0, pdd_heads={"video_out.weight": torch.zeros(8, 1, 1)}
    )
    pipeline._turbo_lora_specs = {1: spec}
    sampling = SimpleNamespace(
        **{"lora_request": LoRARequest("pdd", 1, "/models/pdd"), "lora_scale": 1.0, "extra_args": {}, **request_fields}
    )
    with pytest.raises(OmniClientError, match=message):
        pipeline._validate_pdd_sampling(sampling, spec)
    sampling = SimpleNamespace(lora_request=LoRARequest("pdd", 1, "/models/pdd"), lora_scale=1.0, extra_args={})
    pipeline._validate_pdd_sampling(sampling, spec)
