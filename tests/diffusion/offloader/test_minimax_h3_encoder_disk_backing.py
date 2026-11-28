# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MiniMax-H3 text encoder declares one-shot checkpoint-mmap backing.

The declaration is what turns the generic disk backing into behavior: the
resolver must resolve it to ``CHECKPOINT_MMAP`` for the single-rank setup the
feature targets, fail closed to ``PINNED`` when the configuration cannot stream
rank-local weights, and the loader must only hand the encoder to the plan when
its own preflight agrees.
"""

import pytest
from torch import nn

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.offloader.config import TEXT_ENCODER_COMPONENT
from vllm_omni.diffusion.offloader.module_collector import PipelineModules
from vllm_omni.diffusion.offloader.offload_plan import BackingKind, OffloadPlan

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]

CUDA = hardware_marks(res={"cuda": "H100"}, num_cards=1)

ENCODER_PATH = "text_encoder"


def _declaration() -> OffloadPlan:
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    return MiniMaxH3Pipeline._offload_plan


def _modules(encoder: nn.Module) -> PipelineModules:
    return PipelineModules(
        dits=[],
        dit_names=[],
        encoders=[encoder],
        encoder_names=[ENCODER_PATH],
        vaes=[],
        vae_names=[],
    )


def test_pipeline_declares_one_shot_checkpoint_mmap_encoder() -> None:
    plan = _declaration()

    assert TEXT_ENCODER_COMPONENT in plan.one_shot_component_paths
    assert plan.component_backing[TEXT_ENCODER_COMPONENT] is BackingKind.CHECKPOINT_MMAP
    assert TEXT_ENCODER_COMPONENT in plan.on_demand_component_paths


def _select(encoder: nn.Module, plan: OffloadPlan, **overrides) -> tuple[tuple[str, nn.Module], ...]:
    arguments = {
        "selected": True,
        "use_hsdp": False,
        "dp_size": 1,
        "uses_allgather": False,
        "dit_mmap_ready": True,
    }
    arguments.update(overrides)
    return DiffusersPipelineLoader._mmap_encoder_modules(_modules(encoder), plan, **arguments)


def test_loader_adds_the_encoder_for_single_rank_streaming() -> None:
    encoder = nn.Identity()

    selected = _select(encoder, _declaration())

    assert selected == ((ENCODER_PATH, encoder),)


def test_loader_fails_closed_when_hsdp_needs_the_ordinary_loader() -> None:
    assert _select(nn.Identity(), _declaration(), use_hsdp=True) == ()


def test_loader_fails_closed_for_multi_rank_allgather() -> None:
    assert _select(nn.Identity(), _declaration(), dp_size=2, uses_allgather=True) == ()


def test_loader_keeps_encoder_on_ordinary_loader_when_dit_allgather_disables_mmap() -> None:
    # The backend ties rank-local mmap to the DiT transport, so a multi-rank
    # AllGather DiT group cannot realize a planned rank-local encoder.
    assert (
        _select(
            nn.Identity(),
            _declaration(),
            dp_size=2,
            uses_allgather=False,
            dit_mmap_ready=False,
        )
        == ()
    )


def test_loader_ignores_encoders_the_user_did_not_select() -> None:
    assert _select(nn.Identity(), _declaration(), selected=False) == ()


def test_loader_ignores_encoders_without_a_disk_declaration() -> None:
    plan = OffloadPlan(encoder_component_types={ENCODER_PATH: TEXT_ENCODER_COMPONENT})

    assert _select(nn.Identity(), plan) == ()


def test_encoder_maps_checkpoint_names_onto_runtime_parameters() -> None:
    from vllm_omni.diffusion.models.minimax_h3.encoder import MiniMaxH3Qwen3VLEncoder

    encoder = MiniMaxH3Qwen3VLEncoder.__new__(MiniMaxH3Qwen3VLEncoder)
    expected = {
        "model.visual.blocks.0.attn.qkv.weight": ("vision.blocks.0.attn.qkv.weight", None),
        "model.language_model.embed_tokens.weight": ("text_model.embed_tokens.weight", None),
        "model.language_model.layers.0.input_layernorm.weight": (
            "text_model.layers.0.input_layernorm.weight",
            None,
        ),
        "model.language_model.layers.0.self_attn.q_proj.weight": (
            "text_model.layers.0.self_attn.qkv_proj.weight",
            "q",
        ),
        "model.language_model.layers.0.mlp.gate_proj.weight": (
            "text_model.layers.0.mlp.gate_up_proj.weight",
            0,
        ),
        "model.language_model.layers.0.mlp.up_proj.weight": (
            "text_model.layers.0.mlp.gate_up_proj.weight",
            1,
        ),
        # Not part of the streamed set.
        "lm_head.weight": None,
        "model.language_model.norm.weight": None,
    }

    for name, want in expected.items():
        assert encoder.map_checkpoint_weight_name(name) == want


def test_encoder_maps_qkv_slices_onto_one_runtime_tensor() -> None:
    from vllm_omni.diffusion.models.minimax_h3.encoder import MiniMaxH3Qwen3VLEncoder

    encoder = MiniMaxH3Qwen3VLEncoder.__new__(MiniMaxH3Qwen3VLEncoder)
    mapped = [
        encoder.map_checkpoint_weight_name(f"model.language_model.layers.3.self_attn.{part}_proj.weight")
        for part in ("q", "k", "v")
    ]

    assert {runtime_name for runtime_name, _ in mapped} == {"text_model.layers.3.self_attn.qkv_proj.weight"}
    assert [shard for _, shard in mapped] == ["q", "k", "v"]
