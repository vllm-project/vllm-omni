# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Component-isolated planning and placement (RFC #10943)."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from vllm_omni.diffusion.model_loader.checkpoint_adapters.direct_mmap import (
    _MiniMaxH3DirectMmapAdapter,
    get_direct_mmap_adapter,
)
from vllm_omni.diffusion.model_loader.host_weight_plan import build_checkpoint_binding_plan
from vllm_omni.diffusion.offloader.component_utils import prepare_pipeline_components
from vllm_omni.diffusion.offloader.plan_resolver import ResolvedComponent, ResolvedOffloadPlan

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


class _FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(2, 2))

    def map_checkpoint_weight_name(self, name: str):
        return name, None


def test_encoder_only_plan_owns_only_the_encoder_source(tmp_path) -> None:
    save_file({"weight": torch.ones(2, 2)}, str(tmp_path / "model.safetensors"))
    encoder = _FakeEncoder()
    source = SimpleNamespace(
        model_or_path=str(tmp_path),
        subfolder=None,
        prefix="text_encoder.",
        revision=None,
        fall_back_to_pt=False,
    )

    result = build_checkpoint_binding_plan(
        encoder,
        dit_modules=(),
        sources=(source,),
        model_path=str(tmp_path),
        tensor_parallel_size=1,
        online_quantization=False,
        encoder_modules=(("text_encoder", encoder),),
    )

    assert result.fallback_reason is None
    assert result.plan is not None
    assert result.plan.planned_source_prefixes == frozenset({"text_encoder."})
    assert tuple(result.plan.bindings) == ("text_encoder.weight",)


def test_adapter_is_selected_from_a_pipeline_without_a_dit() -> None:
    from vllm_omni.diffusion.models.minimax_h3.encoder import MiniMaxH3Qwen3VLEncoder

    encoder = MiniMaxH3Qwen3VLEncoder.__new__(MiniMaxH3Qwen3VLEncoder)
    nn.Module.__init__(encoder)
    pipeline = nn.Module()
    pipeline.text_encoder = encoder

    adapter = get_direct_mmap_adapter(pipeline)

    assert isinstance(adapter, _MiniMaxH3DirectMmapAdapter)


class _PlacementProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.moved = False

    def to(self, *args, **kwargs):
        self.moved = True
        return self


def _resolved_unselected_dit(module: nn.Module) -> ResolvedOffloadPlan:
    return ResolvedOffloadPlan(
        dits=(ResolvedComponent(path="transformer", module=module, selected=False),),
        encoders=(),
        vaes=(),
        residents=(),
    )


def test_an_unselected_dit_is_placed_for_generation() -> None:
    dit = _PlacementProbe()

    prepare_pipeline_components(
        _resolved_unselected_dit(dit),
        device=torch.device("cpu"),
        staged_components=[],
        enable_encoder_blocks=lambda _component: False,
    )

    assert dit.moved
