# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU regression tests for Wan VACE TeaCache support."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

import pytest
import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from pytest_mock import MockerFixture
from torch import nn

from vllm_omni.diffusion.cache.teacache.backend import TeaCacheBackend
from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.extractors import (
    EXTRACTOR_REGISTRY,
    _build_vace_hint_cache_indicator,
    extract_wan_vace_context,
)
from vllm_omni.diffusion.cache.teacache.hook import TeaCacheHook, apply_teacache_hook
from vllm_omni.diffusion.cache.teacache.state import TeaCacheState
from vllm_omni.diffusion.data import DiffusionCacheConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.forward_context import (
    ForwardContext,
    get_forward_context,
    override_forward_context,
    set_forward_context_cfg_branch,
)

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cpu,
    pytest.mark.diffusion,
    pytest.mark.cache,
]


@dataclass
class _TransformerConfig:
    patch_size: tuple[int, int, int] = (1, 1, 1)


@dataclass
class _ParallelConfig:
    sequence_parallel_size: int = 1
    mask_sp_padding: bool = False


@dataclass
class _OmniDiffusionConfig:
    parallel_config: _ParallelConfig


@dataclass
class _ForwardContext:
    omni_diffusion_config: _OmniDiffusionConfig
    sp_original_seq_len: int | None = None
    sp_padding_size: int = 0
    denoise_step_idx: int | None = 0


@dataclass
class _SPDecisionGroup:
    global_distance: float
    local_distances: list[float]

    def all_reduce(self, distance: torch.Tensor, op: object) -> torch.Tensor:
        assert op == torch.distributed.ReduceOp.MAX
        self.local_distances.append(float(distance.item()))
        distance.fill_(self.global_distance)
        return distance


class _PatchEmbedding(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.cat((hidden_states, hidden_states * 2), dim=1)


class _ConditionEmbedder(nn.Module):
    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None,
        *,
        timestep_seq_len: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        del timestep_seq_len
        temb = timestep.float().reshape(-1, 1).repeat(1, 2)
        timestep_proj = temb.repeat(1, 6)
        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class _TimestepProjPrepare(nn.Module):
    def forward(self, timestep_proj: torch.Tensor, timestep_seq_len: int | None) -> torch.Tensor:
        if timestep_seq_len is None:
            return timestep_proj.unflatten(1, (6, -1))
        return timestep_proj.unflatten(2, (6, -1))


class _ShardPoint(nn.Module):
    def __init__(self, local_seq_len: int | None = None) -> None:
        super().__init__()
        self.local_seq_len = local_seq_len

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.local_seq_len is None:
            return hidden_states
        return hidden_states[:, : self.local_seq_len]


class _ScaleShiftNorm(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        return hidden_states * (1 + scale) + shift


class _MainBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.zeros(1, 6, 2))
        self.norm1 = _ScaleShiftNorm()
        self.calls = 0
        self.last_hidden_states_mask: torch.Tensor | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_proj: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        hidden_states_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        del encoder_hidden_states, timestep_proj, rotary_emb
        self.calls += 1
        self.last_hidden_states_mask = hidden_states_mask
        return hidden_states + 1


class _VACEBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.last_hidden_states_mask: torch.Tensor | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        control_hidden_states: torch.Tensor,
        timestep_proj: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        hidden_states_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del encoder_hidden_states, timestep_proj, rotary_emb
        self.calls += 1
        self.last_hidden_states_mask = hidden_states_mask
        control_hidden_states = control_hidden_states + hidden_states
        return control_hidden_states * 2, control_hidden_states


class _OutputScaleShiftPrepare(nn.Module):
    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros_like(temb), torch.zeros_like(temb)


class _OutputNorm(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        del scale, shift
        return hidden_states


class _OutputProjection(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states[..., :1]


class WanVACETransformer3DModel(nn.Module):
    """Small deterministic VACE-shaped transformer used by L1 tests."""

    def __init__(self, local_seq_len: int | None = None) -> None:
        super().__init__()
        self.config = _TransformerConfig()
        self.patch_embedding = _PatchEmbedding()
        self.condition_embedder = _ConditionEmbedder()
        self.timestep_proj_prepare = _TimestepProjPrepare()
        self._sp_shard_point = _ShardPoint(local_seq_len)
        self.vace_blocks = nn.ModuleList([_VACEBlock()])
        self.blocks = nn.ModuleList([_MainBlock()])
        self.vace_layers_mapping = {0: 0}
        self.output_scale_shift_prepare = _OutputScaleShiftPrepare()
        self.norm_out = _OutputNorm()
        self.proj_out = _OutputProjection()
        self._cached_rope_emb: tuple[torch.Tensor, torch.Tensor] | None = None
        self._cached_rope_resolution: tuple[int, int, int] | None = None

    @property
    def main_block(self) -> _MainBlock:
        return cast(_MainBlock, self.blocks[0])

    @property
    def vace_block(self) -> _VACEBlock:
        return cast(_VACEBlock, self.vace_blocks[0])

    def rope(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_length = hidden_states.shape[2] * hidden_states.shape[3] * hidden_states.shape[4]
        frequencies = torch.zeros(sequence_length, 4, device=hidden_states.device)
        return frequencies, frequencies

    def embed_vace_context(
        self,
        vace_context: torch.Tensor,
        seq_len: int,
        sp_size: int,
    ) -> torch.Tensor:
        control_hidden_states = vace_context.flatten(2).transpose(1, 2).repeat(1, 1, 2)
        if control_hidden_states.shape[1] < seq_len:
            padding = control_hidden_states.new_zeros(
                control_hidden_states.shape[0],
                seq_len - control_hidden_states.shape[1],
                control_hidden_states.shape[2],
            )
            control_hidden_states = torch.cat((control_hidden_states, padding), dim=1)
        if sp_size > 1:
            control_hidden_states = control_hidden_states[:, : seq_len // sp_size]
        return control_hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        return_dict: bool = True,
        attention_kwargs: dict[str, object] | None = None,
        vace_context: torch.Tensor | None = None,
        vace_context_scale: float | list[float] = 1.0,
    ) -> torch.Tensor:
        del (
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_hidden_states_image,
            return_dict,
            attention_kwargs,
            vace_context,
            vace_context_scale,
        )
        raise AssertionError("TeaCache hook should intercept the original forward")


# The class name intentionally matches the backend's custom-enabler dispatch key.
class Wan22VACEPipeline:
    def __init__(
        self,
        transformer: WanVACETransformer3DModel | None,
        transformer_2: WanVACETransformer3DModel | None,
    ) -> None:
        self.transformer = transformer
        self.transformer_2 = transformer_2


class _CFGBranchObserver(CFGParallelMixin):
    def __init__(self) -> None:
        self.observed_branches: list[str | None] = []

    def predict_noise(self, value: float) -> torch.Tensor:
        self.observed_branches.append(get_forward_context().cfg_branch)
        return torch.tensor([value])


def _forward_context(
    *,
    sp_size: int = 1,
    mask_sp_padding: bool = False,
    original_seq_len: int | None = None,
    padding_size: int = 0,
) -> _ForwardContext:
    return _ForwardContext(
        omni_diffusion_config=_OmniDiffusionConfig(
            parallel_config=_ParallelConfig(
                sequence_parallel_size=sp_size,
                mask_sp_padding=mask_sp_padding,
            )
        ),
        sp_original_seq_len=original_seq_len,
        sp_padding_size=padding_size,
    )


def _forward_inputs(width: int = 2) -> dict[str, torch.Tensor]:
    return {
        "hidden_states": torch.zeros(1, 1, 1, 1, width),
        "timestep": torch.zeros(1, dtype=torch.long),
        "encoder_hidden_states": torch.zeros(1, 1, 2),
        "vace_context": torch.zeros(1, 1, 1, 1, width),
    }


def test_wan_vace_extractor_and_coefficients_are_registered() -> None:
    assert EXTRACTOR_REGISTRY["WanVACETransformer3DModel"] is extract_wan_vace_context
    config = TeaCacheConfig(transformer_type="WanVACETransformer3DModel")
    assert config.coefficients == [
        -3.03318725e05,
        4.90537029e04,
        -2.65530556e03,
        5.87365115e01,
        -3.15583525e-01,
    ]


def test_vace_hint_indicator_has_fixed_spatial_bound() -> None:
    hints = [torch.ones(2, 100, 4), torch.full((2, 100, 4), 2.0)]

    indicator = _build_vace_hint_cache_indicator(hints, [1.0, 0.5])

    # Each layer retains 2*C channel statistics plus at most 2*64 token bins.
    assert indicator.shape == (2, 2, 136)
    assert indicator.dtype == torch.float32


def test_wan_vace_extractor_full_path_matches_reference_value(mocker: MockerFixture) -> None:
    model = WanVACETransformer3DModel()
    mocker.patch(
        "vllm_omni.diffusion.cache.teacache.extractors.get_forward_context",
        return_value=_forward_context(),
    )
    inputs = _forward_inputs()
    inputs["vace_context"] = torch.ones_like(inputs["vace_context"])

    context = extract_wan_vace_context(model, **inputs)
    output = context.postprocess(context.run_transformer_blocks()[0])

    # Fake-model reference: zero main tokens become one after the main block,
    # then receive a two-valued VACE hint, producing three at every output token.
    expected = torch.full_like(inputs["hidden_states"], 3)
    assert isinstance(output, Transformer2DModelOutput)
    torch.testing.assert_close(output.sample, expected)


def test_wan_vace_extractor_preserves_tuple_output_contract(mocker: MockerFixture) -> None:
    model = WanVACETransformer3DModel()
    mocker.patch(
        "vllm_omni.diffusion.cache.teacache.extractors.get_forward_context",
        return_value=_forward_context(),
    )
    inputs = _forward_inputs()
    inputs["vace_context"] = torch.ones_like(inputs["vace_context"])

    context = extract_wan_vace_context(model, **inputs, return_dict=False)
    output = context.postprocess(context.run_transformer_blocks()[0])

    assert isinstance(output, tuple)
    assert len(output) == 1
    torch.testing.assert_close(output[0], torch.full_like(inputs["hidden_states"], 3))


def test_cache_hit_recomputes_vace_and_hint_change_forces_main_blocks(mocker: MockerFixture) -> None:
    model = WanVACETransformer3DModel()
    mocker.patch(
        "vllm_omni.diffusion.cache.teacache.extractors.get_forward_context",
        return_value=_forward_context(),
    )
    apply_teacache_hook(
        model,
        TeaCacheConfig(
            transformer_type="WanVACETransformer3DModel",
            rel_l1_thresh=0.2,
            coefficients=[0.0, 0.0, 0.0, 1.0, 0.0],
        ),
    )

    inputs = _forward_inputs()
    first_output = model(**inputs).sample
    cached_output = model(**inputs).sample

    assert model.vace_block.calls == 2
    assert model.main_block.calls == 1
    torch.testing.assert_close(cached_output, first_output)

    inputs["vace_context"] = torch.ones_like(inputs["vace_context"])
    changed_output = model(**inputs).sample

    assert model.vace_block.calls == 3
    assert model.main_block.calls == 2
    assert not torch.equal(changed_output, cached_output)


def test_cfg_branches_keep_independent_vace_cache_state(mocker: MockerFixture) -> None:
    model = WanVACETransformer3DModel()
    model.do_true_cfg = True
    mocker.patch(
        "vllm_omni.diffusion.cache.teacache.extractors.get_forward_context",
        return_value=_forward_context(),
    )
    mocker.patch(
        "vllm_omni.diffusion.cache.teacache.hook.get_classifier_free_guidance_world_size",
        return_value=1,
    )
    apply_teacache_hook(
        model,
        TeaCacheConfig(
            transformer_type="WanVACETransformer3DModel",
            rel_l1_thresh=0.2,
            coefficients=[0.0, 0.0, 0.0, 1.0, 0.0],
        ),
    )

    inputs = _forward_inputs()
    for _ in range(2):
        model(**inputs)  # positive branch
        model(**inputs)  # negative branch

    assert model.vace_block.calls == 4
    assert model.main_block.calls == 2


def test_explicit_cfg_branch_context_survives_call_order_changes(mocker: MockerFixture) -> None:
    model = WanVACETransformer3DModel()
    mocker.patch(
        "vllm_omni.diffusion.cache.teacache.extractors.get_forward_context",
        return_value=_forward_context(),
    )
    mocker.patch(
        "vllm_omni.diffusion.cache.teacache.hook.get_classifier_free_guidance_world_size",
        return_value=1,
    )
    apply_teacache_hook(
        model,
        TeaCacheConfig(
            transformer_type="WanVACETransformer3DModel",
            rel_l1_thresh=0.2,
            coefficients=[0.0, 0.0, 0.0, 1.0, 0.0],
        ),
    )

    negative_inputs = _forward_inputs()
    positive_inputs = _forward_inputs()
    positive_inputs["vace_context"] = torch.ones_like(positive_inputs["vace_context"])

    # The first step deliberately visits negative before positive. The second
    # reverses that order; explicit branch identity must still select the right
    # residual instead of alternating by a global call counter.
    with override_forward_context(ForwardContext()):
        with set_forward_context_cfg_branch("negative"):
            negative_full = model(**negative_inputs).sample
        with set_forward_context_cfg_branch("positive"):
            positive_full = model(**positive_inputs).sample
        with set_forward_context_cfg_branch("positive"):
            positive_cached = model(**positive_inputs).sample
        with set_forward_context_cfg_branch("negative"):
            negative_cached = model(**negative_inputs).sample

    assert model.main_block.calls == 2
    torch.testing.assert_close(positive_cached, positive_full)
    torch.testing.assert_close(negative_cached, negative_full)
    assert not torch.equal(positive_cached, negative_cached)


def test_cfg_mixin_publishes_and_restores_sequential_branch_context(mocker: MockerFixture) -> None:
    pipeline = _CFGBranchObserver()
    context = ForwardContext()
    mocker.patch(
        "vllm_omni.diffusion.distributed.cfg_parallel.get_classifier_free_guidance_world_size",
        return_value=1,
    )

    with override_forward_context(context):
        pipeline.predict_noise_maybe_with_cfg(
            do_true_cfg=True,
            true_cfg_scale=2.0,
            positive_kwargs={"value": 3.0},
            negative_kwargs={"value": 1.0},
            cfg_normalize=False,
        )
        assert context.cfg_branch is None

    assert pipeline.observed_branches == ["positive", "negative"]


def test_unknown_denoise_step_is_full_compute_without_cache_mutation(mocker: MockerFixture) -> None:
    model = WanVACETransformer3DModel()
    mocker.patch(
        "vllm_omni.diffusion.cache.teacache.extractors.get_forward_context",
        return_value=replace(_forward_context(), denoise_step_idx=None),
    )
    apply_teacache_hook(
        model,
        TeaCacheConfig(
            transformer_type="WanVACETransformer3DModel",
            rel_l1_thresh=0.2,
            coefficients=[0.0, 0.0, 0.0, 1.0, 0.0],
        ),
    )

    inputs = _forward_inputs()
    first_output = model(**inputs).sample
    second_output = model(**inputs).sample
    hook = model._hook_registry.get_hook(TeaCacheHook._HOOK_NAME)

    assert isinstance(hook, TeaCacheHook)
    assert model.vace_block.calls == 2
    assert model.main_block.calls == 2
    assert hook.state_manager.get_state().cnt == 0
    assert hook._forward_cnt == 0
    torch.testing.assert_close(second_output, first_output)


def test_spatially_rearranged_vace_hints_force_main_blocks(mocker: MockerFixture) -> None:
    model = WanVACETransformer3DModel()
    mocker.patch(
        "vllm_omni.diffusion.cache.teacache.extractors.get_forward_context",
        return_value=_forward_context(),
    )
    apply_teacache_hook(
        model,
        TeaCacheConfig(
            transformer_type="WanVACETransformer3DModel",
            rel_l1_thresh=0.2,
            coefficients=[0.0, 0.0, 0.0, 1.0, 0.0],
        ),
    )

    inputs = _forward_inputs()
    inputs["vace_context"] = torch.tensor([[[[[0.0, 1.0]]]]])
    first_output = model(**inputs).sample

    # Channel-wise signed and absolute means are unchanged by this permutation;
    # the spatial component of the hint indicator must still detect it.
    inputs["vace_context"] = torch.tensor([[[[[1.0, 0.0]]]]])
    rearranged_output = model(**inputs).sample

    assert model.vace_block.calls == 2
    assert model.main_block.calls == 2
    assert not torch.equal(rearranged_output, first_output)


def test_vace_scale_change_forces_main_blocks(mocker: MockerFixture) -> None:
    model = WanVACETransformer3DModel()
    mocker.patch(
        "vllm_omni.diffusion.cache.teacache.extractors.get_forward_context",
        return_value=_forward_context(),
    )
    apply_teacache_hook(
        model,
        TeaCacheConfig(
            transformer_type="WanVACETransformer3DModel",
            rel_l1_thresh=0.2,
            coefficients=[0.0, 0.0, 0.0, 1.0, 0.0],
        ),
    )

    inputs = _forward_inputs()
    inputs["vace_context"] = torch.ones_like(inputs["vace_context"])

    full_scale_output = model(**inputs, vace_context_scale=1.0).sample
    reduced_scale_output = model(**inputs, vace_context_scale=0.5).sample

    assert model.vace_block.calls == 2
    assert model.main_block.calls == 2
    assert not torch.equal(reduced_scale_output, full_scale_output)

    cached_output = model(**inputs, vace_context_scale=0.5).sample
    assert model.vace_block.calls == 3
    assert model.main_block.calls == 2
    torch.testing.assert_close(cached_output, reduced_scale_output)


def test_vace_scale_count_must_match_hint_count(mocker: MockerFixture) -> None:
    model = WanVACETransformer3DModel()
    mocker.patch(
        "vllm_omni.diffusion.cache.teacache.extractors.get_forward_context",
        return_value=_forward_context(),
    )

    with pytest.raises(ValueError, match="one value per VACE hint"):
        extract_wan_vace_context(
            model,
            **_forward_inputs(),
            vace_context_scale=[],
        )


def test_vace_presence_change_forces_main_blocks(mocker: MockerFixture) -> None:
    model = WanVACETransformer3DModel()
    mocker.patch(
        "vllm_omni.diffusion.cache.teacache.extractors.get_forward_context",
        return_value=_forward_context(),
    )
    apply_teacache_hook(
        model,
        TeaCacheConfig(
            transformer_type="WanVACETransformer3DModel",
            rel_l1_thresh=0.2,
            coefficients=[0.0, 0.0, 0.0, 1.0, 0.0],
        ),
    )

    inputs = _forward_inputs()
    vace_context = inputs.pop("vace_context")

    model(**inputs, vace_context=None)
    model(**inputs, vace_context=vace_context)

    assert model.vace_block.calls == 1
    assert model.main_block.calls == 2


def test_wan_vace_extractor_keeps_sp_local_states_and_padding_mask(mocker: MockerFixture) -> None:
    model = WanVACETransformer3DModel(local_seq_len=2)
    mocker.patch(
        "vllm_omni.diffusion.cache.teacache.extractors.get_forward_context",
        return_value=_forward_context(
            sp_size=2,
            mask_sp_padding=True,
            original_seq_len=3,
            padding_size=1,
        ),
    )

    context = extract_wan_vace_context(model, **_forward_inputs(width=4))
    outputs = context.run_transformer_blocks()

    assert context.hidden_states.shape == (1, 2, 2)
    assert context.modulated_input.shape == context.hidden_states.shape
    assert outputs[0].shape == context.hidden_states.shape
    assert context.synchronize_cache_decision
    expected_mask = torch.tensor([[True, True, True, False]])
    torch.testing.assert_close(model.vace_block.last_hidden_states_mask, expected_mask)
    torch.testing.assert_close(model.main_block.last_hidden_states_mask, expected_mask)


def test_sp_cache_decision_uses_group_max_distance(mocker: MockerFixture) -> None:
    group = _SPDecisionGroup(global_distance=0.25, local_distances=[])
    mocker.patch(
        "vllm_omni.diffusion.cache.teacache.hook.get_sp_group",
        return_value=group,
    )
    hook = TeaCacheHook(
        TeaCacheConfig(
            transformer_type="WanVACETransformer3DModel",
            rel_l1_thresh=0.2,
            coefficients=[0.0, 0.0, 0.0, 1.0, 0.0],
        )
    )
    state = TeaCacheState()
    state.cnt = 1
    state.previous_modulated_input = torch.ones(1, 2, 2)

    should_compute = hook._should_compute_full_transformer(
        state,
        torch.ones(1, 2, 2),
        synchronize_cache_decision=True,
    )

    # This rank sees identical local inputs, but another SP rank reports a
    # threshold-crossing distance, so every rank must recompute main blocks.
    assert group.local_distances == [0.0]
    assert should_compute


def test_dual_vace_transformers_have_independent_state_and_refresh() -> None:
    high_noise_transformer = WanVACETransformer3DModel()
    low_noise_transformer = WanVACETransformer3DModel()
    pipeline = Wan22VACEPipeline(high_noise_transformer, low_noise_transformer)
    backend = TeaCacheBackend(DiffusionCacheConfig(coefficients=[0.0, 0.0, 0.0, 1.0, 0.0]))

    backend.enable(pipeline)

    high_hook = high_noise_transformer._hook_registry.get_hook(TeaCacheHook._HOOK_NAME)
    low_hook = low_noise_transformer._hook_registry.get_hook(TeaCacheHook._HOOK_NAME)
    assert isinstance(high_hook, TeaCacheHook)
    assert isinstance(low_hook, TeaCacheHook)
    assert high_hook is not low_hook
    assert high_hook.state_manager is not low_hook.state_manager

    high_hook.state_manager.get_state().cnt = 3
    low_hook.state_manager.get_state().cnt = 7
    high_hook._forward_cnt = 3
    low_hook._forward_cnt = 7

    backend.refresh(pipeline, num_inference_steps=20, verbose=False)

    assert high_hook.state_manager.get_state().cnt == 0
    assert low_hook.state_manager.get_state().cnt == 0
    assert high_hook._forward_cnt == 0
    assert low_hook._forward_cnt == 0


@pytest.mark.parametrize("loaded_expert", ["transformer", "transformer_2"])
def test_single_loaded_vace_expert_is_supported(loaded_expert: str) -> None:
    transformer = WanVACETransformer3DModel()
    pipeline = Wan22VACEPipeline(
        transformer=transformer if loaded_expert == "transformer" else None,
        transformer_2=transformer if loaded_expert == "transformer_2" else None,
    )
    backend = TeaCacheBackend(DiffusionCacheConfig(coefficients=[0.0, 0.0, 0.0, 1.0, 0.0]))

    backend.enable(pipeline)

    hook = transformer._hook_registry.get_hook(TeaCacheHook._HOOK_NAME)
    assert isinstance(hook, TeaCacheHook)
