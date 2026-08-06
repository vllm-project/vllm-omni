# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_branch(layout: str):
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import (
        MiniMaxH3DenoiseBranch,
    )

    update_mask = {
        "t2va": [True, True],
        "fl2va": [False, True],
        "ref2va": [True, True],
    }[layout]
    audio_update_mask = {
        "t2va": [True, True],
        "fl2va": [True, True],
        "ref2va": [False, True],
    }[layout]
    packed = {
        "seq_len": 8,
        "text_pos": torch.tensor([0, 1]),
        "img_pos": torch.tensor([2, 3]),
        "audio_pos": torch.tensor([4, 5]),
        "update_mask": torch.tensor(update_mask),
        "audio_update_mask": torch.tensor(audio_update_mask),
        "img_position_ids": torch.zeros(8, 3, dtype=torch.long),
        "cu_seqlens": torch.tensor([0, 6, 8], dtype=torch.int32),
    }
    return MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.zeros(2, 4),
        token_tags=torch.zeros(8, dtype=torch.long),
        device=torch.device("cpu"),
    )


def _reference_full_vector_unique(
    branch,
    *,
    sigma_video: float,
    sigma_audio: float,
    imgvid_cond_timestep: float,
    audio_ref_cond_timestep: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    t_video = 1.0 - sigma_video
    t_audio = 1.0 - sigma_audio
    timesteps = torch.full((branch.seq_len,), t_video, dtype=torch.float32)
    timesteps[branch.img_pos[branch.update_mask]] = t_video
    timesteps[branch.img_pos[~branch.update_mask]] = max(t_video, imgvid_cond_timestep)
    timesteps[branch.audio_pos[branch.audio_update_mask]] = t_audio
    timesteps[branch.audio_pos[~branch.audio_update_mask]] = max(t_audio, audio_ref_cond_timestep)
    return torch.unique(timesteps, sorted=True, return_inverse=True)


@pytest.mark.parametrize("layout", ["t2va", "fl2va", "ref2va"])
def test_timestep_metadata_matches_full_vector_unique(layout: str):
    branch = _make_branch(layout)
    sigmas_video = [1.0, 0.5, 0.0]
    sigmas_audio = [1.0, 0.25, 0.0]

    cached = branch.prepare_timestep_metadata(
        sigmas_video=sigmas_video,
        sigmas_audio=sigmas_audio,
        imgvid_cond_noise_aug_for_inference=0.999,
        audio_cond_noise_aug_for_inference=1.0,
    )

    assert len(cached) == 2
    for step, metadata in enumerate(cached):
        expected_unique, expected_inverse = _reference_full_vector_unique(
            branch,
            sigma_video=sigmas_video[step],
            sigma_audio=sigmas_audio[step],
            imgvid_cond_timestep=0.999,
            audio_ref_cond_timestep=1.0,
        )
        torch.testing.assert_close(metadata.unique_timesteps.cpu(), expected_unique, rtol=0, atol=0)
        torch.testing.assert_close(metadata.inverse_indices.cpu(), expected_inverse, rtol=0, atol=0)


def test_adaln_schedule_key_uses_exact_float32_bits():
    from vllm_omni.diffusion.models.minimax_h3.adaln_schedule_cache import (
        minimax_h3_adaln_schedule_key,
    )

    branch = _make_branch("t2va")
    first = branch.prepare_timestep_metadata(
        sigmas_video=[1.0, 0.5, 0.0],
        sigmas_audio=[1.0, 0.25, 0.0],
        imgvid_cond_noise_aug_for_inference=0.999,
        audio_cond_noise_aug_for_inference=1.0,
    )
    same_after_float32_rounding = branch.prepare_timestep_metadata(
        sigmas_video=[1.0, 0.5 + 1e-12, 0.0],
        sigmas_audio=[1.0, 0.25 + 1e-12, 0.0],
        imgvid_cond_noise_aug_for_inference=0.999,
        audio_cond_noise_aug_for_inference=1.0,
    )
    changed = branch.prepare_timestep_metadata(
        sigmas_video=[1.0, 0.5, 0.0],
        sigmas_audio=[1.0, 0.25000003, 0.0],
        imgvid_cond_noise_aug_for_inference=0.999,
        audio_cond_noise_aug_for_inference=1.0,
    )

    first_key = minimax_h3_adaln_schedule_key(first)
    assert first_key == minimax_h3_adaln_schedule_key(same_after_float32_rounding)
    assert first_key != minimax_h3_adaln_schedule_key(changed)


class _FakeAdalnLinear(torch.nn.Module):
    def __init__(self, out_features: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(out_features * 4, dtype=torch.bfloat16).reshape(out_features, 4))
        self.bias = torch.nn.Parameter(torch.arange(out_features, dtype=torch.bfloat16))
        self.out_features = out_features
        self.calls = 0

    def forward(self, x):
        assert self.weight.numel() > 0
        assert self.bias.numel() > 0
        self.calls += 1
        rows = torch.arange(
            x.shape[0] * self.out_features,
            device=x.device,
            dtype=torch.float32,
        ).reshape(x.shape[0], self.out_features)
        rows = rows + x.float().sum(dim=-1, keepdim=True)
        return rows.to(torch.bfloat16), None


def _make_tiny_adaln_projection(*, expand_ratio: int = 2, modality_num: int = 3):
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3AdalnProj,
    )

    projection = object.__new__(MiniMaxH3AdalnProj)
    torch.nn.Module.__init__(projection)
    projection.expand_ratio = expand_ratio
    projection.modality_num = modality_num
    projection.hidden_size = 2
    projection.linear = _FakeAdalnLinear(out_features=expand_ratio * modality_num * projection.hidden_size)
    projection.register_buffer("_precomputed_table", None, persistent=False)
    projection._precomputed_step = None
    projection._offloaded_projection_state = None
    return projection


def test_adaln_projection_reuses_exact_precomputed_rows_without_linear_calls():
    projection = _make_tiny_adaln_projection()
    first_t_emb = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    second_t_emb = torch.tensor([[4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]])
    first_direct = projection.compute_flat(first_t_emb)
    second_direct = projection.compute_flat(second_t_emb)
    first_padded = torch.cat(
        [first_direct, first_direct.new_zeros(second_direct.shape[0] - first_direct.shape[0], first_direct.shape[1])]
    )
    table = torch.stack([first_padded, second_direct])
    cursor = torch.zeros((), dtype=torch.long)
    direct_calls = projection.linear.calls

    projection.install_precomputed(table, cursor)
    first_cached = projection(first_t_emb)
    cursor.fill_(1)
    second_cached = projection(second_t_emb)

    assert projection.linear.calls == direct_calls
    for actual, expected in zip(first_cached, first_direct.chunk(2, dim=-1)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(second_cached, second_direct.chunk(2, dim=-1)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_adaln_projection_offload_preserves_cache_and_clear_restores_weights():
    projection = _make_tiny_adaln_projection()
    t_emb = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    direct = projection.compute_flat(t_emb)
    expected_state = {name: tensor.clone() for name, tensor in projection.linear.state_dict().items()}
    expected_bytes = sum(tensor.numel() * tensor.element_size() for tensor in expected_state.values())
    projection.install_precomputed(direct.unsqueeze(0), torch.zeros((), dtype=torch.long))

    freed_bytes = projection.offload_projection()

    assert freed_bytes == expected_bytes
    assert all(parameter.numel() == 0 for parameter in projection.linear.parameters())
    cached = projection(t_emb)
    for actual, expected in zip(cached, direct.chunk(projection.expand_ratio, dim=-1)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    projection.clear_precomputed()
    assert not projection.projection_weights_offloaded
    for name, tensor in projection.linear.state_dict().items():
        torch.testing.assert_close(tensor, expected_state[name], rtol=0, atol=0)
    torch.testing.assert_close(projection.compute_flat(t_emb), direct, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for allocator accounting")
def test_adaln_projection_offload_releases_cuda_allocator_storage():
    device = torch.device("cuda")
    torch.accelerator.empty_cache()
    projection = _make_tiny_adaln_projection().to(device)
    t_emb = torch.tensor([[0.0, 1.0, 2.0, 3.0]], device=device)
    direct = projection.compute_flat(t_emb)
    projection.install_precomputed(
        direct.unsqueeze(0),
        torch.zeros((), dtype=torch.long, device=device),
    )
    torch.accelerator.synchronize()
    allocated_before = torch.cuda.memory_allocated(device)

    freed_bytes = projection.offload_projection()
    torch.accelerator.synchronize()
    allocated_after = torch.cuda.memory_allocated(device)

    assert freed_bytes > 0
    assert allocated_before - allocated_after >= freed_bytes
    assert all(parameter.is_cuda and parameter.numel() == 0 for parameter in projection.linear.parameters())

    projection.clear_precomputed()
    assert all(parameter.is_cuda and parameter.numel() > 0 for parameter in projection.linear.parameters())


class _CountingTimeEmbedder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, timesteps):
        self.calls += 1
        return timesteps[:, None]


class _TinyAdalnBlock(torch.nn.Module):
    def __init__(self, *, modality_num: int = 3):
        super().__init__()
        self.adaln_proj = _make_tiny_adaln_projection(modality_num=modality_num)


def _make_adaln_schedule_model():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3DiTModel,
    )

    model = object.__new__(MiniMaxH3DiTModel)
    torch.nn.Module.__init__(model)
    model.time_embedder = _CountingTimeEmbedder()
    model.blocks = torch.nn.ModuleList([_TinyAdalnBlock(), _TinyAdalnBlock()])
    model.final_layer = _TinyAdalnBlock(modality_num=1)
    model._adaln_projections = tuple(block.adaln_proj for block in model.blocks) + (model.final_layer.adaln_proj,)
    model._adaln_schedule_cache = None
    return model


def _schedule(branch, *, middle_video: float, middle_audio: float):
    return branch.prepare_timestep_metadata(
        sigmas_video=[1.0, middle_video, 0.0],
        sigmas_audio=[1.0, middle_audio, 0.0],
        imgvid_cond_noise_aug_for_inference=0.999,
        audio_cond_noise_aug_for_inference=1.0,
    )


def test_model_cache_hit_and_changed_schedule_rebuild_with_weight_offload():
    from vllm_omni.diffusion.models.minimax_h3.adaln_schedule_cache import (
        minimax_h3_adaln_schedule_key,
    )

    branch = _make_branch("t2va")
    first_plan = _schedule(branch, middle_video=0.5, middle_audio=0.25)
    changed_plan = _schedule(branch, middle_video=0.4, middle_audio=0.2)
    model = _make_adaln_schedule_model()
    expected_projection_bytes = sum(
        parameter.numel() * parameter.element_size()
        for projection in model._adaln_projections
        for parameter in projection.linear.parameters()
    )

    first = model.prepare_adaln_schedule_cache(
        unique_timestep_plan=tuple(step.unique_timesteps for step in first_plan),
        schedule_key=minimax_h3_adaln_schedule_key(first_plan),
        offload_weights=True,
    )
    calls_after_first = [projection.linear.calls for projection in model._adaln_projections]
    hit = model.prepare_adaln_schedule_cache(
        unique_timestep_plan=tuple(step.unique_timesteps for step in first_plan),
        schedule_key=minimax_h3_adaln_schedule_key(first_plan),
        offload_weights=True,
    )
    replaced = model.prepare_adaln_schedule_cache(
        unique_timestep_plan=tuple(step.unique_timesteps for step in changed_plan),
        schedule_key=minimax_h3_adaln_schedule_key(changed_plan),
        offload_weights=True,
    )

    assert first is hit
    assert replaced is not None and replaced is not first
    assert first is not None
    assert first.offloaded_projection_bytes == expected_projection_bytes
    assert first.net_memory_saved_bytes == expected_projection_bytes - first.table_bytes
    assert calls_after_first == [2, 2, 2]
    assert [projection.linear.calls for projection in model._adaln_projections] == [4, 4, 4]
    assert all(projection.projection_weights_offloaded for projection in model._adaln_projections)


def test_model_does_not_manually_offload_dtensor_owned_parameters(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer
    from vllm_omni.diffusion.models.minimax_h3.adaln_schedule_cache import (
        minimax_h3_adaln_schedule_key,
    )

    class FakeDTensor(torch.nn.Parameter):
        def to_local(self):
            return self

    monkeypatch.setattr(minimax_h3_transformer, "DTensor", FakeDTensor)
    branch = _make_branch("t2va")
    plan = _schedule(branch, middle_video=0.5, middle_audio=0.25)
    model = _make_adaln_schedule_model()
    projection = model._adaln_projections[0]
    projection.linear.weight = FakeDTensor(projection.linear.weight.detach())

    cache = model.prepare_adaln_schedule_cache(
        unique_timestep_plan=tuple(step.unique_timesteps for step in plan),
        schedule_key=minimax_h3_adaln_schedule_key(plan),
        offload_weights=True,
    )

    assert cache is not None
    assert cache.offloaded_projection_bytes == 0
    assert all(not item.projection_weights_offloaded for item in model._adaln_projections)
    assert projection.linear.weight.numel() > 0


def test_model_weight_offload_falls_back_when_tables_are_larger():
    model = _make_adaln_schedule_model()
    unique_timestep_plan = tuple(torch.tensor([float(step)]) for step in range(12))

    cache = model.prepare_adaln_schedule_cache(
        unique_timestep_plan=unique_timestep_plan,
        schedule_key=tuple((step,) for step in range(12)),
        offload_weights=True,
    )

    assert cache is None
    assert all(projection._precomputed_table is None for projection in model._adaln_projections)
    assert all(not projection.projection_weights_offloaded for projection in model._adaln_projections)


def test_denoise_loop_prepares_cache_once_and_passes_step_indices(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import denoise_loop

    monkeypatch.setenv("VLLM_OMNI_H3_ADALN_SCHEDULE_CACHE", "1")
    monkeypatch.setenv("VLLM_OMNI_H3_ADALN_OFFLOAD_WEIGHTS", "1")

    class FakeModel:
        def __init__(self):
            self.prepare_calls = 0
            self.prepare_for_request_calls = 0
            self.step_indices = []

        def prepare_for_request_caches(self):
            assert torch.is_inference_mode_enabled()
            self.prepare_for_request_calls += 1

        def prepare_adaln_schedule_cache(self, *, unique_timestep_plan, schedule_key, offload_weights):
            assert torch.is_inference_mode_enabled()
            assert len(unique_timestep_plan) == len(schedule_key) == 2
            assert offload_weights
            self.prepare_calls += 1
            return object()

        def __call__(self, **kwargs):
            self.step_indices.append(kwargs["adaln_step_index"])
            return torch.zeros(2, 96), torch.zeros(2, 32)

    model = FakeModel()
    denoise_loop.minimax_h3_denoise_loop(
        model=model,
        positive=_make_branch("t2va"),
        initial_video_rows=torch.zeros(2, 96),
        initial_audio_rows=torch.zeros(2, 32),
        keyframe_cond_rows=None,
        sigmas_video=[1.0, 0.5, 0.0],
        sigmas_audio=[1.0, 0.25, 0.0],
        device=torch.device("cpu"),
    )

    assert model.prepare_for_request_calls == 1
    assert model.prepare_calls == 1
    assert model.step_indices == [0, 1]


def test_weight_offload_env_requires_schedule_cache(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3.adaln_schedule_cache import (
        minimax_h3_adaln_weight_offload_enabled,
    )

    monkeypatch.delenv("VLLM_OMNI_H3_ADALN_SCHEDULE_CACHE", raising=False)
    monkeypatch.setenv("VLLM_OMNI_H3_ADALN_OFFLOAD_WEIGHTS", "1")
    assert not minimax_h3_adaln_weight_offload_enabled()
    monkeypatch.setenv("VLLM_OMNI_H3_ADALN_SCHEDULE_CACHE", "1")
    assert minimax_h3_adaln_weight_offload_enabled()
