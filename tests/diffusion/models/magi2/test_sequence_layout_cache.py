# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MAGI-2 reuses rope and modality metadata within a request, bit for bit."""

from __future__ import annotations

import pytest
import torch

from tests.diffusion.models.magi2.test_native_packing import (
    _longer_text_tensors,
    _sampler_tensors,
    _tiny_model,
    _tiny_sampler,
)
from vllm_omni.diffusion.models.magi2.layers import ModalityDispatcher
from vllm_omni.diffusion.models.magi2.modeling_magi2 import Magi2PreviewTransformer, Modality
from vllm_omni.diffusion.models.magi2.parallel import Magi2SequenceDispatcher, balanced_split_sizes
from vllm_omni.diffusion.models.magi2.sampler_magi2 import CFGConfig

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]

V, A, T, TIME = int(Modality.VIDEO), int(Modality.AUDIO), int(Modality.TEXT), int(Modality.TIME)
DISPATCHER_TENSORS = (
    "modality_mapping",
    "permute_mapping",
    "inv_permute_mapping",
    "permuted_modality_mapping",
    "group_size",
    "cu_group_sizes",
)


def _rebuilt_metadata(model: Magi2PreviewTransformer, coords_mapping: torch.Tensor, modality_mapping: torch.Tensor):
    """Rope and modality metadata computed from scratch, as every forward did without the cache."""
    rope = model.pre_adapter.rope(coords_mapping)
    time_mask = modality_mapping == int(Modality.TIME)
    modality_mapping = torch.where(time_mask, int(Modality.TEXT), modality_mapping)
    modality_dispatcher = ModalityDispatcher(modality_mapping, 3)
    video_indices = torch.nonzero(modality_mapping == int(Modality.VIDEO)).flatten()
    audio_indices = torch.nonzero(modality_mapping == int(Modality.AUDIO)).flatten()
    text_indices = torch.nonzero(modality_mapping == int(Modality.TEXT)).flatten()
    return rope, modality_dispatcher, video_indices, audio_indices, text_indices


def _rebuilding_forward(
    model: Magi2PreviewTransformer,
    x: torch.Tensor,
    coords_mapping: torch.Tensor,
    modality_mapping: torch.Tensor,
    varlen_handler,
    time_token_sequence: torch.Tensor | None = None,
) -> torch.Tensor:
    """``Magi2PreviewTransformer.forward`` with its metadata rebuilt on every call."""
    dispatcher = Magi2SequenceDispatcher()
    x = dispatcher.dispatch(x)
    coords_mapping = dispatcher.dispatch(coords_mapping)
    modality_mapping = dispatcher.dispatch(modality_mapping)
    if time_token_sequence is not None:
        time_token_sequence = dispatcher.dispatch(time_token_sequence)
    assert dispatcher.split_sizes is not None
    cp_split_sizes = dispatcher.split_sizes

    rope, modality_dispatcher, video_indices, audio_indices, text_indices = _rebuilt_metadata(
        model, coords_mapping, modality_mapping
    )
    hidden_states = model.pre_adapter(x, video_indices, audio_indices, text_indices)
    if time_token_sequence is not None and time_token_sequence.shape[-1] > 0:
        hidden_states[:, : time_token_sequence.shape[-1]] = time_token_sequence.to(hidden_states.dtype)
    hidden_states = model.block(hidden_states, rope, varlen_handler, modality_dispatcher, cp_split_sizes)
    output = model.post_adapter(hidden_states, video_indices, audio_indices)
    return dispatcher.undispatch(output)


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.int32) if tensor.dtype == torch.float32 else tensor


def _assert_same_metadata(actual, expected) -> None:
    rope, dispatcher, *indices = actual
    expected_rope, expected_dispatcher, *expected_indices = expected
    assert rope.dtype == expected_rope.dtype
    assert torch.equal(_bits(rope), _bits(expected_rope))
    for name in DISPATCHER_TENSORS:
        assert torch.equal(getattr(dispatcher, name), getattr(expected_dispatcher, name)), name
    assert dispatcher.group_size_cpu == expected_dispatcher.group_size_cpu
    assert dispatcher.num_modalities == expected_dispatcher.num_modalities
    for actual_indices, expected_indices_ in zip(indices, expected_indices, strict=True):
        assert torch.equal(actual_indices, expected_indices_)


def _count_rope_builds(model: Magi2PreviewTransformer) -> list[None]:
    builds: list[None] = []
    model.pre_adapter.rope.register_forward_hook(lambda *_: builds.append(None))
    return builds


def _coords(modalities: list[int], seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    tokens = len(modalities)
    xyz = torch.randint(-4, 5, (tokens, 3), generator=generator)
    sizes = torch.randint(2, 6, (tokens, 3), generator=generator)
    references = torch.randint(2, 6, (tokens, 3), generator=generator)
    return torch.cat((xyz, sizes, references), dim=-1).float()


def _layout_inputs(modalities: list[int], seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    return _coords(modalities, seed), torch.tensor(modalities, dtype=torch.int32)


def _model_args(sampler, tensors: dict[str, torch.Tensor], t: float):
    model_input = sampler.prepare_model_input(**tensors, t=torch.tensor([t]), cfg_config=CFGConfig())
    return sampler.data_proxy.process_input(model_input).model_args


def _transposed_video(seed: int) -> dict[str, torch.Tensor]:
    """Same token count and modality split, different video coordinates."""
    tensors = _sampler_tensors(seed)
    tensors["latent"] = tensors["latent"].transpose(3, 4).contiguous()
    return tensors


def _resplit(seed: int) -> dict[str, torch.Tensor]:
    """Same token count, different audio/text split."""
    generator = torch.Generator().manual_seed(seed + 200)
    tensors = _sampler_tensors(seed)
    tensors["audio_latent"] = torch.randn(1, 6, 4, generator=generator)
    tensors["txt_feat"] = torch.randn(1, 2, 4, generator=generator)
    tensors["null_txt_feat"] = torch.randn(1, 1, 4, generator=generator)
    return tensors


def _longer_audio(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed + 300)
    tensors = _sampler_tensors(seed)
    tensors["audio_latent"] = torch.randn(1, 7, 4, generator=generator)
    return tensors


@pytest.mark.cpu
def test_forward_reuses_metadata_within_a_request_and_matches_rebuilding_bitwise() -> None:
    model = _tiny_model()
    sampler = _tiny_sampler(model)
    builds = _count_rope_builds(model)
    # (inputs, timestep, whether the metadata must be built)
    calls = [
        (_sampler_tensors(0), 900.0, True),
        (_sampler_tensors(1), 450.0, False),
        (_transposed_video(2), 900.0, True),
        (_transposed_video(3), 450.0, False),
        (_sampler_tensors(4), 900.0, True),
        (_resplit(5), 900.0, True),
        (_longer_text_tensors(6), 900.0, True),
        (_resplit(7), 450.0, False),
        (_longer_text_tensors(8), 450.0, False),
        (_longer_audio(9), 900.0, True),
        (_resplit(10), 300.0, True),
        (_longer_audio(11), 450.0, False),
    ]

    for index, (tensors, t, must_build) in enumerate(calls):
        args = _model_args(sampler, tensors, t)
        with torch.inference_mode():
            before = len(builds)
            actual = model(*args)
            built = len(builds) - before
            expected = _rebuilding_forward(model, *args)
        assert built == int(must_build), index
        assert torch.equal(_bits(actual), _bits(expected)), index
        assert len(model._sequence_layouts) <= 2


@pytest.mark.cpu
@pytest.mark.parametrize("world_size", [1, 2, 3, 4, 11, 12])
def test_sequence_layout_matches_rebuilding_for_uneven_shards(world_size: int) -> None:
    model = _tiny_model()
    builds = _count_rope_builds(model)
    coords, modalities = _layout_inputs([V, V, V, A, A, T, T, TIME, V, A, T])
    start = 0
    for size in balanced_split_sizes(coords.shape[0], world_size):
        shard_coords = coords.narrow(0, start, size).contiguous()
        shard_modalities = modalities.narrow(0, start, size).contiguous()
        start += size
        with torch.inference_mode():
            expected = _rebuilt_metadata(model, shard_coords, shard_modalities)
            before = len(builds)
            first = model._sequence_layout(shard_coords, shard_modalities)
            second = model._sequence_layout(shard_coords.clone(), shard_modalities.clone())
        assert len(builds) - before == 1
        assert second is first
        _assert_same_metadata(first, expected)
        assert not bool((first.modality_dispatcher.modality_mapping == TIME).any())


@pytest.mark.cpu
@pytest.mark.parametrize(
    "change",
    ["modality", "coordinate", "signed_zero", "rope_bands", "coordinate_dtype", "in_place_input"],
)
def test_sequence_layout_rebuilds_when_an_input_changes_bitwise(change: str) -> None:
    model = _tiny_model()
    builds = _count_rope_builds(model)
    coords, modalities = _layout_inputs([V, V, A, T, TIME, V])
    coords[:, 0] = 0.0
    with torch.inference_mode():
        model._sequence_layout(coords, modalities)
    assert len(builds) == 1

    if change == "modality":
        modalities = modalities.clone()
        modalities[1] = A
    elif change == "coordinate":
        coords = coords.clone()
        coords[2, 1] += 1
    elif change == "signed_zero":
        # -0.0 == 0.0, but sin(-0.0) keeps the sign in the rope table.
        coords = coords.clone()
        coords[:, 0] = -0.0
    elif change == "rope_bands":
        with torch.no_grad():
            model.pre_adapter.rope.bands.mul_(2)
    elif change == "coordinate_dtype":
        coords = coords.double()
    else:
        # The cache keeps its own copy, so an in-place edit of the caller's tensor is seen.
        with torch.no_grad():
            coords[3, 2] += 1

    with torch.inference_mode():
        actual = model._sequence_layout(coords, modalities)
        expected = _rebuilt_metadata(model, coords, modalities)
    assert len(builds) == 3
    _assert_same_metadata(actual, expected)
    with torch.inference_mode():
        assert model._sequence_layout(coords, modalities) is actual
    assert len(builds) == 3


@pytest.mark.cpu
def test_sequence_layout_keeps_one_layout_per_cfg_branch() -> None:
    model = _tiny_model()
    builds = _count_rope_builds(model)
    branches = [_layout_inputs([V, A, T][: index % 3 + 1] * (index + 1), seed=index) for index in range(3)]
    with torch.inference_mode():
        for _ in range(3):
            for branch in branches[:2]:
                model._sequence_layout(*branch)
        assert len(builds) == 2
        model._sequence_layout(*branches[2])
        assert len(model._sequence_layouts) == 2
        model._sequence_layout(*branches[1])
        assert len(builds) == 3
        model._sequence_layout(*branches[0])
        assert len(builds) == 4


@pytest.mark.cpu
def test_sequence_layout_is_rebuilt_while_autograd_is_enabled() -> None:
    model = _tiny_model()
    builds = _count_rope_builds(model)
    coords, modalities = _layout_inputs([V, A, T])
    with torch.enable_grad():
        first = model._sequence_layout(coords, modalities)
        second = model._sequence_layout(coords, modalities)
    assert len(builds) == 2
    assert first is not second
    assert not model._sequence_layouts
    _assert_same_metadata(second, _rebuilt_metadata(model, coords, modalities))


@pytest.mark.cpu
def test_sequence_layout_keeps_inference_mode_entries_apart() -> None:
    model = _tiny_model()
    builds = _count_rope_builds(model)
    coords, modalities = _layout_inputs([V, A, T])
    with torch.inference_mode():
        inference = model._sequence_layout(coords, modalities)
    with torch.no_grad():
        no_grad = model._sequence_layout(coords, modalities)
        assert model._sequence_layout(coords, modalities) is no_grad
    with torch.inference_mode():
        assert model._sequence_layout(coords, modalities) is inference
    assert len(builds) == 2
    assert inference.rope.is_inference()
    assert not no_grad.rope.is_inference()


@pytest.mark.cpu
def test_invalid_coordinates_are_rejected_on_every_call() -> None:
    model = _tiny_model()
    coords, modalities = _layout_inputs([V, A, T])
    coords[1, 3] = 1.0  # size 1 with a reference other than 1
    for _ in range(2):
        with torch.inference_mode(), pytest.raises(ValueError, match="invalid MAGI coordinate scale"):
            model._sequence_layout(coords, modalities)
    assert not model._sequence_layouts


@pytest.mark.musa
def test_musa_device_sequence_layout_matches_rebuilding() -> None:
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("requires a MUSA device")
    model = _tiny_model().to("musa")
    builds = _count_rope_builds(model)
    modalities = [V] * 2048 + [A] * 250 + [T] * 429 + [TIME]
    coords, modality_mapping = (tensor.to("musa") for tensor in _layout_inputs(modalities))
    with torch.inference_mode():
        expected = _rebuilt_metadata(model, coords, modality_mapping)
        first = model._sequence_layout(coords, modality_mapping)
        second = model._sequence_layout(coords.clone(), modality_mapping.clone())
    assert len(builds) == 2
    assert second is first
    _assert_same_metadata(first, expected)
