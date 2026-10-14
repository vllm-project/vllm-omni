# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import dataclasses
import gc
import hashlib
import inspect
import math
import weakref

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from vllm_omni.diffusion.models.magi2.configuration_magi2 import Magi2RefinerConfig
from vllm_omni.diffusion.models.magi2.modeling_magi2_refiner import Magi2RefinerTransformer
from vllm_omni.diffusion.models.magi2.refiner_data_proxy import Magi2RefinerDataProxy
from vllm_omni.diffusion.models.magi2.window_attention import build_reference_window_ranges

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]

_PRODUCTION_GRID = (63, 68, 120)
# sha256[:16] of the int32 token_order and token_restore_order of SandAI's
# Magi2RefinerDataProxy._build_block_order at the production grid with 512 text tokens.
_TOKEN_ORDER_SHA16 = "9553ee706fd89a31"
_TOKEN_RESTORE_SHA16 = "a832fe5bba68b1bb"
_VIDEO, _AUDIO, _TEXT = 0, 1, 2


def _sha16(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()[:16]


def _latent(grid: tuple[int, int, int], channels: int = 48, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, channels, *grid, generator=generator)


def _text(length: int, channels: int = 5120, seed: int = 1, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, length, channels, generator=generator).to(dtype)


# Transcription of SandAI's refiner_data_proxy.get_coords.
def _ref_get_coords(shape, ref_feat_shape, offset_thw=(0, 0, 0), dtype=torch.float32):
    ori_t, ori_h, ori_w = shape
    ref_t, ref_h, ref_w = ref_feat_shape
    offset_t, offset_h, offset_w = offset_thw
    time_rng = torch.arange(ori_t, dtype=dtype) + offset_t
    height_rng = torch.arange(ori_h, dtype=dtype) + offset_h
    width_rng = torch.arange(ori_w, dtype=dtype) + offset_w
    time_grid, height_grid, width_grid = torch.meshgrid(time_rng, height_rng, width_rng, indexing="ij")
    coords_flat = torch.stack([time_grid, height_grid, width_grid], dim=-1).reshape(-1, 3)
    meta = torch.tensor([ori_t, ori_h, ori_w, ref_t, ref_h, ref_w], dtype=dtype)
    return torch.cat([coords_flat, meta.expand(coords_flat.size(0), -1)], dim=-1)


def _ref_packed_sequence(latent: torch.Tensor, text: torch.Tensor):
    """SandAI's block-ordered ``(x, coords_mapping, modality_mapping)`` for one sample."""

    _, _, t, h, w = latent.shape
    text_len = text.shape[0]
    video_tokens = rearrange(latent, "1 C T H W -> (T H W) C")
    width = max(video_tokens.shape[-1], text.shape[-1])
    x = torch.cat([F.pad(video_tokens, (0, width - video_tokens.shape[-1])), F.pad(text, (0, width - text.shape[-1]))])
    coords = torch.cat(
        [
            _ref_get_coords((t, h, w), (t, h, w)),
            _ref_get_coords((text_len, 1, 1), (1, 1, 1), offset_thw=(-text_len, 0, 0)),
        ]
    )
    modality = torch.cat([torch.full((t * h * w,), _VIDEO), torch.full((text_len,), _TEXT)])
    token_order = build_reference_window_ranges((t, h, w), text_len).token_order.long()
    return x[token_order], coords[token_order], modality[token_order]


def test_block_order_reproduces_the_reference_token_order() -> None:
    proxy = Magi2RefinerDataProxy()
    order, restore = proxy.block_order(_PRODUCTION_GRID)
    num_video, text_len = math.prod(_PRODUCTION_GRID), 512
    tail = torch.arange(num_video, num_video + text_len)

    assert order.dtype == restore.dtype == torch.int64
    assert _sha16(torch.cat((order, tail)).to(torch.int32)) == _TOKEN_ORDER_SHA16
    assert _sha16(torch.cat((restore, tail)).to(torch.int32)) == _TOKEN_RESTORE_SHA16
    assert torch.equal(order[restore], torch.arange(num_video))
    assert torch.equal(restore[order], torch.arange(num_video))


@pytest.mark.parametrize("grid", [(27, 8, 24), (23, 16, 40), (9, 4, 8)])
@pytest.mark.parametrize("text_len", [37, 0])
def test_pack_then_unpack_is_the_identity(grid: tuple[int, int, int], text_len: int) -> None:
    proxy = Magi2RefinerDataProxy()
    latent = _latent(grid)
    packed = proxy.pack(latent, _text(text_len))
    order, restore = proxy.block_order(grid)
    raster = rearrange(latent, "1 C T H W -> (T H W) C")

    assert packed.video.dtype == torch.float32
    assert torch.equal(packed.video, raster[order])
    assert torch.equal(packed.video.index_select(0, restore), raster)
    assert torch.equal(proxy.unpack(packed.video, grid), latent)


@pytest.mark.parametrize(("grid", "text_len"), [((23, 16, 40), 37), ((9, 4, 8), 1), ((9, 4, 8), 0)])
def test_pack_matches_the_reference_sequence(grid: tuple[int, int, int], text_len: int) -> None:
    proxy = Magi2RefinerDataProxy()
    latent = _latent(grid)
    # The Qwen text features arrive in bf16; the refiner sees them upcast to fp32.
    text = _text(text_len, dtype=torch.bfloat16)
    packed = proxy.pack(latent, text)
    ref_x, ref_coords, ref_modality = _ref_packed_sequence(latent, text[0].float())
    num_video = math.prod(grid)

    assert packed.coords.dtype == torch.float32 and packed.modality.dtype == torch.int64
    assert torch.equal(packed.coords, ref_coords)
    assert torch.equal(packed.modality, ref_modality)
    assert torch.equal(packed.video, ref_x[:num_video, :48])
    assert torch.equal(packed.text, ref_x[num_video:, :5120])
    assert packed.text.dtype == torch.float32
    assert packed.window_plan.grid == grid and packed.window_plan.text_len == text_len
    assert packed.split_sizes == (num_video + text_len,) and packed.rank == 0
    # Text positions count up to -1; the video uses its own size as the reference size.
    if text_len:
        assert packed.coords[num_video:, 0].tolist() == list(range(-text_len, 0))
        assert torch.equal(packed.coords[num_video:, 3:], torch.tensor([text_len, 1, 1, 1, 1, 1.0]).expand(text_len, 6))
    assert torch.equal(
        packed.coords[:num_video, 3:], torch.tensor([*grid, *grid], dtype=torch.float32).expand(num_video, 6)
    )


def test_pack_accepts_two_dimensional_and_empty_text() -> None:
    proxy = Magi2RefinerDataProxy()
    latent = _latent((9, 4, 8))
    text = _text(5)

    assert torch.equal(proxy.pack(latent, text[0]).text, proxy.pack(latent, text).text)
    empty = proxy.pack(latent, text[:, :0])
    assert empty.text.shape == (0, 5120)
    assert empty.coords.shape == (288, 9)
    assert empty.window_plan.text_len == 0
    assert (empty.modality == _VIDEO).all()


def test_pack_reuses_the_window_plan_per_grid_and_text_length() -> None:
    proxy = Magi2RefinerDataProxy()
    latent = _latent((9, 4, 8))

    first = proxy.pack(latent, _text(7))
    assert proxy.pack(latent * 2, _text(7, seed=3)).window_plan is first.window_plan
    other = proxy.pack(latent, _text(8))
    assert other.window_plan is not first.window_plan
    assert other.window_plan.text_len == 8


def test_window_plan_is_built_once_and_reused_by_pack() -> None:
    grid = (9, 4, 8)
    proxy = Magi2RefinerDataProxy()
    with torch.inference_mode():
        plan = proxy.window_plan(grid, 7)

    assert plan.grid == grid and plan.text_len == 7
    assert not plan.page_table.is_inference()
    assert proxy.window_plan(list(grid), 7, torch.device("cpu")) is plan
    assert proxy.pack(_latent(grid), _text(7)).window_plan is plan
    empty = proxy.window_plan(grid, 0)
    assert empty is not plan and empty.text_len == 0
    assert proxy.pack(_latent(grid), _text(0)).window_plan is empty


def test_caches_belong_to_the_proxy_and_hold_at_most_eight_plans() -> None:
    grid = (9, 4, 8)
    latent = _latent(grid)
    proxy = Magi2RefinerDataProxy()
    plans = [proxy.pack(latent, _text(text_len)).window_plan for text_len in range(9)]

    other = Magi2RefinerDataProxy()
    assert other.pack(latent, _text(8)).window_plan is not plans[8]
    assert other.block_order(grid)[0] is not proxy.block_order(grid)[0]
    # The least recently used plan is evicted first.
    assert proxy.pack(latent, _text(8)).window_plan is plans[8]
    assert proxy.pack(latent, _text(1)).window_plan is plans[1]
    assert proxy.pack(latent, _text(0)).window_plan is not plans[0]
    # The cached plans are freed with the proxy.
    survivor = weakref.ref(plans[8])
    del proxy, plans
    gc.collect()
    assert survivor() is None


def test_cache_filled_in_inference_mode_serves_autograd() -> None:
    grid = (9, 4, 8)
    proxy = Magi2RefinerDataProxy()
    with torch.inference_mode():
        cached = proxy.pack(_latent(grid), _text(7))

    latent = _latent(grid).requires_grad_()
    packed = proxy.pack(latent, _text(7))
    order, restore = proxy.block_order(grid)
    plan = packed.window_plan
    fields = [getattr(plan, field.name) for field in dataclasses.fields(plan)]
    plan_tensors = [value for value in fields if isinstance(value, torch.Tensor)]

    assert plan is cached.window_plan and len(plan_tensors) == 9
    assert not any(tensor.is_inference() for tensor in (order, restore, *plan_tensors))
    # pack and unpack index with the cached order and restore, which autograd saves.
    proxy.unpack(packed.video * 2.0, grid).sum().backward()
    assert torch.equal(latent.grad, torch.full_like(latent, 2.0))


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_shards_follow_the_reference_split_and_concatenate_back(world_size: int) -> None:
    proxy = Magi2RefinerDataProxy()
    # 288 video tokens and 100 text tokens: at 8 ranks the text spans three of them.
    packed = proxy.pack(_latent((9, 4, 8)), _text(100))
    num_tokens = 388
    shards = [packed.shard(rank, world_size) for rank in range(world_size)]

    # SandAI's ulysses_scheduler: the first N % world_size ranks get one extra token.
    remainder = num_tokens % world_size
    base = (num_tokens - remainder) // world_size
    expected_sizes = (base + 1,) * remainder + (base,) * (world_size - remainder)
    assert all(shard.split_sizes == expected_sizes for shard in shards)
    assert [shard.rank for shard in shards] == list(range(world_size))
    for field in ("video", "text", "coords", "modality"):
        assert torch.equal(torch.cat([getattr(shard, field) for shard in shards]), getattr(packed, field)), field
    for shard in shards:
        start = sum(expected_sizes[: shard.rank])
        end = start + expected_sizes[shard.rank]
        assert torch.equal(shard.coords, packed.coords[start:end])
        assert torch.equal(shard.modality, packed.modality[start:end])
        assert int((shard.modality == _VIDEO).sum()) == shard.video.shape[0]
        assert int((shard.modality == _TEXT).sum()) == shard.text.shape[0]
        assert shard.window_plan is packed.window_plan
        assert shard.video_split_sizes == [s.video.shape[0] for s in shards]
    assert shards[-1].text.shape[0] > 0


def test_shard_rejects_sliced_inputs_and_bad_ranks() -> None:
    packed = Magi2RefinerDataProxy().pack(_latent((9, 4, 8)), _text(3))

    with pytest.raises(ValueError, match="rank must be in"):
        packed.shard(2, 2)
    with pytest.raises(ValueError, match="only an unsliced"):
        packed.shard(0, 2).shard(0, 1)


def test_per_modality_embedding_equals_the_padded_pack_embedding() -> None:
    config = Magi2RefinerConfig(num_layers=1, hidden_size=64, head_dim=32, num_query_groups=1, multimodal_layers=(0,))
    model = Magi2RefinerTransformer(config)
    generator = torch.Generator().manual_seed(4)
    with torch.no_grad():
        for parameter in model.pre_adapter.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
    grid, text_len = (9, 4, 8), 37
    latent, text = _latent(grid), _text(text_len)
    packed = Magi2RefinerDataProxy(config).pack(latent, text)
    x, _, modality = _ref_packed_sequence(latent, text[0])
    adapter = model.pre_adapter

    with torch.no_grad():
        expected = torch.zeros(x.shape[0], config.hidden_size)
        text_mask, video_mask = modality == _TEXT, modality == _VIDEO
        expected[text_mask] = adapter.text_embedder(x[text_mask, : config.text_in_channels])
        expected[video_mask] = adapter.video_embedder(x[video_mask, : config.video_in_channels])
        actual = adapter(
            packed.video,
            packed.text,
            torch.nonzero(packed.modality == _VIDEO).flatten(),
            torch.nonzero(packed.modality == _TEXT).flatten(),
        )

    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)


def test_model_kwargs_drive_the_refiner_and_unpack_its_velocity() -> None:
    config = Magi2RefinerConfig(
        num_layers=1, hidden_size=64, head_dim=32, num_query_groups=1, multimodal_layers=(0,), text_in_channels=16
    )
    proxy = Magi2RefinerDataProxy(config)
    packed = proxy.pack(_latent((9, 4, 8)), _text(5, channels=16))
    kwargs = packed.model_kwargs()

    assert set(kwargs) == set(inspect.signature(Magi2RefinerTransformer.forward).parameters) - {"self"}
    model = Magi2RefinerTransformer(config)
    with torch.no_grad():
        velocity = model(**kwargs)
    assert velocity.shape == (288, 48)
    assert proxy.unpack(velocity, packed.grid).shape == (1, 48, 9, 4, 8)


def test_pack_and_unpack_validate_shapes() -> None:
    proxy = Magi2RefinerDataProxy()
    latent, text = _latent((9, 4, 8)), _text(3)

    with pytest.raises(ValueError, match="one \\[1, 48, T, H, W\\] latent"):
        proxy.pack(torch.cat((latent, latent)), text)
    with pytest.raises(ValueError, match="one \\[1, 48, T, H, W\\] latent"):
        proxy.pack(latent[:, :47], text)
    with pytest.raises(ValueError, match="text features"):
        proxy.pack(latent, torch.cat((text, text)))
    with pytest.raises(ValueError, match="text features"):
        proxy.pack(latent, text[..., :5119])
    with pytest.raises(ValueError, match="H % 4 == 0"):
        proxy.pack(_latent((9, 6, 8)), text)
    with pytest.raises(ValueError, match="block-major velocity"):
        proxy.unpack(torch.zeros(288, 47), (9, 4, 8))
