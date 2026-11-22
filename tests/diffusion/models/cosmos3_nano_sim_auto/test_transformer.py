# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sim-Auto transformer: token embedding (rig identity, timestep gating), control K/V reuse and Ulysses."""

import os
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.helpers.distributed import configure_cpu_collective_worker, start_collective_workers
from tests.helpers.runtime import get_open_port
from vllm_omni.diffusion.models.cosmos3_nano_sim_auto.causal_plan import build_causal_maskless_plan
from vllm_omni.diffusion.models.cosmos3_nano_sim_auto.geometry import Chunk, SensorGrid
from vllm_omni.diffusion.models.cosmos3_nano_sim_auto.layout import (
    ITEM_ORDER,
    CaptionDocument,
    RingSlot,
    build_chunk_items,
    caption_metadata,
    chunk_token_metadata,
    committed_metadata,
)
from vllm_omni.diffusion.models.cosmos3_nano_sim_auto.parallel import UlyssesGroup, current_ulysses_group
from vllm_omni.diffusion.models.cosmos3_nano_sim_auto.transformer_cosmos3_nano_sim_auto import (
    CausalMasklessRuntime,
    Cosmos3NanoSimAutoGenDecoderLayer,
    Cosmos3NanoSimAutoJointAttention,
    Cosmos3NanoSimAutoTransformer,
    item_token_ranges,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

CHANNELS, HIDDEN, RIG_ROWS, LIDAR_ROW = 3, 5, 12, 11


class _Host(nn.Module):
    """The attributes ``embed_items`` and ``validate_loaded_weights`` read, with 1x1 patches."""

    def __init__(self, *, rig: bool) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.proj_in = nn.Linear(CHANNELS, HIDDEN)
        self.lidar_proj_in = nn.Linear(CHANNELS, HIDDEN)
        self.rig_view_embed = nn.Embedding(RIG_ROWS, HIDDEN) if rig else None
        self.rig_lidar_id = LIDAR_ROW if rig else None
        self.time = torch.randn(HIDDEN, generator=generator)

    def _item_patch(self, item) -> tuple[int, int]:
        return (1, 1)

    def _embed_timestep(self, timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return self.time.to(dtype).view(1, -1)


def _chunk_items(chunk: Chunk):
    return build_chunk_items(
        chunk,
        num_views=2,
        rgb_grid=SensorGrid(2, 2, 1, 1),
        rgb_seconds_per_frame=4 / 30,
        lidar_grid=SensorGrid(1, 3, 1, 1),
        lidar_seconds_per_frame=0.1,
        rgb_condition_frames={1},
    )


def _joint_items():
    items = _chunk_items(Chunk(step=1, rgb_frames=(1, 2), lidar_sweeps=(1,), rgb_prefix_end=3, lidar_prefix_end=2))
    generator = torch.Generator().manual_seed(1)
    latents = []
    for item in items:
        shape = (1, CHANNELS, len(item.view_ids) * len(item.frames), item.grid.latent_height, item.grid.latent_width)
        latents.append(torch.randn(shape, generator=generator))
    return items, latents


@torch.no_grad()
def test_rig_rows_follow_request_cameras_and_lidar_row_before_timestep() -> None:
    host = _Host(rig=True)
    items, latents = _joint_items()
    rig_view_ids = torch.tensor([5, 0])  # request order: cross-left, then front-wide
    hidden = Cosmos3NanoSimAutoTransformer.embed_items(
        host, items, latents, timestep=torch.tensor([500.0]), rig_view_ids=rig_view_ids
    )

    expected = []
    for item, latent in zip(items, latents, strict=True):
        project = host.lidar_proj_in if item.is_lidar else host.proj_in
        tokens = project(latent[0].permute(1, 2, 3, 0).reshape(-1, CHANNELS))  # (frame, h, w) order
        per_view = tokens.shape[0] // len(item.view_ids)
        if item.is_lidar:
            rows = host.rig_view_embed.weight[LIDAR_ROW].expand_as(tokens)
        else:
            rows = host.rig_view_embed(rig_view_ids).repeat_interleave(per_view, dim=0)
        tokens = tokens + rows
        if not item.is_control:
            frame_of_token = (
                torch.tensor(item.frames).repeat_interleave(item.tokens_per_frame).repeat(len(item.view_ids))
            )
            generated = torch.tensor([int(frame) not in item.condition_frames for frame in frame_of_token])
            tokens = tokens + host.time * generated[:, None]
        expected.append(tokens)
    torch.testing.assert_close(hidden[0], torch.cat(expected))


@torch.no_grad()
def test_rig_view_ids_are_required_exactly_with_a_rig_table() -> None:
    items, latents = _joint_items()
    with pytest.raises(ValueError, match="exactly when the artifact declares a rig table"):
        Cosmos3NanoSimAutoTransformer.embed_items(_Host(rig=True), items, latents, timestep=None)
    with pytest.raises(ValueError, match="exactly when the artifact declares a rig table"):
        Cosmos3NanoSimAutoTransformer.embed_items(
            _Host(rig=False), items, latents, timestep=None, rig_view_ids=torch.tensor([0, 1])
        )
    host = _Host(rig=False)
    hidden = Cosmos3NanoSimAutoTransformer.embed_items(host, items, latents, timestep=None)
    plain = [
        (host.lidar_proj_in if item.is_lidar else host.proj_in)(latent[0].permute(1, 2, 3, 0).reshape(-1, CHANNELS))
        for item, latent in zip(items, latents, strict=True)
    ]
    torch.testing.assert_close(hidden[0], torch.cat(plain))


def test_rig_weights_must_match_the_declared_table() -> None:
    with_rig = _Host(rig=True)
    names = {f"transformer.{name}" for name, _ in with_rig.named_parameters()}
    Cosmos3NanoSimAutoTransformer.validate_loaded_weights(with_rig, names)
    with pytest.raises(ValueError, match="missing required transformer weights: transformer.rig_view_embed.weight"):
        Cosmos3NanoSimAutoTransformer.validate_loaded_weights(with_rig, names - {"transformer.rig_view_embed.weight"})
    without_rig = _Host(rig=False)
    names = {f"transformer.{name}" for name, _ in without_rig.named_parameters()}
    with pytest.raises(ValueError, match="declares no rig_view_embedding"):
        Cosmos3NanoSimAutoTransformer.validate_loaded_weights(
            without_rig, names | {"transformer.rig_view_embed.weight"}
        )


# -- Control K/V reuse ------------------------------------------------------------

HEADS, KV_HEADS, HEAD_DIM, LAYERS = 2, 1, 4, 2


def _module(cls, **attributes):
    """An instance of ``cls`` holding only ``attributes`` (skips the vLLM-parallel constructor)."""
    module = cls.__new__(cls)
    nn.Module.__init__(module)
    for name, value in attributes.items():
        setattr(module, name, value)
    return module


class _ForwardHost(_Host):
    """Real forward, GEN layers and joint attention over plain linears; item-local fake RoPE."""

    forward = Cosmos3NanoSimAutoTransformer.forward
    embed_items = Cosmos3NanoSimAutoTransformer.embed_items
    local_kv_heads = Cosmos3NanoSimAutoTransformer.local_kv_heads
    _inductor_cudagraphs = False
    num_hidden_layers = LAYERS
    head_dim = HEAD_DIM
    ulysses: UlyssesGroup | None = None

    def __init__(self, *, heads: int = HEADS, kv_heads: int = KV_HEADS) -> None:
        super().__init__(rig=False)
        self.num_kv_heads_local = kv_heads

        def qk_norm():
            return SimpleNamespace(weight=torch.rand(HEAD_DIM, dtype=torch.float64) + 0.5, variance_epsilon=1e-6)

        def attention():
            return _module(
                Cosmos3NanoSimAutoJointAttention,
                to_q=nn.Linear(HIDDEN, heads * HEAD_DIM),
                to_k=nn.Linear(HIDDEN, kv_heads * HEAD_DIM),
                to_v=nn.Linear(HIDDEN, kv_heads * HEAD_DIM),
                to_out=nn.Linear(heads * HEAD_DIM, HIDDEN),
                num_heads_local=heads,
                num_kv_heads_local=kv_heads,
                head_dim=HEAD_DIM,
                qk_norm=True,
                norm_q=qk_norm(),
                norm_k=qk_norm(),
            )

        self.gen_layers = nn.ModuleList(
            _module(
                Cosmos3NanoSimAutoGenDecoderLayer,
                input_layernorm=nn.LayerNorm(HIDDEN),
                cross_attention=attention(),
                post_attention_layernorm=nn.LayerNorm(HIDDEN),
                mlp=nn.Sequential(nn.Linear(HIDDEN, 2 * HIDDEN), nn.SiLU(), nn.Linear(2 * HIDDEN, HIDDEN)),
            )
            for _ in range(LAYERS)
        )
        self.norm_moe_gen = nn.LayerNorm(HIDDEN)
        self.proj_out = nn.Linear(HIDDEN, CHANNELS)
        self.lidar_proj_out = nn.Linear(HIDDEN, CHANNELS)

    def _offload_context(self, name: str):
        return nullcontext()

    def chunk_rope(self, items, *, text_extent: int, device: torch.device, dtype: torch.dtype):
        # Like the real mRoPE ids, every item's angles depend on that item alone.
        positions = torch.cat(
            [
                torch.arange(item.num_tokens) + 7.0 * item.frames[0] + 31.0 * ITEM_ORDER.index(item.kind)
                for item in items
            ]
        )
        angles = positions[:, None] * torch.linspace(0.1, 1.0, HEAD_DIM // 2)[None, :]
        angles = torch.cat([angles, angles], dim=-1).to(device=device, dtype=dtype)
        return angles.cos()[None, :, None], angles.sin()[None, :, None]


def _context(items, pass_kind: str, generator: torch.Generator, *, kv_heads: int = KV_HEADS):
    """Plans (full and targets-only) plus the ring and caption K/V of chunk 1."""
    chunk0 = _chunk_items(Chunk(step=0, rgb_frames=(0,), lidar_sweeps=(0,), rgb_prefix_end=1, lidar_prefix_end=1))
    ring = RingSlot(0, 0, committed_metadata(chunk_token_metadata(chunk0, step=0, pass_kind="clean")[0]))
    documents = [CaptionDocument(view, 2 + view, start_seconds=0.0, end_seconds=10.0) for view in range(2)]
    current, _ = chunk_token_metadata(items, step=1, pass_kind=pass_kind)

    def runtime(reuse: bool) -> CausalMasklessRuntime:
        plan = build_causal_maskless_plan(
            current=current,
            history=[ring],
            captions=caption_metadata(documents),
            window_seconds=0.4,
            reuse_control_kv=reuse,
        )
        return CausalMasklessRuntime(plan.flatten(), len(plan.passes), "torch")

    def kv(*shape):
        return tuple(torch.randn(*shape, kv_heads, HEAD_DIM, generator=generator, dtype=torch.float64) for _ in "kv")

    streams = {
        "text_extent": 0,
        "und_kv": [kv(1, sum(doc.num_tokens for doc in documents)) for _ in range(LAYERS)],
        "history_kv": [kv(ring.metadata.num_tokens) for _ in range(LAYERS)],
    }
    return runtime(False), runtime(True), streams


@torch.no_grad()
def test_targets_only_forwards_with_cached_control_kv_match_full_forwards() -> None:
    torch.manual_seed(0)
    host = _ForwardHost().double()
    generator = torch.Generator().manual_seed(5)
    items, latents = _joint_items()
    latents = [latent.double() for latent in latents]
    controls = torch.cat([torch.arange(a, a + n) for a, n in item_token_ranges(items, controls=True)])
    targets = torch.cat([torch.arange(a, a + n) for a, n in item_token_ranges(items, controls=False)])
    full_noisy, reuse_noisy, streams = _context(items, "noisy", generator)

    first = host.forward(
        items=items,
        latents=latents,
        timestep=torch.tensor([900.0]),
        runtime=full_noisy,
        collect_control_kv=True,
        **streams,
    )
    assert len(first.control_kv) == LAYERS
    assert all(key.shape == (controls.numel(), KV_HEADS, HEAD_DIM) for key, _ in first.control_kv)
    later = [latent if item.is_control else torch.randn_like(latent) for item, latent in zip(items, latents)]
    expected = host.forward(
        items=items,
        latents=later,
        timestep=torch.tensor([500.0]),
        runtime=full_noisy,
        collect_control_kv=True,
        **streams,
    )
    # Neither the targets nor the timestep reach the controls' K/V.
    for (k_first, v_first), (k_later, v_later) in zip(first.control_kv, expected.control_kv, strict=True):
        assert torch.equal(k_first, k_later) and torch.equal(v_first, v_later)
    # A targets-only forward ignores control latents entirely.
    poisoned = [torch.full_like(x, float("nan")) if item.is_control else x for item, x in zip(items, later)]
    actual = host.forward(
        items=items,
        latents=poisoned,
        timestep=torch.tensor([500.0]),
        runtime=reuse_noisy,
        control_kv=first.control_kv,
        **streams,
    )
    assert actual.control_kv is None
    for got, want in zip(actual.targets, expected.targets, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)

    # Clean refresh: the forward publishes only its own (target) tokens; the pipeline adds the controls.
    full_clean, reuse_clean, _ = _context(items, "clean", generator)
    num_tokens = sum(item.num_tokens for item in items)
    slots = torch.arange(num_tokens) + 3

    def pools():
        return [
            tuple(torch.zeros(num_tokens + 3, KV_HEADS, HEAD_DIM, dtype=torch.float64) for _ in "kv")
            for _ in range(LAYERS)
        ]

    full_pools, reuse_pools = pools(), pools()
    refreshed = host.forward(
        items=items,
        latents=later,
        timestep=None,
        runtime=full_clean,
        write_slots=slots,
        kv_pools=full_pools,
        collect_kv=True,
        **streams,
    )
    reused = host.forward(
        items=items,
        latents=poisoned,
        timestep=None,
        runtime=reuse_clean,
        write_slots=slots[targets],
        kv_pools=reuse_pools,
        collect_kv=True,
        control_kv=first.control_kv,
        **streams,
    )
    for got, want in zip(reused.targets, refreshed.targets, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)
    for layer in range(LAYERS):
        for got, want, cached in zip(reused.current_kv[layer], refreshed.current_kv[layer], first.control_kv[layer]):
            torch.testing.assert_close(got[0], want[0, targets], rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(cached, want[0, controls], rtol=0, atol=0)
        for got, want in zip(reuse_pools[layer], full_pools[layer], strict=True):
            torch.testing.assert_close(got[slots[targets]], want[slots[targets]], rtol=1e-6, atol=1e-6)
            assert not got[slots[controls]].any()


def test_control_kv_reuse_rejects_mismatched_caches() -> None:
    host = _ForwardHost().double()
    items, latents = _joint_items()
    full_noisy, reuse_noisy, streams = _context(items, "noisy", torch.Generator().manual_seed(0))
    cache = [(torch.zeros(1, KV_HEADS, HEAD_DIM), torch.zeros(1, KV_HEADS, HEAD_DIM))] * LAYERS
    with pytest.raises(ValueError, match="control tokens"):
        host.forward(items=items, latents=latents, timestep=None, runtime=reuse_noisy, control_kv=cache, **streams)
    with pytest.raises(ValueError, match="either collects control K/V or reuses it"):
        host.forward(
            items=items,
            latents=latents,
            timestep=None,
            runtime=full_noisy,
            control_kv=cache,
            collect_control_kv=True,
            **streams,
        )


# -- Ulysses -------------------------------------------------------------------------

#: Enough heads for Ulysses 4; the 38-token chunk (19 non-control) pads at degrees 2 and 4.
ULYSSES_HEADS, ULYSSES_KV_HEADS = 8, 4


def _heads(tensor: torch.Tensor, heads: slice) -> torch.Tensor:
    """Head group of a ``[..., H, D]`` K/V tensor."""
    return tensor[..., heads, :]


def _rollout_forwards(host, items, latents, context, *, und_kv, history_kv):
    """A chunk's forwards: denoise (collecting control K/V), targets-only denoise, targets-only refresh."""
    full_noisy, reuse_noisy, reuse_clean = context
    streams = {"text_extent": 0, "und_kv": und_kv, "history_kv": history_kv}
    num_tokens = sum(item.num_tokens for item in items)
    targets = torch.cat([torch.arange(a, a + n) for a, n in item_token_ranges(items, controls=False)])
    first = host.forward(
        items=items,
        latents=latents,
        timestep=torch.tensor([900.0]),
        runtime=full_noisy,
        collect_control_kv=True,
        **streams,
    )
    later = host.forward(
        items=items,
        latents=latents,
        timestep=torch.tensor([500.0]),
        runtime=reuse_noisy,
        control_kv=first.control_kv,
        **streams,
    )
    heads = und_kv[0][0].shape[2]
    pools = [
        tuple(torch.zeros(num_tokens + 3, heads, HEAD_DIM, dtype=torch.float64) for _ in "kv") for _ in range(LAYERS)
    ]
    refreshed = host.forward(
        items=items,
        latents=latents,
        timestep=None,
        runtime=reuse_clean,
        write_slots=(torch.arange(num_tokens) + 3)[targets],
        kv_pools=pools,
        collect_kv=True,
        control_kv=first.control_kv,
        **streams,
    )
    return first, later, refreshed, pools


@torch.no_grad()
def _check_ulysses_matches_one_rank(ulysses: UlyssesGroup) -> None:
    torch.manual_seed(0)
    host = _ForwardHost(heads=ULYSSES_HEADS, kv_heads=ULYSSES_KV_HEADS).double()
    items, latents = _joint_items()
    latents = [latent.double() for latent in latents]
    generator = torch.Generator().manual_seed(5)
    full_noisy, reuse_noisy, streams = _context(items, "noisy", generator, kv_heads=ULYSSES_KV_HEADS)
    _, reuse_clean, _ = _context(items, "clean", generator, kv_heads=ULYSSES_KV_HEADS)
    context = (full_noisy, reuse_noisy, reuse_clean)
    expected = _rollout_forwards(
        host, items, latents, context, und_kv=streams["und_kv"], history_kv=streams["history_kv"]
    )

    host.ulysses = ulysses
    host.num_kv_heads_local = ULYSSES_KV_HEADS // ulysses.world_size
    heads = ulysses.head_slice(ULYSSES_KV_HEADS)
    und_kv = host.local_kv_heads(streams["und_kv"])
    for (key, value), (full_key, full_value) in zip(und_kv, streams["und_kv"], strict=True):
        assert torch.equal(key, _heads(full_key, heads)) and torch.equal(value, _heads(full_value, heads))
    history_kv = [(_heads(key, heads), _heads(value, heads)) for key, value in streams["history_kv"]]
    actual = _rollout_forwards(host, items, latents, context, und_kv=und_kv, history_kv=history_kv)

    # Every rank gathers the full prediction. Per-token math is unchanged; only GEMM shapes differ, which the
    # executor's FP32 pass merge can round differently.
    for got_output, want_output in zip(actual[:3], expected[:3], strict=True):
        for got, want in zip(got_output.targets, want_output.targets, strict=True):
            torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)
    first, _, refreshed, pools = actual
    for layer in range(LAYERS):
        for got, want in zip(first.control_kv[layer], expected[0].control_kv[layer], strict=True):
            torch.testing.assert_close(got, _heads(want, heads), rtol=1e-6, atol=1e-6)
        for got, want in zip(refreshed.current_kv[layer], expected[2].current_kv[layer], strict=True):
            torch.testing.assert_close(got, _heads(want, heads), rtol=1e-6, atol=1e-6)
        for got, want in zip(pools[layer], expected[3][layer], strict=True):
            torch.testing.assert_close(got, _heads(want, heads), rtol=1e-6, atol=1e-6)


def _ulysses_worker(rank: int, world_size: int, port: int) -> None:
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_distributed_env,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
        MASTER_ADDR="localhost",
        MASTER_PORT=str(port),
    )
    configure_cpu_collective_worker()
    init_distributed_environment(backend="gloo")
    try:
        initialize_model_parallel(ulysses_degree=world_size)
        ulysses = current_ulysses_group()
        assert ulysses is not None and (ulysses.rank, ulysses.world_size) == (rank, world_size)
        _check_ulysses_matches_one_rank(ulysses)
    finally:
        destroy_distributed_env()


@pytest.mark.parametrize("world_size", [2, 4])
def test_ulysses_forwards_match_one_rank(world_size: int) -> None:
    """Real Gloo ranks: padded token shards, packed Q/K/V exchange, head-group caches and pool writes."""
    start_collective_workers(_ulysses_worker, args=(world_size, get_open_port()), nprocs=world_size, device_kind="cpu")


def test_without_distributed_state_there_is_no_ulysses_group() -> None:
    assert current_ulysses_group() is None
    assert _ForwardHost().local_kv_heads([(torch.zeros(1, 2, 1, 4),) * 2])[0][0].shape == (1, 2, 1, 4)
