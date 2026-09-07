# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SP unit tests for SANA-Video: balanced frame split sizes, linear-attention
state reduction and the uneven frame gather.

CPU-only. The SP world size, rank and the collectives are mocked so a single
process simulates the shard/reduce logic a real multi-rank run depends on.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.sana_video.transformer_sana_video import (
    GLUMBTempConv,
    SanaAdaLayerNormSingle,
    SanaLinearAttention,
    _sp_frame_split_sizes,
    _sp_gather_frames,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_MODULE = "vllm_omni.diffusion.models.sana_video.transformer_sana_video"


@pytest.fixture
def tp1_group():
    """Real single-process TP group so the parallel linear layers construct and
    run on CPU at tensor_parallel_size=1."""
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29501")
    init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
    initialize_model_parallel()
    yield
    cleanup_dist_env_and_memory()


@pytest.fixture
def force_default_gemm(monkeypatch):
    """Force CPU-compatible GEMM dispatch for the parallel linear layers."""
    from vllm.model_executor.layers.utils import default_unquantized_gemm

    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.dispatch_unquantized_gemm",
        lambda: default_unquantized_gemm,
    )


# ── frame split sizes ──


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("num_frames", [5, 6, 9, 11, 21])
def test_frame_split_sizes_are_balanced_and_cover_all_frames(num_frames, world_size) -> None:
    """Exactly world_size non-empty near-equal chunks that sum to num_frames.

    torch.chunk's ceil semantics produce fewer than world_size chunks for
    5/6/9 frames at world size 4, which is why this helper exists.
    """
    sizes = _sp_frame_split_sizes(num_frames, world_size)

    assert len(sizes) == world_size
    assert all(size >= 1 for size in sizes)
    assert sum(sizes) == num_frames
    assert max(sizes) - min(sizes) <= 1


def test_frame_split_sizes_known_layouts() -> None:
    assert _sp_frame_split_sizes(21, 2) == [11, 10]
    assert _sp_frame_split_sizes(11, 2) == [6, 5]
    assert _sp_frame_split_sizes(9, 4) == [3, 2, 2, 2]
    assert _sp_frame_split_sizes(5, 4) == [2, 1, 1, 1]


def test_frame_split_sizes_rejects_fewer_frames_than_ranks() -> None:
    with pytest.raises(ValueError, match="latent frame per rank"):
        _sp_frame_split_sizes(3, 4)


# ── linear attention state reduction ──


class _FakeSpAllReduce:
    """Two-pass stand-in for the SP all-reduce: pass 1 records each rank's
    partial, pass 2 replays their sum to every rank."""

    def __init__(self) -> None:
        self.partials: list[torch.Tensor] = []
        self.total: torch.Tensor | None = None

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        self.partials.append(tensor.clone())
        if self.total is not None:
            return self.total.clone()
        return tensor


def _make_attn(dim: int, num_heads: int, head_dim: int) -> SanaLinearAttention:
    attn = SanaLinearAttention(
        dim=dim, num_heads=num_heads, head_dim=head_dim, dropout=0.0, bias=False, qk_norm="rms_norm_across_heads"
    )
    attn.eval()
    torch.manual_seed(0)
    for _, param in sorted(attn.named_parameters()):
        param.data.normal_()
    return attn


def _attn_inputs(seq_len: int, dim: int, head_dim: int):
    """Second half of the sequence drawn from a shifted, scaled distribution so
    per-rank partial sums differ; a missing state reduction cannot cancel out."""
    torch.manual_seed(1)
    x = torch.randn(2, seq_len, dim)
    x[:, seq_len // 2 :] = x[:, seq_len // 2 :] * 3.0 + 1.0
    freqs_cos = torch.randn(1, seq_len, 1, head_dim)
    freqs_sin = torch.randn(1, seq_len, 1, head_dim)
    return x, (freqs_cos, freqs_sin)


def test_attn1_sp2_state_reduction_matches_dense(tp1_group, force_default_gemm, mocker) -> None:
    """Two token shards with a summed packed state must reproduce the dense
    full-sequence linear attention output."""
    dim, num_heads, head_dim, seq_len = 24, 4, 6, 12
    attn = _make_attn(dim, num_heads, head_dim)
    x, (freqs_cos, freqs_sin) = _attn_inputs(seq_len, dim, head_dim)
    bounds = [(0, 8), (8, 12)]

    with torch.no_grad():
        ref = attn(x, rotary_emb=(freqs_cos, freqs_sin))

    fake_group = _FakeSpAllReduce()
    mocker.patch(f"{_MODULE}.get_sequence_parallel_world_size", return_value=2)
    mocker.patch(f"{_MODULE}.get_sp_group", return_value=fake_group)

    def run_ranks() -> list[torch.Tensor]:
        outs = []
        for start, stop in bounds:
            local_rotary = (freqs_cos[:, start:stop], freqs_sin[:, start:stop])
            with torch.no_grad():
                outs.append(attn(x[:, start:stop], rotary_emb=local_rotary))
        return outs

    run_ranks()
    fake_group.total = fake_group.partials[0] + fake_group.partials[1]
    out = torch.cat(run_ranks(), dim=1)

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_attn1_issues_single_packed_all_reduce(tp1_group, force_default_gemm, mocker) -> None:
    """One collective per forward, carrying scores and k_sum packed together."""
    dim, num_heads, head_dim, seq_len = 24, 4, 6, 8
    attn = _make_attn(dim, num_heads, head_dim)
    x, rotary_emb = _attn_inputs(seq_len, dim, head_dim)

    fake_group = _FakeSpAllReduce()
    mocker.patch(f"{_MODULE}.get_sequence_parallel_world_size", return_value=2)
    mocker.patch(f"{_MODULE}.get_sp_group", return_value=fake_group)

    with torch.no_grad():
        attn(x, rotary_emb=rotary_emb)

    assert len(fake_group.partials) == 1
    packed = fake_group.partials[0]
    assert packed.shape == (2, num_heads, head_dim, head_dim + 1)
    assert packed.dtype == torch.float32


def test_attn1_sp1_path_is_untouched(tp1_group, force_default_gemm, mocker) -> None:
    """At SP1 no collective may run: the dense operator sequence must not
    change shape or route through get_sp_group at all."""
    dim, num_heads, head_dim, seq_len = 24, 4, 6, 8
    attn = _make_attn(dim, num_heads, head_dim)
    x, rotary_emb = _attn_inputs(seq_len, dim, head_dim)

    mock_group = mocker.patch(f"{_MODULE}.get_sp_group")

    with torch.no_grad():
        attn(x, rotary_emb=rotary_emb)

    mock_group.assert_not_called()


# ── uneven frame gather ──


class _FakeSpAllGather:
    """Stand-in for the SP all-gather: returns precomputed equal-shape per-rank
    contributions, enforcing the equal-shard contract of the real collective."""

    def __init__(self, parts: list[torch.Tensor]) -> None:
        self.parts = parts

    def all_gather(self, tensor: torch.Tensor, dim: int = 0, separate_tensors: bool = False):
        assert separate_tensors, "frame gather must request per-rank tensors"
        assert all(part.shape == tensor.shape for part in self.parts), "all_gather requires equal shards"
        return [part.clone() for part in self.parts]


@pytest.mark.parametrize("num_frames,world_size", [(21, 2), (6, 2), (5, 4)])
def test_gather_frames_roundtrips_uneven_shards(num_frames, world_size, mocker) -> None:
    """shard -> pad -> all_gather -> per-rank narrow -> cat must reproduce the
    full sequence bitwise, with communication pads never reaching the output."""
    batch, hw, channels = 2, 3, 4
    torch.manual_seed(0)
    full = torch.randn(batch, num_frames * hw, channels)

    sizes = _sp_frame_split_sizes(num_frames, world_size)
    frame_view = full.unflatten(1, (num_frames, hw))
    locals_ = [t.flatten(1, 2) for t in frame_view.split(sizes, dim=1)]

    # Per-rank padded contributions as the real collective would deliver them;
    # NaN pads prove a leaked pad frame cannot go unnoticed.
    parts = []
    for local, size in zip(locals_, sizes):
        part = local.unflatten(1, (size, hw))
        pad = max(sizes) - size
        if pad:
            part = torch.cat([part, torch.full((batch, pad, hw, channels), torch.nan)], dim=1)
        parts.append(part)

    mocker.patch(f"{_MODULE}.get_sequence_parallel_world_size", return_value=world_size)
    mocker.patch(f"{_MODULE}.get_sp_group", return_value=_FakeSpAllGather(parts))
    for rank in range(world_size):
        mocker.patch(f"{_MODULE}.get_sequence_parallel_rank", return_value=rank)

        gathered = _sp_gather_frames(locals_[rank], sizes)

        torch.testing.assert_close(gathered, full, rtol=0, atol=0)


# ── GLUMB conv_temp halo exchange ──


class _NoopReq:
    def wait(self) -> None:
        pass


class _ReplayP2PDist:
    """Two-pass stand-in for torch.distributed P2P: pass 1 records each rank's
    boundary sends, pass 2 replays them into the matching neighbour recvs."""

    irecv = "irecv"
    isend = "isend"

    def __init__(self) -> None:
        self.rank = 0
        self.sent: dict[tuple[int, int], torch.Tensor] = {}
        self.replay = False
        self.batches = 0
        self.peers: set[tuple[int, int]] = set()

    def P2POp(self, op, tensor, peer, group):
        return (op, tensor, peer)

    def batch_isend_irecv(self, ops):
        self.batches += 1
        for op, tensor, peer in ops:
            self.peers.add((self.rank, peer))
            if op == "isend":
                self.sent[(self.rank, peer)] = tensor.clone()
            elif self.replay:
                tensor.copy_(self.sent[(peer, self.rank)])
        return [_NoopReq() for _ in ops]


class _FakeSpGroup:
    def __init__(self, world_size: int) -> None:
        self.ranks = list(range(world_size))
        self.device_group = None


def _make_glumb(mocker) -> GLUMBTempConv:
    mocker.patch(f"{_MODULE}.get_tensor_model_parallel_world_size", return_value=1)
    mocker.patch(f"{_MODULE}.get_tensor_model_parallel_rank", return_value=0)
    torch.manual_seed(0)
    glumb = GLUMBTempConv(4, 4, 2.0, norm_type=None, residual_connection=False)
    glumb.eval()
    for _, param in sorted(glumb.named_parameters()):
        param.data.normal_()
    return glumb


@pytest.mark.parametrize("world_size", [2, 4])
def test_glumb_conv_temp_sp_halo_matches_dense(world_size, mocker) -> None:
    """Frame shards with neighbour boundary frames exchanged must reproduce the
    dense temporal conv, including zero padding at the global first/last frame.
    The exchange must talk to the two neighbouring ranks only."""
    glumb = _make_glumb(mocker)
    num_frames = 5
    torch.manual_seed(1)
    x = torch.randn(2, num_frames, 3, 3, 4)

    with torch.no_grad():
        ref = glumb(x)

    sizes = _sp_frame_split_sizes(num_frames, world_size)
    shards = [shard.contiguous() for shard in x.split(sizes, dim=1)]
    fake_dist = _ReplayP2PDist()
    mocker.patch(f"{_MODULE}.get_sequence_parallel_world_size", return_value=world_size)
    mocker.patch(f"{_MODULE}.get_sp_group", return_value=_FakeSpGroup(world_size))
    mocker.patch(f"{_MODULE}.dist", fake_dist)

    def run_ranks() -> list[torch.Tensor]:
        outs = []
        for rank in range(world_size):
            fake_dist.rank = rank
            mocker.patch(f"{_MODULE}.get_sequence_parallel_rank", return_value=rank)
            with torch.no_grad():
                outs.append(glumb(shards[rank]))
        return outs

    run_ranks()
    assert fake_dist.batches == world_size
    assert fake_dist.peers == {(a, b) for a in range(world_size) for b in (a - 1, a + 1) if 0 <= b < world_size}
    fake_dist.replay = True
    out = torch.cat(run_ranks(), dim=1)

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_glumb_sp_halo_rejects_non_three_frame_kernel(mocker) -> None:
    """The single-frame halo is only correct for a three-frame temporal kernel;
    any other width must fail loudly instead of corrupting parity."""
    glumb = _make_glumb(mocker)
    glumb.conv_temp = nn.Conv2d(4, 4, kernel_size=(5, 1), stride=1, padding=(2, 0), bias=False)
    mocker.patch(f"{_MODULE}.get_sequence_parallel_world_size", return_value=2)
    mocker.patch(f"{_MODULE}.get_sequence_parallel_rank", return_value=0)
    mocker.patch(f"{_MODULE}.get_sp_group", return_value=_FakeSpGroup(2))
    mocker.patch(f"{_MODULE}.dist", _ReplayP2PDist())
    with pytest.raises(AssertionError, match="temporal kernel"):
        with torch.no_grad():
            glumb(torch.randn(1, 3, 3, 3, 4))


# ── I2V per-token timestep frame slicing ──


def test_i2v_timestep_frame_slice_matches_dense_token_region() -> None:
    """Slicing the 5D timestep along its frame dim before the per-token
    time-embed MLP must match the dense modulation for the same token region.

    The sinusoidal projection is bitwise under slicing; the MLP is only equal
    up to CPU GEMM blocking, which depends on the row count. A wrongly sliced
    frame produces O(1) differences, far outside this tolerance."""
    dim = 24
    torch.manual_seed(0)
    layer = SanaAdaLayerNormSingle(dim).eval()

    batch, num_frames, height, width = 2, 5, 2, 3
    timestep = torch.rand(batch, 1, num_frames, height, width) * 1000.0

    with torch.no_grad():
        dense_mod, dense_emb = layer(timestep.flatten(), batch_size=batch, hidden_dtype=torch.float32)
    dense_mod = dense_mod.view(batch, -1, dense_mod.size(-1))
    dense_emb = dense_emb.view(batch, -1, dense_emb.size(-1))

    sizes = _sp_frame_split_sizes(num_frames, 2)
    start = 0
    for rank, size in enumerate(sizes):
        local = timestep.split(sizes, dim=2)[rank]
        with torch.no_grad():
            local_mod, local_emb = layer(local.flatten(), batch_size=batch, hidden_dtype=torch.float32)
        local_mod = local_mod.view(batch, -1, local_mod.size(-1))
        local_emb = local_emb.view(batch, -1, local_emb.size(-1))

        lo, hi = start * height * width, (start + size) * height * width
        torch.testing.assert_close(local_mod, dense_mod[:, lo:hi], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(local_emb, dense_emb[:, lo:hi], rtol=1e-5, atol=1e-6)
        start += size


# ── full-model lockstep parity ──


_TINY_SP_CONFIG = {
    "in_channels": 4,
    "out_channels": 4,
    "num_attention_heads": 2,
    "attention_head_dim": 12,
    "num_layers": 2,
    "num_cross_attention_heads": 2,
    "cross_attention_head_dim": 12,
    "cross_attention_dim": 24,
    "caption_channels": 8,
    "mlp_ratio": 2.0,
    "patch_size": (1, 2, 2),
    "sample_size": 4,
    "rope_max_seq_len": 64,
}


class _LockstepSpGroup:
    """Barrier-synchronized collectives so rank threads run true lockstep SP in
    one process; records per-call inputs so tests can prove ranks computed on
    different shards."""

    def __init__(self, world_size: int, tls: threading.local) -> None:
        self.world_size = world_size
        self.ranks = list(range(world_size))
        self.device_group = None
        self.tls = tls
        self.barrier = threading.Barrier(world_size)
        self.slots: list[torch.Tensor | None] = [None] * world_size
        self.reduce_inputs: list[list[torch.Tensor]] = []
        self.gather_calls = 0
        self.lock = threading.Lock()

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        self.slots[self.tls.rank] = tensor
        self.barrier.wait(timeout=60)
        total = self.slots[0].clone()
        for part in self.slots[1:]:
            total = total + part
        if self.tls.rank == 0:
            self.reduce_inputs.append([part.clone() for part in self.slots])
        self.barrier.wait(timeout=60)
        return total

    def all_gather(self, tensor: torch.Tensor, dim: int = 0, separate_tensors: bool = False):
        assert separate_tensors
        self.slots[self.tls.rank] = tensor
        self.barrier.wait(timeout=60)
        assert all(part.shape == tensor.shape for part in self.slots), "all_gather requires equal shards"
        parts = [part.clone() for part in self.slots]
        if self.tls.rank == 0:
            with self.lock:
                self.gather_calls += 1
        self.barrier.wait(timeout=60)
        return parts


class _LockstepP2PDist:
    """Barrier-synchronized torch.distributed P2P stand-in: sends land in a
    mailbox, all rank threads rendezvous, then recvs read their neighbour's
    send of the same batch."""

    irecv = "irecv"
    isend = "isend"

    def __init__(self, world_size: int, tls: threading.local) -> None:
        self.tls = tls
        self.barrier = threading.Barrier(world_size)
        self.mailbox: dict[tuple[int, int], torch.Tensor] = {}
        self.peers: set[tuple[int, int]] = set()
        self.batches = 0
        self.lock = threading.Lock()

    def P2POp(self, op, tensor, peer, group):
        return (op, tensor, peer)

    def batch_isend_irecv(self, ops):
        rank = self.tls.rank
        with self.lock:
            self.batches += 1
            for op, tensor, peer in ops:
                self.peers.add((rank, peer))
                if op == "isend":
                    self.mailbox[(rank, peer)] = tensor.clone()
        self.barrier.wait(timeout=60)
        for op, tensor, peer in ops:
            if op == "irecv":
                tensor.copy_(self.mailbox[(peer, rank)])
        self.barrier.wait(timeout=60)
        return [_NoopReq() for _ in ops]


@pytest.mark.parametrize("task", ["t2v", "i2v"])
def test_tiny_transformer_sp2_lockstep_matches_dense(tp1_group, force_default_gemm, mocker, task) -> None:
    """Two lockstep rank threads over uneven frame shards must reproduce the
    dense tiny-transformer output, and must actually have sharded the work."""
    from vllm_omni.diffusion.models.sana_video import SanaVideoTransformer3DModel

    torch.manual_seed(3)
    model = SanaVideoTransformer3DModel(**_TINY_SP_CONFIG).eval()
    for _, param in sorted(model.named_parameters()):
        param.data.normal_()
    batch, frames, height, width = 2, 5, 4, 4
    torch.manual_seed(11)
    latent = torch.randn(batch, 4, frames, height, width)
    encoder_hidden_states = torch.randn(batch, 6, 8)
    encoder_attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])
    if task == "t2v":
        timestep = torch.tensor([500.0, 700.0])
    else:
        timestep = torch.rand(batch, 1, frames, height // 2, width // 2) * 1000.0

    with torch.no_grad():
        ref = model(
            latent,
            encoder_hidden_states,
            timestep,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=False,
        )[0]

    world_size = 2
    tls = threading.local()
    group = _LockstepSpGroup(world_size, tls)
    fake_dist = _LockstepP2PDist(world_size, tls)
    mocker.patch(f"{_MODULE}.get_sequence_parallel_world_size", return_value=world_size)
    mocker.patch(f"{_MODULE}.get_sequence_parallel_rank", side_effect=lambda: tls.rank)
    mocker.patch(f"{_MODULE}.get_sp_group", return_value=group)
    mocker.patch(f"{_MODULE}.dist", fake_dist)

    def run_rank(rank: int) -> torch.Tensor:
        tls.rank = rank
        with torch.no_grad():
            return model(
                latent,
                encoder_hidden_states,
                timestep,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]

    with ThreadPoolExecutor(max_workers=world_size) as pool:
        outs = list(pool.map(run_rank, range(world_size)))

    num_layers = _TINY_SP_CONFIG["num_layers"]
    # One attn1 state reduce per block, its inputs distinct per rank (the ranks
    # really processed different frames), one neighbour halo exchange per block
    # per rank, and only the final frame gather is a full collective.
    assert len(group.reduce_inputs) == num_layers
    for parts in group.reduce_inputs:
        assert not torch.equal(parts[0], parts[1])
    assert group.gather_calls == 1
    assert fake_dist.batches == world_size * num_layers
    assert fake_dist.peers == {(0, 1), (1, 0)}

    torch.testing.assert_close(outs[1], outs[0], rtol=0, atol=0)
    # Reduction-reorder noise amplified by the unit-normal test weights; any
    # sharding misalignment is orders of magnitude above this.
    torch.testing.assert_close(outs[0], ref, rtol=1e-3, atol=1e-3)


def test_transformer_declares_empty_sp_plan() -> None:
    """Empty plan: the registry takes its normal SP enablement path with zero
    hooks and no misleading 'no _sp_plan found' warning; sharding is manual."""
    from vllm_omni.diffusion.models.sana_video import SanaVideoTransformer3DModel

    assert SanaVideoTransformer3DModel._sp_plan == {}


def test_dummy_run_num_frames_covers_max_sp_degree() -> None:
    """Engine warmup must produce enough latent frames for every allowed SP
    degree on both VAE variants (temporal /4 and /8); one pixel frame would
    fail the per-rank frame check at startup."""
    from vllm_omni.diffusion.models.sana_video import SanaImageToVideoPipeline, SanaVideoPipeline

    for temporal_scale in (4, 8):
        latent_frames = (SanaVideoPipeline.dummy_run_num_frames - 1) // temporal_scale + 1
        assert latent_frames >= 4
    assert SanaImageToVideoPipeline.dummy_run_num_frames == SanaVideoPipeline.dummy_run_num_frames
