# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for the pinned tiled-VAE gather buffer in the MiniMax-H3 video VAE.

The checkpoint's ``_all_gather_tiled_results`` allocates its gather output on
every call, so with a per-request ``empty_cache()`` the collective lands on a
new device address every request and XCCL's per-address registrations pile up.
On XPU the adapter replaces that method with an equal-shape gather into a
buffer it keeps alive. These tests pin the properties that make it a fix
rather than a rewrite: the address is stable, the tiles come back
bit-identical and in the same order, the caller owns what it gets back, and no
other device is touched.
"""

import logging
import sys
import types
from typing import Any

import pytest
import torch
import torch.distributed as dist

from vllm_omni.diffusion.models.minimax_h3 import vae as vae_module
from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

TILE = (1, 3, 2, 4, 4)


class _CheckpointGatherSentinel:
    """Stands in for the checkpoint's own bound method.

    Identity is the point: a test asserts this exact object is still installed
    on the devices the adapter must leave alone.
    """

    def __call__(self, tasks, num_tiles):  # pragma: no cover - never invoked
        raise AssertionError("the checkpoint gather must not run in these tests")


class _FakeCheckpointModel:
    def __init__(self):
        self._all_gather_tiled_results = _CheckpointGatherSentinel()


def _local_tiles(all_tiles, num_tiles, rank, sp_size):
    """One rank's round-robin share, as ``klvae.py:275`` computes it."""
    return [all_tiles[index] for index in range(rank, num_tiles, sp_size)]


def _reference_gather(all_tiles, num_tiles, sp_size):
    """The checkpoint's variable-shape list gather, reimplemented verbatim.

    Written from the control flow of ``_all_gather_tiled_results`` in the
    checkpoint's ``klvae.py`` (lines 252-273) -- stack each rank's share, gather
    the list, walk it back into global tile order -- so that it is an
    independent oracle rather than a restatement of the code under test.
    """
    gathered = [torch.stack(_local_tiles(all_tiles, num_tiles, rank, sp_size), dim=0) for rank in range(sp_size)]
    results = [None] * num_tiles
    for rank, rank_tensors in enumerate(gathered):
        for k in range(rank_tensors.shape[0]):
            global_index = k * sp_size + rank
            if global_index >= num_tiles:
                break
            results[global_index] = rank_tensors[k]
    return results


def _peer_stacks(all_tiles, num_tiles, sp_size):
    """What every rank hands to the collective, padded the way the adapter pads."""
    max_tasks = -(-num_tiles // sp_size)
    stacks = []
    for rank in range(sp_size):
        share = _local_tiles(all_tiles, num_tiles, rank, sp_size)
        stacked = share[0].new_zeros((max_tasks, *share[0].shape))
        stacked[: len(share)] = torch.stack(share, dim=0)
        stacks.append(stacked)
    return stacks


def _tiles(num_tiles, seed=0, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    return [torch.rand(TILE, generator=generator).to(dtype) for _ in range(num_tiles)]


def _workspace_bytes(num_tiles, sp_size, dtype=torch.float32):
    """Bytes one gather needs: ``sp_size`` copies of the padded local stack."""
    max_tasks = -(-num_tiles // sp_size)
    element_size = torch.empty(0, dtype=dtype).element_size()
    numel = 1
    for dim in TILE:
        numel *= dim
    return sp_size * max_tasks * numel * element_size


@pytest.fixture
def fake_collective(monkeypatch):
    """Replace the collective with a single-process stand-in.

    ``all_gather_into_tensor`` is the only distributed call on this path, so
    filling the output buffer from a caller-supplied peer list exercises the
    real padding, keying, addressing and unpacking code without a process
    group.
    """
    calls: list[dict[str, Any]] = []

    def fake_all_gather_into_tensor(output, input_tensor, group=None, async_op=False):
        peers = calls[-1]["peers"]
        assert output.shape == (len(peers), *input_tensor.shape)
        calls[-1]["sent"] = input_tensor.clone()
        calls[-1]["buf_ptr"] = output.data_ptr()
        for rank, peer in enumerate(peers):
            output[rank].copy_(peer)

    monkeypatch.setattr(dist, "all_gather_into_tensor", fake_all_gather_into_tensor)
    return calls


class _Records(logging.Handler):
    """vllm's logger sets ``propagate=False``, so caplog never sees these."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.messages = []
        self.records = []

    def emit(self, record):
        self.messages.append(record.getMessage())
        self.records.append((record.levelno, record.getMessage()))


@pytest.fixture
def gather_log():
    handler = _Records()
    logger = vae_module.logger
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


def _vae(sp_size, sp_rank=0):
    """An adapter instance with the override installed, no checkpoint needed."""
    state = {
        "sp_size": sp_size,
        "sp_rank": sp_rank,
        "sp_enabled": sp_size > 1,
        "sp_process_group": object(),
    }
    module = types.ModuleType("fake_gather_ckpt.parallel")
    module.get_parallel_state = lambda: state  # type: ignore[attr-defined]
    sys.modules.setdefault("fake_gather_ckpt", types.ModuleType("fake_gather_ckpt"))
    sys.modules["fake_gather_ckpt.parallel"] = module

    vae = object.__new__(MiniMaxH3VideoVAE)
    vae.remote = type("Remote", (), {"__module__": "fake_gather_ckpt.klvae"})()
    vae.model = _FakeCheckpointModel()
    vae._device_target = torch.device("xpu")
    vae._tile_gather_workspace = None
    vae._tile_gather_stats = {"hits": 0, "allocs": 0, "workspace_bytes": 0}
    vae._checkpoint_tile_gather = None
    # What decode_latent records from the grid geometry before dispatching
    # tiles; these tests exercise the equal-shape protocol directly.
    vae._tile_gather_grid_uniform = True
    vae._tile_gather_expected_hw = TILE[-2:]
    vae._install_persistent_tile_gather()
    return vae


def _run(vae, calls, all_tiles, num_tiles, sp_size, sp_rank):
    calls.append({"peers": _peer_stacks(all_tiles, num_tiles, sp_size)})
    return vae.model._all_gather_tiled_results(_local_tiles(all_tiles, num_tiles, sp_rank, sp_size), num_tiles)


# --------------------------------------------------------------------------
# The address is stable across calls. This is what the accumulation is about:
# drop the workspace and every call registers a new receive address. And the
# workspace is one grow-only block, so varying request geometries cannot keep
# adding resident buffers the way a per-geometry cache would.
# --------------------------------------------------------------------------


def test_the_landing_buffer_address_is_stable_across_calls(fake_collective):
    vae = _vae(4)
    pointers = []
    for request in range(3):
        _run(vae, fake_collective, _tiles(8, seed=request), 8, 4, 0)
        pointers.append(fake_collective[-1]["buf_ptr"])

    assert len(set(pointers)) == 1, f"gather buffer moved across calls: {pointers}"
    assert vae._tile_gather_stats == {
        "hits": 2,
        "allocs": 1,
        "workspace_bytes": _workspace_bytes(8, 4),
    }
    assert vae._tile_gather_workspace.numel() == _workspace_bytes(8, 4)
    assert vae._tile_gather_workspace.dtype is torch.uint8


def test_distinct_shapes_share_the_one_workspace(fake_collective):
    """One decode gathers several temporal clips, and the head/tail clips differ."""
    vae = _vae(4)
    _run(vae, fake_collective, _tiles(8), 8, 4, 0)
    first = fake_collective[-1]["buf_ptr"]
    _run(vae, fake_collective, _tiles(12), 12, 4, 0)
    second = fake_collective[-1]["buf_ptr"]
    _run(vae, fake_collective, _tiles(8, seed=5), 8, 4, 0)
    third = fake_collective[-1]["buf_ptr"]

    # The larger geometry did not fit, so the block was replaced -- not added to.
    assert second == third
    assert vae._tile_gather_stats["allocs"] == 2
    assert vae._tile_gather_stats["hits"] == 1
    assert vae._tile_gather_stats["workspace_bytes"] == _workspace_bytes(12, 4)
    # ``first`` is deliberately not compared: the replaced block is freed, so
    # the allocator is free to hand its address back for the larger one.
    assert isinstance(first, int)


def test_workspace_is_bounded_across_geometries(fake_collective):
    """Alternating geometries must not each retain a buffer of their own."""
    sp_size = 4
    large, small, largest = 12, 8, 20
    vae = _vae(sp_size)

    pointers = []
    for round_index in range(3):
        for num_tiles in (large, small):
            _run(vae, fake_collective, _tiles(num_tiles, seed=round_index), num_tiles, sp_size, 0)
            pointers.append(fake_collective[-1]["buf_ptr"])

    assert len(pointers) == 6
    assert len(set(pointers)) == 1, f"workspace moved across geometries: {pointers}"
    assert vae._tile_gather_stats["allocs"] == 1
    assert vae._tile_gather_stats["workspace_bytes"] == _workspace_bytes(large, sp_size)
    assert vae._tile_gather_workspace.numel() == _workspace_bytes(large, sp_size)

    # A geometry that does not fit grows the single block; it does not add one.
    _run(vae, fake_collective, _tiles(largest), largest, sp_size, 0)
    grown = fake_collective[-1]["buf_ptr"]
    assert vae._tile_gather_stats["allocs"] == 2
    assert vae._tile_gather_stats["workspace_bytes"] == _workspace_bytes(largest, sp_size)
    assert vae._tile_gather_workspace.numel() == _workspace_bytes(largest, sp_size)

    # ...and the geometries seen before it now fit, so nothing reallocates again.
    for num_tiles in (large, small):
        _run(vae, fake_collective, _tiles(num_tiles, seed=7), num_tiles, sp_size, 0)
        assert fake_collective[-1]["buf_ptr"] == grown
    assert vae._tile_gather_stats["allocs"] == 2
    assert vae._tile_gather_stats["workspace_bytes"] == _workspace_bytes(largest, sp_size)


def test_workspace_grows_only_upward(fake_collective):
    """Resident bytes track the largest geometry seen, not the sum of them."""
    sp_size = 4
    small, large = 8, 12
    vae = _vae(sp_size)

    _run(vae, fake_collective, _tiles(small), small, sp_size, 0)
    assert vae._tile_gather_stats["workspace_bytes"] == _workspace_bytes(small, sp_size)

    _run(vae, fake_collective, _tiles(large), large, sp_size, 0)

    assert vae._tile_gather_stats["allocs"] == 2
    assert vae._tile_gather_stats["hits"] == 0
    assert vae._tile_gather_stats["workspace_bytes"] == _workspace_bytes(large, sp_size)
    assert vae._tile_gather_stats["workspace_bytes"] < _workspace_bytes(small, sp_size) + _workspace_bytes(
        large, sp_size
    )
    assert vae._tile_gather_workspace.numel() == _workspace_bytes(large, sp_size)


def test_mixed_dtype_shares_one_workspace(fake_collective):
    """The workspace is bytes, so a narrower dtype reuses it instead of adding one."""
    sp_size, num_tiles = 4, 8
    vae = _vae(sp_size)
    fp32_bytes = _workspace_bytes(num_tiles, sp_size, dtype=torch.float32)

    pointers = []
    for round_index in range(2):
        for dtype in (torch.float32, torch.bfloat16):
            tiles = _tiles(num_tiles, seed=round_index, dtype=dtype)
            produced = _run(vae, fake_collective, tiles, num_tiles, sp_size, 0)
            pointers.append(fake_collective[-1]["buf_ptr"])
            assert all(tile.dtype is dtype for tile in produced)
            for index, tile in enumerate(produced):
                assert torch.equal(tile, tiles[index])

    assert len(set(pointers)) == 1, f"workspace moved across dtypes: {pointers}"
    assert vae._tile_gather_stats["allocs"] == 1
    assert vae._tile_gather_stats["hits"] == 3
    assert vae._tile_gather_stats["workspace_bytes"] == fp32_bytes
    assert vae._tile_gather_workspace.numel() == fp32_bytes


# --------------------------------------------------------------------------
# Equivalence with the checkpoint's own gather, padded branch included. A
# reordered tile grid is silent at runtime, so order is asserted as well.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("num_tiles", [5, 8, 9, 28])
@pytest.mark.parametrize("sp_rank", [0, 3])
def test_gathered_tiles_match_the_checkpoint_gather(fake_collective, num_tiles, sp_rank):
    sp_size = 4
    all_tiles = _tiles(num_tiles, seed=num_tiles)
    vae = _vae(sp_size, sp_rank=sp_rank)

    produced = _run(vae, fake_collective, all_tiles, num_tiles, sp_size, sp_rank)
    expected = _reference_gather(all_tiles, num_tiles, sp_size)

    assert len(produced) == num_tiles
    assert all(tile is not None for tile in produced)
    for index, (got, want) in enumerate(zip(produced, expected)):
        assert torch.equal(got, want), f"tile {index} differs from the checkpoint gather"
    # ...and it is the tile that belongs at that index, not a permutation that
    # happens to agree with an equally permuted oracle.
    for index, tile in enumerate(produced):
        assert torch.equal(tile, all_tiles[index])


def test_the_padded_branch_is_actually_exercised(fake_collective):
    """9 tiles over 4 ranks: ranks 1-3 are one short, so the padding runs."""
    sp_size, num_tiles = 4, 9
    all_tiles = _tiles(num_tiles, seed=9)
    assert len(_local_tiles(all_tiles, num_tiles, 1, sp_size)) < -(-num_tiles // sp_size)

    vae = _vae(sp_size, sp_rank=1)
    produced = _run(vae, fake_collective, all_tiles, num_tiles, sp_size, 1)

    sent = fake_collective[-1]["sent"]
    assert sent.shape[0] == 3
    assert torch.equal(sent[2], torch.zeros_like(sent[2]))
    for index, tile in enumerate(produced):
        assert torch.equal(tile, all_tiles[index])


def test_an_empty_local_share_still_raises(fake_collective):
    """A rank with no tiles fails loudly, exactly as the checkpoint does."""
    vae = _vae(4)
    with pytest.raises(ValueError, match="empty tasks on sp rank"):
        vae.model._all_gather_tiled_results([], 8)


def test_more_tiles_than_round_robin_allows_is_refused(fake_collective):
    """The unpacking arithmetic is only valid for the checkpoint's ownership."""
    vae = _vae(4)
    with pytest.raises(ValueError, match="round-robin"):
        vae.model._all_gather_tiled_results(_tiles(5), 8)


# --------------------------------------------------------------------------
# Ownership: the buffer is overwritten by the next gather, so returned tiles
# must not alias it.
# --------------------------------------------------------------------------


def test_returned_tiles_do_not_alias_the_pinned_buffer(fake_collective):
    sp_size, num_tiles = 4, 8
    first_tiles = _tiles(num_tiles, seed=1)
    vae = _vae(sp_size)
    produced = _run(vae, fake_collective, first_tiles, num_tiles, sp_size, 0)

    buffer_storage = vae._tile_gather_workspace.untyped_storage().data_ptr()
    for tile in produced:
        assert tile.untyped_storage().data_ptr() != buffer_storage

    # The next gather reuses that memory; the first call's tiles must survive it.
    _run(vae, fake_collective, _tiles(num_tiles, seed=2), num_tiles, sp_size, 0)
    for index, tile in enumerate(produced):
        assert torch.equal(tile, first_tiles[index])


# --------------------------------------------------------------------------
# Non-XPU behaviour is unchanged: the checkpoint's own method stays installed.
# --------------------------------------------------------------------------


def _construct(monkeypatch, device, model):
    """Run the real ``__init__`` against a stub checkpoint component."""

    class _Remote(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.model = model

    monkeypatch.setattr(
        vae_module,
        "_load_component_config",
        lambda path: {"latent_channels": 1, "latents_mean": [0.0], "latents_std": [1.0]},
    )
    # **kwargs: the loader gained a trust_remote_code argument at some point,
    # and this stub does not care either way.
    monkeypatch.setattr(vae_module, "_load_remote_component", lambda path, config, **kwargs: _Remote())
    # The adapter only stores this handle; resolving it for real needs an
    # accelerator, which a CPU test runner does not have.
    monkeypatch.setattr(torch, "get_device_module", lambda *args, **kwargs: types.SimpleNamespace())
    return MiniMaxH3VideoVAE("unused", device=torch.device(device))


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_non_xpu_devices_keep_the_checkpoint_gather(monkeypatch, device):
    model = _FakeCheckpointModel()
    original = model._all_gather_tiled_results

    vae = _construct(monkeypatch, device, model)

    assert vae._tile_gather_reuse_enabled() is False
    assert vae.model._all_gather_tiled_results is original
    assert vae._checkpoint_tile_gather is None


@pytest.mark.parametrize(
    ("device", "enabled"),
    [("xpu", True), ("cpu", False), ("cuda", False), ("meta", False)],
)
def test_only_xpu_asks_for_the_pinned_buffer(device, enabled):
    vae = object.__new__(MiniMaxH3VideoVAE)
    vae._device_target = torch.device(device)
    assert vae._tile_gather_reuse_enabled() is enabled


def test_construction_installs_the_override_when_the_device_asks_for_it(monkeypatch):
    """Close the loop: ``__init__`` must consult the predicate, not ignore it."""
    monkeypatch.setattr(MiniMaxH3VideoVAE, "_tile_gather_reuse_enabled", lambda self: True)
    model = _FakeCheckpointModel()
    original = model._all_gather_tiled_results

    vae = _construct(monkeypatch, "cpu", model)

    assert vae.model._all_gather_tiled_results is not original
    assert vae._checkpoint_tile_gather is original


# --------------------------------------------------------------------------
# The decision-point receipt carries quantities, not just presence: a line
# that only says "installed" cannot tell reuse from reallocation.
# --------------------------------------------------------------------------


def test_the_receipt_line_reports_hits_allocs_and_the_address(fake_collective, gather_log):
    vae = _vae(4)
    install = [
        (level, line) for level, line in gather_log.records if "persistent tile gather installed device=xpu" in line
    ]
    # The one-shot install line is what an operator needs at info level.
    assert install and all(level == logging.INFO for level, _ in install)
    gather_log.messages.clear()
    gather_log.records.clear()

    _run(vae, fake_collective, _tiles(8), 8, 4, 0)
    first_ptr = fake_collective[-1]["buf_ptr"]
    _run(vae, fake_collective, _tiles(8, seed=3), 8, 4, 0)

    records = [(level, line) for level, line in gather_log.records if "[H3_VAE_GATHER] reuse=" in line]
    # The per-gather line fires once per decoder tile batch, so it is debug.
    assert all(level == logging.DEBUG for level, _ in records)
    lines = [line for _, line in records]
    assert len(lines) == 2
    assert "reuse=alloc" in lines[0] and "hits=0 allocs=1" in lines[0]
    assert "reuse=hit" in lines[1] and "hits=1 allocs=1" in lines[1]
    # Resident bytes are reported, not just the bytes this one gather needed.
    needed = _workspace_bytes(8, 4) / (1024.0 * 1024.0)
    for line in lines:
        assert f"need_mib={needed:.2f}" in line
        assert f"workspace_mib={needed:.2f}" in line
    assert f"buf_ptr=0x{first_ptr:x}" in lines[0]
    assert f"buf_ptr=0x{first_ptr:x}" in lines[1]


# --------------------------------------------------------------------------
# Protocol selection is never rank-local. decode_latent records one
# geometry-derived decision that every rank resolves identically; the gather
# reads only that. A rank-local choice would split the ranks between the
# equal-shape collective and the checkpoint's variable-shape gather (only the
# ranks holding edge tiles would fall back), mismatching or hanging the group.
# --------------------------------------------------------------------------


def _install_checkpoint_gather(vae, sp_rank, sp_size, collective_entries):
    """Install the checkpoint's real ``_all_gather_tiled_results`` packing.

    Transcribed from the checkpoint's ``klvae.py`` ``_all_gather_tiled_results``
    (lines 252-273): the local share is ``torch.stack``-ed *before* the
    variable-shape collective, then walked back into global tile order. Only
    ``all_gather_var_shape`` itself is replaced -- by a recorder that notes
    this rank entered the variable-shape collective and echoes the local stack
    (peer ranks are out of process) -- so these tests exercise the
    checkpoint's real local packing step instead of an always-succeeding stub.
    """

    def checkpoint_gather(tasks, num_tiles):
        if not tasks:
            raise ValueError(f"Found empty tasks on sp rank {sp_rank}")
        stacked = torch.stack(tasks, dim=0)  # the checkpoint's local packing step
        collective_entries.append((sp_rank, tuple(stacked.shape)))
        gathered = [stacked]
        results = [None] * num_tiles
        for rank, rank_tensors in enumerate(gathered):
            for k in range(rank_tensors.shape[0]):
                global_index = k * sp_size + rank
                if global_index >= num_tiles:
                    break
                results[global_index] = rank_tensors[k]
        return results

    vae._checkpoint_tile_gather = checkpoint_gather


def test_no_recorded_uniform_grid_sends_every_rank_to_the_checkpoint_gather(monkeypatch):
    """Counterexample 1: rank 0 holds mixed shapes, rank 1 a uniform pair.

    The grid producing them is not equal-shape, so geometry records False on
    every rank and both ranks take the checkpoint's variable-shape gather --
    including the rank whose local share looks uniform. The checkpoint's own
    packing then ``torch.stack``s the local share, which is where a rank-local
    mix of shapes fails: a pre-existing checkpoint limitation, pinned here as
    such (rank 1 has already entered the variable-shape collective when
    rank 0's stack raises), not a success this branch claims to provide.
    """
    tile_a, tile_b = torch.rand(TILE), torch.rand((1, 3, 2, 4, 6))
    entries: list[tuple] = []
    equal_shape_calls: list[tuple] = []
    monkeypatch.setattr(dist, "all_gather_into_tensor", lambda *a, **k: equal_shape_calls.append(a))

    # rank 1, locally uniform share: must leave the equal-shape gather and
    # enter the checkpoint's variable-shape collective.
    vae = _vae(2, sp_rank=1)
    vae._tile_gather_grid_uniform = False
    vae._tile_gather_expected_hw = None
    _install_checkpoint_gather(vae, 1, 2, entries)
    vae.model._all_gather_tiled_results([tile_a, tile_a], 4)
    assert entries == [(1, (2, 1, 3, 2, 4, 4))], "rank 1 left the common protocol"

    # rank 0, mixed local share: also routed to the checkpoint gather, whose
    # local stack cannot pack mixed shapes (known checkpoint limitation).
    vae = _vae(2, sp_rank=0)
    vae._tile_gather_grid_uniform = False
    vae._tile_gather_expected_hw = None
    _install_checkpoint_gather(vae, 0, 2, entries)
    with pytest.raises(RuntimeError):
        vae.model._all_gather_tiled_results([tile_a, tile_b], 4)
    assert len(entries) == 1, "rank 0 must fail in the checkpoint's local stack, before its collective entry"
    assert not equal_shape_calls, "no rank may enter the equal-shape collective"


def test_locally_uniform_but_globally_unequal_shares_also_agree_on_the_protocol(monkeypatch):
    """Counterexample 2: every rank's share is uniform, but the payloads differ
    across ranks. Rank-local checks pass everywhere, so only the shared
    geometry decision keeps the ranks in one collective: both must reach the
    checkpoint's variable-shape collective through its real packing step, and
    neither may enter the equal-shape collective."""
    per_rank_tasks = {
        0: [torch.rand(TILE), torch.rand(TILE)],
        1: [torch.rand((1, 3, 2, 4, 6)), torch.rand((1, 3, 2, 4, 6))],
    }
    equal_shape_calls: list[tuple] = []
    monkeypatch.setattr(
        dist,
        "all_gather_into_tensor",
        lambda output, input_tensor, group=None, async_op=False: equal_shape_calls.append(tuple(input_tensor.shape)),
    )
    entries: list[tuple] = []
    for rank, tasks in per_rank_tasks.items():
        vae = _vae(2, sp_rank=rank)
        vae._tile_gather_grid_uniform = False
        vae._tile_gather_expected_hw = None
        _install_checkpoint_gather(vae, rank, 2, entries)
        vae.model._all_gather_tiled_results(tasks, 4)

    assert entries == [
        (0, (2, 1, 3, 2, 4, 4)),
        (1, (2, 1, 3, 2, 4, 6)),
    ], "a rank left the common protocol"
    assert not equal_shape_calls, "no rank may enter the equal-shape collective"


def test_a_wrong_uniformity_prediction_raises_before_the_collective(monkeypatch):
    """Geometry said equal shapes but this rank's share is mixed: report the
    violated prediction on the spot. Peer ranks may already be waiting in the
    collective; the executor tears the group down."""
    vae = _vae(2)
    entries: list[tuple] = []
    _install_checkpoint_gather(vae, 0, 2, entries)
    equal_shape_calls: list[tuple] = []
    monkeypatch.setattr(dist, "all_gather_into_tensor", lambda *a, **k: equal_shape_calls.append(a))
    tasks = [torch.rand(TILE), torch.rand((1, 3, 2, 4, 6))]

    with pytest.raises(ValueError, match="equal-shape decoder tile grid"):
        vae.model._all_gather_tiled_results(tasks, 4)

    assert not entries, "must not silently fall back on a rank-local observation"
    assert not equal_shape_calls, "must not carry a violated prediction into the collective"


def test_a_locally_uniform_share_violating_the_predicted_shape_raises(monkeypatch):
    """Geometry predicted (4, 4) tiles but this rank's share is uniformly
    (..., 4, 6). A check of the share only against its own first tile passes
    and carries the wrong payload into the collective; checking against the
    geometry-derived expected shape catches the cross-rank inconsistency on
    the rank that holds it."""
    vae = _vae(2, sp_rank=1)
    entries: list[tuple] = []
    _install_checkpoint_gather(vae, 1, 2, entries)
    equal_shape_calls: list[tuple] = []
    monkeypatch.setattr(dist, "all_gather_into_tensor", lambda *a, **k: equal_shape_calls.append(a))
    tasks = [torch.rand((1, 3, 2, 4, 6)), torch.rand((1, 3, 2, 4, 6))]

    with pytest.raises(ValueError, match="equal-shape decoder tile grid"):
        vae.model._all_gather_tiled_results(tasks, 4)

    assert not entries and not equal_shape_calls


def test_decode_latent_records_the_geometry_decision_and_clears_it_after():
    """The flag and expected tile shape the gather reads are written per decode
    from the grid geometry (via _decoder_tile_grid_hw) and cleared once the
    decode returns."""
    vae = _vae(2)
    observed = {}

    def decode_base(sample):
        observed["during"] = vae._tile_gather_grid_uniform
        observed["during_hw"] = vae._tile_gather_expected_hw
        return torch.zeros(1, 3, 2, 4, 4)

    vae.config_dict = {"latent_channels": 4, "latents_mean": [0.0] * 4, "latents_std": [1.0] * 4}
    vae.parallel_size = 1
    vae.model.decode_base = decode_base
    vae.model.processor = types.SimpleNamespace(revert_tensor=lambda decoded: decoded)
    for uniform in (True, False):
        vae.model.vae_ratio = 1
        vae.model.split_tiles = lambda length, is_decoder, u=uniform: (
            ([0, 1], [3, 3], [2]) if u else ([0, 1], [3, 1], [2])
        )
        vae._tile_gather_grid_uniform = None  # sentinel: decode must overwrite it
        vae._tile_gather_expected_hw = (9, 9)  # sentinel: decode must overwrite it

        MiniMaxH3VideoVAE.decode_latent(vae, torch.zeros(1, 4, 2, 4, 4))

        assert observed["during"] is uniform
        assert observed["during_hw"] == ((3, 3) if uniform else None)
        assert vae._tile_gather_grid_uniform is False, "decision must not outlive the decode"
        assert vae._tile_gather_expected_hw is None, "expected shape must not outlive the decode"
