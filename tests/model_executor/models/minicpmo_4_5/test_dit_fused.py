# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The fused Code2Wav DiT body computes the ragged body's function (``dit_fused.py``)."""

import gc

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import BatchedToken2Wav
from vllm_omni.model_executor.models.minicpmo_4_5.dit_fused import (
    blocks_forward_chunk_fused,
    dit_modulation,
    supports_fused_body,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _upstream_dit(device: str) -> nn.Module:
    """The shipped DiT architecture at toy width with non-trivial adaLN weights."""
    decoder_dit = pytest.importorskip("stepaudio2.cosyvoice2.flow.decoder_dit")
    torch.manual_seed(0)
    estimator = decoder_dit.DiT(in_channels=320, out_channels=80, depth=2, num_heads=4, head_dim=16, hidden_size=64)
    with torch.no_grad():
        # The adaLN-Zero init makes every block an identity; any weights will do here.
        for parameter in estimator.parameters():
            parameter.normal_(0.0, 0.08)
    return estimator.eval().to(device)


def _chunk(estimator: nn.Module, device: str, *, rows: int, frames: int, cached: int):
    torch.manual_seed(1)
    depth = len(estimator.blocks)
    channels = estimator.blocks[0].conv.block[1].in_channels
    attn = estimator.blocks[0].attn
    estimator_input = torch.randn(rows, 320, frames, device=device)
    time_embedding = estimator.t_embedder(torch.rand((), device=device).expand(rows)).unsqueeze(1)
    mask = torch.rand(rows, frames, cached + frames, device=device) > 0.3
    mask[..., 0] = True
    cnn = [torch.randn(rows, 2 * channels, 2, device=device) for _ in range(depth)]
    att = [torch.randn(rows, attn.num_heads, cached, 2 * attn.head_dim, device=device) for _ in range(depth)]
    return estimator_input, time_embedding, mask, cnn, att


def _run(body, estimator, chunk, lengths, **kwargs):
    """``body`` over ``chunk`` (``_chunk``'s tuple; ``None`` caches for a first chunk) into fresh output caches."""
    estimator_input, time_embedding, mask, cnn, att = chunk
    rows, _, frames = estimator_input.shape
    depth = len(estimator.blocks)
    attn = estimator.blocks[0].attn
    channels = estimator.blocks[0].conv.block[1].in_channels
    cached = 0 if att[0] is None else int(att[0].shape[2])
    cnn_out = torch.zeros((depth, rows, 2 * channels, 2), device=estimator_input.device)
    att_out = torch.zeros(
        (depth, rows, attn.num_heads, cached + frames, 2 * attn.head_dim), device=estimator_input.device
    )
    with torch.no_grad():
        out = body(estimator, estimator_input, time_embedding, mask, cnn, att, cnn_out, att_out, lengths, **kwargs)
    return out, cnn_out, att_out


def _solve_inputs(batch_size: int, width: int, seed: int) -> dict[str, torch.Tensor]:
    """One chunk's Whole-Euler solve inputs (``WholeEulerCFMGraphWrapper.replay``)."""
    torch.manual_seed(seed)
    return {
        "x": torch.randn(batch_size, 80, width, device="cuda"),
        "mu_cfg": torch.randn(2 * batch_size, 80, width, device="cuda"),
        "speakers_cfg": torch.randn(2 * batch_size, 80, device="cuda"),
        "cond_cfg": torch.randn(2 * batch_size, 80, width, device="cuda"),
    }


@pytest.fixture
def fp32_graphs(monkeypatch: pytest.MonkeyPatch) -> None:
    """A private graph pool, and fp32 cuDNN convolutions (they default to TF32).

    Both bodies then differ only in reduction order across the 10 Euler steps.
    """
    from vllm.platforms import current_platform

    pool = torch.cuda.graph_pool_handle()
    monkeypatch.setattr(current_platform, "get_global_graph_pool", lambda: pool)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)


@pytest.mark.parametrize("device", _DEVICES)
def test_fused_body_matches_ragged_body(device: str) -> None:
    estimator = _upstream_dit(device)
    assert supports_fused_body(estimator)
    chunk = _chunk(estimator, device, rows=4, frames=6, cached=5)
    lengths = [6, 4]

    expected = _run(BatchedToken2Wav._blocks_forward_chunk_ragged, estimator, chunk, lengths)
    # On CUDA the fused body attends with the tiled tf32x3 kernel, the reference with SDPA.
    actual = _run(blocks_forward_chunk_fused, estimator, chunk, lengths)
    for want, got in zip(expected, actual, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=2e-5)

    # A precomputed per-timestep table is the one computed on the fly.
    with torch.no_grad():
        table = dit_modulation(estimator, chunk[1][:1])
    tabled = _run(blocks_forward_chunk_fused, estimator, chunk, lengths, modulation=table)
    torch.testing.assert_close(tabled[0], actual[0], rtol=0, atol=0)


@pytest.mark.parametrize("device", _DEVICES)
def test_fused_body_first_chunk_without_caches(device: str) -> None:
    estimator = _upstream_dit(device)
    estimator_input, time_embedding, _, _, _ = _chunk(estimator, device, rows=2, frames=8, cached=0)
    depth = len(estimator.blocks)
    mask = torch.ones(2, 8, 8, dtype=torch.bool, device=device)
    chunk = (estimator_input, time_embedding, mask, [None] * depth, [None] * depth)
    expected = _run(BatchedToken2Wav._blocks_forward_chunk_ragged, estimator, chunk, [8])
    actual = _run(blocks_forward_chunk_fused, estimator, chunk, [8])
    for want, got in zip(expected, actual, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=2e-5)


def test_modulation_table_layout() -> None:
    estimator = _upstream_dit("cpu")
    embedding = estimator.t_embedder(torch.tensor([0.3])).unsqueeze(1)
    with torch.no_grad():
        table = dit_modulation(estimator, embedding)
        block = estimator.blocks[1].adaLN_modulation(embedding).reshape(9, -1)
        final_shift, final_scale = estimator.final_layer.adaLN_modulation(embedding).reshape(2, -1)
    depth = len(estimator.blocks)
    assert table.shape == (depth + 1, 9, 64)
    torch.testing.assert_close(table[1, 0], block[0])
    torch.testing.assert_close(table[1, 1], block[1] + 1)  # scale_msa -> 1 + scale
    torch.testing.assert_close(table[1, 2], block[2])  # gates stay as they are
    torch.testing.assert_close(table[depth, 0], final_shift)
    torch.testing.assert_close(table[depth, 1], final_scale + 1)


def test_unsupported_estimator_is_refused() -> None:
    estimator = _upstream_dit("cpu")
    estimator.blocks[0].attn.q_norm = nn.Identity()
    assert not supports_fused_body(estimator)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture")
def test_whole_euler_graph_with_fused_body_matches_ragged_body(fp32_graphs: None) -> None:
    """Captured with the arena's per-timestep modulation, the fused body replays the ragged body's solve."""
    from vllm_omni.model_executor.models.minicpmo_4_5.cuda_graph_wrapper import WholeEulerCFMGraphWrapper

    estimator = _upstream_dit("cuda")
    batch_size, width = 3, 8
    results = []
    for body, modulation_fn in (
        (BatchedToken2Wav._blocks_forward_chunk_ragged, None),
        (blocks_forward_chunk_fused, lambda embedding: dit_modulation(estimator, embedding)),
    ):
        wrapper = WholeEulerCFMGraphWrapper(
            estimator=estimator,
            n_timesteps=10,
            max_graphs=8,
            query_bucket_frames=16,
            ragged_body=body,
            modulation_fn=modulation_fn,
        )
        first = wrapper.replay(**_solve_inputs(batch_size, width, 2), cnn_cache=None, att_cache=None)
        assert first is not None and wrapper.stats_snapshot()["eager"] == 0
        second = wrapper.replay(
            **_solve_inputs(batch_size, width, 3), cnn_cache=first[1], att_cache=first[2], valid_lengths=[8, 5, 3]
        )
        assert second is not None
        results.append((first, second))
        if modulation_fn is not None:
            # One table per timestep, shared by every graph.
            assert wrapper.arena._modulation.shape[:2] == (10, len(estimator.blocks) + 1)

    for want, got in zip(results[0], results[1], strict=True):
        torch.testing.assert_close(got[0], want[0], rtol=1e-4, atol=2e-4)
        torch.testing.assert_close(got[1], want[1], rtol=1e-4, atol=2e-4)
        want_att = want[2] if isinstance(want[2], torch.Tensor) else torch.cat([a.flatten() for a in want[2]])
        got_att = got[2] if isinstance(got[2], torch.Tensor) else torch.cat([a.flatten() for a in got[2]])
        torch.testing.assert_close(got_att, want_att, rtol=1e-4, atol=2e-4)


def test_slot_pool_adopt_materialize_round_trip() -> None:
    from vllm_omni.model_executor.models.minicpmo_4_5.cuda_graph_wrapper import AttSlotPool

    pool = AttSlotPool(
        n_timesteps=2,
        depth=3,
        heads=2,
        width=4,
        slots=2,
        frames=16,
        suffix=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    cache = torch.randn(2, 3, 2, 2, 7, 4)
    first, second = pool.adopt(cache), pool.adopt(cache)
    assert first is not None and second is not None and pool.free_slots == 0
    assert pool.adopt(cache) is None
    # The suffix sits at the fixed frames, the rest in order behind it.
    assert first.physical_frames() == [3, 4, 5, 6, 0, 1, 2]
    assert first.shape == cache.shape
    torch.testing.assert_close(first.materialize(), cache, rtol=0, atol=0)
    del first
    gc.collect()
    assert pool.free_slots == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture")
def test_slot_pool_replay_matches_arena_replay(fp32_graphs: None) -> None:
    """Resident caches give the arena path's chunks, CNN caches and trimmed attention caches."""
    from vllm_omni.model_executor.models.minicpmo_4_5.cuda_graph_wrapper import (
        ResidentAttCache,
        WholeEulerCFMGraphWrapper,
        _materialize_att_rows,
    )

    estimator = _upstream_dit("cuda")
    depth, heads, att_width = (
        len(estimator.blocks),
        estimator.blocks[0].attn.num_heads,
        2 * estimator.blocks[0].attn.head_dim,
    )
    batch_size, width, prompt_len, suffix = 3, 8, 12, 6
    keep = (prompt_len, suffix)
    torch.manual_seed(5)
    # One shared prompt state, as ``setup_batch`` hands every fresh request.
    prompt = torch.randn(10, depth, 2, heads, prompt_len, att_width, device="cuda")
    cnn = torch.randn(10, depth, 2 * batch_size, 2 * 64, 2, device="cuda")
    lengths = [None, [8, 5, 3], None, [4, 8, 6]]
    results = []
    for slots in (0, 4):
        wrapper = WholeEulerCFMGraphWrapper(
            estimator=estimator,
            n_timesteps=10,
            max_graphs=16,
            query_bucket_frames=16,
            ragged_body=blocks_forward_chunk_fused,
            modulation_fn=lambda embedding: dit_modulation(estimator, embedding),
            att_slots=slots,
        )
        att: list = [prompt] * batch_size
        cnn_cache = cnn
        outputs = []
        for step, valid in enumerate(lengths):
            mask = None
            if valid is not None:
                # ``_decode_cfm``'s ragged mask: each row's own queries and current keys, all of its cache.
                offset = int(att[0].shape[4])
                cfg = torch.tensor(valid * 2, device="cuda")
                queries = torch.arange(width, device="cuda").unsqueeze(0) < cfg.unsqueeze(1)
                keys = torch.cat((queries, torch.ones(2 * batch_size, offset, dtype=torch.bool, device="cuda")), dim=1)
                mask = queries.unsqueeze(2) & keys.unsqueeze(1)
            out = wrapper.replay(
                **_solve_inputs(batch_size, width, 10 + step),
                cnn_cache=cnn_cache,
                att_cache=att,
                attn_mask=mask,
                valid_lengths=valid,
                att_keep=keep,
            )
            assert out is not None
            mel, cnn_cache, att = out
            if valid is not None:
                # Each row keeps its own frames; the next chunk needs one cache length again.
                shortest = min(int(row.shape[4]) for row in att)
                assert all(int(row.shape[4]) == shortest for row in att)
            outputs.append((mel, cnn_cache.clone(), [row.clone() for row in _materialize_att_rows(att)]))
        stats = wrapper.stats_snapshot()
        assert stats["eager"] == 0
        if slots:
            assert stats["slot_replays"] == len(lengths)
            assert all(isinstance(row, ResidentAttCache) for row in att)
            # Each request holds one slot; dropping its state frees it.
            assert wrapper.slot_pool.free_slots == slots - batch_size
            del att, out
            gc.collect()
            assert wrapper.slot_pool.free_slots == slots
        results.append(outputs)

    for step, (want, got) in enumerate(zip(results[0], results[1], strict=True)):

        def close(actual: torch.Tensor, expected: torch.Tensor, step: int = step) -> None:
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-4, msg=lambda text: f"chunk {step}: {text}")

        close(got[0], want[0])
        close(got[1], want[1])
        for want_att, got_att in zip(want[2], got[2], strict=True):
            assert got_att.shape == want_att.shape, f"chunk {step}"
            close(got_att, want_att)


def _cfg_rows(tensor: torch.Tensor, batch_size: int, rows: list[int]) -> torch.Tensor:
    """Requests ``rows`` of a CFG-stacked ``[cond x B | uncond x B]`` tensor (axis 0)."""
    return tensor[[*rows, *(batch_size + row for row in rows)]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture")
@pytest.mark.parametrize("grid", ["4", ""], ids=["one-graph-batch", "graph-grid"])
def test_slot_pool_replay_merges_rows_of_different_cache_lengths(
    fp32_graphs: None, monkeypatch: pytest.MonkeyPatch, grid: str
) -> None:
    """A stream's second chunk and steady chunks in one slot-pool replay (``row_offsets``).

    Each row attends its own slot frames and the columns past a shorter cache
    are masked, so on one graph batch the merged solve is bit-identical to
    the separate replays; across graph batches only the GEMM tiling differs.
    """
    from vllm_omni.model_executor.models.minicpmo_4_5.cuda_graph_wrapper import (
        WholeEulerCFMGraphWrapper,
        _materialize_att_rows,
    )

    monkeypatch.setenv("VLLM_OMNI_CFM_GRAPH_GRID", grid)
    estimator = _upstream_dit("cuda")
    depth, heads, att_width = (
        len(estimator.blocks),
        estimator.blocks[0].attn.num_heads,
        2 * estimator.blocks[0].attn.head_dim,
    )
    # A slot keeps the last ``suffix`` frames fixed, so the prompt holds at least that many.
    width, prompt_len, suffix = 8, 20, 16
    keep = (prompt_len, suffix)
    torch.manual_seed(5)
    prompt = torch.randn(10, depth, 2, heads, prompt_len, att_width, device="cuda")
    # Requests 0 and 1 run two chunks (steady cache 20 + 16 frames), request 2
    # one (second-chunk cache 20 + 8); then all three take a chunk together.
    cnn = [torch.randn(10, depth, 2, 2 * 64, 2, device="cuda") for _ in range(3)]
    final = _solve_inputs(3, width, 30)

    def stacked_cnn(rows: list) -> torch.Tensor:
        return torch.cat([*(row[:, :, 0:1] for row in rows), *(row[:, :, 1:2] for row in rows)], dim=2)

    def solve(wrapper, requests: list[int], att: list, cnn_rows: list, inputs: dict, **kwargs):
        out = wrapper.replay(**inputs, cnn_cache=stacked_cnn(cnn_rows), att_cache=att, att_keep=keep, **kwargs)
        assert out is not None
        mel, out_cnn, out_att = out
        batch_size = len(requests)
        return mel, [out_cnn[:, :, [row, batch_size + row]] for row in range(batch_size)], out_att

    def history(row_offsets: bool):
        wrapper = WholeEulerCFMGraphWrapper(
            estimator=estimator,
            n_timesteps=10,
            max_graphs=16,
            micro_batch_size=4,
            pad_max_rows=3,
            query_bucket_frames=16,
            ragged_body=blocks_forward_chunk_fused,
            modulation_fn=lambda embedding: dit_modulation(estimator, embedding),
            att_slots=4,
            row_offsets=row_offsets,
        )
        att: list = [prompt, prompt, prompt]
        cnn_rows = list(cnn)
        for step, requests in enumerate(([0, 1], [0, 1], [2])):
            inputs = _solve_inputs(len(requests), width, 10 + step)
            _, new_cnn, new_att = solve(
                wrapper, requests, [att[r] for r in requests], [cnn_rows[r] for r in requests], inputs
            )
            for request, row_cnn, row_att in zip(requests, new_cnn, new_att, strict=True):
                cnn_rows[request], att[request] = row_cnn, row_att
        assert [int(row.shape[4]) for row in att] == [prompt_len + suffix, prompt_len + suffix, prompt_len + width]
        return wrapper, att, cnn_rows

    def subset(rows: list[int]) -> dict:
        return {
            "x": final["x"][rows],
            **{name: _cfg_rows(final[name], 3, rows) for name in ("mu_cfg", "speakers_cfg", "cond_cfg")},
        }

    # Switch off: rows of different cache lengths never reach the slot pool.
    wrapper, att, cnn_rows = history(row_offsets=False)
    layouts = [list(row.layout) for row in att]
    assert (
        wrapper.replay(**final, cnn_cache=stacked_cnn(cnn_rows), att_cache=[att[2], att[0], att[1]], att_keep=keep)
        is None
    )
    assert [list(row.layout) for row in att] == layouts

    # Separate buckets: the second chunk alone, then the steady pair.
    split_mel, split_cnn, split_att = {}, {}, {}
    for requests in ([2], [0, 1]):
        rows = [{2: 0, 0: 1, 1: 2}[r] for r in requests]
        mel, new_cnn, new_att = solve(
            wrapper, requests, [att[r] for r in requests], [cnn_rows[r] for r in requests], subset(rows)
        )
        for index, request in enumerate(requests):
            split_mel[request], split_cnn[request] = mel[index], new_cnn[index]
            split_att[request] = _materialize_att_rows([new_att[index]])[0].clone()

    # One replay: rows [second chunk, steady, steady] with ``_decode_cfm``'s row-offset mask.
    merged_wrapper, merged_att, merged_cnn_rows = history(row_offsets=True)
    order = [2, 0, 1]
    offsets = [int(merged_att[r].shape[4]) for r in order]
    lengths = torch.tensor([width] * 6, device="cuda")
    queries = torch.arange(width, device="cuda").unsqueeze(0) < lengths.unsqueeze(1)
    cached = torch.arange(max(offsets), device="cuda").unsqueeze(0) < torch.tensor(
        offsets * 2, device="cuda"
    ).unsqueeze(1)
    mask = queries.unsqueeze(2) & torch.cat((queries, cached), dim=1).unsqueeze(1)
    replays = merged_wrapper.stats_snapshot()["slot_replays"]
    mel, new_cnn, new_att = solve(
        merged_wrapper,
        order,
        [merged_att[r] for r in order],
        [merged_cnn_rows[r] for r in order],
        final,
        attn_mask=mask,
        valid_lengths=[width] * 3,
    )
    stats = merged_wrapper.stats_snapshot()
    assert stats["slot_replays"] == replays + 1 and stats["eager"] == 0
    # Each row keeps its own cache length and slot layout.
    assert [int(row.shape[4]) for row in new_att] == [prompt_len + suffix] * 3
    assert [row.layout for row in new_att] == [att[r].layout for r in order]

    exact = grid == "4"
    for index, request in enumerate(order):

        def close(actual: torch.Tensor, expected: torch.Tensor, what: str, request: int = request) -> None:
            if exact:
                assert torch.equal(actual, expected), f"request {request} {what}"
            else:
                torch.testing.assert_close(
                    actual, expected, rtol=1e-4, atol=2e-4, msg=lambda t: f"{request} {what}: {t}"
                )

        close(mel[index], split_mel[request], "mel")
        close(new_cnn[index], split_cnn[request], "cnn")
        close(_materialize_att_rows([new_att[index]])[0], split_att[request], "att")
