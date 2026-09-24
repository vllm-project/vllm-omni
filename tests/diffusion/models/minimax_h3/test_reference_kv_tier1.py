# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3.reference_kv_tier1 import (
    MiniMaxH3ReferenceKVObserverState,
    MiniMaxH3ReferenceKVTier1State,
    MiniMaxH3ReferenceKVTier2State,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _kv(layer: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.arange(4 * 2 * 3, dtype=torch.float32).view(4, 2, 3)
    return values + base + layer * 100, values + base + layer * 100 + 50


def test_tier1_caches_global_target_positions_for_compact_reuse() -> None:
    state = MiniMaxH3ReferenceKVTier1State(
        num_layers=1,
        global_reference_rows=2,
        device=torch.device("cpu"),
        refresh_interval=2,
        pin_memory=False,
    )

    state.set_global_reference_mask(torch.tensor([True, False, False, True, False]))
    assert state.global_non_reference_positions.tolist() == [1, 2, 4]
    rope_table = torch.zeros(3, 6, dtype=torch.bfloat16)
    assert state.cache_compact_rope_table(rope_table) is rope_table
    assert state.compact_rope_table is rope_table

    state.close()
    assert state.compact_rope_table is None
    try:
        state.global_non_reference_positions
    except RuntimeError as exc:
        assert "was not initialized" in str(exc)
    else:
        raise AssertionError("close must release cached global positions")


def test_tier1_populates_host_and_bulk_stages_once_per_reuse_step() -> None:
    state = MiniMaxH3ReferenceKVTier1State(
        num_layers=2,
        global_reference_rows=2,
        device=torch.device("cpu"),
        refresh_interval=2,
        pin_memory=False,
        skip_reference_projection=True,
    )
    mask = torch.tensor([False, True, False, True])

    captured: list[tuple[torch.Tensor, torch.Tensor]] = []
    state.begin_step(0)
    state.set_local_reference_mask(mask)
    for layer in range(2):
        k, v = _kv(layer, 0)
        captured.append((k[mask].clone(), v[mask].clone()))
        out_k, out_v = state.process_layer(layer, k, v)
        assert out_k.data_ptr() == k.data_ptr()
        assert out_v.data_ptr() == v.data_ptr()
    state.end_step()

    assert state.host_pool is not None
    assert state.host_pool.is_contiguous()
    assert tuple(state.host_pool.shape) == (2, 2, 2, 2, 3)
    assert state.device_pool is None

    state.begin_step(1)
    assert state.device_pool is not None
    assert state.should_skip_reference_projection
    assert state.local_non_reference_positions.tolist() == [0, 2]
    for layer in range(2):
        state.record_projection_skip(layer, total_rows=4)
        k, v = _kv(layer, 1000)
        current_k = k.clone()
        current_v = v.clone()
        out_k, out_v = state.process_layer(layer, k, v)
        torch.testing.assert_close(out_k[mask], captured[layer][0])
        torch.testing.assert_close(out_v[mask], captured[layer][1])
        torch.testing.assert_close(out_k[~mask], current_k[~mask])
        torch.testing.assert_close(out_v[~mask], current_v[~mask])
    state.end_step()

    assert state.device_pool is None
    assert state.stats.refresh_steps == 1
    assert state.stats.h2d_steps == 1
    assert state.stats.captured_layers == 2
    assert state.stats.substituted_layers == 2
    assert state.stats.h2d_bytes == state.host_bytes
    assert state.stats.projection_skipped_layers == 2
    assert state.stats.projection_skipped_rows == 4
    assert state.stats.projection_total_rows == 8


def test_tier1_interval_zero_is_strict_populate_once() -> None:
    state = MiniMaxH3ReferenceKVTier1State(
        num_layers=1,
        global_reference_rows=1,
        device=torch.device("cpu"),
        refresh_interval=0,
        pin_memory=False,
    )
    mask = torch.tensor([True, False])

    state.begin_step(0)
    state.set_local_reference_mask(mask)
    k0 = torch.tensor([[[1.0]], [[2.0]]])
    v0 = torch.tensor([[[3.0]], [[4.0]]])
    state.process_layer(0, k0, v0)
    state.end_step()

    for step in (1, 2):
        state.begin_step(step)
        k = torch.full_like(k0, 10.0 + step)
        v = torch.full_like(v0, 20.0 + step)
        state.process_layer(0, k, v)
        assert k[0].item() == 1.0
        assert v[0].item() == 3.0
        state.end_step()

    assert state.stats.refresh_steps == 1
    assert state.stats.h2d_steps == 2
    assert state.stats.captured_layers == 1
    assert state.stats.substituted_layers == 2


def test_tier2_layerwise_ring_reuses_cached_reference_rows() -> None:
    state = MiniMaxH3ReferenceKVTier2State(
        num_layers=4,
        global_reference_rows=2,
        device=torch.device("cpu"),
        refresh_interval=2,
        ring_size=2,
        pin_memory=False,
        skip_reference_projection=True,
    )
    mask = torch.tensor([False, True, False, True])

    captured: list[tuple[torch.Tensor, torch.Tensor]] = []
    state.begin_step(0)
    state.set_local_reference_mask(mask)
    for layer in range(4):
        k, v = _kv(layer, 0)
        captured.append((k[mask].clone(), v[mask].clone()))
        state.process_layer(layer, k, v)
    state.end_step()

    state.begin_step(1)
    assert state.should_skip_reference_projection
    for layer in range(4):
        state.record_projection_skip(layer, total_rows=4)
        k, v = _kv(layer, 1000)
        current_k = k.clone()
        current_v = v.clone()
        out_k, out_v = state.process_layer(layer, k, v)
        torch.testing.assert_close(out_k[mask], captured[layer][0])
        torch.testing.assert_close(out_v[mask], captured[layer][1])
        torch.testing.assert_close(out_k[~mask], current_k[~mask])
        torch.testing.assert_close(out_v[~mask], current_v[~mask])
    state.end_step()

    assert state.device_pool is not None
    assert tuple(state.device_pool.shape) == (2, 2, 2, 2, 3)
    assert state.device_staging_bytes == state.host_bytes // 2
    assert state.stats.refresh_steps == 1
    assert state.stats.h2d_steps == 1
    assert state.stats.captured_layers == 4
    assert state.stats.substituted_layers == 4
    assert state.stats.prefetched_layers == 4
    assert state.stats.h2d_bytes == state.host_bytes
    assert state.stats.max_device_staging_bytes == state.device_staging_bytes
    assert state.stats.projection_skipped_layers == 4
    assert state.stats.projection_skipped_rows == 8
    assert state.stats.projection_total_rows == 16


def test_tier2_zero_reference_rank_is_inactive() -> None:
    state = MiniMaxH3ReferenceKVTier2State(
        num_layers=2,
        global_reference_rows=2,
        device=torch.device("cpu"),
        refresh_interval=2,
        ring_size=2,
        pin_memory=False,
    )
    mask = torch.zeros(4, dtype=torch.bool)

    for step in range(2):
        state.begin_step(step)
        state.set_local_reference_mask(mask)
        for layer in range(2):
            k, v = _kv(layer, float(step))
            state.process_layer(layer, k, v)
        state.end_step()

    assert state.host_pool is None
    assert state.device_pool is None
    assert state.stats.refresh_steps == 0
    assert state.stats.h2d_steps == 0
    assert state.stats.captured_layers == 0
    assert state.stats.substituted_layers == 0
    assert state.stats.prefetched_layers == 0


def test_environment_selects_tier2(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_OMNI_MINIMAX_H3_REF_KV_OBSERVER", raising=False)
    monkeypatch.delenv("VLLM_OMNI_MINIMAX_H3_REF_KV_TIER1", raising=False)
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_REF_KV_TIER2", "1")
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_REF_KV_RING_SIZE", "3")
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_REF_KV_SKIP_PROJECTION", "1")
    state = MiniMaxH3ReferenceKVTier1State.from_environment(
        num_layers=4,
        global_reference_rows=2,
        device=torch.device("cpu"),
    )

    assert isinstance(state, MiniMaxH3ReferenceKVTier2State)
    assert state.ring_size == 3
    assert state.skip_reference_projection


def test_observer_measures_multiple_intervals_without_substitution(tmp_path) -> None:
    state = MiniMaxH3ReferenceKVObserverState(
        num_layers=2,
        global_reference_rows=2,
        device=torch.device("cpu"),
        intervals=(2, 3),
        sample_rows=2,
        sample_heads=1,
        output_path=str(tmp_path / "drift.jsonl"),
    )
    mask = torch.tensor([False, True, False, True])

    for step in range(3):
        state.begin_step(step, sigma=1.0 - 0.1 * step)
        state.set_local_reference_mask(mask)
        for layer in range(2):
            k, v = _kv(layer, float(step * 10))
            expected_k = k.clone()
            expected_v = v.clone()
            out_k, out_v = state.process_layer(layer, k, v)
            assert out_k.data_ptr() == k.data_ptr()
            assert out_v.data_ptr() == v.data_ptr()
            torch.testing.assert_close(out_k, expected_k)
            torch.testing.assert_close(out_v, expected_v)
        state.end_step()

    assert state.stats.observed_steps == 3
    assert state.stats.sampled_layers == 6
    assert state.stats.metric_records == 12
    state.close()

    records = [json.loads(line) for line in (tmp_path / "drift.cpu.jsonl").read_text().splitlines()]
    assert [record["kind"] for record in records].count("run_start") == 1
    assert [record["kind"] for record in records].count("layout") == 1
    assert [record["kind"] for record in records].count("run_end") == 1

    metrics = [record for record in records if record["kind"] == "metric"]
    assert len(metrics) == 12
    step_one = [record for record in metrics if record["step"] == 1]
    assert {(record["interval"], record["source_step"], record["age"]) for record in step_one} == {(2, 0, 1), (3, 0, 1)}
    step_two = [record for record in metrics if record["step"] == 2]
    assert {(record["interval"], record["source_step"], record["age"]) for record in step_two} == {(3, 0, 2)}
    assert {record["kv"] for record in metrics} == {"K", "V"}
    assert all(record["rel_l2"] > 0 for record in metrics)
    assert all(record["abs_error_max"] > 0 for record in metrics)
    assert all(record["source_sigma"] == 1.0 for record in metrics)
    assert all(record["sigma_delta"] < 0 for record in metrics)


def test_observer_zero_reference_rank_is_inactive(tmp_path) -> None:
    state = MiniMaxH3ReferenceKVObserverState(
        num_layers=2,
        global_reference_rows=2,
        device=torch.device("cpu"),
        intervals=(2,),
        sample_rows=2,
        sample_heads=1,
        output_path=str(tmp_path / "inactive.jsonl"),
    )
    mask = torch.zeros(4, dtype=torch.bool)

    for step in range(2):
        state.begin_step(step, sigma=float(1 - step))
        state.set_local_reference_mask(mask)
        for layer in range(2):
            k, v = _kv(layer, float(step))
            out_k, out_v = state.process_layer(layer, k, v)
            assert out_k.data_ptr() == k.data_ptr()
            assert out_v.data_ptr() == v.data_ptr()
        state.end_step()

    assert state.host_bytes == 0
    assert state.stats.observed_steps == 0
    assert state.stats.sampled_layers == 0
    assert state.stats.metric_records == 0
    state.close()

    records = [json.loads(line) for line in (tmp_path / "inactive.cpu.jsonl").read_text().splitlines()]
    end = next(record for record in records if record["kind"] == "run_end")
    assert end["inactive"] is True
    assert end["metric_records"] == 0


def test_environment_observer_takes_precedence(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_REF_KV_OBSERVER", "1")
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_REF_KV_TIER1", "1")
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_REF_KV_TIER2", "1")
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_REF_KV_SKIP_PROJECTION", "1")
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_REF_KV_REFRESH_INTERVAL", "invalid")
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_REF_KV_OBSERVER_INTERVALS", "4,2")
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_REF_KV_OBSERVER_ROWS", "3")
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_REF_KV_OBSERVER_HEADS", "1")
    monkeypatch.setenv(
        "VLLM_OMNI_MINIMAX_H3_REF_KV_OBSERVER_OUTPUT",
        str(tmp_path / "observer.jsonl"),
    )

    state = MiniMaxH3ReferenceKVTier1State.from_environment(
        num_layers=2,
        global_reference_rows=2,
        device=torch.device("cpu"),
    )

    assert isinstance(state, MiniMaxH3ReferenceKVObserverState)
    assert state.intervals == (2, 4)
    assert state.sample_rows == 3
    assert state.sample_heads == 1
    assert not state.skip_reference_projection
    state.close()


def _post_ulysses_kv(layer: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.arange(6 * 2 * 3, dtype=torch.float32).view(1, 6, 2, 3)
    return (
        values + base + layer * 100,
        values + base + layer * 100 + 50,
    )


def _populate_post_ulysses_state(
    *, host_quantization: str
) -> tuple[
    MiniMaxH3ReferenceKVTier2State,
    list[tuple[torch.Tensor, torch.Tensor]],
]:
    state = MiniMaxH3ReferenceKVTier2State(
        num_layers=2,
        global_reference_rows=2,
        device=torch.device("cpu"),
        refresh_interval=2,
        ring_size=2,
        pin_memory=False,
        skip_reference_projection=True,
        host_quantization=host_quantization,
    )
    global_mask = torch.tensor([True, False, False, True, False, False])
    state.begin_step(0)
    state.set_global_reference_mask(global_mask)
    # This emulates a pre-Ulysses rank with no local reference rows. It must
    # remain active because post-Ulysses every rank owns the full sequence.
    state.set_local_reference_mask(torch.zeros(4, dtype=torch.bool))
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer in range(2):
        key, value = _post_ulysses_kv(layer, 0)
        captured.append((key[0, global_mask].clone(), value[0, global_mask].clone()))
        out_key, out_value = state.process_post_parallel_layer(
            layer,
            key,
            value,
            compact=False,
            parallel_strategy="ulysses",
        )
        assert out_key.data_ptr() == key.data_ptr()
        assert out_value.data_ptr() == value.data_ptr()
    state.end_step()
    return state, captured


def test_tier2_post_ulysses_cache_is_balanced_and_reused() -> None:
    state, captured = _populate_post_ulysses_state(host_quantization="none")

    assert state.host_pool is not None
    assert tuple(state.host_pool.shape) == (2, 2, 2, 2, 3)
    assert not state._inactive

    state.begin_step(1)
    for layer in range(2):
        target_key = torch.full((1, 4, 2, 3), 1000.0 + layer)
        target_value = torch.full((1, 4, 2, 3), 2000.0 + layer)
        out_key, out_value = state.process_post_parallel_layer(
            layer,
            target_key,
            target_value,
            compact=True,
            parallel_strategy="ulysses",
        )
        torch.testing.assert_close(out_key[0, :2], captured[layer][0])
        torch.testing.assert_close(out_value[0, :2], captured[layer][1])
        torch.testing.assert_close(out_key[:, 2:], target_key)
        torch.testing.assert_close(out_value[:, 2:], target_value)
    state.end_step()

    assert state.stats.refresh_steps == 1
    assert state.stats.h2d_steps == 1
    assert state.stats.captured_layers == 2
    assert state.stats.substituted_layers == 2
    assert state.stats.h2d_bytes == state.host_bytes


def test_tier2_post_ulysses_int8_host_quantization() -> None:
    state, captured = _populate_post_ulysses_state(host_quantization="int8")

    assert state.host_pool is not None
    assert state.host_pool.dtype == torch.int8
    assert state._host_scale_pool is not None
    assert tuple(state._host_scale_pool.shape) == (2, 2, 2)
    quantized_host_bytes = state.host_bytes

    state.begin_step(1)
    for layer in range(2):
        target_key = torch.zeros((1, 4, 2, 3))
        target_value = torch.zeros_like(target_key)
        out_key, out_value = state.process_post_parallel_layer(
            layer,
            target_key,
            target_value,
            compact=True,
            parallel_strategy="ulysses",
        )
        torch.testing.assert_close(out_key[0, :2], captured[layer][0], rtol=0.02, atol=1.0)
        torch.testing.assert_close(out_value[0, :2], captured[layer][1], rtol=0.02, atol=1.0)
    state.end_step()

    bf32_bytes = 2 * 2 * 2 * 2 * 3 * torch.tensor([], dtype=torch.float32).element_size()
    assert quantized_host_bytes < bf32_bytes
    assert state.stats.dequantized_layers == 2
    assert state.stats.h2d_bytes == quantized_host_bytes


def test_environment_selects_tier2_host_quantization(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_OMNI_MINIMAX_H3_REF_KV_OBSERVER", raising=False)
    monkeypatch.delenv("VLLM_OMNI_MINIMAX_H3_REF_KV_TIER1", raising=False)
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_REF_KV_TIER2", "1")
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_REF_KV_HOST_QUANTIZATION", "int8")
    state = MiniMaxH3ReferenceKVTier1State.from_environment(
        num_layers=2,
        global_reference_rows=2,
        device=torch.device("cpu"),
    )

    assert isinstance(state, MiniMaxH3ReferenceKVTier2State)
    assert state.post_parallel_cache
    assert state.host_quantization == "int8"


def test_tier2_post_ulysses_fp8_host_quantization() -> None:
    if not hasattr(torch, "float8_e4m3fn"):
        return
    state, captured = _populate_post_ulysses_state(host_quantization="fp8")

    assert state.host_pool is not None
    assert state.effective_host_quantization == "fp8"
    assert state.host_pool.dtype == torch.float8_e4m3fn
    assert state._host_scale_pool is None

    state.begin_step(1)
    for layer in range(2):
        target_key = torch.zeros((1, 4, 2, 3))
        target_value = torch.zeros_like(target_key)
        out_key, out_value = state.process_post_parallel_layer(
            layer,
            target_key,
            target_value,
            compact=True,
            parallel_strategy="ulysses",
        )
        torch.testing.assert_close(out_key[0, :2], captured[layer][0], rtol=0.1, atol=4.0)
        torch.testing.assert_close(out_value[0, :2], captured[layer][1], rtol=0.1, atol=4.0)
    state.end_step()

    assert state.stats.quantization_fallbacks == 0
    assert state.stats.dequantized_layers == 2
    assert state.stats.h2d_bytes == state.host_bytes
