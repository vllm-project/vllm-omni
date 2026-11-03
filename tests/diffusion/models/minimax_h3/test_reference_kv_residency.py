# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.kv_offload_config import parse_reference_kv_config
from vllm_omni.diffusion.models.minimax_h3.reference_kv_cache import MiniMaxH3ReferenceKVTier2State

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _state(residency, layers=3):
    state = MiniMaxH3ReferenceKVTier2State(
        num_layers=layers,
        global_reference_rows=2,
        device=torch.device("cpu"),
        refresh_interval=2,
        pin_memory=False,
        cache_residency=residency,
    )
    state.set_global_reference_mask(torch.tensor([False, True, False, True, False]))
    return state


def _kv(step, layer, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    key = (torch.arange(40).reshape(1, 5, 2, 4) + 100 * step + 10 * layer).to(dtype)
    return key, key + 1000


def _process(state, layer, key, value, compact):
    return state.process_post_parallel_layer(layer, key, value, compact=compact, parallel_strategy="none")


def test_reference_kv_default_remains_host():
    config = parse_reference_kv_config({"mode": "on"})
    assert config is not None
    assert config.kv_cache_residency == "host"


@pytest.mark.parametrize("residency", ["host", "device"])
def test_reference_kv_residency_config_reaches_runtime(residency):
    config = parse_reference_kv_config({"mode": "on", "kv_cache_residency": residency})
    state = MiniMaxH3ReferenceKVTier2State.from_config(
        config=config, num_layers=3, global_reference_rows=2, device=torch.device("cpu")
    )
    assert state is not None
    assert state.cache_residency == residency
    state.close()


@pytest.mark.parametrize("quantization", ["fp8", "int8"])
def test_device_residency_rejects_host_quantization(quantization):
    with pytest.raises(ValueError, match="requires kv_host_quantization=none"):
        parse_reference_kv_config({"mode": "on", "kv_cache_residency": "device", "kv_host_quantization": quantization})
    with pytest.raises(ValueError, match="requires host_quantization=none"):
        MiniMaxH3ReferenceKVTier2State(
            num_layers=1,
            global_reference_rows=2,
            device=torch.device("cpu"),
            refresh_interval=2,
            cache_residency="device",
            host_quantization=quantization,
        )


@pytest.mark.parametrize("residency", ["auto", None, []])
def test_invalid_reference_kv_residency_is_rejected(residency):
    with pytest.raises(ValueError, match="kv_cache_residency must be host or device"):
        parse_reference_kv_config({"mode": "on", "kv_cache_residency": residency})


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_resident_kv_matches_host_capture_refresh_and_reuse(dtype):
    host, resident = _state("host"), _state("device")
    reference = torch.tensor([1, 3])
    target = torch.tensor([0, 2, 4])
    captured: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for step in range(4):
        compact = bool(step % 2)
        host.begin_step(step)
        resident.begin_step(step)
        for layer in range(3):
            key, value = _kv(step, layer, dtype)
            if compact:
                key, value = key[:, target], value[:, target]
                expected = tuple(
                    torch.cat((old.unsqueeze(0), new), dim=1) for old, new in zip(captured[layer], (key, value))
                )
            else:
                captured[layer] = (key[0, reference].clone(), value[0, reference].clone())
                expected = (key, value)
            for state in (host, resident):
                actual = _process(state, layer, key, value, compact)
                for result, wanted in zip(actual, expected):
                    torch.testing.assert_close(result, wanted, rtol=0, atol=0)
        host.end_step()
        resident.end_step()
    assert host.stats.refresh_steps == resident.stats.refresh_steps == 2
    assert host.stats.substituted_layers == resident.stats.substituted_layers == 6
    assert host.stats.d2h_bytes == host.stats.h2d_bytes == 2 * host.host_bytes
    assert resident.device_resident_bytes == host.host_bytes
    assert resident.host_bytes == resident.device_staging_bytes == 0
    assert resident.stats.d2h_bytes == resident.stats.h2d_bytes == resident.stats.h2d_steps == 0
    assert resident.stats.device_reuse_steps == 2
    assert resident._device_ring is None and resident._copy_stream is None
    host.close()
    resident.close()
    assert resident.device_resident_bytes == 0


@pytest.mark.parametrize("residency", ["host", "device"])
def test_returned_reference_payload_survives_later_refresh(residency):
    state = _state(residency, layers=1)
    key, value = _kv(0, 0)
    state.begin_step(0)
    _process(state, 0, key, value, False)
    state.end_step()
    state.begin_step(1)
    cached = state.take_cached_reference_layer(0, key)
    saved = tuple(t.clone() for t in cached)
    state.end_step()
    state.begin_step(2)
    _process(state, 0, *_kv(2, 0), False)
    state.end_step()
    for actual, expected in zip(cached, saved):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    state.close()


def test_resident_kv_supports_ordered_block_cache_skips_and_abort():
    state = _state("device", layers=4)
    state.begin_step(0)
    for layer in range(4):
        _process(state, layer, *_kv(0, layer), False)
    state.end_step()
    state.begin_step(1)
    key, value = _kv(1, 0)
    _process(state, 0, key[:, [0, 2, 4]], value[:, [0, 2, 4]], True)
    with pytest.raises(RuntimeError, match="in order"):
        state.mark_block_cache_skipped([2])
    state.mark_block_cache_skipped([1, 2])
    key, value = _kv(1, 3)
    _process(state, 3, key[:, [0, 2, 4]], value[:, [0, 2, 4]], True)
    state.end_step()
    assert state.stats.block_cached_layers == 2
    assert state.stats.substituted_layers == 2
    assert state.stats.h2d_bytes == 0
    state.begin_step(2)
    _process(state, 0, *_kv(2, 0), False)
    state.abort_step()
    state.close()
    assert state.current_step is None
    assert state.device_resident_bytes == 0
    assert state._resident_pool is None
