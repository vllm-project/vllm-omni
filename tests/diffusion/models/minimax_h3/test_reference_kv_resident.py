# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tensor contracts for resident reference KV and compact Ulysses transport."""

import ast
import dataclasses
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]
ROOT = Path(__file__).resolve().parents[4]


def load_source(name: str, relative: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


config = load_source("_h3_reuse_config", "vllm_omni/diffusion/kv_offload_config.py")
transport = load_source("_h3_reuse_transport", "vllm_omni/diffusion/attention/ops/reference_kv_alignment.py")


@pytest.fixture
def cache_module(monkeypatch) -> ModuleType:
    # Only import-time logging/platform dependencies are replaced. All cache
    # allocation, capture, reuse and cleanup below execute the published code
    # with real CPU tensors. No NPU/communication behavior is simulated.
    logger = ModuleType("vllm.logger")
    logger.__dict__["init_logger"] = logging.getLogger
    platform = ModuleType("vllm_omni.platforms")
    platform.__dict__["current_omni_platform"] = object()
    for name, module in {
        "vllm.logger": logger,
        "vllm_omni.platforms": platform,
        "vllm_omni.diffusion.kv_offload_config": config,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return load_source("_h3_reuse_cache", "vllm_omni/diffusion/models/minimax_h3/reference_kv_cache.py")


@torch.inference_mode()
@pytest.mark.parametrize("steps,interval", [(8, 8), (6, 2), (12, 4)])
def test_reference_values_refresh_and_cleanup(cache_module: ModuleType, steps: int, interval: int) -> None:
    state = cache_module.MiniMaxH3ReferenceKVResidentState(
        num_layers=2, global_reference_rows=2, device=torch.device("cpu"), refresh_interval=interval
    )
    state.set_global_reference_mask(torch.tensor([False, True, False, True, False]))
    original = torch.arange(20, dtype=torch.float32).reshape(1, 5, 2, 2)
    for step in range(steps):
        refresh = step % interval == 0
        state.begin_step(step)
        for layer in range(2):
            full = original + 100 * layer + 1000 * step
            key = full if refresh else full[:, [0, 2, 4]]
            out_k, out_v = state.process_post_parallel_layer(
                layer, key, key + 50, compact=not refresh, parallel_strategy="none"
            )
            if refresh:
                assert out_k is key
            else:
                reference = (original + 100 * layer + 1000 * (step // interval * interval))[:, [1, 3]]
                torch.testing.assert_close(out_k, torch.cat((reference, key), dim=1))
                torch.testing.assert_close(out_v, torch.cat((reference + 50, key + 50), dim=1))
        state.end_step()
    refresh_steps = len(range(0, steps, interval))
    reuse_steps = steps - refresh_steps
    assert state.stats.refresh_steps == refresh_steps and state.stats.resident_reuse_steps == reuse_steps
    assert state.stats.captured_layers == 2 * refresh_steps and state.stats.substituted_layers == 2 * reuse_steps
    assert state.stats.h2d_bytes == 0 and state.host_bytes == 0
    assert state.device_cache_bytes > 0 and state._device_ring is None
    state.close()
    assert state.device_cache_bytes == 0 and state._concat_scratch is None


@torch.inference_mode()
def test_host_cache_retains_contiguous_layer_layout(cache_module: ModuleType) -> None:
    state = cache_module.MiniMaxH3ReferenceKVTier2State(
        num_layers=2, global_reference_rows=1, device=torch.device("cpu"), refresh_interval=2
    )
    state.set_global_reference_mask(torch.tensor([False, True, False]))
    key = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    state.begin_step(0)
    for layer in range(2):
        state.process_post_parallel_layer(layer, key + layer, key + layer + 10, compact=False, parallel_strategy="none")
    state.end_step()
    assert state._host_pool.shape == (2, 2, 1, 2, 2)
    assert all(state._host_pool[layer].is_contiguous() for layer in range(2))
    state.begin_step(1)
    for layer in range(2):
        target = (key + layer + 100)[:, [0, 2]]
        restored, _ = state.process_post_parallel_layer(layer, target, target, compact=True, parallel_strategy="none")
        torch.testing.assert_close(restored, torch.cat(((key + layer)[:, [1]], target), dim=1))
    state.end_step()
    assert state.stats.h2d_steps == 1
    state.close()


@torch.inference_mode()
def test_partial_capture_fails_and_request_cleanup_releases_pool(cache_module: ModuleType) -> None:
    state = cache_module.MiniMaxH3ReferenceKVResidentState(
        num_layers=2, global_reference_rows=1, device=torch.device("cpu"), refresh_interval=8
    )
    state.set_global_reference_mask(torch.tensor([True, False]))
    key = torch.ones(1, 2, 1, 2)
    state.begin_step(0)
    state.process_post_parallel_layer(0, key, key, compact=False, parallel_strategy="none")
    with pytest.raises(RuntimeError, match="1/2 layers"):
        state.end_step()
    # The denoise loop closes this request-local state in finally. It never
    # resumes an aborted step with the partially populated cache object.
    state.close()
    assert state.device_cache_bytes == 0 and state._host_pool is None


def test_compact_transport_metadata_reaches_attention(cache_module: ModuleType, mocker) -> None:
    # Exercise the model-owned packing method with a recording attention call.
    # The transport state must survive the transformer-to-block boundary.
    source = ROOT / "vllm_omni/diffusion/models/minimax_h3/minimax_h3_blocks.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MiniMaxH3Attention")
    method = next(
        node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "_run_packed_attention"
    )
    method.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {
        "torch": torch,
        "nn": torch.nn,
        "dataclass": dataclasses.dataclass,
        "field": dataclasses.field,
        "_ring_sequence_parallel_is_active": lambda attention: False,
        "MINIMAX_H3_LASER_INPUT_SCALE": 1.0,
    }
    metadata_source = ROOT / "vllm_omni/diffusion/attention/backends/abstract.py"
    metadata_nodes = [
        node
        for node in ast.parse(metadata_source.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef) and node.name in {"PackedPaddingMetadata", "AttentionMetadata"}
    ]
    module = ModuleType("_h3_transport_metadata")
    sys.modules[module.__name__] = module
    module.__dict__.update(namespace)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[future, *metadata_nodes, owner], type_ignores=[])),
            str(source),
            "exec",
        ),
        module.__dict__,
    )

    class Backend:
        supports_prefix_kv_slicing = True

        @staticmethod
        def supports_packed_mask_free():
            return True

    attention = mocker.Mock(side_effect=lambda q, k, v, metadata: q)
    attention.effective_attention.return_value = (None, Backend, None)
    block_cls = module.__dict__["MiniMaxH3Attention"]
    block = block_cls.__new__(block_cls)
    torch.nn.Module.__init__(block)
    block.attention = attention
    state = cache_module.MiniMaxH3ReferenceKVResidentState(
        num_layers=1, global_reference_rows=2, device=torch.device("cpu"), refresh_interval=8
    )
    state.set_transport_alignment(13, 3)
    q = torch.ones(4, 1, 2)  # Equal physical local shard; logical global Q has 13 rows.
    result = block._run_packed_attention(
        q,
        q,
        q,
        cu_seqlens=torch.tensor([0, 13]),
        max_seqlen=13,
        packed_total=13,
        reference_kv_state=state,
        reference_kv_layer_index=0,
        reference_kv_compact=True,
    )
    metadata = attention.call_args.args[3]
    assert metadata.extra["minimax_h3_transport_alignment"] == {"compact_rows": 13, "transport_padding": 3}
    assert metadata.extra["minimax_h3_reference_kv_post_parallel"] == (state, 0, True)
    assert metadata.extra["minimax_h3_resident_reference_kv"] is True
    assert result.shape == q.shape
    with pytest.raises(RuntimeError, match="layout changed"):
        state.set_transport_alignment(14, 2)
    state.close()
    assert state.transport_alignment is None


@pytest.mark.parametrize("rows,world", [(13, 4), (16, 4), (2, 8), (17, 1), (89754, 4), (89754, 8)])
def test_transport_excludes_poisoned_padding(rows: int, world: int) -> None:
    # Values beyond the logical boundary must never reach attention, even if
    # communication padding contains arbitrary nonzero values.
    original = torch.arange(rows * 4, dtype=torch.float32).reshape(1, rows, 1, 4)
    shards = []
    coverage: list[int] = []
    for rank in range(world):
        start, valid, physical = transport.shard_bounds(rows, world, rank)
        coverage.extend(range(start, start + valid))
        shard = transport.append_rows(original[:, start : start + valid], physical - valid, 1)
        shard[:, valid:] = 999999
        shards.append(shard)
    assert coverage == list(range(rows))
    packed = torch.cat(shards, dim=1)
    padding = packed.shape[1] - rows
    q, k, v = transport.strip_qkv(packed, packed, packed, rows, padding)
    for tensor in (q, k, v):
        torch.testing.assert_close(tensor, original)
    restored = transport.append_rows(q, padding, 1)
    assert restored.shape == packed.shape
    assert torch.count_nonzero(restored[:, rows:]) == 0


@pytest.mark.parametrize("rows,world,rank", [(0, 4, 0), (2, 0, 0), (2, 4, 4), (2, 4, -1)])
def test_invalid_transport(rows: int, world: int, rank: int) -> None:
    with pytest.raises(ValueError):
        transport.shard_bounds(rows, world, rank)


def test_mismatched_qkv_fails() -> None:
    q = torch.zeros(1, 8, 1, 4)
    with pytest.raises(ValueError, match="shape"):
        transport.strip_qkv(q, q[:, :7], q, 5, 3)
    with pytest.raises(ValueError, match="layout"):
        transport.strip_qkv(q, q, q, 5, 2)


def test_reference_cache_is_opt_in() -> None:
    assert config.parse_reference_kv_config(None) is None
    assert config.parse_reference_kv_config({"mode": "on"}).kv_storage == "host"
    parsed = config.parse_reference_kv_config(
        {"mode": "on", "kv_storage": "npu", "kv_refresh_interval": 8, "kv_host_quantization": "none"}
    )
    assert parsed.kv_refresh_interval == 8 and parsed.kv_storage == "npu"


@pytest.mark.parametrize("quantization", ["fp8", "int8"])
def test_device_cache_rejects_quantized_storage(quantization: str) -> None:
    with pytest.raises(ValueError, match="unquantized"):
        config.parse_reference_kv_config({"mode": "on", "kv_storage": "npu", "kv_host_quantization": quantization})


@pytest.mark.parametrize(
    "value", [{"mode": "off"}, {"mode": "on", "unknown": 1}, {"mode": "on", "kv_refresh_interval": -1}]
)
def test_invalid_cache_config(value: dict) -> None:
    with pytest.raises(ValueError):
        config.parse_reference_kv_config(value)
