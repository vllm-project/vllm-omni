# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("vllm_ascend")

import vllm.v1.worker.gpu.spec_decode.speculator  # noqa: F401
import vllm_ascend.ops  # noqa: F401
from vllm.config import CacheConfig, ModelConfig, ParallelConfig, SpeculativeConfig, VllmConfig
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor
from vllm_ascend.ascend_config import KVPPConfig
from vllm_ascend.worker import worker as ascend_worker
from vllm_ascend.worker.v2 import attn_utils

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_kv import MiniCPMO45DuplexWindowSpec
from vllm_omni.platforms.npu.minicpmo_shared_kv import can_share_minicpmo_kv, install_minicpmo_shared_kv

pytestmark = [pytest.mark.core_model, pytest.mark.omni, *hardware_marks(res={"npu": "A3"}, num_cards=1)]


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda _: False)
    monkeypatch.setattr(KVPPConfig, "from_vllm_config", lambda _: SimpleNamespace(size=1))
    return MagicMock(
        spec=VllmConfig,
        model_config=MagicMock(
            spec=ModelConfig,
            use_v2_model_runner=True,
            model_stage="llm",
            architectures=["MiniCPMO45OmniForConditionalGeneration"],
            hf_text_config=SimpleNamespace(model_type="qwen3"),
        ),
        speculative_config=MagicMock(spec=SpeculativeConfig, method="dspark"),
        cache_config=MagicMock(
            spec=CacheConfig,
            cache_dtype="auto",
            get_resolved_kv_cache_layout=lambda: SimpleNamespace(is_layer_compact=True, is_block_compact=True),
        ),
        parallel_config=MagicMock(spec=ParallelConfig, decode_context_parallel_size=1, prefill_context_parallel_size=1),
        kv_transfer_config=None,
    )


@pytest.fixture
def specs():
    fields = dict(block_size=128, num_kv_heads=2, head_size=64, dtype=torch.bfloat16)
    return (
        FullAttentionSpec(**fields),
        MiniCPMO45DuplexWindowSpec(
            **fields, sliding_window=1024, chunk_size=128, window_chunks=8, sink_chunks=1, reset_at_boundary=False
        ),
    )


def test_budget_and_allocator_share_the_same_policy(config, specs, monkeypatch):
    # The installed Ascend source rejects this layout before Omni's extension.
    assert not attn_utils.can_share_full_and_sliding_kv_cache(config, specs)
    monkeypatch.setattr(
        attn_utils, "can_share_full_and_sliding_kv_cache", attn_utils.can_share_full_and_sliding_kv_cache
    )
    groups = [KVCacheGroupSpec(layer_names=[name], kv_cache_spec=spec) for name, spec in zip(("full", "window"), specs)]
    worker = ascend_worker.NPUWorker.__new__(ascend_worker.NPUWorker)
    worker.vllm_config = config
    worker.model_config = config.model_config
    worker.use_v2_model_runner = True
    worker.model_runner = SimpleNamespace(supports_standardized_shared_kv_backing=True)
    worker.get_kv_cache_spec = lambda: dict(zip(("full", "window"), specs))
    monkeypatch.setattr(ascend_worker, "get_kv_cache_groups", lambda *_: groups)
    assert worker._scale_kv_cache_memory_for_multi_group(1024) == 512
    install_minicpmo_shared_kv(config)
    installed = attn_utils.can_share_full_and_sliding_kv_cache
    install_minicpmo_shared_kv(config)
    assert installed is attn_utils.can_share_full_and_sliding_kv_cache
    assert installed(config, iter(specs))
    assert worker._scale_kv_cache_memory_for_multi_group(1024) == 1024

    blocks = 4
    page_size = specs[0].page_size_bytes
    cache = KVCacheConfig(
        num_blocks=blocks,
        kv_cache_groups=groups,
        kv_cache_tensors=[
            KVCacheTensor(
                size=blocks * page_size,
                layers=[name],
                layer_stride=blocks * page_size,
                block_stride=page_size,
                offset=0,
            )
            for name in ("full", "window")
        ],
    )
    allocations = []

    def allocate(size, alignment, device):
        allocations.append(size)
        return torch.zeros(size, dtype=torch.int8, device=device)

    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(attn_utils, "_allocate_int8_cache_tensor", allocate)
    result = attn_utils._allocate_kv_cache(cache, {}, torch.device("cpu"))
    assert allocations == [blocks * page_size]
    full_k, full_v = result["full"]
    window_k, window_v = result["window"]
    assert full_k.data_ptr() == window_k.data_ptr()
    assert full_v.data_ptr() == window_v.data_ptr()
    assert full_k.is_contiguous() and full_v.is_contiguous()
    # Different live block IDs are disjoint; after release the same ID can be
    # reused through the other group without overwriting the still-live page.
    half_page = page_size // 2
    full_k[:half_page].fill_(7)
    window_k[half_page : 2 * half_page].fill_(9)
    assert torch.all(full_k[:half_page] == 7)
    window_k[:half_page].fill_(3)
    assert torch.all(window_k[half_page : 2 * half_page] == 9)


@pytest.mark.parametrize(
    "scope,value",
    [
        ("use_v2_model_runner", False),
        ("model_stage", "tts"),
        ("architectures", ["Qwen3ForCausalLM"]),
        ("method", "ngram"),
        ("method", None),
        ("cache_dtype", "int8"),
        ("kv_transfer_config", object()),
        ("decode_context_parallel_size", 2),
        ("prefill_context_parallel_size", 2),
    ],
)
def test_rejects_other_paths(config, specs, scope, value):
    owner = config
    for candidate in (config.model_config, config.speculative_config, config.cache_config, config.parallel_config):
        if hasattr(candidate, scope):
            owner = candidate
            break
    setattr(owner, scope, value)
    assert not can_share_minicpmo_kv(config, specs)


@pytest.mark.parametrize("change", [{"num_kv_heads": 4}, {"dtype": torch.float16}, {"head_size_v": 128}])
def test_rejects_incompatible_pages(config, specs, change):
    assert not can_share_minicpmo_kv(config, (specs[0], replace(specs[1], **change)))


def test_preserves_original_policy(config, specs, monkeypatch):
    calls = []

    def original(cfg, items):
        calls.append((cfg, tuple(items)))
        return cfg.model_config.hf_text_config.model_type == "gemma4_text"

    monkeypatch.setattr(attn_utils, "can_share_full_and_sliding_kv_cache", original)
    install_minicpmo_shared_kv(config)
    config.model_config.architectures = ["Gemma4ForCausalLM"]
    config.model_config.hf_text_config.model_type = "gemma4_text"
    assert attn_utils.can_share_full_and_sliding_kv_cache(config, iter(specs))
    assert calls == [(config, specs)]
    config.model_config.hf_text_config.model_type = "qwen3"
    assert not attn_utils.can_share_full_and_sliding_kv_cache(config, specs)


def test_no_install_for_other_models_or_runners(config, monkeypatch):
    original = attn_utils.can_share_full_and_sliding_kv_cache
    config.model_config.use_v2_model_runner = False
    install_minicpmo_shared_kv(config)
    assert attn_utils.can_share_full_and_sliding_kv_cache is original
    config.model_config.use_v2_model_runner = True
    config.model_config.architectures = ["Qwen3ForCausalLM"]
    install_minicpmo_shared_kv(config)
    assert attn_utils.can_share_full_and_sliding_kv_cache is original


@pytest.mark.parametrize("setting", ["sfa", "kvpp", "layout"])
def test_rejects_specialized_allocators(config, specs, monkeypatch, setting):
    if setting == "sfa":
        monkeypatch.setattr(attn_utils, "enable_sfa", lambda _: True)
    elif setting == "kvpp":
        monkeypatch.setattr(KVPPConfig, "from_vllm_config", lambda _: SimpleNamespace(size=2))
    else:
        config.cache_config.get_resolved_kv_cache_layout = lambda: SimpleNamespace(
            is_layer_compact=False, is_block_compact=True
        )
    assert not can_share_minicpmo_kv(config, specs)
