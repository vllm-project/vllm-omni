# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("vllm_ascend")

import vllm.v1.worker.gpu.spec_decode.speculator  # noqa: F401 -- Populate the module referenced by Ascend V2.
import vllm_ascend.ops  # noqa: F401 -- Register Ascend operations before importing runners.
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor
from vllm_ascend.attention import utils
from vllm_ascend.worker import model_runner_v1
from vllm_ascend.worker.v2 import attn_utils

from tests.helpers.mark import hardware_marks
from vllm_omni.platforms.npu.platform import apply_contiguous_kv_patch

pytestmark = [pytest.mark.core_model, pytest.mark.omni, *hardware_marks(res={"npu": "A3"}, num_cards=1)]


def make_kv_cache_config(name: str, spec: FullAttentionSpec) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[
            KVCacheTensor(
                size=2 * spec.page_size_bytes,
                layers=[name],
                layer_stride=2 * spec.page_size_bytes,
                block_stride=spec.page_size_bytes,
                offset=0,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[name], kv_cache_spec=spec)],
    )


def test_contiguous_kv_cache_layout_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    original = utils.requires_contiguous_pa_kv_cache
    monkeypatch.setattr(original, "__code__", (lambda layer, vllm_config, spec: False).__code__)
    apply_contiguous_kv_patch()
    apply_contiguous_kv_patch()
    assert utils.requires_contiguous_pa_kv_cache is original
    assert model_runner_v1.requires_contiguous_pa_kv_cache is original
    assert attn_utils.requires_contiguous_pa_kv_cache is original

    runner = model_runner_v1.NPUModelRunner.__new__(model_runner_v1.NPUModelRunner)
    runner.device = torch.device("cpu")
    runner.ascend_config = SimpleNamespace(
        kvpp_config=SimpleNamespace(size=1), xlite_graph_config=SimpleNamespace(enabled=False)
    )
    runner.dcp_size = 1
    runner.runner_only_attn_layers = set()
    runner.use_sparse = False
    runner.use_compress = False
    runner.use_hybrid_blocks = False
    runner.sparse_kv_offload_enabled = False
    runner.sparse_kv_offload_config = None
    runner.model_config = MagicMock()
    runner.model_config.use_mla = False
    runner.model_config.is_hybrid = False
    runner.vllm_config = MagicMock()
    runner.vllm_config.model_config = runner.model_config
    runner.vllm_config.kv_transfer_config = None
    runner.vllm_config.quant_config = None
    runner.vllm_config.cache_config.cache_dtype = "auto"

    name = "model.layers.0.self_attn.attn"
    spec = FullAttentionSpec(block_size=128, num_kv_heads=2, head_size=64, dtype=torch.bfloat16)
    runner._kv_cache_spec_attn_group_iterator = lambda: iter(
        [
            SimpleNamespace(
                kv_cache_group_id=0,
                kv_cache_spec=spec,
                backend=model_runner_v1.AscendAttentionBackend,
                layer_names=[name],
            )
        ]
    )
    config = make_kv_cache_config(name, spec)
    raw = runner._allocate_kv_cache_tensors(config)[name]
    key, value = (tensor.view(spec.dtype).view(2, 128, 2, 64) for tensor in raw)
    assert key.is_contiguous()
    assert value.is_contiguous()
    assert key.untyped_storage().data_ptr() != value.untyped_storage().data_ptr()


@pytest.mark.parametrize("padded", [False, True], ids=["unpadded", "padded"])
def test_contiguous_kv_cache_layout_v2(monkeypatch: pytest.MonkeyPatch, padded: bool) -> None:
    original = utils.requires_contiguous_pa_kv_cache
    monkeypatch.setattr(original, "__code__", original.__code__)
    apply_contiguous_kv_patch()
    name = "model.layers.0.self_attn.attn"
    spec = FullAttentionSpec(block_size=128, num_kv_heads=2, head_size=64, dtype=torch.bfloat16)
    if padded:
        spec = replace(spec, page_size_padded=spec.real_page_size_bytes + 256)
    assert original(None, None, spec) is (not padded)
    config = make_kv_cache_config(name, spec)
    vllm_config = MagicMock()
    vllm_config.parallel_config.decode_context_parallel_size = 1
    vllm_config.kv_transfer_config = None
    vllm_config.quant_config = None
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: vllm_config)
    monkeypatch.setattr(attn_utils.KVPPConfig, "from_vllm_config", lambda config: SimpleNamespace(size=1))
    monkeypatch.setattr(attn_utils, "_is_dsv4_model", lambda config: False)
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda config: False)
    monkeypatch.setattr(attn_utils, "enable_fa_quant", lambda config: False)
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *args: {})
    raw = attn_utils._allocate_kv_cache(config, {}, torch.device("npu"))
    assert isinstance(raw[name], torch.Tensor if padded else tuple)
    group = SimpleNamespace(
        kv_cache_group_id=0,
        kv_cache_spec=spec,
        backend=model_runner_v1.AscendAttentionBackend,
        layer_names=[name],
    )
    caches = attn_utils._reshape_kv_cache_v2([group], raw, "auto", [128], {}, config)
    key, value = caches[name]
    assert key.shape == value.shape == (2, 128, 2, 64)
    if not padded:
        assert key.is_contiguous() and value.is_contiguous()
        assert key.untyped_storage().data_ptr() != value.untyped_storage().data_ptr()
    else:
        assert key.stride(0) * key.element_size() == spec.page_size_bytes
    key.fill_(1)
    value.fill_(2)
    assert torch.all(key == 1)
    assert torch.all(value == 2)
