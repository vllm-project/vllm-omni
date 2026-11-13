# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Zonos3 AR module tests: architecture contract, layer arithmetic, GDN state
semantics, runtime validation, official-LM oracle parity and the native
release-checkpoint model loader.

The official oracle sections run only with ZONOS3_REFERENCE_ROOT pointing at
an unmodified Zyphra/ZONOS3-Mini checkout.
"""

import ast
import json
import os
import sys
from dataclasses import fields, replace
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, Mock

import pytest
import torch
from torch.nn import functional as F
from transformers import PretrainedConfig
from vllm.config import (
    CacheConfig,
    CompilationConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import SamplingStates
from vllm.v1.worker.gpu.states import RequestState

from vllm_omni.model_executor.models.registry import OmniModelRegistry
from vllm_omni.model_executor.models.zonos3 import modeling_zonos3_mini
from vllm_omni.model_executor.models.zonos3.ar_config import Zonos3Architecture
from vllm_omni.model_executor.models.zonos3.model_state import Zonos3ARModelState
from vllm_omni.model_executor.models.zonos3.modeling_zonos3_mini import (
    FullAttnResWorkspace,
    MultiEmbedding,
    MultiOutputHead,
    Zonos3AttentionProjection,
    Zonos3Backbone,
    Zonos3BackboneBatch,
    Zonos3DenseFFN,
    Zonos3ForCausalLM,
    Zonos3GDN,
    Zonos3MoEFFN,
    Zonos3PrefillSpan,
    Zonos3QuantileRouter,
    full_attnres_route,
    native_attention_factory,
    validate_zonos3_ar_runtime,
)
from vllm_omni.worker_v2.model_states import init_omni_model_state

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def small_ar_config() -> Zonos3Architecture:
    return Zonos3Architecture(
        n_layers=11,
        dim=16,
        n_heads=2,
        n_kv_heads=1,
        head_dim=8,
        norm_eps=1e-5,
        rope_theta=10000.0,
        max_seqlen=512,
        ffn_hidden=24,
        moe_hidden=12,
        moe_routed_dim=8,
        n_experts=4,
        text_vocab=20,
        speaker_embedding_dim=4,
        condition_bucket_counts=(2, 2, 3),
        condition_names=("loudness", "fmax", "speaking_rate"),
        condition_edges={"loudness": (-20.0,), "fmax": (16000.0,), "speaking_rate": (8.0, 14.0)},
        condition_units={},
        codebook_size=8,
    )


def release_params():
    return {
        "n_layers": 11,
        "dim": 1024,
        "head_dim": 128,
        "n_heads": 8,
        "n_kv_heads": 2,
        "norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_seqlen": 2048,
        "loss_softcap": 15.0,
        "attention_pattern": "gdn3_global1",
        "residual_type": "full_attnres",
        "use_xsa": False,
        "n_codebooks": 10,
        "codebook_size": 4096,
        "eoa_id": 4096,
        "audio_pad_id": 4097,
        "text_vocab": 260,
        "speaker_enabled": True,
        "speaker_embedding_dim": 192,
        "condition_tokens_enabled": True,
        "condition_count": 3,
        "condition_n_buckets": 10,
        "condition_names": ["loudness", "fmax", "speaking_rate"],
        "condition_bucket_counts": [10, 10, 30],
        "moe_router_type": "linear",
        "moe_router_use_eda": False,
        "moe_balancing_strategy": "quantile",
        "moe_router_topk": 2,
        "moe_start_from_layer": 1,
        "moe_end_from_layer": 0,
        "moe_alpha": 0.25,
        "moe_n_experts": 8,
        "ffn_dim_multiplier": 3.5,
        "moe_ffn_dim_multiplier": 2.5,
    }


def test_release_architecture_has_nine_gdn_two_global_layers_and_exact_widths():
    c = Zonos3Architecture.from_params({"model": release_params()})
    assert [i for i in range(11) if not c.uses_gdn(i)] == [3, 7]
    assert sum(c.uses_gdn(i) for i in range(11)) == 9
    assert (c.ffn_hidden, c.moe_hidden, c.moe_routed_dim) == (3584, 2560, 256)
    assert c.audio_vocab == 4098 and c.codebook_sizes == (4096,) * 10
    assert c.condition_bucket_counts == (10, 10, 30)


@pytest.mark.parametrize(
    "key,value",
    [
        ("residual_type", "standard"),
        ("attention_pattern", "global"),
        ("n_layers", 12),
        ("moe_router_use_eda", True),
        ("qwen_gating_style", "elementwise"),
        ("eoa_id", 1024),
        ("moe_n_experts", 2),
        ("moe_alpha", 0.0),
        ("codebook_sizes", [4096] * 9 + [1024]),
        ("condition_bucket_counts", [10]),
    ],
)
def test_release_architecture_rejects_incompatible_variants(key, value):
    with pytest.raises(ValueError):
        Zonos3Architecture.from_params({**release_params(), key: value})


@torch.inference_mode()
def test_nonzero_checkpoint_padding_embeddings_are_preserved():
    config = small_ar_config()
    embed = MultiEmbedding(config)
    for i, table in enumerate(embed.embedders):
        table.weight.fill_(i + 1)
    ids = torch.tensor([[config.audio_pad_id] * 10 + [config.text_vocab]])
    assert torch.equal(embed(ids), torch.full((1, config.dim), 66.0))


@torch.inference_mode()
def test_dense_checkpoint_up_gate_order():
    c = replace(small_ar_config(), ffn_hidden=16)
    layer = Zonos3DenseFFN(c)
    layer.w_in.weight[0] = 2 * torch.eye(16)
    layer.w_in.weight[1] = 3 * torch.eye(16)
    layer.w_out.weight.copy_(torch.eye(16))
    x = torch.linspace(-2, 2, 32).reshape(2, 16)
    torch.testing.assert_close(layer(x), 2 * x * F.silu(3 * x))


@torch.inference_mode()
def test_attnres_workspace_is_overwritten_across_bucket_rows():
    torch.manual_seed(41)
    workspace = FullAttnResWorkspace(23, 8, 16, dtype=torch.float32, device="cpu")
    pointer = workspace.bank.data_ptr()
    for rows in (8, 1, 4, 8):
        sources = torch.randn(23, rows, 16)
        query = torch.randn(16)
        for i, source in enumerate(sources):
            workspace.write(i, source)
        expected = full_attnres_route(sources, query, 1e-5)
        torch.testing.assert_close(workspace.route(23, rows, query, 1e-5), expected)
        assert workspace.bank.data_ptr() == pointer


@torch.inference_mode()
def test_quantile_bias_changes_selection_but_not_selected_raw_logit_weights():
    c = small_ar_config()
    router = Zonos3QuantileRouter(c)
    router.router_linear.weight.zero_()
    router.router_linear.weight[:, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    router.balancing_biases[:] = torch.tensor([0.0, 0.0, 100.0, 100.0])
    x = torch.ones(3, 16)
    weights, choices = router(x)
    assert choices.tolist() == [[1, 0]] * 3
    expected = router.router_linear(router.rmsnorm_eda(x)).float().gather(-1, choices.long()).softmax(-1)
    torch.testing.assert_close(weights, expected)


@pytest.mark.parametrize("batch", [1, 4, 8, 16, 32])
@torch.inference_mode()
def test_output_head_ten_codebooks_and_single_row_batch_invariance(batch):
    torch.manual_seed(21)
    c = small_ar_config()
    head = MultiOutputHead(c)
    head.weight.normal_(std=0.2)
    x = torch.randn(batch, 16)
    whole = head(x)
    assert whole.shape == (batch, 10, 10)
    for row in range(batch):
        torch.testing.assert_close(head(x[row : row + 1])[0], whole[row])


@torch.inference_mode()
def test_latent_moe_cpu_batches_have_independent_rows():
    torch.manual_seed(19)
    c = small_ar_config()
    module = Zonos3MoEFFN(c)
    for name, parameter in module.named_parameters():
        if name.endswith("balancing_biases"):
            parameter.zero_()
        else:
            parameter.normal_(std=0.1)
    x = torch.randn(8, 16)
    together = module(x)
    torch.testing.assert_close(together, torch.cat([module(row[None]) for row in x]), atol=1e-7, rtol=2e-5)


@torch.inference_mode()
def test_flat_ar_checkpoint_requires_every_parameter_and_preserves_runtime_buffers():
    c = small_ar_config()
    model = Zonos3Backbone(c, lambda config, index: Zonos3AttentionProjection(config))
    # A native attention backend can add non-checkpoint cache scale buffers.
    model.layers[3].attention.register_buffer("runtime_scale", torch.ones(1))
    weights = {name: torch.randn_like(parameter) for name, parameter in model.named_parameters()}
    loaded = model.load_release_state(weights)
    assert loaded == set(weights) and model.layers[3].attention.runtime_scale.item() == 1.0
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, weights[name], atol=0, rtol=0)
        assert not parameter.requires_grad
    with pytest.raises(ValueError, match="checkpoint keys differ"):
        model.load_release_state({**weights, "encoder.unexpected": torch.ones(1)})
    with pytest.raises(ValueError, match="checkpoint keys differ"):
        model.load_release_state({name: value for name, value in weights.items() if name != "block_attnres_queries"})
    with pytest.raises(RuntimeError, match="size mismatch"):
        model.load_release_state({**weights, "block_attnres_queries": torch.zeros(2, 2)})


def valid_runtime():
    config = MagicMock(spec=VllmConfig)
    config.cache_config = CacheConfig(enable_prefix_caching=False)
    config.model_config = MagicMock(spec=ModelConfig)
    config.model_config.enforce_eager = False
    config.parallel_config = ParallelConfig()
    config.compilation_config = CompilationConfig(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY)
    return config


def test_native_attention_binding_requires_complete_gdn_cache_semantics():
    config = valid_runtime()
    config.cache_config.enable_prefix_caching = True
    with pytest.raises(ValueError, match="KV-only hits cannot restore GDN"):
        native_attention_factory(config)


@pytest.mark.parametrize(
    "mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL_AND_PIECEWISE, CUDAGraphMode.FULL]
)
def test_unsupported_graph_modes_fail_instead_of_downgrading(mode):
    config = valid_runtime()
    config.compilation_config.cudagraph_mode = mode
    with pytest.raises(ValueError, match="requires FULL_DECODE_ONLY"):
        validate_zonos3_ar_runtime(config)


def test_native_full_decode_mode_is_accepted_before_backend_construction():
    config = valid_runtime()
    assert callable(native_attention_factory(config))
    config.model_config.enforce_eager = True
    with pytest.raises(ValueError, match="requires FULL_DECODE_ONLY"):
        validate_zonos3_ar_runtime(config)


@pytest.mark.parametrize(
    "field,value", [("tensor_parallel_size", 2), ("pipeline_parallel_size", 2), ("enable_expert_parallel", True)]
)
def test_unsupported_parallel_state_sharing_fails_explicitly(field, value):
    config = valid_runtime()
    setattr(config.parallel_config, field, value)
    with pytest.raises(ValueError, match="one tensor/pipeline rank"):
        validate_zonos3_ar_runtime(config)


def initialized_gdn():
    torch.manual_seed(42)
    layer = Zonos3GDN(16).eval()
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.normal_(std=0.1)
    return layer


@pytest.mark.parametrize("batch", [1, 4, 8, 16, 32])
@torch.inference_mode()
def test_window_commit_boundary_is_per_slot_and_survives_batch_reordering(batch):
    layer = initialized_gdn()
    state = layer.allocate_state(batch)
    positions = []
    for slot in range(batch):
        count = 63 - slot % 2
        layer.forward_prefill(torch.randn(count, 16), slot, 0, state)
        positions.append(count)
    assert not state.recurrent.any()  # incomplete training chunks never commit
    slots = torch.arange(batch - 1, -1, -1)
    layer.forward_decode(torch.randn(batch, 16), slots, state)
    for slot, before in enumerate(positions):
        assert int(state.total_tokens[slot]) == before + 1
        assert int(state.window_length[slot]) == (before + 1) % 64
        if before == 63:
            assert not state.window[slot].any()
            assert state.recurrent[slot].any()
        else:
            assert not state.recurrent[slot].any()
    state.reset([0])
    assert not state.window[0].any() and not state.recurrent[0].any() and not state.conv[0].any()
    assert state.prefill_tokens[0] == 0 and int(state.total_tokens[0]) == 0


@torch.inference_mode()
def test_stale_prefill_offset_is_rejected_before_state_mutation():
    layer = initialized_gdn()
    state = layer.allocate_state(1)
    layer.forward_prefill(torch.randn(7, 16), 0, 0, state)
    with pytest.raises(ValueError, match="offset is stale"):
        layer.forward_prefill(torch.randn(2, 16), 0, 0, state)
    assert state.prefill_tokens == [7] and int(state.total_tokens[0]) == 7


def test_cuda_kernel_preparation_requires_explicit_reference_backend(monkeypatch):
    layer = initialized_gdn()
    monkeypatch.delenv("FLA_TILELANG", raising=False)
    with pytest.raises(ValueError, match="FLA_TILELANG=0"):
        layer.prepare_cuda([1, 4])
    monkeypatch.setenv("FLA_TILELANG", "0")
    monkeypatch.setattr("importlib.metadata.version", lambda name: "0.5.1")
    with pytest.raises(ImportError, match="fla-core==0.5.2"):
        layer.prepare_cuda([1, 4])


@pytest.fixture(scope="module")
def official_lm():
    root = os.environ.get("ZONOS3_REFERENCE_ROOT")
    if not root:
        pytest.skip("Set ZONOS3_REFERENCE_ROOT for official AR arithmetic checks")
    path = Path(root) / "model/src/zonos3"
    buckets = {"__name__": "_zonos3_reference_buckets"}
    exec(compile((path / "condition_buckets.py").read_text(), str(path / "condition_buckets.py"), "exec"), buckets)
    filename = path / "reference/lm.py"
    tree = ast.parse(filename.read_text(), filename=str(filename))
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.ImportFrom) and node.module in ("zonos3.condition_buckets", "zonos3.tts.probe"))
    ]
    module = ModuleType("_zonos3_official_lm_oracle")
    module.bucket_counts = buckets["bucket_counts"]
    sys.modules[module.__name__] = module
    try:
        exec(compile(tree, str(filename), "exec"), module.__dict__)
        yield module
    finally:
        sys.modules.pop(module.__name__, None)


@pytest.fixture
def oracle(official_lm):
    torch.manual_seed(14)
    c = small_ar_config()
    config = official_lm.Zonos3LMConfig(
        **{field.name: getattr(c, field.name) for field in fields(official_lm.Zonos3LMConfig)}
    )
    model = official_lm.Zonos3LM(config).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("balancing_biases"):
                parameter.uniform_(-0.1, 0.1)
            else:
                parameter.normal_(std=0.15)
    return model


def test_official_release_architecture_derivation(official_lm):
    native = Zonos3Architecture.from_params(release_params())
    reference = official_lm.Zonos3LMConfig.from_params(release_params())
    for field in fields(reference):
        assert getattr(native, field.name) == getattr(reference, field.name), field.name


@torch.inference_mode()
def test_official_multiembedding_and_full_attnres(official_lm, oracle):
    c = small_ar_config()
    embed = MultiEmbedding(c)
    embed.load_state_dict(oracle.multi_embedder.state_dict(), strict=True)
    codes = torch.cat((torch.randint(0, c.audio_vocab, (9, 10)), torch.randint(0, c.text_rows, (9, 1))), -1)
    expected = sum(F.embedding(codes[:, cb], oracle.multi_embedder.embedders[cb].weight) for cb in range(11))
    torch.testing.assert_close(embed(codes), expected, atol=0, rtol=0)
    bank = torch.randn(23, 9, c.dim)
    query = torch.randn(c.dim)
    torch.testing.assert_close(
        full_attnres_route(bank, query, c.norm_eps),
        official_lm.full_attnres_route(list(bank.unbind(0)), query, c.norm_eps),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("batch", [1, 4, 8, 16, 32])
@torch.inference_mode()
def test_official_dense_and_latent_moe(oracle, batch):
    c = small_ar_config()
    dense = Zonos3DenseFFN(c)
    moe = Zonos3MoEFFN(c)
    dense.load_state_dict(oracle.layers[0].feed_forward.state_dict(), strict=True)
    moe.load_state_dict(oracle.layers[1].feed_forward.state_dict(), strict=True)
    x = torch.randn(batch, c.dim)
    torch.testing.assert_close(dense(x), oracle._feed_forward(x, 0), atol=1e-7, rtol=2e-5)
    torch.testing.assert_close(moe(x), oracle._feed_forward(x, 1), atol=1e-7, rtol=2e-5)


@torch.inference_mode()
def test_official_global_qknorm_rope_head_gating_and_output(official_lm, oracle):
    c = small_ar_config()
    native = Zonos3AttentionProjection(c)
    native.load_state_dict(oracle.layers[3].attention.state_dict(), strict=True)
    prefix, x = torch.randn(17, c.dim), torch.randn(5, c.dim)
    cache = official_lm.LMCache()
    oracle._global_attention(prefix, 3, cache, None, {})
    cache.position = 17
    expected = oracle._global_attention(x, 3, cache, None, {})
    past = native.project(prefix, torch.arange(17))
    qkv = native.project(x, torch.arange(17, 22))
    key = torch.cat((past.key, qkv.key)).view(22, c.n_kv_heads, c.head_dim)
    value = torch.cat((past.value, qkv.value)).view(22, c.n_kv_heads, c.head_dim)
    query = qkv.query.view(5, c.n_heads, c.head_dim).transpose(0, 1)
    repeats = c.n_heads // c.n_kv_heads
    out = F.scaled_dot_product_attention(
        query,
        key.repeat_interleave(repeats, 1).transpose(0, 1),
        value.repeat_interleave(repeats, 1).transpose(0, 1),
        attn_mask=torch.arange(22)[None, :] <= torch.arange(17, 22)[:, None],
        scale=c.head_dim**-0.5,
    ).transpose(0, 1)
    torch.testing.assert_close(native.output(out, qkv.gate), expected, atol=1e-7, rtol=2e-5)


@pytest.mark.parametrize("batch", [1, 4, 8, 16, 32])
@torch.inference_mode()
def test_official_multioutput_head_logits(oracle, batch):
    c = small_ar_config()
    head = MultiOutputHead(c)
    head.load_state_dict(oracle.multi_output.state_dict(), strict=True)
    hidden = torch.randn(batch, c.dim)
    expected = F.linear(hidden, oracle.multi_output.weight).view(batch, 10, c.audio_vocab)
    expected = c.loss_softcap * torch.tanh(expected / c.loss_softcap)
    torch.testing.assert_close(head(hidden), expected, atol=1e-7, rtol=2e-5)


@torch.inference_mode()
def test_official_gdn_chunked_prefill_crosses_training_chunk_boundaries(official_lm, oracle):
    native = Zonos3GDN(small_ar_config().dim)
    native.load_state_dict(oracle.layers[0].attention.state_dict(), strict=True)
    state = native.allocate_state(1)
    cache = official_lm.LMCache()
    offset = 0
    for length in (1, 7, 55, 1, 2, 61, 3):
        x = torch.randn(length, native.hidden_size)
        expected = oracle._gdn(x, 0, cache)
        actual = native.forward_prefill(x, 0, offset, state)
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=3e-5)
        offset += length
        assert int(state.window_length[0]) == offset % 64
        if offset % 64 == 0:
            torch.testing.assert_close(state.recurrent[0], cache.gdn_state[0], atol=2e-7, rtol=3e-5)
            # Replaying a 64-row window changes the GEMM row count versus the
            # oracle's one-row chunks; FP32 projection rounding is not bitwise
            # invariant (observed maximum 2.38e-7 in this fixture).
            torch.testing.assert_close(state.conv[0, :, 1:], cache.conv_state[0], atol=3e-7, rtol=3e-5)


@pytest.mark.parametrize("batch", [1, 4, 8, 16, 32])
@torch.inference_mode()
def test_official_gdn_batched_decode_independent_chunk_positions(official_lm, oracle, batch):
    native = Zonos3GDN(small_ar_config().dim)
    native.load_state_dict(oracle.layers[0].attention.state_dict(), strict=True)
    state = native.allocate_state(batch)
    caches = [official_lm.LMCache() for _ in range(batch)]
    for slot in range(batch):
        length = 61 + slot % 3
        x = torch.randn(length, native.hidden_size)
        oracle._gdn(x, 0, caches[slot])
        native.forward_prefill(x, slot, 0, state)
    for step in range(4):
        slots = list(range(batch)) if step % 2 else list(reversed(range(batch)))
        x = torch.randn(batch, native.hidden_size)
        actual = native.forward_decode(x, torch.tensor(slots), state)
        for row, slot in enumerate(slots):
            expected = oracle._gdn(x[row : row + 1], 0, caches[slot])
            torch.testing.assert_close(actual[row : row + 1], expected, atol=2e-7, rtol=3e-5)


class CPUAttentionOracle(Zonos3AttentionProjection):
    """Test-only KV backend; CUDA serving uses Zonos3GlobalAttention/vLLM Attention."""

    def __init__(self, config):
        super().__init__(config)
        self.cache = {}

    def forward(self, x, positions, batch):
        spans = [(span.start, span.length, span.slot) for span in batch.prefill_spans]
        if batch.decode_only:
            spans = [(row, 1, slot) for row, slot in enumerate(batch.decode_slots.tolist())]
        else:
            spans += [
                (row, 1, slot)
                for row, slot in zip(batch.decode_rows.tolist(), batch.decode_slots.tolist(), strict=True)
            ]
        result = torch.zeros_like(x)
        c = self.config
        for start, length, slot in spans:
            qkv = self.project(x[start : start + length], positions[start : start + length])
            old_k, old_v = self.cache.get(slot, (qkv.key[:0], qkv.value[:0]))
            assert old_k.shape[0] == int(positions[start])
            k, v = torch.cat((old_k, qkv.key)), torch.cat((old_v, qkv.value))
            self.cache[slot] = (k, v)
            repeats = c.n_heads // c.n_kv_heads
            q = qkv.query.view(length, c.n_heads, c.head_dim).transpose(0, 1)
            keys = k.view(-1, c.n_kv_heads, c.head_dim).repeat_interleave(repeats, 1).transpose(0, 1)
            values = v.view(-1, c.n_kv_heads, c.head_dim).repeat_interleave(repeats, 1).transpose(0, 1)
            out = F.scaled_dot_product_attention(
                q,
                keys,
                values,
                attn_mask=torch.arange(len(k))[None, :] <= positions[start : start + length, None],
                scale=c.head_dim**-0.5,
            ).transpose(0, 1)
            result[start : start + length] = self.output(out, qkv.gate)
        return result


@torch.inference_mode()
def test_complete_backbone_hidden_logits_mixed_prefill_decode_and_conditioning(official_lm, oracle):
    config = small_ar_config()
    native = Zonos3Backbone(config, lambda c, index: CPUAttentionOracle(c)).eval()
    native.load_release_state(oracle.state_dict())
    states = native.allocate_gdn_states(2)
    workspace = native.allocate_workspace(10)
    pointer = workspace.bank.data_ptr()
    caches = [official_lm.LMCache(), official_lm.LMCache()]
    prompts = [
        torch.tensor([[config.audio_pad_id] * 10 + [token % config.text_vocab] for token in range(n)]) for n in (9, 6)
    ]
    speakers = [torch.randn(config.speaker_embedding_dim) for _ in range(2)]
    conditions = [{1: 0, 2: 1, 3: 2}, {1: 1, 2: 0, 3: 1}]
    head_calls = []
    hook = native.multi_output.register_forward_pre_hook(lambda *_: head_calls.append(True))

    def run(ids, spans, decode_slots, decode_rows, decode_only):
        embeddings = native.multi_embedder(torch.cat([row[1] for row in ids]))
        positions = []
        expected_hidden, expected_logits = [], []
        start = 0
        for slot, frame_ids in ids:
            offset = caches[slot].position
            positions.extend(range(offset, offset + len(frame_ids)))
            capture = {}
            output = oracle(
                frame_ids,
                caches[slot],
                speaker_rows={0: speakers[slot]},
                condition_rows=conditions[slot],
                capture=capture,
            )
            expected_hidden.append(torch.from_numpy(capture["out_norm"]))
            expected_logits.append(output.logits)
            if offset == 0:
                embeddings[start] = native.speaker_projection(speakers[slot])
                for table, row in enumerate(sorted(conditions[slot])):
                    if row < len(frame_ids):
                        embeddings[start + row] = native.condition_embeddings[table].weight[conditions[slot][row]]
            start += len(frame_ids)
        batch = Zonos3BackboneBatch(
            states,
            workspace,
            torch.tensor(decode_slots, dtype=torch.int64),
            torch.tensor(decode_rows, dtype=torch.int64),
            tuple(spans),
            decode_only,
        )
        before = len(head_calls)
        hidden = native(embeddings, torch.tensor(positions), batch)
        assert len(head_calls) == before  # Backbone forward never invokes the RVQ head
        torch.testing.assert_close(hidden, torch.cat(expected_hidden), atol=2e-6, rtol=3e-4)
        torch.testing.assert_close(native.multi_output(hidden), torch.cat(expected_logits), atol=2e-6, rtol=3e-4)
        assert workspace.bank.data_ptr() == pointer

    try:
        run(
            [(0, prompts[0][:4]), (1, prompts[1])],
            [Zonos3PrefillSpan(0, 4, 0, 0), Zonos3PrefillSpan(4, 6, 1, 0)],
            [],
            [],
            False,
        )
        generated = torch.cat(
            (torch.randint(0, config.codebook_size, (1, 10)), torch.full((1, 1), config.text_vocab)), -1
        )
        run([(0, prompts[0][4:]), (1, generated)], [Zonos3PrefillSpan(0, 5, 0, 4)], [1], [5], False)
        run([(1, generated), (0, generated)], [], [1, 0], [0, 1], True)
    finally:
        hook.remove()


@torch.inference_mode()
def test_release_checkpoint_native_factory_and_parallel_codebook_sampling(tmp_path, monkeypatch):
    monkeypatch.setattr("vllm.v1.worker.gpu.buffer_utils.is_uva_available", lambda: False)
    monkeypatch.setattr(
        modeling_zonos3_mini,
        "native_attention_factory",
        lambda config, prefix: lambda arch, index: CPUAttentionOracle(arch),
    )
    params = {
        **release_params(),
        "dim": 16,
        "head_dim": 8,
        "n_heads": 2,
        "n_kv_heads": 1,
        "multiple_of": 4,
        "moe_multiple_of": 4,
        "ffn_dim_multiplier": 1.5,
        "moe_ffn_dim_multiplier": 1.0,
    }
    (tmp_path / "params.json").write_text(json.dumps(params))
    config = valid_runtime()
    config.model_config = MagicMock(spec=ModelConfig)
    config.model_config.model = str(tmp_path)
    config.model_config.revision = None
    config.model_config.hf_config = PretrainedConfig(hidden_size=16)
    config.model_config.enable_prompt_embeds = False
    config.model_config.uses_mrope = False
    config.model_config.get_inputs_embeds_size.return_value = 16
    config.model_config.max_model_len = 128
    config.model_config.dtype = torch.float32
    config.load_config = LoadConfig(load_format="pt")
    config.speculative_config = None
    config.scheduler_config = SchedulerConfig(
        max_model_len=128, is_encoder_decoder=False, max_num_seqs=4, max_num_batched_tokens=32
    )
    config.compilation_config.cudagraph_capture_sizes = [1, 2, 4]
    assert OmniModelRegistry.models["Zonos3ForCausalLM"].load_model_cls() is Zonos3ForCausalLM
    model = Zonos3ForCausalLM(vllm_config=config)
    with pytest.raises(RuntimeError, match="not been loaded"):
        model()
    torch.manual_seed(17)
    weights = {name: torch.randn_like(parameter) * 0.1 for name, parameter in model.named_parameters()}
    torch.save(weights, tmp_path / "model.pth")
    loaded = model.load_weights(DefaultModelLoader(config.load_config).get_all_weights(config.model_config, model))
    assert loaded == set(weights)
    with pytest.raises(RuntimeError, match="already loaded"):
        model.load_weights(iter(()))
    state = init_omni_model_state(config, model, None, torch.device("cpu"))
    assert isinstance(state, Zonos3ARModelState)
    base = Mock(spec=Sampler)
    base.sampling_states = SamplingStates(4, 3)
    base.add_request.side_effect = base.sampling_states.add_request
    base.apply_staged_writes.side_effect = base.sampling_states.apply_staged_writes
    base.req_states = MagicMock(spec=RequestState)
    base.req_states.index_to_req_id = {}
    base.req_states.prefill_len = UvaBackedTensor(4, dtype=torch.int32)
    sampler, _ = state.custom_sampler(base)
    owner = state, sampler, base.req_states
    from tests.model_executor.models.test_zonos3_model_state import add, batch

    add(owner, "request", 0, 5)
    data = batch(owner, [("request", 0, 5, 0, True)])
    inputs = state.prepare_inputs(data, base.req_states)
    state.run_preprocess(data, inputs, base.req_states)
    calls = []
    hook = model.multi_output.register_forward_pre_hook(lambda _, args: calls.append(args[0].shape[0]))
    try:
        hidden = model(input_ids=data.input_ids, positions=data.positions, **inputs)
        assert hidden.shape == (5, 16) and not calls
        result = sampler.sample_step(
            hidden, data, base.req_states, None, lambda *args: pytest.fail("text sampling called")
        )
        assert calls == [1]
    finally:
        hook.remove()
    assert result.num_sampled.tolist() == [1]
    snapshot = result.finalize_multimodal(result.multimodal_outputs, [1])
    assert snapshot.inter_stage[0]["codes"]["audio"].shape == (1, 10)
