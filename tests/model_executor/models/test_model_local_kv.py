# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A declaration must match the cache that actually gets allocated.

Two earlier versions of this file could not fail. The first asserted hand-typed
constants against each other. The second built real `transformers` caches but
compared them to specs written in the test, so no production declaration was
ever executed. What matters here is that a real declarer is called and its
numbers are checked against a real cache built the way the model builds it.
"""

import ast
import pathlib
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config
from transformers.cache_utils import DynamicCache, StaticCache

from vllm_omni.model_executor.models.model_local_kv import (
    HasModelLocalKV,
    ModelLocalKVScope,
    ModelLocalKVSpec,
    RowDriver,
    collect_model_local_kv_specs,
    spec_from_hf_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

MODELS_DIR = pathlib.Path(__file__).resolve().parents[3] / "vllm_omni" / "model_executor" / "models"


def _total_bytes(model: object, max_num_seqs: int = 1) -> int:
    """Sum declared peak bytes across a model's declarations."""
    return sum(spec.peak_bytes(max_num_seqs) for _, spec in collect_model_local_kv_specs(model))


def _cache_bytes(cache) -> int:
    total = 0
    for layer in cache.layers:
        for tensor in (layer.keys, layer.values):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
    return total


def _qwen2_config(layers=4, kv_heads=2, heads=8, hidden=256) -> Qwen2Config:
    return Qwen2Config(
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        vocab_size=1000,
    )


# --------------------------------------------------------------------------
# A real declarer, called, checked against a real cache.
# --------------------------------------------------------------------------


def test_ming_declaration_matches_the_cache_it_describes():
    """Calls MingAudioGenerator.model_local_kv_specs and checks the bytes.

    Builds the declarer without its weights -- only `_llm_config` and `_model`
    are read -- then builds the `StaticCache` that `_init_kv_cache` builds from
    the same config and compares. A wrong layer count, kv-head count, dtype or
    capacity in the production declaration fails here.
    """
    from vllm_omni.model_executor.models.ming_flash_omni.talker_module import MingAudioGenerator

    config = _qwen2_config()
    generator = object.__new__(MingAudioGenerator)
    generator._llm_config = config
    generator._model = torch.nn.Linear(1, 1).to(torch.float16)

    (spec,) = generator.model_local_kv_specs()

    cache = StaticCache(
        config=config,
        max_batch_size=1,
        max_cache_len=MingAudioGenerator._STATIC_CACHE_LEN,
        device="cpu",
        dtype=torch.float16,
    )
    head_dim = config.hidden_size // config.num_attention_heads
    for layer_idx in range(config.num_hidden_layers):
        keys = torch.zeros(1, config.num_key_value_heads, 1, head_dim, dtype=torch.float16)
        cache.update(keys, keys.clone(), layer_idx)

    assert spec.peak_bytes(1) == _cache_bytes(cache)


def test_ming_does_not_scale_with_max_num_seqs():
    """forward() takes runtime_additional_information[0] and loops inline.

    One worker holds one of these no matter how many sequences the scheduler
    admits. A previous revision derived the multiplier from scope and reported
    `max_num_seqs` times the real figure.
    """
    from vllm_omni.model_executor.models.ming_flash_omni.talker_module import MingAudioGenerator

    generator = object.__new__(MingAudioGenerator)
    generator._llm_config = _qwen2_config()
    generator._model = torch.nn.Linear(1, 1).to(torch.float16)

    (spec,) = generator.model_local_kv_specs()
    assert spec.peak_bytes(64) == spec.peak_bytes(1)


def test_growing_cache_declaration_matches_a_real_dynamic_cache():
    """The declared positions are the extent the tensor reaches when full."""
    layers, kv_heads, head_dim, positions = 4, 2, 32, 11
    config = _qwen2_config(layers=layers, kv_heads=kv_heads, heads=kv_heads, hidden=kv_heads * head_dim)
    spec = spec_from_hf_config(
        config,
        name="local_transformer",
        dtype=torch.bfloat16,
        physical_capacity_positions=positions,
        capacity_source="group_size + max(delay_pattern)",
        scope=ModelLocalKVScope.INVOCATION,
        rows=RowDriver.FIXED,
        rows_reason="test",
    )

    cache = DynamicCache()
    for layer_idx in range(layers):
        keys = torch.zeros(1, kv_heads, positions, head_dim, dtype=torch.bfloat16)
        cache.update(keys, torch.zeros_like(keys), layer_idx)

    assert spec.peak_bytes() == _cache_bytes(cache)


def test_multiplicity_does_not_claim_an_object_layout():
    """One allocation B rows wide costs the same as B one-row allocations.

    The protocol reports bytes and deliberately does not distinguish the two.
    An earlier revision carried a separate `allocations` count and got the
    object topology wrong in three of four declarers, which is worse than not
    claiming it: the layout belongs in `allocation_note`, where it cannot be
    mistaken for arithmetic.
    """
    config = _qwen2_config(layers=4, kv_heads=2, heads=2, hidden=64)
    common = dict(
        config=config,
        name="c",
        dtype=torch.bfloat16,
        physical_capacity_positions=11,
        capacity_source="test",
        scope=ModelLocalKVScope.INVOCATION,
    )
    batched = spec_from_hf_config(rows=RowDriver.MAX_NUM_SEQS, **common)

    assert batched.row_count(64) == 64
    assert batched.peak_bytes(64) == batched.bytes_per_row * 64
    assert not hasattr(batched, "allocations"), "the object-count axis was removed on purpose"


def test_a_fixed_driver_ignores_concurrency():
    """A serialized or capture-pinned cache does not widen with max_num_seqs."""
    spec = spec_from_hf_config(
        _qwen2_config(),
        name="graph_pool",
        dtype=torch.bfloat16,
        physical_capacity_positions=11,
        capacity_source="test",
        scope=ModelLocalKVScope.MODEL,
        rows=RowDriver.FIXED,
        rows_fixed=261,
        rows_reason="sum of captured buckets",
    )
    assert spec.row_count(1) == 261
    assert spec.row_count(64) == 261
    assert spec.peak_bytes(64) == spec.bytes_per_row * 261


def test_qwen3_tts_declares_nothing_on_the_stateless_path():
    """Calls the real decoder declarer with no cache-holding mode configured.

    `_forward_exact` runs `pre_transformer` without `use_cache`, so stateless
    decoding allocates no KV. A revision that declared a per-request cache
    unconditionally invented over a gigabyte at max_num_seqs=256, and the
    revision before it declared nothing at all under enforce_eager, where the
    eager async-chunk path does allocate. Both are regressions this pins.
    """
    from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz import (
        modeling_qwen3_tts_tokenizer_v2 as mod,
    )

    decoder = object.__new__(mod.Qwen3TTSTokenizerV2Decoder)
    # nn.Module.__setattr__ needs the machinery __init__ would have built, and
    # this instance deliberately has none of it.
    setattr_ = object.__setattr__
    setattr_(
        decoder,
        "config",
        SimpleNamespace(
            sliding_window=72,
            num_hidden_layers=8,
            num_key_value_heads=16,
            head_dim=64,
            hidden_size=1024,
            num_attention_heads=16,
        ),
    )
    setattr_(decoder, "parameters", lambda: iter([torch.zeros(1, dtype=torch.float32)]))

    setattr_(decoder, "_cudagraph_wrapper", None)
    assert decoder.model_local_kv_specs() == []

    setattr_(decoder, "_cudagraph_wrapper", SimpleNamespace(async_chunk=False, model_local_kv_specs=lambda: []))
    assert decoder.model_local_kv_specs() == []

    setattr_(decoder, "_cudagraph_wrapper", SimpleNamespace(async_chunk=True, model_local_kv_specs=lambda: []))
    names = [s.name for s in decoder.model_local_kv_specs()]
    assert names == ["codec_decoder_request", "codec_decoder_working_copy"], names


def test_fixed_rejects_unreviewable_values():
    """A bare or unexplained constant cannot be reviewed, so it is rejected."""
    common = dict(
        name="x",
        layers=1,
        kv_heads=1,
        head_dim=1,
        dtype=torch.float16,
        physical_capacity_positions=1,
        capacity_source="",
        scope=ModelLocalKVScope.REQUEST,
    )
    with pytest.raises(ValueError, match="rows_reason"):
        ModelLocalKVSpec(rows=RowDriver.FIXED, rows_fixed=1, rows_reason="   ", **common)
    with pytest.raises(ValueError, match=">= 1"):
        ModelLocalKVSpec(rows=RowDriver.FIXED, rows_fixed=0, rows_reason="test", **common)


def test_rows_is_required():
    """Defaulting the batch extent is how the previous guarantee died.

    The dataclass declared it required while the helper every declarer uses
    defaulted it to 1, so nothing enforced it where it mattered.
    """
    with pytest.raises(TypeError):
        spec_from_hf_config(  # type: ignore[call-arg]
            _qwen2_config(),
            name="x",
            dtype=torch.float16,
            physical_capacity_positions=1,
            capacity_source="",
            scope=ModelLocalKVScope.REQUEST,
        )


def test_spec_from_hf_config_raises_rather_than_guessing():
    class Empty:
        pass

    with pytest.raises(ValueError, match="layer count"):
        spec_from_hf_config(
            Empty(),
            name="x",
            dtype=torch.float16,
            physical_capacity_positions=1,
            capacity_source="",
            scope=ModelLocalKVScope.REQUEST,
            rows=RowDriver.FIXED,
            rows_reason="test",
        )


# --------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------


def _spec(**overrides) -> ModelLocalKVSpec:
    kwargs = dict(
        name="c",
        layers=2,
        kv_heads=2,
        head_dim=4,
        dtype=torch.float16,
        physical_capacity_positions=8,
        capacity_source="test",
        scope=ModelLocalKVScope.REQUEST,
        rows=RowDriver.FIXED,
        rows_reason="test",
    )
    kwargs.update(overrides)
    return ModelLocalKVSpec(**kwargs)


class _Owner(torch.nn.Module):
    def model_local_kv_specs(self):
        return [_spec()]


class _CUDAGraphWrapperLike:
    """Mirrors vllm.compilation.cuda_graph.CUDAGraphWrapper.

    Not an nn.Module, and its __getattr__ forwards everything the runnable
    has. That forwarding is why a previous revision's "unwrap fix" was
    unnecessary, and why a walk that does not unwrap can enter the tree twice.
    """

    def __init__(self, runnable):
        self.runnable = runnable

    def unwrap(self):
        return self.runnable

    def __getattr__(self, key):
        return getattr(self.runnable, key)


def test_collector_finds_owners_nested_in_the_module_tree():
    class Mid(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = _Owner()

    class Top(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mid = Mid()

    model = Top()
    assert [path for path, _ in collect_model_local_kv_specs(model)] == ["mid.inner"]
    # 2 layers * 2 kv * 4 head_dim * 8 positions * 2 bytes * 2 (K and V) = 512
    assert _total_bytes(model) == 512


def test_a_root_declarer_behind_a_wrapper_is_counted_once():
    """The wrapper forwards attributes, so a naive walk sees it twice."""

    class Root(_Owner):
        pass

    wrapped = _CUDAGraphWrapperLike(Root())
    collected = collect_model_local_kv_specs(wrapped)

    assert len(collected) == 1, collected
    assert _total_bytes(wrapped) == 512


def test_collector_reaches_owners_through_a_wrapper():
    class Top(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.owner = _Owner()

    wrapped = _CUDAGraphWrapperLike(Top())
    assert [path for path, _ in collect_model_local_kv_specs(wrapped)] == ["owner"]


def test_collector_survives_an_owner_that_raises():
    class Broken(torch.nn.Module):
        def model_local_kv_specs(self):
            raise RuntimeError("checkpoint field missing")

    class Top(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.broken = Broken()
            self.ok = _Owner()

    assert _total_bytes(Top()) == 512


def test_undeclared_models_report_zero_not_an_error():
    assert _total_bytes(object()) == 0
    assert _total_bytes(torch.nn.Linear(2, 2)) == 0


def test_protocol_is_structural():
    assert isinstance(_Owner(), HasModelLocalKV)
    assert not isinstance(torch.nn.Linear(2, 2), HasModelLocalKV)


@pytest.mark.parametrize(
    ("relative_path", "owner"),
    [
        ("qwen3_tts/segmented_graph_wrapper.py", "CUDAGraphDecoderWrapper"),
        ("qwen3_tts/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py", "Qwen3TTSTokenizerV2Decoder"),
        ("mimo_audio/mimo_audio_llm.py", "MiMoAudioLLMForConditionalGeneration"),
        ("minicpmo_4_5/minicpmo_4_5_omni_llm.py", "MiniCPMWhisperEncoder"),
        ("ming_flash_omni/talker_module.py", "MingAudioGenerator"),
        ("ming_flash_omni/ming_flash_omni_talker.py", "MingFlashOmniTalkerForConditionalGeneration"),
    ],
)
def test_known_cache_owners_still_declare(relative_path: str, owner: str):
    """Parsed rather than imported: these modules need CUDA and real weights.

    Weak by construction -- it only proves the method exists. The ming tests
    above are the ones that check a declaration's contents.
    """
    tree = ast.parse((MODELS_DIR / relative_path).read_text())
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == owner]
    assert classes, f"{owner} not found in {relative_path}"
    methods = {child.name for cls in classes for child in cls.body if isinstance(child, ast.FunctionDef)}
    assert "model_local_kv_specs" in methods, f"{owner} no longer declares its model-local KV"
