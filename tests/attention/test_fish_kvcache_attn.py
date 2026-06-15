# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.attention import fish_kvcache_attn, fish_kvcache_backend

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeImpl:
    dcp_world_size = 1
    alibi_slopes = None
    sliding_window = None
    scale = 1.0

    def __init__(self):
        self.original_calls = 0

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        del layer, query, key, value, kv_cache, attn_metadata, output_scale, output_block_scale
        self.original_calls += 1
        output.fill_(2)
        return output


class _FakeAttentionLayer:
    def __init__(self):
        self.impl = _FakeImpl()


class _FakeSelfAttn:
    def __init__(self):
        self.attn = _FakeAttentionLayer()


class _FakeLayer:
    def __init__(self):
        self.self_attn = _FakeSelfAttn()


class _FakeModel:
    def __init__(self):
        self.layers = [_FakeLayer()]


_DEFAULT_UPPER_BOUND = object()


def _metadata(
    *,
    batch_size: int = 2,
    num_actual_tokens: int | None = None,
    max_query_len: int = 1,
    max_seq_len: int = 128,
    use_cascade: bool = False,
    seq_lens_cpu_upper_bound: torch.Tensor | None | object = _DEFAULT_UPPER_BOUND,
    seq_lens: torch.Tensor | None = None,
):
    if num_actual_tokens is None:
        num_actual_tokens = batch_size
    if seq_lens is None:
        seq_lens = torch.ones((batch_size,), dtype=torch.int32)
    if seq_lens_cpu_upper_bound is _DEFAULT_UPPER_BOUND and max_query_len == 1:
        seq_lens_cpu_upper_bound = seq_lens.detach().cpu()
    elif seq_lens_cpu_upper_bound is _DEFAULT_UPPER_BOUND:
        seq_lens_cpu_upper_bound = None
    return type(
        "Metadata",
        (),
        {
            "use_cascade": use_cascade,
            "num_actual_tokens": num_actual_tokens,
            "block_table": torch.zeros((batch_size, 8), dtype=torch.int32),
            "seq_lens": seq_lens,
            "seq_lens_cpu_upper_bound": seq_lens_cpu_upper_bound,
            "max_query_len": max_query_len,
            "max_seq_len": max_seq_len,
        },
    )()


def test_fish_kvcache_enabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_FISH_KVCACHE_ATTN", raising=False)

    assert fish_kvcache_attn.is_fish_kvcache_attn_enabled()
    assert fish_kvcache_backend._fish_kvcache_enabled()
    assert not fish_kvcache_attn.is_fish_kvcache_attn_required()


@pytest.mark.parametrize("value", ["", "1", "true", "yes", "on", "required"])
def test_fish_kvcache_enabled_values_are_consistent(monkeypatch, value):
    monkeypatch.setenv("VLLM_OMNI_FISH_KVCACHE_ATTN", value)
    assert fish_kvcache_attn.is_fish_kvcache_attn_enabled()
    assert fish_kvcache_backend._fish_kvcache_enabled()
    assert fish_kvcache_attn.is_fish_kvcache_attn_required() is (value == "required")


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "disabled", "disable"])
def test_fish_kvcache_disabled_values_are_consistent(monkeypatch, value):
    monkeypatch.setenv("VLLM_OMNI_FISH_KVCACHE_ATTN", value)

    assert not fish_kvcache_attn.is_fish_kvcache_attn_enabled()
    assert not fish_kvcache_backend._fish_kvcache_enabled()
    assert not fish_kvcache_attn.is_fish_kvcache_attn_required()


@pytest.mark.parametrize("sliding_window", [None, (-1, -1)])
def test_fish_kvcache_attn_guard_accepts_decode_only_shape(monkeypatch, sliding_window):
    monkeypatch.setenv("VLLM_OMNI_FISH_KVCACHE_ATTN", "1")
    monkeypatch.setattr(fish_kvcache_attn, "is_available", lambda: True)

    assert fish_kvcache_attn.can_use_fish_kvcache_attn(
        query=torch.empty((4, 32, 128), dtype=torch.float16),
        key_cache=torch.empty((8, 16, 8, 128), dtype=torch.float16),
        value_cache=torch.empty((8, 16, 8, 128), dtype=torch.float16),
        block_table=torch.zeros((4, 8), dtype=torch.int32),
        seq_lens=torch.ones((4,), dtype=torch.int32),
        max_query_len=1,
        max_seq_len=128,
        dcp_world_size=1,
        use_cascade=False,
        alibi_slopes=None,
        sliding_window=sliding_window,
    )


def test_fish_kvcache_attn_guard_rejects_unsupported_block_size(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_FISH_KVCACHE_ATTN", "1")
    monkeypatch.setattr(fish_kvcache_attn, "is_available", lambda: True)

    assert not fish_kvcache_attn.can_use_fish_kvcache_attn(
        query=torch.empty((4, 32, 128), dtype=torch.float16),
        key_cache=torch.empty((8, 32, 8, 128), dtype=torch.float16),
        value_cache=torch.empty((8, 32, 8, 128), dtype=torch.float16),
        block_table=torch.zeros((4, 8), dtype=torch.int32),
        seq_lens=torch.ones((4,), dtype=torch.int32),
        max_query_len=1,
        max_seq_len=128,
        dcp_world_size=1,
        use_cascade=False,
        alibi_slopes=None,
        sliding_window=None,
    )


def test_fish_kvcache_backend_wraps_only_model_instance(monkeypatch):
    fish_kvcache_backend.reset_fish_kvcache_attn_stats()
    monkeypatch.setenv("VLLM_OMNI_FISH_KVCACHE_ATTN", "1")
    monkeypatch.setattr(fish_kvcache_backend, "is_available", lambda: True)
    monkeypatch.setattr(fish_kvcache_backend, "load_error", lambda: None)
    monkeypatch.setattr(fish_kvcache_backend, "can_use_fish_kvcache_attn", lambda **_: True)

    hits = []

    def fake_decode(query, key_cache, value_cache, block_table, seq_lens, out, *, scale, max_seq_len):
        del key_cache, value_cache, block_table, seq_lens, scale, max_seq_len
        hits.append(tuple(query.shape))
        out.fill_(1)
        return out

    monkeypatch.setattr(fish_kvcache_backend, "fish_decode_kvcache_attn", fake_decode)

    model = _FakeModel()
    other_impl = _FakeImpl()

    assert fish_kvcache_backend.install_fish_kvcache_attn_backend(model) == 1
    assert not hasattr(other_impl, "_fish_kvcache_attn_installed")

    metadata = _metadata()
    query = torch.zeros((2, 32, 128), dtype=torch.float16)
    output = torch.zeros_like(query)
    kv_cache = torch.zeros((2, 8, 16, 8, 128), dtype=torch.float16)

    result = model.layers[0].self_attn.attn.impl.forward(
        None,
        query,
        None,
        None,
        kv_cache,
        metadata,
        output,
    )

    assert result is output
    assert hits == [(2, 32, 128)]
    assert model.layers[0].self_attn.attn.impl.original_calls == 0
    assert torch.equal(output, torch.ones_like(output))
    assert fish_kvcache_backend.get_fish_kvcache_attn_stats()["small_hit_count"] == 1


def test_fish_kvcache_backend_install_is_idempotent(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_FISH_KVCACHE_ATTN", "1")
    monkeypatch.setattr(fish_kvcache_backend, "is_available", lambda: True)
    monkeypatch.setattr(fish_kvcache_backend, "load_error", lambda: None)
    monkeypatch.setattr(fish_kvcache_backend, "can_use_fish_kvcache_attn", lambda **_: False)

    model = _FakeModel()
    assert fish_kvcache_backend.install_fish_kvcache_attn_backend(model) == 1
    wrapped_forward = model.layers[0].self_attn.attn.impl.forward
    assert fish_kvcache_backend.install_fish_kvcache_attn_backend(model) == 0
    assert model.layers[0].self_attn.attn.impl.forward is wrapped_forward

    query = torch.zeros((2, 32, 128), dtype=torch.float16)
    output = torch.zeros_like(query)
    kv_cache = torch.zeros((2, 8, 16, 8, 128), dtype=torch.float16)

    model.layers[0].self_attn.attn.impl.forward(
        None,
        query,
        None,
        None,
        kv_cache,
        _metadata(),
        output,
    )

    assert model.layers[0].self_attn.attn.impl.original_calls == 1


def test_fish_kvcache_backend_falls_back_to_original_forward_on_guard_miss(monkeypatch):
    fish_kvcache_backend.reset_fish_kvcache_attn_stats()
    monkeypatch.setenv("VLLM_OMNI_FISH_KVCACHE_ATTN", "1")
    monkeypatch.setattr(fish_kvcache_backend, "is_available", lambda: True)
    monkeypatch.setattr(fish_kvcache_backend, "load_error", lambda: None)
    monkeypatch.setattr(fish_kvcache_backend, "can_use_fish_kvcache_attn", lambda **_: False)

    decode_calls = []

    def fake_decode(*args, **kwargs):
        del args, kwargs
        decode_calls.append(True)

    monkeypatch.setattr(fish_kvcache_backend, "fish_decode_kvcache_attn", fake_decode)

    model = _FakeModel()
    assert fish_kvcache_backend.install_fish_kvcache_attn_backend(model) == 1

    query = torch.zeros((2, 32, 128), dtype=torch.float16)
    output = torch.zeros_like(query)
    kv_cache = torch.zeros((2, 8, 16, 8, 128), dtype=torch.float16)

    result = model.layers[0].self_attn.attn.impl.forward(
        None,
        query,
        None,
        None,
        kv_cache,
        _metadata(),
        output,
    )

    assert result is output
    assert model.layers[0].self_attn.attn.impl.original_calls == 1
    assert decode_calls == []
    assert torch.equal(output, torch.full_like(output, 2))
    stats = fish_kvcache_backend.get_fish_kvcache_attn_stats()
    assert stats["fallback_count_by_reason"] == {"guard_miss": 1}


def test_fish_kvcache_backend_uses_decode_seq_lens_upper_bound(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_FISH_KVCACHE_ATTN", "1")
    monkeypatch.setattr(fish_kvcache_backend, "is_available", lambda: True)
    monkeypatch.setattr(fish_kvcache_backend, "load_error", lambda: None)

    guard_max_seq_lens = []

    def fake_can_use(**kwargs):
        guard_max_seq_lens.append(kwargs["max_seq_len"])
        return True

    decode_max_seq_lens = []

    def fake_decode(query, key_cache, value_cache, block_table, seq_lens, out, *, scale, max_seq_len):
        del query, key_cache, value_cache, block_table, seq_lens, scale
        decode_max_seq_lens.append(max_seq_len)
        out.fill_(1)
        return out

    monkeypatch.setattr(fish_kvcache_backend, "can_use_fish_kvcache_attn", fake_can_use)
    monkeypatch.setattr(fish_kvcache_backend, "fish_decode_kvcache_attn", fake_decode)

    model = _FakeModel()
    assert fish_kvcache_backend.install_fish_kvcache_attn_backend(model) == 1

    query = torch.zeros((2, 32, 128), dtype=torch.float16)
    output = torch.zeros_like(query)
    kv_cache = torch.zeros((2, 8, 16, 8, 128), dtype=torch.float16)

    model.layers[0].self_attn.attn.impl.forward(
        None,
        query,
        None,
        None,
        kv_cache,
        _metadata(
            max_query_len=1,
            max_seq_len=16384,
            seq_lens_cpu_upper_bound=torch.tensor([512, 768], dtype=torch.int32),
        ),
        output,
    )

    assert guard_max_seq_lens == [1024]
    assert decode_max_seq_lens == [1024]


def test_fish_kvcache_backend_uses_cpu_upper_bound_above_metadata_max(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_FISH_KVCACHE_ATTN", "1")
    monkeypatch.setattr(fish_kvcache_backend, "is_available", lambda: True)
    monkeypatch.setattr(fish_kvcache_backend, "load_error", lambda: None)

    guard_max_seq_lens = []

    def fake_can_use(**kwargs):
        guard_max_seq_lens.append(kwargs["max_seq_len"])
        return True

    monkeypatch.setattr(fish_kvcache_backend, "can_use_fish_kvcache_attn", fake_can_use)
    monkeypatch.setattr(
        fish_kvcache_backend,
        "fish_decode_kvcache_attn",
        lambda query, key_cache, value_cache, block_table, seq_lens, out, *, scale, max_seq_len: out.fill_(1),
    )

    model = _FakeModel()
    assert fish_kvcache_backend.install_fish_kvcache_attn_backend(model) == 1

    query = torch.zeros((2, 32, 128), dtype=torch.float16)
    output = torch.zeros_like(query)
    kv_cache = torch.zeros((2, 8, 16, 8, 128), dtype=torch.float16)

    model.layers[0].self_attn.attn.impl.forward(
        None,
        query,
        None,
        None,
        kv_cache,
        _metadata(
            max_query_len=1,
            max_seq_len=64,
            seq_lens_cpu_upper_bound=torch.tensor([1024, 1024], dtype=torch.int32),
        ),
        output,
    )

    assert guard_max_seq_lens == [1024]


def test_fish_kvcache_backend_slices_upper_bound_by_active_batch(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_FISH_KVCACHE_ATTN", "1")
    monkeypatch.setattr(fish_kvcache_backend, "is_available", lambda: True)
    monkeypatch.setattr(fish_kvcache_backend, "load_error", lambda: None)

    guard_max_seq_lens = []

    def fake_can_use(**kwargs):
        guard_max_seq_lens.append(kwargs["max_seq_len"])
        return True

    def fake_decode(query, key_cache, value_cache, block_table, seq_lens, out, *, scale, max_seq_len):
        del query, key_cache, value_cache, block_table, seq_lens, scale, max_seq_len
        out.fill_(1)
        return out

    monkeypatch.setattr(fish_kvcache_backend, "can_use_fish_kvcache_attn", fake_can_use)
    monkeypatch.setattr(fish_kvcache_backend, "fish_decode_kvcache_attn", fake_decode)

    model = _FakeModel()
    assert fish_kvcache_backend.install_fish_kvcache_attn_backend(model) == 1

    query = torch.zeros((2, 32, 128), dtype=torch.float16)
    output = torch.zeros_like(query)
    kv_cache = torch.zeros((2, 8, 16, 8, 128), dtype=torch.float16)

    model.layers[0].self_attn.attn.impl.forward(
        None,
        query,
        None,
        None,
        kv_cache,
        _metadata(
            batch_size=4,
            num_actual_tokens=2,
            max_query_len=1,
            max_seq_len=16384,
            seq_lens_cpu_upper_bound=torch.tensor([512, 768, 4096, 4096], dtype=torch.int32),
        ),
        output,
    )

    assert guard_max_seq_lens == [1024]


def test_fish_kvcache_backend_falls_back_on_short_upper_bound_shape(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_FISH_KVCACHE_ATTN", "1")
    monkeypatch.setattr(fish_kvcache_backend, "is_available", lambda: True)
    monkeypatch.setattr(fish_kvcache_backend, "load_error", lambda: None)

    guard_max_seq_lens = []

    def fake_can_use(**kwargs):
        guard_max_seq_lens.append(kwargs["max_seq_len"])
        return True

    def fake_decode(query, key_cache, value_cache, block_table, seq_lens, out, *, scale, max_seq_len):
        del query, key_cache, value_cache, block_table, seq_lens, scale, max_seq_len
        out.fill_(1)
        return out

    monkeypatch.setattr(fish_kvcache_backend, "can_use_fish_kvcache_attn", fake_can_use)
    monkeypatch.setattr(fish_kvcache_backend, "fish_decode_kvcache_attn", fake_decode)

    model = _FakeModel()
    assert fish_kvcache_backend.install_fish_kvcache_attn_backend(model) == 1

    query = torch.zeros((2, 32, 128), dtype=torch.float16)
    output = torch.zeros_like(query)
    kv_cache = torch.zeros((2, 8, 16, 8, 128), dtype=torch.float16)

    result = model.layers[0].self_attn.attn.impl.forward(
        None,
        query,
        None,
        None,
        kv_cache,
        _metadata(
            batch_size=2,
            max_query_len=1,
            max_seq_len=16384,
            seq_lens_cpu_upper_bound=torch.tensor([512], dtype=torch.int32),
        ),
        output,
    )

    assert result is output
    assert guard_max_seq_lens == []
    assert model.layers[0].self_attn.attn.impl.original_calls == 1
    assert torch.equal(output, torch.full_like(output, 2))


def test_fish_kvcache_backend_falls_back_on_non_cpu_upper_bound(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_FISH_KVCACHE_ATTN", "1")
    monkeypatch.setattr(fish_kvcache_backend, "is_available", lambda: True)
    monkeypatch.setattr(fish_kvcache_backend, "load_error", lambda: None)

    guard_max_seq_lens = []

    def fake_can_use(**kwargs):
        guard_max_seq_lens.append(kwargs["max_seq_len"])
        return True

    def fake_decode(query, key_cache, value_cache, block_table, seq_lens, out, *, scale, max_seq_len):
        del query, key_cache, value_cache, block_table, seq_lens, scale, max_seq_len
        out.fill_(1)
        return out

    monkeypatch.setattr(fish_kvcache_backend, "can_use_fish_kvcache_attn", fake_can_use)
    monkeypatch.setattr(fish_kvcache_backend, "fish_decode_kvcache_attn", fake_decode)

    model = _FakeModel()
    assert fish_kvcache_backend.install_fish_kvcache_attn_backend(model) == 1

    query = torch.zeros((2, 32, 128), dtype=torch.float16)
    output = torch.zeros_like(query)
    kv_cache = torch.zeros((2, 8, 16, 8, 128), dtype=torch.float16)

    result = model.layers[0].self_attn.attn.impl.forward(
        None,
        query,
        None,
        None,
        kv_cache,
        _metadata(
            batch_size=2,
            max_query_len=1,
            max_seq_len=16384,
            seq_lens_cpu_upper_bound=torch.empty((2,), device="meta", dtype=torch.int32),
        ),
        output,
    )

    assert result is output
    assert guard_max_seq_lens == []
    assert model.layers[0].self_attn.attn.impl.original_calls == 1
    assert torch.equal(output, torch.full_like(output, 2))


def test_fish_kvcache_backend_required_rejects_non_cpu_upper_bound(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_FISH_KVCACHE_ATTN", "required")
    monkeypatch.setattr(fish_kvcache_backend, "is_available", lambda: True)
    monkeypatch.setattr(fish_kvcache_backend, "load_error", lambda: None)
    monkeypatch.setattr(fish_kvcache_backend, "can_use_fish_kvcache_attn", lambda **_: True)

    model = _FakeModel()
    assert fish_kvcache_backend.install_fish_kvcache_attn_backend(model) == 1

    query = torch.zeros((2, 32, 128), dtype=torch.float16)
    output = torch.zeros_like(query)
    kv_cache = torch.zeros((2, 8, 16, 8, 128), dtype=torch.float16)

    with pytest.raises(RuntimeError, match="must be a CPU tensor"):
        model.layers[0].self_attn.attn.impl.forward(
            None,
            query,
            None,
            None,
            kv_cache,
            _metadata(
                batch_size=2,
                max_query_len=1,
                max_seq_len=16384,
                seq_lens_cpu_upper_bound=torch.empty((2,), device="meta", dtype=torch.int32),
            ),
            output,
        )


def test_fish_kvcache_backend_required_rejects_underestimated_upper_bound(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_FISH_KVCACHE_ATTN", "required")
    monkeypatch.setattr(fish_kvcache_backend, "is_available", lambda: True)
    monkeypatch.setattr(fish_kvcache_backend, "load_error", lambda: None)
    monkeypatch.setattr(fish_kvcache_backend, "can_use_fish_kvcache_attn", lambda **_: True)

    model = _FakeModel()
    assert fish_kvcache_backend.install_fish_kvcache_attn_backend(model) == 1

    query = torch.zeros((2, 32, 128), dtype=torch.float16)
    output = torch.zeros_like(query)
    kv_cache = torch.zeros((2, 8, 16, 8, 128), dtype=torch.float16)

    with pytest.raises(RuntimeError, match="underestimates"):
        model.layers[0].self_attn.attn.impl.forward(
            None,
            query,
            None,
            None,
            kv_cache,
            _metadata(
                max_query_len=1,
                max_seq_len=16384,
                seq_lens=torch.tensor([512, 2048], dtype=torch.int32),
                seq_lens_cpu_upper_bound=torch.tensor([512, 768], dtype=torch.int32),
            ),
            output,
        )


def test_fish_kvcache_backend_required_rejects_missing_upper_bound(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_FISH_KVCACHE_ATTN", "required")
    monkeypatch.setattr(fish_kvcache_backend, "is_available", lambda: True)
    monkeypatch.setattr(fish_kvcache_backend, "load_error", lambda: None)
    monkeypatch.setattr(fish_kvcache_backend, "can_use_fish_kvcache_attn", lambda **_: True)

    model = _FakeModel()
    assert fish_kvcache_backend.install_fish_kvcache_attn_backend(model) == 1

    query = torch.zeros((2, 32, 128), dtype=torch.float16)
    output = torch.zeros_like(query)
    kv_cache = torch.zeros((2, 8, 16, 8, 128), dtype=torch.float16)

    with pytest.raises(RuntimeError, match="requires seq_lens_cpu_upper_bound"):
        model.layers[0].self_attn.attn.impl.forward(
            None,
            query,
            None,
            None,
            kv_cache,
            _metadata(max_query_len=1, max_seq_len=16384, seq_lens_cpu_upper_bound=None),
            output,
        )


def test_fish_kvcache_backend_required_rejects_zero_installed_layers(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_FISH_KVCACHE_ATTN", "required")
    monkeypatch.setattr(fish_kvcache_backend, "is_available", lambda: True)
    monkeypatch.setattr(fish_kvcache_backend, "load_error", lambda: None)

    with pytest.raises(RuntimeError, match="installed 0 attention layers"):
        fish_kvcache_backend.install_fish_kvcache_attn_backend(type("EmptyModel", (), {"layers": []})())


def test_fish_kvcache_backend_required_raises_on_guard_miss(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_FISH_KVCACHE_ATTN", "required")
    monkeypatch.setattr(fish_kvcache_backend, "is_available", lambda: True)
    monkeypatch.setattr(fish_kvcache_backend, "load_error", lambda: None)
    monkeypatch.setattr(fish_kvcache_backend, "can_use_fish_kvcache_attn", lambda **_: False)

    model = _FakeModel()
    assert fish_kvcache_backend.install_fish_kvcache_attn_backend(model) == 1

    query = torch.zeros((2, 32, 128), dtype=torch.float16)
    output = torch.zeros_like(query)
    kv_cache = torch.zeros((2, 8, 16, 8, 128), dtype=torch.float16)

    with pytest.raises(RuntimeError, match="required"):
        model.layers[0].self_attn.attn.impl.forward(
            None,
            query,
            None,
            None,
            kv_cache,
            _metadata(),
            output,
        )


def test_fish_kvcache_backend_required_allows_prefill_fallback(monkeypatch):
    fish_kvcache_backend.reset_fish_kvcache_attn_stats()
    monkeypatch.setenv("VLLM_OMNI_FISH_KVCACHE_ATTN", "required")
    monkeypatch.setattr(fish_kvcache_backend, "is_available", lambda: True)
    monkeypatch.setattr(fish_kvcache_backend, "load_error", lambda: None)
    monkeypatch.setattr(fish_kvcache_backend, "can_use_fish_kvcache_attn", lambda **_: False)

    model = _FakeModel()
    assert fish_kvcache_backend.install_fish_kvcache_attn_backend(model) == 1

    query = torch.zeros((2, 32, 128), dtype=torch.float16)
    output = torch.zeros_like(query)
    kv_cache = torch.zeros((2, 8, 16, 8, 128), dtype=torch.float16)

    result = model.layers[0].self_attn.attn.impl.forward(
        None,
        query,
        None,
        None,
        kv_cache,
        _metadata(max_query_len=2, max_seq_len=2),
        output,
    )

    assert result is output
    assert model.layers[0].self_attn.attn.impl.original_calls == 1
    assert torch.equal(output, torch.full_like(output, 2))
    stats = fish_kvcache_backend.get_fish_kvcache_attn_stats()
    assert stats["fallback_count_by_reason"] == {"non_decode": 1}


def test_fish_kvcache_backend_falls_back_without_cpu_upper_bound(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_FISH_KVCACHE_ATTN", "1")
    monkeypatch.setattr(fish_kvcache_backend, "is_available", lambda: True)
    monkeypatch.setattr(fish_kvcache_backend, "load_error", lambda: None)

    guard_max_seq_lens = []

    def fake_can_use(**kwargs):
        guard_max_seq_lens.append(kwargs["max_seq_len"])
        return True

    decode_max_seq_lens = []

    def fake_decode(query, key_cache, value_cache, block_table, seq_lens, out, *, scale, max_seq_len):
        del query, key_cache, value_cache, block_table, seq_lens, scale
        decode_max_seq_lens.append(max_seq_len)
        out.fill_(1)
        return out

    monkeypatch.setattr(fish_kvcache_backend, "can_use_fish_kvcache_attn", fake_can_use)
    monkeypatch.setattr(fish_kvcache_backend, "fish_decode_kvcache_attn", fake_decode)

    model = _FakeModel()
    assert fish_kvcache_backend.install_fish_kvcache_attn_backend(model) == 1

    query = torch.zeros((2, 32, 128), dtype=torch.float16)
    output = torch.zeros_like(query)
    kv_cache = torch.zeros((2, 8, 16, 8, 128), dtype=torch.float16)

    result = model.layers[0].self_attn.attn.impl.forward(
        None,
        query,
        None,
        None,
        kv_cache,
        _metadata(max_query_len=1, max_seq_len=16384, seq_lens_cpu_upper_bound=None),
        output,
    )

    assert result is output
    assert guard_max_seq_lens == []
    assert decode_max_seq_lens == []
    assert model.layers[0].self_attn.attn.impl.original_calls == 1
    assert torch.equal(output, torch.full_like(output, 2))


def test_fish_kvcache_attn_guard_accepts_decode_metadata_above_small_cap(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_FISH_KVCACHE_ATTN", "1")
    monkeypatch.setattr(fish_kvcache_attn, "is_available", lambda: True)

    assert fish_kvcache_attn.can_use_fish_kvcache_attn(
        query=torch.empty((4, 32, 128), dtype=torch.float16),
        key_cache=torch.empty((1024, 16, 8, 128), dtype=torch.float16),
        value_cache=torch.empty((1024, 16, 8, 128), dtype=torch.float16),
        block_table=torch.zeros((4, 1024), dtype=torch.int32),
        seq_lens=torch.ones((4,), dtype=torch.int32),
        max_query_len=1,
        max_seq_len=16384,
        dcp_world_size=1,
        use_cascade=False,
        alibi_slopes=None,
        sliding_window=None,
    )


def test_fish_kvcache_attn_guard_rejects_seq_len_above_block_table_capacity(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_FISH_KVCACHE_ATTN", "1")
    monkeypatch.setattr(fish_kvcache_attn, "is_available", lambda: True)

    assert not fish_kvcache_attn.can_use_fish_kvcache_attn(
        query=torch.empty((4, 32, 128), dtype=torch.float16),
        key_cache=torch.empty((8, 16, 8, 128), dtype=torch.float16),
        value_cache=torch.empty((8, 16, 8, 128), dtype=torch.float16),
        block_table=torch.zeros((4, 8), dtype=torch.int32),
        seq_lens=torch.ones((4,), dtype=torch.int32),
        max_query_len=1,
        max_seq_len=129,
        dcp_world_size=1,
        use_cascade=False,
        alibi_slopes=None,
        sliding_window=None,
    )


def test_fish_kvcache_decode_reuses_long_workspace(monkeypatch):
    fish_kvcache_attn._WORKSPACE_CACHE.clear()
    calls = []

    def fake_decode(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        out,
        scale,
        max_seq_len,
        small_path_max_seq_len,
        long_split_tokens,
        partial_m,
        partial_l,
        partial_acc,
    ):
        del query, key_cache, value_cache, block_table, seq_lens, scale, max_seq_len
        del small_path_max_seq_len, long_split_tokens
        calls.append((partial_m.data_ptr(), partial_l.data_ptr(), partial_acc.data_ptr(), tuple(partial_acc.shape)))
        return out

    monkeypatch.setattr(
        fish_kvcache_attn._triton_backend(),
        "fish_decode_kvcache_attn_triton",
        fake_decode,
    )

    q = torch.zeros((2, 4, 128), dtype=torch.float16)
    k_cache = torch.zeros((128, 16, 2, 128), dtype=torch.float16)
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.zeros((2, 128), dtype=torch.int32)
    seq_lens = torch.full((2,), 2048, dtype=torch.int32)
    out = torch.empty_like(q)

    for _ in range(2):
        fish_kvcache_attn.fish_decode_kvcache_attn(
            q,
            k_cache,
            v_cache,
            block_table,
            seq_lens,
            out,
            scale=128**-0.5,
            max_seq_len=2048,
        )

    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert calls[0][3] == (2, 2, 4, 128)


def test_fish_kvcache_decode_workspace_miss_rejects_cuda_graph_capture(monkeypatch):
    fish_kvcache_attn._WORKSPACE_CACHE.clear()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    q = torch.zeros((2, 4, 128), dtype=torch.float16)

    with pytest.raises(RuntimeError, match="workspace was not prewarmed"):
        fish_kvcache_attn._get_decode_workspace(q, 2048)


def test_fish_kvcache_backend_prewarms_capture_workspaces(monkeypatch):
    fish_kvcache_attn._WORKSPACE_CACHE.clear()
    monkeypatch.setenv("VLLM_OMNI_FISH_KVCACHE_ATTN", "1")
    monkeypatch.setattr(fish_kvcache_backend, "is_available", lambda: True)

    text_config = type(
        "TextConfig",
        (),
        {"num_attention_heads": 4, "head_dim": 128},
    )()
    hf_config = type("HFConfig", (), {"text_config": text_config})()
    model_config = type(
        "ModelConfig",
        (),
        {
            "hf_config": hf_config,
            "max_model_len": 2048,
            "model_arch": "FishSpeechSlowARForConditionalGeneration",
        },
    )()

    assert (
        fish_kvcache_backend.prewarm_fish_kvcache_attn_capture_workspaces(
            model_config=model_config,
            device=torch.device("cpu"),
            dtype=torch.float16,
            capture_sizes=[1, 4],
        )
        == 3
    )

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    q = torch.zeros((4, 4, 128), dtype=torch.float16)
    partial_m, partial_l, partial_acc = fish_kvcache_attn._get_decode_workspace(q, 2048)

    assert tuple(partial_m.shape) == (2, 4, 4)
    assert tuple(partial_l.shape) == (2, 4, 4)
    assert tuple(partial_acc.shape) == (2, 4, 4, 128)


def _reference_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    block_size = int(k_cache.shape[1])
    num_q_heads = int(q.shape[1])
    num_kv_heads = int(k_cache.shape[2])
    outputs = []
    for batch_idx in range(q.shape[0]):
        seq_len = int(seq_lens[batch_idx].item())
        if seq_len <= 0:
            outputs.append(torch.zeros_like(q[batch_idx], dtype=torch.float32))
            continue
        k_rows = []
        v_rows = []
        for logical_block in range((seq_len + block_size - 1) // block_size):
            physical_block = int(block_table[batch_idx, logical_block].item())
            start = logical_block * block_size
            take = min(block_size, seq_len - start)
            k_rows.append(k_cache[physical_block, :take])
            v_rows.append(v_cache[physical_block, :take])
        k = torch.cat(k_rows, dim=0).to(torch.float32)
        v = torch.cat(v_rows, dim=0).to(torch.float32)
        per_head = []
        for q_head in range(num_q_heads):
            kv_head = q_head // (num_q_heads // num_kv_heads)
            scores = torch.matmul(k[:, kv_head, :], q[batch_idx, q_head].to(torch.float32)) * scale
            weights = torch.softmax(scores, dim=0)
            per_head.append(torch.matmul(weights, v[:, kv_head, :]))
        outputs.append(torch.stack(per_head, dim=0))
    return torch.stack(outputs, dim=0).to(q.dtype)


@pytest.mark.parametrize("seq_len", [64, 128, 256, 512, 1024, 1025, 2048])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.skipif(
    not torch.cuda.is_available() or not fish_kvcache_attn.is_available(),
    reason="Fish kvcache Triton attention is not available",
)
def test_fish_kvcache_native_op_matches_reference(seq_len, dtype, batch_size):
    torch.manual_seed(0)
    device = torch.device("cuda")
    block_size = 16
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 128
    max_blocks = (seq_len + block_size - 1) // block_size
    num_blocks = batch_size * max_blocks + 3
    scale = head_dim**-0.5

    q = torch.randn((batch_size, num_q_heads, head_dim), device=device, dtype=dtype)
    k_cache = torch.randn((num_blocks, block_size, num_kv_heads, head_dim), device=device, dtype=dtype)
    v_cache = torch.randn((num_blocks, block_size, num_kv_heads, head_dim), device=device, dtype=dtype)
    block_table = torch.empty((batch_size, max_blocks), device=device, dtype=torch.int32)
    for batch_idx in range(batch_size):
        pages = torch.arange(batch_idx * max_blocks, (batch_idx + 1) * max_blocks, device=device, dtype=torch.int32)
        block_table[batch_idx] = torch.flip(pages, dims=[0])
    seq_lens = torch.full((batch_size,), seq_len, device=device, dtype=torch.int32)

    out = torch.empty_like(q)
    fish_kvcache_attn.fish_decode_kvcache_attn(
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        out,
        scale=scale,
        max_seq_len=seq_len,
    )

    expected = _reference_decode_attention(q, k_cache, v_cache, block_table, seq_lens, scale)
    torch.testing.assert_close(out, expected, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not fish_kvcache_attn.is_available(),
    reason="Fish kvcache Triton attention is not available",
)
def test_fish_kvcache_native_op_zero_seq_len_writes_zero():
    device = torch.device("cuda")
    dtype = torch.float16
    block_size = 16
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 128
    max_seq_len = 512
    max_blocks = max_seq_len // block_size
    scale = head_dim**-0.5

    q = torch.randn((1, num_q_heads, head_dim), device=device, dtype=dtype)
    k_cache = torch.randn((max_blocks, block_size, num_kv_heads, head_dim), device=device, dtype=dtype)
    v_cache = torch.randn_like(k_cache)
    block_table = torch.arange(max_blocks, device=device, dtype=torch.int32).reshape(1, max_blocks)
    seq_lens = torch.zeros((1,), device=device, dtype=torch.int32)
    out = torch.full_like(q, float("nan"))

    fish_kvcache_attn.fish_decode_kvcache_attn(
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        out,
        scale=scale,
        max_seq_len=max_seq_len,
    )

    assert torch.isfinite(out).all()
    assert torch.equal(out, torch.zeros_like(out))


@pytest.mark.skipif(
    not torch.cuda.is_available() or not fish_kvcache_attn.is_available(),
    reason="Fish kvcache Triton attention is not available",
)
def test_fish_kvcache_native_op_handles_mixed_short_and_long_seq_lens():
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float16
    block_size = 16
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 128
    seq_lens = torch.tensor([512, 2048, 1025], device=device, dtype=torch.int32)
    max_seq_len = int(seq_lens.max().item())
    batch_size = int(seq_lens.numel())
    max_blocks = (max_seq_len + block_size - 1) // block_size
    num_blocks = batch_size * max_blocks
    scale = head_dim**-0.5

    q = torch.randn((batch_size, num_q_heads, head_dim), device=device, dtype=dtype)
    k_cache = torch.randn((num_blocks, block_size, num_kv_heads, head_dim), device=device, dtype=dtype)
    v_cache = torch.randn((num_blocks, block_size, num_kv_heads, head_dim), device=device, dtype=dtype)
    block_table = torch.empty((batch_size, max_blocks), device=device, dtype=torch.int32)
    for batch_idx in range(batch_size):
        pages = torch.arange(batch_idx * max_blocks, (batch_idx + 1) * max_blocks, device=device, dtype=torch.int32)
        block_table[batch_idx] = pages

    out = torch.empty_like(q)
    fish_kvcache_attn.fish_decode_kvcache_attn(
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        out,
        scale=scale,
        max_seq_len=max_seq_len,
    )

    expected = _reference_decode_attention(q, k_cache, v_cache, block_table, seq_lens, scale)
    torch.testing.assert_close(out, expected, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not fish_kvcache_attn.is_available(),
    reason="Fish kvcache Triton attention is not available",
)
def test_fish_kvcache_native_op_handles_zero_short_and_long_seq_lens():
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float16
    block_size = 16
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 128
    seq_lens = torch.tensor([0, 512, 2048], device=device, dtype=torch.int32)
    max_seq_len = int(seq_lens.max().item())
    batch_size = int(seq_lens.numel())
    max_blocks = (max_seq_len + block_size - 1) // block_size
    num_blocks = batch_size * max_blocks
    scale = head_dim**-0.5

    q = torch.randn((batch_size, num_q_heads, head_dim), device=device, dtype=dtype)
    k_cache = torch.randn((num_blocks, block_size, num_kv_heads, head_dim), device=device, dtype=dtype)
    v_cache = torch.randn((num_blocks, block_size, num_kv_heads, head_dim), device=device, dtype=dtype)
    block_table = torch.empty((batch_size, max_blocks), device=device, dtype=torch.int32)
    for batch_idx in range(batch_size):
        pages = torch.arange(batch_idx * max_blocks, (batch_idx + 1) * max_blocks, device=device, dtype=torch.int32)
        block_table[batch_idx] = pages
    out = torch.full_like(q, float("nan"))

    fish_kvcache_attn.fish_decode_kvcache_attn(
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        out,
        scale=scale,
        max_seq_len=max_seq_len,
    )

    expected = _reference_decode_attention(q, k_cache, v_cache, block_table, seq_lens, scale)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, expected, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not fish_kvcache_attn.is_available(),
    reason="Fish kvcache Triton attention is not available",
)
def test_fish_kvcache_native_op_can_be_captured_by_cuda_graph():
    device = torch.device("cuda")
    dtype = torch.float16
    q = torch.randn((2, 4, 128), device=device, dtype=dtype)
    k_cache = torch.randn((8, 16, 2, 128), device=device, dtype=dtype)
    v_cache = torch.randn((8, 16, 2, 128), device=device, dtype=dtype)
    block_table = torch.arange(8, device=device, dtype=torch.int32).reshape(2, 4)
    seq_lens = torch.full((2,), 64, device=device, dtype=torch.int32)
    out = torch.empty_like(q)
    expected = torch.empty_like(q)

    fish_kvcache_attn.fish_decode_kvcache_attn(
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        expected,
        scale=128**-0.5,
        max_seq_len=64,
    )
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fish_kvcache_attn.fish_decode_kvcache_attn(
            q,
            k_cache,
            v_cache,
            block_table,
            seq_lens,
            out,
            scale=128**-0.5,
            max_seq_len=64,
        )
    graph.replay()
    torch.accelerator.synchronize()

    torch.testing.assert_close(out, expected, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not fish_kvcache_attn.is_available(),
    reason="Fish kvcache Triton attention is not available",
)
def test_fish_kvcache_native_long_op_can_be_captured_by_cuda_graph():
    device = torch.device("cuda")
    dtype = torch.float16
    seq_len = 2048
    block_size = 16
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 128
    max_blocks = (seq_len + block_size - 1) // block_size

    q = torch.randn((2, num_q_heads, head_dim), device=device, dtype=dtype)
    k_cache = torch.randn((2 * max_blocks, block_size, num_kv_heads, head_dim), device=device, dtype=dtype)
    v_cache = torch.randn((2 * max_blocks, block_size, num_kv_heads, head_dim), device=device, dtype=dtype)
    block_table = torch.arange(2 * max_blocks, device=device, dtype=torch.int32).reshape(2, max_blocks)
    seq_lens = torch.full((2,), seq_len, device=device, dtype=torch.int32)
    out = torch.empty_like(q)
    expected = torch.empty_like(q)

    fish_kvcache_attn.fish_decode_kvcache_attn(
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        expected,
        scale=head_dim**-0.5,
        max_seq_len=seq_len,
    )
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fish_kvcache_attn.fish_decode_kvcache_attn(
            q,
            k_cache,
            v_cache,
            block_table,
            seq_lens,
            out,
            scale=head_dim**-0.5,
            max_seq_len=seq_len,
        )
    graph.replay()
    torch.accelerator.synchronize()

    torch.testing.assert_close(out, expected, atol=3e-2, rtol=3e-2)
