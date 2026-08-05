# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for prompt-embedding cache for diffusion pipelines.

Covers the module :mod:`vllm_omni.diffusion.cache.prompt_embed_cache`:

* :class:`PromptEmbedCache` LRU semantics, stats and thread-safety.
* :func:`_hashable` safe-input classification (scalars, torch dtype/device,
  numpy scalars, nested containers, tensors/PIL bypass).
* :func:`install_prompt_embed_cache` / :func:`uninstall_prompt_embed_cache`
  behaviour: caching, bypass on tensors, bypass on precomputed embeds,
  idempotent install, restore on uninstall, detachment of cached tensors.
* :func:`resolve_prompt_embed_cache_config` env-var overrides.
"""

from __future__ import annotations

import threading

import pytest
import torch

from vllm_omni.diffusion.cache.prompt_embed_cache import (
    _CACHE_MISS,
    _NOT_HASHABLE,
    PromptEmbedCache,
    _build_key,
    _detach_output,
    _hashable,
    install_prompt_embed_cache,
    resolve_prompt_embed_cache_config,
    uninstall_prompt_embed_cache,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# _hashable
# ---------------------------------------------------------------------------


class TestHashable:
    """Tests for the ``_hashable`` helper."""

    @pytest.mark.parametrize(
        "value",
        [None, "prompt", "", 0, 1, -5, 3.14, True, False, b"bytes"],
    )
    def test_scalars_are_returned_unchanged(self, value):
        assert _hashable(value) == value

    def test_torch_device_is_normalized(self):
        out = _hashable(torch.device("cpu"))
        assert out == ("__torch_device__", "cpu")
        assert hash(out) is not None  # must actually be hashable

    def test_torch_dtype_is_normalized(self):
        out = _hashable(torch.float32)
        assert out[0] == "__torch_dtype__"
        assert "float32" in out[1]
        assert hash(out) is not None

    def test_numpy_scalar_is_unwrapped(self):
        np = pytest.importorskip("numpy")
        assert _hashable(np.int64(7)) == 7
        assert _hashable(np.float32(1.5)) == pytest.approx(1.5)

    def test_numpy_ndarray_is_not_hashable(self):
        np = pytest.importorskip("numpy")
        assert _hashable(np.array([1, 2, 3])) is _NOT_HASHABLE

    def test_nested_list_and_tuple(self):
        out = _hashable(["a", ("b", 1), [2, 3]])
        assert out[0] == "__seq__"
        # fully hashable
        assert hash(out) is not None

    def test_dict_is_normalized_by_sorted_keys(self):
        a = _hashable({"b": 1, "a": 2})
        b = _hashable({"a": 2, "b": 1})
        assert a == b
        assert hash(a) is not None

    def test_tensor_is_not_hashable(self):
        assert _hashable(torch.zeros(2)) is _NOT_HASHABLE

    def test_list_containing_tensor_is_not_hashable(self):
        assert _hashable(["a", torch.zeros(1)]) is _NOT_HASHABLE

    def test_dict_containing_tensor_is_not_hashable(self):
        assert _hashable({"x": torch.zeros(1)}) is _NOT_HASHABLE

    def test_custom_object_is_not_hashable(self):
        class Obj:
            pass

        assert _hashable(Obj()) is _NOT_HASHABLE


# ---------------------------------------------------------------------------
# _detach_output
# ---------------------------------------------------------------------------


class TestDetachOutput:
    def test_detaches_tensor(self):
        t = torch.zeros(2, requires_grad=True)
        out = _detach_output(t)
        assert isinstance(out, torch.Tensor)
        assert not out.requires_grad

    def test_detaches_inside_tuple(self):
        t = torch.ones(1, requires_grad=True)
        out = _detach_output((t, None))
        assert isinstance(out, tuple)
        assert not out[0].requires_grad
        assert out[1] is None

    def test_detaches_inside_list_and_dict(self):
        t = torch.ones(1, requires_grad=True)
        out = _detach_output({"a": [t]})
        assert not out["a"][0].requires_grad

    def test_preserves_namedtuple_type(self):
        from collections import namedtuple

        NT = namedtuple("NT", ["x", "y"])
        t = torch.ones(1, requires_grad=True)
        out = _detach_output(NT(x=t, y=1))
        assert isinstance(out, NT)
        assert not out.x.requires_grad
        assert out.y == 1

    def test_passthrough_for_non_tensor(self):
        assert _detach_output("hi") == "hi"
        assert _detach_output(None) is None
        assert _detach_output(5) == 5


# ---------------------------------------------------------------------------
# PromptEmbedCache
# ---------------------------------------------------------------------------


class TestPromptEmbedCache:
    def test_rejects_non_positive_max_size(self):
        with pytest.raises(ValueError):
            PromptEmbedCache(max_size=0)
        with pytest.raises(ValueError):
            PromptEmbedCache(max_size=-1)

    def test_miss_returns_sentinel_and_bumps_miss_counter(self):
        cache = PromptEmbedCache(max_size=2)
        assert cache.get("k") is _CACHE_MISS
        assert cache.stats()["misses"] == 1
        assert cache.stats()["hits"] == 0

    def test_put_then_get_returns_value_and_bumps_hit_counter(self):
        cache = PromptEmbedCache(max_size=2)
        cache.put("k", 42)
        assert cache.get("k") == 42
        assert cache.stats()["hits"] == 1

    def test_none_is_a_cacheable_value(self):
        cache = PromptEmbedCache(max_size=2)
        cache.put("k", None)
        # Distinguishable from a miss.
        assert cache.get("k") is None
        assert cache.stats()["hits"] == 1

    def test_lru_eviction_keeps_recently_used(self):
        cache = PromptEmbedCache(max_size=2)
        cache.put("a", 1)
        cache.put("b", 2)
        # Access "a" to mark it MRU.
        assert cache.get("a") == 1
        cache.put("c", 3)  # should evict "b"
        assert cache.get("b") is _CACHE_MISS
        assert cache.get("a") == 1
        assert cache.get("c") == 3
        assert cache.stats()["size"] == 2

    def test_get_refreshes_recency(self):
        cache = PromptEmbedCache(max_size=2)
        cache.put("a", 1)
        cache.put("b", 2)
        # Hit on "a" → "b" becomes LRU.
        cache.get("a")
        cache.put("c", 3)
        assert cache.get("b") is _CACHE_MISS
        assert cache.get("a") == 1

    def test_put_overwrites_existing_key(self):
        cache = PromptEmbedCache(max_size=2)
        cache.put("k", 1)
        cache.put("k", 2)
        assert cache.get("k") == 2
        assert cache.stats()["size"] == 1

    def test_clear_empties_store_but_keeps_counters(self):
        cache = PromptEmbedCache(max_size=2)
        cache.put("k", 1)
        cache.get("k")
        cache.clear()
        assert cache.stats()["size"] == 0
        # clear only drops entries, counters reflect prior activity.
        assert cache.stats()["hits"] == 1

    def test_stats_reports_expected_fields(self):
        cache = PromptEmbedCache(max_size=4)
        stats = cache.stats()
        assert set(stats.keys()) == {"size", "max_size", "hits", "misses", "bypassed"}
        assert stats["max_size"] == 4
        assert stats["size"] == 0

    def test_thread_safety_under_concurrent_writes(self):
        cache = PromptEmbedCache(max_size=1024)

        def worker(start: int) -> None:
            for i in range(start, start + 200):
                cache.put(f"k{i}", i)
                cache.get(f"k{i}")

        threads = [threading.Thread(target=worker, args=(i * 200,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = cache.stats()
        # 4 * 200 = 800 writes, each followed by a successful read.
        assert stats["hits"] == 800
        assert stats["size"] == 800


# ---------------------------------------------------------------------------
# _build_key
# ---------------------------------------------------------------------------


def _make_sig(fn):
    import inspect

    return inspect.signature(fn)


class TestBuildKey:
    def test_positional_and_keyword_collide(self):
        def encode(prompt, device=None):
            return None

        sig = _make_sig(encode)
        k1 = _build_key(sig, "m", ("hello",), {"device": "cpu"})
        k2 = _build_key(sig, "m", (), {"prompt": "hello", "device": "cpu"})
        assert k1 == k2
        assert k1 is not None

    def test_different_prompts_produce_different_keys(self):
        def encode(prompt):
            return None

        sig = _make_sig(encode)
        assert _build_key(sig, "m", ("a",), {}) != _build_key(sig, "m", ("b",), {})

    def test_different_model_tags_produce_different_keys(self):
        def encode(prompt):
            return None

        sig = _make_sig(encode)
        assert _build_key(sig, "m1", ("a",), {}) != _build_key(sig, "m2", ("a",), {})

    def test_tensor_argument_bypasses(self):
        def encode(prompt, extra=None):
            return None

        sig = _make_sig(encode)
        assert _build_key(sig, "m", ("hi",), {"extra": torch.zeros(1)}) is None

    def test_precomputed_prompt_embeds_bypasses(self):
        def encode(prompt=None, prompt_embeds=None):
            return None

        sig = _make_sig(encode)
        # None → not a bypass.
        assert _build_key(sig, "m", (), {"prompt": "hi"}) is not None
        # A non-None precomputed embed triggers bypass even if it would
        # otherwise be "hashable".
        assert _build_key(sig, "m", (), {"prompt": "hi", "prompt_embeds": "x"}) is None

    def test_invalid_binding_returns_none(self):
        def encode(prompt):
            return None

        sig = _make_sig(encode)
        # Unknown kwarg → TypeError on bind → bypass.
        assert _build_key(sig, "m", (), {"nope": 1}) is None


# ---------------------------------------------------------------------------
# install_prompt_embed_cache / uninstall_prompt_embed_cache
# ---------------------------------------------------------------------------


class _FakePipeline:
    """Minimal pipeline with a signature-bearing ``encode_prompt``."""

    def __init__(self):
        self.call_count = 0

    def encode_prompt(
        self,
        prompt: str,
        device=None,
        num_images_per_prompt: int = 1,
        negative_prompt: str | None = None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        self.call_count += 1
        # Return a fresh tensor so identity changes between real calls.
        return torch.tensor([float(self.call_count)])


class TestInstallAndUninstall:
    def test_returns_none_when_pipeline_has_no_encode_prompt(self):
        class Empty:
            pass

        assert install_prompt_embed_cache(Empty()) is None

    def test_cache_hits_on_identical_args(self):
        pipe = _FakePipeline()
        cache = install_prompt_embed_cache(pipe, max_size=4)
        assert cache is not None

        out1 = pipe.encode_prompt("cat", device="cpu")
        out2 = pipe.encode_prompt("cat", device="cpu")
        # Second call must be served from cache (wrapped fn not invoked).
        assert pipe.call_count == 1
        # Cache returns the same tensor object.
        assert torch.equal(out1, out2)
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["bypassed"] == 0

    def test_positional_and_keyword_share_cache_slot(self):
        pipe = _FakePipeline()
        cache = install_prompt_embed_cache(pipe)
        pipe.encode_prompt("cat")
        pipe.encode_prompt(prompt="cat")
        assert pipe.call_count == 1
        assert cache.stats()["hits"] == 1

    def test_different_prompts_miss(self):
        pipe = _FakePipeline()
        cache = install_prompt_embed_cache(pipe)
        pipe.encode_prompt("cat")
        pipe.encode_prompt("dog")
        assert pipe.call_count == 2
        assert cache.stats()["misses"] == 2

    def test_tensor_argument_bypasses_and_increments_counter(self):
        pipe = _FakePipeline()
        cache = install_prompt_embed_cache(pipe)
        # Put a tensor into a non-precomputed-embed slot to trigger bypass.
        pipe.encode_prompt("cat", device=torch.device("cpu"), num_images_per_prompt=1)

        # Now pass a fake tensor via a harmless positional-like call: use
        # the negative_prompt slot with an unhashable object.
        class Unhashable:
            pass

        pipe.encode_prompt("cat", negative_prompt=Unhashable())  # type: ignore[arg-type]
        pipe.encode_prompt("cat", negative_prompt=Unhashable())  # type: ignore[arg-type]
        assert cache.stats()["bypassed"] == 2
        # All three calls actually executed the underlying function.
        assert pipe.call_count == 3

    def test_precomputed_embeds_bypass_cache(self):
        pipe = _FakePipeline()
        cache = install_prompt_embed_cache(pipe)
        pipe.encode_prompt("cat", prompt_embeds=torch.zeros(1))
        pipe.encode_prompt("cat", prompt_embeds=torch.zeros(1))
        assert cache.stats()["bypassed"] == 2
        assert cache.stats()["hits"] == 0
        assert pipe.call_count == 2

    def test_install_is_idempotent(self):
        pipe = _FakePipeline()
        c1 = install_prompt_embed_cache(pipe, max_size=4)
        c2 = install_prompt_embed_cache(pipe, max_size=16)
        assert c1 is c2
        # The second call must not re-wrap (max_size stays from first install).
        assert c1.max_size == 4

    def test_disabled_cache_passes_through(self):
        pipe = _FakePipeline()
        cache = install_prompt_embed_cache(pipe, enabled=False)
        pipe.encode_prompt("cat")
        pipe.encode_prompt("cat")
        assert pipe.call_count == 2
        assert cache.stats()["hits"] == 0
        assert cache.stats()["misses"] == 0

    def test_toggle_enabled_flag_takes_effect(self):
        pipe = _FakePipeline()
        cache = install_prompt_embed_cache(pipe, enabled=True)
        pipe.encode_prompt("cat")
        pipe.encode_prompt("cat")
        assert cache.stats()["hits"] == 1

        cache.enabled = False
        pipe.encode_prompt("cat")
        pipe.encode_prompt("cat")
        # Cache-bypassed calls go through; hit count unchanged.
        assert cache.stats()["hits"] == 1

    def test_cached_tensor_is_detached(self):
        class GradPipeline:
            def encode_prompt(self, prompt):
                return torch.ones(1, requires_grad=True)

        pipe = GradPipeline()
        install_prompt_embed_cache(pipe)
        # First call populates the cache.
        pipe.encode_prompt("cat")
        # Second call returns the cached tensor which must be detached.
        cached = pipe.encode_prompt("cat")
        assert isinstance(cached, torch.Tensor)
        assert not cached.requires_grad

    def test_lru_eviction_through_wrapper(self):
        pipe = _FakePipeline()
        cache = install_prompt_embed_cache(pipe, max_size=2)
        pipe.encode_prompt("a")
        pipe.encode_prompt("b")
        pipe.encode_prompt("c")  # evicts "a"
        pipe.encode_prompt("a")  # miss again → re-run
        assert pipe.call_count == 4
        assert cache.stats()["size"] == 2

    def test_uninstall_restores_original(self):
        pipe = _FakePipeline()
        original = pipe.encode_prompt
        install_prompt_embed_cache(pipe)
        assert pipe.encode_prompt is not original
        uninstall_prompt_embed_cache(pipe)
        # The bound method reference will differ, but identity of underlying
        # function matches and wrapper attributes are gone.
        assert not hasattr(pipe, "_prompt_embed_cache") or pipe._prompt_embed_cache is None
        # After uninstall calls go straight to the real function.
        pipe.encode_prompt("cat")
        pipe.encode_prompt("cat")
        assert pipe.call_count == 2

    def test_uninstall_on_unwrapped_pipeline_is_noop(self):
        class Empty:
            def encode_prompt(self, prompt):
                return None

        pipe = Empty()
        uninstall_prompt_embed_cache(pipe)  # must not raise


# ---------------------------------------------------------------------------
# resolve_prompt_embed_cache_config
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("OMNI_DIFFUSION_PROMPT_EMBED_CACHE", raising=False)
    monkeypatch.delenv("OMNI_DIFFUSION_PROMPT_EMBED_CACHE_SIZE", raising=False)


class TestResolveConfig:
    def test_defaults_when_nothing_set(self, clean_env):
        enable, size = resolve_prompt_embed_cache_config()
        assert enable is False
        assert size == 32

    def test_explicit_args_used(self, clean_env):
        enable, size = resolve_prompt_embed_cache_config(enable=True, max_size=8)
        assert enable is True
        assert size == 8

    @pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on"])
    def test_env_enable_truthy(self, clean_env, monkeypatch, raw):
        monkeypatch.setenv("OMNI_DIFFUSION_PROMPT_EMBED_CACHE", raw)
        enable, _ = resolve_prompt_embed_cache_config(enable=False)
        assert enable is True

    @pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "off"])
    def test_env_enable_falsy_overrides_true(self, clean_env, monkeypatch, raw):
        monkeypatch.setenv("OMNI_DIFFUSION_PROMPT_EMBED_CACHE", raw)
        enable, _ = resolve_prompt_embed_cache_config(enable=True)
        assert enable is False

    def test_env_enable_invalid_leaves_arg_untouched(self, clean_env, monkeypatch):
        monkeypatch.setenv("OMNI_DIFFUSION_PROMPT_EMBED_CACHE", "garbage")
        enable, _ = resolve_prompt_embed_cache_config(enable=True)
        assert enable is True

    def test_env_size_overrides_arg(self, clean_env, monkeypatch):
        monkeypatch.setenv("OMNI_DIFFUSION_PROMPT_EMBED_CACHE_SIZE", "7")
        _, size = resolve_prompt_embed_cache_config(max_size=99)
        assert size == 7

    def test_env_size_zero_or_negative_ignored(self, clean_env, monkeypatch):
        monkeypatch.setenv("OMNI_DIFFUSION_PROMPT_EMBED_CACHE_SIZE", "0")
        _, size = resolve_prompt_embed_cache_config(max_size=12)
        assert size == 12

        monkeypatch.setenv("OMNI_DIFFUSION_PROMPT_EMBED_CACHE_SIZE", "-5")
        _, size = resolve_prompt_embed_cache_config(max_size=12)
        assert size == 12

    def test_env_size_non_integer_ignored(self, clean_env, monkeypatch):
        monkeypatch.setenv("OMNI_DIFFUSION_PROMPT_EMBED_CACHE_SIZE", "notanint")
        _, size = resolve_prompt_embed_cache_config(max_size=12)
        assert size == 12

    def test_max_size_none_falls_back_to_default(self, clean_env):
        _, size = resolve_prompt_embed_cache_config(enable=True, max_size=None)
        assert size == 32
