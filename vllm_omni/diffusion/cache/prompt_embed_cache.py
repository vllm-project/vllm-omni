# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Prompt-embedding cache for diffusion pipelines.

Text encoders are the single most expensive per-request preprocessing step in
most diffusion pipelines, and their inputs are frequently identical across
requests (e.g. in GRPO-style rollouts the same prompt is submitted many times
with different seeds to sample different images). Re-running the text encoder
for each of those requests wastes compute and GPU memory.

This module provides a small LRU cache plus a transparent wrapper for a
pipeline's ``encode_prompt`` method. Because almost every diffusion pipeline in
``vllm_omni/diffusion/models`` routes all text-encoder invocations through
``self.encode_prompt(...)``, wrapping that single method is sufficient to
cache results model-wide without per-pipeline edits.

Design points:
    * The wrapper is installed by :class:`DiffusionModelRunner` after the
      pipeline has loaded, so each runner process owns its own cache.
    * Cache keys are derived from the bound ``encode_prompt`` arguments. Only
      inputs we can safely hash (``str`` / ``int`` / ``float`` / ``bool`` /
      ``None`` / ``bytes`` / ``torch.device`` / ``torch.dtype`` / numpy
      scalars / nested lists/tuples/dicts of those) participate in the key.
      If any argument is a tensor, PIL image, or other non-trivial object, we
      bypass the cache for that call to guarantee correctness.
    * If a caller passes precomputed ``*_embeds`` into ``encode_prompt`` the
      wrapper also bypasses the cache because the call is already short-circuit.
    * Cache values are detached tensors. Downstream pipeline code typically
      does non-inplace ops (``.repeat``, ``.view``, slicing); we do not clone
      on hit since outputs are treated as read-only.
"""

from __future__ import annotations

import functools
import inspect
import os
import threading
from collections import OrderedDict
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# Sentinel returned by :func:`_hashable` when an input cannot be hashed
# safely (e.g. tensor / PIL image / custom object). Any such input disables
# caching for that specific call.
_NOT_HASHABLE = object()

# Sentinel returned by :meth:`PromptEmbedCache.get` to distinguish a cache
# miss from a legitimate cached value of ``None``.
_CACHE_MISS = object()

# Argument names whose presence of a non-None value means the caller is
# providing precomputed embeddings. In every pipeline we inspected, passing
# these makes ``encode_prompt`` a cheap passthrough, so caching it has no
# benefit and just wastes memory on large tensor keys.
_PRECOMPUTED_EMBED_ARGS = (
    "prompt_embeds",
    "negative_prompt_embeds",
    "pooled_prompt_embeds",
    "negative_pooled_prompt_embeds",
    "prompt_embeds_mask",
    "negative_prompt_embeds_mask",
)


def _hashable(obj: Any) -> Any:
    """Convert *obj* into a hashable representation or return ``_NOT_HASHABLE``.

    Only inputs whose textual / scalar identity fully determines the
    ``encode_prompt`` output are considered safe. Tensors and PIL images (which
    flow through e.g. image-edit pipelines) deliberately disable caching for
    that call rather than risk stale results. Common value-like configuration
    objects such as ``torch.device`` and ``torch.dtype`` are normalized to
    stable string forms so they can participate in cache keys safely.
    """
    if obj is None or isinstance(obj, (str, int, float, bool, bytes)):
        return obj
    if isinstance(obj, torch.device):
        return ("__torch_device__", str(obj))
    if isinstance(obj, torch.dtype):
        return ("__torch_dtype__", str(obj))
    if type(obj).__module__.split(".", 1)[0] == "numpy" and hasattr(obj, "item"):
        try:
            return _hashable(obj.item())
        except (ValueError, TypeError):
            return _NOT_HASHABLE
    if isinstance(obj, (list, tuple)):
        out = []
        for item in obj:
            h = _hashable(item)
            if h is _NOT_HASHABLE:
                return _NOT_HASHABLE
            out.append(h)
        return ("__seq__", tuple(out))
    if isinstance(obj, dict):
        items = []
        for k in sorted(obj.keys(), key=repr):
            v = _hashable(obj[k])
            if v is _NOT_HASHABLE:
                return _NOT_HASHABLE
            items.append((repr(k), v))
        return ("__dict__", tuple(items))
    # torch.Tensor, PIL.Image, np.ndarray, and any other object → not safe.
    return _NOT_HASHABLE


def _detach_output(value: Any) -> Any:
    """Detach tensors in the return value so the cache never holds grad state."""
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, tuple):
        detached = tuple(_detach_output(v) for v in value)
        # Preserve namedtuple types so attribute access keeps working.
        if hasattr(value, "_fields"):
            return type(value)(*detached)
        return detached
    if isinstance(value, list):
        return [_detach_output(v) for v in value]
    if isinstance(value, dict):
        return {k: _detach_output(v) for k, v in value.items()}
    return value


class PromptEmbedCache:
    """Thread-safe LRU cache for ``encode_prompt`` outputs.

    The cache stores whatever ``encode_prompt`` returns (tensor, tuple of
    tensors, ``None``, etc.). Lookup / insertion is O(1) amortized. Eviction
    is least-recently-used.
    """

    def __init__(self, max_size: int = 32, enabled: bool = True) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self.max_size = max_size
        self.enabled = enabled
        self._store: OrderedDict[Any, Any] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.bypassed = 0

    def get(self, key: Any) -> Any:
        """Return the cached value or ``_CACHE_MISS`` if absent.

        Returning a sentinel (rather than ``None``) lets callers cache
        legitimate ``None`` results from the wrapped function.
        """
        with self._lock:
            if key not in self._store:
                self.misses += 1
                return _CACHE_MISS
            self._store.move_to_end(key)
            self.hits += 1
            return self._store[key]

    def put(self, key: Any, value: Any) -> None:
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "size": len(self._store),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "bypassed": self.bypassed,
            }


def _build_key(
    signature: inspect.Signature,
    model_tag: str,
    args: tuple,
    kwargs: dict,
) -> Any:
    """Return a hashable cache key or ``None`` if this call must bypass cache.

    We bind the raw positional / keyword arguments to the method signature so
    that keyword and positional invocations with the same underlying values
    collide in the cache.
    """
    try:
        bound = signature.bind_partial(*args, **kwargs)
    except TypeError:
        return None
    bound.apply_defaults()

    # Bypass when the caller supplies precomputed embeddings.
    for name in _PRECOMPUTED_EMBED_ARGS:
        if name in bound.arguments and bound.arguments[name] is not None:
            return None

    items: list[tuple[str, Any]] = []
    for name in sorted(bound.arguments.keys()):
        h = _hashable(bound.arguments[name])
        if h is _NOT_HASHABLE:
            return None
        items.append((name, h))
    return (model_tag, tuple(items))


def install_prompt_embed_cache(
    pipeline: Any,
    *,
    max_size: int = 32,
    enabled: bool = True,
    model_tag: str | None = None,
) -> PromptEmbedCache | None:
    """Wrap ``pipeline.encode_prompt`` so results are cached by argument identity.

    Idempotent: calling twice on the same pipeline is a no-op and returns the
    existing cache. Returns ``None`` if the pipeline has no ``encode_prompt``
    method.
    """
    existing = getattr(pipeline, "_prompt_embed_cache", None)
    if existing is not None:
        return existing

    encode_fn = getattr(pipeline, "encode_prompt", None)
    if encode_fn is None or not callable(encode_fn):
        logger.debug(
            "Prompt-embed cache: pipeline %s has no encode_prompt(); skipping.",
            type(pipeline).__name__,
        )
        return None

    try:
        signature = inspect.signature(encode_fn)
    except (TypeError, ValueError) as e:
        logger.warning(
            "Prompt-embed cache: cannot introspect encode_prompt on %s (%s); skipping.",
            type(pipeline).__name__,
            e,
        )
        return None

    cache = PromptEmbedCache(max_size=max_size, enabled=enabled)
    tag = model_tag or type(pipeline).__name__

    @functools.wraps(encode_fn)
    def cached_encode_prompt(*args: Any, **kwargs: Any) -> Any:
        if not cache.enabled:
            return encode_fn(*args, **kwargs)
        key = _build_key(signature, tag, args, kwargs)
        if key is None:
            with cache._lock:
                cache.bypassed += 1
            return encode_fn(*args, **kwargs)
        hit = cache.get(key)
        if hit is not _CACHE_MISS:
            return hit
        out = encode_fn(*args, **kwargs)
        cache.put(key, _detach_output(out))
        return out

    cached_encode_prompt.__wrapped__ = encode_fn  # type: ignore[attr-defined]
    # Use object.__setattr__ to bypass diffusers ModelMixin attribute checks.
    try:
        pipeline.encode_prompt = cached_encode_prompt  # type: ignore[assignment]
    except Exception:
        object.__setattr__(pipeline, "encode_prompt", cached_encode_prompt)
    object.__setattr__(pipeline, "_prompt_embed_cache", cache)
    object.__setattr__(pipeline, "_prompt_embed_cache_original_fn", encode_fn)

    logger.info(
        "Prompt-embed cache installed on %s (max_size=%d, enabled=%s).",
        tag,
        max_size,
        enabled,
    )
    return cache


def uninstall_prompt_embed_cache(pipeline: Any) -> None:
    """Restore the original ``encode_prompt`` on *pipeline* if wrapped."""
    original = getattr(pipeline, "_prompt_embed_cache_original_fn", None)
    if original is None:
        return
    try:
        pipeline.encode_prompt = original  # type: ignore[assignment]
    except Exception:
        object.__setattr__(pipeline, "encode_prompt", original)
    for attr in ("_prompt_embed_cache", "_prompt_embed_cache_original_fn"):
        if hasattr(pipeline, attr):
            try:
                delattr(pipeline, attr)
            except Exception:
                object.__setattr__(pipeline, attr, None)


def resolve_prompt_embed_cache_config(
    enable: bool | None = None,
    max_size: int | None = None,
) -> tuple[bool, int]:
    """Combine explicit args with env-var overrides.

    Environment variables (useful for quick enablement in GRPO jobs without
    touching config files):

        ``OMNI_DIFFUSION_PROMPT_EMBED_CACHE`` (``1``/``0``/``true``/``false``)
        ``OMNI_DIFFUSION_PROMPT_EMBED_CACHE_SIZE`` (positive int)
    """
    env_enable = os.environ.get("OMNI_DIFFUSION_PROMPT_EMBED_CACHE")
    if env_enable is not None:
        parsed = env_enable.strip().lower()
        if parsed in ("1", "true", "yes", "on"):
            enable = True
        elif parsed in ("0", "false", "no", "off"):
            enable = False

    env_size = os.environ.get("OMNI_DIFFUSION_PROMPT_EMBED_CACHE_SIZE")
    if env_size is not None:
        try:
            env_size_int = int(env_size)
            if env_size_int > 0:
                max_size = env_size_int
        except ValueError:
            logger.warning(
                "Ignoring non-integer OMNI_DIFFUSION_PROMPT_EMBED_CACHE_SIZE=%r.",
                env_size,
            )

    return bool(enable), int(max_size or 32)
