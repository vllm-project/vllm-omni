# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SPARSE_ATTN: an in-tree dispatcher to external sparse attention kernels.

vllm-omni owns only this dispatcher (the ``SPARSE_ATTN`` slot in
``DiffusionAttentionBackendEnum``). The actual kernels live in external,
pip-installable packages that register a *callable* via the
``vllm_omni.sparse_attn`` entry-point group::

    [project.entry-points."vllm_omni.sparse_attn"]
    dummy_sparse = "dummy_sparse_attn:attn_fn"

Plugin contract (a callable; device-agnostic self-attention)::

    fn(query, key, value, params: dict, is_causal: bool, softmax_scale: float) -> torch.Tensor

``query``/``key``/``value`` use the BSND layout ``[batch, seq_len, num_heads,
head_dim]`` and the callable returns the same layout. ``is_causal`` and
``softmax_scale`` are the standard attention scalars set by the framework
(``softmax_scale`` defaults to ``head_dim ** -0.5``); kernel-specific knobs and
per-forward state arrive in ``params``.

``params`` is a flat dict merged from three tiers (later wins, ``None`` dropped
so kernels can use ``params.get(key, default)``):

    1. static  — the kernel's configured params (``AttentionSpec.extra["params"]``)
    2. per-forward — canonical :class:`PerForwardState` fields
    3. dynamic — the opaque :attr:`AttentionMetadata.extra`

Well-known ``params`` keys carry attention inputs that change the result and are
forwarded only when set: ``attn_mask`` and ``full_attn_spans``. A plugin that
consumes either must advertise it, or the dispatcher raises rather than let the
kernel ignore it. Advertise via an optional ``capabilities`` dict on the
callable::

    attn_fn.capabilities = {"supports_attention_mask": True}

vllm-omni interprets no kernel-level parameter; the plugin reads what it needs.
"""

from collections.abc import Callable
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)

logger = init_logger(__name__)

SPARSE_ATTN_ENTRYPOINT_GROUP = "vllm_omni.sparse_attn"


def resolve_sparse_attn_plugin(name: str | None) -> Callable[..., torch.Tensor]:
    """Resolve a sparse attention kernel callable by name from entry points.

    Raises ``ValueError`` when ``name`` is missing or no matching plugin is
    installed, and re-raises (with context) when a matching plugin is found but
    fails to import — so a broken plugin is never silently treated as
    "not installed" (and silently degraded to dense).
    """
    if not name:
        raise ValueError(
            "SPARSE_ATTN requires a kernel name. Set it via the per-role "
            "attention config, e.g. extra={'backend': 'dummy_sparse'}."
        )
    from importlib.metadata import entry_points

    eps = list(entry_points(group=SPARSE_ATTN_ENTRYPOINT_GROUP))
    matches = [ep for ep in eps if ep.name.lower() == name.lower()]
    if not matches:
        available = sorted(ep.name for ep in eps)
        raise ValueError(
            f"No sparse attention plugin named {name!r} in entry-point group "
            f"{SPARSE_ATTN_ENTRYPOINT_GROUP!r}. Installed: {available or 'none'}. "
            f"Install the kernel package (e.g. `pip install dummy-sparse-attn`)."
        )
    ep = matches[0]
    try:
        return ep.load()
    except Exception as e:  # found but broken -> fail loud, never silent-dense
        logger.warning("Failed to load sparse attention plugin %r (%s)", name, ep.value, exc_info=True)
        raise ValueError(f"Sparse attention plugin {name!r} failed to import: {e}") from e


def merge_sparse_params(
    static: dict[str, Any] | None,
    per_forward: dict[str, Any] | None,
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    """Flatten the three param tiers into one dict (Policy 1).

    Precedence is static < per-forward < dynamic ``extra`` (later wins), and
    ``None`` values are dropped so the kernel can rely on
    ``params.get(key, default)`` semantics.
    """
    merged = {**(static or {}), **(per_forward or {}), **(extra or {})}
    return {k: v for k, v in merged.items() if v is not None}


class SparseAttentionBackend(AttentionBackend):
    """In-tree dispatcher backend; inherits the conservative ABC defaults.

    ``accept_output_buffer=False`` and the default ``supports_kv_cache_dtype()``
    (no quant) match the sparse callable contract. ``get_supported_head_sizes()
    == []`` means "any" — the external kernel owns its head-size support. Mask
    support is per-plugin: declared via the callable's ``capabilities`` and
    enforced at dispatch (``supports_attention_mask()`` is queried before a
    plugin is resolved, so it stays at the default).
    """

    @staticmethod
    def get_name() -> str:
        return "SPARSE_ATTN"

    @staticmethod
    def get_impl_cls() -> type["_SparseAttentionImpl"]:
        return _SparseAttentionImpl

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return []


class _SparseAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args: Any,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        # backend_kwargs == AttentionSpec.extra for this role. Accept both the
        # nested form {"backend": "dummy_sparse", "params": {"topk": 0.3}} and the
        # flat form {"backend": "dummy_sparse", "topk": 0.3} (unknown keys -> params).
        cfg = extra_impl_args.get("backend_kwargs") or {}
        self._kernel_name: str | None = cfg.get("backend")
        if "params" in cfg:
            self._static_params: dict[str, Any] = dict(cfg["params"] or {})
        else:
            self._static_params = {k: v for k, v in cfg.items() if k != "backend"}
        # Resolve eagerly so a missing/broken plugin fails at model init, not on
        # the first denoise step.
        self._callable = resolve_sparse_attn_plugin(self._kernel_name)
        # Optional capability advertisement on the callable; a missing or
        # malformed attribute degrades to "no capabilities", so semantics-changing
        # inputs get refused rather than silently passed to a kernel that ignores
        # them. Read once here, not per forward.
        raw_caps = getattr(self._callable, "capabilities", None)
        caps = raw_caps if isinstance(raw_caps, dict) else {}
        self._supports_attn_mask = bool(caps.get("supports_attention_mask", False))
        self._supports_full_attn_spans = bool(caps.get("supports_full_attn_spans", False))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        per_forward = (
            attn_metadata.per_forward.to_dict(exclude_none=True)
            if attn_metadata is not None and attn_metadata.per_forward is not None
            else {}
        )
        extra = attn_metadata.extra if attn_metadata is not None else None
        params = merge_sparse_params(self._static_params, per_forward, extra)
        if attn_metadata is not None:
            if attn_metadata.attn_mask is not None:
                if not self._supports_attn_mask:
                    raise ValueError(
                        f"sparse attention plugin {self._kernel_name!r} received an "
                        f"attn_mask but does not advertise 'supports_attention_mask'."
                    )
                params["attn_mask"] = attn_metadata.attn_mask
            if attn_metadata.full_attn_spans is not None:
                if not self._supports_full_attn_spans:
                    raise ValueError(
                        f"sparse attention plugin {self._kernel_name!r} received "
                        f"full_attn_spans but does not advertise 'supports_full_attn_spans'."
                    )
                params["full_attn_spans"] = attn_metadata.full_attn_spans
        return self._callable(query, key, value, params, is_causal=self.causal, softmax_scale=self.softmax_scale)
