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

Video geometry reaches ``params`` as raw, model-agnostic primitives published
generically by the framework (no per-model code): ``latent_shape`` ``(T, H, W)``
and ``patch_size`` ``(p_t, p_h, p_w)``, both per-sample ``[num_samples, 3]``
tensors. The dispatcher's default resolver derives ``total_latent_frames`` /
``patches_per_frame`` from them for plugins that just want frame / patch counts.
The default resolver writes Python ints; when a value instead arrives through
:class:`PerForwardState` or ``extra`` it may be a per-sample tensor — plugins
should accept either (``int(...)`` / ``_row0``-style reads). The resolver only
fills the derived fields when their product equals the actual sequence length
(``query.shape[1]``); under SP auto-padding or for models mixing non-video
tokens into self-attention they are dropped and only the raw primitives remain.
A plugin needing a bespoke layout may expose an optional ``geometry_fn``::

    def geometry_fn(query_shape, params) -> dict: ...
    attn_fn.geometry_fn = geometry_fn

which runs after the default resolver and whose (non-``None``) returns are merged
into ``params``.

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


def _row0(v: Any) -> tuple[int, int, int]:
    """Read the first sample's ``(a, b, c)`` triple from a raw geometry value.

    The bridge ships ``latent_shape`` / ``patch_size`` as ``[num_samples, 3]``
    tensors (homogeneous batch -> equal rows), but also tolerates a plain
    list/tuple so plugins and tests can pass either form.
    """
    if isinstance(v, torch.Tensor):
        row = (v[0] if v.dim() == 2 else v).tolist()
    elif v and isinstance(v[0], (list, tuple)):
        row = list(v[0])
    else:
        row = list(v)
    return int(row[0]), int(row[1]), int(row[2])


def _apply_default_geometry(params: dict[str, Any], seq_len: int | None = None) -> None:
    """Framework default resolver: derive frame / patch counts from raw primitives.

    Fills ``total_latent_frames`` / ``patches_per_frame`` (as Python ints) from
    ``latent_shape`` ``(T, H, W)`` and ``patch_size`` ``(p_t, p_h, p_w)`` using the
    standard factorization ``(T // p_t, (H // p_h) * (W // p_w))``. A no-op when
    the derived value is already present (so a plugin's ``geometry_fn`` or an
    explicit ``extra`` override wins) or when the raw primitives are absent
    (image/audio models, or a model that does not publish geometry).

    When ``seq_len`` is given and the derived product does not equal it — e.g.
    Ulysses SP auto-padding appended pad tokens, or the model's self-attention
    sequence carries non-video tokens alongside the video patches — the derived
    fields are DROPPED (with a one-time warning) instead of handing the kernel a
    (frame, patch) mapping that does not describe the actual sequence. The raw
    primitives stay in ``params`` so a ``geometry_fn`` can still resolve a bespoke
    layout. Mutates ``params``.
    """
    if params.get("total_latent_frames") is not None:
        return
    ls, ps = params.get("latent_shape"), params.get("patch_size")
    if ls is None or ps is None:
        return
    t, h, w = _row0(ls)
    p_t, p_h, p_w = _row0(ps)
    if not (p_t and p_h and p_w):
        return
    total_latent_frames = t // p_t
    patches_per_frame = (h // p_h) * (w // p_w)
    if seq_len is not None and total_latent_frames * patches_per_frame != seq_len:
        logger.warning_once(
            "Derived video geometry (total_latent_frames=%d * patches_per_frame=%d) "
            "does not match the attention sequence length (%d); dropping the derived "
            "fields. Likely causes: SP auto-padding, or non-video tokens in the "
            "self-attention sequence. Raw latent_shape/patch_size are still passed.",
            total_latent_frames,
            patches_per_frame,
            seq_len,
        )
        return
    params["total_latent_frames"] = total_latent_frames
    params["patches_per_frame"] = patches_per_frame


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
    # This dispatcher reads per_forward (step state + raw video geometry) and
    # flattens it into the plugin ``params``, so the Attention layer must build it.
    consumes_per_forward = True

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
        # The plugin contract mandates BSND. Other impls transpose on the model's
        # qkv_layout hint; an external kernel would silently get mislabeled axes,
        # so refuse any non-BSND layout up front.
        qkv_layout = extra_impl_args.get("qkv_layout")
        if qkv_layout not in (None, "BSND"):
            raise ValueError(
                f"SPARSE_ATTN requires the BSND qkv layout; the model declared "
                f"qkv_layout={qkv_layout!r}. Select a dense backend for this layer."
            )
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
        # Optional per-plugin geometry resolver. When present, it runs after the
        # framework default resolver and may override/extend the derived geometry
        # with a bespoke representation (block layout, CSR, ...). Read once here.
        geometry_fn = getattr(self._callable, "geometry_fn", None)
        self._geometry_fn = geometry_fn if callable(geometry_fn) else None

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
        # Geometry: framework default first, then optional per-plugin override.
        # query is BSND, so shape[1] is the sequence length the kernel will see;
        # the resolver drops derived fields that don't factorize it.
        _apply_default_geometry(params, seq_len=query.shape[1])
        if self._geometry_fn is not None:
            geo = self._geometry_fn(query.shape, params) or {}
            params.update({k: v for k, v in geo.items() if v is not None})
        if attn_metadata is not None:
            if attn_metadata.attn_mask is not None:
                if not self._supports_attn_mask:
                    raise ValueError(
                        f"sparse attention plugin {self._kernel_name!r} received an "
                        f"attn_mask but does not advertise 'supports_attention_mask'. "
                        f"Note: Ulysses SP auto-padding injects an attn_mask when the "
                        f"sequence length is not divisible by the SP world size."
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
