# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) Microsoft Corporation and Jiarui Fang
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team & Jiarui Fang
# Adapted from
# https://github.com/feifeibear/long-context-attention/blob/main/yunchang/attention/layer.py


import functools
import itertools
import json
import weakref
from collections.abc import Callable, Mapping
from dataclasses import MISSING, dataclass, fields, replace
from typing import TYPE_CHECKING, Any, TypeGuard, cast

import torch
import torch.nn as nn
from torch.library import Library
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    VideoTokenLayout,
)
from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend, SDPAImpl
from vllm_omni.diffusion.attention.backends.trtllm_calibration import (
    calibration_for_candidate,
    layer_calibration_curve,
    layer_calibration_is_ignored,
    resolve_effective_calibration,
)
from vllm_omni.diffusion.attention.capabilities import (
    AttentionComputation,
    CapabilityResult,
    CrossAttentionParallelLayout,
    ExecutionContext,
    ExecutionPathResult,
    OuterBoundary,
    ParallelStrategy,
)
from vllm_omni.diffusion.attention.parallel import build_parallel_attention_strategy
from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
from vllm_omni.diffusion.attention.parallel.ring import RingParallelAttention
from vllm_omni.diffusion.attention.selector import get_attn_backend_for_role
from vllm_omni.diffusion.config import get_current_diffusion_config_or_none
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.diffusion_kv.layout import assert_backend_layout_supported
from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelInput
from vllm_omni.diffusion.forward_context import (
    get_attention_schedule_profile,
    get_forward_context,
    get_ulysses_mode,
    is_forward_context_available,
)
from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import AttentionScheduleConfig, AttentionSpec, OmniDiffusionConfig

logger = init_logger(__name__)


def _canonical_json(value: object) -> str | object:
    """Order-insensitive canonical form for dedup keys; ``""`` for None or empty containers.

    An unserializable value is not shareable, so it gets a fresh sentinel that equals nothing else.
    ``repr`` is not used: two different objects can share a repr and would otherwise collapse onto
    one impl.
    """
    if not value:
        return ""
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return object()


def _attention_identity(
    backend_cls: type[AttentionBackend] | None, backend_explicit: bool, spec: "AttentionSpec | None"
) -> tuple:
    """Backend name, explicit-vs-default selection and canonical backend kwargs.

    The ring runner is bound to the baseline's backend preference and ignores backend kwargs, and
    paged attention runs the native implementation bound to the baseline, so a candidate can run
    on the ring or paged path only when this identity equals the baseline's. An unserializable
    kwargs value never matches.
    """
    return (
        backend_cls.get_name() if backend_cls is not None else None,
        bool(backend_explicit),
        _canonical_json(spec.backend_kwargs() if spec is not None else None),
    )


def _promote_marked_paged_default(
    backend_cls: type[AttentionBackend],
    spec: "AttentionSpec | None",
    *,
    scheduler_paged_kv: bool,
    paged_kv_cache_role: str | None,
) -> type[AttentionBackend]:
    """Return FLASH_ATTN for a marked paged layer whose platform default lacks paged KV support.

    The promotion applies only when the Scheduler owns paged KV (PAGED_SCHEDULER mode), the layer has
    a paged KV cache role and no explicit spec chose ``backend_cls``; otherwise ``backend_cls`` is
    returned unchanged.
    """
    if scheduler_paged_kv and paged_kv_cache_role is not None and spec is None and not backend_cls.supports_paged_kv:
        # FLASH_ATTN is an Omni selector, not a device-specific kernel.
        # vLLM resolves it to CUDA FlashAttention on GPU and Ascend native
        # attention on NPU. A platform may legitimately default dense
        # attention to CUDNN/SDPA (for example when optional dense FA
        # extras are absent), but a paged layer must advertise the
        # capability so the Worker can register its native cache view.
        # Load that selector from the registry: this is not a user-explicit
        # dense FLASH_ATTN request, so skip platform explicit-backend
        # validation (Blackwell would otherwise require CuTe FA4). Formal
        # paged execution still delegates to vLLM's native paged backend.
        return DiffusionAttentionBackendEnum.FLASH_ATTN.get_class()
    return backend_cls


def _reject_allgather_trtllm(
    backend_cls: type[AttentionBackend], skip_sequence_parallel: bool, allgather_degree: int
) -> None:
    # TODO: Move AllGather-KV compatibility into an AttentionBackend capability
    # so validation does not depend on backend names.
    if not skip_sequence_parallel and allgather_degree > 1 and backend_cls.get_name() == "TRTLLM_ATTN":
        raise ValueError(
            "TRTLLM_ATTN does not support AllGather-KV sequence parallelism. "
            "Set --allgather-degree 1 or select another diffusion attention backend."
        )


def _select_impl_cls(
    backend_cls: type[AttentionBackend], impl_overrides: Mapping[str, type[AttentionImpl]] | None
) -> type[AttentionImpl]:
    impl_cls = backend_cls.get_impl_cls()
    override = impl_overrides.get(backend_cls.get_name()) if impl_overrides is not None else None
    if override is None:
        return impl_cls
    if not issubclass(override, impl_cls):
        raise TypeError(
            f"Attention implementation override {override.__qualname__} must subclass "
            f"the selected implementation {impl_cls.__qualname__} "
            f"for backend {backend_cls.__qualname__}"
        )
    return override


# Layers built with a startup schedule, keyed by the integer id the scheduled-attention op carries.
# Weak values, so the registry never keeps a dropped model alive.
_SCHEDULED_ATTENTION_LAYERS: "weakref.WeakValueDictionary[int, Attention]" = weakref.WeakValueDictionary()
_SCHEDULED_ATTENTION_IDS = itertools.count()

# AttentionMetadata fields the scheduled-attention op carries (see _encode_scheduled_op_metadata).
_SCHEDULED_OP_CARRIED_FIELDS = frozenset({"attn_mask", "extra", "video_layout"})
# A field without a declared default never matches, so it keeps the eager boundary.
_NO_FIELD_DEFAULT = object()


def _scheduled_op_field_defaults() -> tuple[tuple[str, object], ...]:
    """Every other AttentionMetadata field with its declared default.

    Derived from the dataclass, so a field added later keeps the eager boundary until the op carries
    it, instead of being dropped by ``_decode_scheduled_op_metadata``.
    """
    defaults = []
    for metadata_field in fields(AttentionMetadata):
        if metadata_field.name in _SCHEDULED_OP_CARRIED_FIELDS:
            continue
        if metadata_field.default is not MISSING:
            default = metadata_field.default
        elif metadata_field.default_factory is not MISSING:
            default = metadata_field.default_factory()
        else:
            default = _NO_FIELD_DEFAULT
        defaults.append((metadata_field.name, default))
    return tuple(defaults)


# Uncarried fields that must hold their default for a compiled call to use the scheduled-attention
# op; any other value keeps the eager boundary.
_SCHEDULED_OP_FIELD_DEFAULTS = _scheduled_op_field_defaults()


def _holds_field_default(value: object, default: object) -> bool:
    if value is default:
        return True
    # A tensor never compares equal to a default; the type check also keeps bool apart from int.
    return (
        default is not _NO_FIELD_DEFAULT
        and type(value) is type(default)
        and not isinstance(value, torch.Tensor)
        and value == default
    )


# Spec of a call without metadata; any other spec starts with "M".
_NO_METADATA_SPEC = "-"


def _is_op_int(value: object) -> TypeGuard[int | torch.SymInt]:
    # A bool is an int subclass; bools travel in the spec instead.
    return (isinstance(value, int) and not isinstance(value, bool)) or isinstance(value, torch.SymInt)


def _encode_scheduled_op_metadata(
    attn_metadata: AttentionMetadata | None,
) -> tuple[str, list[int | torch.SymInt]] | None:
    """Split metadata into a structure string and its integers for ``vllm_omni::scheduled_attention``.

    Runs while tracing. The string holds only names, types and booleans, which are the same on every
    call from one call site, so it never adds a guard that a new shape or step would fail. The
    integers may be symbolic sizes; they become ``SymInt[]`` operands, so a new resolution does not
    recompile the block either. ``attn_mask`` travels as its own tensor operand.

    Carried: ``video_layout`` with plain ``prefix_len``/``latent_grid``/``used_len`` and no
    ``video_spans``, and ``extra`` entries whose values are None, a bool, an int or a tuple/list of
    ints. Returns None for anything else (tensors, floats or strings in ``extra``, video spans, joint
    tensors, packed padding, ranges, cross-attention mask metadata, a non-default ``joint_strategy``, any
    other field off its declared default, a metadata subclass), which keeps the eager boundary.
    """
    if attn_metadata is None:
        return _NO_METADATA_SPEC, []
    if type(attn_metadata) is not AttentionMetadata:
        return None
    for name, default in _SCHEDULED_OP_FIELD_DEFAULTS:
        if not _holds_field_default(getattr(attn_metadata, name), default):
            return None
    spec = "M"
    ints: list[int | torch.SymInt] = []
    layout = attn_metadata.video_layout
    if layout is not None:
        if type(layout) is not VideoTokenLayout or layout.video_spans:
            return None
        spec += ";L"
        for field_value in (layout.prefix_len, layout.used_len):
            if field_value is None:
                spec += "n"
            elif _is_op_int(field_value):
                spec += "i"
                ints.append(field_value)
            else:
                return None
        grid = layout.latent_grid
        if grid is None:
            spec += "n"
        elif isinstance(grid, tuple) and len(grid) == 3 and all(_is_op_int(dim) for dim in grid):
            spec += "g"
            ints.extend(grid)
        else:
            return None
    for key in sorted(attn_metadata.extra):
        if not isinstance(key, str) or not key or ";" in key or "=" in key:
            return None
        value = attn_metadata.extra[key]
        if value is None:
            kind = "n"
        elif isinstance(value, bool):
            kind = "b1" if value else "b0"
        elif _is_op_int(value):
            kind = "i"
            ints.append(value)
        elif isinstance(value, (tuple, list)) and all(_is_op_int(item) for item in value):
            kind = ("t" if isinstance(value, tuple) else "l") + str(len(value))
            ints.extend(value)
        else:
            return None
        spec += ";X" + key + "=" + kind
    return spec, ints


@functools.lru_cache(maxsize=256)
def _parse_scheduled_op_spec(spec: str) -> tuple[tuple[str, ...] | None, tuple[tuple[str, str, int], ...]]:
    """Parse a spec from ``_encode_scheduled_op_metadata`` once per call site.

    Returns the layout field codes (or None) and, per ``extra`` key, ``(key, kind, int count)``.
    """
    if not spec.startswith("M"):
        raise RuntimeError(f"malformed scheduled attention metadata spec {spec!r}")
    layout: tuple[str, ...] | None = None
    extra: list[tuple[str, str, int]] = []
    for part in spec[1:].split(";")[1:]:
        if part.startswith("L") and len(part) == 4 and layout is None:
            layout = tuple(part[1:])
            continue
        key, sep, kind = part[1:].partition("=")
        if not part.startswith("X") or not sep:
            raise RuntimeError(f"malformed scheduled attention metadata spec {spec!r}")
        if kind == "i":
            count = 1
        elif kind[:1] in ("t", "l"):
            count = int(kind[1:])
        else:
            count = 0
        extra.append((key, kind, count))
    return layout, tuple(extra)


def _decode_scheduled_op_metadata(
    attn_mask: torch.Tensor | None, spec: str, ints: list[int]
) -> AttentionMetadata | None:
    """Rebuild the metadata ``_encode_scheduled_op_metadata`` split, with fresh containers."""
    if spec == _NO_METADATA_SPEC:
        return None
    layout_codes, extra_codes = _parse_scheduled_op_spec(spec)
    values = iter(ints)
    video_layout = None
    if layout_codes is not None:
        prefix_code, used_code, grid_code = layout_codes
        prefix_len = next(values) if prefix_code == "i" else None
        used_len = next(values) if used_code == "i" else None
        grid = (next(values), next(values), next(values)) if grid_code == "g" else None
        video_layout = VideoTokenLayout(prefix_len=prefix_len, latent_grid=grid, used_len=used_len)
    extra: dict[str, Any] = {}
    for key, kind, count in extra_codes:
        if kind == "n":
            extra[key] = None
        elif kind in ("b0", "b1"):
            extra[key] = kind == "b1"
        elif kind == "i":
            extra[key] = next(values)
        else:
            items = [next(values) for _ in range(count)]
            extra[key] = tuple(items) if kind[0] == "t" else items
    if next(values, None) is not None:
        raise RuntimeError(f"scheduled attention metadata spec {spec!r} does not match {len(ints)} integers")
    return AttentionMetadata(attn_mask=attn_mask, extra=extra, video_layout=video_layout)


def _run_scheduled_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
    layer_id: torch.Tensor,
    metadata_spec: str,
    metadata_ints: list[int],
) -> torch.Tensor:
    """Eager body of ``vllm_omni::scheduled_attention``: the layer's full ``_forward_impl``.

    Candidate selection, capability checks, backend-private gates and the parallel strategy run
    here at execution time, exactly as behind ``_forward_schedule_compile_boundary``, so the step,
    sigma and timestep never reach Dynamo.
    """
    layer = _SCHEDULED_ATTENTION_LAYERS.get(int(layer_id))
    if layer is None:
        raise RuntimeError(f"scheduled attention layer {int(layer_id)} is not registered")
    metadata = _decode_scheduled_op_metadata(attn_mask, metadata_spec, metadata_ints)
    out = layer._forward_impl(query, key, value, metadata)
    expected = (*query.shape[:-1], value.shape[-1])
    if tuple(out.shape) != expected or out.dtype != query.dtype or out.device != query.device:
        raise RuntimeError(
            f"scheduled attention on layer {layer.prefix!r} returned {tuple(out.shape)} {out.dtype} on {out.device}; "
            f"the compiled graph expects {expected} {query.dtype} on {query.device}"
        )
    # Compiled code relies on the fake's contiguous strides, and a custom op may not return a view
    # of one of its inputs.
    out = out.contiguous()
    if any(out.untyped_storage().data_ptr() == t.untyped_storage().data_ptr() for t in (query, key, value)):
        out = out.clone()
    return out


def _scheduled_attention_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
    layer_id: torch.Tensor,
    metadata_spec: str,
    metadata_ints: list[int],
) -> torch.Tensor:
    return query.new_empty((*query.shape[:-1], value.shape[-1]))


# direct_register_custom_op skips torch.library.custom_op's Python dispatch (~21us/call).
# CompositeExplicitAutograd stays device-agnostic (CPU layer id beside device Q/K/V).
# No autograd formula; Attention._schedule_op_eligible keeps gradient calls eager.
_SCHEDULED_ATTENTION_LIB = Library("vllm_omni", "FRAGMENT")
if not hasattr(torch.ops.vllm_omni, "scheduled_attention"):
    direct_register_custom_op(
        op_name="scheduled_attention",
        op_func=_run_scheduled_attention,
        mutates_args=[],
        fake_impl=_scheduled_attention_fake,
        target_lib=_SCHEDULED_ATTENTION_LIB,
        dispatch_key="CompositeExplicitAutograd",
    )


_scheduled_attention_op = torch.ops.vllm_omni.scheduled_attention


def _try_extract_layer_index(prefix: str) -> int | None:
    if not prefix:
        return None
    try:
        return extract_layer_index(prefix)
    except (AssertionError, ValueError):
        return None


@dataclass(frozen=True)
class ScheduleCandidate:
    """A pre-constructed per-profile attention candidate.

    Carries the resolved backend class, spec, constructed impl and explicit-selection flag so
    calibration, startup capability validation and runtime selection can
    use them without re-resolving. Deduplicated profiles share one instance, so the
    record intentionally carries no single profile name (the dict key is the name).
    """

    backend_cls: type[AttentionBackend]
    spec: "AttentionSpec | None"
    impl: AttentionImpl
    backend_explicit: bool


class Attention(nn.Module):
    _scheduler_paged_kv = False
    _has_custom_attention = False
    # Whether this layer saw a schedule at construction. Attention can be built with no current
    # diffusion config, and such a layer legitimately prepares no candidates, so startup
    # validation must not report it as a missing profile.
    _schedule_configured = False
    # Id the compiled scheduled-attention op uses to find this layer (see _register_scheduled_op).
    _schedule_op_id: torch.Tensor | None = None

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        # Per-role backend selection (RFC: per-role attention backend)
        role: str = "self",
        role_category: str | None = None,
        # Model-defined Q/K/V tensor layout hint for backend execution.
        qkv_layout: str | None = None,
        # ulysses attention
        scatter_idx: int = 2,
        gather_idx: int = 1,
        use_sync: bool = False,
        skip_sequence_parallel: bool = False,
        # Opt-out for KV-cache quantization at this specific attention layer.
        # Set by the model author when quant is known to degrade quality or
        # perf for this layer (e.g. Wan2.2 cross-attn has short sequences and
        # block-FP8 quant offers no win). Default False = follow global config.
        disable_kv_quant: bool = False,
        # Opt-in marker for Scheduler-managed paged KV. Unmarked diffusion
        # attention remains dense and contributes no native KVCacheSpec.
        paged_kv_cache_role: str | None = None,
        paged_kv_cache_dtype: torch.dtype | None = None,
        # Model-owned kernel for architectures whose attention contract cannot
        # be represented by the generic backend interface (for example packed
        # varlen attention with learned sink logits). The shared Attention
        # layer still owns parallel dispatch and compile boundaries.
        custom_attention: nn.Module | None = None,
        # Preserve dense FP32 inference for models opting into CUDA auto fallback.
        allow_fp32_fallback: bool = False,
        # Model-owned implementation subclasses, keyed by backend name.
        # Keep the platform-selected backend and its capabilities unchanged.
        # Each override must preserve the selected implementation's contract.
        impl_overrides: Mapping[str, type[AttentionImpl]] | None = None,
        # Computation semantics are independent of the role used for config.
        # Omitted for legacy callers; new cross-attention callers set it.
        computation: AttentionComputation | str | None = None,
        cross_parallel_layout: CrossAttentionParallelLayout | str | None = None,
        cross_mask_kind: str | None = None,
    ):
        super().__init__()

        self.role = role
        self.role_category = role_category
        self.qkv_layout = qkv_layout
        default_computation = "cross" if role == "cross" else "self"
        self.computation = AttentionComputation(computation if computation is not None else default_computation)
        self._explicit_cross_contract = computation is not None and self.computation is AttentionComputation.CROSS
        self.cross_parallel_layout = (
            CrossAttentionParallelLayout(cross_parallel_layout) if cross_parallel_layout is not None else None
        )
        self.cross_mask_kind = cross_mask_kind
        if self._explicit_cross_contract:
            if causal:
                raise NotImplementedError("The first cross-attention contract supports noncausal execution only")
            if qkv_layout != "BSND":
                raise ValueError("The first cross-attention contract requires qkv_layout='BSND'")
            if skip_sequence_parallel and self.cross_parallel_layout is None:
                raise ValueError("Skipping SP for cross attention requires an explicit Q/KV placement")
            if (
                self.cross_parallel_layout in (CrossAttentionParallelLayout.REPLICATED_Q_SHARDED_KV,)
                and skip_sequence_parallel
            ):
                raise ValueError("Sharded-KV cross attention requires SP communication")
            if paged_kv_cache_role is not None:
                raise ValueError("Conditioning cross attention does not use Scheduler-managed paged KV")
            if cross_mask_kind not in (None, "kv", "pair"):
                raise ValueError("Cross mask kind must be 'kv' or 'pair'")
        # ``prefix`` is also the stable layer identity used by vLLM's native
        # KV-cache metadata.  Keep it on the Omni layer so the active paged
        # adapter can dispatch the already-resharded Q/K/V to the matching
        # native cache tensor without replacing this Omni execution path.
        self.prefix = prefix
        if paged_kv_cache_role == "":
            raise ValueError("paged_kv_cache_role must be non-empty when provided")
        self.paged_kv_cache_role = paged_kv_cache_role
        self.paged_kv_cache_dtype = paged_kv_cache_dtype
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_size = head_size
        if self._explicit_cross_contract and (
            num_heads <= 0 or self.num_kv_heads <= 0 or num_heads % self.num_kv_heads or head_size <= 0
        ):
            raise ValueError("Cross attention requires positive head dimensions and Q heads divisible by KV heads")

        self._has_custom_attention = custom_attention is not None

        # Per-step schedule candidates. Empty unless a startup schedule
        # is configured, so without a schedule no extra impl is built.
        # AttentionImpl is not an nn.Module, so keeping these in a plain dict
        # registers no submodule and leaves state_dict untouched.
        self._schedule_profiles: tuple[str, ...] = ()
        self._schedule_candidates: dict[str, ScheduleCandidate] = {}
        # Model-owned requirements on every candidate of this layer (add_schedule_candidate_check).
        self._schedule_candidate_checks: list[Callable[[ScheduleCandidate], str | None]] = []

        # Resolve backend via role-aware config.
        # The global diffusion config is set during model init via
        # set_current_diffusion_config(); no env-var re-parsing needed here.
        backend_kwargs: dict | None = None
        self.backend_pref = None
        self.backend_explicit = False

        config = get_current_diffusion_config_or_none()
        attention_config = config.diffusion_attention_config if config is not None else None
        parallel_config = getattr(config, "parallel_config", None)
        self._hsdp_compile_boundary_enabled = bool(getattr(parallel_config, "use_hsdp", False))

        from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata

        model_class_name = getattr(config, "model_class_name", None) if config is not None else None
        allow_trtllm_default = get_diffusion_model_metadata(model_class_name).attention_mask_free

        scheduler_paged_kv = (
            config is not None
            and getattr(config, "diffusion_kv_mode", DiffusionKVCacheMode.DENSE_LEGACY)
            is DiffusionKVCacheMode.PAGED_SCHEDULER
        )
        self._scheduler_paged_kv = scheduler_paged_kv
        self.attn_backend: type[AttentionBackend] | None
        self.attention: AttentionImpl | nn.Module
        self.sdpa_fallback: SDPAImpl | None
        if custom_attention is None:
            selected_backend_cls, spec = get_attn_backend_for_role(
                role=role,
                head_size=head_size,
                attention_config=attention_config,
                role_category=role_category,
                allow_trtllm_default=allow_trtllm_default,
            )
            attn_backend_cls = _promote_marked_paged_default(
                selected_backend_cls,
                spec,
                scheduler_paged_kv=scheduler_paged_kv,
                paged_kv_cache_role=paged_kv_cache_role,
            )
            if attn_backend_cls is not selected_backend_cls:
                logger.info(
                    "Resolved marked paged diffusion attention role=%r to %r because the platform default %r "
                    "does not support Scheduler-owned KV",
                    role,
                    attn_backend_cls.get_name(),
                    selected_backend_cls.get_name(),
                )
            allgather_degree = getattr(parallel_config, "allgather_degree", 1)
            _reject_allgather_trtllm(attn_backend_cls, skip_sequence_parallel, allgather_degree)
            self.attn_spec = spec
            if spec is not None:
                backend_kwargs = spec.backend_kwargs()
                self.backend_pref = spec.backend
                self.backend_explicit = True
                logger.debug("Attention(role=%s) → backend=%s", role, spec.backend)
            else:
                # Propagate the resolved platform default so Ring Attention can
                # make a compatible automatic selection on the current GPU.
                self.backend_pref = attn_backend_cls.get_name()
                logger.debug("Attention(role=%s) → platform default (%s)", role, self.backend_pref)

            self.attn_backend = attn_backend_cls
            self.attn_impl_cls = _select_impl_cls(attn_backend_cls, impl_overrides)
            # Every impl of this layer, the baseline and each schedule candidate, gets these
            # arguments; only the backend selection differs.
            impl_kwargs: dict[str, Any] = dict(
                num_heads=num_heads,
                head_size=head_size,
                softmax_scale=softmax_scale,
                causal=causal,
                num_kv_heads=num_kv_heads,
                qkv_layout=qkv_layout,
                prefix=prefix,
                role=role,
                computation=self.computation,
            )
            self.attention = self.attn_impl_cls(
                **impl_kwargs, backend_kwargs=backend_kwargs, backend_explicit=self.backend_explicit
            )
            # Compatibility kernels run inside shared dispatch, between the
            # parallel strategy's input preparation and output restoration.
            self.sdpa_fallback = SDPABackend.get_impl_cls()(
                num_heads=num_heads,
                head_size=head_size,
                softmax_scale=softmax_scale,
                causal=causal,
                num_kv_heads=num_kv_heads,
                qkv_layout=qkv_layout,
            )
            # Pre-construct one deduplicated candidate per configured
            # profile. Runs only when a startup schedule exists; the baseline impl
            # above is never replaced and stays the gap/no-selection choice.
            schedule_config = getattr(config, "diffusion_attention_schedule", None)
            if schedule_config is not None:
                self._schedule_configured = True
                self._register_scheduled_op()
                self._prepare_schedule_candidates(
                    schedule_config,
                    impl_kwargs,
                    role=role,
                    role_category=role_category,
                    head_size=head_size,
                    allow_trtllm_default=allow_trtllm_default,
                    impl_overrides=impl_overrides,
                    skip_sequence_parallel=skip_sequence_parallel,
                    allgather_degree=allgather_degree,
                )
        else:
            if paged_kv_cache_role is not None:
                raise ValueError("custom_attention does not support Scheduler-managed paged KV")
            if not skip_sequence_parallel:
                raise ValueError("custom_attention must own its communication and requires skip_sequence_parallel=True")
            self.attn_spec = None
            self.attn_backend = None
            self.attn_impl_cls = type(custom_attention)
            self.attention = custom_attention
            self.sdpa_fallback = None
            logger.debug("Attention(role=%s) → custom kernel=%s", role, type(custom_attention).__name__)

        # Backend and head geometry are fixed for the layer. Resolve these
        # capabilities once rather than repeating platform queries per step.
        self._cross_head_limitation = None
        self._cross_mask_limitation = None
        if self._explicit_cross_contract:
            if self.attn_backend is None:
                raise NotImplementedError("Explicit cross attention requires a shared attention backend")
            if not self.attn_backend.supports_cross_head_geometry(self.num_heads, self.num_kv_heads):
                self._cross_head_limitation = f"Q/KV head geometry {self.num_heads}/{self.num_kv_heads}"
            if cross_mask_kind == "kv" and not self.attn_backend.supports_cross_kv_mask(self.attn_spec):
                self._cross_mask_limitation = "KV-only validity mask"
            elif cross_mask_kind == "pair" and not self.attn_backend.supports_cross_pair_mask(self.attn_spec):
                self._cross_mask_limitation = "pair restrictions"
            limitation = self._cross_head_limitation or self._cross_mask_limitation
            if limitation is not None and not self.backend_explicit:
                logger.warning_once(
                    "Cross attention will use SDPA for calls requiring %s because %s does not support it",
                    limitation,
                    self.attn_backend.get_name(),
                )

        self.softmax_scale = softmax_scale
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.use_sync = use_sync
        self.causal = causal
        self.skip_sequence_parallel = skip_sequence_parallel
        self.allow_fp32_fallback = allow_fp32_fallback

        self.use_ring = False
        self.ring_pg = None
        self.ring_runner = None

        if config is not None:
            if config.parallel_config.ring_degree > 1 and not self._explicit_cross_contract:
                self.use_ring = True
                sp_group = get_sp_group()
                self.ring_pg = sp_group.ring_group
                self.ring_runner = RingParallelAttention(
                    sp_group,
                    attn_backend_pref=self.backend_pref,
                    attn_backend_explicit=self.backend_explicit,
                )

        self.parallel_strategy = build_parallel_attention_strategy(
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            use_sync=use_sync,
            causal=causal,
            cross_parallel_layout=self.cross_parallel_layout if self._explicit_cross_contract else None,
        )
        # Local strategy when SP is intentionally inactive outside sharded regions.
        self._no_parallel_strategy = NoParallelAttention()

        self.layer_idx: int | None = _try_extract_layer_index(prefix)

        self._kv_cache_dtype: str | None = None
        self._kv_cache_skip_steps: set[int] | None = None
        self._kv_cache_skip_layers: set[int] | None = None
        # Per-layer opt-out from KV-cache quantization (set by model author).
        self._disable_kv_quant: bool = disable_kv_quant
        self._init_kv_cache_quantization(config)

    def _prepare_schedule_candidates(
        self,
        schedule_config: "AttentionScheduleConfig",
        impl_kwargs: Mapping[str, Any],
        *,
        role: str,
        role_category: str | None,
        head_size: int,
        allow_trtllm_default: bool,
        impl_overrides: Mapping[str, type[AttentionImpl]] | None,
        skip_sequence_parallel: bool,
        allgather_degree: int,
    ) -> None:
        """Pre-construct one deduplicated candidate record per configured profile.

        Candidates are additional prepared objects only; the baseline impl built in
        ``__init__`` is never replaced and remains the gap/no-selection choice. Candidate impls
        are built with the baseline's ``impl_kwargs`` and their own backend selection. Each
        profile is resolved independently through the same role-aware selector, so it
        keeps its own role precedence and never merges the baseline. Profiles
        not referenced by the service default schedule are still prepared, because
        startup compatibility validation covers every configured profile.

        The baseline guards that decide which backend a layer uses apply to candidates too: the
        marked-paged default->FLASH_ATTN promotion, the AllGather-KV + TRTLLM_ATTN rejection and
        implementation overrides. Per-candidate capability checks (paged KV, SP auto-pad, ring, kv
        dtype support) run in the post-load startup traversal
        (``validate_attention_schedule_candidates``).

        Identical effective identities for this role share one record. The dedup key carries the
        backend name, explicit-vs-default selection, the impl class object, and canonical JSON of
        backend_kwargs and of the spec's skip_calibration. An unserializable kwargs value does not
        share. Two profiles that differ only in their calibration curve never share one impl. A
        candidate whose key equals the baseline's, with no calibration of its own, reuses the
        baseline impl.
        """
        profiles = schedule_config.profiles
        self._schedule_profiles = tuple(sorted(profiles))
        # A candidate with the baseline's identity reuses the baseline impl instead of building a
        # second one. Calibration stamping writes the fallback dict onto the baseline, so only a
        # candidate that carries no calibration of its own (and is stamped from that same fallback)
        # can share it.
        baseline_key = (
            cast(type[AttentionBackend], self.attn_backend).get_name(),
            self.backend_explicit,
            self.attn_impl_cls,
            _canonical_json(self.attn_spec.backend_kwargs() if self.attn_spec is not None else None),
            "",
        )
        prepared: dict[tuple, ScheduleCandidate] = {}
        candidates: dict[str, ScheduleCandidate] = {}
        for name in self._schedule_profiles:
            backend_cls, spec = get_attn_backend_for_role(
                role=role,
                head_size=head_size,
                attention_config=profiles[name],
                role_category=role_category,
                allow_trtllm_default=allow_trtllm_default,
            )
            backend_cls = _promote_marked_paged_default(
                backend_cls,
                spec,
                scheduler_paged_kv=self._scheduler_paged_kv,
                paged_kv_cache_role=self.paged_kv_cache_role,
            )
            _reject_allgather_trtllm(backend_cls, skip_sequence_parallel, allgather_degree)
            impl_cls = _select_impl_cls(backend_cls, impl_overrides)
            backend_explicit = spec is not None
            backend_kwargs = spec.backend_kwargs() if spec is not None else None
            dedup_key = (
                backend_cls.get_name(),
                backend_explicit,
                impl_cls,
                _canonical_json(backend_kwargs),
                _canonical_json(spec.skip_calibration if spec is not None else None),
            )
            record = prepared.get(dedup_key)
            if record is None:
                if dedup_key == baseline_key:
                    impl = self.attention
                else:
                    impl = impl_cls(**impl_kwargs, backend_kwargs=backend_kwargs, backend_explicit=backend_explicit)
                record = ScheduleCandidate(
                    backend_cls=backend_cls,
                    spec=spec,
                    impl=impl,
                    backend_explicit=backend_explicit,
                )
                prepared[dedup_key] = record
            candidates[name] = record
        self._schedule_candidates = candidates

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        """Return native rank-local geometry for an opted-in paged cache."""

        if self.paged_kv_cache_role is None:
            return None
        assert self.attn_backend is not None  # Custom attention cannot opt into paged KV.
        dtype = self.paged_kv_cache_dtype or vllm_config.model_config.dtype
        # Keep backend layout discovery under the same config context used by
        # upstream vLLM's attention-spec collector.  vLLM 0.29 moved the
        # block-stride decision off the spec and onto the single physical
        # ``CacheConfig.kv_cache_layout`` resolved before the cache is built,
        # so the backend's preference is enforced there (see
        # ``vllm_omni.diffusion.diffusion_kv.initialization``) rather than
        # carried per layer.
        with set_current_vllm_config(vllm_config):
            assert_backend_layout_supported(vllm_config, self.attn_backend)
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            dtype=dtype,
            non_causal=not self.causal,
        )

    def _get_active_parallel_strategy(self):
        """Get the parallel strategy based on current SP active state.

        Returns NoParallelAttention if we're outside an SP sharded region
        (e.g., in noise_refiner/context_refiner before unified_prepare in Z-Image).
        This avoids unnecessary SP communication for layers not covered by _sp_plan.
        """
        if self.skip_sequence_parallel:
            return self._no_parallel_strategy
        if is_forward_context_available():
            ctx = get_forward_context()
            if not ctx.sp_active:
                return self._no_parallel_strategy
        return self.parallel_strategy

    def _init_kv_cache_quantization(self, config) -> None:
        if config is None or self._has_custom_attention:
            return
        assert self.attn_backend is not None
        dtype = getattr(config, "diffusion_kv_cache_dtype", None)
        if dtype == "auto":
            dtype = None
        parallel_config = getattr(config, "parallel_config", None)
        ring_degree = getattr(parallel_config, "ring_degree", 1)
        if dtype and dtype != "float":
            if ring_degree > 1:
                raise ValueError(
                    "KV quantization is not compatible with ring attention "
                    "(ring_degree > 1). Ring kernels do not propagate quantization descale "
                    "factors. Use Ulysses SP instead."
                )
            platform_key = current_omni_platform.device_name
            if not self.attention.supports_kv_cache_dtype(dtype, platform_key):
                raise ValueError(
                    f"Attention backend {self.attn_backend.get_name()} does not support "
                    f"kv_cache_dtype={dtype!r} on {platform_key}. Select a compatible "
                    "backend or set diffusion_kv_cache_dtype='auto'."
                )
        self._kv_cache_dtype = dtype
        self._kv_cache_skip_steps = getattr(config, "diffusion_kv_cache_skip_step_indices", None)
        self._kv_cache_skip_layers = getattr(config, "diffusion_kv_cache_skip_layer_indices", None)
        if self._kv_cache_skip_layers and self.layer_idx is None and not self._disable_kv_quant:
            raise ValueError("Attention quantization skip_layers requires a parseable transformer block index.")

    def _should_apply_kv_cache_quant(self) -> bool:
        skip_steps = self._kv_cache_skip_steps
        skip_layers = self._kv_cache_skip_layers
        if skip_steps is not None:
            step_idx = get_forward_context().denoise_step_idx if is_forward_context_available() else None
            if skip_steps and (step_idx is None or step_idx in skip_steps):
                return False
        if skip_layers is not None:
            if self.layer_idx is not None and self.layer_idx in skip_layers:
                return False
        return True

    def _with_kv_cache_dtype(self, attn_metadata: AttentionMetadata | None) -> AttentionMetadata | None:
        disabled = self._disable_kv_quant or not self._should_apply_kv_cache_quant()
        dtype = self._kv_cache_dtype
        if dtype in (None, "float"):
            dtype = None
        elif disabled:
            dtype = "float"
        if dtype is None and (attn_metadata is None or "kv_cache_dtype" not in attn_metadata.extra):
            return attn_metadata
        extra = dict(attn_metadata.extra) if attn_metadata is not None else {}
        # Recompute per forward so shared metadata cannot retain another step's policy.
        extra.pop("kv_cache_dtype", None)
        if dtype is not None:
            extra["kv_cache_dtype"] = dtype
        if attn_metadata is None:
            return AttentionMetadata(extra=extra) if extra else None
        return replace(attn_metadata, extra=extra)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        if torch.compiler.is_compiling() and self._schedule_configured:
            # Selecting a scheduled candidate reads the bound schedule from the forward context,
            # with the step and total for step ranges or the denoise sigma for sigma windows, and
            # backend-private gates read the step or timestep. Traced, those values would become
            # guards and each new step or sigma value would recompile the enclosing graph. The
            # decision uses only the construction-time flag, so a layer without a startup schedule
            # adds no boundary.
            encoded = (
                _encode_scheduled_op_metadata(attn_metadata) if self._schedule_op_eligible(query, key, value) else None
            )
            if encoded is not None:
                # One opaque node: the enclosing block stays a single graph, and the selection
                # runs when the op executes.
                return _scheduled_attention_op(
                    query,
                    key,
                    value,
                    None if attn_metadata is None else attn_metadata.attn_mask,
                    self._schedule_op_id,
                    *encoded,
                )
            return self._forward_schedule_compile_boundary(query, key, value, attn_metadata)
        if torch.compiler.is_compiling() and self._uses_hsdp_compile_boundary():
            # Keep HSDP/FSDP2 parameter all-gather outside Inductor's
            # attention graph; otherwise scheduler dependency analysis can
            # fail on the fused attention region.
            return self._forward_hsdp_compile_boundary(query, key, value, attn_metadata)

        return self._forward_impl(query, key, value, attn_metadata)

    def _register_scheduled_op(self) -> None:
        """Give this layer the id the compiled scheduled-attention op carries.

        The id is a plain CPU tensor attribute, not a buffer: ``Module.to()`` leaves it on the CPU,
        so reading it never synchronizes a device, and ``state_dict`` is unchanged. Dynamo passes it
        as a graph input, so repeated blocks share one graph instead of guarding on a per-layer
        constant (a string or int attribute would recompile once per block).
        """
        op_id = next(_SCHEDULED_ATTENTION_IDS)
        _SCHEDULED_ATTENTION_LAYERS[op_id] = self
        # Keep the id on the CPU even under ``with torch.device(...)``: reading a
        # device tensor here would force a host sync on every attention call.
        self._schedule_op_id = torch.tensor(op_id, dtype=torch.int64, device="cpu")

    def __setstate__(self, state: dict[str, Any]) -> None:
        # copy.copy, copy.deepcopy and unpickling all restore through here. The copied id would still
        # resolve to the original layer, so a copy that carries one registers its own.
        super().__setstate__(state)
        if self._schedule_op_id is not None:
            self._register_scheduled_op()

    def _schedule_op_eligible(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> bool:
        """Whether this layer and call can use the scheduled-attention op instead of the eager boundary.

        Evaluated while tracing, on structure only (never on the step). HSDP (its own boundary),
        Scheduler-managed paged KV and autograd (the op has no backward) keep
        ``_forward_schedule_compile_boundary``, as does metadata that ``_encode_scheduled_op_metadata``
        cannot carry.

        The op is registered with ``mutates_args=[]``, so Inductor may reorder it against other reads
        of state it writes outside its arguments. Paged KV writes the Scheduler's cache through the
        Worker adapter, so it stays behind the boundary. KV-cache quantization quantizes per call into
        fresh tensors and keeps no state between calls. Prompt K/V capture (HunyuanImage-3.0) runs in
        the model's attention module around this call, not inside ``_forward_impl``.
        """
        if self._schedule_op_id is None or self._uses_hsdp_compile_boundary():
            return False
        if self._scheduler_paged_kv and self.paged_kv_cache_role is not None:
            return False
        return not (torch.is_grad_enabled() and (query.requires_grad or key.requires_grad or value.requires_grad))

    def _uses_hsdp_compile_boundary(self) -> bool:
        if self._hsdp_compile_boundary_enabled:
            return True
        if not is_forward_context_available():
            return False
        od_config = get_forward_context().omni_diffusion_config
        parallel_config = getattr(od_config, "parallel_config", None)
        return bool(getattr(parallel_config, "use_hsdp", False))

    def resolve_execution_path(
        self,
        context: ExecutionContext,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> ExecutionPathResult:
        """Compose backend capabilities with outer Attention boundaries."""
        boundaries = set(context.outer_boundaries)
        if self._uses_hsdp_compile_boundary():
            boundaries.add(OuterBoundary.HSDP)
        active_strategy = self._get_active_parallel_strategy()
        strategy_name = active_strategy.name
        if self.use_ring and active_strategy.enabled and strategy_name == "ulysses":
            parallel_strategy = ParallelStrategy.HYBRID_ULYSSES_RING
        elif self.use_ring and active_strategy.enabled:
            parallel_strategy = ParallelStrategy.RING
        else:
            parallel_strategy = {
                "allgather_kv": ParallelStrategy.ALLGATHER_KV,
                "cross_reshard_kv": ParallelStrategy.CROSS_RESHARD_KV,
                "ulysses": ParallelStrategy.ULYSSES,
            }.get(strategy_name, ParallelStrategy.NONE)
        resolved_context = replace(
            context,
            outer_boundaries=frozenset(boundaries),
            paged_kv=self.is_paged_kv_active(),
            parallel_strategy=parallel_strategy,
            computation=getattr(self, "computation", AttentionComputation.SELF),
            q_heads=getattr(self, "num_heads", context.q_heads),
            kv_heads=getattr(self, "num_kv_heads", context.kv_heads),
            head_dim=getattr(self, "head_size", context.head_dim),
            cross_parallel_layout=getattr(self, "cross_parallel_layout", context.cross_parallel_layout),
            cross_mask_kind=getattr(self, "cross_mask_kind", context.cross_mask_kind),
        )
        if not resolved_context.paged_kv:
            attn_metadata = self._with_kv_cache_dtype(attn_metadata)
        if getattr(self, "_explicit_cross_contract", False):
            reason = self._cross_backend_limitation(attn_metadata)
            if reason is not None:
                blocked = self._cross_fallback_blocker(attn_metadata)
                result = self.sdpa_fallback.resolve_execution_path(resolved_context, query, key, value, attn_metadata)
                return replace(
                    result,
                    backend=self.attn_backend.get_name() if blocked else "SDPA",
                    path="cross_unsupported" if blocked else "cross_sdpa_fallback",
                    support=CapabilityResult.unsupported(blocked) if blocked else CapabilityResult.unmigrated(reason),
                )
        resolver = getattr(self.attention, "resolve_execution_path", None)
        if not callable(resolver):
            return ExecutionPathResult.unmigrated(
                type(self.attention).__name__,
                resolved_context,
            )
        return resolver(
            resolved_context,
            query,
            key,
            value,
            attn_metadata,
        )

    @torch.compiler.disable
    def _forward_hsdp_compile_boundary(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        return self._forward_impl(query, key, value, attn_metadata)

    @torch.compiler.disable
    def _forward_schedule_compile_boundary(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        # Runs eagerly: candidate selection, capability checks, backend-private gates and the
        # parallel strategy of this layer. Projections, norms and RoPE outside it stay compiled.
        return self._forward_impl(query, key, value, attn_metadata)

    def _forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        if getattr(self, "_explicit_cross_contract", False):
            self._validate_cross_inputs(query, key, value, attn_metadata)
        # Get the appropriate parallel strategy based on SP active state
        strategy = self._get_active_parallel_strategy()
        if self._explicit_cross_contract and self.cross_parallel_layout is None and strategy.enabled:
            raise ValueError(
                "Cross attention with SP requires an explicit Q/KV placement; "
                "declare local_q_replicated_kv when K/V is replicated on each rank"
            )
        paged_adapter = self._active_paged_kv_adapter()
        in_kv_memory_profile = is_forward_context_available() and get_forward_context().in_diffusion_kv_memory_profile
        if (
            self._scheduler_paged_kv
            and self.paged_kv_cache_role is not None
            and paged_adapter is None
            and not in_kv_memory_profile
        ):
            raise RuntimeError(
                "Scheduler-paged diffusion attention reached model forward without an active Worker adapter. "
                "Only the startup KV memory profile may execute before paged KV initialization."
            )
        use_paged_attention = paged_adapter is not None and self.paged_kv_cache_role is not None
        if use_paged_attention and not getattr(self.attn_backend, "supports_paged_kv", False):
            backend_name = (
                self.attn_backend.get_name() if self.attn_backend is not None else type(self.attention).__name__
            )
            raise NotImplementedError(
                f"Diffusion paged KV requires an Omni backend with paged support; selected {backend_name}"
            )
        if use_paged_attention and strategy is not self._no_parallel_strategy:
            strategy_name = strategy.name
            if self.use_ring or strategy_name == "ring":
                raise NotImplementedError(
                    "paged Scheduler KV is not supported with Ring attention; use strict Ulysses or no SP"
                )
            if strategy_name in ("allgather_kv", "ulysses_allgather_kv"):
                raise NotImplementedError(
                    f"paged Scheduler KV is not supported with the {strategy_name!r} sequence-parallel strategy"
                )
            if strategy_name == "ulysses" and get_ulysses_mode(default="strict") != "strict":
                raise NotImplementedError("paged Scheduler KV currently supports only strict Ulysses")

        # 1. Prepare inputs (Communication / Resharding)
        # For Ulysses: AllToAll Q/K/V; Slicing joint_q/k/v
        # For Ring: Concat joint_q
        query, key, value, attn_metadata, ctx = strategy.pre_attention(query, key, value, attn_metadata)

        # MiniMax-H3 Tier2 stores reference K/V in the balanced layout produced
        # by Ulysses. Keeping this hook at the generic post-parallel boundary
        # avoids a model import in the parallel strategy and leaves other models
        # untouched unless they explicitly provide the request-local state.
        post_parallel_ref = (
            attn_metadata.extra.get("minimax_h3_reference_kv_post_parallel") if attn_metadata is not None else None
        )
        if post_parallel_ref is not None:
            if not isinstance(post_parallel_ref, tuple) or len(post_parallel_ref) != 3:
                raise TypeError("minimax_h3_reference_kv_post_parallel must be a (state, layer_index, compact) tuple")
            if self.use_ring:
                raise NotImplementedError("post-Ulysses MiniMax-H3 reference KV requires ring_degree=1")
            state, layer_index, compact = post_parallel_ref
            processor = getattr(state, "process_post_parallel_layer", None)
            if not callable(processor):
                raise TypeError("MiniMax-H3 reference-KV state does not implement process_post_parallel_layer")
            key, value = processor(
                int(layer_index),
                key,
                value,
                compact=bool(compact),
                parallel_strategy=strategy.name,
            )

        # Scheduler rows describe the logical sequence, while strict Ulysses
        # may append synthetic tokens solely to make the image shard divisible.
        # Remove those tokens after the all-to-all and put zero placeholders
        # back before the reverse all-to-all.
        paged_sp_padding = 0
        paged_sp_padding_offset = query.shape[1]
        if use_paged_attention and strategy is not self._no_parallel_strategy and strategy.name == "ulysses":
            forward_ctx = get_forward_context() if is_forward_context_available() else None
            paged_sp_padding = int(getattr(forward_ctx, "sp_padding_size", 0))
            if paged_sp_padding:
                joint_len = int(getattr(ctx, "joint_len", 0))
                joint_strategy = str(getattr(ctx, "joint_strategy", "front"))
                paged_sp_padding_offset = query.shape[1] - (joint_len if joint_strategy == "rear" else 0)
                padding_start = paged_sp_padding_offset - paged_sp_padding
                if padding_start < 0:
                    raise ValueError(
                        "Paged Ulysses padding exceeds the post-all-to-all sequence: "
                        f"padding={paged_sp_padding}, sequence={query.shape[1]}"
                    )

                def _remove_paged_sp_padding(tensor: torch.Tensor) -> torch.Tensor:
                    return torch.cat(
                        (tensor[:, :padding_start], tensor[:, paged_sp_padding_offset:]),
                        dim=1,
                    ).contiguous()

                query = _remove_paged_sp_padding(query)
                key = _remove_paged_sp_padding(key)
                value = _remove_paged_sp_padding(value)

        # 2. This is the shared GPU/NPU boundary. The Worker adapter prepares
        # the native page-table context after SP has produced rank-local Q/K/V;
        # backend resolution below selects CUDA or Ascend execution.
        if use_paged_attention:
            assert paged_adapter is not None
            paged_kv_context = paged_adapter.prepare_layer_context(
                self.prefix,
                query,
                key,
                value,
                omni_attn_metadata=attn_metadata,
            )
            out = self.attention.forward_paged(paged_kv_context)
        else:
            attn_metadata = self._with_kv_cache_dtype(attn_metadata)
            if self.use_ring and strategy is not self._no_parallel_strategy:
                out = self._run_ring_attention(query, key, value, attn_metadata)
            else:
                out = self._run_local_attention(query, key, value, attn_metadata)

        if paged_sp_padding:
            padding_start = paged_sp_padding_offset - paged_sp_padding
            output_padding = out.new_zeros((out.shape[0], paged_sp_padding, *out.shape[2:]))
            out = torch.cat((out[:, :padding_start], output_padding, out[:, padding_start:]), dim=1)

        # 3. Post-processing (Reverse Communication)
        # For Ulysses: AllToAll Output, and AllGather Joint Output
        out = strategy.post_attention(out, ctx)
        if getattr(self, "_explicit_cross_contract", False) and attn_metadata is not None:
            if attn_metadata.cross_query_mask is not None:
                out = out.masked_fill(~attn_metadata.cross_query_mask[:, :, None, None], 0)

        return out

    def _validate_cross_inputs(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> None:
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("Cross attention requires BSND Q/K/V tensors")
        q_shape, k_shape, v_shape = query.shape, key.shape, value.shape
        if q_shape[0] != k_shape[0] or k_shape[0] != v_shape[0]:
            raise ValueError("Cross attention requires matching Q/K/V batch sizes")
        if q_shape[1] <= 0 or k_shape[1] <= 0 or k_shape[1] != v_shape[1]:
            raise ValueError("Cross attention requires nonempty Q/K/V and matching K/V sequence lengths")
        if q_shape[2] != self.num_heads or k_shape[2] != self.num_kv_heads or v_shape[2] != self.num_kv_heads:
            raise ValueError("Cross attention Q/K/V head counts do not match the declared local head geometry")
        if q_shape[3] != self.head_size or k_shape[3] != self.head_size or v_shape[3] != self.head_size:
            raise ValueError("Cross attention Q/K/V head dimensions must match head_size")
        if (
            query.dtype != key.dtype
            or key.dtype != value.dtype
            or query.device != key.device
            or key.device != value.device
        ):
            raise ValueError("Cross attention Q/K/V must have the same dtype and device")
        if attn_metadata is None:
            return
        if (
            attn_metadata.joint_attn_mask is not None
            or attn_metadata.joint_query is not None
            or attn_metadata.joint_key is not None
            or attn_metadata.joint_value is not None
            or attn_metadata.full_attn_spans is not None
            or attn_metadata.query_ranges is not None
            or attn_metadata.video_layout is not None
            or attn_metadata.packed_padding is not None
            or set(attn_metadata.extra) - {"attention_mask_mode"}
        ):
            raise NotImplementedError("Explicit cross attention does not accept joint or packed metadata")
        q_mask = attn_metadata.cross_query_mask
        if q_mask is not None and (
            q_mask.dtype != torch.bool or q_mask.shape != q_shape[:2] or q_mask.device != query.device
        ):
            raise ValueError("Cross Q validity must be a Boolean [B, Q_len] tensor on the Q device")
        if attn_metadata.attn_mask is None:
            return
        mask = attn_metadata.attn_mask
        if mask.device != query.device:
            raise ValueError("Cross attention mask must be on the Q/K/V device")
        if self.cross_mask_kind == "pair":
            if (
                mask.ndim != 4
                or mask.shape[0] not in (1, q_shape[0])
                or mask.shape[1] not in (1, self.num_heads)
                or mask.shape[2] not in (1, q_shape[1])
                or mask.shape[3] != k_shape[1]
                or mask.dtype not in (torch.bool, torch.float32, query.dtype)
            ):
                raise ValueError("Cross pair mask must broadcast to [B, Q_heads, Q_len, KV_len]")
            if not attn_metadata.cross_pair_mask_validated:
                raise ValueError("Cross pair masks must be prepared with AttentionMetadata.for_cross_pair_mask")
            return
        if self.cross_mask_kind != "kv":
            raise ValueError("Cross attention requires cross_mask_kind='kv' or 'pair' when a mask is supplied")
        if mask.ndim != 2 or mask.shape != (q_shape[0], k_shape[1]) or mask.dtype != torch.bool:
            raise ValueError("The cross-attention mask must be a Boolean KV validity mask of shape [B, KV_len]")
        if mask.device.type != "cpu" and not attn_metadata.cross_kv_rows_validated:
            raise ValueError("GPU cross K/V masks must be prepared with AttentionMetadata.for_cross_kv_mask")
        if mask.device.type == "cpu" and not attn_metadata.cross_kv_rows_validated and not bool(mask.any(dim=1).all()):
            raise ValueError("Each cross-attention batch row must contain at least one valid K/V position")

    def _cross_backend_limitation(
        self,
        attn_metadata: AttentionMetadata | None,
        *,
        backend: type[AttentionBackend] | None = None,
        spec: "AttentionSpec | None" = None,
    ) -> str | None:
        """Return what the selected backend cannot do for this cross call.

        ``backend`` and ``spec`` name a scheduled candidate; without them the baseline's limitations,
        resolved once at construction, apply.
        """
        if backend is None:
            if self._cross_head_limitation is not None:
                return self._cross_head_limitation
            if attn_metadata is not None and attn_metadata.attn_mask is not None:
                return self._cross_mask_limitation
            return None
        if not backend.supports_cross_head_geometry(self.num_heads, self.num_kv_heads):
            return f"Q/KV head geometry {self.num_heads}/{self.num_kv_heads}"
        if attn_metadata is None or attn_metadata.attn_mask is None:
            return None
        if self.cross_mask_kind == "kv" and not backend.supports_cross_kv_mask(spec):
            return "KV-only validity mask"
        if self.cross_mask_kind == "pair" and not backend.supports_cross_pair_mask(spec):
            return "pair restrictions"
        return None

    def _cross_fallback_blocker(
        self,
        attn_metadata: AttentionMetadata | None,
        *,
        backend: type[AttentionBackend] | None = None,
        spec: "AttentionSpec | None" = None,
    ) -> str | None:
        explicit = self.backend_explicit if backend is None else spec is not None
        if explicit:
            selected = self.attn_backend if backend is None else backend
            return (
                f"Explicit backend {selected.get_name()} does not support "
                f"{self._cross_backend_limitation(attn_metadata, backend=backend, spec=spec)}"
            )
        if attn_metadata is not None and attn_metadata.extra.get("kv_cache_dtype") not in (None, "float"):
            return (
                "Cross SDPA fallback does not support quantized K/V; "
                "disable KV quantization or select a compatible backend"
            )
        return None

    @staticmethod
    def _active_paged_kv_adapter():
        """Return the Worker adapter selected by Runner-owned metadata."""

        if not is_forward_context_available():
            return None
        return getattr(get_forward_context(), "paged_kv_adapter", None)

    def is_paged_kv_active(self) -> bool:
        """Return whether this layer will use Scheduler-managed paged KV."""

        return self.paged_kv_cache_role is not None and self._active_paged_kv_adapter() is not None

    def _run_local_attention(self, query, key, value, attn_metadata):
        if self._has_custom_attention:
            return cast(nn.Module, self.attention)(query, key, value, attn_metadata)

        effective_impl, effective_backend, effective_spec = self.effective_attention()
        if getattr(self, "_explicit_cross_contract", False):
            # A scheduled candidate is checked against its own backend; the baseline keeps the
            # limitations resolved at construction.
            cross_backend, cross_spec = (
                (None, None) if effective_impl is self.attention else (effective_backend, effective_spec)
            )
            reason = self._cross_backend_limitation(attn_metadata, backend=cross_backend, spec=cross_spec)
            if reason is not None:
                blocked = self._cross_fallback_blocker(attn_metadata, backend=cross_backend, spec=cross_spec)
                if blocked is not None:
                    raise NotImplementedError(blocked)
                return self.sdpa_fallback.forward(query, key, value, attn_metadata)
        dense_mask = attn_metadata.attn_mask if attn_metadata is not None else None
        # Piecewise backends honor full_attn_spans ahead of attn_mask and never read the
        # dense mask (e.g. HunyuanImage3, which sends both), so there is nothing to divert.
        spans_take_precedence = (
            attn_metadata is not None
            and attn_metadata.full_attn_spans is not None
            and getattr(effective_backend, "supports_piecewise_spans", False)
        )
        if (
            dense_mask is not None
            and dense_mask.ndim == 4
            and effective_backend is not None
            and not spans_take_precedence
        ):
            # A dense 4D mask (for example a padded DiT batch) is a semantic
            # requirement, not a backend hint. Backends that only implement a
            # 2D key-validity mask must not reach their kernel wrapper, which
            # asserts deeper down. Degrade this single call to SDPA instead;
            # explicit selections and paged KV must still run or fail.
            capability = getattr(effective_backend, "supports_dense_4d_mask", None)
            if callable(capability):
                capability = capability(effective_spec)
            if capability is False:
                backend_name = cast(type[AttentionBackend], effective_backend).get_name()
                # A scheduled candidate is explicit when its profile names a backend for
                # this layer; a role left on the platform default is not.
                backend_explicit = (
                    self.backend_explicit if effective_impl is self.attention else effective_spec is not None
                )
                paged_kv = self.is_paged_kv_active()
                if backend_explicit or self.sdpa_fallback is None or paged_kv:
                    raise NotImplementedError(
                        f"Attention backend '{backend_name}' does not support a dense 4D attn_mask "
                        f"(shape={tuple(dense_mask.shape)}, dtype={dense_mask.dtype}, "
                        f"backend_explicit={backend_explicit}, paged_kv={paged_kv}). "
                        "Use a mask-capable backend (TORCH_SDPA or CUDNN_ATTN), or avoid padding "
                        "inside the attention batch."
                    )
                logger.warning_once(
                    "Attention backend '%s' cannot consume a dense 4D attn_mask; "
                    "falling back to SDPA for this layer. "
                    "role=%s mask_ndim=%d mask_shape=%s mask_dtype=%s mask_device=%s "
                    "backend_explicit=%s paged_kv=%s",
                    backend_name,
                    getattr(self, "role", None),
                    dense_mask.ndim,
                    tuple(dense_mask.shape),
                    dense_mask.dtype,
                    dense_mask.device,
                    backend_explicit,
                    paged_kv,
                )
                return cast(AttentionImpl, self.sdpa_fallback).forward(query, key, value, attn_metadata)
        self._assert_metadata_compatible(attn_metadata, backend=effective_backend, spec=effective_spec)

        if (
            self.allow_fp32_fallback
            and query.is_cuda
            and query.dtype == torch.float32
            and query.ndim == 4
            and not self.backend_explicit
            and cast(type[AttentionBackend], self.attn_backend).get_name() == "FLASH_ATTN"
            and (
                attn_metadata is None
                or (
                    attn_metadata.full_attn_spans is None
                    and attn_metadata.query_ranges is None
                    and attn_metadata.video_layout is None
                    and attn_metadata.packed_padding is None
                    and not attn_metadata.extra
                )
            )
        ):
            logger.warning_once("Using SDPA for this layer's FP32 input with automatic CUDA FlashAttention selection.")
            return cast(AttentionImpl, self.sdpa_fallback).forward(query, key, value, attn_metadata)

        in_kv_memory_profile = is_forward_context_available() and get_forward_context().in_diffusion_kv_memory_profile
        # The startup KV-capacity profile needs tensor shapes, not a paged
        # attention result. If dense FLASH_ATTN deps are absent (NPU MindIE-SD
        # or CUDA CuTe FA4), SDPA provides that profile forward. Formal paged
        # requests never use this branch because their Worker adapter is active.
        # A user-explicit backend must run or raise.
        if (
            self._scheduler_paged_kv
            and self.paged_kv_cache_role is not None
            and in_kv_memory_profile
            and self.attn_backend is not None
            and self.attn_backend.get_name() == "FLASH_ATTN"
            and not current_omni_platform.supports_diffusion_dense_flash_attention()
        ):
            assert self.sdpa_fallback is not None
            logger.warning_once(
                "The startup KV memory profile is using SDPA because dense FLASH_ATTN is unavailable. "
                "Formal paged requests still use the platform-native paged attention backend."
            )
            return cast(AttentionImpl, self.sdpa_fallback).forward(query, key, value, attn_metadata)

        return effective_impl.forward(query, key, value, attn_metadata)

    def add_schedule_candidate_check(self, check: Callable[[ScheduleCandidate], str | None]) -> None:
        """Register a model-owned requirement that startup validation applies to every candidate.

        ``check(record)`` returns None when the candidate is usable on this layer, or the reason it is
        not. Models use it for decisions the generic traversal cannot see, such as a layout the model
        builds for every forward or a construction-time choice made from the baseline backend. Register
        before the loader validates candidates, i.e. during model construction or weight loading.
        """
        self._schedule_candidate_checks.append(check)

    def effective_attention(
        self,
    ) -> tuple[AttentionImpl | nn.Module, type[AttentionBackend] | None, "AttentionSpec | None"]:
        """Baseline outside a scheduled denoise step; the step's prepared candidate inside one.

        A layer built without a schedule config has no candidates and always runs its baseline.
        """
        name = get_attention_schedule_profile() if self._schedule_configured else None
        if name is None:
            return self.attention, self.attn_backend, getattr(self, "attn_spec", None)
        record = self._schedule_candidates.get(name)
        if record is None:
            raise RuntimeError(f"attention schedule profile {name!r} was not prepared on this layer")
        return record.impl, record.backend_cls, record.spec

    def _assert_metadata_compatible(
        self,
        attn_metadata: AttentionMetadata | None,
        *,
        backend: type[AttentionBackend] | None = None,
        spec: "AttentionSpec | None" = None,
    ) -> None:
        """Reject metadata that ``backend`` with ``spec`` cannot run; without ``backend``, check the baseline."""
        if attn_metadata is None:
            return
        if backend is None:
            selected, selected_spec = self.attn_backend, getattr(self, "attn_spec", None)
        else:
            selected, selected_spec = backend, spec
        if selected is None:
            return
        backend_name = selected.get_name()
        if attn_metadata.attn_mask is not None and not selected.supports_attention_mask(selected_spec):
            raise ValueError(
                f"Attention backend '{backend_name}' does not support attn_mask. Select a mask-capable backend."
            )
        if attn_metadata.full_attn_spans is None:
            return
        if attn_metadata.attn_mask is not None and attn_metadata.attn_mask.ndim == 4:
            return
        if not selected.supports_piecewise_spans:
            raise ValueError(
                f"Attention backend '{backend_name}' does not support "
                f"piecewise attention (full_attn_spans without a 4D attn_mask). "
                f"Use a Flash backend (FLASH_ATTN / FLASH_ATTN_HUB / FLASH_ATTN_3_HUB), "
                f"or provide a 4D attn_mask that encodes the mixed causal/full pattern."
            )

    def _run_ring_attention(self, query, key, value, attn_metadata):
        if attn_metadata is not None and attn_metadata.attn_mask is not None:
            raise ValueError("Ring attention does not support attn_mask; use Ulysses SP or disable mask_sp_padding.")
        backend_name = self.attn_backend.get_name() if self.attn_backend is not None else None
        if backend_name == "SPARGE_ATTN":
            raise NotImplementedError(
                "SPARGE_ATTN is not supported with ring sequence parallelism: "
                "the ring path bypasses the backend, so the sparse config would be silently ignored. "
                "Use Ulysses SP instead, or disable ring SP."
            )
        skip = getattr(self.attention, "skip", None)
        if skip is not None and getattr(skip, "configured", False):
            raise NotImplementedError(
                "Skip-Softmax (TRTLLM_ATTN) is not supported with ring sequence parallelism: "
                "the ring path bypasses the backend, so the skip config would be silently ignored. "
                "Use Ulysses SP instead, or remove the skip_softmax config."
            )
        # Delegate to RingParallelAttention strategy if available
        if self.ring_runner is not None:
            return self.ring_runner.run_attention(
                query, key, value, attn_metadata, softmax_scale=self.softmax_scale, causal=self.causal
            )

        raise RuntimeError("Ring attention is enabled but strategy is not RingParallelAttention")


def _plans_auto_pad(value: object) -> bool:
    """Whether one SP plan value declares ``auto_pad`` at any nesting level.

    Plan values nest dicts and lists, e.g. ``{"rope": {0: SequenceParallelInput(..., auto_pad=True)}}``.
    """
    if isinstance(value, dict):
        return any(_plans_auto_pad(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_plans_auto_pad(item) for item in value)
    return isinstance(value, SequenceParallelInput) and value.auto_pad


def _sp_auto_pad_roots(model: nn.Module) -> tuple[str, ...]:
    """Qualified names of the components whose own ``_sp_plan`` declares ``auto_pad``.

    SP hooks are applied per component, so one component's plan does not require mask
    support from another component's layers. The root module has the empty name.
    """
    return tuple(
        name
        for name, module in model.named_modules()
        if isinstance(getattr(module, "_sp_plan", None), dict) and _plans_auto_pad(module._sp_plan)
    )


def _validate_candidate_calibration(
    layer_name: str,
    profile_name: str,
    record: "ScheduleCandidate",
    effective_calibration: dict | None,
) -> None:
    """A candidate that asked for ``target_sparsity`` needs a curve for THIS layer.

    ``effective_calibration`` is the dict that stamping writes onto this candidate. The caller passes
    ``calibration_for_candidate(record, fallback)``: the candidate's own ``skip_calibration`` when it
    has one, otherwise the fallback dict, so validation answers the same question stamping does. An
    ignore-rule hit is a preserved legitimate dense fallback and passes; a layer that dict gives no
    curve for would silently stay dense at runtime, so it is rejected at startup.
    """
    skip = record.spec.skip_softmax if record.spec is not None else None
    if skip is None or skip.target_sparsity is None or skip.threshold is not None:
        return  # no curve requested, or the calibration-free path
    if layer_calibration_is_ignored(layer_name, effective_calibration):
        return
    if layer_calibration_curve(layer_name, effective_calibration) is None:
        raise ValueError(
            f"attention schedule profile {profile_name!r} requests skip_softmax.target_sparsity but the "
            f"calibration stamped onto this candidate resolves no curve for layer {layer_name!r}. "
            f"Load a calibrated checkpoint for that expert, set skip_softmax.threshold for the "
            f"calibration-free path, or remove the profile."
        )


def validate_attention_schedule_candidates(model: nn.Module, od_config: "OmniDiffusionConfig") -> int:
    """Post-load startup traversal over every prepared schedule candidate.

    Construction (``Attention._prepare_schedule_candidates``) reproduces the guards that decide
    WHICH backend a layer uses. The checks here need the loaded model and the resolved parallel
    plan instead. The checks that run without this traversal do not read each prepared candidate,
    so an incompatible candidate would surface inside the first kernel after the layer switches to
    it, be ignored on a paged KV or ring layer, or run dense when its calibration curve is missing:

    * paged KV: ``_forward_impl`` checks only the baseline backend for paged support, and paged
      forward runs the native implementation bound to the baseline, so a candidate must support
      paged KV and match the baseline backend, explicit selection and backend kwargs;
    * KV-cache quantization: ``_init_kv_cache_quantization`` probes the baseline impl only;
    * SP auto-pad: the pad-time mask probe checks only the baseline, so when the model masks SP
      padding (``mask_sp_padding``), a self-attention layer in a component that plans auto_pad
      needs mask-capable candidates;
    * ring: the ring path bypasses the backend and the ring runner is bound to the baseline backend
      preference and explicit flag, and ignores backend kwargs, so a candidate carrying skip_softmax
      or differing from the baseline in backend name, explicit selection or backend kwargs cannot be
      represented;
    * calibration: a candidate asking for ``target_sparsity`` needs a curve for its own layer in the
      calibration dict that stamping writes onto that candidate.

    Returns the number of validated candidates. Without a schedule this returns 0 immediately and
    walks nothing. A schedule that finds no attention layer, or no layer that saw it, is rejected
    instead of passing silently.
    """
    schedule = od_config.diffusion_attention_schedule
    if schedule is None:
        return 0

    profiles = tuple(sorted(schedule.profiles))
    kv_dtype = od_config.diffusion_kv_cache_dtype
    if kv_dtype in ("auto", "float"):
        # The baseline guard treats both as "no quantization" (see _init_kv_cache_quantization,
        # which probes only `if dtype and dtype != "float"`). Probing them here would reject a
        # configuration that loads fine without a schedule.
        kv_dtype = None
    platform_key = current_omni_platform.device_name
    sp_size = od_config.parallel_config.sequence_parallel_size
    # mask_sp_padding turns on the SP padding mask in Wan2.2 T2V, I2V and VACE; without it, padded tokens are unmasked.
    # HunyuanImage-3.0 masks its own padding either way; its image attention layer registers its own profile check.
    masks_sp_padding = sp_size is not None and sp_size > 1 and od_config.parallel_config.mask_sp_padding
    auto_pad_roots = _sp_auto_pad_roots(model) if masks_sp_padding else ()
    ring_degree = od_config.parallel_config.ring_degree
    skip_layers = od_config.diffusion_kv_cache_skip_layer_indices
    effective_calibration = resolve_effective_calibration(od_config.diffusion_attention_config, schedule)

    validated = 0
    attention_layers = 0
    schedule_aware_layers = 0
    for layer_name, module in model.named_modules():
        if not isinstance(module, Attention):
            continue
        attention_layers += 1
        if not module._schedule_configured:
            continue  # built with no current diffusion config; it can never take part in a schedule
        schedule_aware_layers += 1
        # Only self-attention sees the SP padding mask; cross-attention keys come from the unpadded context.
        requires_mask = module.computation is AttentionComputation.SELF and any(
            root == "" or layer_name == root or layer_name.startswith(root + ".") for root in auto_pad_roots
        )
        paged = module._scheduler_paged_kv and module.paged_kv_cache_role is not None
        # Construction prepares a candidate for every configured profile, and only from a backend:
        # custom-attention layers never take part in a schedule.
        baseline_backend = cast(type[AttentionBackend], module.attn_backend).get_name()
        baseline_identity = _attention_identity(module.attn_backend, module.backend_explicit, module.attn_spec)
        for name in profiles:
            record = module._schedule_candidates[name]
            backend_name = record.backend_cls.get_name()
            matches_baseline = (
                _attention_identity(record.backend_cls, record.backend_explicit, record.spec) == baseline_identity
            )
            if paged and not record.backend_cls.supports_paged_kv:
                raise ValueError(
                    f"attention schedule profile {name!r} selects backend {backend_name} which does not "
                    f"support Scheduler-managed paged KV (layer {layer_name!r}, role "
                    f"{module.paged_kv_cache_role!r}). Select a paged-capable backend for that profile "
                    f"or disable scheduler paged KV."
                )
            if paged and not matches_baseline:
                raise ValueError(
                    f"attention schedule profile {name!r} selects backend {backend_name} "
                    f"(explicit={record.backend_explicit}) on a Scheduler-managed paged KV layer "
                    f"(layer {layer_name!r}, role {module.paged_kv_cache_role!r}), but paged attention runs "
                    f"the native implementation bound to the baseline {baseline_backend!r} "
                    f"(explicit={module.backend_explicit}) and cannot represent a candidate whose backend, "
                    f"explicit selection or backend kwargs differ from it. Align that profile with the "
                    f"baseline or disable scheduler paged KV."
                )
            # Layers that can run ring: use_ring at construction (never explicit cross-attention), minus
            # layers that skip sequence parallelism, which always run local attention.
            if ring_degree > 1 and not module._explicit_cross_contract and not module.skip_sequence_parallel:
                # The ring path bypasses the backend, so a candidate carrying skip_softmax
                # would be silently ignored (_run_ring_attention only reads the baseline impl). The
                # ring runner was constructed once from the baseline backend preference and explicit
                # flag and never reads backend kwargs: an explicit candidate over an automatic
                # baseline would fall back to SDPA ring where it must raise, and quant or sparse
                # kwargs would be dropped (ring.py). Kernel-availability probing is deliberately not
                # duplicated here.
                if record.spec is not None and record.spec.skip_softmax is not None:
                    raise ValueError(
                        f"attention schedule profile {name!r} configures skip_softmax, which ring sequence "
                        f"parallelism cannot honor: the ring path bypasses the backend, so the skip config "
                        f"would be silently ignored (layer {layer_name!r}). Use Ulysses SP instead, or "
                        f"remove skip_softmax from that profile."
                    )
                if not matches_baseline:
                    raise ValueError(
                        f"attention schedule profile {name!r} resolves to backend {backend_name} "
                        f"(explicit={record.backend_explicit}) but this layer's ring runner is bound to "
                        f"{baseline_backend!r} (explicit={module.backend_explicit}) (layer {layer_name!r}). "
                        f"Ring attention runs from the baseline's backend preference and ignores backend "
                        f"kwargs, so it cannot represent a candidate whose backend, explicit selection or "
                        f"backend kwargs differ from the baseline; use Ulysses SP, or align that profile "
                        f"with the baseline."
                    )
            kv_opted_out = module._disable_kv_quant or (
                skip_layers is not None and module.layer_idx is not None and module.layer_idx in skip_layers
            )
            if kv_dtype and not kv_opted_out and not record.impl.supports_kv_cache_dtype(kv_dtype, platform_key):
                raise ValueError(
                    f"attention schedule profile {name!r} selects backend {backend_name} which does "
                    f"not support kv_cache_dtype={kv_dtype!r} on {platform_key} (layer {layer_name!r}). "
                    f"Select a compatible backend for that profile or set "
                    f"diffusion_kv_cache_dtype='auto'."
                )
            if requires_mask and not record.backend_cls.supports_attention_mask(record.spec):
                raise ValueError(
                    f"attention schedule profile {name!r} selects backend {backend_name} which does not "
                    f"support attention_mask, but this model plans SP auto_pad with sequence_parallel_size="
                    f"{sp_size} and mask_sp_padding=True (layer {layer_name!r}). Remove that profile, select "
                    f"a mask-capable backend for it, or disable mask_sp_padding."
                )
            for check in module._schedule_candidate_checks:
                reason = check(record)
                if reason is not None:
                    raise ValueError(
                        f"attention schedule profile {name!r} selects backend {backend_name}, which this model "
                        f"cannot use on layer {layer_name!r}: {reason}"
                    )
            stamped = calibration_for_candidate(record, effective_calibration)
            _validate_candidate_calibration(layer_name, name, record, stamped)
            validated += 1
    if attention_layers == 0:
        raise ValueError(
            f"an attention schedule with profile(s) {', '.join(repr(p) for p in profiles)} is configured but "
            f"no attention layer was discovered in {type(model).__name__}; nothing was validated and the "
            f"schedule could never be selected. Remove the schedule, or load a model whose attention uses "
            f"vllm_omni.diffusion.attention.layer.Attention."
        )
    if schedule_aware_layers == 0:
        raise ValueError(
            f"an attention schedule with profile(s) {', '.join(repr(p) for p in profiles)} is configured but "
            f"none of the {attention_layers} attention layer(s) in {type(model).__name__} saw it at "
            f"construction, so nothing was validated. Those layers were built without a current diffusion "
            f"config; bind the config before constructing them, or remove the schedule."
        )
    return validated
