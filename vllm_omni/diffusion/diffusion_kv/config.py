# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# Attention backend selectors that can serve Scheduler-owned paged diffusion KV.
# Today this is the logical FLASH_ATTN selector, which resolves to a native
# paged backend per platform (CUDA FlashAttention/FA3, Ascend FIA). Other
# diffusion backends have no paged path and would have to fall back to dense
# attention, which a paged request must never do silently.
PAGED_KV_ATTENTION_BACKENDS = frozenset({"FLASH_ATTN"})

# Whether an omitted ``diffusion_kv_mode`` may select paged execution. The
# resolution machinery is in place, but the default stays on the dense path
# until the parity evidence required by RFC #9192 (accuracy, memory sizing,
# prefix caching, lifecycle) is attached to each paged model integration. Flip
# this constant in the follow-up that makes paged the default.
PAGED_KV_IMPLICIT_DEFAULT_ENABLED = False

# Reasons returned by ``resolve_diffusion_kv_cache_mode`` when an implicit (no
# ``diffusion_kv_mode`` given) selection stays on the dense path.
REASON_IMPLICIT_DISABLED = "paged KV is not the default yet (pending parity evidence)"
REASON_MODEL_NOT_PAGED = "model has no paged KV integration"
REASON_STEP_EXECUTION = "step_execution is not implemented on the paged path"
REASON_OFFLOAD = "offload is not validated with paged KV"
REASON_LEGACY_IMPORT = "imported AR KV is not connector-aware yet"
REASON_ATTENTION_BACKEND = "attention backend does not advertise paged KV"
REASON_ROW_CAPACITY = "no BlockTable row capacity is known for the request"


class DiffusionKVCacheMode(str, Enum):
    """Migration mode for diffusion KV ownership."""

    DENSE_LEGACY = "dense_legacy"
    # Reserved for the experimental Worker-owned paging design. Production
    # configuration rejects it until that engine is migrated to the common
    # Scheduler-owned path.
    PAGED_WORKER_LOCAL = "paged_worker_local"
    PAGED_SCHEDULER = "paged_scheduler"


def parse_diffusion_kv_cache_mode(value: object) -> DiffusionKVCacheMode | None:
    """Parse a selectable Diffusion KV ownership mode.

    ``None`` means "not specified": the mode is then resolved from the model's
    paged-KV capability and the runtime configuration by
    :func:`resolve_diffusion_kv_cache_mode`. Every other value must name a
    supported mode.

    ``PAGED_WORKER_LOCAL`` remains in the enum to document the migration
    topology, but it is not a valid production selection until that runtime is
    wired. Keep this validation shared by structured and runtime configs so an
    unsupported mode fails before managed-process startup.
    """

    if value is None:
        return None
    try:
        mode = DiffusionKVCacheMode(value)
    except (TypeError, ValueError) as exc:
        supported = ", ".join(mode.value for mode in DiffusionKVCacheMode)
        raise ValueError(f"Unsupported Diffusion KV diffusion_kv_mode {value!r}; expected one of: {supported}") from exc
    if mode is DiffusionKVCacheMode.PAGED_WORKER_LOCAL:
        raise ValueError(
            "Diffusion KV diffusion_kv_mode 'paged_worker_local' is reserved but not implemented; "
            "use 'dense_legacy' or 'paged_scheduler'"
        )
    return mode


def is_scheduler_paged_kv_mode(mode: DiffusionKVCacheMode | None) -> bool:
    """Return whether a parsed cache mode uses Scheduler-owned paging."""
    return mode is DiffusionKVCacheMode.PAGED_SCHEDULER


def normalize_attention_backend(value: object) -> str | None:
    """Normalize an attention backend selector for capability comparison."""

    if value is None:
        return None
    name = getattr(value, "value", value)
    if not isinstance(name, str):
        name = getattr(name, "__name__", None) or str(name)
    normalized = name.strip().upper()
    if normalized.endswith("BACKEND"):
        normalized = normalized[: -len("BACKEND")]
    return normalized


def attention_backend_supports_paged_kv(value: object) -> bool:
    """Return whether an explicitly selected attention backend may serve paged KV.

    ``None`` means no explicit selector was configured, so the platform default
    is used; that default is validated when the attention layer is built.
    """

    normalized = normalize_attention_backend(value)
    if normalized is None:
        return True
    return normalized in PAGED_KV_ATTENTION_BACKENDS


@dataclass(frozen=True)
class DiffusionKVResolution:
    """Resolved KV mode plus the facts that produced it."""

    mode: DiffusionKVCacheMode
    max_rows_per_request: int | None
    reason: str

    @property
    def is_paged(self) -> bool:
        return self.mode is DiffusionKVCacheMode.PAGED_SCHEDULER


def _paged_incompatibility(
    *,
    step_execution: bool,
    offload_enabled: bool,
    legacy_imported_kv: bool,
    attention_backend: object,
) -> str | None:
    """Return why paged execution cannot run, or ``None`` when it can."""

    if step_execution:
        return (
            "paged_scheduler currently supports request-level execution only; "
            "disable step_execution or set diffusion_kv_mode='dense_legacy'"
        )
    if offload_enabled:
        return (
            "paged_scheduler has not been validated with CPU, layerwise, or distributed "
            "layerwise offload; disable offload or set diffusion_kv_mode='dense_legacy'"
        )
    if legacy_imported_kv:
        return (
            "paged_scheduler Diffusion KV does not support imported AR KV; "
            "disable need_recv_cache until connector-aware import is implemented"
        )
    if not attention_backend_supports_paged_kv(attention_backend):
        return (
            "paged_scheduler requires an attention backend that advertises paged KV support "
            f"({', '.join(sorted(PAGED_KV_ATTENTION_BACKENDS))}); "
            f"got {normalize_attention_backend(attention_backend)!r}"
        )
    return None


def resolve_diffusion_kv_cache_mode(
    mode: object,
    *,
    model_supports_paged_kv: bool = False,
    max_rows_per_request: int | None = None,
    model_default_max_rows: int | None = None,
    step_execution: bool = False,
    offload_enabled: bool = False,
    legacy_imported_kv: bool = False,
    attention_backend: object = None,
    allow_implicit_paged: bool = PAGED_KV_IMPLICIT_DEFAULT_ENABLED,
) -> DiffusionKVResolution:
    """Resolve the diffusion KV ownership mode.

    An explicit ``dense_legacy`` selection is always honored (opt-out), and an
    explicit ``paged_scheduler`` selection is validated strictly: an
    incompatible combination raises instead of silently degrading. When no mode
    is given, paged execution is selected only when implicit paged execution is
    enabled, the model declares support, and the runtime configuration can
    actually run it; every other case stays on the dense path and reports why.
    """

    parsed = parse_diffusion_kv_cache_mode(mode)

    if parsed is DiffusionKVCacheMode.DENSE_LEGACY:
        return DiffusionKVResolution(
            mode=DiffusionKVCacheMode.DENSE_LEGACY,
            max_rows_per_request=max_rows_per_request,
            reason="explicit dense_legacy opt-out",
        )

    if parsed is DiffusionKVCacheMode.PAGED_SCHEDULER:
        incompatibility = _paged_incompatibility(
            step_execution=step_execution,
            offload_enabled=offload_enabled,
            legacy_imported_kv=legacy_imported_kv,
            attention_backend=attention_backend,
        )
        if incompatibility is not None:
            raise ValueError(incompatibility)
        rows = max_rows_per_request if max_rows_per_request is not None else model_default_max_rows
        if rows is None:
            raise ValueError("paged_scheduler requires diffusion_kv_max_rows_per_request to be set")
        return DiffusionKVResolution(
            mode=DiffusionKVCacheMode.PAGED_SCHEDULER,
            max_rows_per_request=rows,
            reason="explicit paged_scheduler",
        )

    # No mode requested: resolve from capability and compatibility.
    if not model_supports_paged_kv:
        return DiffusionKVResolution(
            mode=DiffusionKVCacheMode.DENSE_LEGACY,
            max_rows_per_request=max_rows_per_request,
            reason=REASON_MODEL_NOT_PAGED,
        )
    if not allow_implicit_paged:
        return DiffusionKVResolution(
            mode=DiffusionKVCacheMode.DENSE_LEGACY,
            max_rows_per_request=max_rows_per_request,
            reason=REASON_IMPLICIT_DISABLED,
        )
    incompatibility = _paged_incompatibility(
        step_execution=step_execution,
        offload_enabled=offload_enabled,
        legacy_imported_kv=legacy_imported_kv,
        attention_backend=attention_backend,
    )
    if incompatibility is not None:
        return DiffusionKVResolution(
            mode=DiffusionKVCacheMode.DENSE_LEGACY,
            max_rows_per_request=max_rows_per_request,
            reason=incompatibility,
        )
    rows = max_rows_per_request if max_rows_per_request is not None else model_default_max_rows
    if rows is None:
        return DiffusionKVResolution(
            mode=DiffusionKVCacheMode.DENSE_LEGACY,
            max_rows_per_request=max_rows_per_request,
            reason=REASON_ROW_CAPACITY,
        )
    return DiffusionKVResolution(
        mode=DiffusionKVCacheMode.PAGED_SCHEDULER,
        max_rows_per_request=rows,
        reason="paged_scheduler selected by default for a paged-capable model",
    )
