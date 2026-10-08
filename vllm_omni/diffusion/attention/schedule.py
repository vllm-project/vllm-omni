# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Immutable step and normalized-noise schedules, independent of torch backends."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

from vllm_omni.errors import OmniClientError

_RANGE_FIELDS = frozenset({"start", "end", "profile"})


class InvalidAttentionScheduleError(OmniClientError):
    """A request's schedule failed admission; entrypoints report it as HTTP 400, not as a server error."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_type="invalid_attention_schedule")


def validate_attention_profile_name(name: str) -> None:
    if not isinstance(name, str):
        raise TypeError("attention schedule profile names must be strings")
    if not name or not name.isascii() or not name[0].isalpha() or not all(c.isalnum() or c in "_-" for c in name):
        raise ValueError(f"Invalid attention schedule profile name {name!r}; expected [A-Za-z][A-Za-z0-9_-]*")


def _validate_step(value: int, name: str, *, minimum: int = 0) -> None:
    # bool is an int subclass, but accepting it makes JSON true a step boundary.
    if type(value) is not int:
        raise TypeError(f"attention_schedule {name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"attention_schedule {name} must be >= {minimum}, got {value!r}")


@dataclass(frozen=True)
class AttentionScheduleRange:
    """Select a prepared profile on [start, end); None ends at the actual total."""

    start: int
    end: int | None
    profile: str

    def __post_init__(self) -> None:
        _validate_step(self.start, "start")
        if self.end is not None:
            _validate_step(self.end, "end")
            if self.end <= self.start:
                raise ValueError("attention_schedule end must be greater than start")
        validate_attention_profile_name(self.profile)


AttentionSchedule: TypeAlias = tuple[AttentionScheduleRange, ...]


def parse_attention_schedule(value: Any) -> AttentionSchedule | None:
    """Detach and normalize request intervals; None inherits, an empty tuple disables.

    JSON request bodies supply lists of mappings. Tuples and typed ranges also
    support normalized sampling parameters and dataclass serialization.
    """
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise TypeError("attention_schedule must be a list of ranges or None")

    ranges: list[AttentionScheduleRange] = []
    for item in value:
        if isinstance(item, AttentionScheduleRange):
            entry = item
        elif isinstance(item, Mapping):
            if set(item) != _RANGE_FIELDS:
                raise ValueError("attention_schedule ranges require exactly start, end and profile")
            entry = AttentionScheduleRange(**dict(item))
        else:
            raise TypeError("attention_schedule entries must be mappings or AttentionScheduleRange objects")
        if ranges and (ranges[-1].end is None or entry.start < ranges[-1].end):
            raise ValueError("attention_schedule ranges must be ordered and must not overlap")
        ranges.append(entry)
    return tuple(ranges)


def validate_attention_schedule(
    schedule: AttentionSchedule,
    *,
    profiles: Collection[str] | None = None,
    total_steps: int | None = None,
) -> None:
    """Check normalized intervals against prepared names and the actual sequence.

    Startup can validate names without a sequence. Request preparation must
    validate the actual total before any denoising forward, including inherited
    defaults. No range is clipped to fit a shorter request.
    """
    if profiles is not None:
        unknown = sorted({entry.profile for entry in schedule if entry.profile not in profiles})
        if unknown:
            raise ValueError(f"attention_schedule references unknown profile(s): {unknown}")
    if total_steps is not None:
        _validate_step(total_steps, "total_steps", minimum=1)
        for entry in schedule:
            if entry.start >= total_steps or (entry.end is not None and entry.end > total_steps):
                raise ValueError(f"attention_schedule range {entry!r} exceeds total_steps={total_steps}")


def require_attention_schedule_fits(schedule: AttentionSchedule, total_steps: int) -> None:
    """Reject, as a client error, a schedule whose ranges do not fit the actual denoise sequence.

    Publishers call this once they know the sequence they will run and before its first denoise
    forward. The actual total can differ from the requested num_inference_steps (fixed DMD tables,
    FastH3 positions), so callers pass the length of the sequence they built, not the request field.
    """
    try:
        validate_attention_schedule(schedule, total_steps=total_steps)
    except InvalidAttentionScheduleError:
        raise
    except ValueError as exc:
        raise InvalidAttentionScheduleError(str(exc)) from exc


def resolve_attention_schedule(
    request_schedule: Any,
    default_schedule: Any,
    *,
    profiles: Collection[str],
    total_steps: int | None = None,
) -> AttentionSchedule:
    """Resolve inherit/disable/replace without changing either input."""
    effective = parse_attention_schedule(default_schedule if request_schedule is None else request_schedule)
    schedule = () if effective is None else effective
    validate_attention_schedule(schedule, profiles=profiles, total_steps=total_steps)
    return schedule


def validate_request_attention_schedule(request: Any, od_config: Any) -> AttentionSchedule:
    """Resolve one request against the service schedule. Raises before any denoiser runs."""
    sampling = getattr(request, "sampling_params", None)
    request_schedule = getattr(sampling, "attention_schedule", None) if sampling is not None else None
    service = getattr(od_config, "diffusion_attention_schedule", None) if od_config is not None else None
    profiles = tuple(getattr(service, "profiles", None) or ())
    default = getattr(service, "default", None)
    return resolve_attention_schedule(request_schedule, default, profiles=profiles)


def validate_request_attention_sigma_schedule(request: Any, od_config: Any) -> AttentionSigmaSchedule:
    """Resolve and validate noise windows before request admission."""
    sampling = SimpleRequest(request).sampling_params
    service = getattr(od_config, "diffusion_attention_schedule", None) if od_config is not None else None
    return resolve_attention_sigma_schedule(
        getattr(sampling, "attention_sigma_schedule", None),
        getattr(service, "sigma", None),
        profiles=tuple(getattr(service, "profiles", None) or ()),
    )


def validate_request_attention_schedules(request: Any, od_config: Any) -> None:
    step = validate_request_attention_schedule(SimpleRequest(request), od_config)
    sigma = validate_request_attention_sigma_schedule(request, od_config)
    reject_mixed_attention_schedules(step, sigma)


def resolve_batch_attention_sigma_schedule(states: Any, od_config: Any) -> AttentionSigmaSchedule:
    """One resolved sigma schedule for a batch. Different windows must not share a denoise forward."""
    service = getattr(od_config, "diffusion_attention_schedule", None) if od_config is not None else None
    profiles = tuple(getattr(service, "profiles", None) or ())
    default = getattr(service, "sigma", None)
    resolved = []
    for state in states:
        sampling = getattr(state, "sampling_params", None)
        if sampling is None:
            sampling = getattr(state, "sampling", None)
        request_schedule = getattr(sampling, "attention_sigma_schedule", None) if sampling is not None else None
        resolved.append(resolve_attention_sigma_schedule(request_schedule, default, profiles=profiles))
    if not resolved:
        return ()
    first = resolved[0]
    if any(item != first for item in resolved[1:]):
        raise ValueError("attention_sigma_schedule values in one batch must be identical")
    return first


def resolve_batch_attention_schedule(states: Any, od_config: Any) -> AttentionSchedule:
    """One resolved schedule for a batch. Different ranges must not share a denoise forward."""
    resolved = [validate_request_attention_schedule(SimpleRequest(state), od_config) for state in states]
    if not resolved:
        return ()
    first = resolved[0]
    if any(item != first for item in resolved[1:]):
        raise ValueError("attention_schedule values in one batch must be identical")
    return first


def require_request_attention_schedule_fits(request: Any, od_config: Any, total_steps: int) -> AttentionSchedule:
    """Step-mode form of require_attention_schedule_fits for one request or runner state.

    Step-mode preparation runs before the runner opens the forward context, so the bound schedule is
    not available there; this resolves the request's own schedule against the service default.
    """
    schedule = validate_request_attention_schedule(SimpleRequest(request), od_config)
    if schedule:
        require_attention_schedule_fits(schedule, total_steps)
    return schedule


class SimpleRequest:
    """Adapter so a runner state with ``sampling`` satisfies the request validator."""

    def __init__(self, state: Any) -> None:
        self.sampling_params = getattr(state, "sampling_params", None)
        if self.sampling_params is None:
            self.sampling_params = getattr(state, "sampling", None)


def require_no_cache_backend(od_config: Any, schedule: AttentionSchedule | AttentionSigmaSchedule) -> None:
    """A non-empty schedule cannot run with a cache backend that reuses or skips transformer evaluations.

    TeaCache-style backends skip DiT evaluations on some steps and add a residual cached at an
    earlier step, which may have run a different profile. The selected attention would then not be
    what ran, so the combination is rejected instead of invalidating caches at profile switches.
    """
    if not schedule:
        return
    cache_backend = getattr(od_config, "cache_backend", None) if od_config is not None else None
    if cache_backend in (None, "none"):
        return
    raise ValueError(
        f"attention_schedule cannot be combined with cache_backend={cache_backend!r}: the cache backend "
        "reuses or skips transformer evaluations across denoise steps, so the scheduled attention would "
        "not run as selected. Disable the cache backend or send attention_schedule=[]."
    )


def require_denoise_progress_publisher(pipeline: Any, schedule: AttentionSchedule | AttentionSigmaSchedule) -> None:
    """A non-empty schedule cannot enter denoise on a pipeline that never publishes a step."""
    if not schedule:
        return
    if callable(getattr(pipeline, "record_denoise_step", None)):
        return
    raise ValueError(
        "attention_schedule requires a pipeline that publishes denoise progress via record_denoise_step; "
        "rejecting before denoise"
    )


def select_attention_profile(schedule: AttentionSchedule, step_index: int, *, total_steps: int) -> str | None:
    """Return the selected name, or None for the original configuration in gaps."""
    _validate_step(total_steps, "total_steps", minimum=1)
    _validate_step(step_index, "step_index")
    if step_index >= total_steps:
        raise ValueError(f"attention_schedule step_index={step_index} must be less than total_steps={total_steps}")
    validate_attention_schedule(schedule, total_steps=total_steps)
    for entry in schedule:
        if step_index < entry.start:
            break
        if entry.end is None or step_index < entry.end:
            return entry.profile
    return None


_SIGMA_FIELDS = frozenset({"low", "high", "profile"})


@dataclass(frozen=True)
class AttentionSigmaWindow:
    """Select a profile on [low, high); a window ending at 1 also includes 1."""

    low: float
    high: float
    profile: str

    def __post_init__(self) -> None:
        _validate_sigma(self.low, "low")
        _validate_sigma(self.high, "high")
        if self.high <= self.low:
            raise ValueError("attention_sigma_schedule high must be greater than low")
        validate_attention_profile_name(self.profile)


AttentionSigmaSchedule: TypeAlias = tuple[AttentionSigmaWindow, ...]


def _validate_sigma(value: float, name: str) -> None:
    if isinstance(value, bool) or type(value) not in (int, float):
        raise TypeError(f"attention_sigma_schedule {name} must be a number, got {value!r}")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")) or not 0.0 <= number <= 1.0:
        raise ValueError(f"attention_sigma_schedule {name} must be in [0, 1], got {value!r}")


def parse_attention_sigma_schedule(value: Any) -> AttentionSigmaSchedule | None:
    """Detach sigma windows. None inherits, an empty tuple disables."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise TypeError("attention_sigma_schedule must be a list of windows or None")
    windows: list[AttentionSigmaWindow] = []
    for item in value:
        if isinstance(item, AttentionSigmaWindow):
            entry = item
        elif isinstance(item, Mapping):
            if set(item) != _SIGMA_FIELDS:
                raise ValueError("attention_sigma_schedule windows require exactly low, high and profile")
            entry = AttentionSigmaWindow(**dict(item))
        else:
            raise TypeError("attention_sigma_schedule entries must be mappings or AttentionSigmaWindow objects")
        if windows and entry.low < windows[-1].high:
            raise ValueError("attention_sigma_schedule windows must be ordered and must not overlap")
        windows.append(entry)
    return tuple(windows)


def validate_attention_sigma_schedule(
    schedule: AttentionSigmaSchedule,
    *,
    profiles: Collection[str] | None = None,
) -> None:
    if profiles is not None:
        unknown = sorted({entry.profile for entry in schedule if entry.profile not in profiles})
        if unknown:
            raise ValueError(f"attention_sigma_schedule references unknown profile(s): {unknown}")


def resolve_attention_sigma_schedule(
    request_schedule: Any,
    default_schedule: Any,
    *,
    profiles: Collection[str],
) -> AttentionSigmaSchedule:
    effective = parse_attention_sigma_schedule(default_schedule if request_schedule is None else request_schedule)
    schedule = () if effective is None else effective
    validate_attention_sigma_schedule(schedule, profiles=profiles)
    return schedule


def normalized_sigma(current: float, reference: float = 1.0) -> float:
    """Normalize scheduler noise; flow-matching sigma already uses reference=1."""
    import math

    if isinstance(current, bool) or isinstance(reference, bool):
        raise TypeError("normalized sigma inputs must be numbers")
    current, reference = float(current), float(reference)
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError("normalized sigma reference must be finite and positive")
    value = current / reference
    _validate_sigma(value, "sigma")
    return value


def select_attention_profile_by_sigma(schedule: AttentionSigmaSchedule, sigma: float) -> str | None:
    """Return the selected name, or None when the normalized sigma is in a gap."""
    _validate_sigma(sigma, "sigma")
    validate_attention_sigma_schedule(schedule)
    for entry in schedule:
        if entry.low <= float(sigma) < entry.high or entry.high == sigma == 1.0:
            return entry.profile
    return None


def reject_mixed_attention_schedules(step_schedule: AttentionSchedule, sigma_schedule: AttentionSigmaSchedule) -> None:
    if step_schedule and sigma_schedule:
        raise ValueError("a request cannot combine attention_schedule with attention_sigma_schedule")
