# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Immutable integer-step schedules, independent of torch and attention backends."""

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


def require_no_cache_backend(od_config: Any, schedule: AttentionSchedule) -> None:
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


def require_denoise_progress_publisher(pipeline: Any, schedule: AttentionSchedule) -> None:
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
