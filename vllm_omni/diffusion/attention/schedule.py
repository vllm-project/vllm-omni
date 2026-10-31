# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Immutable integer-step schedules, independent of torch and attention backends."""

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from vllm_omni.diffusion.models.interface import role_loads_component
from vllm_omni.errors import OmniClientError

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import AttentionScheduleConfig
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

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


def parse_attention_schedule(value: object) -> AttentionSchedule | None:
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

    Startup can validate names without a sequence. Request preparation validates
    a request's own ranges against the actual total before any denoising forward.
    A bounded range must fit the sequence. An open-ended range that starts at or
    after the end selects nothing, like a gap, so a service default such as
    ``[6, null)`` leaves shorter requests on the base attention.
    """
    if profiles is not None:
        unknown = sorted({entry.profile for entry in schedule if entry.profile not in profiles})
        if unknown:
            raise ValueError(f"attention_schedule references unknown profile(s): {unknown}")
    if total_steps is not None:
        _validate_step(total_steps, "total_steps", minimum=1)
        for entry in schedule:
            if entry.end is not None and entry.end > total_steps:
                raise ValueError(f"attention_schedule range {entry!r} exceeds total_steps={total_steps}")


def require_attention_schedule_fits(schedule: AttentionSchedule, total_steps: int) -> None:
    """Reject, as a client error, a schedule whose ranges do not fit the actual denoise sequence.

    Callers pass the length of the sequence they built, which can differ from the requested
    num_inference_steps (fixed DMD tables, FastH3 positions), before its first denoise forward.
    """
    try:
        validate_attention_schedule(schedule, total_steps=total_steps)
    except ValueError as exc:
        raise InvalidAttentionScheduleError(str(exc)) from exc


def resolve_attention_schedule(
    request_schedule: object,
    default_schedule: object,
    *,
    profiles: Collection[str],
) -> AttentionSchedule:
    """Resolve inherit/disable/replace without changing either input."""
    effective = parse_attention_schedule(default_schedule if request_schedule is None else request_schedule)
    schedule = () if effective is None else effective
    validate_attention_schedule(schedule, profiles=profiles)
    return schedule


def resolve_request_attention_schedule(
    sampling_params: "OmniDiffusionSamplingParams", service: "AttentionScheduleConfig | None"
) -> AttentionSchedule:
    """Resolve one request's schedule against the service configuration."""
    if service is None:
        return resolve_attention_schedule(sampling_params.attention_schedule, (), profiles=())
    return resolve_attention_schedule(sampling_params.attention_schedule, service.default, profiles=service.profiles)


def batch_attention_schedule(schedules: Iterable[AttentionSchedule]) -> AttentionSchedule:
    """One schedule for a batch of resolved schedules. Different ranges must not share a denoise forward."""
    distinct = set(schedules)
    if len(distinct) > 1:
        raise ValueError("attention_schedule values in one batch must be identical")
    return distinct.pop() if distinct else ()


def resolve_batch_attention_schedule(
    sampling_params: Iterable["OmniDiffusionSamplingParams"], service: "AttentionScheduleConfig | None"
) -> AttentionSchedule:
    """Resolve each request's schedule and return the one schedule of the batch."""
    return batch_attention_schedule(resolve_request_attention_schedule(params, service) for params in sampling_params)


def require_request_attention_schedule_fits(
    sampling_params: "OmniDiffusionSamplingParams", service: "AttentionScheduleConfig | None", total_steps: int
) -> AttentionSchedule:
    """Resolve one request's schedule and check it against the sequence the pipeline built.

    Step-mode preparation runs before the runner binds a schedule, so pipelines resolve the
    request's own schedule here instead of reading the forward context. An inherited service
    default serves requests of any length, so it is not checked: a default range that ends after
    this sequence applies up to its last step.
    """
    schedule = resolve_request_attention_schedule(sampling_params, service)
    if schedule and sampling_params.attention_schedule is not None:
        require_attention_schedule_fits(schedule, total_steps)
    return schedule


# Pipeline classes (as registered in the diffusion model registry) that publish denoise progress
# for every denoise forward and honour configured profiles, mapped to the model name in errors.
ATTENTION_SCHEDULE_PIPELINES: Mapping[str, str] = {
    "HunyuanImage3Pipeline": "HunyuanImage-3.0",
    "MiniMaxH3Pipeline": "MiniMax-H3",
    "Wan22I2VPipeline": "Wan2.2",
    "Wan22Pipeline": "Wan2.2",
    "Wan22S2VPipeline": "Wan2.2",
    "Wan22VACEPipeline": "Wan2.2",
}

# Two kinds of diffusion stage run no denoising: a decode-only pipeline class listed here, and a
# stage whose role loads no DiT, such as the encode stage of a Wan2.2 EG or EGD deployment and the
# EGD decode stage, which use the same pipeline class as the denoising stage. Multi-stage endpoints
# copy the request's diffusion params to every diffusion stage, so these stages drop the request
# schedule instead of rejecting it.
DECODE_ONLY_PIPELINES = frozenset({"MiniMaxH3DecoderPipeline"})


def stage_runs_denoising(model_class_name: str | None, stage_role: str) -> bool:
    """Whether a diffusion stage runs the denoise loop that attention schedules apply to.

    ``stage_role`` is the stage's role as ``resolve_diffusion_stage_role`` returns it.
    """
    return model_class_name not in DECODE_ONLY_PIPELINES and role_loads_component(stage_role, "dit")


def unsupported_attention_schedule_error(reason: str) -> ValueError:
    """The one startup error for a schedule configuration this service cannot run yet."""
    models = ", ".join(sorted(set(ATTENTION_SCHEDULE_PIPELINES.values())))
    return ValueError(
        f"Unsupported diffusion_attention_schedule: {reason}. Attention schedules currently support "
        f"step ranges ('profiles' and 'default') on {models} only. Remove the schedule, or the "
        "unsupported part, from this model's config."
    )


def require_attention_schedule_pipeline(pipeline_class_name: str | None, model_class_name: str | None) -> None:
    """Reject configured profiles at startup on a pipeline that is not known to honour them."""
    if pipeline_class_name in ATTENTION_SCHEDULE_PIPELINES:
        return
    raise unsupported_attention_schedule_error(f"model {model_class_name!r} does not support attention profiles yet")


def select_attention_profile(schedule: AttentionSchedule, step_index: int, *, total_steps: int) -> str | None:
    """Return the profile selected at ``step_index``, or None for the base attention in gaps.

    The schedule was validated against ``total_steps`` before denoising started, so this
    only walks the ranges.
    """
    if not 0 <= step_index < total_steps:
        raise ValueError(f"attention_schedule step_index={step_index} must be in [0, total_steps={total_steps})")
    for entry in schedule:
        if step_index < entry.start:
            break
        if entry.end is None or step_index < entry.end:
            return entry.profile
    return None
