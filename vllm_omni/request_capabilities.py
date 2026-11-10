# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Typed, optional request facts shared by model owners and discovery.

Omission means unknown. These are implemented input contracts, not recommended
settings or a general-purpose validation language. Route projections contain
the complete effective contract; clients must not merge unrelated routes.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Generic, Literal, TypeVar

from pydantic import Field, TypeAdapter, model_validator
from pydantic.dataclasses import dataclass as validated_dataclass
from typing_extensions import Self

RequestSupport = Literal["supported", "unsupported", "unknown"]
Scalar = str | int | float | bool


@validated_dataclass(frozen=True)
class RequestFact:
    """Remain dataclasses so PipelineConfig's asdict/JSON contract survives."""

    def model_copy(self, *, update: dict[str, object]) -> Self:
        return replace(self, **update)

    def model_dump(self, *, exclude_none: bool = False) -> dict[str, Any]:
        adapter: TypeAdapter[RequestFact] = TypeAdapter(type(self))
        return adapter.dump_python(self, mode="json", exclude_none=exclude_none)

    @classmethod
    def model_validate(cls, value: object) -> Self:
        return TypeAdapter(cls).validate_python(value)


@validated_dataclass(frozen=True)
class InputConstraint(RequestFact):
    support: RequestSupport = "unknown"
    min_count: int | None = Field(default=None, ge=0)
    max_count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def ordered_counts(self) -> "InputConstraint":
        if self.min_count is not None and self.max_count is not None and self.min_count > self.max_count:
            raise ValueError("min_count must not exceed max_count")
        return self


@validated_dataclass(frozen=True)
class ParameterConstraint(RequestFact):
    support: RequestSupport = "unknown"
    type: Literal["string", "integer", "number", "boolean"] | None = None
    enum: tuple[Scalar, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = Field(default=None, gt=0)


@validated_dataclass(frozen=True)
class ImageGeometry(RequestFact):
    state: Literal["known", "unknown"] = "unknown"
    multiple_of: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def known_value(self) -> "ImageGeometry":
        if (self.state == "known") != (self.multiple_of is not None):
            raise ValueError("Known geometry requires an initialized multiple_of")
        return self


@validated_dataclass(frozen=True)
class SourceResize(RequestFact):
    max_pixels: int = Field(gt=0)
    policy: Literal["downscale_preserve_aspect_ratio"] = "downscale_preserve_aspect_ratio"


@validated_dataclass(frozen=True)
class MaskConstraint(RequestFact):
    support: RequestSupport = "unknown"
    source_role: str = "image"
    max_source_count: int | None = Field(default=None, ge=1)
    white: Literal["edit", "preserve"] | None = None
    black: Literal["edit", "preserve"] | None = None
    binarize_threshold: float | None = Field(default=None, ge=0, le=1)
    requires_same_size: bool | None = None
    geometry: ImageGeometry = Field(default_factory=ImageGeometry)
    source_resize: SourceResize | None = None


@validated_dataclass(frozen=True)
class RequestConstraints(RequestFact):
    inputs: dict[str, InputConstraint] = Field(default_factory=dict)
    required_any_of: tuple[tuple[str, ...], ...] = ()
    parameters: dict[str, ParameterConstraint] = Field(default_factory=dict)
    mask: MaskConstraint | None = None
    # Operator-owned limit; not a checkpoint's universal maximum resolution.
    max_output_pixels: int | None = Field(default=None, gt=0)


@validated_dataclass(frozen=True)
class RuntimeServingMetadata(RequestFact):
    """Small public-safe facts carried by a completed startup handshake."""

    image_geometry: ImageGeometry | None = None


_Input = TypeVar("_Input")
_Output = TypeVar("_Output")


@dataclass(frozen=True)
class ServingPostProcess(Generic[_Input, _Output]):
    """A normal callable with optional facts from its existing initialization."""

    process: Callable[[_Input], _Output]
    serving_metadata: RuntimeServingMetadata

    def __call__(self, value: _Input) -> _Output:
        return self.process(value)
