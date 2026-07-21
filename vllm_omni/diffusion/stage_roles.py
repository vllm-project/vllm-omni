# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diffusion stage-role vocabulary, component ownership, and topology rules.

RFC #4590 (generic disaggregated diffusion execution) represents all three
disaggregated stages as ``StageExecutionType.DIFFUSION`` and distinguishes their
role through ``StagePipelineConfig.model_stage``:

* ``encode``  — validate / text-image-VAE encode / prepare latents+timesteps.
* ``denoise`` — DiT / scheduler loop / AR-Diffusion KV (session state lives here).
* ``decode``  — VAE decode / postprocess / final user-visible output.

The historical single-worker value ``diffusion`` (and the empty / ``None`` role)
denotes the monolithic fallback: one worker owns the whole ``forward()``.

This module is intentionally free of ``torch`` and of any heavy intra-package
import so that role resolution, component ownership, and topology validation can
be reasoned about (and unit-tested) without the model runtime. Keep it a leaf.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, fields
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Role vocabulary
# ---------------------------------------------------------------------------

#: The disaggregated diffusion stage roles, in their canonical linear order.
ENCODE = "encode"
DENOISE = "denoise"
DECODE = "decode"

#: The monolithic single-worker role (one worker owns the whole ``forward()``).
#: Kept for backward compatibility with every existing diffusion deployment.
MONOLITHIC = "diffusion"

#: Roles that mean "run a portion of the diffusion model as its own stage".
DISAGGREGATED_ROLES: frozenset[str] = frozenset({ENCODE, DENOISE, DECODE})

#: Every role this module understands. Unknown roles are not rejected outright
#: (a model may declare support for a custom role), but they never take the
#: built-in encode/denoise/decode execution paths.
KNOWN_ROLES: frozenset[str] = DISAGGREGATED_ROLES | {MONOLITHIC}

#: Canonical linear order for the first-cut ``encode -> denoise -> decode`` path.
LINEAR_ROLE_ORDER: tuple[str, ...] = (ENCODE, DENOISE, DECODE)


def normalize_stage_role(model_stage: str | None) -> str:
    """Return a canonical role string for a raw ``model_stage`` value.

    ``None`` and the empty string collapse to :data:`MONOLITHIC`; every other
    value is returned verbatim after stripping surrounding whitespace (case is
    preserved) so unknown roles stay intact for model-declared extensions rather
    than being silently rewritten. Role matching against the built-in constants
    is therefore case-sensitive.
    """
    if model_stage is None:
        return MONOLITHIC
    role = str(model_stage).strip()
    if not role:
        return MONOLITHIC
    return role


def is_monolithic_role(model_stage: str | None) -> bool:
    """True when the stage owns the whole ``forward()`` (legacy behavior)."""
    return normalize_stage_role(model_stage) == MONOLITHIC


def is_disaggregated_role(model_stage: str | None) -> bool:
    """True when the stage is one of the built-in encode/denoise/decode roles."""
    return normalize_stage_role(model_stage) in DISAGGREGATED_ROLES


def is_known_role(model_stage: str | None) -> bool:
    """True when the role is one this framework has a built-in path for."""
    return normalize_stage_role(model_stage) in KNOWN_ROLES


#: Names of the runner execution path a role selects. Pure mapping, kept here so
#: the runner's dispatch decision is testable without importing the torch-heavy
#: runner module. The runner mirrors this table in ``execute_model``.
EXECUTION_PATH_MONOLITHIC = "monolithic"
EXECUTION_PATH_ENCODE = "encode_stage"
EXECUTION_PATH_DENOISE = "denoise_stage"
EXECUTION_PATH_DECODE = "decode_stage"
EXECUTION_PATH_MODEL_DEFINED = "model_defined_stage"


def resolve_execution_path(model_stage: str | None) -> str:
    """Return which runner path a stage role selects (see EXECUTION_PATH_*).

    * ``diffusion`` / ``None`` / empty -> monolithic (unchanged legacy path).
    * ``encode`` / ``denoise`` / ``decode`` -> the matching disaggregated path.
    * any other declared role -> the model-defined-stage hook.
    """
    role = normalize_stage_role(model_stage)
    if role == MONOLITHIC:
        return EXECUTION_PATH_MONOLITHIC
    if role == ENCODE:
        return EXECUTION_PATH_ENCODE
    if role == DENOISE:
        return EXECUTION_PATH_DENOISE
    if role == DECODE:
        return EXECUTION_PATH_DECODE
    return EXECUTION_PATH_MODEL_DEFINED


# ---------------------------------------------------------------------------
# Component ownership
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageComponentSpec:
    """Which pipeline components a stage must construct and load.

    Queried *before* expensive module construction so a stage worker builds and
    loads only the weights its role needs (RFC #4590 §8). A component the spec
    marks ``False`` is skipped at build time; weight loading then self-gates
    because the loader only fills parameters that actually exist on the module.

    The fields are deliberately generic (not DreamZero-specific): a pipeline
    maps its own submodules onto these coarse buckets. ``action_modules`` covers
    robotics/action heads; models without one simply leave it ``False``.
    """

    tokenizer: bool = False
    text_encoder: bool = False
    image_encoder: bool = False
    vae_encoder: bool = False
    dit: bool = False
    scheduler: bool = False
    vae_decoder: bool = False
    action_modules: bool = False

    def enabled_components(self) -> tuple[str, ...]:
        """Return the names of the components this spec requires, in order."""
        return tuple(f.name for f in fields(self) if getattr(self, f.name))

    def requires(self, component: str) -> bool:
        """Return whether ``component`` is required (unknown names -> False)."""
        return bool(getattr(self, component, False))

    def union(self, other: StageComponentSpec) -> StageComponentSpec:
        """Return a spec requiring every component either spec requires (per-field OR).

        Useful for composing a superset spec from per-stage specs (e.g. a
        "build everything" spec for a monolithic worker). The built-in monolithic
        path instead returns the :data:`ALL_COMPONENTS` singleton directly.
        """
        return StageComponentSpec(
            **{f.name: (getattr(self, f.name) or getattr(other, f.name)) for f in fields(self)}
        )

    def describe(self) -> str:
        """Human-readable component list for startup logging."""
        enabled = self.enabled_components()
        return "+".join(enabled) if enabled else "none"


#: Spec for the monolithic fallback: build everything. Used when a pipeline does
#: not declare disaggregated support or the role is ``diffusion``/``None``.
ALL_COMPONENTS = StageComponentSpec(
    tokenizer=True,
    text_encoder=True,
    image_encoder=True,
    vae_encoder=True,
    dit=True,
    scheduler=True,
    vae_decoder=True,
    action_modules=True,
)


# ---------------------------------------------------------------------------
# Topology validation
# ---------------------------------------------------------------------------


class DiffusionTopologyError(ValueError):
    """Raised when a disaggregated diffusion topology is structurally invalid."""


def _stage_role(stage: Any) -> str:
    return normalize_stage_role(getattr(stage, "model_stage", None))


def validate_linear_diffusion_topology(stages: Iterable[Any]) -> list[str]:
    """Validate a linear ``encode -> denoise -> decode`` diffusion topology.

    ``stages`` is any iterable of stage-like objects exposing ``stage_id``,
    ``model_stage``, ``input_sources`` (iterable of upstream stage ids) and
    ``final_output`` (bool). Returns a list of human-readable error strings;
    empty means valid. This is duck-typed on purpose so it validates both
    ``StagePipelineConfig`` objects and lightweight test doubles, and stays
    free of any config-package import.

    Rules enforced (RFC #4590 §5):

    * unique stage ids;
    * every ``input_sources`` entry references an existing stage (no self-ref);
    * a ``denoise`` stage has at least one ``encode`` (or upstream ``denoise``)
      source;
    * a ``decode`` stage has at least one ``denoise`` source;
    * exactly one stage is marked ``final_output`` and (when a decode stage is
      present) it is the decode stage;
    * a linear path admits a single upstream source per consumer stage.

    Stages whose role is not one of encode/denoise/decode are accepted (a model
    may declare a custom role) but do not participate in the role-adjacency
    checks beyond id/reference validation.
    """
    stage_list = list(stages)
    errors: list[str] = []
    if not stage_list:
        return ["Diffusion topology has no stages."]

    by_id: dict[int, Any] = {}
    for stage in stage_list:
        sid = getattr(stage, "stage_id", None)
        if sid in by_id:
            errors.append(f"Duplicate stage id {sid}.")
        by_id[sid] = stage

    disaggregated = [s for s in stage_list if _stage_role(s) in DISAGGREGATED_ROLES]

    for stage in stage_list:
        sid = getattr(stage, "stage_id", None)
        role = _stage_role(stage)
        sources = list(getattr(stage, "input_sources", ()) or ())

        for src in sources:
            if src == sid:
                errors.append(f"Stage {sid} references itself as an input source.")
            elif src not in by_id:
                errors.append(f"Stage {sid} references non-existent input source {src}.")

        if role == ENCODE and sources:
            errors.append(f"Encode stage {sid} must be an entry point (input_sources must be empty).")

        if role == DENOISE:
            if not sources:
                errors.append(f"Denoise stage {sid} has no upstream source; expected an encode stage.")
            else:
                upstream_roles = {_stage_role(by_id[s]) for s in sources if s in by_id}
                if not (upstream_roles & {ENCODE, DENOISE}):
                    errors.append(
                        f"Denoise stage {sid} has no upstream encode/denoise source (got roles {sorted(upstream_roles)})."
                    )
                if len(sources) > 1:
                    errors.append(
                        f"Denoise stage {sid} has {len(sources)} upstream sources; the linear "
                        "diffusion processor merges exactly one."
                    )

        if role == DECODE:
            if not sources:
                errors.append(f"Decode stage {sid} has no upstream source; expected a denoise stage.")
            else:
                upstream_roles = {_stage_role(by_id[s]) for s in sources if s in by_id}
                if DENOISE not in upstream_roles:
                    errors.append(
                        f"Decode stage {sid} has no upstream denoise source (got roles {sorted(upstream_roles)})."
                    )
                if len(sources) > 1:
                    errors.append(
                        f"Decode stage {sid} has {len(sources)} upstream sources; the linear "
                        "diffusion processor merges exactly one."
                    )

    # Final-output placement: exactly one stage owns the user-visible result and,
    # when the topology is disaggregated, it must be the decode stage.
    final_stages = [s for s in stage_list if getattr(s, "final_output", False)]
    if len(final_stages) != 1:
        errors.append(f"Expected exactly one final_output stage, found {len(final_stages)}.")
    elif disaggregated:
        decode_stages = [s for s in stage_list if _stage_role(s) == DECODE]
        if decode_stages and _stage_role(final_stages[0]) != DECODE:
            errors.append(
                f"final_output must be placed on the decode stage, not the "
                f"{_stage_role(final_stages[0])!r} stage {getattr(final_stages[0], 'stage_id', None)}."
            )

    return errors
