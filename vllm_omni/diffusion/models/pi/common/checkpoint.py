# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Mechanical checkpoint-name translation shared by Pi-family models."""

from collections.abc import Collection, Sequence

PrefixAlias = tuple[str, str]

_PALIGEMMA_PREFIX = "paligemma_with_expert.paligemma."
_PALIGEMMA_MODEL_PREFIX = _PALIGEMMA_PREFIX + "model."
_PALIGEMMA_SUBMODULES = ("vision_tower", "multi_modal_projector", "language_model")
_VISION_TOWER_PREFIX = _PALIGEMMA_MODEL_PREFIX + "vision_tower."
_LM_HEAD_WEIGHT = _PALIGEMMA_PREFIX + "lm_head.weight"
_EMBED_TOKENS_WEIGHT = _PALIGEMMA_MODEL_PREFIX + "language_model.embed_tokens.weight"


def resolve_parameter_name(
    name: str,
    model_keys: Collection[str],
    *,
    prefix_aliases: Sequence[PrefixAlias] = (),
) -> str:
    """Map a LeRobot checkpoint name onto the installed Transformers layout.

    This function only translates names. Callers own compatibility checks,
    tensor loading, completeness audits, and warning/error policy.

    ``prefix_aliases`` expresses variant vocabulary explicitly, for example
    Pi0's ``time_mlp_in.`` → ``action_time_mlp_in.`` mapping. The remaining
    rules are common: strip LeRobot's policy-wrapper prefix, nest PaliGemma
    submodules, and redirect the tied LM head to token embeddings.

    SigLIP nesting is resolved adaptively against ``model_keys``. Transformers
    <=5.3 wraps the encoder in ``vision_tower.vision_model.*`` while >=5.4
    flattens it to ``vision_tower.*``. Rewriting only when the candidate exists
    prevents hundreds of vision parameters from silently missing their target.
    """
    if name.startswith("model."):
        name = name[len("model.") :]

    for source, target in prefix_aliases:
        if name.startswith(source):
            name = target + name[len(source) :]
            break

    for submodule in _PALIGEMMA_SUBMODULES:
        flat = f"{_PALIGEMMA_PREFIX}{submodule}."
        if name.startswith(flat):
            name = f"{_PALIGEMMA_MODEL_PREFIX}{submodule}." + name[len(flat) :]
            break

    if name == _LM_HEAD_WEIGHT:
        name = _EMBED_TOKENS_WEIGHT

    if not name.startswith(_VISION_TOWER_PREFIX) or name in model_keys:
        return name

    rest = name[len(_VISION_TOWER_PREFIX) :]
    if rest.startswith("vision_model."):
        candidate = _VISION_TOWER_PREFIX + rest[len("vision_model.") :]
    else:
        candidate = _VISION_TOWER_PREFIX + "vision_model." + rest
    return candidate if candidate in model_keys else name
