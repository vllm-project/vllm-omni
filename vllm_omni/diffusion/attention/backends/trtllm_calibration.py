# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import fnmatch
import logging
from typing import Any

logger = logging.getLogger(__name__)

_SUPPORTED_FORMULA = "a*exp(b*target_sparsity)"


def parse_sparse_attention_config(sac: dict | None) -> dict | None:
    if not sac:
        return None
    group: dict = {}
    for g in (sac.get("config_groups") or {}).values():
        if isinstance(g, dict) and (g.get("algorithm") == "skip_softmax" or g.get("threshold_scale_factor")):
            group = g
            break
    tsf = group.get("threshold_scale_factor") or sac.get("threshold_scale_factor") or {}
    ab = tsf.get("coefficients") or tsf.get("prefill") or {}
    if "a" not in ab or "b" not in ab:
        return None

    formula = (tsf.get("formula") or _SUPPORTED_FORMULA).replace(" ", "")
    if formula != _SUPPORTED_FORMULA:
        raise ValueError(
            f"unsupported skip-softmax formula {tsf.get('formula')!r}; this backend implements '{_SUPPORTED_FORMULA}'."
        )

    return {"a": float(ab["a"]), "b": float(ab["b"]), "ignore": list(group.get("ignore") or [])}


def parse_secondary_expert_calibration(model: str | None) -> dict | None:
    from vllm.transformers_utils.config import get_hf_file_to_dict

    if not model:
        return None
    try:
        cfg = get_hf_file_to_dict("transformer_2/config.json", model)
        return parse_sparse_attention_config(cfg.get("sparse_attention_config")) if cfg else None
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not read transformer_2 skip-softmax calibration (%s); transformer_2 will stay uncalibrated (dense).",
            exc,
        )
        return None


def propagate_skip_softmax_calibration(specs: list, model: str | None, tf_config: Any) -> None:
    calibration = parse_sparse_attention_config(
        tf_config.get("sparse_attention_config") if tf_config is not None else None
    )
    if calibration is not None:
        by_expert = {"transformer": calibration}
        ckpt2 = parse_secondary_expert_calibration(model)
        if ckpt2 is not None:
            by_expert["transformer_2"] = ckpt2
        calibration = {"by_expert": by_expert}

    if calibration is None:
        for spec in specs:
            ss = spec.skip_softmax
            if ss is not None and ss.target_sparsity is not None and ss.threshold is None:
                raise ValueError(
                    f"target_sparsity was requested but the checkpoint for model '{model}' carries no "
                    f"skip-softmax calibration. Either set skip_softmax.threshold for the "
                    f"calibration-free path or load a calibrated checkpoint."
                )
        return

    for spec in specs:
        if spec.skip_calibration is None:
            spec.skip_calibration = calibration

    experts = list(calibration["by_expert"])
    logger.info("Loaded skip-softmax calibration from checkpoint (%d curve(s): %s).", len(experts), ", ".join(experts))


def collect_calibration_specs(attention_config: Any, schedule: Any = None) -> list:
    """Baseline specs plus every schedule profile's specs, in a stable order.

    Both calibration discovery and the target-sparsity-without-calibration check have to see
    non-active profiles, otherwise a profile the service default never references keeps an
    undetected calibration gap. Without a schedule this returns exactly the baseline specs.
    """
    specs: list = []

    def _extend(cfg: Any) -> None:
        if cfg is None:
            return
        default = getattr(cfg, "default", None)
        if default is not None:
            specs.append(default)
        per_role = getattr(cfg, "per_role", None) or {}
        specs.extend(spec for spec in per_role.values() if spec is not None)

    _extend(attention_config)
    profiles = getattr(schedule, "profiles", None) or {}
    for name in sorted(profiles):
        _extend(profiles[name])
    return specs


def layer_calibration_is_ignored(module_name: str, calibration: dict | None) -> bool:
    """Whether this layer stays dense because the calibration explicitly ignores it.

    Separates a preserved legitimate dense fallback from a layer whose expert simply carries no
    curve: ``resolve_layer_calibration`` returns ``None`` for both, but only the first one is
    acceptable when a profile asked for ``target_sparsity``.
    """
    if not calibration:
        return False
    by_expert = calibration.get("by_expert")
    if by_expert:
        key = select_expert(module_name, by_expert.keys())
        if key is None:
            return False
        entry = by_expert[key]
    else:
        entry = calibration
    return is_ignored(module_name, entry.get("ignore"))


def resolve_effective_calibration(attention_config: Any, schedule: Any = None) -> dict | None:
    """Fallback calibration for the baseline and for candidates that carry none of their own.

    This is the first non-empty ``skip_calibration`` over ``collect_calibration_specs`` (baseline
    default, baseline per-role, then profiles in sorted order). A prepared candidate with its own
    ``skip_calibration`` is stamped and validated from that dict instead.
    """
    for spec in collect_calibration_specs(attention_config, schedule):
        calibration = getattr(spec, "skip_calibration", None)
        if calibration:
            return calibration
    return None


def calibration_for_candidate(record: Any, fallback: dict | None) -> dict | None:
    """The dict that stamping writes onto this candidate impl."""
    spec = getattr(record, "spec", None)
    own = getattr(spec, "skip_calibration", None) if spec is not None else None
    if own:
        return own
    return fallback


def apply_skip_softmax_calibration(attention_config: Any, model: Any, schedule: Any = None) -> None:
    fallback = resolve_effective_calibration(attention_config, schedule)
    stamped = apply_to_pipeline(model, fallback)
    if not stamped:
        return
    logger.debug("Skip-softmax: stamped calibration onto %d attention impl(s).", stamped)


_EXPERT_PREFIXES = ("transformer.", "transformer_2.")


def layer_match_names(module_name: str) -> tuple[str, ...]:
    names = {module_name, module_name.replace("._orig_mod.", ".")}
    for name in tuple(names):
        for pfx in _EXPERT_PREFIXES:
            if name.startswith(pfx):
                names.add(name[len(pfx) :])
    return tuple(names)


def is_ignored(module_name: str, ignore_patterns) -> bool:
    names = layer_match_names(module_name)
    for p in ignore_patterns or ():
        for n in names:
            if fnmatch.fnmatch(n, p) or fnmatch.fnmatch(n, p + ".*"):
                return True
    return False


def select_expert(module_name: str, expert_keys) -> str | None:
    for key in sorted(expert_keys or (), key=len, reverse=True):
        if module_name == key or module_name.startswith(key + "."):
            return key
    return None


def resolve_layer_calibration(module_name: str, calibration: dict) -> dict | None:
    if not calibration:
        return None
    by_expert = calibration.get("by_expert")
    if by_expert:
        key = select_expert(module_name, by_expert.keys())
        if key is None:
            return None
        entry = by_expert[key]
    else:
        entry = calibration
    if is_ignored(module_name, entry.get("ignore")):
        return None
    return {"a": entry.get("a"), "b": entry.get("b")}


def _stamp_impl(impl, module_name: str, calibration: dict | None, seen: set[int]) -> int:
    if impl is None or calibration is None or id(impl) in seen:
        return 0
    set_layer = getattr(impl, "set_layer_calibration", None)
    if set_layer is None:
        return 0
    per = resolve_layer_calibration(module_name, calibration)
    if not per or per.get("a") is None or per.get("b") is None:
        return 0
    set_layer(per["a"], per["b"])
    seen.add(id(impl))
    return 1


def apply_to_pipeline(pipeline, calibration: dict | None) -> int:
    """Stamp the baseline from ``calibration`` and each candidate from its own dict or that fallback.

    A shared candidate impl is stamped once. With no schedule, candidates are empty and this is the
    baseline path. ``calibration`` may be None when every stampable curve lives on a candidate spec.
    """
    stamped = 0
    for name, module in pipeline.named_modules():
        seen: set[int] = set()
        baseline = getattr(module, "attention", None)
        stamped += _stamp_impl(baseline, name, calibration, seen)
        candidates = getattr(module, "_schedule_candidates", None) or {}
        # Records are not necessarily hashable (tests use SimpleNamespace). Identity dedup is
        # by impl object inside _stamp_impl, so a shared impl is stamped once even if two names
        # point at it.
        for record in candidates.values():
            own = calibration_for_candidate(record, calibration)
            stamped += _stamp_impl(getattr(record, "impl", None), name, own, seen)
    return stamped
