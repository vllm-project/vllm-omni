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


def apply_skip_softmax_calibration(attention_config: Any, model: Any) -> None:
    cfg = attention_config
    specs = getattr(cfg, "per_role", None)
    calibration = None
    for spec in (getattr(cfg, "default", None), *(specs.values() if specs else ())):
        calibration = calibration or getattr(spec, "skip_calibration", None)
    if not calibration:
        return

    stamped = apply_to_pipeline(model, calibration)
    logger.info("Skip-softmax: stamped calibration onto %d attention layer(s).", stamped)


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


def apply_to_pipeline(pipeline, calibration: dict) -> int:
    if not calibration:
        return 0
    stamped = 0
    for name, module in pipeline.named_modules():
        impl = getattr(module, "attention", None)
        set_layer = getattr(impl, "set_layer_calibration", None)
        if set_layer is None:
            continue
        per = resolve_layer_calibration(name, calibration)
        if per and per.get("a") is not None and per.get("b") is not None:
            set_layer(per["a"], per["b"])
            stamped += 1
    return stamped
