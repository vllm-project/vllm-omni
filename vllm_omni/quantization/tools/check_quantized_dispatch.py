#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that every layer got the quantization method its checkpoint expects.

A quantized checkpoint and a built model agree on which layers are quantized
only by convention: the checkpoint ships an exclusion list, the model presents
prefixes, and a quantization config matches one against the other. When that
match is wrong the model still builds. A layer that should have been excluded
is created quantized and then fails on a scale tensor it has no parameter for;
worse, a layer whose scales are merged incorrectly loads cleanly and produces
confidently wrong output that no latency or memory metric can detect.

So compare the two directly. The checkpoint's own index says which modules
carry scale tensors -- that is ground truth for "this module is quantized" --
and the built model says which modules resolved to a quantized method. Any
disagreement is reported per module.

This reads the safetensors *index* only, never the shards, so it costs a few
kilobytes of I/O regardless of model size.

Usage::

    # Inspect a checkpoint. Reads the index only, so it needs no device and
    # no shard download.
    python -m vllm_omni.quantization.tools.check_quantized_dispatch \\
        --checkpoint /path/to/quantized-transformer

    # Compare against a model you have already built:
    from vllm_omni.quantization.tools.check_quantized_dispatch import (
        checkpoint_quantized_modules, model_quantized_modules, compare,
    )
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

# Suffixes ModelOpt / compressed-tensors ship alongside a packed weight. A
# module carrying any of them was quantized by whoever produced the checkpoint.
SCALE_SUFFIXES = (
    "weight_scale",
    "weight_scale_2",
    "weight_global_scale",
    "input_scale",
    "input_global_scale",
    "weight_scale_inv",
)


def _strip_scale_suffix(name: str) -> str | None:
    """Return the module a scale tensor belongs to, or None if it is not one."""
    for suffix in SCALE_SUFFIXES:
        if name.endswith(f".{suffix}"):
            return name[: -len(suffix) - 1]
    return None


def checkpoint_quantized_modules(index: dict[str, str] | Iterable[str]) -> set[str]:
    """Modules the checkpoint quantized, from its tensor names alone."""
    names = index.keys() if isinstance(index, dict) else index
    modules = set()
    for name in names:
        module = _strip_scale_suffix(name)
        if module is not None:
            modules.add(module)
    return modules


def checkpoint_linear_modules(index: dict[str, str] | Iterable[str]) -> set[str]:
    """Every module carrying a ``.weight``, quantized or not."""
    names = index.keys() if isinstance(index, dict) else index
    return {name[: -len(".weight")] for name in names if name.endswith(".weight")}


def load_index(checkpoint: Path) -> dict[str, str]:
    """Read a safetensors index, or synthesise one from a single-file model."""
    for candidate in sorted(checkpoint.glob("*.safetensors.index.json")):
        return json.loads(candidate.read_text(encoding="utf-8"))["weight_map"]

    shards = sorted(checkpoint.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"{checkpoint} contains neither a safetensors index nor any shard")
    from safetensors import safe_open

    index: dict[str, str] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt") as handle:
            for key in handle.keys():
                index[key] = shard.name
    return index


def model_quantized_modules(model, *, prefix: str = "") -> tuple[set[str], set[str]]:
    """Split a built model's linear modules into quantized and unquantized.

    Membership is decided by the resolved ``quant_method``, which is what the
    kernel will actually run -- not by whether a config was passed in.
    """
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    quantized: set[str] = set()
    unquantized: set[str] = set()
    for name, module in model.named_modules():
        method = getattr(module, "quant_method", None)
        if method is None or not hasattr(module, "weight"):
            continue
        target = quantized if not isinstance(method, UnquantizedLinearMethod) else unquantized
        target.add(f"{prefix}{name}" if prefix else name)
    return quantized, unquantized


@dataclass
class DispatchReport:
    """Where the model and the checkpoint disagree about what is quantized."""

    quantized_in_both: set[str] = field(default_factory=set)
    unquantized_in_both: set[str] = field(default_factory=set)
    # The model quantized it; the checkpoint shipped no scales for it. This
    # fails at load on a missing parameter, or silently uses uninitialised
    # scales if the loader is permissive.
    quantized_without_scales: set[str] = field(default_factory=set)
    # The checkpoint quantized it; the model built it unquantized. Its scale
    # tensors have nowhere to go -- an exclusion list that matched too much.
    scales_without_a_quantized_layer: set[str] = field(default_factory=set)
    # Present in the checkpoint but absent from the model, or the reverse.
    missing_from_model: set[str] = field(default_factory=set)
    missing_from_checkpoint: set[str] = field(default_factory=set)

    @property
    def agrees(self) -> bool:
        return not (
            self.quantized_without_scales
            or self.scales_without_a_quantized_layer
            or self.missing_from_model
            or self.missing_from_checkpoint
        )


def compare(
    *,
    checkpoint_index: dict[str, str] | Iterable[str],
    model_quantized: Iterable[str],
    model_unquantized: Iterable[str],
    ignore: Sequence[str] = (),
) -> DispatchReport:
    """Compare a built model's dispatch against the checkpoint's own tensors.

    ``ignore`` accepts glob-ish patterns for modules the model legitimately
    does not build one-to-one -- a fused projection, for instance, has no
    single checkpoint counterpart.
    """
    patterns = [re.compile(re.escape(p).replace(r"\*", ".*")) for p in ignore]

    def skipped(name: str) -> bool:
        return any(pattern.fullmatch(name) for pattern in patterns)

    ckpt_quantized = {m for m in checkpoint_quantized_modules(checkpoint_index) if not skipped(m)}
    ckpt_linear = {m for m in checkpoint_linear_modules(checkpoint_index) if not skipped(m)}
    quantized = {m for m in model_quantized if not skipped(m)}
    unquantized = {m for m in model_unquantized if not skipped(m)}
    built = quantized | unquantized

    return DispatchReport(
        quantized_in_both=quantized & ckpt_quantized,
        unquantized_in_both=unquantized & (ckpt_linear - ckpt_quantized),
        quantized_without_scales=quantized - ckpt_quantized,
        scales_without_a_quantized_layer=ckpt_quantized - quantized,
        missing_from_model=ckpt_linear - built,
        missing_from_checkpoint=built - ckpt_linear,
    )


def format_report(report: DispatchReport, *, limit: int = 12) -> str:
    """Render the comparison, leading with disagreement rather than totals."""

    def block(title: str, names: set[str], explanation: str) -> list[str]:
        if not names:
            return []
        lines = [f"{title}: {len(names)}", f"  {explanation}"]
        for name in sorted(names)[:limit]:
            lines.append(f"    {name}")
        if len(names) > limit:
            lines.append(f"    ... and {len(names) - limit} more")
        return [*lines, ""]

    lines: list[str] = []
    lines += block(
        "QUANTIZED BUT THE CHECKPOINT HAS NO SCALES",
        report.quantized_without_scales,
        "The exclusion list did not match these prefixes. They will fail on a missing scale.",
    )
    lines += block(
        "CHECKPOINT SCALES WITH NO QUANTIZED LAYER",
        report.scales_without_a_quantized_layer,
        "The exclusion list matched too much, or the module was fused away.",
    )
    lines += block(
        "IN THE CHECKPOINT, NOT IN THE MODEL",
        report.missing_from_model,
        "Fused projections land here legitimately; pass --ignore for those.",
    )
    lines += block(
        "IN THE MODEL, NOT IN THE CHECKPOINT",
        report.missing_from_checkpoint,
        "A fused parameter, or a module the checkpoint does not carry.",
    )

    lines.append(f"agreed: {len(report.quantized_in_both)} quantized, {len(report.unquantized_in_both)} unquantized")
    lines.append("VERDICT: dispatch matches the checkpoint" if report.agrees else "VERDICT: DISAGREEMENT, see above")
    return "\n".join(lines)


def summarise_checkpoint(index: dict[str, str]) -> str:
    """Describe a checkpoint's quantization without building anything."""
    quantized = checkpoint_quantized_modules(index)
    linear = checkpoint_linear_modules(index)
    scales = sum(1 for name in index if _strip_scale_suffix(name) is not None)
    lines = [
        f"tensors:            {len(index)}",
        f"modules with weight:{len(linear):>5}",
        f"  quantized:        {len(quantized):>5}",
        f"  unquantized:      {len(linear - quantized):>5}",
        f"scale tensors:      {scales:>5}",
    ]
    grouped: dict[str, int] = {}
    for module in sorted(quantized):
        head = module.split(".")[0]
        grouped[head] = grouped.get(head, 0) + 1
    lines.append("quantized modules by top-level group:")
    for head, count in sorted(grouped.items()):
        lines.append(f"  {head:<28} {count}")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check quantization dispatch against a checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Directory holding the quantized transformer.")
    parser.add_argument("--output", type=Path, default=None, help="Also write the report here.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Describe a checkpoint's quantization. Needs no device.

    Comparing against a *built* model needs a pipeline, so that half is a
    library call: import :func:`model_quantized_modules` and :func:`compare`
    from wherever the model already exists.
    """
    args = parse_args(argv)
    text = summarise_checkpoint(load_index(args.checkpoint))
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
