# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``vllm-omni doctor``: report what this machine can run.

The command answers two questions for a consumer-device deployment before any
weights are downloaded: how much device memory is available, and which
deployment variants document a minimum at or below that. The variant data is
generated from ``recipes/models/`` by ``tools/gen_memory_tiers.py``, so the
answer tracks the recipes instead of a hand-maintained list.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from vllm.entrypoints.cli.types import CLISubcommand
from vllm.logger import init_logger

from vllm_omni.utils.hardware import MIB_PER_GIB, Device, detect_devices

if TYPE_CHECKING:
    from vllm_omni.utils.tracking_parser import TrackingArgumentParser

logger = init_logger(__name__)

TIERS_PATH = Path(__file__).resolve().parents[2] / "utils" / "memory_tiers.json"
_MAX_SHOWN_MODELS = 8
_UNMEASURED_NOTE = "not yet measured, not unsupported"

# A documented hardware slug names one card or one manufacturer. The slugs are
# owned by the recipe renderer; this table only maps the values recipes actually
# use to the substrings a local device reports. A card that does not appear here
# is never assumed to match an allowlist.
_HARDWARE_HINTS: dict[str, tuple[str, ...]] = {
    "h100": ("h100",),
    "h200": ("h200",),
    "b200": ("b200",),
    "gb200": ("gb200",),
    "b300": ("b300",),
    "gb300": ("gb300",),
    "rtx_pro_6000_8x": ("rtx pro 6000",),
    "rtx_pro_6000": ("rtx pro 6000",),
    "rtx_pro_5000": ("rtx pro 5000",),
    "dgx_spark_gb10": ("gb10",),
    "mi300x": ("mi300x",),
    "mi325x": ("mi325x",),
    "mi355x": ("mi355x",),
}

_DEVICE_BRAND = {"cuda": "NVIDIA", "rocm": "AMD", "mps": "Apple", "npu": "Ascend", "xpu": "Intel"}

# Reason codes travel through the JSON output; the text report groups by them.
_REASON_LABELS = {
    "system-memory": "system memory",
    "unified-memory": "unified memory",
    "backend-not-declared": "recipe declares no backend",
    "backend-excluded": "recipe documents another backend",
    "brand-mismatch": "recipe documents another vendor",
    "hardware-allowlist": "recipe lists specific hardware",
}

COMPATIBLE = "compatible"
CAPACITY_ONLY = "capacity_only"
NEEDS_HARDWARE = "needs_hardware"


def load_documented_variants() -> list[dict[str, object]]:
    """Read the packaged minimum-memory index generated from ``recipes/models/``."""
    return json.loads(TIERS_PATH.read_text(encoding="utf-8"))


def device_tier_gib(device: Device) -> int | None:
    """Whole-GiB tier a device belongs to.

    Devices are marketed in whole GiB but report slightly less (a 16 GiB GPU
    exposes ~15.9 GiB), and recipes bound memory in whole GiB, so round to the
    nearest tier instead of flooring the device out of its own band.
    """
    if device.total_mib is None:
        return None
    return round(device.total_mib / MIB_PER_GIB)


def fitting_variants(device: Device, variants: list[dict[str, object]]) -> list[dict[str, object]]:
    """Variants whose documented minimum fits the device's memory.

    Capacity alone is not a fit: a variant is only compatible when its memory
    bound fits, its recipe declares the device's backend, the device has a
    dedicated memory pool, and any hardware allowlist includes the card. Rows
    that fit on capacity but fail one of those gates are reported separately
    with their reason instead of being counted as fits.
    """
    return [row[0] for row in classify_variants(device, variants) if row[1] == COMPATIBLE]


def device_hardware_slugs(device: Device) -> set[str]:
    """Documented hardware slugs this device matches by name."""
    name = device.name.lower()
    return {slug for slug, hints in _HARDWARE_HINTS.items() if any(hint in name for hint in hints)}


def _gate(device: Device, variant: dict[str, object]) -> tuple[str, str]:
    """Compatibility gate for one capacity-fitting variant: ``(status, reason)``."""
    if device.kind == "cpu":
        return CAPACITY_ONLY, "system-memory"
    if device.unified_memory:
        return CAPACITY_ONLY, "unified-memory"
    backends = tuple(str(backend) for backend in variant.get("backends") or ())
    if not backends:
        return CAPACITY_ONLY, "backend-not-declared"
    if device.kind not in backends:
        return CAPACITY_ONLY, "backend-excluded"
    brand = str(variant.get("brand") or "")
    if brand and _DEVICE_BRAND.get(device.kind) != brand:
        return CAPACITY_ONLY, "brand-mismatch"
    allowlist = {str(slug) for slug in variant.get("hardware") or ()}
    if allowlist and not (device_hardware_slugs(device) & allowlist):
        return NEEDS_HARDWARE, "hardware-allowlist"
    return COMPATIBLE, ""


def classify_variants(device: Device, variants: list[dict[str, object]]) -> list[tuple[dict[str, object], str, str]]:
    """Every variant that fits the device's memory, with its compatibility status."""
    tier_gib = device_tier_gib(device)
    if tier_gib is None:
        return []
    rows: list[tuple[dict[str, object], str, str]] = []
    for variant in variants:
        if int(variant["minimum_gib"]) > tier_gib:
            continue
        status, reason = _gate(device, variant)
        rows.append((variant, status, reason))
    return rows


def _models(rows: list[tuple[dict[str, object], str, str]]) -> list[str]:
    return sorted({str(row[0]["model_id"]) for row in rows})


def _shown(rows: list[tuple[dict[str, object], str, str]]) -> str:
    models = _models(rows)
    shown = ", ".join(f"`{model}`" for model in models[:_MAX_SHOWN_MODELS])
    if len(models) > _MAX_SHOWN_MODELS:
        shown += f", and {len(models) - _MAX_SHOWN_MODELS} more"
    return shown


def _counted(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _verb(count: int, singular: str, plural: str) -> str:
    return singular if count == 1 else plural


def _slugs(variant: dict[str, object]) -> str:
    return ", ".join(str(slug) for slug in variant.get("hardware") or ()) or "-"


def _reasons(rows: list[tuple[dict[str, object], str, str]]) -> str:
    return ", ".join(sorted({_REASON_LABELS.get(reason, reason) for _, _, reason in rows}))


def render_device(device: Device, variants: list[dict[str, object]]) -> list[str]:
    lines = [f"- {device.label}"]
    if device.total_mib is None:
        lines.append("  memory could not be read, so no documented tier was compared")
        return lines

    rows = classify_variants(device, variants)
    tier = device_tier_gib(device)
    compatible = [row for row in rows if row[1] == COMPATIBLE]
    capacity = [row for row in rows if row[1] == CAPACITY_ONLY]
    needs = [row for row in rows if row[1] == NEEDS_HARDWARE]

    if compatible:
        models = _models(compatible)
        lines.append(
            f"  {_counted(len(compatible), 'documented variant')} across {_counted(len(models), 'model')}"
            f" fit: {_shown(compatible)}"
        )
    else:
        lines.append(f"  no variant is documented for this device at or below {tier} GiB")

    if capacity:
        lines.append(
            f"  {_counted(len(capacity), 'more variant')} {_verb(len(capacity), 'fits', 'fit')}"
            f" on capacity alone, compatibility unknown"
            f" ({_reasons(capacity)}): {_shown(capacity)}"
        )
    if needs:
        restricted = ", ".join(
            f"`{row[0]['model_id']}` ({row[0]['variant']}: {_slugs(row[0])})" for row in needs[:_MAX_SHOWN_MODELS]
        )
        if len(needs) > _MAX_SHOWN_MODELS:
            restricted += f", and {len(needs) - _MAX_SHOWN_MODELS} more"
        lines.append(
            f"  {_counted(len(needs), 'variant')} {_verb(len(needs), 'fits', 'fit')} on capacity but"
            f" {_verb(len(needs), 'needs', 'need')} specific hardware: {restricted}"
        )
    return lines


def render_report(devices: list[Device], variants: list[dict[str, object]]) -> str:
    lines = [
        "vLLM-Omni doctor",
        "",
        "Devices:",
        *[line for device in devices for line in render_device(device, variants)],
        "",
        "A documented minimum is what a recipe measured or bounded, not a promise for",
        f"your workload. Anything below it is {_UNMEASURED_NOTE}.",
        "A fit also requires the recipe to declare your backend on dedicated memory and,",
        "where it lists hardware, to include your card; capacity alone is reported apart.",
        "Full index: docs/models/memory_tiers.md",
    ]
    return "\n".join(lines)


def report_json(devices: list[Device], variants: list[dict[str, object]]) -> dict[str, object]:
    """Machine-readable form: every capacity-fitting variant with its status."""
    reports = []
    for device in devices:
        reports.append(
            {
                "device": device.as_dict(),
                "tier_gib": device_tier_gib(device),
                "variants": [
                    {
                        "model_id": variant["model_id"],
                        "variant": variant["variant"],
                        "minimum_gib": variant["minimum_gib"],
                        "precision": variant["precision"],
                        "backends": variant.get("backends") or [],
                        "hardware": variant.get("hardware") or [],
                        "status": status,
                        "reason": reason,
                    }
                    for variant, status, reason in classify_variants(device, variants)
                ],
            }
        )
    return {"devices": [device.as_dict() for device in devices], "reports": reports}


class OmniDoctorSubcommand(CLISubcommand):
    """The `doctor` subcommand: report detected devices and documented tiers."""

    name = "doctor"
    help = "Report local devices and the deployment tiers documented for them."

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        devices = detect_devices()
        variants = load_documented_variants()
        if args.json:
            print(json.dumps(report_json(devices, variants), indent=2))
            return
        print(render_report(devices, variants))

    def validate(self, args: argparse.Namespace) -> None:
        pass

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> TrackingArgumentParser:
        parser = subparsers.add_parser(
            self.name,
            description=self.help,
            usage="vllm-omni doctor [--json]",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Emit devices and per-variant compatibility as JSON.",
        )
        return parser


def cmd_init() -> list[CLISubcommand]:
    return [OmniDoctorSubcommand()]
