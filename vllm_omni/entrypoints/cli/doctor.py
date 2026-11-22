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
    """Variants whose documented minimum fits the device's memory."""
    tier_gib = device_tier_gib(device)
    if tier_gib is None:
        return []
    return [variant for variant in variants if int(variant["minimum_gib"]) <= tier_gib]


def _models(variants: list[dict[str, object]]) -> list[str]:
    return sorted({str(variant["model_id"]) for variant in variants})


def _counted(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def render_device(device: Device, variants: list[dict[str, object]]) -> list[str]:
    lines = [f"- {device.label}"]
    fitting = fitting_variants(device, variants)
    if device.total_mib is None:
        lines.append("  memory could not be read, so no documented tier was compared")
        return lines
    if not fitting:
        lines.append(f"  no recipe documents a minimum at or below {device_tier_gib(device)} GiB")
        return lines

    models = _models(fitting)
    shown = ", ".join(f"`{model}`" for model in models[:_MAX_SHOWN_MODELS])
    if len(models) > _MAX_SHOWN_MODELS:
        shown += f", and {len(models) - _MAX_SHOWN_MODELS} more"
    lines.append(
        f"  {_counted(len(fitting), 'documented variant')} across {_counted(len(models), 'model')} fit: {shown}"
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
        "Full index: docs/models/memory_tiers.md",
    ]
    return "\n".join(lines)


class OmniDoctorSubcommand(CLISubcommand):
    """The `doctor` subcommand: report detected devices and documented tiers."""

    name = "doctor"
    help = "Report local devices and the deployment tiers documented for them."

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        devices = detect_devices()
        variants = load_documented_variants()
        if args.json:
            print(
                json.dumps(
                    {
                        "devices": [device.as_dict() for device in devices],
                        "documented_fits": [
                            {
                                "device": device.as_dict(),
                                "variants": fitting_variants(device, variants),
                            }
                            for device in devices
                        ],
                    },
                    indent=2,
                )
            )
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
            help="Emit devices and documented fits as JSON.",
        )
        return parser


def cmd_init() -> list[CLISubcommand]:
    return [OmniDoctorSubcommand()]
