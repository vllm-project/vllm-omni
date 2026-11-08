# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Print the current machine's acceleration tier before serving a model."""

from __future__ import annotations

import argparse
import json

from vllm.entrypoints.cli.types import CLISubcommand

from vllm_omni.utils.hardware import HardwareInfo, detect_hardware, format_hardware

# Only documented minimums from the recipes; an entry here is not an
# end-to-end qualification for the tier.
DOCUMENTED_RECIPES = (
    ("VoxCPM2", 24.0, "recipes/OpenBMB/VoxCPM2.md"),
    ("ERNIE-Image", 24.0, "recipes/Baidu/ERNIE-Image.md"),
)


def render_doctor(info: HardwareInfo) -> str:
    lines = ["vLLM-Omni Doctor", format_hardware(info), "", "Documented recipes:"]
    for name, minimum_gb, source in DOCUMENTED_RECIPES:
        lines.append(f"  {name}: >= {minimum_gb:.0f} GiB ({source})")
    lines.append("")
    if info.memory_gb is not None and info.memory_gb >= min(minimum_gb for _, minimum_gb, _ in DOCUMENTED_RECIPES):
        lines.append("Your tier meets the documented minimum; an end-to-end run is still required.")
    else:
        lines.append(
            "Your tier is below every documented recipe; it is not yet measured. Do not treat this as unsupported."
        )
    return "\n".join(lines)


def doctor_payload(info: HardwareInfo) -> dict[str, object]:
    """Machine-readable form of the doctor report."""
    minimum = min(minimum_gb for _, minimum_gb, _ in DOCUMENTED_RECIPES)
    meets_minimum = info.memory_gb is not None and info.memory_gb >= minimum
    return {
        "hardware": {
            "backend": info.backend,
            "device": info.device,
            "memory_gb": info.memory_gb,
            "os_name": info.os_name,
            "detail": info.detail,
        },
        "recipes": [
            {"name": name, "minimum_gb": minimum_gb, "source": source}
            for name, minimum_gb, source in DOCUMENTED_RECIPES
        ],
        "tier": "meets_documented_minimum" if meets_minimum else "not_yet_measured",
    }


def render_doctor_json(info: HardwareInfo) -> str:
    return json.dumps(doctor_payload(info), indent=2, sort_keys=True)


class DoctorCommand(CLISubcommand):
    """The `doctor` subcommand for the vLLM CLI."""

    name = "doctor"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        info = detect_hardware()
        print(render_doctor_json(info) if args.json else render_doctor(info))

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
        parser = subparsers.add_parser(
            self.name,
            help="Print the current machine's acceleration tier and documented recipe minimums.",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Emit the report as JSON.",
        )
        return parser


def cmd_init() -> list[CLISubcommand]:
    return [DoctorCommand()]
