# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Print the current machine's acceleration tier before serving a model."""

from __future__ import annotations

import argparse
import json

from vllm.entrypoints.cli.types import CLISubcommand

from vllm_omni.utils.hardware import HardwareInfo, detect_hardware, format_hardware

# Only documented minimums from the recipes; an entry here is not an
# end-to-end qualification for the tier. ``backend`` and ``memory_kind`` keep
# host memory or Apple unified memory from being reported as meeting an
# accelerator-specific recipe.
DOCUMENTED_RECIPES = (
    ("VoxCPM2", 24.0, "cuda", "device", "recipes/OpenBMB/VoxCPM2.md"),
    ("ERNIE-Image", 24.0, "cuda", "device", "recipes/Baidu/ERNIE-Image.md"),
)


def _matching_recipes(info: HardwareInfo) -> list[tuple[str, float, str, str, str]]:
    return [recipe for recipe in DOCUMENTED_RECIPES if recipe[2] == info.backend]


def _meets_documented_minimum(info: HardwareInfo) -> bool:
    if info.memory_gb is None:
        return False
    return any(info.memory_gb >= minimum_gb for _, minimum_gb, _, _, _ in _matching_recipes(info))


def render_doctor(info: HardwareInfo) -> str:
    lines = ["vLLM-Omni Doctor", format_hardware(info), "", "Documented recipes:"]
    for name, minimum_gb, backend, memory_kind, source in DOCUMENTED_RECIPES:
        lines.append(f"  {name}: >= {minimum_gb:.0f} GiB {backend.upper()} {memory_kind} memory ({source})")
    lines.append("")
    if info.memory_gb is None:
        lines.append("Your tier's memory is unknown; it is not yet measured. Do not treat this as unsupported.")
    elif not _matching_recipes(info):
        lines.append(
            f"No documented recipe targets {info.backend.upper()}; this tier is not yet measured. "
            "Do not treat this as unsupported."
        )
    elif _meets_documented_minimum(info):
        lines.append("Your tier meets the documented minimum; an end-to-end run is still required.")
    else:
        lines.append(
            "Your tier is below every documented recipe; it is not yet measured. Do not treat this as unsupported."
        )
    return "\n".join(lines)


def doctor_payload(info: HardwareInfo) -> dict[str, object]:
    """Machine-readable form of the doctor report."""
    return {
        "hardware": {
            "backend": info.backend,
            "device": info.device,
            "memory_gb": info.memory_gb,
            "os_name": info.os_name,
            "detail": info.detail,
        },
        "recipes": [
            {
                "name": name,
                "minimum_gb": minimum_gb,
                "backend": backend,
                "memory_kind": memory_kind,
                "source": source,
            }
            for name, minimum_gb, backend, memory_kind, source in DOCUMENTED_RECIPES
        ],
        "tier": "meets_documented_minimum" if _meets_documented_minimum(info) else "not_yet_measured",
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
