# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Print the current machine's acceleration tier before serving a model."""

from __future__ import annotations

import argparse

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


class DoctorCommand(CLISubcommand):
    """The `doctor` subcommand for the vLLM CLI."""

    name = "doctor"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        del args
        print(render_doctor(detect_hardware()))

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
        return subparsers.add_parser(
            self.name,
            help="Print the current machine's acceleration tier and documented recipe minimums.",
        )


def cmd_init() -> list[CLISubcommand]:
    return [DoctorCommand()]
