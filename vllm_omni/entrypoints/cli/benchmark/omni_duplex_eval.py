# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import argparse

from vllm_omni.benchmarks.duplex.omni_duplex_eval_cli import add_cli_args, run
from vllm_omni.entrypoints.cli.benchmark.base import OmniBenchmarkSubcommandBase


class OmniDuplexEvalSubcommand(OmniBenchmarkSubcommandBase):
    """Run the Omni-DuplexEval generation or scoring workflow."""

    name = "omni-duplex-eval"
    help = "Generate, evaluate, or summarize Omni-DuplexEval artifacts."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        run(args)
