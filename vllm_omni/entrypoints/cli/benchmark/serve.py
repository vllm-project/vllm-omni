# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse

from vllm_omni.benchmarks.serve import add_cli_args, main
from vllm.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase


class OmniBenchmarkServingSubcommand(BenchmarkSubcommandBase):
    """ The `serve` subcommand for vllm bench. """
    name = "serve"
    help = "Benchmark the online serving throughput."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_cli_args(parser)
        parser.add_argument(
            "--omni",
            action="store_true",
            help="Enable vLLM-benchmark-Omni mode"
        )

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)
