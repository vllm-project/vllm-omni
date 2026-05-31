from __future__ import annotations

import argparse
import sys
import typing

from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG

from vllm_omni.entrypoints.cli.benchmark.base import OmniBenchmarkSubcommandBase

if typing.TYPE_CHECKING:
    from vllm.utils.argparse_utils import FlexibleArgumentParser


class OmniBenchmarkSubcommand(CLISubcommand):
    """The `bench` subcommand for the vLLM CLI."""

    name = "bench"
    help = "vLLM-omni bench subcommand."

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        args.dispatch_function(args)

    def validate(self, args: argparse.Namespace) -> None:
        pass

    @staticmethod
    def _load_requested_benchmark_subcommands() -> None:
        is_benchmark_invocation = False
        for arg in sys.argv[1:]:
            if arg.startswith("-"):
                continue
            is_benchmark_invocation = arg == "bench"
            break
        if not is_benchmark_invocation:
            return

        # Registers OmniBenchmarkServingSubcommand via __subclasses__(). This is
        # intentionally lazy: importing it loads vllm.benchmarks and the Omni
        # benchmark monkey patches, which can probe CUDA/NVML.
        import vllm_omni.entrypoints.cli.benchmark.serve  # noqa: F401

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        self._load_requested_benchmark_subcommands()

        bench_parser = subparsers.add_parser(
            self.name, description=self.help, usage=f"vllm {self.name} <bench_type> [options]"
        )
        bench_subparsers = bench_parser.add_subparsers(required=True, dest="bench_type")

        for cmd_cls in OmniBenchmarkSubcommandBase.__subclasses__():
            cmd_subparser = bench_subparsers.add_parser(
                cmd_cls.name,
                help=cmd_cls.help,
                description=cmd_cls.help,
                usage=f"vllm {self.name} {cmd_cls.name} [--omni] [options]",
            )
            cmd_subparser.add_argument(
                "--omni",
                action="store_true",
                help="Enable benchmark-Omni mode (always enabled for omni commands)",
            )
            cmd_subparser.set_defaults(dispatch_function=cmd_cls.cmd)
            cmd_cls.add_cli_args(cmd_subparser)

            cmd_subparser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(subcmd=f"{self.name} {cmd_cls.name}")

        return bench_parser


def cmd_init() -> list[CLISubcommand]:
    return [OmniBenchmarkSubcommand()]
