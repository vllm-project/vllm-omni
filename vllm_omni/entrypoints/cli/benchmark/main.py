from __future__ import annotations

import argparse
import typing

from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG

from vllm_omni.entrypoints.cli.benchmark.base import OmniBenchmarkSubcommandBase

if typing.TYPE_CHECKING:
    from vllm.utils import FlexibleArgumentParser


class OmniBenchmarkSubcommand(CLISubcommand):
    """The `bench` subcommand for the vLLM CLI."""

    name = "bench"
    help = "vLLM-omni bench subcommand."

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        args.dispatch_function(args)

    def validate(self, args: argparse.Namespace) -> None:
        pass

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
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
