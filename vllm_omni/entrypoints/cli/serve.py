"""
Omni serve command for vLLM-Omni.
"""

import argparse

import uvloop
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG
from vllm.logger import init_logger
from vllm.utils import FlexibleArgumentParser

from vllm_omni.entrypoints.openai.api_server import omni_run_server

logger = init_logger(__name__)

DESCRIPTION = """Launch a local OpenAI-compatible API server to serve Omni LLM
completions via HTTP. Defaults to Qwen/Qwen2.5-Omni-7B if no model is specified.

Search by using: `--help=<ConfigGroup>` to explore options by section (e.g.,
--help=ModelConfig, --help=Frontend)
  Use `--help=all` to show all available flags at once.
"""


class OmniServeCommand(CLISubcommand):
    """The `serve` subcommand for the vLLM CLI."""

    name = "serve"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        # If model is specified in CLI (as positional arg), it takes precedence
        if hasattr(args, "model_tag") and args.model_tag is not None:
            args.model = args.model_tag

        uvloop.run(omni_run_server(args))

    def validate(self, args: argparse.Namespace) -> None:
        validate_parsed_serve_args(args)

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        serve_parser = subparsers.add_parser(
            self.name,
            description=DESCRIPTION,
            usage="vllm serve [model_tag] --omni [options]",
        )

        serve_parser = make_arg_parser(serve_parser)
        serve_parser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(subcmd=self.name)
        serve_parser.add_argument(
            "--omni",
            action="store_true",
            help="Enable vLLM-Omni mode",
        )
        serve_parser.add_argument(
            "--stage-configs-path",
            type=str,
            default=None,
            help="Path to the stage configs file. If not specified, the stage configs will be loaded from the model.",
        )
        serve_parser.add_argument(
            "--init-sleep-seconds",
            type=int,
            default=30,
            help="The number of seconds to sleep before initializing the stages.",
        )
        serve_parser.add_argument(
            "--init-timeout",
            type=int,
            default=60000,
            help="The timeout for initializing the stages.",
        )
        serve_parser.add_argument(
            "--shm-threshold-bytes",
            type=int,
            default=65536,
            help="The threshold for the shared memory size.",
        )
        serve_parser.add_argument(
            "--log-stats",
            action="store_true",
            help="Enable logging the stats.",
        )
        serve_parser.add_argument(
            "--log-file",
            type=str,
            default=None,
            help="The path to the log file.",
        )
        serve_parser.add_argument(
            "--batch-timeout",
            type=int,
            default=10,
            help="The timeout for the batch.",
        )
        serve_parser.add_argument(
            "--worker-backend",
            type=str,
            default="multi_process",
            choices=["multi_process", "ray"],
            help="The backend to use for stage workers.",
        )
        serve_parser.add_argument(
            "--ray-address",
            type=str,
            default=None,
            help="The address of the Ray cluster to connect to.",
        )
        return serve_parser


def cmd_init() -> list[CLISubcommand]:
    return [OmniServeCommand()]
