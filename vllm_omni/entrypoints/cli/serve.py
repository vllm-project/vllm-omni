"""
Omni serve command for vLLM-Omni.

Supports both multi-stage LLM models (e.g., Qwen2.5-Omni) and
diffusion models (e.g., Qwen-Image) through the same CLI interface.
"""

import argparse

import uvloop
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.entrypoints.openai.api_server import omni_run_server

logger = init_logger(__name__)

DESCRIPTION = """Launch a local OpenAI-compatible API server to serve Omni models
via HTTP. Supports both multi-stage LLM models and diffusion models.

The server automatically detects the model type:
- LLM models: Served via /v1/chat/completions endpoint
- Diffusion models: Served via /v1/images/generations endpoint

Examples:
  # Start an Omni LLM server
  vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8091

  # Start a diffusion model server
  vllm serve Qwen/Qwen-Image --omni --port 8091

Search by using: `--help=<ConfigGroup>` to explore options by section (e.g.,
--help=OmniConfig)
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
        # Skip validation for diffusion models as they have different requirements
        from vllm_omni.diffusion.utils.hf_utils import is_diffusion_model

        model = getattr(args, "model_tag", None) or getattr(args, "model", None)
        if model and is_diffusion_model(model):
            logger.info("Detected diffusion model: %s", model)
            return
        validate_parsed_serve_args(args)

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        serve_parser = subparsers.add_parser(
            self.name,
            description=DESCRIPTION,
            usage="vllm serve [model_tag] --omni [options]",
        )

        serve_parser = make_arg_parser(serve_parser)
        serve_parser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(subcmd=self.name)

        # Create OmniConfig argument group for omni-related parameters
        # This ensures the parameters appear in --help output
        omni_config_group = serve_parser.add_argument_group(
            title="OmniConfig", description="Configuration for vLLM-Omni multi-stage and diffusion models."
        )

        omni_config_group.add_argument(
            "--omni",
            action="store_true",
            help="Enable vLLM-Omni mode for multi-modal and diffusion models",
        )
        omni_config_group.add_argument(
            "--stage-configs-path",
            type=str,
            default=None,
            help="Path to the stage configs file. If not specified, the stage configs will be loaded from the model.",
        )
        omni_config_group.add_argument(
            "--stage-init-timeout",
            type=int,
            default=300,
            help="The timeout for initializing a single stage in seconds (default: 300)",
        )
        omni_config_group.add_argument(
            "--init-timeout",
            type=int,
            default=60000,
            help="The timeout for initializing the stages.",
        )
        omni_config_group.add_argument(
            "--shm-threshold-bytes",
            type=int,
            default=65536,
            help="The threshold for the shared memory size.",
        )
        omni_config_group.add_argument(
            "--log-stats",
            action="store_true",
            help="Enable logging the stats.",
        )
        omni_config_group.add_argument(
            "--log-file",
            type=str,
            default=None,
            help="The path to the log file.",
        )
        omni_config_group.add_argument(
            "--batch-timeout",
            type=int,
            default=10,
            help="The timeout for the batch.",
        )
        omni_config_group.add_argument(
            "--worker-backend",
            type=str,
            default="multi_process",
            choices=["multi_process", "ray"],
            help="The backend to use for stage workers.",
        )
        omni_config_group.add_argument(
            "--ray-address",
            type=str,
            default=None,
            help="The address of the Ray cluster to connect to.",
        )

        # Diffusion model specific arguments
        omni_config_group.add_argument(
            "--num-gpus",
            type=int,
            default=None,
            help="Number of GPUs to use for diffusion model inference.",
        )
        omni_config_group.add_argument(
            "--usp",
            "--ulysses-degree",
            dest="ulysses_degree",
            type=int,
            default=None,
            help="Ulysses Sequence Parallelism degree for diffusion models. "
            "Equivalent to setting DiffusionParallelConfig.ulysses_degree.",
        )
        omni_config_group.add_argument(
            "--ring",
            dest="ring_degree",
            type=int,
            default=None,
            help="Ring Sequence Parallelism degree for diffusion models. "
            "Equivalent to setting DiffusionParallelConfig.ring_degree.",
        )

        # Cache optimization parameters
        omni_config_group.add_argument(
            "--cache-backend",
            type=str,
            default="none",
            help="Cache backend for diffusion models, options: 'tea_cache', 'cache_dit'",
        )
        omni_config_group.add_argument(
            "--cache-config",
            type=str,
            default=None,
            help="JSON string of cache configuration (e.g., '{\"rel_l1_thresh\": 0.2}').",
        )
        omni_config_group.add_argument(
            "--enable-cache-dit-summary",
            action="store_true",
            help="Enable cache-dit summary logging after diffusion forward passes.",
        )

        # VAE memory optimization parameters
        omni_config_group.add_argument(
            "--vae-use-slicing",
            action="store_true",
            help="Enable VAE slicing for memory optimization (useful for mitigating OOM issues).",
        )
        omni_config_group.add_argument(
            "--vae-use-tiling",
            action="store_true",
            help="Enable VAE tiling for memory optimization (useful for mitigating OOM issues).",
        )

        # diffusion model offload parameters
        serve_parser.add_argument(
            "--enable-cpu-offload",
            action="store_true",
            help="Enable CPU offloading for diffusion models.",
        )
        serve_parser.add_argument(
            "--enable-layerwise-offload",
            action="store_true",
            help="Enable layerwise (blockwise) offloading on DiT modules.",
        )
        serve_parser.add_argument(
            "--layerwise-num-gpu-layers",
            type=int,
            default=1,
            help="Number of layers (blocks) to keep on GPU during generation.",
        )

        # Video model parameters (e.g., Wan2.2) - engine-level
        omni_config_group.add_argument(
            "--boundary-ratio",
            type=float,
            default=None,
            help="Boundary split ratio for low/high DiT in video models (e.g., 0.875 for Wan2.2).",
        )
        omni_config_group.add_argument(
            "--flow-shift",
            type=float,
            default=None,
            help="Scheduler flow_shift for video models (e.g., 5.0 for 720p, 12.0 for 480p).",
        )
        omni_config_group.add_argument(
            "--cfg-parallel-size",
            type=int,
            default=1,
            choices=[1, 2],
            help="Number of devices for CFG parallel computation for diffusion models. "
            "Equivalent to setting DiffusionParallelConfig.cfg_parallel_size.",
        )
        return serve_parser


def cmd_init() -> list[CLISubcommand]:
    return [OmniServeCommand()]
