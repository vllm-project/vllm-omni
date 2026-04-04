"""
Omni serve command for vLLM-Omni.

Supports both multi-stage LLM models (e.g., Qwen2.5-Omni) and
diffusion models (e.g., Qwen-Image) through the same CLI interface.
"""

import argparse
import json
import os
from typing import Any

import uvloop
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.entrypoints.cli.logo import log_logo
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


def _ensure_vllm_platform():
    """Ensure vLLM's current_platform is valid before arg parsing.

    Upstream vLLM's argument parser now instantiates DeviceConfig during
    ``make_arg_parser``, which requires a resolved platform with a non-empty
    ``device_type``.  In some environments (e.g. editable installs with
    broken package metadata), vLLM's own platform auto-detection may fail
    and fall back to ``UnspecifiedPlatform``.  When that happens, use the
    Omni platform (which has its own detection logic) as a drop-in
    replacement so that argument parsing succeeds.
    """
    from vllm import platforms as vllm_platforms

    if vllm_platforms.current_platform.is_unspecified():
        from vllm_omni.platforms import current_omni_platform

        if not current_omni_platform.is_unspecified():
            vllm_platforms.current_platform = current_omni_platform
            logger.debug(
                "Replaced vLLM UnspecifiedPlatform with omni platform %s",
                type(current_omni_platform).__name__,
            )
        else:
            from vllm.platforms.cpu import CpuPlatform

            vllm_platforms.current_platform = CpuPlatform()
            logger.debug(
                "Both vLLM and omni platforms are unspecified, falling back to CpuPlatform for arg parsing",
            )


class OmniServeCommand(CLISubcommand):
    """The `serve` subcommand for the vLLM CLI."""

    name = "serve"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        if not os.environ.get("VLLM_DISABLE_LOG_LOGO"):
            os.environ["VLLM_DISABLE_LOG_LOGO"] = "1"
            log_logo()

        # If model is specified in CLI (as positional arg), it takes precedence
        if hasattr(args, "model_tag") and args.model_tag is not None:
            args.model = args.model_tag

        if args.headless:
            run_headless(args)
        else:
            uvloop.run(omni_run_server(args))

    def validate(self, args: argparse.Namespace) -> None:
        if args.stage_id is not None and (args.omni_master_address is None or args.omni_master_port is None):
            raise ValueError("--stage-id requires both --omni-master-address and --omni-master-port to be set")

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

        _ensure_vllm_platform()
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
            "--task-type",
            type=str,
            default=None,
            choices=["CustomVoice", "VoiceDesign", "Base"],
            help="Default task type for TTS models (CustomVoice, VoiceDesign, or Base). "
            "If not specified, will be inferred from model path.",
        )
        omni_config_group.add_argument(
            "--stage-configs-path",
            type=str,
            default=None,
            help="Path to the stage configs file. If not specified, the stage configs will be loaded from the model.",
        )
        omni_config_group.add_argument(
            "--stage-id",
            type=int,
            default=None,
            help="Select and launch a single stage by stage_id.",
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
            default=600,
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
        omni_config_group.add_argument(
            "--omni-master-address",
            "-oma",
            type=str,
            help="Hostname or IP address of the Omni orchestrator (master).",
        )
        omni_config_group.add_argument(
            "--omni-master-port",
            "-omp",
            type=int,
            help="Port of the Omni orchestrator (master).",
        )

        # Diffusion model specific arguments
        omni_config_group.add_argument(
            "--num-gpus",
            type=int,
            default=None,
            help="Number of GPUs to use for diffusion model inference.",
        )
        omni_config_group.add_argument(
            "--model-class-name",
            dest="model_class_name",
            type=str,
            default=None,
            help="Override the diffusion pipeline class name (e.g. LTX2ImageToVideoPipeline).",
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
            "--ulysses-mode",
            type=str,
            default="strict",
            choices=["strict", "advanced_uaa"],
            help="Ulysses sequence-parallel mode for diffusion models. "
            "'strict' keeps the original divisibility requirements; "
            "'advanced_uaa' enables the experimental UAA path for uneven sequence/head shapes.",
        )
        omni_config_group.add_argument(
            "--ring",
            "--ring-degree",
            dest="ring_degree",
            type=int,
            default=None,
            help="Ring Sequence Parallelism degree for diffusion models. "
            "Equivalent to setting DiffusionParallelConfig.ring_degree.",
        )
        omni_config_group.add_argument(
            "--quantization-config",
            type=json.loads,
            default=None,
            help=(
                "JSON string for diffusion quantization_config. "
                'Example: \'{"method":"gguf","gguf_model":"/path/to/model.gguf"}\'.'
            ),
        )

        # HSDP (Hybrid Sharded Data Parallel) parameters
        omni_config_group.add_argument(
            "--use-hsdp",
            dest="use_hsdp",
            action="store_true",
            help="Enable HSDP (Hybrid Sharded Data Parallel) for diffusion models. "
            "Shards model weights across GPUs to reduce per-GPU memory usage.",
        )
        omni_config_group.add_argument(
            "--hsdp-shard-size",
            type=int,
            default=-1,
            help="Number of GPUs to shard weights across. -1 = auto (world_size / replicate_size).",
        )
        omni_config_group.add_argument(
            "--hsdp-replicate-size",
            type=int,
            default=1,
            help="Number of replica groups for HSDP. Each group holds a full sharded copy.",
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
        omni_config_group.add_argument(
            "--step-execution",
            action="store_true",
            help="Enable per-step diffusion execution so running requests can be aborted between denoise steps.",
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

        # Parallel weight loading (faster diffusion startup)
        omni_config_group.add_argument(
            "--disable-multithread-weight-load",
            action="store_false",
            dest="enable_multithread_weight_load",
            default=True,
            help="Disable multi-threaded safetensors loading (default: enabled with 4 threads).",
        )
        omni_config_group.add_argument(
            "--num-weight-load-threads",
            type=int,
            default=4,
            help="Number of threads for parallel weight loading (default: 4).",
        )

        # diffusion model offload parameters
        omni_config_group.add_argument(
            "--enable-cpu-offload",
            action="store_true",
            help="Enable CPU offloading for diffusion models.",
        )
        omni_config_group.add_argument(
            "--enable-layerwise-offload",
            action="store_true",
            help="Enable layerwise (blockwise) offloading on DiT modules.",
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
        omni_config_group.add_argument(
            "--vae-patch-parallel-size",
            type=int,
            default=1,
            help="VAE Patch Parallelism degree for diffusion models. "
            "Distributes VAE decode workload across multiple ranks by splitting the latent spatially. "
            "Equivalent to setting DiffusionParallelConfig.vae_patch_parallel_size.",
        )

        # Default sampling parameters
        omni_config_group.add_argument(
            "--default-sampling-params",
            type=str,
            help="Json str for Default sampling parameters, \n"
            'Structure: {"<stage_id>": {<sampling_param>: value, ...}, ...}\n'
            'e.g., \'{"0": {"num_inference_steps":50, "guidance_scale":1}}\'. '
            "Currently only supports diffusion models.",
        )
        # Diffusion model mixed precision
        omni_config_group.add_argument(
            "--max-generated-image-size",
            type=int,
            help="The max size of generate image (height * width).",
        )

        # TTS-specific parameters
        omni_config_group.add_argument(
            "--tts-max-instructions-length",
            type=int,
            default=None,
            help="Maximum length for TTS voice style instructions (overrides stage config, default: 500).",
        )

        # Enable diffusion pipeline profiling
        omni_config_group.add_argument(
            "--enable-diffusion-pipeline-profiler",
            action="store_true",
            help="Enable diffusion pipeline profiler to display stage durations.",
        )
        return serve_parser


def _create_default_diffusion_stage_cfg(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Create default diffusion stage configuration.

    Uses AsyncOmniEngine's implementation which doesn't have OmegaConf
    compatibility issues.
    """
    from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

    return AsyncOmniEngine._create_default_diffusion_stage_cfg(vars(args))


def run_headless(args: argparse.Namespace) -> None:
    """Run a single stage in headless mode.

    .. deprecated:: 0.x.x
        Headless mode is deprecated and will be removed in a future version.
        It is only compatible with the old OmniStage-based runtime.
        The current AsyncOmniEngine-based runtime does not support headless mode.

    Raises:
        RuntimeError: Always raises an error indicating headless mode is deprecated.
    """
    raise RuntimeError(
        "Headless mode is deprecated and not supported in the current runtime. "
        "Please use the standard orchestrator mode (without --headless flag). "
        "If you need distributed deployment, consider using Ray backend or "
        "other distributed serving solutions."
    )


def cmd_init() -> list[CLISubcommand]:
    return [OmniServeCommand()]
