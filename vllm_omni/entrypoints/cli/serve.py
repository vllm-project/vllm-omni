"""
Omni serve command for vLLM-Omni.

Supports both multi-stage LLM models (e.g., Qwen2.5-Omni) and
diffusion models (e.g., Qwen-Image) through the same CLI interface.
"""

import argparse
import json
import os
import signal
from types import FrameType
from typing import Any

import uvloop
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.engine.arg_utils import nullify_stage_engine_defaults
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
    # Parser stashed at subparser_init so ``cmd`` can resolve each user-typed
    # flag to its real ``dest`` via the parser's action table.
    _parser: FlexibleArgumentParser | None = None

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

        try:
            omni_config_group.add_argument(
                "--enable-sleep-mode",
                action="store_true",
                default=False,
                help="Enable GPU memory pool for sleep mode.",
            )
        except argparse.ArgumentError:
            pass

        omni_config_group.add_argument(
            "--task-type",
            type=str,
            default=None,
            choices=["CustomVoice", "VoiceDesign", "Base"],
            help="Default task type for TTS models (CustomVoice, VoiceDesign, or Base). "
            "If not specified, will be inferred from model path.",
        )
        # TODO(@lishunyang12): deprecate once all models migrate to --deploy-config
        omni_config_group.add_argument(
            "--stage-configs-path",
            type=str,
            default=None,
            help="[Deprecated — will be removed in a future release] Path to a legacy "
            "stage configs YAML (stage_args format). Prefer --deploy-config for new-format deploy YAMLs.",
        )
        omni_config_group.add_argument(
            "--deploy-config",
            type=str,
            default=None,
            help="Path to a deploy config YAML (new format with stages/engine_args). "
            "Mutually exclusive with --stage-configs-path.",
        )
        omni_config_group.add_argument(
            "--stage-overrides",
            type=str,
            default=None,
            help="Per-stage JSON overrides. Example: "
            '\'{"0": {"gpu_memory_utilization": 0.8}, "2": {"enforce_eager": true}}\'',
        )
        omni_config_group.add_argument(
            "--async-chunk",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Override the deploy YAML's ``async_chunk:`` bool. Unset leaves the YAML value in force.",
        )
        omni_config_group.add_argument(
            "--stage-id",
            type=int,
            default=None,
            help="Select and launch a single stage by stage_id.",
        )
        omni_config_group.add_argument(
            "--replica-id",
            type=int,
            default=0,
            help="Replica id to register when launching a single headless stage.",
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
            "--diffusion-load-format",
            dest="diffusion_load_format",
            type=str,
            default=None,
            choices=["default", "custom_pipeline", "dummy", "diffusers"],
            help=(
                "How to load the diffusion pipeline: native/registry (default), "
                "custom_pipeline, dummy, or diffusers for the HF diffusers adapter."
            ),
        )
        omni_config_group.add_argument(
            "--diffusers-load-kwargs",
            dest="diffusers_load_kwargs",
            type=json.loads,
            default="{}",
            help=(
                "JSON object passed to DiffusionPipeline.from_pretrained()."
                "It overrides corresponding parameters in the standard vLLM-Omni interface."
                '(e.g. \'{"use_safetensors": true, "variant": "fp16"}\').'
            ),
        )
        omni_config_group.add_argument(
            "--diffusers-call-kwargs",
            dest="diffusers_call_kwargs",
            type=json.loads,
            default="{}",
            help=(
                "JSON object passed to pipeline.__call__(). "
                "Useful for model-specific sampling parameters not covered by the vLLM-Omni interface."
                "During request time, it is overridden by corresponding parameters in the vLLM-Omni interface."
                '(e.g. \'{"num_inference_steps": 30, "guidance_scale": 7.5}\').'
            ),
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
        omni_config_group.add_argument(
            "--force-cutlass-fp8",
            action="store_true",
            default=None,
            help=(
                "Diffusion-only runtime override for ModelOpt FP8 checkpoints: "
                "force CUTLASS FP8 linear kernels on CUDA SM89+ devices. "
                "Ignored for BF16, non-ModelOpt FP8, ROCm, and older CUDA GPUs."
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

        # Attention backend configuration
        omni_config_group.add_argument(
            "--diffusion-attention-backend",
            dest="diffusion_attention_backend",
            type=str,
            default=None,
            help="Diffusion attention backend (shorthand). "
            "Sets the default backend for all diffusion attention roles, e.g. 'FLASH_ATTN'. "
            "May be combined with --diffusion-attention-config.per_role.* overrides, "
            "but mutually exclusive with --diffusion-attention-config.default.backend.",
        )
        omni_config_group.add_argument(
            "--diffusion-attention-config",
            "-dac",
            dest="diffusion_attention_config",
            type=json.loads,
            default=None,
            help="Diffusion attention config. Accepts JSON or vLLM-style dotted flags. "
            "Examples: "
            "--diffusion-attention-config.default.backend FLASH_ATTN, "
            "--diffusion-attention-config.per_role.self.backend SPARSE_BLOCK, "
            "--diffusion-attention-config.per_role.cross.backend SAGE_ATTN, "
            '--diffusion-attention-config \'{"default": {"backend": "FLASH_ATTN"}, '
            '"per_role": {"cross": {"backend": "SAGE_ATTN"}}}\'.',
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
        omni_config_group.add_argument(
            "--enable-ar-profiler",
            action="store_true",
            help="Enable AR stage profiler to include AR stage timing in stage_durations.",
        )
        # Stash via type(self) so the docs hook (which execs this function in a
        # sandboxed globals dict via ``DummySelf``) doesn't fail on a NameError.
        type(self)._parser = serve_parser

        nullify_stage_engine_defaults(serve_parser)
        return serve_parser


def _create_default_diffusion_stage_cfg(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Create default diffusion stage configuration.

    Uses AsyncOmniEngine's implementation which doesn't have OmegaConf
    compatibility issues.
    """
    from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

    return AsyncOmniEngine._create_default_diffusion_stage_cfg(vars(args))


def run_headless(args: argparse.Namespace) -> None:
    """Run a single stage in headless mode."""
    from vllm.v1.engine.coordinator import DPCoordinator
    from vllm.v1.engine.utils import CoreEngineProcManager
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor
    from vllm.version import __version__ as VLLM_VERSION

    from vllm_omni.diffusion.stage_diffusion_proc import (
        complete_diffusion_handshake,
        spawn_diffusion_proc,
    )
    from vllm_omni.distributed.omni_connectors.utils.initialization import resolve_omni_kv_config_for_stage
    from vllm_omni.engine.stage_engine_startup import register_stage_with_omni_master
    from vllm_omni.engine.stage_init_utils import (
        build_diffusion_config,
        build_engine_args_dict,
        build_vllm_config,
        extract_stage_metadata,
        get_stage_connector_spec,
        inject_kv_stage_info,
        load_omni_transfer_config_for_model,
        prepare_engine_environment,
        terminate_alive_proc,
    )
    from vllm_omni.entrypoints.utils import inject_omni_kv_config, load_and_resolve_stage_configs

    model = args.model
    stage_id: int | None = args.stage_id
    replica_id: int = args.replica_id
    omni_master_address: str | None = args.omni_master_address
    omni_master_port: int | None = args.omni_master_port

    if stage_id is None:
        raise ValueError("--stage-id is required in headless mode")
    if replica_id < 0:
        raise ValueError("--replica-id must be >= 0 in headless mode")
    if omni_master_address is None or omni_master_port is None:
        raise ValueError("--omni-master-address and --omni-master-port are required in headless mode")
    api_server_count = args.api_server_count or 0
    if api_server_count > 1:
        raise ValueError("api_server_count can't be set in headless mode")
    if args.worker_backend != "multi_process":
        raise ValueError("headless mode requires worker_backend=multi_process")

    args_dict = vars(args).copy()
    args_dict.pop("_cli_explicit_keys", None)
    config_path, stage_configs = load_and_resolve_stage_configs(
        model,
        args_dict.get("stage_configs_path"),
        args_dict,
        deploy_config_path=args_dict.get("deploy_config"),
    )

    # Locate the stage config that matches stage_id.
    stage_cfg = None
    for cfg in stage_configs:
        if cfg.stage_id == stage_id:
            stage_cfg = cfg
            break
    if stage_cfg is None:
        raise ValueError(
            f"No stage config found for stage_id={stage_id}. Available stage ids: {[c.stage_id for c in stage_configs]}"
        )

    prepare_engine_environment()
    omni_transfer_config = load_omni_transfer_config_for_model(model, config_path)
    omni_conn_cfg, omni_from, omni_to = resolve_omni_kv_config_for_stage(omni_transfer_config, stage_id)

    if stage_cfg.stage_type == "diffusion":
        metadata = extract_stage_metadata(stage_cfg)
        if omni_conn_cfg:
            inject_omni_kv_config(stage_cfg, omni_conn_cfg, omni_from, omni_to)
        inject_kv_stage_info(stage_cfg, stage_id)
        od_config = build_diffusion_config(model, stage_cfg, metadata)

        logger.info(
            "[Headless] Launching diffusion stage %d replica %d via OmniMasterServer at %s:%d",
            stage_id,
            replica_id,
            omni_master_address,
            omni_master_port,
        )

        proc = None
        try:
            handshake_address, request_address, response_address = register_stage_with_omni_master(
                omni_master_address=omni_master_address,
                omni_master_port=omni_master_port,
                omni_stage_id=stage_id,
                omni_stage_config=stage_cfg,
                return_addresses=True,
                replica_id=replica_id,
            )
            proc, _, _, _ = spawn_diffusion_proc(
                model,
                od_config,
                handshake_address=handshake_address,
                request_address=request_address,
                response_address=response_address,
            )
            complete_diffusion_handshake(proc, handshake_address, args.stage_init_timeout)
            proc.join()
            if proc.exitcode not in (None, 0):
                raise RuntimeError(f"Diffusion stage {stage_id} replica {replica_id} exited with code {proc.exitcode}")
            return
        finally:
            logger.info("[Headless] Shutting down stage %d replica %d.", stage_id, replica_id)
            if proc is not None and proc.is_alive():
                terminate_alive_proc(proc)

    stage_connector_spec = get_stage_connector_spec(
        omni_transfer_config=omni_transfer_config,
        stage_id=stage_id,
        async_chunk=False,
    )

    # Device assignment is managed externally (e.g. CUDA_VISIBLE_DEVICES);
    # runtime_cfg is intentionally ignored in headless mode.
    engine_args_dict = build_engine_args_dict(
        stage_cfg,
        model,
        stage_connector_spec=stage_connector_spec,
        cli_tokenizer=getattr(args, "tokenizer", None),
    )

    # Inject omni KV connector config so the engine runner can initialize the
    # correct connector (sender/receiver role, type, addresses, etc.).
    if omni_conn_cfg:
        omni_kv = engine_args_dict.get("omni_kv_config") or {}
        if not isinstance(omni_kv, dict):
            omni_kv = dict(omni_kv)
        omni_kv["connector_config"] = omni_conn_cfg
        omni_kv["omni_from_stage"] = omni_from
        omni_kv["omni_to_stage"] = omni_to
        omni_kv.setdefault("stage_id", stage_id)
        engine_args_dict["omni_kv_config"] = omni_kv

    vllm_config, executor_class = build_vllm_config(
        stage_cfg,
        model,
        stage_connector_spec=stage_connector_spec,
        engine_args_dict=engine_args_dict,
        headless=True,
    )
    parallel_config = vllm_config.parallel_config
    local_engine_count = parallel_config.data_parallel_size_local

    if local_engine_count <= 0:
        raise ValueError("data_parallel_size_local must be > 0 in headless mode")

    shutdown_requested = False

    def signal_handler(signum: int, frame: FrameType | None) -> None:
        nonlocal shutdown_requested
        logger.debug("Received %d signal.", signum)
        if not shutdown_requested:
            shutdown_requested = True
            raise SystemExit

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    if parallel_config.node_rank_within_dp > 0:
        head_node_address = f"{parallel_config.master_addr}:{parallel_config.master_port}"
        logger.info(
            "Launching vLLM-Omni (v%s) headless multiproc executor, "
            "with head node address %s for torch.distributed process group.",
            VLLM_VERSION,
            head_node_address,
        )

        executor = MultiprocExecutor(vllm_config, monitor_workers=False)
        executor.start_worker_monitor(inline=True)
        return

    dp_rank = parallel_config.data_parallel_rank if parallel_config.data_parallel_rank is not None else 0
    coordinator = None
    if vllm_config.needs_dp_coordinator and dp_rank == 0:
        coordinator = DPCoordinator(
            parallel_config,
            enable_wave_coordination=vllm_config.model_config.is_moe,
        )
        logger.info(
            "[Headless] Started DP Coordinator process for stage %d replica %d (PID: %d)",
            stage_id,
            replica_id,
            coordinator.proc.pid,
        )

    logger.info(
        "[Headless] Launching %d engine core(s) for stage %d replica %d via OmniMasterServer at %s:%d",
        local_engine_count,
        stage_id,
        replica_id,
        omni_master_address,
        omni_master_port,
    )

    # Headless mode launches all local engine cores for a single stage.
    # The OmniMasterServer allocates one handshake endpoint per stage, so we
    # register the stage once here and let every local engine core reuse the
    # returned handshake address directly.
    handshake_address = register_stage_with_omni_master(
        omni_master_address=omni_master_address,
        omni_master_port=omni_master_port,
        omni_stage_id=stage_id,
        omni_stage_config=stage_cfg,
        coordinator=coordinator,
        replica_id=replica_id,
    )

    engine_manager = None
    log_stats = bool(args.log_stats)
    if args.disable_log_stats:
        log_stats = False

    try:
        engine_manager = CoreEngineProcManager(
            local_engine_count=local_engine_count,
            start_index=dp_rank,
            local_start_index=0,
            vllm_config=vllm_config,
            local_client=False,
            handshake_address=handshake_address,
            executor_class=executor_class,
            log_stats=log_stats,
        )
        # vllm>=0.19 renamed CoreEngineProcManager.join_first() to
        # monitor_engine_liveness() (see upstream PR #35862).
        engine_manager.monitor_engine_liveness()
    finally:
        logger.info("[Headless] Shutting down stage %d.", stage_id)
        if engine_manager is not None:
            engine_manager.shutdown()
        if coordinator is not None:
            coordinator.shutdown()


def cmd_init() -> list[CLISubcommand]:
    return [OmniServeCommand()]
