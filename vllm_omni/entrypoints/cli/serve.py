"""
Omni serve command for vLLM-Omni.

Supports both multi-stage LLM models (e.g., Qwen2.5-Omni) and
diffusion models (e.g., Qwen-Image) through the same CLI interface.
"""

import argparse
import json
import os
import signal
import threading
from multiprocessing import connection
from types import FrameType
from typing import Any

import uvloop
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.serve.utils.api_utils import VLLM_SUBCMD_PARSER_EPILOG
from vllm.logger import init_logger

from vllm_omni.entrypoints.cli.logo import log_logo
from vllm_omni.entrypoints.openai.api_server import omni_run_server
from vllm_omni.utils.tracking_parser import TrackingArgumentParser, TrackingNamespace

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
    _parser: TrackingArgumentParser

    @staticmethod
    def cmd(args: TrackingNamespace) -> None:
        if not os.environ.get("VLLM_DISABLE_LOG_LOGO"):
            os.environ["VLLM_DISABLE_LOG_LOGO"] = "1"
            log_logo()

        # If model is specified in CLI (as positional arg), it takes precedence
        if hasattr(args, "model_tag") and args.model_tag is not None:
            args.model = args.model_tag

        if getattr(args, "no_guardrails", False):
            existing = getattr(args, "model_config", None)
            model_config = dict(existing) if isinstance(existing, dict) else {}
            model_config["guardrails"] = False
            args.model_config = model_config
            explicit_keys = getattr(args, "explicit_keys", None)
            if explicit_keys is not None:
                args.explicit_keys = explicit_keys | {"model_config"}

        if args.headless:
            run_headless(args)
        else:
            uvloop.run(omni_run_server(args))

    def validate(self, args: argparse.Namespace) -> None:
        if args.stage_id is not None and (args.omni_master_address is None or args.omni_master_port is None):
            raise ValueError("--stage-id requires both --omni-master-address and --omni-master-port to be set")

        # --omni-replica-address is only consulted in run_headless(); reject it
        # on the head so a misconfigured launch fails loudly instead of being
        # silently ignored.
        if getattr(args, "omni_replica_address", None) is not None and not args.headless:
            raise ValueError("--omni-replica-address requires --headless to be set")

        # --omni-dp-size-local is process-local. A value other than 1 only
        # makes sense when this process owns a stage (head or headless).
        omni_dp_size_local = getattr(args, "omni_dp_size_local", None)
        if omni_dp_size_local is not None:
            if omni_dp_size_local < 1:
                raise ValueError(f"--omni-dp-size-local must be >= 1, got {omni_dp_size_local}")
            if omni_dp_size_local != 1 and args.stage_id is None:
                raise ValueError("--omni-dp-size-local != 1 requires --stage-id to be set")

        # vLLM CLI args that omni does not honor: parallelism comes from the
        # per-stage YAML (parallel_config:, enable_expert_parallel:) and the
        # process-local replica count from --omni-dp-size-local. Passing the
        # vLLM equivalents on the command line would silently disagree with
        # those sources of truth, so reject them at parse time.
        if getattr(args, "omni", False):
            explicit_cli_keys: set[str] = getattr(args, "_cli_explicit_keys", set()) or set()
            prohibited_with_omni: dict[str, str] = {
                "data_parallel_size": "--data-parallel-size",
                "data_parallel_size_local": "--data-parallel-size-local",
                "data_parallel_address": "--data-parallel-address",
                "data_parallel_rpc_port": "--data-parallel-rpc-port",
                "data_parallel_start_rank": "--data-parallel-start-rank",
                "data_parallel_backend": "--data-parallel-backend",
                "api_server_count": "--api-server-count",
                "enable_expert_parallel": "--enable-expert-parallel",
            }
            offenders = sorted(flag for dest, flag in prohibited_with_omni.items() if dest in explicit_cli_keys)
            if offenders:
                raise ValueError(
                    "The following CLI args are not supported under --omni: "
                    f"{', '.join(offenders)}. Configure parallelism through the "
                    "per-stage YAML (`--deploy-config` / `--stage-configs-path`) "
                    "and replica count via `--omni-dp-size-local`."
                )

        # --omni-lb-policy is validated against the LoadBalancingPolicy enum.
        omni_lb_policy = getattr(args, "omni_lb_policy", None)
        if omni_lb_policy is not None:
            from vllm_omni.distributed.omni_coordinator import LoadBalancingPolicy

            try:
                LoadBalancingPolicy(omni_lb_policy)
            except ValueError as exc:
                valid = ", ".join(p.value for p in LoadBalancingPolicy)
                raise ValueError(f"--omni-lb-policy={omni_lb_policy!r} is not one of: {valid}") from exc

        omni_heartbeat_timeout = getattr(args, "omni_heartbeat_timeout", None)
        if omni_heartbeat_timeout is not None and omni_heartbeat_timeout <= 0:
            raise ValueError(f"--omni-heartbeat-timeout must be > 0, got {omni_heartbeat_timeout}")

        # Skip validation for diffusion models as they have different requirements
        from vllm_omni.diffusion.utils.hf_utils import is_diffusion_model

        model = getattr(args, "model_tag", None) or getattr(args, "model", None)
        if model and is_diffusion_model(model):
            logger.info("Detected diffusion model: %s", model)
            return
        validate_parsed_serve_args(args)

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> TrackingArgumentParser:
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
            default=None,
            help=(
                "Deprecated and ignored — replica ids are auto-assigned by the "
                "master server. Specifying this flag prints a warning and has "
                "no effect."
            ),
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
        omni_config_group.add_argument(
            "--omni-replica-address",
            "-ora",
            type=str,
            default=None,
            help=(
                "Local bind address (this host's IP) that the headless stage "
                "advertises to the Omni master for its handshake/input/output "
                "ZMQ sockets. If unset, auto-detected via a UDP-connect "
                "routing probe against --omni-master-address. Override only "
                "when the auto-detected IP is wrong (e.g. multi-NIC host "
                "where the master is reachable on the wrong interface)."
            ),
        )
        omni_config_group.add_argument(
            "--omni-dp-size-local",
            type=int,
            default=1,
            help=(
                "Number of stage replicas this runtime launches locally for its "
                "own --stage-id. Process-local: head and every headless invocation "
                "read their own copy; values may differ across invocations. "
                "Requires --stage-id to be set when not equal to 1."
            ),
        )
        omni_config_group.add_argument(
            "--omni-lb-policy",
            type=str,
            default="random",
            choices=["random", "round-robin", "least-queue-length"],
            help=(
                "Per-stage load-balancing policy used by the head's StagePool to "
                "route requests across UP replicas. Only consulted on the head runtime."
            ),
        )
        omni_config_group.add_argument(
            "--omni-heartbeat-timeout",
            type=float,
            default=30.0,
            help=(
                "Seconds before an unreporting replica is marked ERROR in the "
                "OmniCoordinator. Only consulted on the head runtime."
            ),
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
            "--diffusion-quantization-config",
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
            help="Cache backend for diffusion models, options: 'tea_cache', 'cache_dit', 'mag_cache'",
        )
        omni_config_group.add_argument(
            "--cache-config",
            type=str,
            default=None,
            help="JSON string of cache configuration. "
            "TeaCache: '{\"rel_l1_thresh\": 0.2}'. "
            'MagCache: \'{"mag_threshold": 0.24, "mag_max_skip_steps": 5, "mag_retention_ratio": 0.1}\'. '
            "Calibration mode: add '\"mag_calibrate\": true'",
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
        # Diffusion KV-cache quantization uses dedicated flags so we do not reuse
        # vLLM's --kv-cache-dtype (AR cache dtype, default "auto").
        omni_config_group.add_argument(
            "--diffusion-kv-cache-dtype",
            type=str,
            default=None,
            help="Diffusion attention KV cache dtype (e.g. fp8). Separate from vLLM --kv-cache-dtype.",
        )
        omni_config_group.add_argument(
            "--diffusion-kv-cache-skip-steps",
            type=str,
            default=None,
            help="Diffusion KV-cache quantization skip-step selector, e.g. '0-9,20,25-30'.",
        )
        omni_config_group.add_argument(
            "--diffusion-kv-cache-skip-layers",
            type=str,
            default=None,
            help="Diffusion KV-cache quantization skip-layer selector, e.g. '0,1,4-8'.",
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

        # Disable safety guardrails for this server (currently only applicable for Cosmos3)
        # TODO: drop once --model-config-override lands (3/N config refactor)
        omni_config_group.add_argument(
            "--no-guardrails",
            dest="no_guardrails",
            action="store_true",
            help="Disable Cosmos3 text/video safety guardrails for this server.",
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

        # Supplementary auxiliary text encoder parameters
        # (e.g., the meta llama/meta llama-3.1-8b-instrument used by hidream)
        omni_config_group.add_argument(
            "--auxiliary-text-encoder",
            type=str,
            default=None,
            help="Auxiliary text encoder parameters model name or path (especially for Hidream-l1-full).",
        )

        # Stash via type(self) so the docs hook (which execs this function in a
        # sandboxed globals dict via ``DummySelf``) doesn't fail on a NameError.
        type(self)._parser = serve_parser

        return serve_parser


def run_headless(args: TrackingNamespace) -> None:
    """Run a single stage in headless mode.

    Honors ``--omni-dp-size-local``: launches that many replicas locally for
    ``--stage-id``. Each replica registers with the head's OmniMasterServer
    (auto-assigned replica id when ``--omni-dp-size-local > 1`` so multiple
    headless invocations can coexist) and reports heartbeats to the head's
    OmniCoordinator.
    """
    from vllm.v1.engine.coordinator import DPCoordinator
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor
    from vllm.version import __version__ as VLLM_VERSION

    from vllm_omni.diffusion.stage_diffusion_proc import (
        complete_diffusion_handshake,
        spawn_diffusion_proc,
    )
    from vllm_omni.distributed.omni_connectors.utils.initialization import resolve_omni_kv_config_for_stage
    from vllm_omni.engine.omni_core_engine_proc_manager import OmniCoreEngineProcManager
    from vllm_omni.engine.stage_engine_startup import register_stage_with_omni_master
    from vllm_omni.engine.stage_init_utils import (
        build_diffusion_config,
        build_engine_args_dict,
        build_vllm_config,
        extract_stage_metadata,
        get_stage_connector_spec,
        get_stage_devices_per_replica,
        inject_kv_stage_info,
        load_omni_transfer_config_for_model,
        prepare_engine_environment,
        setup_stage_devices,
        split_devices_for_replicas,
        terminate_alive_proc,
    )
    from vllm_omni.entrypoints.utils import inject_omni_kv_config, load_and_resolve_stage_configs
    from vllm_omni.platforms import current_omni_platform

    model = args.model
    stage_id: int | None = args.stage_id
    omni_master_address: str | None = args.omni_master_address
    omni_master_port: int | None = args.omni_master_port
    worker_backend: str | None = args.worker_backend
    stage_configs_path: str | None = args.stage_configs_path
    omni_replica_address: str | None = getattr(args, "omni_replica_address", None)
    omni_dp_size_local: int = max(1, int(getattr(args, "omni_dp_size_local", 1) or 1))

    if not model:
        raise ValueError("Failed to pass model from kwargs")
    if stage_id is None:
        raise ValueError("--stage-id is required in headless mode")
    if omni_master_address is None or omni_master_port is None:
        raise ValueError("--omni-master-address and --omni-master-port are required in headless mode")
    if worker_backend != "multi_process":
        raise ValueError("headless mode requires worker_backend=multi_process")

    # Filter down to a dict of things explicitly requested by the user
    args_dict = args.get_explicit_kwargs_dict()

    # ``--replica-id`` is deprecated and ignored — replica ids are
    # auto-assigned by ``OmniMasterServer`` so headless processes carry
    # no knowledge of their per-replica id at launch time. Warn (don't
    # error) when the operator still supplies it so existing launchers
    # keep working with a single log line.
    if "replica_id" in args_dict:
        logger.warning(
            "[Headless] --replica-id is deprecated and ignored "
            "(supplied value: %s). Replica ids are auto-assigned by the "
            "master server.",
            args.replica_id,
        )

    config_path, stage_configs = load_and_resolve_stage_configs(
        model,
        stage_configs_path,
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

    # When ``--omni-dp-size-local > 1``, slice the YAML's ``devices:`` field
    # into per-replica subsets so each subprocess we spawn below sees a
    # narrowed ``CUDA_VISIBLE_DEVICES`` and doesn't stack on cuda:0. Mirrors
    # the head-side per-replica device application at
    # ``async_omni_engine.py`` (setup_stage_devices around each launch).
    runtime_cfg = getattr(stage_cfg, "runtime", None)
    devices_str: str | None = None
    if runtime_cfg is not None:
        devices_str = (
            runtime_cfg.get("devices") if hasattr(runtime_cfg, "get") else getattr(runtime_cfg, "devices", None)
        )
    devices_per_replica = get_stage_devices_per_replica(stage_cfg)
    if devices_str:
        # Always remap YAML's logical devices through setup_stage_devices,
        # even for omni_dp_size_local==1. The launcher's CUDA_VISIBLE_DEVICES
        # is dropped from the engine-subprocess env between vllm-serve and
        # OmniCoreEngineProcManager.Process, so the worker would otherwise
        # default cuda:0 to physical GPU 0 and collide with a co-located
        # head on the same host (see hyi3_multi_host_1 reproducer).
        per_replica_devices: list[str | None] = split_devices_for_replicas(
            devices_str, omni_dp_size_local, devices_per_replica, stage_id
        )
        logger.info(
            "[Headless] Stage %d: %d local replicas, devices_per_replica=%d, per-replica devices: %s",
            stage_id,
            omni_dp_size_local,
            devices_per_replica,
            per_replica_devices,
        )
    else:
        per_replica_devices = [None] * omni_dp_size_local
    device_control_env = current_omni_platform.device_control_env_var

    if stage_cfg.stage_type == "diffusion":
        metadata = extract_stage_metadata(stage_cfg)
        if omni_conn_cfg:
            inject_omni_kv_config(stage_cfg, omni_conn_cfg, omni_from, omni_to)
        # Headless single-stage launch must still infer cross-stage TP topology
        # from the loaded deploy config so heterogeneous KV routing keys match
        # the head process (e.g. from_tp=2, to_tp=1).
        inject_kv_stage_info(stage_cfg, stage_id, stage_configs)
        od_config = build_diffusion_config(model, stage_cfg, metadata)

        logger.info(
            "[Headless] Launching %d diffusion replica(s) for stage %d via OmniMasterServer at %s:%d",
            omni_dp_size_local,
            stage_id,
            omni_master_address,
            omni_master_port,
        )

        procs: list[Any] = []
        try:
            for _rep_idx in range(omni_dp_size_local):
                # Always auto-assign: headless processes carry no knowledge
                # of their per-replica id and the master server is the sole
                # authority on the per-stage id namespace.
                response = register_stage_with_omni_master(
                    omni_master_address=omni_master_address,
                    omni_master_port=omni_master_port,
                    omni_stage_id=stage_id,
                    omni_stage_config=stage_cfg,
                    replica_id=None,
                    return_full_response=True,
                    replica_bind_address=omni_replica_address,
                )
                # Apply this replica's CUDA_VISIBLE_DEVICES (only when
                # ``--omni-dp-size-local > 1`` and the YAML's stage devices
                # field is set). The spawned subprocess inherits the env at
                # spawn time; we restore the parent env afterwards so the
                # next replica's setup sees the same baseline.
                previous_visible_devices = os.environ.get(device_control_env)
                try:
                    if per_replica_devices[_rep_idx] is not None:
                        setup_stage_devices(stage_id, {"devices": per_replica_devices[_rep_idx]})
                    # Each StageDiffusionProc starts its own
                    # torch.distributed group bound to
                    # ``od_config.master_port``. Without an explicit
                    # per-replica override all spawned subprocesses
                    # share the value ``OmniDiffusionConfig.__post_init__``
                    # picked once (and the second binder hits EADDRINUSE
                    # on ``init_process_group``). We can't use
                    # kernel-ephemeral allocation either, because the
                    # master server's pre-allocated ZMQ ports (returned
                    # by ``register_stage_with_omni_master``) also live
                    # in the ephemeral range and are not actually bound
                    # until the headless ``_perform_diffusion_handshake``
                    # runs — so picking an ephemeral port here can steal
                    # a port the master server already promised to a
                    # sibling headless. Use ``settle_port`` from a base
                    # above the Linux default ephemeral range
                    # (32768-60999) so torch.distributed master ports
                    # never overlap with ZMQ allocations.
                    if omni_dp_size_local > 1:
                        od_config.master_port = od_config.settle_port(
                            61000 + _rep_idx * 100,
                            port_inc=37,
                        )
                    proc, _, _, _ = spawn_diffusion_proc(
                        model,
                        od_config,
                        handshake_address=response.handshake_address,
                        request_address=response.input_address,
                        response_address=response.output_address,
                        omni_coordinator_address=response.coordinator_router_address,
                        omni_stage_id=stage_id,
                        omni_replica_id=response.replica_id,
                    )
                finally:
                    if previous_visible_devices is None:
                        current_omni_platform.unset_device_control_env_var()
                    else:
                        current_omni_platform.set_device_control_env_var(previous_visible_devices)
                complete_diffusion_handshake(proc, response.handshake_address, args.stage_init_timeout)
                procs.append(proc)
                logger.info(
                    "[Headless] Diffusion replica id=%d for stage %d is up (coord=%s)",
                    response.replica_id,
                    stage_id,
                    response.coordinator_router_address,
                )

            # Block on the sentinel set so any replica crash is detected
            # immediately (the previous per-proc join loop only noticed
            # crashes in registration order). Any exit triggers fleet
            # shutdown via the finally block; non-zero exits propagate.
            sentinel_to_proc = {p.sentinel: p for p in procs}
            died = connection.wait(list(sentinel_to_proc.keys()))
            first = sentinel_to_proc[died[0]]
            logger.info(
                "[Headless] Diffusion replica %s exited (code=%s); shutting down stage %d.",
                first.name,
                first.exitcode,
                stage_id,
            )
            if first.exitcode not in (None, 0):
                raise RuntimeError(
                    f"Diffusion stage {stage_id} replica {first.name!r} exited with code {first.exitcode}"
                )
            return
        finally:
            logger.info("[Headless] Shutting down %d diffusion replica(s) for stage %d.", len(procs), stage_id)
            for p in procs:
                if p.is_alive():
                    terminate_alive_proc(p)

    stage_connector_spec = get_stage_connector_spec(
        omni_transfer_config=omni_transfer_config,
        stage_id=stage_id,
        async_chunk=False,
    )

    # ``runtime_cfg`` is mostly inherited from the parent's
    # CUDA_VISIBLE_DEVICES; when ``--omni-dp-size-local > 1`` we additionally
    # bracket each replica's spawn below with setup_stage_devices so they
    # don't all stack on cuda:0 (see ``per_replica_devices`` above).
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
            "[Headless] Started DP Coordinator process for stage %d (PID: %d)",
            stage_id,
            coordinator.proc.pid,
        )

    logger.info(
        "[Headless] Launching %d omni replica(s) (vLLM dp_size_local=%d each) for stage %d "
        "via OmniMasterServer at %s:%d",
        omni_dp_size_local,
        local_engine_count,
        stage_id,
        omni_master_address,
        omni_master_port,
    )

    # One OmniMasterServer registration per omni replica; each registration
    # yields its own (handshake, input, output) allocation and the head's
    # OmniCoordinator ROUTER address. We then spawn one
    # OmniCoreEngineProcManager per replica so its subprocess gets the
    # right replica id wired into its OmniCoordClientForStage.
    log_stats = bool(args.log_stats)
    if args.disable_log_stats:
        log_stats = False

    engine_managers: list[Any] = []
    monitor_threads: list[threading.Thread] = []

    def _monitor_target(mgr: Any) -> None:
        try:
            mgr.monitor_engine_liveness()
        except Exception:
            logger.exception("[Headless] monitor_engine_liveness raised")

    try:
        for _rep_idx in range(omni_dp_size_local):
            # Always auto-assign: see the diffusion branch comment above
            # for the rationale (headless owns no replica-id namespace).
            response = register_stage_with_omni_master(
                omni_master_address=omni_master_address,
                omni_master_port=omni_master_port,
                omni_stage_id=stage_id,
                omni_stage_config=stage_cfg,
                coordinator=coordinator,
                replica_id=None,
                return_full_response=True,
                replica_bind_address=omni_replica_address,
                # LLM headless: the head binds *all* three sockets —
                # handshake ROUTER (``connect_remote_engine_cores``),
                # input ROUTER and output PULL (``CoreClient`` —
                # ``make_zmq_socket`` defaults bind=True for PULL).
                # The remote LLM worker is purely a connector: it
                # opens 3 outbound TCP connections to the master's
                # host. So the master must keep every address on
                # its own host; rewriting any of them to this
                # replica's NIC makes the head's ``bind`` go
                # EADDRNOTAVAIL on a cross-host launch.
                replica_binds_sockets=False,
            )
            # Per-replica CUDA_VISIBLE_DEVICES, same pattern as the diffusion
            # branch above. OmniCoreEngineProcManager.__init__ spawns its
            # subprocesses via context.Process inside the constructor, so we
            # must set the env *before* instantiation and restore after.
            previous_visible_devices = os.environ.get(device_control_env)
            try:
                if per_replica_devices[_rep_idx] is not None:
                    setup_stage_devices(stage_id, {"devices": per_replica_devices[_rep_idx]})
                mgr = OmniCoreEngineProcManager(
                    local_engine_count=local_engine_count,
                    start_index=dp_rank,
                    local_start_index=0,
                    vllm_config=vllm_config,
                    local_client=False,
                    handshake_address=response.handshake_address,
                    executor_class=executor_class,
                    log_stats=log_stats,
                    omni_stage_id=stage_id,
                    omni_coordinator_address=response.coordinator_router_address,
                    omni_replica_base_id=response.replica_id,
                )
            finally:
                if previous_visible_devices is None:
                    current_omni_platform.unset_device_control_env_var()
                else:
                    current_omni_platform.set_device_control_env_var(previous_visible_devices)
            engine_managers.append(mgr)
            logger.info(
                "[Headless] Stage %d replica id=%d up (coord=%s)",
                stage_id,
                response.replica_id,
                response.coordinator_router_address,
            )

        # Run all managers' liveness monitors in parallel. Each blocks
        # until its own subprocesses exit (or fail).
        if len(engine_managers) == 1:
            engine_managers[0].monitor_engine_liveness()
        else:
            for mgr in engine_managers:
                t = threading.Thread(target=_monitor_target, args=(mgr,), name=f"omni-replica-monitor-{id(mgr):x}")
                t.start()
                monitor_threads.append(t)
            for t in monitor_threads:
                t.join()
    finally:
        logger.info("[Headless] Shutting down stage %d (%d managers).", stage_id, len(engine_managers))
        for mgr in engine_managers:
            try:
                mgr.shutdown()
            except Exception:
                logger.exception("[Headless] engine manager shutdown failed")
        if coordinator is not None:
            coordinator.shutdown()


def cmd_init() -> list[CLISubcommand]:
    return [OmniServeCommand()]
