from dataclasses import dataclass, field
import torch
import random
import tempfile
from vllm_omni.diffusion.utils import is_port_available
import json
from typing import Any

from vllm.logger import init_logger
logger = init_logger(__name__)

@dataclass
class EngineArgs:
    # Model and path configuration (for convenience)
    model_path: str

    # Attention
    # attention_backend: str = None

    # Running mode
    # mode: ExecutionMode = ExecutionMode.INFERENCE

    # Workload type
    # workload_type: WorkloadType = WorkloadType.T2V

    # Cache strategy
    cache_strategy: str = "none"

    # Distributed executor backend
    distributed_executor_backend: str = "mp"
    nccl_port: int | None = None

    # HuggingFace specific parameters
    trust_remote_code: bool = False
    revision: str | None = None

    # Parallelism
    num_gpus: int = 1
    tp_size: int = -1
    sp_degree: int = -1
    # sequence parallelism
    ulysses_degree: int | None = None
    ring_degree: int | None = None
    # data parallelism
    # number of data parallelism groups
    dp_size: int = 1
    # number of gpu in a dp group
    dp_degree: int = 1
    # cfg parallel
    enable_cfg_parallel: bool = False

    hsdp_replicate_dim: int = 1
    hsdp_shard_dim: int = -1
    dist_timeout: int | None = None  # timeout for torch.distributed

    # pipeline_config: PipelineConfig = field(default_factory=PipelineConfig, repr=False)

    # LoRA parameters
    # (Wenxuan) prefer to keep it here instead of in pipeline config to not make it complicated.
    lora_path: str | None = None
    lora_nickname: str = "default"  # for swapping adapters in the pipeline
    # can restrict layers to adapt, e.g. ["q_proj"]
    # Will adapt only q, k, v, o by default.
    lora_target_modules: list[str] | None = None

    output_type: str = "pil"

    # CPU offload parameters
    dit_cpu_offload: bool = True
    use_fsdp_inference: bool = False
    text_encoder_cpu_offload: bool = True
    image_encoder_cpu_offload: bool = True
    vae_cpu_offload: bool = True
    pin_cpu_memory: bool = True

    # STA (Sliding Tile Attention) parameters
    mask_strategy_file_path: str | None = None
    # STA_mode: STA_Mode = STA_Mode.STA_INFERENCE
    skip_time_steps: int = 15

    # Compilation
    enable_torch_compile: bool = False

    disable_autocast: bool = False

    # VSA parameters
    VSA_sparsity: float = 0.0  # inference/validation sparsity

    # V-MoBA parameters
    moba_config_path: str | None = None
    # moba_config: dict[str, Any] = field(default_factory=dict)

    # Master port for distributed inference
    # TODO: do not hard code
    master_port: int | None = None

    # http server endpoint config, would be ignored in local mode
    host: str | None = None
    port: int | None = None

    scheduler_port: int = 5555

    # Stage verification
    enable_stage_verification: bool = True

    # Prompt text file for batch processing
    prompt_file_path: str | None = None

    # model paths for correct deallocation
    model_paths: dict[str, str] = field(default_factory=dict)
    model_loaded: dict[str, bool] = field(
        default_factory=lambda: {
            "transformer": True,
            "vae": True,
        }
    )
    override_transformer_cls_name: str | None = None

    # # DMD parameters
    # dmd_denoising_steps: List[int] | None = field(default=None)

    # MoE parameters used by Wan2.2
    boundary_ratio: float | None = None

    # Logging
    log_level: str = "info"

    def scheduler_endpoint(self):
        """
        Internal endpoint for scheduler

        """
        scheduler_host = self.host or "localhost"
        return f"tcp://{scheduler_host}:{self.scheduler_port}"
    
    @property
    def is_local_mode(self) -> bool:
        """
        If no server is running when a generation task begins, 'local_mode' will be enabled: a dedicated server will be launched
        """
        return self.host is None or self.port is None
    
    def settle_port(
        self, port: int, port_inc: int = 42, max_attempts: int = 100
    ) -> int:
        """
        Find an available port with retry logic.

        Args:
            port: Initial port to check
            port_inc: Port increment for each attempt
            max_attempts: Maximum number of attempts to find an available port

        Returns:
            An available port number

        Raises:
            RuntimeError: If no available port is found after max_attempts
        """
        attempts = 0
        original_port = port

        while attempts < max_attempts:
            if is_port_available(port):
                if attempts > 0:
                    logger.info(
                        f"Port {original_port} was unavailable, using port {port} instead"
                    )
                return port

            attempts += 1
            if port < 60000:
                port += port_inc
            else:
                # Wrap around with randomization to avoid collision
                port = 5000 + random.randint(0, 1000)

        raise RuntimeError(
            f"Failed to find available port after {max_attempts} attempts "
            f"(started from port {original_port})"
        )

    def __post_init__(self):
        # Add randomization to avoid race condition when multiple servers start simultaneously
        # if self.attention_backend in ["fa3", "fa4"]:
        #     self.attention_backend = "fa"

        initial_scheduler_port = self.scheduler_port + random.randint(0, 100)
        self.scheduler_port = self.settle_port(initial_scheduler_port)
        # TODO: remove hard code
        initial_master_port = (self.master_port or 30005) + random.randint(0, 100)
        self.master_port = self.settle_port(initial_master_port, 37)
        if self.moba_config_path:
            try:
                with open(self.moba_config_path) as f:
                    self.moba_config = json.load(f)
                logger.info("Loaded V-MoBA config from %s", self.moba_config_path)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.error(
                    "Failed to load V-MoBA config from %s: %s", self.moba_config_path, e
                )
                raise
        # self.check_server_args()

        # configure_logger(server_args=self)

        # log clean server_args
        # try:
        #     safe_args = _sanitize_for_logging(self, key_hint="server_args")
        #     logger.info("server_args: %s", json.dumps(safe_args, ensure_ascii=False))
        # except Exception:
        #     # Fallback to default repr if sanitization fails
        #     logger.info(f"server_args: {self}")
    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "EngineArgs":
        # # Convert mode string to enum if necessary
        # if "mode" in kwargs and isinstance(kwargs["mode"], str):
        #     kwargs["mode"] = ExecutionMode.from_string(kwargs["mode"])

        # # Convert workload_type string to enum if necessary
        # if "workload_type" in kwargs and isinstance(kwargs["workload_type"], str):
        #     kwargs["workload_type"] = WorkloadType.from_string(kwargs["workload_type"])

        # kwargs["pipeline_config"] = PipelineConfig.from_kwargs(kwargs)
        return cls(**kwargs)

@dataclass
class PortArgs:
    # The ipc filename for scheduler (rank 0) to receive inputs from tokenizer (zmq)
    scheduler_input_ipc_name: str

    # The port for nccl initialization (torch.dist)
    nccl_port: int

    # The ipc filename for rpc call between Engine and Scheduler
    rpc_ipc_name: str

    # The ipc filename for Scheduler to send metrics
    metrics_ipc_name: str

    # Master port for distributed inference
    master_port: int | None = None

    @staticmethod
    def from_engine_args(
        engine_args: EngineArgs, dp_rank: int | None = None
    ) -> "PortArgs":
        if engine_args.nccl_port is None:
            nccl_port = engine_args.scheduler_port + random.randint(100, 1000)
            while True:
                if is_port_available(nccl_port):
                    break
                if nccl_port < 60000:
                    nccl_port += 42
                else:
                    nccl_port -= 43
        else:
            nccl_port = engine_args.nccl_port

        # Normal case, use IPC within a single node
        return PortArgs(
            scheduler_input_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            nccl_port=nccl_port,
            rpc_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            metrics_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            master_port=engine_args.master_port,
        )


@dataclass
class OutputBatch:
    """
    Final output (after pipeline completion)
    """

    output: torch.Tensor | None = None
    error: str | None = None