# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import multiprocessing as mp
import os
import time
import uuid
import weakref
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal, overload

import huggingface_hub
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from vllm import SamplingParams
from vllm.logger import init_logger

from vllm_omni.distributed.omni_connectors import (
    get_stage_connector_config,
    initialize_orchestrator_connectors,
)
from vllm_omni.distributed.omni_connectors.adapter import try_send_via_connector
from vllm_omni.distributed.omni_connectors.utils.initialization import (
    resolve_omni_kv_config_for_stage,
)
from vllm_omni.distributed.ray_utils.utils import (
    create_placement_group,
    get_ray_queue_class,
    try_close_ray,
)
from vllm_omni.entrypoints.omni_stage import OmniStage
from vllm_omni.entrypoints.stage_utils import SHUTDOWN_TASK, OmniStageTaskType
from vllm_omni.entrypoints.stage_utils import maybe_load_from_ipc as _load
from vllm_omni.entrypoints.utils import (
    get_final_stage_id_for_e2e,
    inject_omni_kv_config,
    load_stage_configs_from_model,
    load_stage_configs_from_yaml,
    resolve_model_config_path,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType, OmniSamplingParams
from vllm_omni.metrics import OrchestratorAggregator, StageRequestStats
from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


def _weak_close_cleanup(stage_list, stage_in_queues, ray_pg):
    """Weak reference cleanup function for OmniBase instances."""
    if stage_list:
        for q in stage_in_queues:
            try:
                q.put_nowait(SHUTDOWN_TASK)
            except Exception as e:
                logger.warning(f"Failed to send shutdown signal to stage input queue: {e}")
        for stage in stage_list:
            try:
                stage.stop_stage_worker()
            except Exception as e:
                logger.warning(f"Failed to stop stage worker: {e}")
    try_close_ray(ray_pg)


def _dummy_snapshot_download(model_id):
    return model_id


def omni_snapshot_download(model_id) -> str:
    # If it's already a local path, just return it
    if os.path.exists(model_id):
        return model_id
    # TODO: this is just a workaround for quickly use modelscope, we should support
    # modelscope in weight loading feature instead of using `snapshot_download`
    if os.environ.get("VLLM_USE_MODELSCOPE", False):
        from modelscope.hub.snapshot_download import snapshot_download

        return snapshot_download(model_id)
    # For other cases (Hugging Face), perform a real download to ensure all
    # necessary files (including *.pt for audio/diffusion) are available locally
    # before stage workers are spawned. This prevents initialization timeouts.
    try:
        return download_weights_from_hf_specific(
            model_name_or_path=model_id,
            cache_dir=None,
            allow_patterns=["*"],
            require_all=True,
        )
    except huggingface_hub.errors.RepositoryNotFoundError:
        logger.warning(f"Repository not found for '{model_id}'.")
        return model_id


class OmniBase:
    """Base class for serving Omni models.

    Args:
        model: Model name or path to load.
        **kwargs: Arbitrary keyword arguments.
            - stage_configs_path: Optional path to YAML file containing stage
              configurations. If None, configurations are loaded from the model.
            - log_stats: Whether to enable statistics logging
              be written to files with stage-specific suffixes.
            - stage_init_timeout: Per-stage init watchdog (seconds). Measured from
              when the previous stage finished (possibly a prior Omni run with GPU
              reuse/overlap) to when the current stage starts to initialize.
            - shm_threshold_bytes: Threshold in bytes for using shared memory
              for IPC. Objects larger than this threshold will use shared memory.
            - worker_backend: Backend for worker processes. Default is "multi_process".
            - ray_address: Address of Ray cluster for Ray backend, if using Ray backend.
            - batch_timeout: Timeout in seconds for batching requests within a stage
            - init_timeout: Timeout in seconds for waiting for all stages to initialize
            - Additional keyword arguments passed to stage engines.
    """

    def __init__(self, model: str, **kwargs: Any) -> None:
        model = omni_snapshot_download(model)
        kwargs["model"] = model

        # Stage management attributes
        self.stage_list: list[OmniStage] = []
        self._stage_in_queues: list[mp.Queue] = []
        self._stage_out_queues: list[mp.Queue] = []
        self._stages_ready: set[int] = set()
        self._ray_pg = None
        self._queue_cls = None
        self._ctx = None

        # Initialize stages - each stage will create appropriate instance based on stage_type
        # Stage workers will automatically create OmniLLM or OmniDiffusion instances
        # based on stage_type in YAML config (handled in omni_stage.py)
        logger.info(f"Initializing stages for model: {model}")
        self._initialize_stages(model, kwargs)

    def _get_default_cache_config(self, cache_backend: str | None) -> dict[str, Any] | None:
        if cache_backend == "cache_dit":
            return {
                "Fn_compute_blocks": 1,
                "Bn_compute_blocks": 0,
                "max_warmup_steps": 4,
                "residual_diff_threshold": 0.24,
                "max_continuous_cached_steps": 3,
                "enable_taylorseer": False,
                "taylorseer_order": 1,
                "scm_steps_mask_policy": None,
                "scm_steps_policy": "dynamic",
            }
        if cache_backend == "tea_cache":
            return {
                "rel_l1_thresh": 0.2,
            }
        return None

    def _normalize_cache_config(self, cache_backend: str | None, cache_config: Any | None) -> Any | None:
        if isinstance(cache_config, str):
            try:
                cache_config = json.loads(cache_config)
            except json.JSONDecodeError:
                logger.warning("Invalid cache_config JSON, using defaults.")
                cache_config = None
        if cache_config is None and cache_backend not in (None, "", "none"):
            cache_config = self._get_default_cache_config(cache_backend)
        return cache_config

    def _create_default_diffusion_stage_cfg(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Create default diffusion stage configuration."""
        # We temporally create a default config for diffusion stage.
        # In the future, we should merge the default config with the user-provided config.
        # TODO: hack, convert dtype to string to avoid non-premitive omegaconf create error.
        if "dtype" in kwargs:
            kwargs["dtype"] = str(kwargs["dtype"])
        cache_backend = kwargs.get("cache_backend", "none")
        cache_config = self._normalize_cache_config(cache_backend, kwargs.get("cache_config", None))
        # TODO: hack, calculate devices based on parallel config.
        devices = "0"
        if "parallel_config" in kwargs:
            num_devices = kwargs["parallel_config"].world_size
            for i in range(1, num_devices):
                devices += f",{i}"
        default_stage_cfg = [
            {
                "stage_id": 0,
                "stage_type": "diffusion",
                "runtime": {
                    "process": True,
                    "devices": devices,
                    "max_batch_size": 1,
                },
                "engine_args": OmegaConf.create(
                    {
                        **kwargs,
                        "cache_backend": cache_backend,
                        "cache_config": cache_config,
                    }
                ),
                "final_output": True,
                "final_output_type": "image",
            }
        ]
        default_stage_cfg[0]["engine_args"]["model_stage"] = "diffusion"
        return default_stage_cfg

    def _initialize_stages(self, model: str, kwargs: dict[str, Any]) -> None:
        """Initialize stage list management."""
        stage_init_timeout = kwargs.get("stage_init_timeout", 20)
        shm_threshold_bytes = kwargs.get("shm_threshold_bytes", 65536)
        init_timeout = kwargs.get("init_timeout", 300)
        worker_backend = kwargs.get("worker_backend", "multi_process")
        ray_address = kwargs.get("ray_address", None)
        batch_timeout = kwargs.get("batch_timeout", 10)
        stage_configs_path = kwargs.get("stage_configs_path", None)
        log_stats = kwargs.get("log_stats", False)

        ### base engine args
        tokenizer = kwargs.get("tokenizer", None)

        base_engine_args = {"tokenizer": tokenizer} if tokenizer is not None else None

        # Load stage configurations from YAML
        if stage_configs_path is None:
            self.config_path = resolve_model_config_path(model)
            self.stage_configs = load_stage_configs_from_model(model, base_engine_args=base_engine_args)
            if not self.stage_configs:
                default_stage_cfg = self._create_default_diffusion_stage_cfg(kwargs)
                self.stage_configs = OmegaConf.create(default_stage_cfg)
        else:
            self.config_path = stage_configs_path
            self.stage_configs = load_stage_configs_from_yaml(stage_configs_path, base_engine_args=base_engine_args)

        # Inject diffusion LoRA-related knobs from kwargs if not present in the stage config.
        for cfg in self.stage_configs:
            try:
                if getattr(cfg, "stage_type", None) != "diffusion":
                    continue
                if not hasattr(cfg, "engine_args") or cfg.engine_args is None:
                    cfg.engine_args = OmegaConf.create({})
                if kwargs.get("lora_path") is not None:
                    if not hasattr(cfg.engine_args, "lora_path") or cfg.engine_args.lora_path is None:
                        cfg.engine_args.lora_path = kwargs["lora_path"]
                lora_scale = kwargs.get("lora_scale")
                if lora_scale is None:
                    # Backwards compatibility for older callers.
                    lora_scale = kwargs.get("static_lora_scale")
                if lora_scale is not None:
                    if not hasattr(cfg.engine_args, "lora_scale") or cfg.engine_args.lora_scale is None:
                        cfg.engine_args.lora_scale = lora_scale
            except Exception as e:
                logger.warning("Failed to inject LoRA config for stage: %s", e)

        # Initialize connectors
        self.omni_transfer_config, self.connectors = initialize_orchestrator_connectors(
            self.config_path, worker_backend=worker_backend, shm_threshold_bytes=shm_threshold_bytes
        )

        # Initialize stats paths
        self.log_stats: bool = bool(log_stats)

        self.worker_backend = worker_backend
        self.ray_address = ray_address
        self.batch_timeout = batch_timeout
        # async chunk remains the same for each stage
        self.async_chunk = self._is_async_chunk_enable(self.stage_configs)

        # Build OmniStage instances in parallel, preserve original order
        def _build_stage(idx_cfg: tuple[int, Any]) -> tuple[int, OmniStage]:
            idx, cfg = idx_cfg
            return idx, OmniStage(cfg, stage_init_timeout=stage_init_timeout)

        with ThreadPoolExecutor(max_workers=min(len(self.stage_configs), max(1, os.cpu_count() or 1))) as executor:
            futures = [executor.submit(_build_stage, (idx, cfg)) for idx, cfg in enumerate(self.stage_configs)]
            results: list[tuple[int, OmniStage]] = []
            for fut in as_completed(futures):
                results.append(fut.result())
        results.sort(key=lambda x: x[0])
        self.stage_list = [st for _, st in results]
        self.default_sampling_params_list = [st.default_sampling_params for st in self.stage_list]
        self.output_modalities = [st.final_output_type for st in self.stage_list]
        logger.debug(f"[{self._name}] Loaded {len(self.stage_list)} stages")

        if self.worker_backend == "ray":
            self._queue_cls = get_ray_queue_class()
        else:
            self._ctx = mp.get_context("spawn")
            self._queue_cls = lambda: self._ctx.Queue(maxsize=0)

        self._stage_init_timeout = max(0, int(stage_init_timeout))
        self._shm_threshold_bytes = max(0, int(shm_threshold_bytes))
        self._start_stages(model)
        # Wait for all stages to report readiness before seeding
        self._wait_for_stages_ready(timeout=init_timeout)

    def _is_async_chunk_enable(self, stage_args: list) -> bool:
        """get async chunk flag"""
        engine_args = getattr(stage_args[0], "engine_args", None)
        return bool(getattr(engine_args, "async_chunk", False))

    def _start_stages(self, model: str) -> None:
        """Start all stage processes."""
        if self.worker_backend == "ray":
            # Initialize Ray Cluster
            self._ray_pg = create_placement_group(
                number_of_stages=len(self.stage_list), address=self.ray_address, strategy="PACK"
            )

        for stage_id, stage in enumerate[OmniStage](self.stage_list):
            in_q = self._queue_cls()
            out_q = self._queue_cls()
            self._stage_in_queues.append(in_q)
            self._stage_out_queues.append(out_q)
            stage.attach_queues(in_q, out_q)

            stage_connectors_config = get_stage_connector_config(
                self.omni_transfer_config,
                stage_id,
            )

            # Inject YAML-resolved connector config into omni_kv_config for
            # in-engine usage (GPU model runner reads model_config.omni_kv_config).
            try:
                omni_conn_cfg, omni_from, omni_to = resolve_omni_kv_config_for_stage(
                    self.omni_transfer_config, stage_id
                )
                if omni_conn_cfg:
                    inject_omni_kv_config(stage, omni_conn_cfg, omni_from, omni_to)  # type: ignore

            except Exception as e:
                logger.debug("[Omni] Failed to inject omni connector config into stage-%s: %s", stage_id, e)

            stage.init_stage_worker(
                model,
                is_async=self.is_async,
                shm_threshold_bytes=self._shm_threshold_bytes,
                ctx=self._ctx if self.worker_backend != "ray" else None,
                batch_timeout=self.batch_timeout,
                connectors_config=stage_connectors_config,
                worker_backend=self.worker_backend,
                ray_placement_group=self._ray_pg,
            )

            logger.debug(f"[{self._name}] Stage-{stage_id} process started")

    def _process_stage_ready(self, stage: OmniStage, stage_id: int, result: dict[str, Any]) -> None:
        self._stages_ready.add(stage_id)
        logger.info(f"[{self._name}] Stage-{stage_id} reported ready")

    def _wait_for_stages_ready(self, timeout: int = 120) -> None:
        """Wait for all stages to report readiness with optimized polling."""
        num_stages = len(self.stage_list)
        deadline = time.time() + max(0, int(timeout))

        logger.info(f"[{self._name}] Waiting for {num_stages} stages to initialize (timeout: {timeout}s)")

        while len(self._stages_ready) < num_stages and time.time() < deadline:
            progressed = False
            for stage_id, stage in enumerate(self.stage_list):
                if stage_id in self._stages_ready:
                    continue

                # Check if the stage has reported status
                if result := stage.try_collect():
                    progressed = True
                    if result.get("type") == "stage_ready":
                        self._process_stage_ready(stage, stage_id, result)

            if not progressed:
                time.sleep(0.05)

        # Handle Final State
        if len(self._stages_ready) == num_stages:
            logger.info(f"[{self._name}] All stages initialized successfully")
            return

        # Handle Timeout/Failure
        not_ready = sorted(set(range(num_stages)) - set(self._stages_ready))
        logger.warning(
            f"[{self._name}] Initialization timeout: {len(self._stages_ready)}/{num_stages} "
            f"stages ready. Missing stages: {not_ready}"
        )

        suggestions = [
            f"Ignore this warning if the model weight download / load from disk time is longer than {timeout}s.",
            "Verify GPU/device assignment in config (runtime.devices) is correct.",
            "Check GPU/host memory availability; reduce model or batch size if needed.",
            "Check model weights path and network reachability (if loading remotely).",
            "Increase initialization wait time (stage_init_timeout or call-site timeout).",
        ]

        formatted_suggestions = "\n".join(f"  {i + 1}) {msg}" for i, msg in enumerate(suggestions))

        logger.warning(f"[{self._name}] Stage initialization timeout. Troubleshooting Steps:\n{formatted_suggestions}")

    def start_profile(self, stages: list[int] | None = None) -> None:
        """Start profiling for specified stages.

        Sends start_profile command to stage workers. Profiling must be enabled
        via VLLM_TORCH_PROFILER_DIR environment variable.

        Args:
            stages: List of stage IDs to start profiling. If None, starts
                profiling for all stages that have profiling enabled.

        Example:
            >>> # Profile all stages
            >>> omni.start_profile()
            >>> outputs = omni.generate(prompts, sampling_params)
            >>> omni.stop_profile()

            >>> # Profile only stage 0 and 2
            >>> omni.start_profile(stages=[0, 2])
        """
        if stages is None:
            stages = list(range(len(self.stage_list)))

        for stage_id in stages:
            if stage_id < len(self.stage_list):
                try:
                    self.stage_list[stage_id].submit({"type": OmniStageTaskType.PROFILER_START})
                    logger.info("[%s] Sent start_profile to stage-%s", self._name, stage_id)
                except Exception as e:
                    logger.warning(
                        "[%s] Failed to send start_profile to stage-%s: %s",
                        self._name,
                        stage_id,
                        e,
                    )

    def stop_profile(self, stages: list[int] | None = None) -> dict:
        """
        Synchronously stop profiling for specified stages and collect
        the file paths for traces and tables.
        """
        if stages is None:
            stages = list(range(len(self.stage_list)))

        all_results = {"traces": [], "tables": []}

        for stage_id in stages:
            if stage_id < len(self.stage_list):
                stage = self.stage_list[stage_id]

                # Check if the stage object has our new bridge method
                if hasattr(stage, "stop_profile"):
                    logger.info("[%s] Requesting profile data collection from stage-%s", self._name, stage_id)

                    # This is the blocking call that triggers the RPC chain
                    stage_data = stage.stop_profile()

                    if isinstance(stage_data, dict):
                        # FIX: Handle both single key and list key formats
                        traces = stage_data.get("trace") or stage_data.get("traces")
                        tables = stage_data.get("table") or stage_data.get("tables")

                        # Debug logging
                        logger.debug(f"[{self._name}] Stage-{stage_id} returned: {stage_data.keys()}")
                        if traces:
                            logger.debug(f"[{self._name}] Stage-{stage_id} traces type: {type(traces)}")
                        if tables:
                            logger.debug(f"[{self._name}] Stage-{stage_id} tables type: {type(tables)}")

                        # Handle single strings
                        if traces:
                            if isinstance(traces, str):
                                all_results["traces"].append(traces)
                            elif isinstance(traces, list):
                                all_results["traces"].extend(traces)

                        # Handle single strings
                        if tables:
                            if isinstance(tables, str):
                                all_results["tables"].append(tables)
                            elif isinstance(tables, list):
                                all_results["tables"].extend(tables)
                        else:
                            logger.warning(f"[{self._name}] Stage-{stage_id} returned no table data")
                    else:
                        logger.warning(f"[{self._name}] Stage-{stage_id} returned non-dict data: {type(stage_data)}")
                else:
                    # Fallback for non-diffusion stages
                    logger.warning(
                        "[%s] Stage-%s does not support synchronous stop_profile. Falling back to async.",
                        self._name,
                        stage_id,
                    )
                    stage.submit({"type": OmniStageTaskType.PROFILER_STOP})

        # Final debug output
        logger.info(
            f"[{self._name}] Collected {len(all_results['traces'])} trace(s) and {len(all_results['tables'])} table(s)"
        )

        return all_results

    def close(self) -> None:
        """Close all stage processes and clean up resources."""
        if hasattr(self, "_weak_finalizer"):
            self._weak_finalizer()

    @property
    def _name(self) -> str:
        return "OmniBase"

    @property
    def is_async(self) -> bool:
        return False


class Omni(OmniBase):
    """Unified entrypoint for both LLM and Diffusion models for better usability.

    Args:
        model: Model name or path to load.
        **kwargs: Arbitrary keyword arguments.
            - stage_configs_path: Optional path to YAML file containing stage
              configurations. If None, configurations are loaded from the model.
            - log_stats: Whether to enable statistics logging
              be written to files with stage-specific suffixes.
            - stage_init_timeout: Per-stage init watchdog (seconds). Measured from
              when the previous stage finished (possibly a prior Omni run with GPU
              reuse/overlap) to when the current stage starts to initialize.
            - shm_threshold_bytes: Threshold in bytes for using shared memory
              for IPC. Objects larger than this threshold will use shared memory.
            - worker_backend: Backend for worker processes. Default is "multi_process".
            - ray_address: Address of Ray cluster for Ray backend, if using Ray backend.
            - batch_timeout: Timeout in seconds for batching requests within a stage
            - init_timeout: Timeout in seconds for waiting for all stages to initialize
            - Additional keyword arguments passed to stage engines.

    Example:
        >>> omni = Omni(model="Qwen/Qwen2.5-Omni-7B")
        >>> outputs = omni.generate(prompts="Hello, world!", sampling_params_list=[SamplingParams()])
        >>> print(outputs)
    """

    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)

        # Register weak reference cleanup (called on garbage collection)
        self._weak_finalizer = weakref.finalize(
            self,
            _weak_close_cleanup,
            self.stage_list,
            self._stage_in_queues,
            self._ray_pg,
        )

    @overload
    def generate(
        self,
        prompts: OmniPromptType | Sequence[OmniPromptType],
        sampling_params_list: OmniSamplingParams | Sequence[OmniSamplingParams] | None = None,
        *,
        py_generator: Literal[True],
    ) -> Generator[OmniRequestOutput, None, None]: ...

    @overload
    def generate(
        self,
        prompts: OmniPromptType | Sequence[OmniPromptType],
        sampling_params_list: OmniSamplingParams | Sequence[OmniSamplingParams] | None = None,
        *,
        py_generator: Literal[False] = False,
    ) -> list[OmniRequestOutput]: ...

    def generate(
        self,
        prompts: OmniPromptType | Sequence[OmniPromptType],
        sampling_params_list: OmniSamplingParams | Sequence[OmniSamplingParams] | None = None,
        *,
        py_generator: bool = False,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> Generator[OmniRequestOutput, None, None] | list[OmniRequestOutput]:
        """Generate outputs for the given prompts.

        Orchestrates the multi-stage pipeline based on YAML configuration.
        Each stage will use OmniLLM or OmniDiffusion based on stage_type.

        Args:
            prompts: Input prompt(s) for generation.
            sampling_params_list: Optional list of per-stage parameters.
            py_generator: Whether the returned result(s) are wrapped in a generator instead of a list.
            use_tqdm: Whether to use tqdm progress bar

        Returns:
            List of OmniRequestOutput objects, one for each input prompt.
            Each output contains the stage_id, final_output_type, and
            the request_output from the final stage.

        Raises:
            ValueError: If sampling_params_list is None or has incorrect length.
        """
        if sampling_params_list is None:
            sampling_params_list = self.default_sampling_params_list
        elif not isinstance(sampling_params_list, Sequence):
            # TODO: After the recent introduction of BAGEL model (one LLM and one Diffusion),
            # expect the text_to_image example code to run when only passing one OmniDiffusionSamplingParams
            # This behavior may be confusing, and future PR can improve it.
            per_stage_params: list[OmniSamplingParams] = []
            for default_stage_sp in self.default_sampling_params_list:
                default_sp_type = default_stage_sp.__class__
                if default_sp_type == sampling_params_list.__class__:
                    per_stage_params.append(sampling_params_list)
                else:
                    per_stage_params.append(default_stage_sp)
            sampling_params_list = per_stage_params

        try:
            if py_generator:
                return self._run_generation_with_generator(prompts, sampling_params_list)
            else:
                outputs = list(self._run_generation(prompts, sampling_params_list, use_tqdm))
                return outputs
        except Exception as e:
            logger.exception("[Orchestrator] Failed to run generation: %s", e)
            # Always close on exception to ensure cleanup
            self.close()
            raise e

    def _run_generation_with_generator(
        self,
        prompts: OmniPromptType | Sequence[OmniPromptType],
        sampling_params_list: Sequence[OmniSamplingParams],
    ) -> Generator[OmniRequestOutput, None, None]:
        """Run generation through all stages in the pipeline and return a generator."""
        gen = self._run_generation(prompts, sampling_params_list)
        try:
            yield from gen
        except Exception as e:
            logger.exception("[Orchestrator] Failed to run generation: %s", e)
            raise e
        finally:
            # Cleanup when generator is exhausted or closed
            self.close()

    def _run_generation(
        self,
        prompts: OmniPromptType | Sequence[OmniPromptType],
        sampling_params_list: Sequence[OmniSamplingParams],
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> Generator[OmniRequestOutput, None, None]:
        """Run generation through all stages in the pipeline."""
        logger.debug(f"[{self._name}] generate() called")
        if sampling_params_list is None:
            raise ValueError("sampling_params_list is required for pipelined generation")

        if len(sampling_params_list) != len(self.stage_list):
            raise ValueError(f"Expected {len(self.stage_list)} sampling params, got {len(sampling_params_list)}")

        for i, (stage, sp) in enumerate(zip(self.stage_list, sampling_params_list)):
            ExpectedSPType = OmniDiffusionSamplingParams if stage.stage_type == "diffusion" else SamplingParams
            if not isinstance(sp, ExpectedSPType):
                raise ValueError(
                    f"Expected sampling parameters with type {ExpectedSPType} in stage {i}, got {sp.__class__}"
                )

        # Normalize prompts to a list for per-request iteration
        # str is also Sequence but only test list-like containers here
        if isinstance(prompts, str) or not isinstance(prompts, Sequence):
            request_prompts: list[OmniPromptType] = [prompts]
        else:
            request_prompts = list(prompts)

        # Orchestrator keeps stage objects for input derivation
        num_stages = len(self.stage_list)

        # Generate globally unique request IDs and map them to original prompts
        request_ids = [f"{i}_{uuid.uuid4()}" for i in range(len(request_prompts))]
        request_id_to_prompt = {rid: p for rid, p in zip(request_ids, request_prompts)}

        # Track per-request start time for end-to-end timing
        _req_start_ts: dict[str, float] = {}
        _wall_start_ts: float = time.time()

        # Determine the final stage for E2E stats (highest stage_id with final_output=True; fallback to last stage)
        final_stage_id_to_prompt: dict[str, int] = {}
        for rid, prompt in request_id_to_prompt.items():
            if isinstance(prompt, dict):
                prompt_modalities = prompt.get("modalities", None)
            else:
                prompt_modalities = None
            final_stage_id_for_e2e = get_final_stage_id_for_e2e(
                prompt_modalities, self.output_modalities, self.stage_list
            )
            final_stage_id_to_prompt[rid] = final_stage_id_for_e2e

        # Metrics/aggregation helper
        metrics = OrchestratorAggregator(
            num_stages,
            self.log_stats,
            _wall_start_ts,
            final_stage_id_to_prompt,
        )

        it = request_id_to_prompt.items()
        if use_tqdm:
            tqdm_func = use_tqdm if callable(use_tqdm) else tqdm
            it = tqdm_func(it, desc="Adding requests")

        # Seed stage-0 queue with all requests
        logger.debug(f"[{self._name}] Seeding {len(request_prompts)} requests into stage-0")
        # Mark first input time for stage-0
        metrics.stage_first_ts[0] = metrics.stage_first_ts[0] or time.time()

        for req_id, prompt in request_id_to_prompt.items():
            sp0 = sampling_params_list[0]  # type: ignore[index]
            task = {
                "request_id": req_id,
                "engine_inputs": prompt,
                "sampling_params": sp0,
            }
            self.stage_list[0].submit(task)
            _req_start_ts[req_id] = time.time()
            logger.debug(f"[{self._name}] Enqueued request {req_id} to stage-0")

        pbar = None
        if use_tqdm:
            tqdm_func = use_tqdm if callable(use_tqdm) else tqdm
            pbar = tqdm_func(
                total=len(request_prompts),
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(f"est. speed input: {0:.2f} unit/s, output: {0:.2f} unit/s"),
            )
        # For each stage, forward results to next stage; collect finals at the end
        # We pipeline by continually polling output queues in stage order
        remaining_by_stage: list[int] = [len(request_prompts)] + [0] * (num_stages - 1)
        completed_requests = 0
        total_requests = len(request_prompts)

        logger.debug(
            f"[{self._name}] Entering scheduling loop: total_requests={total_requests}, stages={num_stages}",
        )
        while completed_requests < total_requests:
            made_progress = False
            for stage_id, stage in enumerate(self.stage_list):
                result = stage.try_collect()
                if result is None:
                    continue

                made_progress = True
                req_id = result.get("request_id")
                if "error" in result:
                    logger.error(
                        f"[{self._name}] Stage {stage_id} error on request {req_id}: {result['error']}",
                    )
                    continue

                if result.get("type") == "stage_ready":
                    # Only happens when stage is initialized slower than expected,
                    # so we wait for a short time and try again
                    time.sleep(0.05)
                    continue

                engine_outputs = _load(result, obj_key="engine_outputs", shm_key="engine_outputs_shm")
                # Mark last output time for this stage whenever we receive outputs
                metrics.stage_last_ts[stage_id] = max(metrics.stage_last_ts[stage_id] or 0.0, time.time())
                try:
                    _m: StageRequestStats = result.get("metrics")
                    if _m is not None:
                        # Accumulate generation time
                        metrics.accumulated_gen_time_ms[req_id][stage_id] += _m.stage_gen_time_ms

                        # For diffusion stages, we also accumulate diffusion time
                        metrics.accumulate_diffusion_metrics(stage.stage_type, req_id, engine_outputs)

                        metrics.on_stage_metrics(stage_id, req_id, _m, stage.final_output_type)
                        if pbar:
                            elapsed = pbar.format_dict["elapsed"] or 1e-6
                            # Aggregate total tokens/images across all stages
                            total_out = sum(metrics.stage_total_tokens)
                            out_spd = total_out / elapsed

                            modality = self.output_modalities[stage_id]
                            unit = "img" if modality == "image" else "tok"

                            # Pre-calculate for cleaner string formatting
                            if metrics.e2e_count > 0:
                                avg_lat = metrics.e2e_total_ms / metrics.e2e_count
                            else:
                                avg_lat = 0

                            # Align with vLLM's wording "est. speed" using multi-line parentheses
                            pbar.postfix = (
                                f"est. speed stage-{stage_id} {unit}/s: {out_spd:.2f}, avg e2e_lat: {avg_lat:.1f}ms"
                            )
                except Exception as e:
                    logger.exception(
                        f"[{self._name}] Failed to process metrics for stage {stage_id}, req {req_id}: {e}",
                    )
                logger.debug(
                    f"[{self._name}] Stage-{stage_id} completed request {req_id}; forwarding or finalizing",
                )
                stage.set_engine_outputs(engine_outputs)

                if getattr(stage, "final_output", False):
                    logger.debug(
                        f"[{self._name}] Request {req_id} finalized at stage-{stage_id}",
                    )

                    # End-to-end timing and time-per-token for final output
                    # (only once per request at the designated final stage)
                    try:
                        if stage_id == final_stage_id_to_prompt[req_id]:
                            metrics.on_finalize_request(
                                stage_id,
                                req_id,
                                _req_start_ts.get(req_id, _wall_start_ts),
                            )
                    except Exception as e:
                        logger.exception(
                            f"[{self._name}] Finalize request handling error for req {req_id} at stage {stage_id}: {e}",
                        )
                    output_to_yield = OmniRequestOutput(
                        stage_id=stage_id,
                        final_output_type=stage.final_output_type,  # type: ignore[attr-defined]
                        request_output=engine_outputs,
                    )

                    # Record audio generated frames with unified signature
                    try:
                        finished = (
                            engine_outputs.finished
                            if hasattr(engine_outputs, "finished")
                            else (
                                engine_outputs[0].finished
                                if isinstance(engine_outputs, list)
                                and engine_outputs
                                and hasattr(engine_outputs[0], "finished")
                                else False
                            )
                        )
                        metrics.record_audio_generated_frames(output_to_yield, finished, stage_id, req_id)
                    except Exception as e:
                        logger.exception(
                            f"[{self._name}] Failed to record audio metrics for req {req_id} at stage {stage_id}: {e}",
                        )

                    yield output_to_yield

                next_stage_id = stage_id + 1
                if next_stage_id <= final_stage_id_to_prompt[req_id]:
                    next_stage: OmniStage = self.stage_list[next_stage_id]
                    try:
                        # Derive inputs for the next stage, record preprocess time
                        with metrics.stage_postprocess_timer(stage_id, req_id):
                            next_inputs = next_stage.process_engine_inputs(
                                self.stage_list, [request_id_to_prompt[req_id]]
                            )
                    except Exception as e:
                        logger.exception(
                            f"[{self._name}] Process engine inputs error for req {req_id}"
                            f" at stage {next_stage_id}: {e}",
                        )
                        continue
                    sp_next = sampling_params_list[next_stage_id]  # type: ignore[index]

                    # Check if we have a connector for this edge
                    connector_key = (str(stage_id), str(next_stage_id))
                    connector = self.connectors.get(connector_key)
                    sent_via_connector = False
                    if connector:
                        sent_via_connector = try_send_via_connector(
                            connector=connector,
                            stage_id=stage_id,
                            next_stage_id=next_stage_id,
                            req_id=req_id,
                            next_inputs=next_inputs,
                            sampling_params=sp_next,
                            original_prompt=request_id_to_prompt[req_id],
                            next_stage_queue_submit_fn=self.stage_list[next_stage_id].submit,
                            metrics=metrics,
                        )

                    if not sent_via_connector:
                        raise RuntimeError(
                            f"[{self._name}] Failed to send request {req_id} to stage-{next_stage_id} via connector. "
                            "Configure a connector for this edge or inspect connector logs for details."
                        )
                    logger.debug(
                        f"[{self._name}] Forwarded request {req_id} to stage-{next_stage_id}",
                    )
                    remaining_by_stage[next_stage_id] += 1
                else:
                    completed_requests += 1
                    if pbar:
                        final_mod = self.output_modalities[final_stage_id_to_prompt[req_id]]
                        pbar.unit = "img" if final_mod == "image" else "req"
                        pbar.update(1)
                    logger.debug(
                        f"[{self._name}] Request {req_id} fully completed ({completed_requests}/{total_requests})",
                    )

            if not made_progress:
                time.sleep(0.005)
        logger.debug(f"[{self._name}] All requests completed")

        if pbar:
            pbar.close()

        # Summarize and print stats
        try:
            metrics.build_and_log_summary()
        except Exception as e:
            logger.exception(f"[{self._name}] Failed to build/log summary: {e}")

    @property
    def _name(self) -> str:
        return "Orchestrator"
