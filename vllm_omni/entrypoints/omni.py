# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import multiprocessing as mp
import os
import time
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pprint import pformat
from typing import Any

import msgspec
from omegaconf import OmegaConf
from tqdm import tqdm  # Added for progress visualization

from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.distributed.omni_connectors import (
    get_stage_connector_config,
    initialize_orchestrator_connectors,
)
from vllm_omni.distributed.omni_connectors.adapter import try_send_via_connector
from vllm_omni.distributed.ray_utils.utils import (
    create_placement_group,
    get_ray_queue_class,
    try_close_ray,
)
from vllm_omni.entrypoints.log_utils import OrchestratorMetrics
from vllm_omni.entrypoints.omni_stage import OmniStage
from vllm_omni.entrypoints.stage_utils import maybe_load_from_ipc as _load
from vllm_omni.entrypoints.utils import (
    get_final_stage_id_for_e2e,
    load_stage_configs_from_model,
    load_stage_configs_from_yaml,
    resolve_model_config_path,
)
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


def _dummy_snapshot_download(model_id):
    return model_id


def omni_snapshot_download(model_id) -> str:
    # TODO: this is just a workaround for quickly use modelscope, we should support
    # modelscope in weight loading feature instead of using `snapshot_download`
    if os.environ.get("VLLM_USE_MODELSCOPE", False):
        from modelscope.hub.snapshot_download import snapshot_download
        return snapshot_download(model_id)
    else:
        return _dummy_snapshot_download(model_id)


class Omni:
    """
    Unified entrypoint for both LLM and Diffusion models for better usability.

    Args:
        *args: Variable length argument list.
            - args[0]: Model name or path to load.
        **kwargs: Arbitrary keyword arguments.
            - model: Model name or path to load (if not in args).
            - stage_configs_path: Optional path to YAML file containing stage
              configurations. If None, configurations are loaded from the model.
            - log_stats: Whether to enable statistics logging
    """

    def __init__(self, *args: Any, **kwargs: dict[str, Any]) -> None:
        model = args[0] if args else kwargs.get("model", "")
        assert model != "", "Null model id detected, please specify a model id."
        model = omni_snapshot_download(model)
        if args:
            args[0] = model
        elif kwargs.get("model", "") != "":
            kwargs["model"] = model

        # Stage management attributes
        self.stage_list: list[OmniStage] = []
        self._stage_in_queues: list[mp.Queue] = []
        self._stage_out_queues: list[mp.Queue] = []
        self._stages_ready: set[int] = set()
        self._ray_pg = None
        self._queue_cls = None
        self._ctx = None

        logger.info(f"Initializing stages for model: {model}")
        self._initialize_stages(model, kwargs)

    def _initialize_stages(self, model: str, kwargs: dict[str, Any]) -> None:
        """Initialize stage list management.
        Each stage will create appropriate instance (OmniLLM or OmniDiffusion)
        based on stage_type in YAML config (handled in omni_stage.py).
        """
        init_sleep_seconds = kwargs.get("init_sleep_seconds", 20)
        shm_threshold_bytes = kwargs.get("shm_threshold_bytes", 65536)
        init_timeout = kwargs.get("init_timeout", 300)
        worker_backend = kwargs.get("worker_backend", "multi_process")
        ray_address = kwargs.get("ray_address", None)
        batch_timeout = kwargs.get("batch_timeout", 10)
        stage_configs_path = kwargs.get("stage_configs_path", None)
        log_stats = kwargs.get("log_stats", False)

        # Load stage configurations from YAML or model
        if stage_configs_path is None:
            self.config_path = resolve_model_config_path(model)
            self.stage_configs = load_stage_configs_from_model(model)
            if not self.stage_configs:
                # Fallback for diffusion-only or custom models
                if "dtype" in kwargs:
                    kwargs["dtype"] = str(kwargs["dtype"])
                devices = "0"
                if "parallel_config" in kwargs:
                    num_devices = kwargs["parallel_config"].world_size
                    for i in range(1, num_devices):
                        devices += f",{i}"
                logger.info(f"model: {model}, kwargs: {kwargs}")
                default_stage_cfg = [
                    {
                        "stage_id": 0,
                        "stage_type": "diffusion",
                        "runtime": {
                            "process": True,
                            "devices": devices,
                            "max_batch_size": 1,
                        },
                        "engine_args": OmegaConf.create(kwargs),
                        "final_output": True,
                        "final_output_type": "image",
                    }
                ]
                default_stage_cfg[0]["engine_args"]["model_stage"] = "diffusion"
                self.stage_configs = OmegaConf.create(default_stage_cfg)
        else:
            self.config_path = stage_configs_path
            self.stage_configs = load_stage_configs_from_yaml(stage_configs_path)

        # SKIP CODE2WAV: Filter to keep only thinker (stage 0) and talker (stage 1)
        self.stage_configs = [s for s in self.stage_configs if s['stage_id'] in [0, 1]]
        for s in self.stage_configs:
            if s['stage_id'] == 1:
                s['final_output'] = True

        # Initialize connectors for inter-stage communication
        self.omni_transfer_config, self.connectors = initialize_orchestrator_connectors(
            self.config_path, worker_backend=worker_backend, shm_threshold_bytes=shm_threshold_bytes
        )

        self._enable_stats: bool = bool(log_stats)
        self.worker_backend = worker_backend
        self.ray_address = ray_address
        self.batch_timeout = batch_timeout

        # Build OmniStage instances in parallel, preserve original order
        def _build_stage(idx_cfg: tuple[int, Any]) -> tuple[int, OmniStage]:
            idx, cfg = idx_cfg
            return idx, OmniStage(cfg)

        with ThreadPoolExecutor(max_workers=min(len(self.stage_configs), max(1, os.cpu_count() or 1))) as executor:
            futures = [executor.submit(_build_stage, (idx, cfg)) for idx, cfg in enumerate(self.stage_configs)]
            results: list[tuple[int, OmniStage]] = []
            for fut in as_completed(futures):
                results.append(fut.result())
        results.sort(key=lambda x: x[0])
        self.stage_list = [st for _, st in results]
        self.output_modalities = [st.final_output_type for st in self.stage_list]
        logger.debug("[Orchestrator] Loaded %d stages", len(self.stage_list))

        # Set up queue backend (Ray or multiprocessing)
        if self.worker_backend == "ray":
            self._queue_cls = get_ray_queue_class()
        else:
            self._ctx = mp.get_context("spawn")
            self._queue_cls = lambda: self._ctx.Queue(maxsize=0)

        self._init_sleep_seconds = max(0, int(init_sleep_seconds))
        self._shm_threshold_bytes = max(0, int(shm_threshold_bytes))
        self._start_stages(model)
        self._wait_for_stages_ready(timeout=init_timeout)

    def _start_stages(self, model: str) -> None:
        """Start all stage processes."""
        if self.worker_backend == "ray":
            self._ray_pg = create_placement_group(
                number_of_stages=len(self.stage_list), address=self.ray_address, strategy="PACK"
            )
        for stage_id, stage in enumerate(self.stage_list):
            in_q = self._queue_cls()
            out_q = self._queue_cls()
            self._stage_in_queues.append(in_q)
            self._stage_out_queues.append(out_q)
            stage.attach_queues(in_q, out_q)
            stage_connectors_config = get_stage_connector_config(
                self.omni_transfer_config,
                stage_id,
            )
            stage.init_stage_worker(
                model,
                shm_threshold_bytes=self._shm_threshold_bytes,
                ctx=self._ctx if self.worker_backend != "ray" else None,
                batch_timeout=self.batch_timeout,
                connectors_config=stage_connectors_config,
                worker_backend=self.worker_backend,
                ray_placement_group=self._ray_pg,
            )
            logger.debug("[Orchestrator] Stage-%s process started", stage_id)
            time.sleep(self._init_sleep_seconds)

    def _wait_for_stages_ready(self, timeout: int = 120) -> None:
        """Wait for all stages to report readiness."""
        deadline = time.time() + max(0, int(timeout))
        num_stages = len(self.stage_list)
        while len(self._stages_ready) < num_stages and time.time() < deadline:
            progressed = False
            for stage_id, stage in enumerate(self.stage_list):
                if stage_id in self._stages_ready:
                    continue
                result = stage.try_collect()
                if result is None:
                    continue
                progressed = True
                if result.get("type") == "stage_ready":
                    self._stages_ready.add(stage_id)
                    logger.info("[Orchestrator] Stage-%s reported ready", stage_id)
            if not progressed:
                time.sleep(0.01)

        if len(self._stages_ready) < num_stages:
            not_ready = sorted(set(range(num_stages)) - set(self._stages_ready))
            logger.warning(
                "[Orchestrator] Initialization timeout: only %s/%s stages are ready; not ready: %s",
                len(self._stages_ready),
                num_stages,
                not_ready,
            )
            try:
                suggestions = [
                    "Verify GPU/device assignment in config (runtime.devices) is correct.",
                    "Check GPU/host memory availability; reduce model or batch size if needed.",
                    "Check model weights path and network reachability (if loading remotely).",
                    "Increase initialization wait time (init_sleep_seconds or call-site timeout).",
                ]
                logger.error(
                    "[Orchestrator] Stage initialization failed, shutting down. Suggestions:\n- %s",
                    "\n- ".join(suggestions),
                )
            except Exception:
                logger.error(
                    "[Orchestrator] Stage initialization failed and an error occurred while logging suggestions",
                )
        else:
            logger.info("[Orchestrator] All stages initialized successfully")

    def generate(self, *args: Any, **kwargs: dict[str, Any]) -> list[OmniRequestOutput]:
        """
        Generate outputs for the given prompts.
        Orchestrates the multi-stage pipeline based on YAML configuration.

        Args:
            *args: Variable length argument list.
                - args[0]: Input prompts for generation.
                - args[1]: Optional list of per-stage parameters.
            **kwargs: Arbitrary keyword arguments.
                - prompt: Input prompts for generation (if not in args).
                - sampling_params_list: Optional list of per-stage parameters.

        Returns:
            List of OmniRequestOutput objects, one for each input prompt.
        """
        prompts = args[0] if args else kwargs.get("prompts")
        sampling_params_list = args[1] if len(args) > 1 else kwargs.get("sampling_params_list")

        if prompts is None:
            if kwargs.get("prompt") is None:
                raise ValueError("prompts is required for generation")
            prompts = kwargs.get("prompt")

        if sampling_params_list is None:
            omni_params_kwargs = {k: v for k, v in kwargs.items() if k not in ["prompts", "sampling_params_list"]}
            per_stage_params: list[Any] = []
            for stage in self.stage_list:
                stage_type = getattr(stage, "stage_type", "llm")
                default_dict = msgspec.to_builtins(getattr(stage, "default_sampling_params", {}))
                merged = {**default_dict, **omni_params_kwargs}
                if stage_type == "diffusion":
                    per_stage_params.append(merged)
                else:
                    per_stage_params.append(SamplingParams(**merged))
            sampling_params_list = per_stage_params

        if len(sampling_params_list) > len(self.stage_list):
            sampling_params_list = sampling_params_list[:len(self.stage_list)]

        return self._run_generation(prompts, sampling_params_list)

    def _run_generation(
        self,
        prompts: PromptType | Sequence[PromptType] | OmniDiffusionRequest | Sequence[OmniDiffusionRequest],
        sampling_params_list: Any | Sequence[Any] | None = None,
    ) -> list[OmniRequestOutput]:
        """Run generation through all stages in the pipeline."""
        logger.debug("[Orchestrator] generate() called")
        if sampling_params_list is None:
            raise ValueError("sampling_params_list is required for pipelined generation")

        if not isinstance(sampling_params_list, (list, tuple)):
            sampling_params_list = [sampling_params_list]
        else:
            sampling_params_list = list(sampling_params_list)

        if len(sampling_params_list) != len(self.stage_list):
            raise ValueError(f"Expected {len(self.stage_list)} sampling params, got {len(sampling_params_list)}")

        if not isinstance(prompts, (list, tuple)):
            request_prompts: list[PromptType] = [prompts]
        else:
            request_prompts = list(prompts)

        final_outputs: list[OmniRequestOutput] = []
        num_stages = len(self.stage_list)

        # Generate unique request IDs
        request_ids: list[str] = [f"{i}_{uuid.uuid4()}" for i in range(len(request_prompts))]
        request_id_to_prompt: dict[str, PromptType] = {rid: p for rid, p in zip(request_ids, request_prompts)}

        _req_start_ts: dict[str, float] = {}
        _wall_start_ts: float = time.time()

        # Determine final stage per request for E2E metrics
        final_stage_id_to_prompt: dict[str, int] = {}
        for rid, prompt in request_id_to_prompt.items():
            prompt_modalities = prompt.get("modalities", None) if isinstance(prompt, dict) else None
            final_stage_id_for_e2e = get_final_stage_id_for_e2e(
                prompt_modalities, self.output_modalities, self.stage_list
            )
            final_stage_id_to_prompt[rid] = final_stage_id_for_e2e

        metrics = OrchestratorMetrics(
            num_stages,
            self._enable_stats,
            _wall_start_ts,
        )

        # === Enhanced Progress Bar with real-time req/sec and tok/sec ===
        total_requests = len(request_prompts)
        pbar = tqdm(
            total=total_requests,
            desc="[Omni] Generating",
            unit="req",
            dynamic_ncols=True,
            mininterval=0.5,
            smoothing=0.1,
        )

        completed_requests = 0

        # Seed stage-0 with all requests
        logger.debug("[Orchestrator] Seeding %d requests into stage-0", len(request_prompts))
        metrics.stage_first_ts[0] = metrics.stage_first_ts[0] or time.time()
        for req_id, prompt in request_id_to_prompt.items():
            sp0 = sampling_params_list[0]
            task = {
                "request_id": req_id,
                "engine_inputs": prompt,
                "sampling_params": sp0,
                "orchestrator_ts": time.time(),
            }
            self.stage_list[0].submit(task)
            _req_start_ts[req_id] = time.time()
            logger.debug("[Orchestrator] Enqueued request %s to stage-0", req_id)

        # Main scheduling loop: poll stages and forward outputs
        while completed_requests < total_requests:
            made_progress = False
            for stage_id, stage in enumerate(self.stage_list):
                result = stage.try_collect()
                if result is None:
                    continue
                made_progress = True
                req_id = result.get("request_id")

                if "error" in result:
                    logger.error("Stage %s error on request %s: %s", stage_id, req_id, result["error"])
                    continue

                if result.get("type") == "stage_ready":
                    time.sleep(0.05)
                    continue

                engine_outputs = _load(result, obj_key="engine_outputs", shm_key="engine_outputs_shm")
                metrics.stage_last_ts[stage_id] = max(metrics.stage_last_ts[stage_id] or 0.0, time.time())

                # Process stage metrics
                try:
                    _m = asdict(result.get("metrics"))
                    if _m is not None:
                        metrics.on_stage_metrics(stage_id, req_id, _m)
                except Exception as e:
                    logger.exception("[Orchestrator] Failed to process metrics for stage %s, req %s: %s", stage_id, req_id, e)

                logger.debug("[Orchestrator] Stage-%s completed request %s", stage_id, req_id)
                stage.set_engine_outputs(engine_outputs)

                # Final stage: collect output
                if getattr(stage, "final_output", False):
                    final_outputs.append(
                        OmniRequestOutput(
                            stage_id=stage_id,
                            final_output_type=stage.final_output_type,
                            request_output=engine_outputs,
                        )
                    )
                    logger.debug("[Orchestrator] Request %s finalized at stage-%s", req_id, stage_id)

                    completed_requests += 1
                    pbar.update(1)

                    elapsed = time.time() - _wall_start_ts
                    if elapsed > 0:
                        rps = completed_requests / elapsed
                        tps = metrics.e2e_total_tokens / elapsed if metrics.e2e_total_tokens > 0 else 0.0
                        pbar.set_postfix({
                            "req/s": f"{rps:.2f}",
                            "tok/s": f"{tps:.1f}",
                            "done": f"{completed_requests}/{total_requests}"
                        })

                    # Record E2E latency only once per request
                    try:
                        if stage_id == final_stage_id_to_prompt[req_id] and str(req_id) not in metrics.e2e_done:
                            metrics.on_finalize_request(
                                stage_id,
                                req_id,
                                engine_outputs,
                                _req_start_ts.get(req_id, _wall_start_ts),
                            )
                    except Exception as e:
                        logger.exception("[Orchestrator] Finalize request error for req %s: %s", req_id, e)

                # Forward to next stage if needed
                next_stage_id = stage_id + 1
                if next_stage_id <= final_stage_id_to_prompt[req_id]:
                    next_stage: OmniStage = self.stage_list[next_stage_id]
                    try:
                        next_inputs = next_stage.process_engine_inputs(self.stage_list, [request_id_to_prompt[req_id]])
                    except Exception as e:
                        logger.exception("[Orchestrator] Process inputs error for req %s: %s", req_id, e)
                        continue

                    sp_next = sampling_params_list[next_stage_id]
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
                        raise RuntimeError(f"[Orchestrator] Failed to send request {req_id} to stage-{next_stage_id}")

                    logger.debug("[Orchestrator] Forwarded request %s to stage-%s", req_id, next_stage_id)

            if not made_progress:
                time.sleep(0.005)

        pbar.close()
        logger.debug("[Orchestrator] All requests completed")

        # Log final performance summary
        try:
            summary = metrics.build_and_log_summary(final_stage_id_to_prompt)
            logger.info("[Summary] %s", pformat(summary, sort_dicts=False))
        except Exception as e:
            logger.exception("[Orchestrator] Failed to build/log summary: %s", e)

        return final_outputs

    def close(self) -> None:
        """Close all stage processes and clean up resources."""
        if self.stage_list:
            for q in self._stage_in_queues:
                try:
                    q.put_nowait(None)
                except Exception as e:
                    logger.warning("[Orchestrator] Failed to send shutdown signal: %s", e)
            for stage in self.stage_list:
                try:
                    stage.stop_stage_worker()
                except Exception as e:
                    logger.warning("[Orchestrator] Failed to stop stage worker: %s", e)
            try_close_ray(self._ray_pg)

    def __del__(self):  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            logger.debug("[Orchestrator] __del__ close() raised", exc_info=True)