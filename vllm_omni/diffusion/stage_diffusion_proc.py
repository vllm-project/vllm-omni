"""Subprocess entry point for the diffusion engine.

StageDiffusionProc runs DiffusionEngine in a child process,
communicating with StageDiffusionClient via ZMQ (PUSH/PULL).
"""

from __future__ import annotations

import asyncio
import signal
import time
from collections.abc import Awaitable
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.process import BaseProcess
from typing import TYPE_CHECKING, Any

import msgspec
import torch
import zmq
import zmq.asyncio
from PIL import Image
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_hf_file_to_dict
from vllm.utils.network_utils import get_open_zmq_ipc_path, zmq_socket_ctx
from vllm.utils.system_utils import get_mp_context
from vllm.v1.utils import shutdown

from vllm_omni.diffusion.data import DiffusionRequestAbortedError, TransformerConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.distributed.omni_connectors.utils.serialization import (
    OmniMsgpackDecoder,
    OmniMsgpackEncoder,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)

_HANDSHAKE_POLL_TIMEOUT_S = 600


class StageDiffusionProc:
    """Subprocess entry point for diffusion inference.

    Manages DiffusionEngine lifecycle, async request processing,
    and ZMQ-based communication with StageDiffusionClient.
    """

    def __init__(self, model: str, od_config: OmniDiffusionConfig) -> None:
        self._model = model
        self._od_config = od_config
        self._engine: DiffusionEngine | None = None
        self._utility_executor: ThreadPoolExecutor | None = None
        self._closed = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Enrich config, create DiffusionEngine, and allocate helper workers."""
        self._enrich_config()
        self._engine = DiffusionEngine.make_engine(self._od_config)
        self._utility_executor = ThreadPoolExecutor(max_workers=2)
        logger.info("StageDiffusionProc initialized with model: %s", self._model)

    def _enrich_config(self) -> None:
        """Load model metadata from HuggingFace and populate od_config fields.

        Diffusers-style models expose ``model_index.json`` with ``_class_name``.
        Non-diffusers models (e.g. Bagel, NextStep) only have ``config.json``,
        so we fall back to reading that and mapping model_type manually.
        """
        od_config = self._od_config

        try:
            config_dict = get_hf_file_to_dict("model_index.json", od_config.model)
            if config_dict is not None:
                if od_config.model_class_name is None:
                    od_config.model_class_name = config_dict.get("_class_name", None)
                od_config.update_multimodal_support()

                tf_config_dict = get_hf_file_to_dict("transformer/config.json", od_config.model)
                od_config.tf_model_config = TransformerConfig.from_dict(tf_config_dict)
            else:
                raise FileNotFoundError("model_index.json not found")
        except (AttributeError, OSError, ValueError, FileNotFoundError):
            cfg = get_hf_file_to_dict("config.json", od_config.model)
            if cfg is None:
                raise ValueError(f"Could not find config.json or model_index.json for model {od_config.model}")

            od_config.tf_model_config = TransformerConfig.from_dict(cfg)
            model_type = cfg.get("model_type")
            architectures = cfg.get("architectures") or []

            if model_type == "bagel" or "BagelForConditionalGeneration" in architectures:
                od_config.model_class_name = "BagelPipeline"
                od_config.tf_model_config = TransformerConfig()
                od_config.update_multimodal_support()
            elif model_type == "nextstep":
                if od_config.model_class_name is None:
                    od_config.model_class_name = "NextStep11Pipeline"
                od_config.tf_model_config = TransformerConfig()
                od_config.update_multimodal_support()
            elif architectures and len(architectures) == 1:
                od_config.model_class_name = architectures[0]
            else:
                raise

    # ------------------------------------------------------------------
    # Request processing
    # ------------------------------------------------------------------

    def _reconstruct_sampling_params(self, sampling_params_dict: dict) -> OmniDiffusionSamplingParams:
        """Reconstruct OmniDiffusionSamplingParams from a dict, handling LoRA."""
        lora_req = sampling_params_dict.get("lora_request")
        if lora_req is not None:
            from vllm.lora.request import LoRARequest

            if not isinstance(lora_req, LoRARequest):
                sampling_params_dict["lora_request"] = msgspec.convert(lora_req, LoRARequest)

        return OmniDiffusionSamplingParams(**sampling_params_dict)

    async def _run_on_utility_executor(self, fn, *args):
        """Run CPU-side helper work off the asyncio event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._utility_executor, fn, *args)

    async def _execute_step_request_async(
        self,
        request: OmniDiffusionRequest,
    ) -> list[OmniRequestOutput]:
        """Execute one request through the engine core loop.

        This mirrors ``DiffusionEngine.step()`` without serializing the whole
        step call on a single worker thread:

        1. preprocess on the utility executor
        2. submit the prepared request to ``DiffusionEngine``'s core loop
        3. await the terminal request future
        4. materialize final outputs on the utility executor
        """
        diffusion_engine_start_time = time.perf_counter()
        request, preprocess_time = await self._run_on_utility_executor(
            self._engine._prepare_step_request,
            request,
        )

        exec_start_time = time.perf_counter()
        future = self._engine.submit_request(request)
        output = await asyncio.wrap_future(future)
        exec_total_time = time.perf_counter() - exec_start_time

        return await self._run_on_utility_executor(
            self._engine._materialize_step_outputs,
            request,
            output,
            preprocess_time,
            exec_total_time,
            diffusion_engine_start_time,
        )

    async def _await_engine_rpc(
        self,
        method: str,
        timeout: float | None,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
        unique_reply_rank: int | None = None,
    ) -> Any:
        """Await one queue-backed engine RPC from asyncio code."""
        future = self._engine.submit_rpc(
            method=method,
            timeout=timeout,
            args=args,
            kwargs=kwargs or {},
            unique_reply_rank=unique_reply_rank,
        )
        try:
            return await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            future.cancel()
            raise

    async def _process_request(
        self,
        request_id: str,
        prompt: Any,
        sampling_params_dict: dict,
        kv_sender_info: dict[str, Any] | None = None,
    ) -> OmniRequestOutput:
        """Build a diffusion request and run it through the engine core loop."""
        sampling_params = self._reconstruct_sampling_params(sampling_params_dict)

        request = OmniDiffusionRequest(
            prompts=[prompt],
            sampling_params=sampling_params,
            request_ids=[request_id],
            request_id=request_id,
            kv_sender_info=kv_sender_info,
        )

        results = await self._execute_step_request_async(request)
        result = results[0]
        if not result.request_id:
            result.request_id = request_id
        return result

    async def _process_batch_request(
        self,
        request_id: str,
        prompts: list[Any],
        sampling_params_dict: dict,
        kv_sender_info: dict[str, Any] | None = None,
    ) -> OmniRequestOutput:
        """Build a batched diffusion request and run one engine request.

        All prompts are processed in a single engine submission. The per-prompt
        results are merged into one :class:`OmniRequestOutput` whose
        ``images`` list contains every generated image, matching the
        contract expected by the orchestrator and tests.
        """
        sampling_params = self._reconstruct_sampling_params(sampling_params_dict)

        request = OmniDiffusionRequest(
            prompts=prompts,
            sampling_params=sampling_params,
            request_ids=[f"{request_id}-{i}" for i in range(len(prompts))],
            request_id=request_id,
            kv_sender_info=kv_sender_info,
        )

        results = await self._execute_step_request_async(request)

        # Merge per-prompt results into a single combined output.
        all_images: list = []
        merged_mm: dict[str, Any] = {}
        merged_metrics: dict[str, Any] = {}
        merged_durations: dict[str, float] = {}
        merged_custom: dict[str, Any] = {}
        peak_mem = 0.0
        latents = None
        trajectory_latents: list[torch.Tensor] | None = None
        trajectory_timesteps: list[torch.Tensor] | None = None
        trajectory_log_probs: torch.Tensor | None = None
        trajectory_decoded: list[Image.Image] | None = None
        final_output_type = "image"

        for r in results:
            all_images.extend(r.images)
            # The current diffusion engine emits step-scoped metrics and stage
            # timings that are shared by every per-prompt result produced from
            # the same engine submission. ``dict.update()`` is therefore safe
            # here because identical keys are expected to carry identical
            # values across all members of the batch.
            merged_mm.update(r._multimodal_output)
            merged_metrics.update(r.metrics)
            merged_durations.update(r.stage_durations)
            merged_custom.update(r._custom_output)
            peak_mem = max(peak_mem, r.peak_memory_mb)
            if latents is None and r.latents is not None:
                latents = r.latents
            if trajectory_latents is None:
                trajectory_latents = r.trajectory_latents
            if trajectory_timesteps is None:
                trajectory_timesteps = r.trajectory_timesteps
            if trajectory_log_probs is None:
                trajectory_log_probs = r.trajectory_log_probs
            if trajectory_decoded is None:
                trajectory_decoded = r.trajectory_decoded
            if r.final_output_type != "image":
                final_output_type = r.final_output_type

        return OmniRequestOutput.from_diffusion(
            request_id=request_id,
            images=all_images,
            prompt=prompts[0] if len(prompts) == 1 else None,
            metrics=merged_metrics,
            latents=latents,
            trajectory_latents=trajectory_latents,
            trajectory_timesteps=trajectory_timesteps,
            trajectory_log_probs=trajectory_log_probs,
            trajectory_decoded=trajectory_decoded,
            custom_output=merged_custom or None,
            multimodal_output=merged_mm or None,
            final_output_type=final_output_type,
            stage_durations=merged_durations,
            peak_memory_mb=peak_mem,
        )

    # ------------------------------------------------------------------
    # Collective RPC dispatch
    # ------------------------------------------------------------------

    async def _handle_collective_rpc(
        self,
        method: str,
        timeout: float | None,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        """Dispatch collective RPC calls to DiffusionEngine.

        LoRA methods remap arguments and post-process results to match
        the contract that ``AsyncOmni`` provides.
        """
        if method == "profile":
            is_start = args[0] if args else True
            profile_prefix = args[1] if len(args) > 1 else None
            action = "start" if is_start else "stop"
            try:
                await self._await_engine_rpc(
                    method="profile",
                    timeout=timeout,
                    args=(is_start, profile_prefix),
                )
            except Exception as e:
                logger.error("Failed to %s profiling on workers", action, exc_info=True)
                if is_start:
                    raise RuntimeError(f"Could not {action} profiler: {e}") from e
            return None

        if method == "add_lora":
            # Reconstruct LoRARequest after IPC if needed.
            lora_request = args[0] if args else kwargs.get("lora_request")
            if lora_request is not None:
                from vllm.lora.request import LoRARequest

                if not isinstance(lora_request, LoRARequest):
                    lora_request = msgspec.convert(lora_request, LoRARequest)
            results = await self._await_engine_rpc(
                method="add_lora",
                timeout=timeout,
                args=(),
                kwargs={"lora_request": lora_request},
            )
            return all(results) if isinstance(results, list) else results

        if method == "remove_lora":
            results = await self._await_engine_rpc(
                method="remove_lora",
                timeout=timeout,
                args=args,
                kwargs=kwargs or {},
            )
            return all(results) if isinstance(results, list) else results

        if method == "list_loras":
            results = await self._await_engine_rpc(
                method="list_loras",
                timeout=timeout,
                args=(),
                kwargs={},
            )
            if not isinstance(results, list):
                return results or []
            merged: set[int] = set()
            for part in results:
                merged.update(part or [])
            return sorted(merged)

        if method == "pin_lora":
            lora_id = args[0] if args else kwargs.get("adapter_id")
            results = await self._await_engine_rpc(
                method="pin_lora",
                timeout=timeout,
                args=(),
                kwargs={"adapter_id": lora_id},
            )
            return all(results) if isinstance(results, list) else results

        # Fall back to DiffusionEngine.collective_rpc for all other methods
        # (e.g. worker extension RPCs like "test_extension_name").
        return await self._await_engine_rpc(
            method=method,
            timeout=timeout,
            args=args,
            kwargs=kwargs or {},
        )

    # ------------------------------------------------------------------
    # ZMQ event loop
    # ------------------------------------------------------------------

    async def run_loop(
        self,
        request_address: str,
        response_address: str,
    ) -> None:
        """Async event loop handling ZMQ messages from StageDiffusionClient."""
        ctx = zmq.asyncio.Context()

        request_socket = ctx.socket(zmq.PULL)
        request_socket.bind(request_address)

        response_socket = ctx.socket(zmq.PUSH)
        response_socket.bind(response_address)

        encoder = OmniMsgpackEncoder()
        decoder = OmniMsgpackDecoder()

        tasks: dict[str, asyncio.Task] = {}

        async def _dispatch_and_send(
            request_id: str,
            processor: Awaitable[Any],
            *,
            failure_log_label: str,
        ) -> None:
            """Run one request coroutine and send either a result or error."""
            try:
                result = await processor
                await response_socket.send(encoder.encode({"type": "result", "output": result}))
            except DiffusionRequestAbortedError as e:
                logger.info(
                    "request_id: %s aborted: %s",
                    request_id,
                    str(e),
                )
            except Exception as e:
                logger.exception("%s %s failed: %s", failure_log_label, request_id, e)
                await response_socket.send(
                    encoder.encode(
                        {
                            "type": "error",
                            "request_id": request_id,
                            "error": str(e),
                        }
                    )
                )
            finally:
                tasks.pop(request_id, None)

        async def _dispatch_request(
            request_id: str,
            prompt: Any,
            sampling_params_dict: dict,
            kv_sender_info: dict[str, Any] | None = None,
        ) -> None:
            """Process a single diffusion request and send the response."""
            await _dispatch_and_send(
                request_id,
                self._process_request(
                    request_id,
                    prompt,
                    sampling_params_dict,
                    kv_sender_info=kv_sender_info,
                ),
                failure_log_label="Diffusion request",
            )

        async def _dispatch_batch_request(
            request_id: str,
            prompts: list[Any],
            sampling_params_dict: dict,
            kv_sender_info: dict[str, Any] | None = None,
        ) -> None:
            """Process a batched diffusion request and send the response."""
            await _dispatch_and_send(
                request_id,
                self._process_batch_request(
                    request_id,
                    prompts,
                    sampling_params_dict,
                    kv_sender_info=kv_sender_info,
                ),
                failure_log_label="Batch diffusion request",
            )

        try:
            while True:
                raw = await request_socket.recv()
                msg = decoder.decode(raw)
                msg_type = msg.get("type")

                if msg_type == "add_request":
                    request_id = msg["request_id"]
                    task = asyncio.create_task(
                        _dispatch_request(
                            request_id,
                            msg["prompt"],
                            msg["sampling_params"],
                            msg.get("kv_sender_info"),
                        )
                    )
                    tasks[request_id] = task

                elif msg_type == "add_batch_request":
                    request_id = msg["request_id"]
                    task = asyncio.create_task(
                        _dispatch_batch_request(
                            request_id,
                            msg["prompts"],
                            msg["sampling_params"],
                            msg.get("kv_sender_info"),
                        )
                    )
                    tasks[request_id] = task

                elif msg_type == "abort":
                    for rid in msg.get("request_ids", []):
                        task = tasks.pop(rid, None)
                        if task:
                            task.cancel()
                        self._engine.abort(rid)

                elif msg_type == "collective_rpc":
                    rpc_id = msg["rpc_id"]
                    try:
                        result = await self._handle_collective_rpc(
                            msg["method"],
                            msg.get("timeout"),
                            tuple(msg.get("args", ())),
                            msg.get("kwargs", {}),
                        )
                        await response_socket.send(
                            encoder.encode(
                                {
                                    "type": "rpc_result",
                                    "rpc_id": rpc_id,
                                    "result": result,
                                }
                            )
                        )
                    except Exception as e:
                        logger.exception("Collective RPC %s failed: %s", msg["method"], e)
                        await response_socket.send(
                            encoder.encode(
                                {
                                    "type": "error",
                                    "rpc_id": rpc_id,
                                    "error": str(e),
                                }
                            )
                        )

                elif msg_type == "shutdown":
                    break

        finally:
            for task in tasks.values():
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks.values(), return_exceptions=True)

            request_socket.close()
            response_socket.close()
            ctx.term()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release engine and helper thread-pool resources."""
        if self._closed:
            return
        self._closed = True

        if self._engine is not None:
            try:
                self._engine.close()
            except Exception as e:
                logger.warning("Error closing diffusion engine: %s", e)

        if self._utility_executor is not None:
            try:
                self._utility_executor.shutdown(wait=False)
            except Exception as e:
                logger.warning("Error shutting down executor: %s", e)

    # ------------------------------------------------------------------
    # Subprocess entry point
    # ------------------------------------------------------------------

    @classmethod
    def run_diffusion_proc(
        cls,
        model: str,
        od_config: OmniDiffusionConfig,
        handshake_address: str,
        request_address: str,
        response_address: str,
    ) -> None:
        """Entry point for the diffusion subprocess."""
        shutdown_requested = False

        def signal_handler(signum: int, frame: Any) -> None:
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        proc = cls(model, od_config)
        try:
            proc.initialize()

            # Send READY via handshake socket
            handshake_ctx = zmq.Context()
            handshake_socket = handshake_ctx.socket(zmq.DEALER)
            handshake_socket.connect(handshake_address)
            handshake_socket.send(msgspec.msgpack.encode({"status": "READY"}))
            handshake_socket.close()
            handshake_ctx.term()

            # Run async event loop
            asyncio.run(proc.run_loop(request_address, response_address))

        except SystemExit:
            logger.debug("StageDiffusionProc exiting.")
            raise
        except Exception:
            logger.exception("StageDiffusionProc encountered a fatal error.")
            raise
        finally:
            proc.close()


# -- Free functions for backward compatibility with StageDiffusionClient ------


def spawn_diffusion_proc(
    model: str,
    od_config: OmniDiffusionConfig,
    handshake_address: str | None = None,
    request_address: str | None = None,
    response_address: str | None = None,
) -> tuple[BaseProcess, str, str, str]:
    """Spawn a StageDiffusionProc subprocess.

    Returns ``(proc, handshake_address, request_address, response_address)``.
    """
    handshake_address = handshake_address or get_open_zmq_ipc_path()
    request_address = request_address or get_open_zmq_ipc_path()
    response_address = response_address or get_open_zmq_ipc_path()

    ctx = get_mp_context()
    proc = ctx.Process(
        target=StageDiffusionProc.run_diffusion_proc,
        name="StageDiffusionProc",
        kwargs={
            "model": model,
            "od_config": od_config,
            "handshake_address": handshake_address,
            "request_address": request_address,
            "response_address": response_address,
        },
    )
    proc.start()
    # Wait for the process to become alive before returning.
    deadline = time.monotonic() + 10
    while not proc.is_alive():
        if proc.exitcode is not None:
            raise RuntimeError(f"StageDiffusionProc failed to start (exit code {proc.exitcode})")
        if time.monotonic() > deadline:
            raise TimeoutError("StageDiffusionProc did not become alive within 10s")
        time.sleep(0.01)
    return proc, handshake_address, request_address, response_address


def complete_diffusion_handshake(
    proc: BaseProcess,
    handshake_address: str,
) -> None:
    """Wait for the diffusion subprocess to signal READY.

    On failure the process is terminated before re-raising.
    """
    try:
        _perform_diffusion_handshake(proc, handshake_address)
    except Exception:
        shutdown([proc])
        raise


def _perform_diffusion_handshake(
    proc: BaseProcess,
    handshake_address: str,
) -> None:
    """Run the handshake with the diffusion subprocess."""
    with zmq_socket_ctx(handshake_address, zmq.ROUTER, bind=True) as handshake_socket:
        poller = zmq.Poller()
        poller.register(handshake_socket, zmq.POLLIN)
        poller.register(proc.sentinel, zmq.POLLIN)

        timeout_ms = _HANDSHAKE_POLL_TIMEOUT_S * 1000
        while True:
            events = dict(poller.poll(timeout=timeout_ms))
            if not events:
                raise TimeoutError("Timed out waiting for READY from StageDiffusionProc")
            if handshake_socket in events:
                identity, raw = handshake_socket.recv_multipart()
                msg = msgspec.msgpack.decode(raw)
                if msg.get("status") == "READY":
                    return
                raise RuntimeError(f"Expected READY, got: {msg}")
            if proc.exitcode is not None:
                raise RuntimeError(f"StageDiffusionProc died during handshake (exit code {proc.exitcode})")
