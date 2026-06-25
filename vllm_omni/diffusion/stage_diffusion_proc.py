"""Subprocess entry point for the diffusion engine.

StageDiffusionProc runs DiffusionEngine in a child process,
communicating with StageDiffusionClient via ZMQ (PUSH/PULL).
"""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing.connection
import signal
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import msgspec
import torch
import zmq
import zmq.asyncio
from PIL import Image
from vllm.logger import init_logger
from vllm.utils.network_utils import get_open_zmq_ipc_path, zmq_socket_ctx
from vllm.utils.system_utils import get_mp_context
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.utils import CoreEngine, EngineZmqAddresses, wait_for_engine_startup
from vllm.v1.utils import shutdown

from vllm_omni.diffusion.data import DiffusionRequestAbortedError
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.distributed.omni_connectors.utils.serialization import (
    OmniMsgpackDecoder,
    OmniMsgpackEncoder,
)
from vllm_omni.distributed.omni_coordinator import OmniCoordClientForStage
from vllm_omni.engine.stage_init_utils import set_death_signal
from vllm_omni.errors import client_error_metadata
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)


_SIGNAL_EXIT_BASE = 128


def _signal_exit_code(signum: int) -> int:
    """Return the conventional process exit code for signal-driven exits."""
    return _SIGNAL_EXIT_BASE + signum


class StageDiffusionProc:
    """Subprocess entry point for diffusion inference.

    Manages DiffusionEngine lifecycle, async request processing,
    and ZMQ-based communication with StageDiffusionClient.
    """

    DIFFUSION_PROC_DEAD = b"DIFFUSION_PROC_DEAD"

    def __init__(self, model: str, od_config: OmniDiffusionConfig) -> None:
        self._model = model
        self._od_config = od_config
        self._engine: DiffusionEngine | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._closed = False
        # Set by ``run_loop`` to the live dispatch task dict so
        # :attr:`queue_length` can report in-flight requests for the
        # OmniCoordinator heartbeat hook.
        self._active_tasks: dict[str, asyncio.Task] | None = None
        # Set when a request-handler detects that the engine's multiproc
        # executor has died (e.g. a worker process crashed and the executor's
        # monitor thread closed it). Once set, ``run_loop`` breaks out so the
        # outer ``except``/``finally`` can send ``DIFFUSION_PROC_DEAD`` and
        # ``ReplicaStatus.DOWN``, then the subprocess exits non-zero. Without
        # this, the run_loop would swallow per-request errors and keep
        # serving 500s indefinitely while heartbeats still report UP.
        self._fatal_event: asyncio.Event | None = None

    @property
    def queue_length(self) -> int:
        """Number of in-flight diffusion requests.

        Returns 0 before :meth:`run_loop` starts and after it exits.
        """
        tasks = self._active_tasks
        return 0 if tasks is None else len(tasks)

    def _is_executor_dead(self) -> bool:
        """True iff the multiproc executor has been closed or marked failed.

        Detects the "workers died but the diffusion proc is still pulling
        requests" case: ``MultiprocDiffusionExecutor`` sets ``_closed = True``
        and ``is_failed = True`` from its worker-monitor thread the moment any
        worker process exits; every subsequent ``execute_request`` /
        ``collective_rpc`` then raises ``RuntimeError("DiffusionExecutor is
        closed.")`` inside the engine. Callers in ``run_loop`` use this to
        decide whether a per-request failure is recoverable or fatal.
        """
        if self._engine is None:
            return False
        executor = getattr(self._engine, "executor", None)
        if executor is None:
            return False
        return bool(getattr(executor, "_closed", False) or getattr(executor, "is_failed", False))

    def _signal_fatal_engine_failure(self, reason: str) -> None:
        """Idempotently signal ``run_loop`` to tear down on a fatal engine error."""
        if self._fatal_event is None or self._fatal_event.is_set():
            return
        logger.error(
            "[StageDiffusionProc] fatal engine failure detected (%s); "
            "signaling run_loop to send DIFFUSION_PROC_DEAD and exit.",
            reason,
        )
        self._fatal_event.set()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Enrich config, create DiffusionEngine and thread pool."""
        self._enrich_config()
        self._engine = DiffusionEngine.make_engine(self._od_config)
        self._executor = ThreadPoolExecutor(max_workers=1)
        logger.info("StageDiffusionProc initialized with model: %s", self._model)

    def _enrich_config(self) -> None:
        """Load model metadata from HuggingFace and populate od_config fields."""
        self._od_config.enrich_config()

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

    async def _process_request(
        self,
        request_id: str,
        prompt: Any,
        sampling_params_dict: dict,
        kv_sender_info: dict[str, Any] | None = None,
    ) -> OmniRequestOutput:
        """Build a diffusion request and run DiffusionEngine.step()."""
        sampling_params = self._reconstruct_sampling_params(sampling_params_dict)

        request = OmniDiffusionRequest(
            prompts=[prompt],
            sampling_params=sampling_params,
            request_id=request_id,
            kv_sender_info=kv_sender_info,
        )

        results = await self._engine.step(request)
        result = results[0]
        if not result.request_id:
            result.request_id = request_id
        return result

    async def _process_streaming_request(
        self,
        request_id: str,
        prompt: Any,
        sampling_params_dict: dict,
        kv_sender_info: dict[str, Any] | None = None,
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Process a streaming diffusion request and yield the results from DiffusionEngine.step_streaming()."""
        sampling_params = self._reconstruct_sampling_params(sampling_params_dict)

        request = OmniDiffusionRequest(
            prompts=[prompt],
            sampling_params=sampling_params,
            request_id=request_id,
            kv_sender_info=kv_sender_info,
        )

        async for results in self._engine.step_streaming(request):  # pyright: ignore[reportOptionalMemberAccess]
            result = results[0]
            if not result.request_id:
                result.request_id = request_id
            yield result

    async def _process_batch_request(
        self,
        request_id: str,
        prompts: list[Any],
        sampling_params_dict: dict,
        kv_sender_info: dict[str, Any] | None = None,
    ) -> OmniRequestOutput:
        """Build a batched diffusion request and run DiffusionEngine.step().

        All prompts are processed in a single step() call.  The per-prompt
        results are merged into one :class:`OmniRequestOutput` whose
        ``images`` list contains every generated image, matching the
        contract expected by the orchestrator and tests.
        """
        if self._od_config.streaming_output:
            raise NotImplementedError("Streaming output is not supported for batched requests")

        sampling_params = self._reconstruct_sampling_params(sampling_params_dict)

        request = OmniDiffusionRequest(
            prompts=prompts,
            sampling_params=sampling_params,
            request_id=request_id,
            kv_sender_info=kv_sender_info,
        )

        results = await self._engine.step(request)

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
        loop = asyncio.get_running_loop()

        if method == "profile":
            is_start = args[0] if args else True
            profile_prefix = args[1] if len(args) > 1 else None
            return await loop.run_in_executor(
                self._executor,
                self._engine.profile,
                is_start,
                profile_prefix,
            )

        if method == "add_lora":
            # Reconstruct LoRARequest after IPC if needed.
            lora_request = args[0] if args else kwargs.get("lora_request")
            if lora_request is not None:
                from vllm.lora.request import LoRARequest

                if not isinstance(lora_request, LoRARequest):
                    lora_request = msgspec.convert(lora_request, LoRARequest)
            results = await loop.run_in_executor(
                self._executor,
                self._engine.collective_rpc,
                "add_lora",
                timeout,
                (),
                {"lora_request": lora_request},
                None,
            )
            return all(results) if isinstance(results, list) else results

        if method == "remove_lora":
            results = await loop.run_in_executor(
                self._executor,
                self._engine.collective_rpc,
                "remove_lora",
                timeout,
                args,
                kwargs or {},
                None,
            )
            return all(results) if isinstance(results, list) else results

        if method == "list_loras":
            results = await loop.run_in_executor(
                self._executor,
                self._engine.collective_rpc,
                "list_loras",
                timeout,
                (),
                {},
                None,
            )
            if not isinstance(results, list):
                return results or []
            merged: set[int] = set()
            for part in results:
                merged.update(part or [])
            return sorted(merged)

        if method == "pin_lora":
            lora_id = args[0] if args else kwargs.get("adapter_id")
            results = await loop.run_in_executor(
                self._executor,
                self._engine.collective_rpc,
                "pin_lora",
                timeout,
                (),
                {"adapter_id": lora_id},
                None,
            )
            return all(results) if isinstance(results, list) else results

        # Fall back to DiffusionEngine.collective_rpc for all other methods
        # (e.g. worker extension RPCs like "test_extension_name").
        return await loop.run_in_executor(
            self._executor,
            self._engine.collective_rpc,
            method,
            timeout,
            args,
            kwargs or {},
            None,
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
        request_socket.connect(request_address)

        response_socket = ctx.socket(zmq.PUSH)
        response_socket.connect(response_address)

        encoder = OmniMsgpackEncoder()
        decoder = OmniMsgpackDecoder()

        tasks: dict[str, asyncio.Task] = {}
        # Expose the live task dict so :attr:`queue_length` (used by the
        # OmniCoordinator heartbeat hook) can read the in-flight count.
        self._active_tasks = tasks
        # Wakes the main recv loop when a request-handler detects a fatal
        # engine failure so we tear down promptly instead of swallowing
        # "DiffusionExecutor is closed" on every subsequent request.
        fatal_event = asyncio.Event()
        self._fatal_event = fatal_event

        async def _dispatch_request(
            request_id: str,
            prompt: Any,
            sampling_params_dict: dict,
            kv_sender_info: dict[str, Any] | None = None,
        ) -> None:
            """Process a single diffusion request and send the response."""
            try:
                if not self._od_config.streaming_output:
                    result = await self._process_request(
                        request_id,
                        prompt,
                        sampling_params_dict,
                        kv_sender_info=kv_sender_info,
                    )
                    await response_socket.send(encoder.encode({"type": "result", "output": result}))
                else:
                    async for result in self._process_streaming_request(
                        request_id,
                        prompt,
                        sampling_params_dict,
                        kv_sender_info=kv_sender_info,
                    ):
                        await response_socket.send(encoder.encode({"type": "result", "output": result}))
            except DiffusionRequestAbortedError as e:
                logger.info(
                    "request_id: %s aborted: %s",
                    request_id,
                    str(e),
                )
            except Exception as e:
                logger.exception("Diffusion request %s failed: %s", request_id, e)
                status_code, error_type = client_error_metadata(e)
                await response_socket.send(
                    encoder.encode(
                        {
                            "type": "error",
                            "request_id": request_id,
                            "error": str(e),
                            "status_code": status_code,
                            "error_type": error_type,
                        }
                    )
                )
                # Per-request errors are usually recoverable, but a closed
                # executor means every future request will get the same
                # "DiffusionExecutor is closed" error. Signal the main loop
                # to send DIFFUSION_PROC_DEAD and exit so the head's hub
                # demotes this replica instead of waiting on the heartbeat
                # timeout (~30 s by default).
                if self._is_executor_dead():
                    self._signal_fatal_engine_failure(f"add_request {request_id}: {e!s}")
            finally:
                tasks.pop(request_id, None)

        try:
            while True:
                # Await recv and fatal_event concurrently so the loop wakes
                # up immediately when a per-request handler signals a fatal
                # engine failure — even if no fresh ZMQ frame arrives.
                recv_task: asyncio.Task = asyncio.ensure_future(request_socket.recv())
                fatal_task: asyncio.Task = asyncio.ensure_future(fatal_event.wait())
                try:
                    done, pending = await asyncio.wait(
                        [recv_task, fatal_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    for waiter in (recv_task, fatal_task):
                        if not waiter.done():
                            waiter.cancel()
                            with contextlib.suppress(asyncio.CancelledError, Exception):
                                await waiter
                if fatal_event.is_set():
                    raise RuntimeError(
                        "StageDiffusionProc executor reported permanent failure; tearing down the diffusion subprocess."
                    )
                raw = recv_task.result()
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

                    async def _dispatch_batch(
                        rid: str,
                        prompts: list,
                        sp_dict: dict,
                        kv_sender_info: dict[str, Any] | None = None,
                    ) -> None:
                        try:
                            result = await self._process_batch_request(
                                rid,
                                prompts,
                                sp_dict,
                                kv_sender_info=kv_sender_info,
                            )
                            await response_socket.send(encoder.encode({"type": "result", "output": result}))
                        except DiffusionRequestAbortedError as e:
                            logger.info(
                                "request_id: %s aborted: %s",
                                rid,
                                str(e),
                            )
                        except Exception as e:
                            logger.exception("Batch diffusion request %s failed: %s", rid, e)
                            status_code, error_type = client_error_metadata(e)
                            await response_socket.send(
                                encoder.encode(
                                    {
                                        "type": "error",
                                        "request_id": rid,
                                        "error": str(e),
                                        "status_code": status_code,
                                        "error_type": error_type,
                                    }
                                )
                            )
                            # Same rationale as the single-request path: a
                            # closed executor turns every subsequent batch
                            # into a 500, so escalate now.
                            if self._is_executor_dead():
                                self._signal_fatal_engine_failure(f"add_batch_request {rid}: {e!s}")
                        finally:
                            tasks.pop(rid, None)

                    task = asyncio.create_task(
                        _dispatch_batch(
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
                        # Collective RPCs run through the same multiproc
                        # executor — a closed executor means every future
                        # RPC fails the same way, so tear down promptly.
                        if self._is_executor_dead():
                            self._signal_fatal_engine_failure(
                                f"collective_rpc {msg['method']} (rpc_id={rpc_id}): {e!s}"
                            )

                elif msg_type == "shutdown":
                    break

        except Exception:
            # Send the death sentinel so the client can detect the
            # fatal failure promptly (mirrors EngineCoreProc._send_engine_dead).
            try:
                response_socket.setsockopt(zmq.LINGER, 4000)
                await response_socket.send(StageDiffusionProc.DIFFUSION_PROC_DEAD)
            except Exception:
                logger.warning("Failed to send DIFFUSION_PROC_DEAD sentinel to client.")
            raise

        finally:
            for task in tasks.values():
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks.values(), return_exceptions=True)

            self._active_tasks = None
            self._fatal_event = None
            request_socket.close()
            response_socket.close()
            ctx.term()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release engine and thread pool resources."""
        if self._closed:
            return
        self._closed = True

        if self._engine is not None:
            try:
                self._engine.close()
            except Exception as e:
                logger.warning("Error closing diffusion engine: %s", e)

        if self._executor is not None:
            try:
                self._executor.shutdown(wait=False)
            except Exception as e:
                logger.warning("Error shutting down executor: %s", e)

    # ------------------------------------------------------------------
    # Subprocess entry point
    # ------------------------------------------------------------------

    @staticmethod
    def _open_startup_handshake(
        handshake_address: str,
        *,
        local_client: bool,
        headless: bool,
    ) -> tuple[zmq.Context, zmq.Socket, EngineZmqAddresses]:
        ctx = zmq.Context()
        socket = ctx.socket(zmq.DEALER)
        socket.setsockopt(zmq.IDENTITY, (0).to_bytes(2, "little"))
        socket.connect(handshake_address)
        addresses = EngineCoreProc.startup_handshake(
            socket,
            local_client=local_client,
            headless=headless,
            parallel_config=None,
        )
        return ctx, socket, addresses

    @staticmethod
    def _send_startup_ready(
        handshake_socket: zmq.Socket,
        *,
        local_client: bool,
        headless: bool,
    ) -> None:
        handshake_socket.send(
            msgspec.msgpack.encode(
                {
                    "status": "READY",
                    "local": local_client,
                    "headless": headless,
                }
            )
        )

    @classmethod
    def run_diffusion_proc(
        cls,
        model: str,
        od_config: OmniDiffusionConfig,
        handshake_address: str,
        *,
        local_client: bool,
        headless: bool,
        omni_coordinator_address: str | None = None,
        omni_stage_id: int | None = None,
        omni_replica_id: int = 0,
    ) -> None:
        """Entry point for the diffusion subprocess.

        Omni-specific kwargs (mirroring :meth:`StageEngineCoreProc.run_stage_core`):
          - ``omni_coordinator_address``: ROUTER address of the head-side
            OmniCoordinator. When set, a :class:`OmniCoordClientForStage`
            reports the diffusion replica's status + queue length.
          - ``omni_stage_id``: logical stage id; required when
            ``omni_coordinator_address`` is set.
          - ``omni_replica_id``: cluster-unique replica id within the
            stage (logging / metrics only).
        """
        shutdown_requested = False

        set_death_signal(signal.SIGTERM)

        def signal_handler(signum: int, frame: Any) -> None:
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit(_signal_exit_code(signum))

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        proc = cls(model, od_config)
        coord_client: OmniCoordClientForStage | None = None
        handshake_ctx: zmq.Context | None = None
        handshake_socket: zmq.Socket | None = None
        try:
            handshake_ctx, handshake_socket, addresses = cls._open_startup_handshake(
                handshake_address,
                local_client=local_client,
                headless=headless,
            )
            request_address = addresses.inputs[0]
            response_address = addresses.outputs[0]

            proc.initialize()

            cls._send_startup_ready(
                handshake_socket,
                local_client=local_client,
                headless=headless,
            )
            handshake_socket.close()
            handshake_ctx.term()
            handshake_socket = None
            handshake_ctx = None

            # Wire OmniCoordClientForStage *after* READY. The address pair is
            # owned by the frontend client; this proc connects to it as the
            # backend runtime.
            if omni_coordinator_address is not None:
                if omni_stage_id is None:
                    raise ValueError("omni_stage_id must be provided when omni_coordinator_address is set")
                coord_client = OmniCoordClientForStage(
                    coord_zmq_addr=omni_coordinator_address,
                    input_addr=request_address,
                    output_addr=response_address,
                    stage_id=int(omni_stage_id),
                )

                def _refresh_queue_length() -> None:
                    coord_client._queue_length = proc.queue_length  # type: ignore[union-attr]

                coord_client._on_heartbeat = _refresh_queue_length

                logger.info(
                    "StageDiffusionProc registered with OmniCoordinator (stage_id=%d replica_id=%d coord=%s)",
                    omni_stage_id,
                    omni_replica_id,
                    omni_coordinator_address,
                )

            asyncio.run(proc.run_loop(request_address, response_address))

        except SystemExit:
            logger.debug("StageDiffusionProc exiting.")
            raise
        except Exception:
            logger.exception("StageDiffusionProc encountered a fatal error.")
            raise
        finally:
            if handshake_socket is not None:
                handshake_socket.close(linger=0)
            if handshake_ctx is not None:
                handshake_ctx.term()
            if coord_client is not None:
                with contextlib.suppress(RuntimeError):
                    coord_client.close()
            proc.close()


class StageDiffusionProcManager:
    """Owns a StageDiffusionProc subprocess.

    Mirrors the small process-lifecycle surface used by vLLM's
    CoreEngineProcManager while keeping diffusion's custom wire protocol.
    """

    def __init__(
        self,
        *,
        model: str,
        od_config: OmniDiffusionConfig,
        stage_init_timeout: int,
        handshake_address: str | None = None,
        addresses: EngineZmqAddresses | None = None,
        omni_coordinator_address: str | None = None,
        omni_stage_id: int | None = None,
        omni_replica_id: int = 0,
    ) -> None:
        handshake_address = handshake_address or get_open_zmq_ipc_path()
        addresses = addresses or EngineZmqAddresses(
            inputs=[get_open_zmq_ipc_path()],
            outputs=[get_open_zmq_ipc_path()],
        )

        ctx = get_mp_context()
        proc = ctx.Process(
            target=StageDiffusionProc.run_diffusion_proc,
            name="StageDiffusionProc",
            kwargs={
                "model": model,
                "od_config": od_config,
                "handshake_address": handshake_address,
                "local_client": True,
                "headless": False,
                "omni_coordinator_address": omni_coordinator_address,
                "omni_stage_id": omni_stage_id,
                "omni_replica_id": omni_replica_id,
            },
        )
        proc.start()
        self.proc = proc
        self.addresses = addresses
        self.manager_stopped = False
        self.failed_proc_name: str | None = None

        self._wait_until_started(handshake_address, stage_init_timeout)

    @classmethod
    def launch_headless(
        cls,
        *,
        model: str,
        od_config: OmniDiffusionConfig,
        handshake_address: str,
        addresses: EngineZmqAddresses,
        omni_coordinator_address: str | None,
        omni_stage_id: int,
        omni_replica_id: int,
    ) -> StageDiffusionProcManager:
        """Launch a headless diffusion backend that connects to head-owned sockets."""
        self = cls.__new__(cls)
        ctx = get_mp_context()
        proc = ctx.Process(
            target=StageDiffusionProc.run_diffusion_proc,
            name="StageDiffusionProc",
            kwargs={
                "model": model,
                "od_config": od_config,
                "handshake_address": handshake_address,
                "local_client": False,
                "headless": True,
                "omni_coordinator_address": omni_coordinator_address,
                "omni_stage_id": omni_stage_id,
                "omni_replica_id": omni_replica_id,
            },
        )
        proc.start()
        self.proc = proc
        self.addresses = addresses
        self.manager_stopped = False
        self.failed_proc_name = None
        return self

    def _wait_until_started(self, handshake_address: str, stage_init_timeout: int) -> None:
        try:
            with zmq_socket_ctx(handshake_address, zmq.ROUTER, bind=True) as handshake_socket:
                wait_for_engine_startup(
                    handshake_socket,
                    self.addresses,
                    [CoreEngine(index=0, local=True)],
                    SimpleNamespace(
                        data_parallel_size_local=1,
                        data_parallel_hybrid_lb=False,
                        data_parallel_external_lb=False,
                    ),
                    False,
                    None,
                    self,
                    None,
                )
        except Exception:
            shutdown([self.proc])
            raise

    def shutdown(self, timeout: float | None = None) -> None:
        self.manager_stopped = True
        shutdown([self.proc], timeout=timeout)

    def sentinels(self) -> list[int]:
        return [self.proc.sentinel]

    def finished_procs(self) -> dict[str, int]:
        if self.proc.exitcode is None:
            return {}
        return {self.proc.name: self.proc.exitcode}

    def monitor_engine_liveness(self) -> None:
        try:
            multiprocessing.connection.wait([self.proc.sentinel])
        except Exception:
            return
        if self.proc.exitcode not in (None, 0) and not self.manager_stopped:
            self.failed_proc_name = self.proc.name
        self.shutdown()
