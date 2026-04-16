"""Stage Diffusion Client for vLLM-Omni multi-stage runtime.

Spawns StageDiffusionProc in a subprocess and communicates via ZMQ
(PUSH/PULL) to expose the same interface the Orchestrator expects
from any stage client.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any

import zmq
from vllm.logger import init_logger

from vllm_omni.diffusion.stage_diffusion_proc import (
    complete_diffusion_handshake,
    spawn_diffusion_proc,
)
from vllm_omni.distributed.omni_connectors.utils.serialization import (
    OmniMsgpackDecoder,
    OmniMsgpackEncoder,
)
from vllm_omni.engine.stage_init_utils import StageMetadata, terminate_alive_proc
from vllm_omni.outputs import OmniRequestOutput

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType

logger = init_logger(__name__)


def create_diffusion_client(
    model: str,
    od_config: OmniDiffusionConfig,
    metadata: StageMetadata,
    stage_init_timeout: int,
    batch_size: int = 1,
    use_inline: bool = False,
) -> Any:
    """Factory to create either an inline or out-of-process diffusion client."""
    if use_inline:
        from vllm_omni.diffusion.inline_stage_diffusion_client import InlineStageDiffusionClient

        return InlineStageDiffusionClient(model, od_config, metadata, batch_size=batch_size)
    return StageDiffusionClient(
        model, od_config, metadata, stage_init_timeout=stage_init_timeout, batch_size=batch_size
    )


class StageDiffusionClient:
    """Communicates with StageDiffusionProc via ZMQ for use inside the Orchestrator.

    Exposes the same attributes and async methods the Orchestrator
    uses on StageEngineCoreClient, but routes execution through
    a StageDiffusionProc subprocess instead of running the diffusion
    engine in-process.
    """

    stage_type: str = "diffusion"

    def __init__(
        self,
        model: str,
        od_config: OmniDiffusionConfig,
        metadata: StageMetadata,
        stage_init_timeout: int,
        batch_size: int = 1,
    ) -> None:
        # Spawn StageDiffusionProc subprocess and wait for READY.
        proc, handshake_address, request_address, response_address = spawn_diffusion_proc(model, od_config)
        complete_diffusion_handshake(proc, handshake_address, stage_init_timeout)
        self._initialize_client(metadata, request_address, response_address, proc=proc, batch_size=batch_size)

    @classmethod
    def from_addresses(
        cls,
        metadata: StageMetadata,
        request_address: str,
        response_address: str,
        *,
        proc: Any = None,
        batch_size: int = 1,
    ) -> StageDiffusionClient:
        """Create a client for an already-running diffusion subprocess."""
        client = cls.__new__(cls)
        client._initialize_client(
            metadata,
            request_address,
            response_address,
            proc=proc,
            batch_size=batch_size,
        )
        return client

    def _initialize_client(
        self,
        metadata: StageMetadata,
        request_address: str,
        response_address: str,
        *,
        proc: Any,
        batch_size: int,
    ) -> None:
        self.stage_id = metadata.stage_id
        self.final_output = metadata.final_output
        self.final_output_type = metadata.final_output_type
        self.default_sampling_params = metadata.default_sampling_params
        self.custom_process_input_func = metadata.custom_process_input_func
        self.engine_input_source = metadata.engine_input_source
        self._proc = proc
        self._owns_process = proc is not None

        self._zmq_ctx = zmq.Context()
        self._request_socket = self._zmq_ctx.socket(zmq.PUSH)
        self._request_socket.connect(request_address)
        self._response_socket = self._zmq_ctx.socket(zmq.PULL)
        self._response_socket.connect(response_address)

        self._encoder = OmniMsgpackEncoder()
        self._decoder = OmniMsgpackDecoder()

        self._output_queue: asyncio.Queue[OmniRequestOutput] = asyncio.Queue()
        self._rpc_results: dict[str, Any] = {}
        self._pending_rpcs: set[str] = set()
        self._tasks: dict[str, asyncio.Task] = {}
        self._shutting_down = False

        logger.info(
            "[StageDiffusionClient] Stage-%s initialized (owns_process=%s, batch_size=%d)",
            self.stage_id,
            self._owns_process,
            batch_size,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _drain_responses(self) -> None:
        """Non-blocking drain of all available responses from the subprocess."""
        while True:
            try:
                raw = self._response_socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                break

            msg = self._decoder.decode(raw)
            msg_type = msg.get("type")

            if msg_type == "result":
                self._output_queue.put_nowait(msg["output"])
            elif msg_type == "rpc_result":
                self._rpc_results[msg["rpc_id"]] = msg["result"]
            elif msg_type == "error":
                req_id = msg.get("request_id")
                rpc_id = msg.get("rpc_id")
                error_msg = msg.get("error")
                logger.error(
                    "[StageDiffusionClient] Stage-%s subprocess error for %s: %s",
                    self.stage_id,
                    rpc_id or req_id,
                    error_msg,
                )
                # Route RPC errors so collective_rpc_async can unblock.
                if rpc_id is not None and rpc_id in self._pending_rpcs:
                    self._rpc_results[rpc_id] = {
                        "error": True,
                        "reason": error_msg,
                    }
                elif req_id is not None:
                    error_output = OmniRequestOutput.from_diffusion(
                        request_id=req_id,
                        images=[],
                    )
                    error_output.error = error_msg
                    self._output_queue.put_nowait(error_output)

    # Fields that are subprocess-local and cannot be serialized across
    # process boundaries.  They are recreated in the subprocess with
    # their default values.
    _NON_SERIALIZABLE_FIELDS = frozenset(
        {
            "generator",  # torch.Generator — recreated from seed
            "modules",  # model components — loaded in subprocess
        }
    )

    @staticmethod
    def _sampling_params_to_dict(sampling_params: Any) -> dict[str, Any]:
        """Convert sampling params to a plain dict for serialization.

        Uses ``dataclasses.fields`` + ``getattr`` instead of ``asdict``
        to avoid deep-copying large tensors, and skips fields that
        cannot cross process boundaries.

        When a ``torch.Generator`` is present but ``seed`` is not set,
        the generator's initial seed is extracted so the subprocess can
        recreate an equivalent generator via ``diffusion_model_runner``.
        """
        if is_dataclass(sampling_params) and not isinstance(sampling_params, type):
            result = {
                f.name: getattr(sampling_params, f.name)
                for f in fields(sampling_params)
                if f.name not in StageDiffusionClient._NON_SERIALIZABLE_FIELDS
            }
        elif not isinstance(sampling_params, dict):
            raise TypeError(f"sampling_params is not a dict but {sampling_params.__class__.__name__}")
        else:
            result = {
                k: v for k, v in sampling_params.items() if k not in StageDiffusionClient._NON_SERIALIZABLE_FIELDS
            }

        # Preserve the generator's seed across the process boundary so
        # the subprocess can recreate deterministic random state.
        if result.get("seed") is None:
            generator = (
                getattr(sampling_params, "generator", None)
                if not isinstance(sampling_params, dict)
                else sampling_params.get("generator")
            )
            if generator is not None:
                if isinstance(generator, list) and generator:
                    generator = generator[0]
                if hasattr(generator, "initial_seed"):
                    result["seed"] = generator.initial_seed()

        return result

    # ------------------------------------------------------------------
    # Public API (matches the interface the Orchestrator expects)
    # ------------------------------------------------------------------

    async def add_request_async(
        self,
        request_id: str,
        prompt: OmniPromptType,
        sampling_params: OmniDiffusionSamplingParams,
        kv_sender_info: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        self._request_socket.send(
            self._encoder.encode(
                {
                    "type": "add_request",
                    "request_id": request_id,
                    "prompt": prompt,
                    "sampling_params": self._sampling_params_to_dict(sampling_params),
                    "kv_sender_info": kv_sender_info,
                }
            )
        )

    # TODO(Long): Temporary solution to boost performance of diffusion stages.
    # Remove this after scheduling algorithm is implemented
    async def add_batch_request_async(
        self,
        request_id: str,
        prompts: list[OmniPromptType],
        sampling_params: OmniDiffusionSamplingParams,
        kv_sender_info: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        """Submit a list of prompts as a single batched engine call.

        All prompts are processed in one ``DiffusionEngine.step()`` call
        and the combined result is placed on the output queue with a single
        *request_id*.
        """
        task = asyncio.create_task(
            self._run_batch(
                request_id,
                prompts,
                sampling_params,
                kv_sender_info,
            ),
            name=f"diffusion-batch-{request_id}",
        )
        self._tasks[request_id] = task

    async def _run_batch(
        self,
        request_id: str,
        prompts: list[OmniPromptType],
        sampling_params: OmniDiffusionSamplingParams,
        kv_sender_info: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        try:
            self._request_socket.send(
                self._encoder.encode(
                    {
                        "type": "add_batch_request",
                        "request_id": request_id,
                        "prompts": prompts,
                        "sampling_params": self._sampling_params_to_dict(sampling_params),
                        "kv_sender_info": kv_sender_info,
                    }
                )
            )
        except Exception as e:
            logger.exception(
                "[StageDiffusionClient] Stage-%s batch req=%s failed: %s",
                self.stage_id,
                request_id,
                e,
            )
        finally:
            self._tasks.pop(request_id, None)

    def get_diffusion_output_nowait(self) -> OmniRequestOutput | None:
        self._drain_responses()
        try:
            return self._output_queue.get_nowait()
        except asyncio.QueueEmpty:
            if not self._shutting_down and self._owns_process and self._proc is not None and not self._proc.is_alive():
                exitcode = self._proc.exitcode
                # One final drain – the last ZMQ frame may have arrived
                # between the first drain and the is_alive() check.
                self._drain_responses()
                try:
                    return self._output_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                if exitcode is not None and exitcode > 128:
                    sig = exitcode - 128
                    logger.warning("StageDiffusionProc was killed by signal %d; treating as external shutdown.", sig)
                    self._shutting_down = True
                    return None
                raise RuntimeError(f"StageDiffusionProc died unexpectedly (exit code {exitcode})")
            return None

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        self._request_socket.send(
            self._encoder.encode(
                {
                    "type": "abort",
                    "request_ids": list(request_ids),
                }
            )
        )

    async def collective_rpc_async(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Forward control RPCs to the diffusion subprocess."""
        # Inject a default profile_prefix that includes stage_id when profiling.
        if method == "profile":
            args_list = list(args)
            is_start = args_list[0] if args_list else True
            profile_prefix = args_list[1] if len(args_list) > 1 else None
            if is_start and profile_prefix is None:
                profile_prefix = f"stage_{self.stage_id}_diffusion_{int(time.time())}"
                if len(args_list) > 1:
                    args_list[1] = profile_prefix
                else:
                    args_list.append(profile_prefix)
                args = tuple(args_list)

        kwargs = kwargs or {}
        rpc_id = uuid.uuid4().hex
        self._pending_rpcs.add(rpc_id)

        self._request_socket.send(
            self._encoder.encode(
                {
                    "type": "collective_rpc",
                    "rpc_id": rpc_id,
                    "method": method,
                    "timeout": timeout,
                    "args": list(args),
                    "kwargs": kwargs,
                }
            )
        )

        deadline = time.monotonic() + timeout if timeout else None
        # Wait for the matching RPC response, buffering result messages.
        try:
            while True:
                self._drain_responses()
                if rpc_id in self._rpc_results:
                    return self._rpc_results.pop(rpc_id)
                if self._owns_process and self._proc is not None and not self._proc.is_alive():
                    raise RuntimeError(
                        f"StageDiffusionProc died while waiting for "
                        f"collective_rpc '{method}' (exit code {self._proc.exitcode})"
                    )
                if deadline and time.monotonic() > deadline:
                    raise TimeoutError(f"collective_rpc_async '{method}' timed out after {timeout}s")
                await asyncio.sleep(0.01)
        finally:
            self._pending_rpcs.discard(rpc_id)

    def shutdown(self) -> None:
        self._shutting_down = True
        try:
            self._request_socket.send(self._encoder.encode({"type": "shutdown"}))
        except Exception:
            pass

        if self._owns_process and self._proc is not None and self._proc.is_alive():
            self._proc.join(timeout=10)
            terminate_alive_proc(self._proc)

        self._request_socket.close(linger=0)
        self._response_socket.close(linger=0)
        self._zmq_ctx.term()
