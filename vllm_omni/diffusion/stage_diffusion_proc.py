"""Subprocess entry point for the diffusion engine.

StageDiffusionProc runs AsyncOmniDiffusion in a child process,
communicating with StageDiffusionClient via ZMQ (PUSH/PULL).
"""

from __future__ import annotations

import asyncio
import signal
from multiprocessing.process import BaseProcess
from typing import TYPE_CHECKING, Any

import msgspec
import zmq
import zmq.asyncio

from vllm.logger import init_logger
from vllm.utils.network_utils import get_open_zmq_ipc_path, zmq_socket_ctx
from vllm.utils.system_utils import get_mp_context
from vllm.v1.utils import shutdown

from vllm_omni.distributed.omni_connectors.utils.serialization import (
    OmniMsgpackDecoder,
    OmniMsgpackEncoder,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)

_HANDSHAKE_POLL_TIMEOUT_S = 600


class StageDiffusionProc:
    """Subprocess entry point for diffusion inference."""

    @staticmethod
    def run_diffusion_proc(
        model: str,
        od_config: OmniDiffusionConfig,
        handshake_address: str,
        request_address: str,
        response_address: str,
    ) -> None:
        """Run the diffusion engine in a subprocess."""
        shutdown_requested = False

        def signal_handler(signum: int, frame: Any) -> None:
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        engine = None
        try:
            from vllm_omni.entrypoints.async_omni_diffusion import (
                AsyncOmniDiffusion,
            )

            engine = AsyncOmniDiffusion(model=model, od_config=od_config)

            # Send READY via handshake socket
            handshake_ctx = zmq.Context()
            handshake_socket = handshake_ctx.socket(zmq.DEALER)
            handshake_socket.connect(handshake_address)
            handshake_socket.send(msgspec.msgpack.encode({"status": "READY"}))
            handshake_socket.close()
            handshake_ctx.term()

            # Run async event loop
            asyncio.run(_run_loop(engine, request_address, response_address))

        except SystemExit:
            logger.debug("StageDiffusionProc exiting.")
            raise
        except Exception:
            logger.exception("StageDiffusionProc encountered a fatal error.")
            raise
        finally:
            if engine is not None:
                engine.close()


async def _run_loop(
    engine: Any,
    request_address: str,
    response_address: str,
) -> None:
    """Async event loop for the diffusion subprocess."""
    ctx = zmq.asyncio.Context()

    request_socket = ctx.socket(zmq.PULL)
    request_socket.bind(request_address)

    response_socket = ctx.socket(zmq.PUSH)
    response_socket.bind(response_address)

    encoder = OmniMsgpackEncoder()
    decoder = OmniMsgpackDecoder()

    tasks: dict[str, asyncio.Task] = {}

    async def process_request(request_id: str, prompt: Any, sampling_params_dict: dict) -> None:
        """Process a single diffusion request."""
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        try:
            # Reconstruct LoRARequest if needed.
            # LoRARequest is a msgspec.Struct with array_like=True,
            # so it deserializes as a list (not dict). Use msgspec.convert
            # to handle both list and dict representations.
            lora_req = sampling_params_dict.get("lora_request")
            if lora_req is not None:
                from vllm.lora.request import LoRARequest

                if not isinstance(lora_req, LoRARequest):
                    sampling_params_dict["lora_request"] = msgspec.convert(lora_req, LoRARequest)

            sampling_params = OmniDiffusionSamplingParams(
                **sampling_params_dict
            )
            result = await engine.generate(prompt, sampling_params, request_id)
            await response_socket.send(encoder.encode({"type": "result", "output": result}))
        except Exception as e:
            logger.exception("Diffusion request %s failed: %s", request_id, e)
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

    try:
        while True:
            raw = await request_socket.recv()
            msg = decoder.decode(raw)
            msg_type = msg.get("type")

            if msg_type == "add_request":
                request_id = msg["request_id"]
                prompt = msg["prompt"]
                sampling_params_dict = msg["sampling_params"]
                task = asyncio.create_task(process_request(request_id, prompt, sampling_params_dict))
                tasks[request_id] = task

            elif msg_type == "abort":
                for rid in msg.get("request_ids", []):
                    task = tasks.pop(rid, None)
                    if task:
                        task.cancel()
                    await engine.abort(rid)

            elif msg_type == "collective_rpc":
                rpc_id = msg["rpc_id"]
                method = msg["method"]
                timeout = msg.get("timeout")
                args = tuple(msg.get("args", ()))
                kwargs = msg.get("kwargs", {})
                try:
                    result = await _handle_collective_rpc(engine, method, timeout, args, kwargs)
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
                    logger.exception("Collective RPC %s failed: %s", method, e)
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


async def _handle_collective_rpc(
    engine: Any,
    method: str,
    timeout: float | None,
    args: tuple,
    kwargs: dict,
) -> Any:
    """Handle collective RPC calls in the subprocess."""
    if method in {
        "add_lora",
        "remove_lora",
        "list_loras",
        "pin_lora",
        "start_profile",
        "stop_profile",
    }:
        # Reconstruct LoRARequest after IPC if needed.
        # LoRARequest is a msgspec.Struct with array_like=True,
        # so msgpack deserializes it as a plain list.
        if method == "add_lora" and args:
            from vllm.lora.request import LoRARequest

            if not isinstance(args[0], LoRARequest):
                args = (msgspec.convert(args[0], LoRARequest), *args[1:])

        target = getattr(engine, method, None)
        if target is None:
            return {
                "supported": False,
                "todo": True,
                "reason": (f"AsyncOmniDiffusion.{method} is not implemented"),
            }
        result = target(*args, **kwargs)
        if asyncio.iscoroutine(result) or asyncio.isfuture(result):
            if timeout is not None:
                return await asyncio.wait_for(result, timeout=timeout)
            return await result
        return result

    if method in {"sleep", "wake_up"}:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            engine._executor,
            engine.engine.collective_rpc,
            method,
            timeout,
            args,
            kwargs,
            None,
        )

    return {
        "supported": False,
        "todo": True,
        "reason": (
            f"Diffusion stage collective_rpc method {method} "
            "is not implemented yet"
        ),
    }


def spawn_diffusion_proc(
    model: str,
    od_config: OmniDiffusionConfig,
) -> tuple[BaseProcess, str, str, str]:
    """Spawn a StageDiffusionProc subprocess.

    Returns ``(proc, handshake_address, request_address, response_address)``.
    """
    handshake_address = get_open_zmq_ipc_path()
    request_address = get_open_zmq_ipc_path()
    response_address = get_open_zmq_ipc_path()

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
