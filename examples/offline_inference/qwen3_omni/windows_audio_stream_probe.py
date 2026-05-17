# SPDX-License-Identifier: Apache-2.0
"""Windows Qwen3-Omni audio-in/audio-out streaming smoke test.

This validates the full thinker -> talker -> code2wav path and records event
metadata only. It stops after the first streamed audio tensor by default and
does not write a WAV file.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import socket
import sys
import time
import types
from pathlib import Path
from typing import Any

from end2end import get_audio_query


def install_windows_runtime_shims() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    module = types.ModuleType("fcntl")
    module.LOCK_EX = 2
    module.LOCK_NB = 4
    module.LOCK_UN = 8
    module.flock = lambda _file, _flags: None
    sys.modules.setdefault("fcntl", module)


def get_open_tcp_zmq_path() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"tcp://127.0.0.1:{port}"


def patch_zmq_ipc_to_tcp() -> None:
    import vllm.utils.network_utils as network_utils

    network_utils.get_open_zmq_ipc_path = get_open_tcp_zmq_path
    try:
        import vllm_omni.engine.stage_engine_core_proc as proc

        proc.get_open_zmq_ipc_path = get_open_tcp_zmq_path
        patch_stage_handshake_for_windows(proc)
    except Exception:
        pass


def patch_stage_handshake_for_windows(proc_module: Any) -> None:
    """Avoid polling a Windows multiprocessing sentinel with pyzmq."""
    import msgspec
    import zmq
    from vllm.utils.network_utils import zmq_socket_ctx
    from vllm.v1.engine.utils import EngineHandshakeMetadata

    def recv_socket_only(poller, handshake_socket, proc, expected: str, timeout_s: int):
        while True:
            events = dict(poller.poll(timeout=timeout_s * 1000))
            if not events:
                if proc.exitcode is not None:
                    raise RuntimeError(f"StageEngineCoreProc died during {expected} (exit code {proc.exitcode})")
                raise TimeoutError(f"Timed out waiting for {expected} from StageEngineCoreProc.")
            if handshake_socket in events:
                identity, raw = handshake_socket.recv_multipart()
                return identity, msgspec.msgpack.decode(raw)
            if proc.exitcode is not None:
                raise RuntimeError(f"StageEngineCoreProc died during {expected} (exit code {proc.exitcode})")

    def perform_handshake(proc, handshake_address, addresses, vllm_config, handshake_timeout):
        with zmq_socket_ctx(handshake_address, zmq.ROUTER, bind=True) as handshake_socket:
            poller = zmq.Poller()
            poller.register(handshake_socket, zmq.POLLIN)

            identity, msg = recv_socket_only(poller, handshake_socket, proc, "HELLO", handshake_timeout)
            if msg.get("status") != "HELLO":
                raise RuntimeError(f"Expected HELLO, got: {msg}")

            init_payload = EngineHandshakeMetadata(addresses=addresses, parallel_config={})
            handshake_socket.send_multipart([identity, msgspec.msgpack.encode(init_payload)])

            identity, msg = recv_socket_only(poller, handshake_socket, proc, "READY", handshake_timeout)
            if msg.get("status") != "READY":
                raise RuntimeError(f"Expected READY, got: {msg}")
            num_gpu_blocks = msg.get("num_gpu_blocks")
            if num_gpu_blocks is not None:
                vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks

    proc_module._perform_handshake = perform_handshake


def tensor_summary(value: Any) -> dict[str, Any]:
    return {
        "shape": list(value.shape) if hasattr(value, "shape") else None,
        "numel": int(value.numel()) if hasattr(value, "numel") else None,
        "type": type(value).__name__,
    }


def summarize_output(output: Any, elapsed_s: float, state: dict[str, int]) -> dict[str, Any]:
    request_output = getattr(output, "request_output", None)
    completions = getattr(request_output, "outputs", None) or []
    completion = completions[0] if completions else None
    final_output_type = getattr(output, "final_output_type", None)
    event: dict[str, Any] = {
        "elapsed_seconds": elapsed_s,
        "stage_id": getattr(output, "stage_id", None),
        "finished": bool(getattr(output, "finished", False)),
        "final_output_type": final_output_type,
        "request_id": getattr(output, "request_id", None),
    }
    if final_output_type == "text" and completion is not None:
        text = getattr(completion, "text", None)
        event["text"] = text
        event["text_len"] = len(text) if isinstance(text, str) else None
    if final_output_type == "audio" and completion is not None:
        mm = getattr(completion, "multimodal_output", None) or {}
        audio = mm.get("audio") if isinstance(mm, dict) else None
        sr = mm.get("sr") if isinstance(mm, dict) else None
        if hasattr(sr, "item"):
            sr = int(sr.item())
        elif sr is not None:
            sr = int(sr)
        event["sample_rate"] = sr
        if isinstance(audio, list):
            consumed = state.get("audio_list_consumed", 0)
            new_chunks = audio[consumed:]
            state["audio_list_consumed"] = len(audio)
            event["audio_chunks_total"] = len(audio)
            event["audio_chunks_new"] = len(new_chunks)
            event["audio_new"] = [tensor_summary(chunk) for chunk in new_chunks]
        elif audio is not None:
            numel = int(audio.numel()) if hasattr(audio, "numel") else 0
            previous = state.get("audio_tensor_numel", 0)
            state["audio_tensor_numel"] = max(previous, numel)
            event["audio_chunks_total"] = 1
            event["audio_chunks_new"] = 1 if numel > previous else 0
            event["audio_new"] = [tensor_summary(audio)]
    return event


async def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "ok": False,
        "model": args.model,
        "deploy_config": args.deploy_config,
        "events": [],
        "error_type": None,
        "error": None,
    }
    omni = None
    try:
        install_windows_runtime_shims()
        patch_zmq_ipc_to_tcp()

        from vllm_omni import AsyncOmni

        query = get_audio_query(question=args.question)
        prompt = dict(query.inputs)
        prompt["modalities"] = ["audio"]
        omni = AsyncOmni(
            model=args.model,
            deploy_config=args.deploy_config,
            init_timeout=args.init_timeout,
            stage_init_timeout=args.stage_init_timeout,
            output_modalities=["audio"],
            log_stats=False,
        )

        stream_state: dict[str, int] = {}
        try:
            async with asyncio.timeout(args.generation_timeout):
                async for output in omni.generate(
                    prompt,
                    request_id=args.request_id,
                    sampling_params_list=None,
                    output_modalities=["audio"],
                ):
                    event = summarize_output(output, time.perf_counter() - started, stream_state)
                    result["events"].append(event)
                    audio_chunk_count = sum(
                        int(item.get("audio_chunks_new") or 0)
                        for item in result["events"]
                        if item.get("final_output_type") == "audio"
                    )
                    if args.stop_after_audio_chunks > 0 and audio_chunk_count >= args.stop_after_audio_chunks:
                        result["stop_reason"] = "audio_chunk_target"
                        break
                    if event["finished"] and event["final_output_type"] == "audio":
                        result["stop_reason"] = "audio_finished"
                        break
        except TimeoutError:
            result["error_type"] = "TimeoutError"
            result["error"] = f"Timed out after {args.generation_timeout}s waiting for audio output."

        close_result = omni.close()
        if inspect.isawaitable(close_result):
            await close_result
        audio_events = [event for event in result["events"] if event.get("final_output_type") == "audio"]
        result["audio_event_count"] = len(audio_events)
        result["audio_chunk_count"] = sum(int(event.get("audio_chunks_new") or 0) for event in audio_events)
        result["ok"] = result["error_type"] is None and result["audio_chunk_count"] > 0
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        if omni is not None:
            try:
                close_result = omni.close()
                if inspect.isawaitable(close_result):
                    await close_result
            except Exception:
                pass
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
    return result


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument(
        "--deploy-config",
        default=str(repo_root / "vllm_omni/deploy/qwen3_omni_moe_windows_single_gpu.yaml"),
    )
    parser.add_argument(
        "--question",
        default="After hearing this audio, respond aloud in one short sentence saying what was recited.",
    )
    parser.add_argument("--request-id", default="qwen3_omni_audio_stream_probe")
    parser.add_argument("--init-timeout", type=int, default=600)
    parser.add_argument("--stage-init-timeout", type=int, default=600)
    parser.add_argument("--generation-timeout", type=float, default=180.0)
    parser.add_argument("--stop-after-audio-chunks", type=int, default=1)
    parser.add_argument("--out", default="windows_qwen3_omni_audio_stream_probe.json")
    args = parser.parse_args()

    result = asyncio.run(run(args))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
