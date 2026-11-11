# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Positronic session envelopes over native Omni robot-policy requests."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING, cast

from fastapi import HTTPException, Request, WebSocket
from starlette.websockets import WebSocketDisconnect
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.stage_params import get_default_sampling_params_list
from vllm_omni.entrypoints.openpi.connection import await_policy
from vllm_omni.entrypoints.openpi.serving import ServingRealtimeRobotOpenPI
from vllm_omni.entrypoints.positronic.codec import IMAGE_CODECS, pack, unpack
from vllm_omni.entrypoints.positronic.praxis import parse_session, session_metadata
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras.praxis1 import PraxisSessionOptions, decode_prompt

if TYPE_CHECKING:
    from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)


def authorized(request: Request | WebSocket) -> bool:
    """Honor vLLM API keys on /api/v1, outside upstream's guarded URL prefixes."""
    from vllm import envs
    from vllm.entrypoints.serve.middleware.authenticate import AuthenticationMiddleware

    args = getattr(request.app.state, "args", None)
    tokens = [key for key in (getattr(args, "api_key", None) or [envs.VLLM_API_KEY]) if key]
    return not tokens or AuthenticationMiddleware(request.app, tokens).verify_token(request.headers)


async def refuse(websocket: WebSocket, reason: str) -> None:
    await websocket.send_bytes(pack({"status": "error", "error": reason}))
    # Close frames carry at most 123 bytes of UTF-8 reason; the full text is in the error payload.
    await websocket.close(code=1008, reason=reason.encode("utf-8")[:120].decode("utf-8", "ignore"))


def _build_policy_request(
    serving: ServingRealtimeRobotOpenPI,
    obs: dict,
    prompt: str,
    session_id: str,
    options: PraxisSessionOptions,
    seed: int,
) -> OmniDiffusionRequest:
    """Keep connection-owned sampling and episode settings out of observations."""
    obs = {k: v for k, v in obs.items() if k not in {"seed", "session_id", "sampling_params", "end_session"}}
    obs["task"] = prompt
    request = serving.build_request(obs, session_id=session_id, reset=False)
    request.sampling_params.seed = seed
    request.sampling_params.num_inference_steps = options.num_steps
    request.sampling_params.extra_args.update(
        {
            "praxis_session_options": options.to_dict(),
            "return_commands": True,
            "return_diagnostics": False,
            "decode_video": False,
        }
    )
    return request


def _session_metadata(
    query: Mapping[str, str],
    options: PraxisSessionOptions,
    values: dict,
    name: str,
    session_id: str,
    policy_config: Mapping | None = None,
) -> dict:
    """Attach a safe rollout identifier to the declared protocol contract."""
    rollout = values.get("rollout_id", "")
    safe_rollout = (
        isinstance(rollout, str)
        and 0 < len(rollout) <= 128
        and rollout[0].isascii()
        and rollout[0].isalnum()
        and all(c.isascii() and (c.isalnum() or c in "._-") for c in rollout)
    )
    meta = session_metadata(options, values, name, list(IMAGE_CODECS), policy_config)
    meta.update(
        {
            "rollout_id": rollout if safe_rollout else session_id,
            "rollout_id_source": "client_supplied"
            if safe_rollout
            else "bridge_generated_after_reject"
            if rollout
            else "bridge_generated",
            "session_overrides": dict(query),
        }
    )
    return meta


async def handle_session(websocket: WebSocket, model_id: str | None = None) -> None:
    """One UUID-isolated policy episode per socket; all inference stays in the engine."""
    if not authorized(websocket):
        await websocket.close(code=1008, reason="Unauthorized")
        return
    await websocket.accept()
    serving: ServingRealtimeRobotOpenPI | None = getattr(websocket.app.state, "openai_serving_realtime_robot", None)
    if serving is None or serving.policy_server_config.values.get("policy") != "praxis":
        await refuse(websocket, "Positronic serving requires the native Praxis deployment")
        return
    if not serving.has_single_worker():
        await refuse(websocket, "Positronic sessions currently require one stage and one worker replica")
        return
    config = serving.policy_server_config.to_dict()
    name = config.get("model_id") or serving.model_name
    if not name:
        await refuse(websocket, "The deployment declares no model_id or served model name")
        return
    aliases = {name, serving.model_name} - {None}
    if model_id is not None and model_id not in aliases:
        await refuse(websocket, "Unknown model; see /api/v1/models")
        return
    defaults = {
        **serving._get_policy_model_config(serving.engine_client),
        **{k: config[k] for k in ("stack_keys", "embodiment") if k in config},
    }
    for params in get_default_sampling_params_list(serving.engine_client):
        if isinstance(params, OmniDiffusionSamplingParams) and params.num_inference_steps is not None:
            defaults["num_steps"] = params.num_inference_steps
    try:
        options, values = parse_session(websocket.query_params, defaults)
    except (TypeError, ValueError) as exc:
        await refuse(websocket, str(exc))
        return
    session_id = "positronic-" + uuid.uuid4().hex
    meta = _session_metadata(websocket.query_params, options, values, name, session_id, config)
    calls = 0
    prompt = str(values.get("prompt", ""))
    allocated = False
    receive = asyncio.create_task(websocket.receive())
    try:
        await websocket.send_bytes(pack({"status": "loading", "message": "Loading policy..."}))
        reservation = _build_policy_request(serving, {}, prompt, session_id, options, cast(int, values["sample_seed"]))
        reservation.sampling_params.extra_args["reserve_session"] = True
        allocated = True
        admission = await await_policy(serving.generate_result(reservation), receive, websocket)
        if not admission.multimodal_output["trajectory"]["session_available"]:
            await refuse(websocket, "Policy session capacity reached; retry after another session closes")
            return
        await websocket.send_bytes(pack({"status": "ready", "meta": meta}))
        while True:
            message = await receive
            if message["type"] == "websocket.disconnect":
                return
            receive = asyncio.create_task(websocket.receive())
            if not isinstance(message.get("bytes"), bytes):
                await refuse(websocket, "Send each observation as one binary msgpack frame")
                return
            budget = values.get("max_chunks_per_episode")
            if isinstance(budget, int) and calls >= budget:
                await websocket.send_bytes(pack({"error": f"max_chunks_per_episode={budget} reached"}))
                await websocket.close(code=1000)
                return
            started = time.monotonic()
            received = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            codec_timing: dict = {}
            try:
                obs = await asyncio.to_thread(
                    unpack,
                    message["bytes"],
                    codec_timing,
                    tuple(config.get("camera_keys", ("image.exterior", "image.wrist"))),
                )
                # Transport settings belong to the connection, never to untrusted observations.
                supplied = obs.get("task")
                if not isinstance(supplied, (str, bytes)) or not supplied:
                    supplied = obs.get("prompt")
                if isinstance(supplied, (str, bytes)) and supplied:
                    prompt = decode_prompt(supplied).strip() or prompt
            except (ValueError, TypeError, OSError) as exc:
                # Malformed, oversized or unsupported client input is the client's error, not a failure.
                logger.info("Positronic observation rejected on %s: %s", session_id, exc)
                await refuse(websocket, f"Invalid observation: {exc}")
                return
            decoded = time.monotonic()
            request = _build_policy_request(serving, obs, prompt, session_id, options, cast(int, values["sample_seed"]))
            # Mark before awaiting: a failed or cancelled inference may already own worker state.
            allocated = True
            output = await await_policy(serving.generate_result(request), receive, websocket)
            payload = output.multimodal_output
            trajectory = payload.get("trajectory") if isinstance(payload, Mapping) else None
            if not isinstance(trajectory, Mapping) or not isinstance(trajectory.get("commands"), list):
                raise RuntimeError("Praxis worker did not return command dictionaries")
            calls += 1
            response = {"result": trajectory["commands"], "effective_prompt": prompt, "prompt_original": None}
            if values.get("return_server_timing", True):
                response["server_timing"] = {
                    **codec_timing,
                    "request_bytes": len(message["bytes"]),
                    "deserialize_ms": round((decoded - started) * 1000, 1),
                    "policy_ms": round((time.monotonic() - decoded) * 1000, 1),
                    "recv_iso": received,
                    "send_iso": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                    "cfg_scale": options.cfg_scale,
                    "cfg_video_scale": options.cfg_video_scale,
                    "cfg_batched": False,
                    "t_shift": options.t_shift,
                    "solver": "euler",
                    "num_evals": options.num_steps,
                }
            await websocket.send_bytes(pack(response))
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Positronic policy session failed: %s", session_id)
        try:
            await websocket.send_bytes(pack({"error": "Policy request failed; this session has ended"}))
            await websocket.close(code=1011)
        except (WebSocketDisconnect, RuntimeError, OSError):
            logger.debug("Positronic peer disconnected before error response")
    finally:
        receive.cancel()
        await asyncio.gather(receive, return_exceptions=True)
        if allocated:
            await serving.close_sessions([session_id])


async def list_models(request: Request) -> dict[str, list[str]]:
    if not authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    serving = getattr(request.app.state, "openai_serving_realtime_robot", None)
    config = serving.policy_server_config.to_dict() if serving is not None else {}
    name = config.get("model_id") or getattr(serving, "model_name", None)
    return {"models": [name] if config.get("policy") == "praxis" and name else []}
