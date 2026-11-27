# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""WebSocket inference for a single Motubrain 2.0 diffusion stage."""

from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from omegaconf import OmegaConf
from vllm.logger import init_logger
from vllm.utils import random_uuid

from vllm_omni.entrypoints.motubrain2_codec import pack_message, unpack_message
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

router = APIRouter()
logger = init_logger(__name__)
_MAX_REQUEST_BYTES = 64 * 1024 * 1024
_NATIVE_OUTPUTS = ("video", "action", "decoded_action", "decoded_valid")


class Motubrain2Serving:
    def __init__(self, engine_client: Any, metadata: Mapping[str, Any]):
        self.engine_client = engine_client
        self.metadata = dict(metadata)

    @classmethod
    def from_engine(cls, engine_client: Any) -> "Motubrain2Serving | None":
        get_config = getattr(engine_client, "get_diffusion_od_config", None)
        configs = [get_config()] if callable(get_config) else []
        # With out-of-process workers the frontend config only identifies the
        # pipeline. The typed stage still has the robot's deployment metadata.
        configs.extend(
            getattr(stage, "diffusion_config", None) for stage in getattr(engine_client, "stage_configs", ())
        )
        for config in configs:
            model_config = getattr(config, "model_config", None)
            if isinstance(model_config, Mapping) and "motubrain2_metadata" in model_config:
                metadata = model_config["motubrain2_metadata"]
                if OmegaConf.is_config(metadata):
                    metadata = OmegaConf.to_container(metadata, resolve=True)
                if not isinstance(metadata, Mapping):
                    raise ValueError("motubrain2_metadata must be a mapping")
                return cls(engine_client, metadata)
        return None

    async def infer(self, obs: dict, *, session_id: str, reset: bool) -> np.ndarray | dict[str, np.ndarray]:
        sampling = obs.get("sampling_params", {})
        if not isinstance(sampling, Mapping):
            raise ValueError("sampling_params must be a mapping")
        native_outputs = obs.get("return_native_outputs", False)
        if not isinstance(native_outputs, (bool, np.bool_)):
            raise ValueError("return_native_outputs must be a boolean")
        count = sampling.get("n", 1)
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count != 1:
            raise ValueError("Motubrain produces a single action chunk per observation")
        for default in getattr(self.engine_client, "default_sampling_params_list", None) or ():
            if isinstance(default, OmniDiffusionSamplingParams):
                params = deepcopy(default)
                break
        else:
            params = OmniDiffusionSamplingParams()
        seed = obs.get("seed")
        if seed is None:
            seed = sampling.get("seed")
        params.extra_args = {
            **(params.extra_args or {}),
            "robot_obs": {key: value for key, value in obs.items() if key not in ("seed", "sampling_params")},
            "session_id": session_id,
            "reset": reset,
        }
        if seed is None:
            # The worker owns the persistent RNG. Explicit seed=0 is an independent call.
            params.extra_args["motubrain2_session_seed"] = params.seed if params.seed is not None else 0
        else:
            params.seed = int(seed)
        if "num_inference_steps" in sampling:
            params.num_inference_steps = sampling["num_inference_steps"]

        output = None
        async for result in self.engine_client.generate(
            prompt=obs.get("prompt") or obs.get("instruction", ""),
            request_id=f"motubrain2-{random_uuid()}",
            sampling_params_list=[params],
        ):
            output = result.multimodal_output
        if not isinstance(output, Mapping):
            raise RuntimeError("Missing Motubrain output")
        if native_outputs:
            tensors = output.get("native_outputs")
            if not isinstance(tensors, Mapping) or set(tensors) != set(_NATIVE_OUTPUTS):
                raise RuntimeError("Missing native Motubrain outputs")
            return {key: np.asarray(tensors[key]) for key in _NATIVE_OUTPUTS}
        actions = np.asarray(output["actions"], dtype=np.float32)
        expected = (self.metadata["action_horizon"], self.metadata["action_dim"])
        if actions.shape != expected or not np.isfinite(actions).all():
            raise ValueError(f"Motubrain requires finite controller actions with shape {expected}")
        return actions


@router.websocket("/v1/realtime/robot/motubrain2")
async def serve_motubrain2(websocket: WebSocket) -> None:
    serving = Motubrain2Serving.from_engine(websocket.app.state.engine_client)
    await websocket.accept()
    if serving is None:
        await websocket.send_bytes(pack_message({"type": "error", "message": "Motubrain robot metadata unavailable"}))
        await websocket.close()
        return
    session_id = "default"
    seen: OrderedDict[str, None] = OrderedDict()
    try:
        await websocket.send_bytes(pack_message(serving.metadata))
        while True:
            event = await websocket.receive()
            if event["type"] == "websocket.disconnect":
                return
            message = event.get("bytes")
            try:
                if not isinstance(message, bytes) or len(message) > _MAX_REQUEST_BYTES:
                    raise ValueError("Expected a binary robot observation of at most 64 MiB")
                obs = unpack_message(message)
                if not isinstance(obs, dict):
                    raise ValueError("Expected a robot observation mapping")
                endpoint = obs.pop("endpoint", "infer")
                if endpoint == "reset":
                    seen.clear()
                    session_id = "default"
                    reply = {"status": "reset successful"}
                elif endpoint == "infer":
                    session_id = str(obs.get("session_id") or session_id)
                    reply = await serving.infer(obs, session_id=session_id, reset=session_id not in seen)
                    seen[session_id] = None
                    seen.move_to_end(session_id)
                    if len(seen) > 1024:
                        seen.popitem(last=False)
                else:
                    raise ValueError("Unknown robot endpoint")
            except ValueError as exc:
                reply = {"type": "error", "message": str(exc)}
            except Exception:
                logger.exception("Motubrain request failed")
                reply = {"type": "error", "message": "Invalid request or inference failed"}
            await websocket.send_bytes(pack_message(reply))
    except WebSocketDisconnect:
        pass
