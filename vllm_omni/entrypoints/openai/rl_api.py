# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stage-aware HTTP control and diffusion rollout interfaces for online RL."""

from __future__ import annotations

import asyncio
from http import HTTPStatus
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Body, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.lora.request import LoRARequest
from vllm.utils import random_uuid
from vllm.v1.engine import PauseMode

from vllm_omni.entrypoints.openai.stage_params import (
    build_stage_sampling_params_list,
    get_default_sampling_params_list,
)
from vllm_omni.entrypoints.openai.utils import get_stage_type
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt
from vllm_omni.lora.utils import stable_lora_int_id

router = APIRouter()


class AbortRequestsRequest(BaseModel):
    request_ids: list[str] | None = None


class WeightTransferBody(BaseModel):
    init_info: dict[str, Any]


class WeightUpdateBody(BaseModel):
    update_info: dict[str, Any]


class WeightsChecksumRequest(BaseModel):
    stage_ids: list[int] | None = None
    component: str | None = None


class RolloutGenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str | None = None
    request_id: str = ""
    stage_id: int | None = None
    sampling_params: dict[str, Any] = Field(default_factory=dict)
    transport: Literal["shm"] = "shm"


class LoadLoRARequest(BaseModel):
    lora_name: str
    lora_path: str
    load_inplace: bool = False
    is_3d_lora_weight: bool = False
    stage_ids: list[int] | None = None


class UnloadLoRARequest(BaseModel):
    lora_name: str
    lora_int_id: int | None = None
    stage_ids: list[int] | None = None


def _engine(raw_request: Request) -> Any:
    return raw_request.app.state.engine_client


def _control_response(message: str, acks: list[dict[str, Any]]) -> JSONResponse:
    return JSONResponse(
        content=jsonable_encoder({"message": message, "acks": acks}),
        status_code=HTTPStatus.OK,
    )


def _raise_control_error(error: Exception) -> None:
    status = (
        HTTPStatus.BAD_REQUEST if isinstance(error, (ValueError, RuntimeError)) else HTTPStatus.INTERNAL_SERVER_ERROR
    )
    raise HTTPException(status_code=status, detail=str(error)) from error


@router.post("/pause")
async def pause_generation(
    raw_request: Request,
    mode: Annotated[PauseMode, Query()] = "abort",
    wait_for_inflight_requests: bool = Query(False),
    clear_cache: bool = Query(True),
) -> JSONResponse:
    try:
        await _engine(raw_request).pause_generation(
            mode=mode,
            wait_for_inflight_requests=wait_for_inflight_requests,
            clear_cache=clear_cache,
        )
    except Exception as error:
        _raise_control_error(error)
    return JSONResponse({"status": "paused"})


@router.post("/resume")
async def resume_generation(raw_request: Request) -> JSONResponse:
    await _engine(raw_request).resume_generation()
    return JSONResponse({"status": "resumed"})


@router.get("/is_paused")
async def is_paused(raw_request: Request) -> JSONResponse:
    engine = _engine(raw_request)
    if hasattr(engine, "get_pause_status"):
        return JSONResponse(await engine.get_pause_status())
    return JSONResponse({"is_paused": await engine.is_paused()})


@router.post("/abort_requests")
async def abort_requests(
    raw_request: Request,
    body: AbortRequestsRequest | None = None,
) -> JSONResponse:
    request_ids = body.request_ids if body is not None else None
    if request_ids:
        await _engine(raw_request).abort(request_ids)
        aborted = len(request_ids)
    else:
        aborted = await _engine(raw_request).abort_all()
    return JSONResponse({"status": "aborted", "aborted": aborted})


@router.post("/init_weight_transfer_engine")
async def init_weight_transfer_engine(body: WeightTransferBody, raw_request: Request) -> JSONResponse:
    try:
        acks = await _engine(raw_request).init_weight_transfer_engine(
            WeightTransferInitRequest(init_info=body.init_info)
        )
    except Exception as error:
        _raise_control_error(error)
    return _control_response("Weight transfer initialized", acks)


@router.post("/start_weight_update")
async def start_weight_update(raw_request: Request) -> JSONResponse:
    try:
        acks = await _engine(raw_request).start_weight_update()
    except Exception as error:
        _raise_control_error(error)
    return _control_response("Weight update started", acks)


@router.post("/start_draft_weight_update")
async def start_draft_weight_update(raw_request: Request) -> JSONResponse:
    try:
        acks = await _engine(raw_request).start_draft_weight_update()
    except Exception as error:
        _raise_control_error(error)
    return _control_response("Draft weight update started", acks)


@router.post("/update_weights")
async def update_weights(body: WeightUpdateBody, raw_request: Request) -> JSONResponse:
    try:
        acks = await _engine(raw_request).update_weights(WeightTransferUpdateRequest(update_info=body.update_info))
    except Exception as error:
        _raise_control_error(error)
    return _control_response("Weights updated", acks)


@router.post("/finish_weight_update")
async def finish_weight_update(
    raw_request: Request,
    weight_version: Annotated[str | None, Body(embed=True)] = None,
) -> JSONResponse:
    try:
        acks = await _engine(raw_request).finish_weight_update(weight_version)
    except Exception as error:
        _raise_control_error(error)
    return _control_response("Weight update finished", acks)


@router.get("/get_world_size")
async def get_world_size(
    raw_request: Request,
    include_dp: bool = Query(True),
    stage_id: int | None = Query(None),
) -> JSONResponse:
    topology = _engine(raw_request).get_stage_topology(include_dp=include_dp)
    if stage_id is None:
        return JSONResponse(topology)
    stage = next((item for item in topology["stages"] if item["stage_id"] == stage_id), None)
    if stage is None:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=f"Unknown stage_id: {stage_id}")
    return JSONResponse({"world_size": stage["world_size"], "stages": [stage]})


@router.post("/get_weights_checksum")
async def get_weights_checksum(
    raw_request: Request,
    body: WeightsChecksumRequest | None = None,
) -> JSONResponse:
    body = body or WeightsChecksumRequest()
    try:
        acks = await _engine(raw_request).get_weights_checksum(
            stage_ids=body.stage_ids,
            component=body.component,
        )
    except Exception as error:
        _raise_control_error(error)
    return _control_response("Weights checksummed", acks)


@router.post("/reset_prefix_cache")
async def reset_prefix_cache(
    raw_request: Request,
    reset_external: bool = Query(False),
    reset_running_requests: bool = Query(False),
    stage_ids: list[int] | None = Query(None),
) -> Response:
    success = await _engine(raw_request).reset_prefix_cache(
        reset_running_requests=reset_running_requests,
        reset_connector=reset_external,
        stage_ids=stage_ids,
    )
    return Response(status_code=HTTPStatus.OK if success else HTTPStatus.INTERNAL_SERVER_ERROR)


@router.post("/reset_mm_cache")
async def reset_mm_cache(raw_request: Request, stage_ids: list[int] | None = Query(None)) -> Response:
    await _engine(raw_request).reset_mm_cache(stage_ids=stage_ids)
    return Response(status_code=HTTPStatus.OK)


@router.post("/reset_encoder_cache")
async def reset_encoder_cache(raw_request: Request, stage_ids: list[int] | None = Query(None)) -> Response:
    await _engine(raw_request).reset_encoder_cache(stage_ids=stage_ids)
    return Response(status_code=HTTPStatus.OK)


@router.post("/sleep")
async def sleep(
    raw_request: Request,
    level: int = Query(1),
    mode: PauseMode = Query("abort"),
    stage_ids: list[int] | None = Query(None),
) -> Response:
    engine = _engine(raw_request)
    await engine.pause_generation(mode=mode, clear_cache=True)
    try:
        await engine.sleep(stage_ids=stage_ids, level=level, mode=mode)
    except Exception:
        await engine.resume_generation()
        raise
    return Response(status_code=HTTPStatus.OK)


@router.post("/wake_up")
async def wake_up(
    raw_request: Request,
    tags: list[str] | None = Query(None),
    stage_ids: list[int] | None = Query(None),
) -> Response:
    engine = _engine(raw_request)
    await engine.wake_up(stage_ids=stage_ids, tags=tags)
    status = await engine.get_sleep_status()
    if not status["is_sleeping"]:
        await engine.resume_generation()
    return Response(status_code=HTTPStatus.OK)


@router.get("/is_sleeping")
async def is_sleeping(raw_request: Request) -> JSONResponse:
    engine = _engine(raw_request)
    if hasattr(engine, "get_sleep_status"):
        return JSONResponse(await engine.get_sleep_status())
    return JSONResponse({"is_sleeping": await engine.is_sleeping()})


@router.post("/v1/load_lora_adapter")
async def load_lora_adapter(body: LoadLoRARequest, raw_request: Request) -> JSONResponse:
    lora_request = LoRARequest(
        lora_name=body.lora_name,
        lora_int_id=stable_lora_int_id(body.lora_path),
        lora_path=body.lora_path,
        load_inplace=body.load_inplace,
        is_3d_lora_weight=body.is_3d_lora_weight,
    )
    try:
        acks = await _engine(raw_request).add_lora_with_acks(
            lora_request,
            stage_ids=body.stage_ids,
        )
    except Exception as error:
        _raise_control_error(error)
    serving_models = getattr(raw_request.app.state, "openai_serving_models", None)
    if serving_models is not None:
        serving_models.lora_requests[body.lora_name] = lora_request
    return _control_response(f"LoRA adapter '{body.lora_name}' loaded", acks)


@router.post("/v1/unload_lora_adapter")
async def unload_lora_adapter(body: UnloadLoRARequest, raw_request: Request) -> JSONResponse:
    serving_models = getattr(raw_request.app.state, "openai_serving_models", None)
    existing = None
    if serving_models is not None:
        existing = serving_models.lora_requests.get(body.lora_name)
    adapter_id = body.lora_int_id or getattr(existing, "lora_int_id", None)
    if adapter_id is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="lora_int_id is required when the adapter is not registered by this server",
        )
    try:
        acks = await _engine(raw_request).remove_lora_with_acks(
            adapter_id,
            stage_ids=body.stage_ids,
        )
    except Exception as error:
        _raise_control_error(error)
    if serving_models is not None:
        serving_models.lora_requests.pop(body.lora_name, None)
    return _control_response(f"LoRA adapter '{body.lora_name}' unloaded", acks)


@router.get("/server_info")
async def server_info(
    raw_request: Request,
    config_format: Literal["text", "json"] = Query("text"),
) -> JSONResponse:
    from vllm.entrypoints.serve.dev.server_info.api_router import (
        PydanticVllmConfig,
        _get_system_env_info_cached,
        _get_vllm_env_vars,
    )

    engine = _engine(raw_request)
    vllm_config = getattr(raw_request.app.state, "vllm_config", None)
    if vllm_config is None:
        serialized_config = None
    elif config_format == "text":
        serialized_config = str(vllm_config)
    else:
        serialized_config = PydanticVllmConfig.dump_python(vllm_config, mode="json", fallback=str)
    topology = engine.get_stage_topology(include_dp=True)
    try:
        system_env = await asyncio.to_thread(_get_system_env_info_cached)
    except Exception as error:
        system_env = {"error": str(error)}
    return JSONResponse(
        {
            "vllm_config": serialized_config,
            "vllm_env": _get_vllm_env_vars(),
            "system_env": system_env,
            "omni_rl": {
                "stage_aware": True,
                "rollout_transport": "shm",
                "topology": topology,
            },
        }
    )


def _public_handle(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or not value.get("preserve_for_client"):
        raise ValueError("Rollout tensors were not returned as shared-memory handles")
    return {key: item for key, item in value.items() if not key.startswith("__") and key != "preserve_for_client"}


@router.post("/rollout/generate")
async def rollout_generate(body: RolloutGenerateRequest, raw_request: Request) -> JSONResponse:
    engine = _engine(raw_request)
    stage_configs = list(getattr(engine, "stage_configs", []))
    diffusion_stage_ids = [index for index, config in enumerate(stage_configs) if get_stage_type(config) == "diffusion"]
    if not diffusion_stage_ids:
        raise HTTPException(status_code=HTTPStatus.NOT_IMPLEMENTED, detail="No diffusion stage is configured")
    stage_id = body.stage_id if body.stage_id is not None else diffusion_stage_ids[-1]
    if stage_id not in diffusion_stage_ids:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=f"stage_id {stage_id} is not a diffusion stage")

    try:
        diffusion_params = OmniDiffusionSamplingParams(**body.sampling_params)
    except TypeError as error:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(error)) from error
    diffusion_params.return_trajectory_latents = True
    diffusion_params.return_trajectory_handles = True
    sampling_params = build_stage_sampling_params_list(
        stage_configs,
        get_default_sampling_params_list(engine),
    )
    sampling_params[stage_id] = diffusion_params
    prompt = OmniTextPrompt(prompt=body.prompt)
    if body.negative_prompt is not None:
        prompt["negative_prompt"] = body.negative_prompt

    output = None
    request_id = body.request_id or f"rollout-{random_uuid()}"
    try:
        async for item in engine.generate(
            prompt=prompt,
            request_id=request_id,
            sampling_params_list=sampling_params,
        ):
            output = item
    except Exception as error:
        _raise_control_error(error)
    if output is None:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="No rollout output was produced")

    try:
        handles = {
            "latents": _public_handle(output.trajectory_latents),
            "timesteps": _public_handle(output.trajectory_timesteps),
            "log_probs": _public_handle(output.trajectory_log_probs),
        }
    except ValueError as error:
        raise HTTPException(status_code=HTTPStatus.UNPROCESSABLE_ENTITY, detail=str(error)) from error
    if all(handle is None for handle in handles.values()):
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            detail="The selected diffusion pipeline did not produce trajectory tensors",
        )
    return JSONResponse(
        {
            "request_id": output.request_id,
            "stage_id": output.stage_id if output.stage_id is not None else stage_id,
            "transport": "shm",
            "handles": handles,
            "metrics": jsonable_encoder(output.metrics),
        }
    )


def remove_overridden_routes(app: Any) -> None:
    """Remove upstream single-engine routes before attaching this router."""
    paths = {
        "/pause",
        "/resume",
        "/is_paused",
        "/abort_requests",
        "/init_weight_transfer_engine",
        "/start_weight_update",
        "/start_draft_weight_update",
        "/update_weights",
        "/finish_weight_update",
        "/get_world_size",
        "/get_weights_checksum",
        "/reset_prefix_cache",
        "/reset_mm_cache",
        "/reset_encoder_cache",
        "/sleep",
        "/wake_up",
        "/is_sleeping",
        "/rollout/generate",
        "/v1/load_lora_adapter",
        "/v1/unload_lora_adapter",
        "/server_info",
    }
    app.routes[:] = [route for route in app.routes if getattr(route, "path", None) not in paths]
