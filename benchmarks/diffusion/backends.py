import asyncio
import base64
import mimetypes
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from tqdm import tqdm


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    model: str
    width: int | None = None
    height: int | None = None
    num_frames: int | None = None
    num_inference_steps: int | None = None
    seed: int | None = None
    fps: int | None = None
    timestamp: float | None = None
    slo_ms: float | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    image_paths: list[str] | None = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class RequestFuncOutput:
    success: bool = False
    latency: float = 0.0
    error: str = ""
    start_time: float = 0.0
    response_body: dict[str, Any] = field(default_factory=dict)
    stage_durations: dict[str, float] = field(default_factory=dict)
    peak_memory_mb: float = 0.0
    slo_achieved: bool | None = None


def _guess_mime_type(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"


def _encode_image_as_data_url(path: str) -> str:
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    mime = _guess_mime_type(path)
    return f"data:{mime};base64,{encoded}"


async def async_request_chat_completions(
    input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
    enable_diffusion_pipeline_profiler: bool = False,
) -> RequestFuncOutput:
    output = RequestFuncOutput()
    output.start_time = time.perf_counter()

    extra_body = dict(input.extra_body)
    if input.width and input.height:
        extra_body.setdefault("height", input.height)
        extra_body.setdefault("width", input.width)
    if input.num_frames:
        extra_body.setdefault("num_frames", input.num_frames)
    if input.num_inference_steps:
        extra_body.setdefault("num_inference_steps", input.num_inference_steps)
    if input.seed is not None:
        extra_body.setdefault("seed", input.seed)
    if input.fps:
        extra_body.setdefault("fps", input.fps)

    if input.image_paths and len(input.image_paths) > 0:
        content = []
        if input.prompt:
            content.append({"type": "text", "text": input.prompt})
        for img_path in input.image_paths:
            if not os.path.exists(img_path):
                output.error = f"Image file not found: {img_path}"
                output.success = False
                if pbar:
                    pbar.update(1)
                return output
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _encode_image_as_data_url(img_path)},
                }
            )
        messages = [{"role": "user", "content": content}]
    else:
        messages = [{"role": "user", "content": input.prompt}]

    payload = {
        "model": input.model,
        "messages": messages,
    }
    if extra_body:
        payload["extra_body"] = extra_body

    try:
        async with session.post(input.api_url, json=payload) as response:
            if response.status == 200:
                resp_json = await response.json()
                output.response_body = resp_json
                output.success = True
                if "peak_memory_mb" in resp_json:
                    output.peak_memory_mb = resp_json["peak_memory_mb"]
                try:
                    choices = resp_json.get("choices", [])
                    if choices and isinstance(choices, list):
                        msg = choices[0].get("message", {})
                        if isinstance(msg, dict):
                            content = msg.get("content", [])
                            if content and isinstance(content, list) and len(content) > 0:
                                first_item = content[0]
                                if isinstance(first_item, dict):
                                    output.stage_durations = first_item.get("stage_durations")
                except (IndexError, TypeError, AttributeError):
                    pass
            else:
                output.error = f"HTTP {response.status}: {await response.text()}"
                output.success = False
    except Exception as e:
        output.error = str(e)
        output.success = False

    output.latency = time.perf_counter() - output.start_time

    if output.success and input.slo_ms is not None:
        output.slo_achieved = (output.latency * 1000.0) <= float(input.slo_ms)

    if pbar:
        pbar.update(1)
    return output


async def async_request_openai_images(
    input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    """
    Send request to OpenAI's /v1/images/generations endpoint.
    """
    output = RequestFuncOutput()
    output.start_time = time.perf_counter()

    # Build size string from width/height
    width = input.width or 1024
    height = input.height or 1024
    size = f"{width}x{height}"

    payload: dict[str, Any] = {
        "model": input.model,
        "prompt": input.prompt,
        "n": 1,
        "size": size,
        "response_format": "b64_json",
    }

    # Add optional parameters
    if input.seed is not None:
        payload["seed"] = input.seed
    if input.num_inference_steps is not None:
        payload["num_inference_steps"] = input.num_inference_steps

    # Add any extra body parameters
    if input.extra_body:
        for key, value in input.extra_body.items():
            if key not in payload:
                payload[key] = value

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer EMPTY",
    }

    try:
        async with session.post(input.api_url, json=payload, headers=headers) as response:
            if response.status == 200:
                resp_json = await response.json()
                output.response_body = resp_json
                output.success = True
                # Check for usage/memory info if available
                if "usage" in resp_json and "peak_memory_mb" in resp_json.get("usage", {}):
                    output.peak_memory_mb = resp_json["usage"]["peak_memory_mb"]
            else:
                output.error = f"HTTP {response.status}: {await response.text()}"
                output.success = False
    except Exception as e:
        output.error = str(e)
        output.success = False

    output.latency = time.perf_counter() - output.start_time

    if output.success and input.slo_ms is not None:
        output.slo_achieved = (output.latency * 1000.0) <= float(input.slo_ms)

    if pbar:
        pbar.update(1)
    return output


async def async_request_v1_videos(
    input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    output = RequestFuncOutput()
    output.start_time = time.perf_counter()

    files = dict(input.extra_body)
    if input.prompt:
        files.setdefault("prompt", input.prompt)
    if input.width and input.height:
        files.setdefault("height", input.height)
        files.setdefault("width", input.width)
    if input.num_frames:
        files.setdefault("num_frames", input.num_frames)
    if input.num_inference_steps:
        files.setdefault("num_inference_steps", input.num_inference_steps)
    if input.seed is not None:
        files.setdefault("seed", input.seed)
    if input.fps:
        files.setdefault("fps", input.fps)

    form = aiohttp.FormData()
    for k, v in files.items():
        form.add_field(k, str(v))

    image_file = None
    if input.image_paths and len(input.image_paths) > 0:
        image_path = input.image_paths[0]
        image_file = open(image_path, "rb")
        form.add_field(
            "input_reference",
            image_file,
            filename=os.path.basename(image_path),
            content_type="application/octet-stream",
        )

    job_id = None
    job_status = None
    poll_json = {}
    resp_json = {}

    try:
        # invoke a post request (POST /v1/videos)
        async with session.post(input.api_url, data=form) as response:
            if response.status == 200:
                resp_json = await response.json()
                job_id = resp_json.get("id")
                job_status = resp_json.get("status")
                if not job_id or not job_status:
                    output.error = "API response missing job 'id' or 'status' field."
                    output.success = False
                    return output
            else:
                output.error = f"HTTP {response.status}: {await response.text()}"
                output.success = False
                return output

        # invoke a poll request (GET /v1/videos/{video_id})
        poll_interval = 2.0  # Unit(s)
        timeout_seconds = 600.0
        deadline = time.perf_counter() + timeout_seconds
        job_url = f"{input.api_url}/{job_id}"

        while job_status not in {"completed", "failed"}:
            await asyncio.sleep(poll_interval)

            async with session.get(job_url) as poll_response:
                if poll_response.status != 200:
                    output.error = f"Polling failed HTTP {poll_response.status}: {await poll_response.text()}"
                    output.success = False
                    return output

                poll_json = await poll_response.json()
                job_status = poll_json.get("status")

                if time.perf_counter() >= deadline:
                    output.error = f"Timed out waiting for video job {job_id} to complete."
                    output.success = False
                    return output

        if job_status == "failed":
            output.error = f"Video job failed: {poll_json}"
            output.success = False
            return output

        # invoke a get request (GET /v1/videos/{video_id}/content)
        content_url = f"{job_url}/content"
        async with session.get(content_url) as content_response:
            if content_response.status != 200:
                output.error = (
                    f"Content retrieval failed HTTP {content_response.status}: {await content_response.text()}"
                )
                output.success = False
                return output

            video_bytes = await content_response.read()
            output.response_body = video_bytes
            output.success = True
            if "peak_memory_mb" in poll_json:
                output.peak_memory_mb = poll_json["peak_memory_mb"]
            elif "peak_memory_mb" in resp_json:
                output.peak_memory_mb = resp_json["peak_memory_mb"]
    except Exception as e:
        output.error = str(e)
        output.success = False
    finally:
        if image_file is not None:
            image_file.close()

        if job_id is not None:
            try:
                async with session.delete(f"{input.api_url}/{job_id}") as _:
                    pass
            except Exception as e:
                print(f"Failed to clean up video job {job_id}: {e}")

    output.latency = time.perf_counter() - output.start_time

    if output.success and input.slo_ms is not None:
        output.slo_achieved = (output.latency * 1000.0) <= float(input.slo_ms)

    if pbar:
        pbar.update(1)
    return output


backends_function_mapping = {
    "2i": {
        "vllm-omni": (async_request_chat_completions, "/v1/chat/completions"),
        "openai": (async_request_openai_images, "/v1/images/generations"),
    },
    "2v": {
        "v1/videos": (async_request_v1_videos, "/v1/videos"),
    },
}
