# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request I/O types and async request functions for the omni serving benchmark.

Reuses ``diffusion/backends.py`` for the base ``RequestFuncInput`` /
``RequestFuncOutput`` shapes and small image-encoding helpers, but keeps all
omni-specific state and senders here so the diffusion benchmark stays
unmodified:

* ``OmniRequest`` — a ``RequestFuncInput`` subclass carrying the client-side
  task label (``i2t`` / ``t2i`` / ``it2i``), the sampled ``bot_task``, and the
  send-time image caches (``image_data_urls`` / ``image_bytes``). None of these
  are sent to the server; the server resolves the task type itself from
  ``modalities`` plus the presence of a reference image (see
  ``serving_chat._resolve_omni_task_type``).
* ``OmniRequestFuncOutput`` — a ``RequestFuncOutput`` subclass adding the
  text-streaming metrics (``ttft`` / ``itl`` / ``output_tokens`` /
  ``generated_text``) captured by the streaming senders; non-stream senders
  leave them at their defaults.
* ``async_request_omni_chat`` — a thin wrapper over
  ``async_request_chat_completions`` that lifts ``modalities`` (and a few
  generation knobs) from ``extra_body`` into the JSON payload so the same
  ``/v1/chat/completions`` endpoint can carry i2t (``modalities=["text"]``) and
  t2i/it2i (``modalities=["image"]``) traffic, and that parses stage metrics
  from both the image-output (``content`` is a list) and text-output
  (``content`` is a string) response shapes.
* ``async_request_image_edits`` — an omni /v1/images/edits (non-stream) sender
  (the diffusion one is not imported) that reuses the shared form builder.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from tqdm import tqdm

# ``backends`` lives in the sibling ``diffusion`` benchmark directory. Make that
# directory importable regardless of the caller's CWD.
_DIFFUSION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "diffusion")
if _DIFFUSION_DIR not in sys.path:
    sys.path.insert(0, _DIFFUSION_DIR)

from backends import (  # noqa: E402  (import after sys.path tweak)
    DEFAULT_EDITS_BOT_TASK,
    RequestFuncInput,
    RequestFuncOutput,
    _encode_image_as_data_url,
    _guess_mime_type,
)

# Top-level request fields the OpenAI chat endpoint understands and that the
# omni benchmark drives through ``extra_body``.
_CHAT_TOPLEVEL_FROM_EXTRA = ("modalities", "max_tokens", "temperature", "top_p", "seed")

# Tasks the it2i bot_task can take on. Keep in sync with
# hunyuan_image3/prompt_utils._BOT_TASK_PRESETS (minus ``vanilla``/``None`` which
# are plain-mode and not part of the "thinking intensity" sweep).
IT2I_BOT_TASKS = ("recaption", "think", "think_recaption")


@dataclass
class OmniRequest(RequestFuncInput):
    """A chat/edits request tagged with its client-side task type.

    ``task_type`` and ``bot_task`` are benchmark-side bookkeeping only — they
    drive per-task / per-bot_task statistics and never enter the HTTP payload.
    The server classifies the request from ``modalities`` (+ reference image).

    ``image_data_urls`` / ``image_bytes`` are send-time caches populated by
    ``prepare_request_images`` (pre-read + pre-encode the input images once) so
    the senders skip file I/O and base64 under concurrency. They live here on
    the omni subclass — the diffusion ``RequestFuncInput`` is left untouched.
    """

    task_type: str = ""
    bot_task: str | None = None
    # Bookkeeping only (never sent). Under --randomize-input this records the
    # generated input image's "WxH" so the tally / JSON can show the realized
    # input-resolution distribution without re-reading the image file.
    input_image_size: str | None = None
    # Send-time caches (never sent). Populated by prepare_request_images.
    image_data_urls: list[str] | None = None
    image_bytes: list[bytes] | None = None


@dataclass
class OmniRequestFuncOutput(RequestFuncOutput):
    """Omni-benchmark output — adds text-streaming metrics on top of the
    diffusion ``RequestFuncOutput``.

    ``ttft`` / ``itl`` / ``output_tokens`` / ``generated_text`` are only set by
    the streaming senders (AR-text TTFT/ITL); non-stream senders leave the
    defaults so the aggregate stats can read them uniformly (``o.ttft <= 0.0``
    marks a request that produced no AR text). Kept on this subclass so the
    diffusion ``RequestFuncOutput`` is left untouched.
    """

    ttft: float = 0.0
    itl: list[float] = field(default_factory=list)
    output_tokens: int = 0
    generated_text: str = ""


def prepare_request_images(req: OmniRequest) -> None:
    """Pre-read + pre-encode ``req``'s input images into on-request caches.

    Run once in a prep pass (outside the timed send) so the senders skip file
    I/O and base64 at send time. This keeps the event loop from blocking under
    concurrency and — together with ordered dispatch in the benchmark driver —
    makes the actual send order match the dry-run order exactly, instead of
    being raced by per-task prep cost: t2i has no image, it2i builds a multipart
    form, i2t base64-encodes a data URL, so without pre-encoding the
    fastest-prep group reaches the server first regardless of list order.
    Idempotent; a no-op for requests without input images.
    """
    if not req.image_paths:
        return
    if req.image_data_urls is not None and req.image_bytes is not None:
        return
    urls: list[str] = []
    raws: list[bytes] = []
    for p in req.image_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        with open(p, "rb") as f:
            raw = f.read()
        raws.append(raw)
        mime = _guess_mime_type(p)
        urls.append(f"data:{mime};base64,{base64.b64encode(raw).decode('utf-8')}")
    if req.image_data_urls is None:
        req.image_data_urls = urls
    if req.image_bytes is None:
        req.image_bytes = raws


def _parse_chat_stage_metrics(resp_json: dict[str, Any]) -> tuple[dict[str, float], float]:
    """Extract (stage_durations, peak_memory_mb) from a chat-completions body.

    Image-output responses put them on ``choices[0].message.content[0]``; the
    text-output path (i2t) puts them on the root-level ``metrics`` object.
    """
    stage_durations: dict[str, float] = {}
    peak_memory_mb = 0.0

    choices = resp_json.get("choices", [])
    if isinstance(choices, list) and choices:
        content = (choices[0] or {}).get("message", {}).get("content")
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict):
                sd = first.get("stage_durations")
                if isinstance(sd, dict):
                    stage_durations = {str(k): float(v) for k, v in sd.items()}
                pm = first.get("peak_memory_mb")
                if isinstance(pm, (int, float)):
                    peak_memory_mb = float(pm)

    if not stage_durations or peak_memory_mb == 0.0:
        metrics = resp_json.get("metrics")
        if isinstance(metrics, dict):
            if not stage_durations:
                sd = metrics.get("stage_durations")
                if isinstance(sd, dict):
                    stage_durations = {str(k): float(v) for k, v in sd.items()}
            if peak_memory_mb == 0.0:
                pm = metrics.get("peak_memory_mb")
                if isinstance(pm, (int, float)):
                    peak_memory_mb = float(pm)

    return stage_durations, peak_memory_mb


def _first_message_content(resp_json: dict[str, Any]) -> Any:
    """Return ``choices[0].message.content`` (str for i2t, list for t2i/it2i) or None."""
    choices = resp_json.get("choices", [])
    if isinstance(choices, list) and choices:
        return (choices[0] or {}).get("message", {}).get("content")
    return None


def extract_chat_outputs(resp_json: dict[str, Any]) -> tuple[str | None, str | None, int]:
    """Extract (returned_text, cot_output, num_images) from a chat-completions body.

    * i2t (text output): ``content`` is a string -> the model's answer.
    * t2i / it2i (image output): ``content`` is a list of ``image_url`` items;
      returned_text is None and num_images counts the image items.

    ``cot_output`` (the AR-stage "thinking" text) is NOT exposed by the current
    chat image response, so it is returned as None on this path. The lookup is
    kept tolerant: if a future server change surfaces it on a content item it
    will be picked up without a benchmark change. Use the /v1/images/edits
    endpoint (see :func:`extract_edits_outputs`) to obtain the AR thinking text
    today — its response carries ``cot_output`` at the root.
    """
    content = _first_message_content(resp_json)
    if isinstance(content, str):
        return content or None, None, 0
    if isinstance(content, list):
        cot_output: str | None = None
        num_images = 0
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image_url":
                num_images += 1
            if cot_output is None:
                co = item.get("cot_output")
                if isinstance(co, str) and co.strip():
                    cot_output = co
        return None, cot_output, num_images
    return None, None, 0


def extract_edits_outputs(resp_json: dict[str, Any]) -> tuple[str | None, int]:
    """Extract (cot_output, num_images) from a /v1/images/edits body.

    The edits response is an ``ImageGenerationResponse``: ``data`` is a list of
    ``{b64_json}`` image items and ``cot_output`` sits at the root — this is the
    only endpoint that currently surfaces the AR-stage "thinking" text.
    """
    cot_output = resp_json.get("cot_output") if isinstance(resp_json, dict) else None
    if not (isinstance(cot_output, str) and cot_output.strip()):
        cot_output = None
    data = resp_json.get("data", []) if isinstance(resp_json, dict) else []
    num_images = sum(1 for d in data if isinstance(d, dict) and d.get("b64_json"))
    return cot_output, num_images


async def async_request_omni_chat(
    input: OmniRequest,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
) -> OmniRequestFuncOutput:
    """POST /v1/chat/completions for any of i2t / t2i / it2i.

    Mirrors ``async_request_chat_completions`` but hoists ``modalities`` (and a
    few generation knobs) from ``extra_body`` to the payload root, and parses
    stage metrics for both image- and text-output response shapes.
    """
    output = OmniRequestFuncOutput()
    output.start_time = time.perf_counter()

    extra_body = dict(input.extra_body)
    modalities = extra_body.pop("modalities", None)

    # Build the message content. Image-conditioned tasks (i2t, it2i) attach the
    # input image; t2i is plain text.
    content: list[dict[str, Any]] = []

    if input.image_paths:
        cached_urls = input.image_data_urls
        for idx, img_path in enumerate(input.image_paths):
            if cached_urls is not None and idx < len(cached_urls):
                url = cached_urls[idx]
            else:
                if not os.path.exists(img_path):
                    output.error = f"Image file not found: {img_path}"
                    output.success = False
                    if pbar:
                        pbar.update(1)
                    return output
                url = _encode_image_as_data_url(img_path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": url},
                }
            )

    if input.prompt:
        content.append({"type": "text", "text": input.prompt})

    messages = [{"role": "user", "content": content}]

    payload: dict[str, Any] = {
        "model": input.model,
        "messages": messages,
        "max_tokens": 512,
    }
    if modalities is not None:
        payload["modalities"] = modalities
    if input.width and input.height:
        payload["height"] = input.height
        payload["width"] = input.width
    if input.num_inference_steps is not None:
        payload["num_inference_steps"] = input.num_inference_steps
    if input.seed is not None:
        payload["seed"] = input.seed

    # Generation knobs the caller may have parked in extra_body.
    for key in _CHAT_TOPLEVEL_FROM_EXTRA:
        if key in extra_body and key not in payload:
            payload[key] = extra_body[key]
    # Anything else (e.g. negative_prompt, guidance_scale, bot_task for it2i)
    # goes through extra_body as the diffusion path expects.
    rest = {k: v for k, v in extra_body.items() if k not in _CHAT_TOPLEVEL_FROM_EXTRA}
    if rest:
        payload["extra_body"] = rest

    try:
        async with session.post(input.api_url, json=payload) as response:
            if response.status == 200:
                resp_json = await response.json()
                output.response_body = resp_json
                output.success = True
                output.stage_durations, output.peak_memory_mb = _parse_chat_stage_metrics(resp_json)
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


def _build_chat_payload(input: OmniRequest) -> dict[str, Any]:
    """Build the /v1/chat/completions JSON body shared by the stream / non-stream senders."""
    extra_body = dict(input.extra_body)
    modalities = extra_body.pop("modalities", None)

    content: list[dict[str, Any]] = []
    if input.prompt:
        content.append({"type": "text", "text": input.prompt})
    if input.image_paths:
        cached_urls = input.image_data_urls
        for idx, img_path in enumerate(input.image_paths):
            if cached_urls is not None and idx < len(cached_urls):
                url = cached_urls[idx]
            else:
                if not os.path.exists(img_path):
                    raise FileNotFoundError(img_path)
                url = _encode_image_as_data_url(img_path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": url},
                }
            )
    messages = [{"role": "user", "content": content}]

    payload: dict[str, Any] = {"model": input.model, "messages": messages}
    if modalities is not None:
        payload["modalities"] = modalities
    if input.width and input.height:
        payload["height"] = input.height
        payload["width"] = input.width
    if input.num_inference_steps is not None:
        payload["num_inference_steps"] = input.num_inference_steps
    if input.seed is not None:
        payload["seed"] = input.seed
    for key in _CHAT_TOPLEVEL_FROM_EXTRA:
        if key in extra_body and key not in payload:
            payload[key] = extra_body[key]
    rest = {k: v for k, v in extra_body.items() if k not in _CHAT_TOPLEVEL_FROM_EXTRA}
    if rest:
        payload["extra_body"] = rest
    return payload


async def async_request_omni_chat_stream(
    input: OmniRequest,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
) -> OmniRequestFuncOutput:
    """Streaming /v1/chat/completions that captures text-output TTFT / ITL.

    Adds ``stream: True`` (+ ``include_usage``) and parses the SSE stream the
    same way vLLM's ``async_request_openai_chat_completions`` does:

    * i2t (``modalities=["text"]``): the AR stage is terminal and streams its
      answer as ``choices[0].delta.content`` text deltas — TTFT is the time to
      the first such delta, ITL is the per-delta gap, ``output_tokens`` comes
      from the final ``usage.completion_tokens`` (falling back to the delta
      count).
    * t2i / it2i (``modalities=["image"]``): the server filters the AR "thinking"
      text out of the chat stream (only the final image chunk is emitted), so
      text TTFT/ITL stay 0 / N/A here — use ``--it2i-endpoint images-edits`` for
      it2i AR-text latency. The image chunk is still parsed so the generated
      image + stage metrics are reconstructed for the existing output path.

    ``response_body`` is reconstructed from the stream so the non-stream
    extraction / image-save logic in ``omni_benchmark_serving`` works unchanged.
    """
    output = OmniRequestFuncOutput()
    output.start_time = time.perf_counter()

    try:
        payload = _build_chat_payload(input)
    except FileNotFoundError as e:
        output.error = f"Image file not found: {e}"
        output.success = False
        if pbar:
            pbar.update(1)
        return output
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}

    generated_text = ""
    ttft = 0.0
    itl: list[float] = []
    most_recent = output.start_time
    output_tokens = 0
    image_content: list[dict[str, Any]] = []
    finish_reason: str | None = None
    metrics: dict[str, Any] | None = None

    try:
        async with session.post(input.api_url, json=payload) as response:
            if response.status != 200:
                output.error = f"HTTP {response.status}: {await response.text()}"
                output.success = False
            else:
                async for data in _iter_sse_events(response):
                    choices = data.get("choices")
                    if isinstance(choices, list) and choices:
                        choice = choices[0] or {}
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        # Text delta (i2t AR answer; also AR CoT if the server
                        # ever streams it for image tasks).
                        if isinstance(content, str) and content:
                            ts = time.perf_counter()
                            if ttft == 0.0:
                                ttft = ts - output.start_time
                            else:
                                itl.append(ts - most_recent)
                            generated_text += content
                            most_recent = ts
                        # Image delta (chat t2i/it2i): content is a list of
                        # {type:image_url, image_url:{url}, stage_durations, ...}.
                        elif isinstance(content, list) and content:
                            image_content.extend(content)
                        if choice.get("finish_reason"):
                            finish_reason = choice.get("finish_reason")
                    usage = data.get("usage")
                    if isinstance(usage, dict) and usage.get("completion_tokens"):
                        output_tokens = int(usage["completion_tokens"])
                    m = data.get("metrics")
                    if isinstance(m, dict):
                        metrics = m
                output.success = True
    except Exception as e:
        output.error = str(e)
        output.success = False

    output.latency = time.perf_counter() - output.start_time
    output.ttft = ttft
    output.itl = itl
    output.generated_text = generated_text
    if output_tokens:
        output.output_tokens = output_tokens
    elif itl:
        # No usage chunk (server may omit usage for image tasks); approximate
        # token count by the number of text deltas (== itl entries + 1).
        output.output_tokens = len(itl) + 1

    # Reconstruct a non-stream-equivalent body so extract_chat_outputs /
    # _collect_image_urls / _parse_chat_stage_metrics keep working.
    if output.success:
        if image_content:
            output.response_body = {
                "choices": [
                    {
                        "index": 0,
                        "message": {"content": image_content},
                        "finish_reason": finish_reason or "stop",
                    }
                ]
            }
        else:
            output.response_body = {
                "choices": [
                    {
                        "index": 0,
                        "message": {"content": generated_text},
                        "finish_reason": finish_reason or "stop",
                    }
                ]
            }
        if metrics is not None:
            output.response_body["metrics"] = metrics
        output.stage_durations, output.peak_memory_mb = _parse_chat_stage_metrics(output.response_body)

    if output.success and input.slo_ms is not None:
        output.slo_achieved = (output.latency * 1000.0) <= float(input.slo_ms)

    if pbar:
        pbar.update(1)
    return output


def _build_edits_form(
    input: OmniRequest,
    stream: bool,
    *,
    with_stage_metrics_flag: bool = True,
) -> aiohttp.FormData:
    """Build the /v1/images/edits multipart form (shared by stream / non-stream).

    Raises FileNotFoundError if a referenced image is missing so the caller can
    surface it as a per-request error without sending.

    ``with_stage_metrics_flag`` gates the ``return_stage_metrics`` form field.
    The streaming sender forwards it (the server emits per-stage timings on the
    stream); the non-stream sender omits it to match the diffusion edits sender
    it replaces (which never forwarded that flag).
    """
    extra_body = dict(input.extra_body)
    width = input.width or extra_body.get("width") or 1024
    height = input.height or extra_body.get("height") or 1024

    form = aiohttp.FormData()
    form.add_field("model", input.model)
    form.add_field("prompt", input.prompt)
    form.add_field("size", f"{width}x{height}")
    form.add_field("response_format", "b64_json")
    if stream:
        form.add_field("stream", "true")

    if input.num_inference_steps is not None:
        form.add_field("num_inference_steps", str(input.num_inference_steps))
    elif extra_body.get("num_inference_steps") is not None:
        form.add_field("num_inference_steps", str(extra_body["num_inference_steps"]))
    if input.seed is not None:
        form.add_field("seed", str(input.seed))
    elif extra_body.get("seed") is not None:
        form.add_field("seed", str(extra_body["seed"]))
    if extra_body.get("guidance_scale") is not None:
        form.add_field("guidance_scale", str(extra_body["guidance_scale"]))
    if extra_body.get("negative_prompt") is not None:
        form.add_field("negative_prompt", str(extra_body["negative_prompt"]))
    if extra_body.get("true_cfg_scale") is not None:
        form.add_field("true_cfg_scale", str(extra_body["true_cfg_scale"]))
    if extra_body.get("sys_type") is not None:
        form.add_field("sys_type", str(extra_body["sys_type"]))
    if extra_body.get("system_prompt") is not None:
        form.add_field("system_prompt", str(extra_body["system_prompt"]))
    if with_stage_metrics_flag and extra_body.get("return_stage_metrics"):
        form.add_field("return_stage_metrics", "true")

    bot_task = input.bot_task or extra_body.get("bot_task") or input.default_bot_task
    if bot_task is not None:
        form.add_field("bot_task", str(bot_task))

    assert input.image_paths is not None
    cached_bytes = input.image_bytes
    for idx, img_path in enumerate(input.image_paths):
        if cached_bytes is not None and idx < len(cached_bytes):
            image_bytes = cached_bytes[idx]
        else:
            if not os.path.exists(img_path):
                raise FileNotFoundError(img_path)
            with open(img_path, "rb") as img_f:
                image_bytes = img_f.read()
        form.add_field(
            "image",
            image_bytes,
            filename=os.path.basename(img_path),
            content_type=_guess_mime_type(img_path),
        )
    return form


async def async_request_image_edits(
    input: OmniRequest,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
) -> OmniRequestFuncOutput:
    """POST /v1/images/edits (multipart, non-streaming) for it2i.

    Omni-benchmark counterpart of the diffusion edits sender: builds the form
    via ``_build_edits_form`` (which honors the ``input.image_bytes`` cache so
    the send skips file I/O) and returns a ``OmniRequestFuncOutput`` so the
    aggregate stats can read ``ttft`` / ``itl`` / ``output_tokens`` /
    ``generated_text`` uniformly — non-stream leaves them at their defaults
    (``ttft == 0.0`` marks "no AR text captured").
    """
    output = OmniRequestFuncOutput()
    output.start_time = time.perf_counter()

    try:
        form = _build_edits_form(input, stream=False, with_stage_metrics_flag=False)
    except FileNotFoundError as e:
        output.error = f"Image file not found: {e}"
        output.success = False
        if pbar:
            pbar.update(1)
        return output

    try:
        async with session.post(input.api_url, data=form) as response:
            if response.status == 200:
                resp_json = await response.json()
                output.response_body = resp_json
                output.success = True
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


async def async_request_image_edits_stream(
    input: OmniRequest,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
) -> OmniRequestFuncOutput:
    """Streaming /v1/images/edits that captures AR-text TTFT / ITL for it2i.

    The multi-stage edits stream emits ``ImageEditARDeltaChunk`` (``type=
    "ar_delta"``, ``delta`` = an AR-stage text token) for each AR decode step,
    then a single ``ImageEditImageChunk`` (``type="image"``) with the final
    image. TTFT is the time to the first ``ar_delta``; ITL is the per-delta gap;
    ``output_tokens`` is the ``ar_delta`` count (each ≈ one AR decode step); the
    concatenated deltas are the AR "thinking" (CoT) text. The image + stage
    metrics are reconstructed into ``response_body`` so the existing edits output
    path (``extract_edits_outputs`` / image save) works unchanged.
    """
    output = OmniRequestFuncOutput()
    output.start_time = time.perf_counter()

    try:
        form = _build_edits_form(input, stream=True)
    except FileNotFoundError as e:
        output.error = f"Image file not found: {e}"
        output.success = False
        if pbar:
            pbar.update(1)
        return output

    generated_text = ""
    ttft = 0.0
    itl: list[float] = []
    most_recent = output.start_time
    ar_delta_count = 0
    images_b64: list[str] = []
    stage_durations: dict[str, float] = {}
    peak_memory_mb = 0.0
    stream_error: str | None = None

    try:
        async with session.post(input.api_url, data=form) as response:
            if response.status != 200:
                output.error = f"HTTP {response.status}: {await response.text()}"
                output.success = False
            else:
                async for data in _iter_sse_events(response):
                    obj = data.get("object")
                    ctype = data.get("type")
                    if obj == "error":
                        err = data.get("error")
                        if isinstance(err, dict):
                            stream_error = str(err.get("message") or err)
                        else:
                            stream_error = str(err)
                        continue
                    if ctype == "ar_delta":
                        delta = data.get("delta") or ""
                        if delta:
                            ts = time.perf_counter()
                            if ttft == 0.0:
                                ttft = ts - output.start_time
                            else:
                                itl.append(ts - most_recent)
                            generated_text += delta
                            most_recent = ts
                            ar_delta_count += 1
                        # AR-stage metrics ride on ar_delta chunks; merge them in
                        # (the DiT-stage metrics come on the final image chunk).
                        m = data.get("metrics")
                        if isinstance(m, dict):
                            _merge_stage_metrics(m, stage_durations)
                    elif ctype == "image":
                        for d in data.get("data", []) or []:
                            if isinstance(d, dict) and d.get("b64_json"):
                                images_b64.append(d["b64_json"])
                        m = data.get("metrics")
                        if isinstance(m, dict):
                            _merge_stage_metrics(m, stage_durations)
                            pm = m.get("peak_memory_mb")
                            if isinstance(pm, (int, float)):
                                peak_memory_mb = float(pm)
                if stream_error:
                    output.error = stream_error
                    output.success = False
                else:
                    output.success = True
    except Exception as e:
        output.error = str(e)
        output.success = False

    output.latency = time.perf_counter() - output.start_time
    output.ttft = ttft
    output.itl = itl
    output.output_tokens = ar_delta_count
    output.generated_text = generated_text
    output.stage_durations = stage_durations
    output.peak_memory_mb = peak_memory_mb

    if output.success:
        output.response_body = {
            "data": [{"b64_json": b64} for b64 in images_b64],
            "cot_output": generated_text or None,
        }

    if output.success and input.slo_ms is not None:
        output.slo_achieved = (output.latency * 1000.0) <= float(input.slo_ms)

    if pbar:
        pbar.update(1)
    return output


def select_request_func(req: OmniRequest, it2i_endpoint: str, stream: bool = False):
    """Pick the sender for a request given the configured it2i endpoint + stream.

    ``it2i_endpoint`` is ``"chat"`` (default, unified) or ``"images-edits"``.
    i2t and t2i always go through chat; it2i follows the configured endpoint.
    ``stream=True`` selects the streaming variant of each sender so the benchmark
    can capture text-output TTFT/ITL (i2t AR answer; it2i AR thinking via edits).
    """
    if req.task_type == "it2i" and it2i_endpoint == "images-edits":
        return async_request_image_edits_stream if stream else async_request_image_edits
    return async_request_omni_chat_stream if stream else async_request_omni_chat


def _merge_stage_metrics(
    metrics: dict[str, Any],
    target: dict[str, float],
) -> None:
    """Fold a chunk's ``metrics.stage_durations`` into ``target`` (first write wins)."""
    sd = metrics.get("stage_durations")
    if isinstance(sd, dict):
        for k, v in sd.items():
            target.setdefault(str(k), float(v))


async def _iter_sse_events(
    response: aiohttp.ClientResponse,
) -> AsyncGenerator[dict[str, Any], None]:
    """Yield parsed JSON payloads from an SSE ``text/event-stream`` response.

    Mirrors vLLM's ``async_request_openai_chat_completions`` chunk loop: skip
    blank lines and keep-alive comments (``: ...``), strip the ``data: `` prefix,
    stop at ``[DONE]``, and ``json.loads`` the rest. Malformed lines are skipped
    rather than raised so a trailing partial line never aborts the whole stream.
    """
    async for raw in response.content:
        line = raw.strip()
        if not line:
            continue
        text = line.decode("utf-8", errors="replace")
        if text.startswith(":"):
            continue
        if not text.startswith("data:"):
            continue
        payload = text.removeprefix("data:").strip()
        if payload == "[DONE]":
            break
        try:
            yield json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            continue


__all__ = [
    "IT2I_BOT_TASKS",
    "OmniRequest",
    "OmniRequestFuncOutput",
    "DEFAULT_EDITS_BOT_TASK",
    "prepare_request_images",
    "async_request_omni_chat",
    "async_request_omni_chat_stream",
    "async_request_image_edits",
    "async_request_image_edits_stream",
    "select_request_func",
    "extract_chat_outputs",
    "extract_edits_outputs",
]
