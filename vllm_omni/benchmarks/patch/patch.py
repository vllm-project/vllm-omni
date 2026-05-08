import asyncio
import contextlib
import io
import json
import os
import random
import ssl
import sys
import time
import traceback
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import aiohttp
import pybase64 as base64
from pydub import AudioSegment
from tqdm.asyncio import tqdm
from vllm.benchmarks import datasets
from vllm.benchmarks.datasets import SampleRequest
from vllm.benchmarks.lib.endpoint_request_func import (
    ASYNC_REQUEST_FUNCS,
    OPENAI_COMPATIBLE_BACKENDS,
    RequestFuncInput,
    RequestFuncOutput,
    StreamedResponseHandler,
    _get_chat_content,
    _update_headers_common,
    _update_payload_common,
    _validate_api_url,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike

logger = init_logger(__name__)

from vllm_omni.benchmarks.data_modules.daily_omni_dataset import DailyOmniDataset, DailyOmniSampleRequest
from vllm_omni.benchmarks.data_modules.random_multi_modal_dataset import OmniRandomMultiModalDataset
from vllm_omni.benchmarks.data_modules.seed_tts_dataset import (
    SEED_TTS_DEFAULT_OMNI_SYSTEM_PROMPT,
    SeedTTSDataset,
    SeedTTSDesignDataset,
    SeedTTSSampleRequest,
    SeedTTSTextDataset,
)

get_samples_old = datasets.get_samples

_DEFAULT_DAILY_OMNI_REPO = "liarliar/Daily-Omni"


def _seed_tts_capture_pcm_for_wer() -> bool:
    return os.environ.get("SEED_TTS_WER_EVAL", "").lower() in (
        "1",
        "true",
        "yes",
    )


def _merge_extra_body_mm_kwargs(base: dict | None, overlay: dict | None) -> dict | None:
    """Shallow-merge ``extra_body`` dicts; deep-merge ``mm_processor_kwargs`` if both set."""
    if not base and not overlay:
        return None
    out = dict(base or {})
    if not overlay:
        return out
    for k, v in overlay.items():
        if k == "mm_processor_kwargs" and isinstance(v, dict):
            prev = out.get("mm_processor_kwargs")
            merged_kw = {**(prev if isinstance(prev, dict) else {}), **v}
            out["mm_processor_kwargs"] = merged_kw
        else:
            out[k] = v
    return out


def _attach_daily_omni_to_request_func_input(sample: SampleRequest, rfi: RequestFuncInput) -> None:
    """Apply per-request OpenAI fields (``mm_processor_kwargs``, messages) for Daily-Omni."""
    if not isinstance(sample, DailyOmniSampleRequest):
        return
    rfi.extra_body = _merge_extra_body_mm_kwargs(rfi.extra_body, sample.omni_extra_body)
    if sample.omni_chat_messages is not None:
        setattr(rfi, "omni_chat_messages", sample.omni_chat_messages)
    else:
        setattr(rfi, "mm_position", sample.omni_chat_mm_position)


def _attach_seed_tts_to_request_func_input(sample: SampleRequest, rfi: RequestFuncInput) -> None:
    """Merge Seed-TTS per-row TTS fields into ``extra_body`` and mark for PCM capture.

    Always sets ``seed_tts_row=True`` on the RequestFuncInput for any
    :class:`SeedTTSSampleRequest` subclass (including text-only and design
    variants that carry no ``ref_audio``).  This enables PCM capture for WER /
    UTMOS evaluation even when there is no reference audio.
    """
    if not isinstance(sample, SeedTTSSampleRequest):
        return
    # Mark for PCM capture (WER / UTMOS eval) regardless of extra body presence.
    setattr(rfi, "seed_tts_row", True)
    sys_prompt = (sample.seed_tts_system_prompt or "").strip() or SEED_TTS_DEFAULT_OMNI_SYSTEM_PROMPT
    setattr(
        rfi,
        "omni_chat_messages",
        [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": sample.prompt}]},
        ],
    )
    ex = sample.seed_tts_speech_extra
    if not ex:
        return  # voice comes from --extra-body in config; no ref_audio to merge
    base = dict(rfi.extra_body) if rfi.extra_body else {}
    base.update(ex)
    rfi.extra_body = base


def _daily_omni_repo_from_args(args) -> str | None:
    """Resolve HuggingFace repo id for Daily-Omni from CLI args.

    vLLM allows ``--dataset-path`` to be a local path while the real HF id is
    passed via ``--hf-name``. Upstream ``get_samples`` for ``hf`` only matches
    a fixed elif-chain and never discovers Omni's loader, so we must detect
    Daily-Omni here using either field.
    """
    dp = getattr(args, "dataset_path", None)
    hn = getattr(args, "hf_name", None)
    if dp in DailyOmniDataset.SUPPORTED_DATASET_PATHS:
        return dp
    if hn in DailyOmniDataset.SUPPORTED_DATASET_PATHS:
        return hn
    return None


def get_samples(args, tokenizer):
    # Daily-Omni: explicit dataset name, or hf + matching path/hf-name
    is_daily_omni = args.dataset_name == "daily-omni" or (
        args.dataset_name == "hf" and _daily_omni_repo_from_args(args) is not None
    )
    is_seed_tts = args.dataset_name in ("seed-tts", "seed-tts-text", "seed-tts-design")

    # Check if we need to handle omni-related backends/datasets
    is_omni_backend = args.backend in ["openai-chat-omni", "openai-audio-speech", "daily-omni"]
    is_omni_dataset = is_daily_omni or is_seed_tts or args.dataset_name == "random-mm"

    if not is_omni_backend and not is_omni_dataset:
        # Not an omni-related request, delegate to original implementation
        return get_samples_old(args, tokenizer)

    # Handle Daily-Omni dataset
    if is_daily_omni:
        # Support:
        #   --dataset-name daily-omni [--dataset-path liarliar/Daily-Omni]
        #   --dataset-name daily-omni --daily-omni-qa-json /path/to/qa.json  (offline QA)
        #   --dataset-name hf --dataset-path liarliar/Daily-Omni
        #   --dataset-name hf --hf-name liarliar/Daily-Omni  (dataset-path may be local)

        # Validate backend supports multimodal (video)
        if args.backend not in ["openai-chat-omni", "daily-omni"]:
            raise ValueError(
                f"Daily-Omni dataset requires a multimodal backend that supports video. "
                f"Got backend='{args.backend}'. Please use '--backend openai-chat-omni'"
            )

        # Determine video directory if specified (for local video files)
        video_dir = getattr(args, "daily_omni_video_dir", None)

        # Get HF split (default to "train"; unused when loading from local qa.json)
        dataset_split = getattr(args, "hf_split", None) or "train"

        qa_json = getattr(args, "daily_omni_qa_json", None)
        if isinstance(qa_json, str):
            qa_json = qa_json.strip() or None

        if qa_json is not None:
            logger.info(
                "Loading Daily-Omni dataset: qa_json=%s, video_dir=%s (Hub not used for QA)",
                qa_json,
                video_dir,
            )
            dataset = DailyOmniDataset(
                qa_json_path=qa_json,
                dataset_path=None,
                dataset_split=dataset_split,
                random_seed=args.seed,
                video_dir=video_dir,
                input_mode=getattr(args, "daily_omni_input_mode", "all"),
                inline_local_video=getattr(args, "daily_omni_inline_local_video", False),
                trust_remote_code=getattr(args, "trust_remote_code", False),
                disable_shuffle=getattr(args, "disable_shuffle", False),
            )
        else:
            repo_id = _daily_omni_repo_from_args(args)
            if args.dataset_name == "daily-omni":
                if repo_id is None:
                    repo_id = _DEFAULT_DAILY_OMNI_REPO
            elif repo_id is None:
                raise ValueError(
                    "Daily-Omni with --dataset-name hf requires "
                    f"--dataset-path {_DEFAULT_DAILY_OMNI_REPO} or "
                    f"--hf-name {_DEFAULT_DAILY_OMNI_REPO}."
                )

            logger.info(
                "Loading Daily-Omni dataset: hf_repo=%s, split=%s, video_dir=%s",
                repo_id,
                dataset_split,
                video_dir,
            )

            dataset = DailyOmniDataset(
                dataset_path=repo_id,
                dataset_split=dataset_split,
                dataset_subset=getattr(args, "hf_subset", None),
                random_seed=args.seed,
                video_dir=video_dir,
                input_mode=getattr(args, "daily_omni_input_mode", "all"),
                inline_local_video=getattr(args, "daily_omni_inline_local_video", False),
                trust_remote_code=getattr(args, "trust_remote_code", False),
                no_stream=getattr(args, "no_stream", False),
                disable_shuffle=getattr(args, "disable_shuffle", False),
            )

        out_len = getattr(args, "output_len", None)
        if out_len is None:
            out_len = getattr(args, "hf_output_len", None)
        if out_len is None:
            out_len = DailyOmniDataset.DEFAULT_OUTPUT_LEN

        input_requests = dataset.sample(
            tokenizer=tokenizer,
            num_requests=args.num_prompts,
            output_len=out_len,
            request_id_prefix=args.request_id_prefix,
            no_oversample=args.no_oversample,
        )
        return input_requests

    if is_seed_tts:
        if args.backend not in ("openai-audio-speech", "openai-chat-omni"):
            raise ValueError(
                "Seed-TTS requires --backend openai-audio-speech (POST /v1/audio/speech) or "
                "--backend openai-chat-omni (POST /v1/chat/completions with ref_audio/ref_text). "
                f"Got backend={args.backend!r}."
            )
        repo_id = getattr(args, "dataset_path", None) or getattr(args, "hf_name", None)
        if not repo_id:
            raise ValueError(
                "Seed-TTS requires --dataset-path (HF dataset repo id or local directory) or "
                "--hf-name for the Hub dataset id."
            )

        _cls_map = {
            "seed-tts": SeedTTSDataset,
            "seed-tts-text": SeedTTSTextDataset,
            "seed-tts-design": SeedTTSDesignDataset,
        }
        DatasetCls = _cls_map[args.dataset_name]
        dataset = DatasetCls(
            dataset_path=repo_id,
            random_seed=args.seed,
            locale=getattr(args, "seed_tts_locale", "en"),
            inline_ref_audio=not getattr(args, "seed_tts_file_ref_audio", False),
            seed_tts_root=getattr(args, "seed_tts_root", None),
            system_prompt=getattr(args, "seed_tts_system_prompt", None),
            disable_shuffle=getattr(args, "disable_shuffle", False),
        )
        out_len = getattr(args, "output_len", None)
        if out_len is None:
            out_len = getattr(args, "hf_output_len", None)
        if out_len is None:
            out_len = SeedTTSDataset.DEFAULT_OUTPUT_LEN
        return dataset.sample(
            tokenizer=tokenizer,
            num_requests=args.num_prompts,
            output_len=out_len,
            request_id_prefix=args.request_id_prefix,
            no_oversample=args.no_oversample,
        )

    # Handle random-mm dataset (Omni's synthetic multimodal dataset)
    if args.dataset_name == "random-mm":
        dataset = OmniRandomMultiModalDataset(random_seed=args.seed, dataset_path=args.dataset_path)
        input_requests = dataset.sample(
            tokenizer=tokenizer,
            num_requests=args.num_prompts,
            prefix_len=args.random_prefix_len,
            range_ratio=args.random_range_ratio,
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            base_items_per_request=args.random_mm_base_items_per_request,
            limit_mm_per_prompt=args.random_mm_limit_mm_per_prompt,
            num_mm_items_range_ratio=args.random_mm_num_mm_items_range_ratio,
            bucket_config=args.random_mm_bucket_config,
            request_id_prefix=args.request_id_prefix,
            no_oversample=args.no_oversample,
        )
        return input_requests
    else:
        return get_samples_old(args, tokenizer)


datasets.get_samples = get_samples

_serve_mod = sys.modules.get("vllm.benchmarks.serve")
if _serve_mod is not None:
    _serve_mod.get_samples = get_samples


@dataclass
class MixRequestFuncOutput(RequestFuncOutput):
    audio_ttfp: float = 0.0
    audio_duration: float = 0.0
    audio_frames: int = 0
    audio_rtf: float = 0.0
    text_latency: float = 0.0
    #: Raw PCM s16le mono at 24 kHz for Seed-TTS WER: from ``/v1/audio/speech`` stream or
    #: resampled export after ``openai-chat-omni`` audio deltas.
    tts_output_pcm_bytes: bytes | None = None


async def async_request_openai_chat_omni_completions(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
    mm_position: Literal["first", "last"] = "last",
) -> MixRequestFuncOutput:
    api_url = request_func_input.api_url
    _validate_api_url(api_url, "OpenAI Chat Completions API", "chat/completions")

    omni_messages = getattr(request_func_input, "omni_chat_messages", None)
    if omni_messages is not None:
        messages_payload = omni_messages
    else:
        effective_mm_position = getattr(request_func_input, "mm_position", mm_position)
        content = _get_chat_content(request_func_input, mm_position=effective_mm_position)
        messages_payload = [{"role": "user", "content": content}]

    payload = {
        "model": request_func_input.model_name if request_func_input.model_name else request_func_input.model,
        "messages": messages_payload,
        "temperature": 0.0,
        "max_tokens": request_func_input.output_len,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
    }
    _update_payload_common(payload, request_func_input)
    # Seed-TTS via chat: voice-clone fields live on the body; ensure audio is streamed.
    if getattr(request_func_input, "seed_tts_row", False):
        if payload.get("modalities") is None:
            payload["modalities"] = ["text", "audio"]

    response_format = payload.get("response_format", "wav")
    if response_format == "pcm":
        raise ValueError(
            "pcm response format is not supported yet. \
        Please use other formats like wav, mp3, etc. instead."
        )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
    }
    _update_headers_common(headers, request_func_input)

    output = MixRequestFuncOutput()
    output.prompt_len = request_func_input.prompt_len
    max_retries = 3
    retry_delay = 0.1
    for attempt in range(max_retries + 1):
        # Reset per-attempt state so that retries do not mix partial
        # outputs or metrics from previous attempts.
        generated_text = ""
        generated_audio = None
        ttft = 0.0
        st = time.perf_counter()
        output.start_time = st
        most_recent_timestamp = st
        timestamp = st
        audio_generate_time = 0.0
        output.itl = []
        output.generated_text = ""
        output.ttft = 0.0
        output.audio_ttfp = 0.0
        output.audio_duration = 0.0
        output.audio_frames = 0
        output.audio_rtf = 0.0
        output.text_latency = 0.0
        output.output_tokens = 0
        output.error = ""
        output.success = False
        try:
            async with session.post(url=api_url, json=payload, headers=headers) as response:
                if response.status == 200:
                    handler = StreamedResponseHandler()
                    async for chunk_bytes in response.content.iter_any():
                        # NOTE: Do NOT strip() here; TCP may fragment the SSE messages,
                        # so stripping here can cause problems depending on how it is split.
                        #
                        # Simple example: [b'data: ',  b'{json}\n\n'] <- stripping the first
                        # chunk will break SSE parsing because the space after 'data:' is required.
                        if not chunk_bytes:
                            continue

                        messages = handler.add_chunk(chunk_bytes)
                        for message in messages:
                            if type(message) is bytes:
                                message = message.decode("utf-8")
                            # NOTE: SSE comments (often used as pings) start with
                            # a colon. These are not JSON data payload and should
                            # be skipped.
                            if message.startswith(":"):
                                continue

                            chunk = message.removeprefix("data: ")
                            if chunk != "[DONE]":
                                timestamp = time.perf_counter()
                                data = json.loads(chunk)
                                if choices := data.get("choices"):
                                    modality = data.get("modality")
                                    delta = choices[0].get("delta") or {}
                                    content = delta.get("content")
                                    if not content and isinstance(delta.get("audio"), dict):
                                        content = delta["audio"].get("data")
                                    if modality == "text":
                                        # First token
                                        if ttft == 0.0:
                                            ttft = timestamp - st
                                            output.ttft = ttft
                                        else:
                                            output.itl.append(timestamp - most_recent_timestamp)
                                        generated_text += content or ""
                                        most_recent_timestamp = timestamp
                                        output.text_latency = timestamp - st
                                    elif modality == "audio":
                                        if output.audio_ttfp == 0.0:
                                            output.audio_ttfp = timestamp - st
                                        audio_generate_time = timestamp - st
                                        if content:
                                            audio_bytes = base64.b64decode(content)
                                            seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
                                            if seg is not None:
                                                if generated_audio is None:
                                                    generated_audio = seg
                                                else:
                                                    generated_audio = generated_audio + seg

                                if metrics := data.get("metrics"):
                                    output.output_tokens = metrics.get("num_tokens_out", 0)

                                if usage := data.get("usage"):
                                    if (pt := usage.get("prompt_tokens")) is not None:
                                        output.prompt_len = pt

                    output.latency = timestamp - st
                    output.generated_text = generated_text
                    if generated_audio is not None:
                        output.audio_duration = len(generated_audio) / 1000.0
                        frame_width = generated_audio.frame_width
                        if frame_width > 0:
                            output.audio_frames = len(generated_audio.raw_data) // frame_width
                        else:
                            output.audio_frames = 0
                            logger.warning("Audio frame width is zero")
                        audio_duration = output.audio_duration
                        if audio_duration > 0:
                            output.audio_rtf = audio_generate_time / output.audio_duration
                        else:
                            output.audio_rtf = 0
                            logger.warning("Audio duration is zero")
                        if _seed_tts_capture_pcm_for_wer() and getattr(request_func_input, "seed_tts_row", False):
                            try:
                                seg = generated_audio.set_frame_rate(24000).set_channels(1).set_sample_width(2)
                                output.tts_output_pcm_bytes = bytes(seg.raw_data)
                            except Exception as ex:
                                logger.warning("seed_tts WER PCM export failed: %s", ex)
                    output.success = True
                else:
                    output.error = response.reason or ""
                    output.success = False
            break
        except aiohttp.ClientError as e:
            # transient transport error: may retry
            output.success = False
            output.error = traceback.format_exc()
            if attempt < max_retries:
                logger.warning(
                    "ClientError in omni benchmark request (will retry): attempt=%d/%d delay=%.2fs: %s",
                    attempt + 1,
                    max_retries + 1,
                    retry_delay,
                    str(e),
                )
                await asyncio.sleep(retry_delay)
                continue
            logger.error(
                "ClientError in omni benchmark request (giving up):\n%s",
                output.error,
            )
            break
        except Exception:
            output.success = False
            output.error = traceback.format_exc()
            logger.error(f"ERROR: send request failed, reason is: {output.error}")
            break

    if pbar:
        pbar.update(1)
    return output


async def async_request_openai_audio_speech(
    request_func_input: RequestFuncInput, session: aiohttp.ClientSession, pbar: tqdm | None = None
) -> MixRequestFuncOutput:
    """Streaming request to /v1/audio/speech endpoint.

    Sends ``stream=true`` with ``response_format=pcm`` so the server returns
    raw PCM chunks as they are decoded. This allows measuring TTFP (time to
    first audio packet) separately from E2EL.
    """
    api_url = request_func_input.api_url
    _validate_api_url(api_url, "OpenAI Audio Speech API", "audio/speech")

    payload = {
        "model": request_func_input.model_name if request_func_input.model_name else request_func_input.model,
        "input": request_func_input.prompt,
        "stream": True,
        "response_format": "pcm",
    }
    _update_payload_common(payload, request_func_input)
    # Seed-TTS + WER: ``--extra-body`` may set stream=false / other formats; speech must stream PCM.
    if getattr(request_func_input, "seed_tts_row", False) and _seed_tts_capture_pcm_for_wer():
        payload["stream"] = True
        payload["response_format"] = "pcm"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
    }
    _update_headers_common(headers, request_func_input)

    output = MixRequestFuncOutput()
    output.prompt_len = request_func_input.prompt_len

    # PCM format: 16-bit signed, 24 kHz, mono
    sample_rate = 24000
    sample_width = 2  # 16-bit = 2 bytes
    channels = 1

    st = time.perf_counter()
    output.start_time = st
    total_pcm_bytes = 0
    capture_wer_pcm = _seed_tts_capture_pcm_for_wer() and getattr(request_func_input, "seed_tts_row", False)
    pcm_capture = bytearray() if capture_wer_pcm else None
    try:
        async with session.post(url=api_url, json=payload, headers=headers) as response:
            if response.status == 200:
                async for chunk in response.content.iter_any():
                    if not chunk:
                        continue
                    timestamp = time.perf_counter()
                    if output.audio_ttfp == 0.0:
                        output.audio_ttfp = timestamp - st
                        output.ttft = output.audio_ttfp
                    total_pcm_bytes += len(chunk)
                    if pcm_capture is not None:
                        pcm_capture.extend(chunk)

                end_time = time.perf_counter()
                output.latency = end_time - st

                total_samples = total_pcm_bytes // (sample_width * channels)
                output.audio_duration = total_samples / sample_rate
                output.audio_frames = total_samples
                if output.audio_duration > 0:
                    output.audio_rtf = output.latency / output.audio_duration
                else:
                    output.audio_rtf = 0
                    logger.warning("Audio duration is zero")
                if pcm_capture is not None and pcm_capture:
                    output.tts_output_pcm_bytes = bytes(pcm_capture)
                elif capture_wer_pcm:
                    ct = response.headers.get("Content-Type", "")
                    logger.warning(
                        "Seed-TTS WER: HTTP 200 but no PCM bytes (Content-Type=%r, url=%s). "
                        "Check stream=true and response_format=pcm on the server.",
                        ct,
                        api_url,
                    )
                output.success = True
            else:
                output.error = response.reason or ""
                output.success = False
    except Exception:
        output.success = False
        output.error = traceback.format_exc()
        logger.error(f"ERROR: send request failed, reason is: {output.error}")

    if pbar:
        pbar.update(1)
    return output


ASYNC_REQUEST_FUNCS["openai-chat-omni"] = async_request_openai_chat_omni_completions
if "openai-chat-omni" not in OPENAI_COMPATIBLE_BACKENDS:
    OPENAI_COMPATIBLE_BACKENDS.append("openai-chat-omni")

ASYNC_REQUEST_FUNCS["openai-audio-speech"] = async_request_openai_audio_speech
if "openai-audio-speech" not in OPENAI_COMPATIBLE_BACKENDS:
    OPENAI_COMPATIBLE_BACKENDS.append("openai-audio-speech")

# Daily-Omni backend for audio-visual reasoning benchmark
# Reuses openai-chat-omni completions for video+text understanding
ASYNC_REQUEST_FUNCS["daily-omni"] = async_request_openai_chat_omni_completions
if "daily-omni" not in OPENAI_COMPATIBLE_BACKENDS:
    OPENAI_COMPATIBLE_BACKENDS.append("daily-omni")

# ruff: noqa: E402
# Prevent import order from causing patch failures
from vllm.benchmarks import serve
from vllm.benchmarks.lib.ready_checker import wait_for_endpoint
from vllm.benchmarks.serve import TaskType, calculate_metrics_for_embeddings, get_request

from vllm_omni.benchmarks.metrics.metrics import MultiModalsBenchmarkMetrics, calculate_metrics

# ruff: noqa: E402

benchmark_old = serve.benchmark


async def benchmark(
    task_type: TaskType,
    endpoint_type: str,
    api_url: str,
    base_url: str,
    model_id: str,
    model_name: str,
    tokenizer: TokenizerLike | None,
    input_requests: list[SampleRequest],
    logprobs: int | None,
    request_rate: float,
    burstiness: float,
    disable_tqdm: bool,
    num_warmups: int,
    profile: bool,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
    ignore_eos: bool,
    goodput_config_dict: dict[str, float],
    max_concurrency: int | None,
    lora_modules: Iterable[str] | None,
    extra_headers: dict | None,
    extra_body: dict | None,
    lora_assignment: Literal["random", "round-robin"] = "random",
    ramp_up_strategy: Literal["linear", "exponential"] | None = None,
    ramp_up_start_rps: int | None = None,
    ramp_up_end_rps: int | None = None,
    ready_check_timeout_sec: int = 600,
    ssl_context: ssl.SSLContext | bool | None = None,
):
    try:
        request_func = ASYNC_REQUEST_FUNCS[endpoint_type]
    except KeyError:
        raise ValueError(f"Unknown backend: {endpoint_type}") from None

    # Reuses connections across requests to reduce TLS handshake overhead.
    ssl_setting = ssl_context if ssl_context is not None else ("https://" in api_url)
    connector = aiohttp.TCPConnector(
        limit=max_concurrency or 0,
        limit_per_host=max_concurrency or 0,
        ttl_dns_cache=300,
        use_dns_cache=True,
        enable_cleanup_closed=True,
        force_close=True,
        ssl=ssl_setting,
    )

    session = aiohttp.ClientSession(
        connector=connector,
        trust_env=True,
        timeout=aiohttp.ClientTimeout(total=6 * 60 * 60),
    )

    print("Starting initial single prompt test run...")
    test_prompt, test_prompt_len, test_output_len, test_mm_content = (
        input_requests[0].prompt,
        input_requests[0].prompt_len,
        input_requests[0].expected_output_len,
        input_requests[0].multi_modal_data,
    )

    assert (
        test_mm_content is None
        or isinstance(test_mm_content, dict)
        or (isinstance(test_mm_content, list) and all(isinstance(item, dict) for item in test_mm_content))
    ), "multi_modal_data must be a dict or list[dict]"
    test_input = RequestFuncInput(
        model=model_id,
        model_name=model_name,
        prompt=test_prompt,
        api_url=api_url,
        prompt_len=test_prompt_len,
        output_len=test_output_len,
        logprobs=logprobs,
        multi_modal_content=test_mm_content,
        ignore_eos=ignore_eos,
        extra_headers=extra_headers,
        extra_body=extra_body,
    )
    _attach_daily_omni_to_request_func_input(input_requests[0], test_input)
    _attach_seed_tts_to_request_func_input(input_requests[0], test_input)

    if ready_check_timeout_sec > 0:
        test_output = await wait_for_endpoint(
            request_func,
            test_input,
            session,
            timeout_seconds=ready_check_timeout_sec,
        )
        if not test_output.success:
            raise ValueError(
                "Initial test run failed - Please make sure benchmark "
                "arguments are correctly specified. "
                f"Error: {test_output.error}"
            )
        else:
            print("Initial test run completed.")
    else:
        print("Skipping endpoint ready check.")

    if num_warmups > 0:
        print(f"Warming up with {num_warmups} requests...")
        warmup_pbar = None if disable_tqdm else tqdm(total=num_warmups)
        warmup_semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else contextlib.nullcontext()
        warmup_tasks = []

        async def warmup_limited_request_func():
            async with warmup_semaphore:
                return await request_func(request_func_input=test_input, session=session, pbar=warmup_pbar)

        for _ in range(num_warmups):
            request_task = asyncio.create_task(warmup_limited_request_func())
            warmup_tasks.append(request_task)
        _ = await asyncio.gather(*warmup_tasks)

        if warmup_pbar is not None:
            warmup_pbar.close()
        print("Warmup run completed.")

    print("Starting main benchmark run...")

    if lora_modules:
        lora_modules_list = list(lora_modules)
        if lora_assignment == "round-robin":
            lora_modules = iter([lora_modules_list[i % len(lora_modules_list)] for i in range(len(input_requests))])
        else:
            lora_modules = iter([random.choice(lora_modules_list) for _ in range(len(input_requests))])

    if profile:
        print("Starting profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            model_name=model_name,
            prompt=test_prompt,
            api_url=base_url + "/start_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=logprobs,
            multi_modal_content=test_mm_content,
            ignore_eos=ignore_eos,
            extra_headers=extra_headers,
            extra_body=extra_body,
        )
        _attach_daily_omni_to_request_func_input(input_requests[0], profile_input)
        _attach_seed_tts_to_request_func_input(input_requests[0], profile_input)
        profile_output = await request_func(request_func_input=profile_input, session=session)
        if profile_output.success:
            print("Profiler started")

    distribution = "Poisson process" if burstiness == 1.0 else "Gamma distribution"

    if ramp_up_strategy is not None:
        print(f"Traffic ramp-up strategy: {ramp_up_strategy}.")
        print(
            f"Will increase RPS from {ramp_up_start_rps} to {ramp_up_end_rps} RPS over the duration of the benchmark."
        )
    else:
        print(f"Traffic request rate: {request_rate}")

    print(f"Burstiness factor: {burstiness} ({distribution})")
    print(f"Maximum request concurrency: {max_concurrency}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else contextlib.nullcontext()

    async def limited_request_func(request_func_input, session, pbar):
        async with semaphore:
            return await request_func(request_func_input=request_func_input, session=session, pbar=pbar)

    benchmark_start_time = time.perf_counter()
    tasks: list[asyncio.Task] = []

    rps_change_events = []
    last_int_rps = -1
    if ramp_up_strategy is not None and ramp_up_start_rps is not None:
        last_int_rps = ramp_up_start_rps
        rps_change_events.append(
            {
                "rps": last_int_rps,
                "timestamp": datetime.now().isoformat(),
            }
        )

    async for request, current_request_rate in get_request(
        input_requests,
        request_rate,
        burstiness,
        ramp_up_strategy,
        ramp_up_start_rps,
        ramp_up_end_rps,
    ):
        if ramp_up_strategy is not None:
            current_int_rps = int(current_request_rate)
            if current_int_rps > last_int_rps:
                timestamp = datetime.now().isoformat()
                for rps_val in range(last_int_rps + 1, current_int_rps + 1):
                    rps_change_events.append({"rps": rps_val, "timestamp": timestamp})
                last_int_rps = current_int_rps
        prompt, prompt_len, output_len, mm_content, request_id = (
            request.prompt,
            request.prompt_len,
            request.expected_output_len,
            request.multi_modal_data,
            request.request_id,
        )
        req_model_id, req_model_name = model_id, model_name
        if lora_modules:
            req_lora_module = next(lora_modules)
            req_model_id, req_model_name = req_lora_module, req_lora_module

        request_func_input = RequestFuncInput(
            model=req_model_id,
            model_name=req_model_name,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            logprobs=logprobs,
            multi_modal_content=mm_content,
            ignore_eos=ignore_eos,
            extra_headers=extra_headers,
            extra_body=extra_body,
            request_id=request_id,
        )
        _attach_daily_omni_to_request_func_input(request, request_func_input)
        _attach_seed_tts_to_request_func_input(request, request_func_input)
        tasks.append(
            asyncio.create_task(limited_request_func(request_func_input=request_func_input, session=session, pbar=pbar))
        )
    outputs: list[MixRequestFuncOutput] = await asyncio.gather(*tasks)

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    if task_type == TaskType.GENERATION:
        metrics, actual_output_lens = calculate_metrics(
            input_requests=input_requests,
            outputs=outputs,
            dur_s=benchmark_duration,
            tokenizer=tokenizer,
            selected_percentiles=selected_percentiles,
            goodput_config_dict=goodput_config_dict,
            task_type=task_type,
            selected_percentile_metrics=selected_percentile_metrics,
            max_concurrency=max_concurrency,
            request_rate=request_rate,
            benchmark_duration=benchmark_duration,
        )
    else:
        metrics = calculate_metrics_for_embeddings(
            outputs=outputs,
            dur_s=benchmark_duration,
            selected_percentiles=selected_percentiles,
        )
        actual_output_lens = 0

    if isinstance(metrics, MultiModalsBenchmarkMetrics):
        result = {
            "duration": benchmark_duration,
            "completed": metrics.completed,
            "failed": metrics.failed,
            "total_input_tokens": metrics.total_input,
            "total_output_tokens": metrics.total_output,
            "request_throughput": metrics.request_throughput,
            "request_goodput": metrics.request_goodput if goodput_config_dict else None,
            "output_throughput": metrics.output_throughput,
            "total_token_throughput": metrics.total_token_throughput,
            "total_audio_duration_s": metrics.total_audio_duration_s,
            "total_audio_frames": metrics.total_audio_frames,
            "audio_throughput": metrics.audio_throughput,
            "input_lens": [output.prompt_len for output in outputs],
            "output_lens": actual_output_lens,
            "ttfts": [output.ttft for output in outputs],
            "itls": [output.itl for output in outputs],
            "generated_texts": [output.generated_text for output in outputs],
            "errors": [output.error for output in outputs],
            "max_output_tokens_per_s": metrics.max_output_tokens_per_s,
            "max_concurrent_requests": metrics.max_concurrent_requests,
            "rtfx": metrics.rtfx,
        }
    else:
        result = {
            "duration": benchmark_duration,
            "completed": metrics.completed,
            "total_input_tokens": metrics.total_input,
            "request_throughput": metrics.request_throughput,
            "total_token_throughput": metrics.total_token_throughput,
            "input_lens": [output.prompt_len for output in outputs],
            "errors": [output.error for output in outputs],
        }

    from vllm_omni.benchmarks.data_modules.daily_omni_eval import (
        compute_daily_omni_accuracy_metrics,
        print_daily_omni_accuracy_summary,
    )

    _save_items = os.environ.get("DAILY_OMNI_SAVE_EVAL_ITEMS", "").lower() in (
        "1",
        "true",
        "yes",
    )
    _daily_acc = compute_daily_omni_accuracy_metrics(input_requests, outputs, include_per_item=_save_items)
    if _daily_acc is not None:
        result.update(_daily_acc)
        print_daily_omni_accuracy_summary(_daily_acc)

    if _seed_tts_capture_pcm_for_wer():
        from vllm_omni.benchmarks.data_modules.seed_tts_eval import (
            compute_seed_tts_wer_metrics,
            print_seed_tts_wer_summary,
        )

        _save_wer = os.environ.get("SEED_TTS_WER_SAVE_ITEMS", "").lower() in (
            "1",
            "true",
            "yes",
        )
        _wer_m = compute_seed_tts_wer_metrics(input_requests, outputs, include_per_item=_save_wer)
        if _wer_m is not None:
            result.update(_wer_m)
            print_seed_tts_wer_summary(_wer_m)

    if rps_change_events:
        result["rps_change_events"] = rps_change_events

    def process_one_metric(
        # E.g., "ttft"
        metric_attribute_name: str,
    ):
        # This function prints and adds statistics of the specified
        # metric.
        if metric_attribute_name not in selected_percentile_metrics:
            return
        is_audio_rtf = metric_attribute_name == "audio_rtf"
        is_audio_duration = metric_attribute_name == "audio_duration"

        suffix = "_ms"
        if is_audio_duration:
            suffix = "_s"
        elif is_audio_rtf:
            suffix = ""
        mean_attr_name = f"mean_{metric_attribute_name}{suffix}"
        mean_value = getattr(metrics, mean_attr_name, 0.0)
        result[mean_attr_name] = mean_value

        median_attr_name = f"median_{metric_attribute_name}{suffix}"
        median_value = getattr(metrics, median_attr_name, 0.0)
        result[median_attr_name] = median_value
        for p, value in getattr(metrics, f"percentiles_{metric_attribute_name}{suffix}"):
            p_word = str(int(p)) if int(p) == p else str(p)
            result[f"p{p_word}_{metric_attribute_name}{suffix}"] = value

    if task_type == TaskType.GENERATION:
        for metric in selected_percentile_metrics:
            process_one_metric(metric)
    else:
        process_one_metric("e2el")

    if profile:
        print("Stopping profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            prompt=test_prompt,
            api_url=base_url + "/stop_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=logprobs,
        )
        profile_output = await request_func(request_func_input=profile_input, session=session)
        if profile_output.success:
            print("Profiler stopped")

    await session.close()
    return result


serve.benchmark = benchmark
