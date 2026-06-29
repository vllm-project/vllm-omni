import asyncio
import contextlib
import io
import json
import mimetypes
import os
import random
import ssl
import sys
import time
import traceback
import wave
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import aiohttp
import numpy as np
import pybase64 as base64
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

from vllm_omni.benchmarks.audio_continuity import compute_continuity_stats
from vllm_omni.benchmarks.data_modules.daily_omni_dataset import DailyOmniDataset, DailyOmniSampleRequest
from vllm_omni.benchmarks.data_modules.random_multi_modal_dataset import OmniRandomMultiModalDataset
from vllm_omni.benchmarks.data_modules.seed_tts_dataset import (
    SEED_TTS_DEFAULT_OMNI_SYSTEM_PROMPT,
    SeedTTSDataset,
    SeedTTSDesignDataset,
    SeedTTSSampleRequest,
    SeedTTSTextDataset,
)
from vllm_omni.benchmarks.data_modules.sound_effect_dataset import SoundEffectDataset
from vllm_omni.benchmarks.data_modules.ttsd_dataset import TTSDDataset
from vllm_omni.metrics import definitions as defs

_AUDIO_CONTINUITY_THRESHOLD_ENV = "VLLM_OMNI_BENCH_AUDIO_CONTINUITY_THRESHOLD_S"
_AUDIO_SAMPLE_RATE_ENV = "VLLM_OMNI_BENCH_AUDIO_SAMPLE_RATE"
_AUDIO_CHANNELS_ENV = "VLLM_OMNI_BENCH_AUDIO_CHANNELS"
RETURN_STAGE_METRICS_FIELD = "return_stage_metrics"
_IMAGE_STAGE_METRICS_BACKENDS = frozenset({"openai-image-edits-omni"})
_PRINT_STAGE = False


def maybe_enable_stage_metrics(extra_body: dict[str, Any] | None, *, enabled: bool) -> dict[str, Any] | None:
    """Return extra_body with stage-metric opt-in when benchmark metrics need it."""
    if not enabled:
        return extra_body
    body = dict(extra_body or {})
    body.setdefault(RETURN_STAGE_METRICS_FIELD, True)
    return body


def should_request_stage_metrics(args: Any) -> bool:
    """Whether this benchmark run needs server-side stage metrics in responses."""
    if getattr(args, "print_stage", False):
        return True

    backend = getattr(args, "backend", None)
    if backend in _IMAGE_STAGE_METRICS_BACKENDS:
        return True

    extra_body = getattr(args, "extra_body", None) or {}
    modalities = extra_body.get("modalities") if isinstance(extra_body, dict) else None
    return backend == "openai-chat-omni" and "image" in (modalities or [])


def set_print_stage(enabled: bool) -> None:
    """Set whether this benchmark run prints the stage benchmark section."""
    global _PRINT_STAGE
    _PRINT_STAGE = bool(enabled)


def _audio_continuity_threshold_s() -> float:
    """Return the per-request underrun budget (s).

    Read from ``VLLM_OMNI_BENCH_AUDIO_CONTINUITY_THRESHOLD_S`` so users can
    re-aim the SLO without rebuilding. Defaults to 100 ms - the standard
    "audible gap" budget for streaming TTS.
    """
    raw = os.environ.get(_AUDIO_CONTINUITY_THRESHOLD_ENV)
    if not raw:
        return defs.AUDIO_CONTINUITY_DEFAULT_THRESHOLD_S
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; using default %.3fs",
            _AUDIO_CONTINUITY_THRESHOLD_ENV,
            raw,
            defs.AUDIO_CONTINUITY_DEFAULT_THRESHOLD_S,
        )
        return defs.AUDIO_CONTINUITY_DEFAULT_THRESHOLD_S
    return max(value, 0.0)


def _audio_pcm_format() -> tuple[int, int]:
    """Return ``(sample_rate, channels)`` of the streamed PCM response.

    Defaults to 24 kHz mono (Qwen3-TTS / VoxCPM2 / most MOSS-TTS variants).
    Override for models with a different codec output format (e.g.
    MOSS-TTS-Local-Transformer-v1.5, which streams 48 kHz stereo PCM) via
    ``VLLM_OMNI_BENCH_AUDIO_SAMPLE_RATE`` / ``VLLM_OMNI_BENCH_AUDIO_CHANNELS`` -
    getting this wrong silently skews audio_duration/audio_rtf/underrun stats.
    """
    sample_rate = 24000
    channels = 1
    raw_sr = os.environ.get(_AUDIO_SAMPLE_RATE_ENV)
    if raw_sr:
        try:
            sample_rate = int(raw_sr)
        except ValueError:
            logger.warning("Invalid %s=%r; using default %d", _AUDIO_SAMPLE_RATE_ENV, raw_sr, sample_rate)
    raw_ch = os.environ.get(_AUDIO_CHANNELS_ENV)
    if raw_ch:
        try:
            channels = int(raw_ch)
        except ValueError:
            logger.warning("Invalid %s=%r; using default %d", _AUDIO_CHANNELS_ENV, raw_ch, channels)
    return sample_rate, channels


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
    is_seed_tts = args.dataset_name in (
        "seed-tts",
        "seed-tts-text",
        "seed-tts-design",
        "ttsd",
        "sound-effect",
    )

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
            "ttsd": TTSDDataset,
            "sound-effect": SoundEffectDataset,
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
    image_count: int = 0
    image_generation_time_ms: float = 0.0
    image_pixels: int = 0
    denoise_step_latency_ms: float = 0.0
    text_latency: float = 0.0
    #: Worst-case streaming-audio underrun (wall-clock seconds the player
    #: would have been starved). Populated by the audio-speech backend; ``0.0``
    #: for backends that do not run continuity analysis.
    audio_underrun_s: float = 0.0
    #: Whether the request stayed under the continuity threshold (default
    #: 100 ms). Mirrors ``audio_underrun_s <= threshold``.
    audio_continuity_ok: bool = True
    #: Number of inter-chunk intervals during which the player buffer went
    #: negative.
    audio_underrun_event_count: int = 0
    #: Raw PCM s16le mono at 24 kHz for Seed-TTS WER: from ``/v1/audio/speech`` stream or
    #: resampled export after ``openai-chat-omni`` audio deltas.
    tts_output_pcm_bytes: bytes | None = None
    #: Per-stage snapshot from orchestrator ``metrics["stage_metrics"]`` (merged across SSE chunks).
    stage_metrics: dict[str, dict] | None = None
    stage_id: int | None = None
    final_output_type: str | None = None


_IMAGE_EDITS_EXTRA_BODY_FORM_FIELDS = (
    "negative_prompt",
    "num_inference_steps",
    "guidance_scale",
    "strength",
    "true_cfg_scale",
    "seed",
    "generator_device",
    "lora",
    "layers",
    "resolution",
    "bot_task",
    "sys_type",
    "system_prompt",
    RETURN_STAGE_METRICS_FIELD,
)


def _guess_mime_type(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"


def _iter_image_edit_inputs(value: Any) -> Iterable[Any]:
    """Yield image references from benchmark multimodal content."""
    if value is None:
        return
    if isinstance(value, list):
        for item in value:
            yield from _iter_image_edit_inputs(item)
        return
    if not isinstance(value, dict):
        yield value
        return

    content_type = value.get("type")
    if content_type == "image_url":
        image_url = value.get("image_url")
        if isinstance(image_url, dict):
            url = image_url.get("url")
            if url:
                yield url
        elif image_url:
            yield image_url
        return

    for key in ("image", "images"):
        if key in value:
            yield from _iter_image_edit_inputs(value[key])


def _add_image_edit_input_to_form(form: aiohttp.FormData, image_input: Any) -> None:
    if isinstance(image_input, dict) and "bytes" in image_input:
        form.add_field(
            "image",
            image_input["bytes"],
            filename="benchmark.png",
            content_type="image/png",
        )
        return

    if isinstance(image_input, str):
        if image_input.startswith(("data:image", "http://", "https://")):
            form.add_field("url", image_input)
            return
        local_path = image_input.removeprefix("file://")
        if os.path.exists(local_path):
            with open(local_path, "rb") as f:
                image_bytes = f.read()
            form.add_field(
                "image",
                image_bytes,
                filename=os.path.basename(local_path),
                content_type=_guess_mime_type(local_path),
            )
            return

    raise ValueError(f"Unsupported image edit input: {type(image_input).__name__}")


def _add_image_edit_extra_body_to_form(form: aiohttp.FormData, extra_body: dict[str, Any]) -> None:
    for key in _IMAGE_EDITS_EXTRA_BODY_FORM_FIELDS:
        value = extra_body.get(key)
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            form.add_field(key, json.dumps(value))
        else:
            form.add_field(key, str(value))


def _coerce_positive_int(value: Any) -> int | None:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return None
    return coerced if coerced > 0 else None


def _extract_output_tokens_from_metrics(metrics: dict[str, Any]) -> int | None:
    top_level_tokens = _coerce_positive_int(metrics.get(defs.NUM_TOKENS_OUT))
    if top_level_tokens is not None:
        return top_level_tokens

    stage_snapshot = metrics.get("stage_metrics")
    if not isinstance(stage_snapshot, dict):
        return None

    fallback_tokens: list[int] = []
    for info in stage_snapshot.values():
        if not isinstance(info, dict):
            continue
        num_tokens_out = _coerce_positive_int(info.get(defs.NUM_TOKENS_OUT))
        if num_tokens_out is None:
            continue
        if info.get("final_output_type") == "text" or info.get("output_unit_type") == "token":
            return num_tokens_out
        fallback_tokens.append(num_tokens_out)
    return max(fallback_tokens, default=None)


def _apply_usage_to_output(output: MixRequestFuncOutput, usage: dict[str, Any]) -> int | None:
    """Apply OpenAI ``usage`` fields to the benchmark output."""
    if (pt := _coerce_positive_int(usage.get("prompt_tokens"))) is not None:
        output.prompt_len = pt
    completion_tokens = _coerce_positive_int(usage.get("completion_tokens"))
    if completion_tokens is not None:
        output.output_tokens = max(int(output.output_tokens or 0), completion_tokens)
    return completion_tokens


def _resolve_token_delta_from_usage(
    completion_tokens: int | None,
    completion_tokens_seen: int,
) -> tuple[int, int]:
    if completion_tokens is None or completion_tokens <= completion_tokens_seen:
        return 0, completion_tokens_seen
    delta = completion_tokens - completion_tokens_seen
    return delta, completion_tokens


def _record_text_token_stream_intervals(
    output: MixRequestFuncOutput,
    *,
    timestamp: float,
    start_time: float,
    token_delta: int,
    most_recent_timestamp: float,
) -> float:
    """Record TTFT/ITL for ``token_delta`` newly generated text tokens."""
    if token_delta <= 0:
        return most_recent_timestamp

    if output.ttft == 0.0:
        output.ttft = timestamp - start_time
        output.text_latency = timestamp - start_time
        most_recent_timestamp = timestamp
        if token_delta > 1:
            output.itl.extend([0.0] * (token_delta - 1))
        return most_recent_timestamp

    interval = max(timestamp - most_recent_timestamp, 0.0)
    per_token = interval / token_delta
    output.itl.extend([per_token] * token_delta)
    output.text_latency = timestamp - start_time
    return timestamp


def _update_output_stage_metrics_from_payload(
    output: MixRequestFuncOutput,
    data: dict[str, Any],
    *,
    update_output_tokens: bool = True,
) -> None:
    metrics = data.get("metrics")
    if not isinstance(metrics, dict):
        return
    if update_output_tokens:
        if (num_tokens_out := _extract_output_tokens_from_metrics(metrics)) is not None:
            output.output_tokens = max(int(output.output_tokens or 0), num_tokens_out)
    if isinstance(sid := metrics.get("stage_id"), int):
        output.stage_id = sid
    if isinstance(final_output_type := metrics.get("final_output_type"), str):
        output.final_output_type = final_output_type
    stage_snapshot = metrics.get("stage_metrics")
    if isinstance(stage_snapshot, dict):
        if output.stage_metrics is None:
            output.stage_metrics = {}
        output.stage_metrics.update(stage_snapshot)


def _image_metrics_from_stage_metrics(metrics: dict[str, Any] | None) -> tuple[int, float, int, float]:
    if not isinstance(metrics, dict):
        return 0, 0.0, 0, 0.0
    stage_snapshot = metrics.get("stage_metrics")
    if not isinstance(stage_snapshot, dict):
        return 0, 0.0, 0, 0.0
    image_count = 0
    image_generation_ms = 0.0
    image_pixels = 0
    denoise_step_latency_ms = 0.0
    for info in stage_snapshot.values():
        if not isinstance(info, dict):
            continue
        final_output_type = info.get("final_output_type")
        output_unit_type = info.get("output_unit_type")
        if final_output_type not in {"image", "images"} and output_unit_type != "image":
            continue
        image_count += int(info.get(defs.OUTPUT_UNIT_COUNT) or 0)
        image_generation_ms += float(info.get(defs.STAGE_GEN_TIME_MS) or 0.0)
        image_pixels += int(info.get(defs.IMAGE_PIXELS) or 0)
        denoise_step_latency_ms = max(
            denoise_step_latency_ms,
            float(info.get(defs.DENOISE_STEP_LATENCY_MS) or 0.0),
        )
    return image_count, image_generation_ms, image_pixels, denoise_step_latency_ms


def _image_generation_ms_from_content(content: Any) -> float:
    if not isinstance(content, list):
        return 0.0
    for item in content:
        if not isinstance(item, dict):
            continue
        stage_durations = item.get("stage_durations")
        if not isinstance(stage_durations, dict):
            continue
        gen_values = [
            float(value)
            for key, value in stage_durations.items()
            if str(key).endswith("_gen_ms") and isinstance(value, (int, float))
        ]
        if gen_values:
            return max(gen_values)
    return 0.0


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
            # Per-chunk completion_tokens lets _resolve_token_delta_from_usage
            # compute the exact token count for each SSE flush.  Without this,
            # one SSE chunk can carry multiple tokens (asyncio coalescing), so
            # len(itl)+1 < actual_tokens and ITL measures per-chunk latency
            # (~1.74× tokens_per_chunk) rather than true per-token latency.
            # NOTE: the vLLM StreamOptions field is "continuous_usage_stats",
            # NOT "include_continuous_usage".
            "continuous_usage_stats": True,
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
        # For wav responses, accumulate decoded PCM bytes per chunk
        # to avoid repeated decode/concat.
        wav_pcm_buffer = bytearray()
        wav_audio_params: tuple[int, int, int] | None = None
        wav_inconsistent_chunk_count = 0
        first_inconsistent_wav_params: tuple[int, int, int] | None = None
        # For non-wav responses, accumulate encoded bytes then decode once.
        audio_bytes_buffer = bytearray()
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
        output.stage_metrics = {}
        output.stage_id = None
        output.final_output_type = None
        output.image_count = 0
        output.image_generation_time_ms = 0.0
        output.image_pixels = 0
        output.denoise_step_latency_ms = 0.0
        completion_tokens_seen = 0
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
                                _update_output_stage_metrics_from_payload(output, data)
                                usage = data.get("usage")
                                completion_tokens = None
                                if isinstance(usage, dict):
                                    completion_tokens = _apply_usage_to_output(output, usage)

                                if choices := data.get("choices"):
                                    modality = data.get("modality")
                                    choice = choices[0]
                                    delta = choice.get("delta") or {}
                                    content = delta.get("content")
                                    if not content and isinstance(delta.get("audio"), dict):
                                        content = delta["audio"].get("data")
                                    if modality == "text":
                                        token_delta, completion_tokens_seen = _resolve_token_delta_from_usage(
                                            completion_tokens,
                                            completion_tokens_seen,
                                        )
                                        token_ids = choice.get("token_ids")
                                        if token_delta == 0 and token_ids:
                                            token_delta = len(token_ids)
                                            if completion_tokens is not None:
                                                completion_tokens_seen = max(
                                                    completion_tokens_seen,
                                                    completion_tokens,
                                                )
                                        has_text_content = bool(content)
                                        if token_delta == 0 and has_text_content and completion_tokens is None:
                                            token_delta = 1
                                        if token_delta > 0:
                                            most_recent_timestamp = _record_text_token_stream_intervals(
                                                output,
                                                timestamp=timestamp,
                                                start_time=st,
                                                token_delta=token_delta,
                                                most_recent_timestamp=most_recent_timestamp,
                                            )
                                        if has_text_content:
                                            generated_text += content
                                    elif modality == "audio":
                                        if output.audio_ttfp == 0.0:
                                            output.audio_ttfp = timestamp - st
                                        audio_generate_time = timestamp - st
                                        if content:
                                            audio_bytes = base64.b64decode(content)
                                            if response_format == "wav":
                                                try:
                                                    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_reader:
                                                        params = (
                                                            wav_reader.getnchannels(),
                                                            wav_reader.getsampwidth(),
                                                            wav_reader.getframerate(),
                                                        )
                                                        if wav_audio_params is None:
                                                            wav_audio_params = params
                                                        elif wav_audio_params != params:
                                                            wav_inconsistent_chunk_count += 1
                                                            if first_inconsistent_wav_params is None:
                                                                first_inconsistent_wav_params = params
                                                            continue
                                                        wav_pcm_buffer.extend(
                                                            wav_reader.readframes(wav_reader.getnframes())
                                                        )
                                                except Exception as ex:
                                                    logger.warning("Failed to parse wav audio chunk: %s", ex)
                                            else:
                                                audio_bytes_buffer.extend(audio_bytes)
                                    elif modality == "image":
                                        output.image_count += 1
                                        content_image_ms = _image_generation_ms_from_content(content)
                                        if content_image_ms > 0:
                                            output.image_generation_time_ms += content_image_ms

                                (
                                    metrics_image_count,
                                    metrics_image_ms,
                                    metrics_image_pixels,
                                    metrics_denoise_step_ms,
                                ) = _image_metrics_from_stage_metrics(data.get("metrics"))
                                if metrics_image_count > output.image_count:
                                    output.image_count = metrics_image_count
                                if metrics_image_ms > output.image_generation_time_ms:
                                    output.image_generation_time_ms = metrics_image_ms
                                if metrics_image_pixels > output.image_pixels:
                                    output.image_pixels = metrics_image_pixels
                                if metrics_denoise_step_ms > output.denoise_step_latency_ms:
                                    output.denoise_step_latency_ms = metrics_denoise_step_ms

                    if wav_inconsistent_chunk_count > 0:
                        logger.warning(
                            "Dropped %d wav chunks with inconsistent params during benchmark "
                            "(expected=%s, first_inconsistent=%s). "
                            "Audio frames/duration may be undercounted.",
                            wav_inconsistent_chunk_count,
                            wav_audio_params,
                            first_inconsistent_wav_params,
                        )

                    output.latency = timestamp - st
                    output.generated_text = generated_text
                    if output.itl:
                        # Align text_latency with ITL so TPOT formula and
                        # mean(ITL) are consistent.  Do NOT infer output_tokens
                        # from len(itl)+1: one SSE chunk may carry multiple
                        # tokens, so the ITL count understates the real count.
                        output.text_latency = output.ttft + sum(output.itl)
                    audio_duration_sec = 0.0
                    audio_frames = 0
                    if response_format == "wav" and wav_pcm_buffer and wav_audio_params is not None:
                        channels, sample_width, frame_rate = wav_audio_params
                        frame_width = sample_width * channels
                        if frame_width > 0:
                            audio_frames = len(wav_pcm_buffer) // frame_width
                            audio_duration_sec = audio_frames / frame_rate
                        else:
                            logger.warning("Audio frame width is zero")
                    elif audio_bytes_buffer:
                        try:
                            from vllm.multimodal.audio import get_audio_duration
                            from vllm.multimodal.media.audio import load_audio

                            waveform, sr = load_audio(
                                io.BytesIO(bytes(audio_bytes_buffer)),
                                sr=None,
                                mono=False,
                            )
                            audio_duration_sec = get_audio_duration(y=waveform, sr=sr)
                            audio_frames = int(audio_duration_sec * sr)
                        except Exception as ex:
                            logger.warning("Failed to decode accumulated audio bytes: %s", ex)
                    if audio_duration_sec > 0 or audio_frames > 0:
                        output.audio_duration = audio_duration_sec
                        output.audio_frames = audio_frames
                        audio_duration = output.audio_duration
                        if audio_duration > 0:
                            output.audio_rtf = audio_generate_time / output.audio_duration
                        else:
                            output.audio_rtf = 0
                            logger.warning("Audio duration is zero")
                        if _seed_tts_capture_pcm_for_wer() and getattr(request_func_input, "seed_tts_row", False):
                            try:
                                if response_format == "wav" and wav_pcm_buffer and wav_audio_params is not None:
                                    from vllm.multimodal.audio import AudioResampler

                                    pcm_channels, pcm_sw, pcm_rate = wav_audio_params
                                    pcm = np.frombuffer(
                                        bytes(wav_pcm_buffer), dtype=np.int16 if pcm_sw == 2 else np.float32
                                    )
                                    if pcm_channels > 1:
                                        pcm = pcm.reshape(-1, pcm_channels).mean(axis=1).astype(pcm.dtype)
                                    pcm_f32 = pcm.astype(np.float32) / 32767.0 if pcm.dtype == np.int16 else pcm
                                    if pcm_rate != 24000:
                                        resampler = AudioResampler(target_sr=24000)
                                        pcm_f32 = resampler.resample(pcm_f32, orig_sr=pcm_rate)
                                    output.tts_output_pcm_bytes = (pcm_f32 * 32767).astype(np.int16).tobytes()
                                elif audio_bytes_buffer:
                                    from vllm.multimodal.media.audio import load_audio

                                    waveform, _ = load_audio(
                                        io.BytesIO(bytes(audio_bytes_buffer)),
                                        sr=24000,
                                        mono=True,
                                    )
                                    output.tts_output_pcm_bytes = (waveform * 32767).astype(np.int16).tobytes()
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


async def async_request_openai_image_edits_omni(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
) -> MixRequestFuncOutput:
    """Streaming request to /v1/images/edits for multi-stage image-edit benchmarks."""
    api_url = request_func_input.api_url
    _validate_api_url(api_url, "OpenAI Image Edits API", "images/edits")

    extra_body = dict(request_func_input.extra_body or {})
    model = request_func_input.model_name if request_func_input.model_name else request_func_input.model
    output = MixRequestFuncOutput()
    output.prompt_len = request_func_input.prompt_len
    output.itl = []
    output.stage_metrics = {}
    output.output_tokens = 0
    output.image_count = 0
    output.image_generation_time_ms = 0.0
    output.image_pixels = 0
    output.denoise_step_latency_ms = 0.0

    form = aiohttp.FormData()
    form.add_field("model", model)
    form.add_field("prompt", request_func_input.prompt)
    form.add_field("response_format", "b64_json")
    form.add_field("output_format", str(extra_body.get("output_format", "png")))
    form.add_field("stream", "true")

    size = extra_body.get("size")
    if size is None:
        width, height = extra_body.get("width"), extra_body.get("height")
        size = f"{width}x{height}" if width is not None and height is not None else "auto"
    form.add_field("size", str(size))

    _add_image_edit_extra_body_to_form(form, extra_body)

    try:
        image_inputs = list(_iter_image_edit_inputs(request_func_input.multi_modal_content))
        if not image_inputs:
            raise ValueError(
                "openai-image-edits-omni requires image multimodal content. "
                "For synthetic inputs, use --dataset-name random-mm with an image bucket."
            )
        for image_input in image_inputs:
            _add_image_edit_input_to_form(form, image_input)
    except Exception:
        output.success = False
        output.error = traceback.format_exc()
        if pbar:
            pbar.update(1)
        return output

    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
    }
    _update_headers_common(headers, request_func_input)

    st = time.perf_counter()
    output.start_time = st
    timestamp = st
    most_recent_text_timestamp = st
    generated_text = ""
    try:
        async with session.post(url=api_url, data=form, headers=headers) as response:
            if response.status == 200:
                handler = StreamedResponseHandler()
                async for chunk_bytes in response.content.iter_any():
                    if not chunk_bytes:
                        continue
                    for message in handler.add_chunk(chunk_bytes):
                        if type(message) is bytes:
                            message = message.decode("utf-8")
                        if message.startswith(":"):
                            continue
                        chunk = message.removeprefix("data: ")
                        if chunk == "[DONE]":
                            continue

                        timestamp = time.perf_counter()
                        data = json.loads(chunk)
                        _update_output_stage_metrics_from_payload(
                            output,
                            data,
                            update_output_tokens=(data.get("type") == "ar_delta"),
                        )

                        chunk_type = data.get("type")
                        if chunk_type == "ar_delta":
                            if output.ttft == 0.0:
                                output.ttft = timestamp - st
                            else:
                                output.itl.append(timestamp - most_recent_text_timestamp)
                            delta = data.get("delta") or ""
                            generated_text += delta
                            most_recent_text_timestamp = timestamp
                            output.text_latency = timestamp - st
                        elif chunk_type == "image":
                            image_data = data.get("data")
                            output.image_count += len(image_data) if isinstance(image_data, list) else 1
                            content_image_ms = _image_generation_ms_from_content(data.get("data"))
                            if content_image_ms > 0:
                                output.image_generation_time_ms += content_image_ms
                        (
                            metrics_image_count,
                            metrics_image_ms,
                            metrics_image_pixels,
                            metrics_denoise_step_ms,
                        ) = _image_metrics_from_stage_metrics(data.get("metrics"))
                        if metrics_image_count > output.image_count:
                            output.image_count = metrics_image_count
                        if metrics_image_ms > output.image_generation_time_ms:
                            output.image_generation_time_ms = metrics_image_ms
                        if metrics_image_pixels > output.image_pixels:
                            output.image_pixels = metrics_image_pixels
                        if metrics_denoise_step_ms > output.denoise_step_latency_ms:
                            output.denoise_step_latency_ms = metrics_denoise_step_ms
                output.latency = timestamp - st
                output.generated_text = generated_text
                output.success = True
            else:
                output.error = f"HTTP {response.status}: {await response.text()}"
                output.success = False
    except Exception:
        output.success = False
        output.error = traceback.format_exc()
        logger.error(f"ERROR: send image edit request failed, reason is: {output.error}")

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

    # PCM format: 16-bit signed; sample_rate/channels are model-dependent
    # (see _audio_pcm_format docstring).
    sample_rate, channels = _audio_pcm_format()
    sample_width = 2  # 16-bit = 2 bytes

    st = time.perf_counter()
    output.start_time = st
    total_pcm_bytes = 0
    capture_wer_pcm = _seed_tts_capture_pcm_for_wer() and getattr(request_func_input, "seed_tts_row", False)
    pcm_capture = bytearray() if capture_wer_pcm else None
    chunk_arrival_times_s: list[float] = []
    chunk_sizes: list[int] = []
    try:
        async with session.post(url=api_url, json=payload, headers=headers) as response:
            if response.status == 200:
                async for chunk in response.content.iter_any():
                    if not chunk:
                        continue
                    timestamp = time.perf_counter()
                    if output.audio_ttfp == 0.0:
                        # TTS speech endpoint emits no text tokens, so TTFT is
                        # not defined here; only audio TTFP is meaningful.
                        output.audio_ttfp = timestamp - st
                    total_pcm_bytes += len(chunk)
                    chunk_arrival_times_s.append(timestamp - st)
                    chunk_sizes.append(len(chunk))
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

                continuity = compute_continuity_stats(
                    chunk_arrival_times_s=chunk_arrival_times_s,
                    chunk_bytes=chunk_sizes,
                    sample_rate=sample_rate,
                    sample_width=sample_width,
                    channels=channels,
                    threshold_s=_audio_continuity_threshold_s(),
                )
                output.audio_underrun_s = continuity.max_underrun_s
                output.audio_continuity_ok = continuity.is_continuous
                output.audio_underrun_event_count = continuity.underrun_event_count
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

ASYNC_REQUEST_FUNCS["openai-image-edits-omni"] = async_request_openai_image_edits_omni
if "openai-image-edits-omni" not in OPENAI_COMPATIBLE_BACKENDS:
    OPENAI_COMPATIBLE_BACKENDS.append("openai-image-edits-omni")

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

from vllm_omni.benchmarks.metrics.metrics import (
    MultiModalsBenchmarkMetrics,
    calculate_metrics,
)

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
    self_timed: bool = False,
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
        self_timed,
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
            print_stage=_PRINT_STAGE,
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
            defs.TOTAL_AUDIO_DURATION_S: getattr(metrics, defs.TOTAL_AUDIO_DURATION_S),
            defs.TOTAL_AUDIO_FRAMES: getattr(metrics, defs.TOTAL_AUDIO_FRAMES),
            defs.AUDIO_THROUGHPUT: getattr(metrics, defs.AUDIO_THROUGHPUT),
            defs.TOTAL_IMAGES: getattr(metrics, defs.TOTAL_IMAGES),
            defs.IMAGE_THROUGHPUT: getattr(metrics, defs.IMAGE_THROUGHPUT),
            defs.AVERAGE_PIXELS_PER_IMAGE: getattr(metrics, defs.AVERAGE_PIXELS_PER_IMAGE),
            defs.MEAN_DENOISE_STEP_LATENCY_MS: getattr(metrics, defs.MEAN_DENOISE_STEP_LATENCY_MS),
            "input_lens": [output.prompt_len for output in outputs],
            "start_times": [output.start_time for output in outputs],
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

    result_percentile_metrics: list[str] = []
    if "ttft" in selected_percentile_metrics:
        result_percentile_metrics.append("ttft")
    if "tpot" in selected_percentile_metrics or "tpop" in selected_percentile_metrics:
        result_percentile_metrics.append("tpot")
    if "itl" in selected_percentile_metrics:
        result_percentile_metrics.append("itl")
    if "e2el" in selected_percentile_metrics:
        result_percentile_metrics.append("e2el")
    for metric in selected_percentile_metrics:
        if metric.startswith("audio") and metric not in result_percentile_metrics:
            result_percentile_metrics.append(metric)

    def process_one_metric(
        # E.g., "ttft"
        metric_attribute_name: str,
    ):
        # This function prints and adds statistics of the specified
        # metric.
        if metric_attribute_name not in result_percentile_metrics:
            return
        # No text tokens generated (e.g. pure TTS speech endpoint): per-token
        # latency metrics (ttft/tpot/itl) are undefined, so skip them.
        is_text_token_metric = not (metric_attribute_name == "e2el" or metric_attribute_name.startswith("audio"))
        if is_text_token_metric and getattr(metrics, "total_output", 0) == 0:
            return
        is_audio_rtf = metric_attribute_name == defs.AUDIO_RTF
        is_audio_duration_or_underrun = metric_attribute_name in (defs.AUDIO_DURATION, defs.AUDIO_UNDERRUN)

        suffix = "_ms"
        if is_audio_duration_or_underrun:
            suffix = "_s"
        elif is_audio_rtf:
            suffix = ""
        mean_attr_name = f"mean_{metric_attribute_name}{suffix}"
        mean_value = getattr(metrics, mean_attr_name, 0.0)
        result[mean_attr_name] = mean_value

        median_attr_name = f"median_{metric_attribute_name}{suffix}"
        median_value = getattr(metrics, median_attr_name, 0.0)
        result[median_attr_name] = median_value
        for p, value in getattr(metrics, f"percentiles_{metric_attribute_name}{suffix}", None) or []:
            p_word = str(int(p)) if int(p) == p else str(p)
            result[f"p{p_word}_{metric_attribute_name}{suffix}"] = value

    if task_type == TaskType.GENERATION:
        for metric in result_percentile_metrics:
            process_one_metric(metric)
    else:
        result_percentile_metrics.append("e2el")
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
