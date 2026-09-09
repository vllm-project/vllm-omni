# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request construction and evaluation phases for SocialOmni."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiohttp

from benchmarks.socialomni.client import RequestResult, run_phase
from benchmarks.socialomni.dataset import (
    SocialOmniLevel1Sample,
    SocialOmniLevel2Sample,
    create_video_prefix,
)
from benchmarks.socialomni.metrics import (
    SOCIALOMNI_JUDGE_NAMES,
    SOCIALOMNI_SCORE_BUCKETS,
)

RETRYABLE_STATUS = frozenset({408, 429})
# Reasoning-capable judges may consume hidden tokens before emitting the score.
JUDGE_MAX_TOKENS = 8192
JUDGE_PARSE_ATTEMPTS = 3
LEVEL1_MAX_TOKENS = 32
LEVEL2_WHEN_MAX_TOKENS = 8
LEVEL2_RESPONSE_MAX_TOKENS = 256


@dataclass(frozen=True)
class JudgeSpec:
    name: str
    model: str
    base_url: str
    api_key_env: str | None
    max_concurrency: int


def chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"


def load_judge_config(path: str | Path) -> list[JudgeSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("judges") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("judge config must contain exactly three judges")
    allowed = {"name", "model", "base_url", "api_key_env", "max_concurrency"}
    judges: list[JudgeSpec] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) - allowed:
            raise ValueError(f"judges[{index}] has invalid fields")
        for field in ("name", "model", "base_url"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"judges[{index}].{field} must be non-empty")
        concurrency = row.get("max_concurrency", 1)
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError(f"judges[{index}].max_concurrency must be >= 1")
        api_key_env = row.get("api_key_env")
        if api_key_env is not None and (
            not isinstance(api_key_env, str) or not api_key_env or api_key_env != api_key_env.strip()
        ):
            raise ValueError(
                f"judges[{index}].api_key_env must be a non-empty string without surrounding whitespace, or null"
            )
        judges.append(
            JudgeSpec(
                name=row["name"].strip(),
                model=row["model"].strip(),
                base_url=row["base_url"].strip(),
                api_key_env=api_key_env,
                max_concurrency=concurrency,
            )
        )
    if {judge.name for judge in judges} != set(SOCIALOMNI_JUDGE_NAMES):
        raise ValueError(f"judge names must be exactly {SOCIALOMNI_JUDGE_NAMES}")
    return judges


def _headers(api_key_env: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key_env:
        token = os.environ.get(api_key_env)
        if not token:
            raise RuntimeError(f"API key environment variable is not set: {api_key_env}")
        headers["Authorization"] = f"Bearer {token}"
    return headers


def validate_judge_credentials(judges: Sequence[JudgeSpec]) -> None:
    """Fail before model inference when a configured judge credential is absent."""
    for judge in judges:
        _headers(judge.api_key_env)


def _response_text(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("invalid completion response: expected non-empty choices")
    if not isinstance(choices[0], dict):
        raise RuntimeError("invalid completion response: expected a choice object")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise RuntimeError("invalid completion response: expected a message object")
    if "content" not in message:
        raise RuntimeError("invalid completion response: missing message content")
    content = message["content"]
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        if any(
            not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str)
            for part in content
        ):
            raise RuntimeError("invalid completion response: expected text content parts")
        return "\n".join(part["text"].strip() for part in content).strip()
    if content is None:
        return ""
    raise RuntimeError("invalid completion response: unsupported message content")


async def request_chat_completion(
    session: aiohttp.ClientSession,
    *,
    api_url: str,
    payload: dict[str, Any],
    request_id: str,
    api_key_env: str | None = None,
    max_attempts: int = 3,
) -> RequestResult:
    """Send a chat completion, retrying transient errors up to max_attempts."""
    request_started = time.perf_counter()
    last = RequestResult(request_id=request_id, error="not attempted")
    for attempt in range(max_attempts):
        try:
            async with session.post(api_url, json=payload, headers=_headers(api_key_env)) as response:
                raw = await response.text()
                if response.status >= 400:
                    last = RequestResult(
                        request_id=request_id,
                        latency_s=time.perf_counter() - request_started,
                        error=f"HTTP {response.status}: {raw[:2000]}",
                    )
                    retry = response.status in RETRYABLE_STATUS or 500 <= response.status < 600
                else:
                    try:
                        body = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        last = RequestResult(
                            request_id=request_id,
                            latency_s=time.perf_counter() - request_started,
                            error=f"invalid JSON response: {exc}: {raw[:1000]}",
                        )
                        retry = True
                    else:
                        if not isinstance(body, dict):
                            last = RequestResult(
                                request_id=request_id,
                                latency_s=time.perf_counter() - request_started,
                                error=f"invalid JSON response object: {raw[:1000]}",
                            )
                            retry = True
                        else:
                            usage = body.get("usage") or {}
                            if not isinstance(usage, dict):
                                usage = {}
                            try:
                                prompt_tokens = usage.get("prompt_tokens", 0)
                                completion_tokens = usage.get("completion_tokens", 0)
                                for count in (prompt_tokens, completion_tokens):
                                    if type(count) is not int or count < 0:
                                        raise ValueError("token counts must be non-negative integers")
                            except (TypeError, ValueError, OverflowError) as exc:
                                return RequestResult(
                                    request_id=request_id,
                                    latency_s=time.perf_counter() - request_started,
                                    error=f"invalid token usage: {exc}",
                                )
                            elapsed = time.perf_counter() - request_started
                            return RequestResult(
                                request_id=request_id,
                                text=_response_text(body),
                                is_success=True,
                                latency_s=elapsed,
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                            )
        except RuntimeError as exc:
            return RequestResult(
                request_id=request_id,
                latency_s=time.perf_counter() - request_started,
                error=str(exc),
            )
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last = RequestResult(
                request_id=request_id,
                latency_s=time.perf_counter() - request_started,
                error=f"{type(exc).__name__}: {exc}",
            )
            retry = True
        if not retry or attempt + 1 == max_attempts:
            return last
        await asyncio.sleep(2**attempt)
    return last


def parse_choice(text: str, choices: Sequence[str]) -> str:
    content = (text or "").strip().upper()
    alphabet = "".join(re.escape(choice) for choice in choices)
    match = re.fullmatch(rf"(?:(?:ANSWER|CHOICE)\s*(?:IS|:)?\s*)?([{alphabet}])[.)]?", content) or re.fullmatch(
        rf"\\?BOXED\s*\{{\s*([{alphabet}])\s*\}}", content
    )
    return match.group(1) if match else ""


def build_level1_result_records(
    samples: Sequence[SocialOmniLevel1Sample], results: Sequence[RequestResult]
) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": sample.sample_id,
            "gold_answer": sample.gold_answer,
            "predicted_answer": (parse_choice(result.text, ("A", "B", "C", "D")) if result.is_success else ""),
            "visibility": sample.visibility,
            "is_success": result.is_success,
            "raw_response": result.text,
            "request": asdict(result),
        }
        for sample, result in zip(samples, results, strict=True)
    ]


def parse_when(text: str) -> str:
    choice = parse_choice(text, ("A", "B"))
    if choice:
        return "YES" if choice == "A" else "NO"
    normalized = (text or "").strip().upper()
    return normalized if normalized in {"YES", "NO"} else ""


def parse_judge_score(text: str) -> int | None:
    values = re.findall(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)", text or "")
    if len(values) != 1 or values[0] not in {"0", "25", "50", "75", "100"}:
        return None
    return int(values[0])


def build_level1_prompt(sample: SocialOmniLevel1Sample) -> str:
    options = "\n".join(
        f"{letter}. {option}" for letter, option in zip(("A", "B", "C", "D"), sample.options, strict=True)
    )
    return (
        f"{sample.question}\n{options}\nUse the video and its audio. "
        "Reply only as Answer: X, where X is A, B, C, or D. Do not include an explanation."
    )


def build_when_prompt(sample: SocialOmniLevel2Sample) -> str:
    return (
        f"Target participant: {sample.target_participant}\n{sample.question_when}\n"
        "A. YES\nB. NO\nUse only the provided audio-video prefix. "
        "Reply exactly as Answer: A or Answer: B."
    )


def build_response_prompt(sample: SocialOmniLevel2Sample) -> str:
    return (
        f"Target participant: {sample.target_participant}\n{sample.question_how}\n"
        "Use only the provided audio-video prefix. Return only the participant's next utterance."
    )


def build_judge_prompt(sample: SocialOmniLevel2Sample, candidate: str) -> str:
    return (
        "Score the candidate continuation for contextual fit, target-role consistency, "
        "intent correctness, and key information completeness. Choose exactly one score "
        "from {0, 25, 50, 75, 100}. Output only the score.\n\n"
        f"Target participant:\n{sample.target_participant}\n\n"
        f"Reference context:\n{sample.reference_context}\n\n"
        f"Reference continuation:\n{sample.reference_response}\n\n"
        f"Candidate continuation:\n{candidate}"
    )


def model_payload(model: str, prompt: str, video_path: str, max_tokens: int) -> dict[str, Any]:
    video = base64.b64encode(Path(video_path).read_bytes()).decode("ascii")
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "mm_processor_kwargs": {"use_audio_in_video": True},
        "modalities": ["text"],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }


def judge_payload(judge: JudgeSpec, prompt: str) -> dict[str, Any]:
    return {
        "model": judge.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": JUDGE_MAX_TOKENS,
        "temperature": 0.0,
        "stream": False,
    }


async def _request_model(
    session: aiohttp.ClientSession,
    *,
    model: str,
    base_url: str,
    prompt: str,
    video_path: str,
    max_tokens: int,
    request_id: str,
) -> RequestResult:
    try:
        payload = await asyncio.to_thread(model_payload, model, prompt, video_path, max_tokens)
    except OSError as exc:
        return RequestResult(request_id=request_id, error=f"media preparation failed: {exc}")
    return await request_chat_completion(
        session, api_url=chat_completions_url(base_url), payload=payload, request_id=request_id
    )


def make_level1_send_fn(
    model: str, base_url: str
) -> Callable[[aiohttp.ClientSession, SocialOmniLevel1Sample], Awaitable[RequestResult]]:
    async def send(session: aiohttp.ClientSession, sample: SocialOmniLevel1Sample) -> RequestResult:
        return await _request_model(
            session,
            model=model,
            base_url=base_url,
            prompt=build_level1_prompt(sample),
            video_path=sample.video_path,
            max_tokens=LEVEL1_MAX_TOKENS,
            request_id=sample.sample_id,
        )

    return send


async def run_level2_model(
    samples: Sequence[SocialOmniLevel2Sample],
    *,
    model: str,
    base_url: str,
    prefix_cache_dir: str | Path,
    max_concurrency: int,
    timeout_s: int,
    warmup: int | None = None,
) -> tuple[list[dict[str, Any]], list[RequestResult], float]:
    """Prepare video prefixes, then run decisions and gold-positive responses."""
    records: list[dict[str, Any]] = []
    prepared: list[tuple[SocialOmniLevel2Sample, Path]] = []
    requests: list[RequestResult] = []
    for sample in samples:
        record = {
            "sample_id": sample.sample_id,
            "gold_when": sample.gold_when,
            "predicted_when": "",
            "when_success": False,
            "when_raw_response": "",
            "gold_response": "",
            "gold_response_success": False if sample.gold_when == "YES" else None,
            "gold_judge_scores": {},
            "judge_results": {},
            "requests": [],
        }
        records.append(record)
        try:
            prefix = await create_video_prefix(sample.video_path, sample.timestamp_s, prefix_cache_dir)
        except (OSError, RuntimeError, ValueError) as exc:
            failure = RequestResult(
                request_id=f"{sample.sample_id}:prefix",
                error=f"{type(exc).__name__}: {exc}",
            )
            requests.append(failure)
            record["requests"].append(asdict(failure))
        else:
            prepared.append((sample, prefix))

    by_id = {record["sample_id"]: record for record in records}
    measured_wall_s = 0.0
    for phase in ("when", "response"):
        cohort = [item for item in prepared if phase == "when" or item[0].gold_when == "YES"]
        if not cohort:
            continue

        async def send(session: aiohttp.ClientSession, item: tuple[SocialOmniLevel2Sample, Path]) -> RequestResult:
            sample, prefix = item
            prompt = build_when_prompt(sample) if phase == "when" else build_response_prompt(sample)
            max_tokens = LEVEL2_WHEN_MAX_TOKENS if phase == "when" else LEVEL2_RESPONSE_MAX_TOKENS
            return await _request_model(
                session,
                model=model,
                base_url=base_url,
                prompt=prompt,
                video_path=str(prefix),
                max_tokens=max_tokens,
                request_id=f"{sample.sample_id}:{phase}",
            )

        outcomes, wall_s = await run_phase(
            cohort, send, max_concurrency=max_concurrency, timeout_s=timeout_s, warmup=warmup
        )
        measured_wall_s += wall_s
        requests.extend(outcomes)
        for (sample, _prefix), result in zip(cohort, outcomes, strict=True):
            record = by_id[sample.sample_id]
            record["requests"].append(asdict(result))
            if phase == "when":
                record["when_success"] = result.is_success
                record["when_raw_response"] = result.text
                record["predicted_when"] = parse_when(result.text) if result.is_success else ""
            else:
                record["gold_response"] = result.text
                record["gold_response_success"] = result.is_success
    return records, requests, measured_wall_s


async def run_judges(
    samples: Sequence[SocialOmniLevel2Sample],
    records: list[dict[str, Any]],
    judges: Sequence[JudgeSpec],
    *,
    timeout_s: int,
) -> tuple[list[RequestResult], list[dict[str, str]]]:
    """Score responses with each judge's configured concurrency limit."""
    by_id = {sample.sample_id: sample for sample in samples}
    eligible = [
        record
        for record in records
        if record["gold_when"] == "YES" and record["gold_response_success"] and str(record["gold_response"]).strip()
    ]

    async def run_judge(judge: JudgeSpec) -> list[RequestResult]:
        async def send(session: aiohttp.ClientSession, record: dict[str, Any]) -> RequestResult:
            sample = by_id[str(record["sample_id"])]
            started = time.perf_counter()
            total_prompt_tokens = 0
            total_completion_tokens = 0
            score = None
            for attempt in range(JUDGE_PARSE_ATTEMPTS):
                result = await request_chat_completion(
                    session,
                    api_url=chat_completions_url(judge.base_url),
                    payload=judge_payload(judge, build_judge_prompt(sample, str(record["gold_response"]))),
                    request_id=f"{sample.sample_id}:judge:{judge.name}",
                    api_key_env=judge.api_key_env,
                )
                total_prompt_tokens += result.prompt_tokens
                total_completion_tokens += result.completion_tokens
                if not result.is_success:
                    break
                score = parse_judge_score(result.text)
                if score in SOCIALOMNI_SCORE_BUCKETS:
                    break
                if attempt + 1 < JUDGE_PARSE_ATTEMPTS:
                    await asyncio.sleep(2**attempt)
            result.latency_s = time.perf_counter() - started
            result.prompt_tokens = total_prompt_tokens
            result.completion_tokens = total_completion_tokens
            if score not in SOCIALOMNI_SCORE_BUCKETS:
                result.is_success = False
                result.error = result.error or (
                    f"invalid judge score after {JUDGE_PARSE_ATTEMPTS} attempts: {result.text!r}"
                )
            return result

        results, _ = await run_phase(
            eligible, send, max_concurrency=judge.max_concurrency, timeout_s=timeout_s, warmup=0
        )
        return results

    outcomes = await asyncio.gather(*(run_judge(judge) for judge in judges))
    failures: list[dict[str, str]] = []
    results: list[RequestResult] = []
    for judge, judge_results in zip(judges, outcomes, strict=True):
        for record, result in zip(eligible, judge_results, strict=True):
            score = parse_judge_score(result.text) if result.is_success else None
            results.append(result)
            record["judge_results"][judge.name] = {
                "score": score,
                "raw_response": result.text,
                "is_success": result.is_success,
                "latency_s": result.latency_s,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "error": result.error,
            }
            if result.is_success and score is not None:
                record["gold_judge_scores"][judge.name] = score
            else:
                failures.append(
                    {
                        "request_id": result.request_id,
                        "sample_id": str(record["sample_id"]),
                        "judge": judge.name,
                        "error": result.error,
                    }
                )
    return results, failures
