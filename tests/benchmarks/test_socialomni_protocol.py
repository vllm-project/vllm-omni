# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import base64
import json
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web

from benchmarks.socialomni import evaluate as entrypoint
from benchmarks.socialomni import protocol
from benchmarks.socialomni.client import RequestResult, run_phase
from benchmarks.socialomni.dataset import SocialOmniLevel1Sample, SocialOmniLevel2Sample
from benchmarks.socialomni.protocol import (
    JUDGE_MAX_TOKENS,
    LEVEL2_RESPONSE_MAX_TOKENS,
    LEVEL2_WHEN_MAX_TOKENS,
    build_judge_prompt,
    build_response_prompt,
    build_when_prompt,
    load_judge_config,
    parse_choice,
    parse_judge_score,
    parse_when,
    request_chat_completion,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _level1(path: str = "/tmp/video.mp4") -> SocialOmniLevel1Sample:
    return SocialOmniLevel1Sample("one", path, "Who?", ("one", "two", "three", "four"), "A", "speaker_visible")


def _level2(index: int = 0) -> SocialOmniLevel2Sample:
    return SocialOmniLevel2Sample(
        str(index),
        "/tmp/video.mp4",
        3.0,
        "Alex",
        "Should Alex speak now?",
        "What should Alex say?",
        "NO",
        "private reference response",
        "private reference transcript",
    )


def _config(**overrides) -> entrypoint.SocialOmniEvalConfig:
    values = {
        "dataset_root": ".",
        "model": "qwen3-omni",
        "base_url": "http://localhost:8000",
        "level": "level1",
        "judge_config": None,
        "prefix_cache_dir": "cache",
        "mini": False,
        "max_samples": None,
        "max_concurrency": 1,
        "timeout_s": 30,
        "output_dir": "results",
    }
    values.update(overrides)
    return entrypoint.SocialOmniEvalConfig(**values)


def test_model_prompts_do_not_leak_reference_material() -> None:
    sample = _level2()
    when = build_when_prompt(sample)
    response = build_response_prompt(sample)
    for secret in (sample.reference_context, sample.reference_response):
        assert secret not in when
        assert secret not in response
    judge = build_judge_prompt(sample, "candidate")
    assert sample.reference_context in judge
    assert sample.reference_response in judge


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Answer: A", "A"),
        ("\\boxed{B}", "B"),
        ("C", "C"),
        ("A or B", ""),
        ("Answer: A or B", ""),
        ("Answer: A/B", ""),
        ("Answer: A and B", ""),
        ("Answer: A. Answer: B", ""),
        ("\\boxed{A} or B", ""),
    ],
)
def test_choice_parser_is_strict(raw: str, expected: str) -> None:
    assert parse_choice(raw, ("A", "B", "C", "D")) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Answer: A", "YES"),
        ("Answer: B", "NO"),
        ("YES", "YES"),
        ("maybe", ""),
        ("Answer: A or B", ""),
    ],
)
def test_when_parser(raw: str, expected: str) -> None:
    assert parse_when(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("75", 75),
        ("Score: 25", 25),
        ("75 or 100", None),
        ("-25", None),
        ("25.5", None),
        (".25", None),
        ("0.75", None),
        ("125", None),
        ("80 or 75", None),
    ],
)
def test_judge_score_parser(raw: str, expected: int | None) -> None:
    assert parse_judge_score(raw) == expected


def test_judge_config_has_only_fixed_public_fields(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SECRET_VALUE", "must-not-appear")
    path = tmp_path / "judges.json"
    path.write_text(
        json.dumps(
            {
                "judges": [
                    {
                        "name": name,
                        "model": name,
                        "base_url": "https://user:password@example.com:443/v1?api_key=secret#token",
                        "api_key_env": "SECRET_VALUE" if index == 0 else None,
                        "max_concurrency": 1,
                    }
                    for index, name in enumerate(("gpt-4o", "gemini-2.5-pro", "qwen3-omni"))
                ]
            }
        ),
        encoding="utf-8",
    )
    public = [judge.public_dict() for judge in load_judge_config(path)]
    assert "must-not-appear" not in json.dumps(public)
    assert public[0] == {
        "name": "gpt-4o",
        "model": "gpt-4o",
        "base_url": "https://example.com:443/v1",
        "max_concurrency": 1,
    }


@pytest.mark.parametrize("api_key_env", ["", " ", " KEY", "KEY ", 1])
def test_judge_config_rejects_invalid_key_names(tmp_path, api_key_env) -> None:
    path = tmp_path / "judges.json"
    path.write_text(
        json.dumps(
            {
                "judges": [
                    {
                        "name": name,
                        "model": name,
                        "base_url": "http://localhost:8000",
                        "api_key_env": api_key_env,
                    }
                    for name in ("gpt-4o", "gemini-2.5-pro", "qwen3-omni")
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="api_key_env"):
        load_judge_config(path)


@asynccontextmanager
async def _server(handler):
    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    server = web.AppRunner(app)
    await server.setup()
    try:
        site = web.TCPSite(server, "127.0.0.1", 0)
        await site.start()
        yield f"http://127.0.0.1:{server.addresses[0][1]}"
    finally:
        await server.cleanup()


def _completion(text, **extra):
    return web.json_response({"choices": [{"message": {"content": text}}], **extra})


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_judge", [None, "qwen3-omni"])
async def test_complete_protocol_over_http(tmp_path, monkeypatch, failed_judge):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"before-query\x00after-query")
    prefix = tmp_path / "prefix.mp4"
    prefix.write_bytes(b"before-query\x00")
    samples = [
        replace(_level2(0), gold_when="YES", video_path=str(source)),
        replace(_level2(1), video_path=str(source)),
    ]
    seen = []

    async def completion(request):
        payload = await request.json()
        seen.append(payload)
        limit = payload["max_tokens"]
        if limit == JUDGE_MAX_TOKENS:
            assert payload["messages"][0]["content"].count("private reference") == 2
            assert "video_url" not in json.dumps(payload)
            if payload["model"] == failed_judge:
                return web.Response(status=400, text="judge unavailable")
        else:
            assert payload["mm_processor_kwargs"] == {"use_audio_in_video": True}
            assert payload["modalities"] == ["text"]
            content = payload["messages"][0]["content"]
            assert [part["type"] for part in content] == ["video_url", "text"]
            video_uri = content[0]["video_url"]["url"]
            media_type, encoded = video_uri.split(",", 1)
            assert media_type == "data:video/mp4;base64"
            expected = source.read_bytes() if limit == 32 else prefix.read_bytes()
            assert base64.b64decode(encoded, validate=True) == expected
            assert "private reference" not in json.dumps(payload)
            assert "videos" not in payload and "use_audio_in_video" not in payload
            if limit == 32:
                assert content[1]["text"].endswith(
                    "Reply only as Answer: X, where X is A, B, C, or D. Do not include an explanation."
                )
        text = {32: "Answer: A", 8: "Answer: B", 256: "candidate", 8192: "75"}[limit]
        return _completion(text, usage={"prompt_tokens": 2, "completion_tokens": 1})

    async def prepared(*args):
        return prefix

    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setattr(protocol, "create_video_prefix", prepared)
    monkeypatch.setattr(entrypoint, "load_socialomni_level1_samples", lambda *a, **k: [_level1(str(source))])
    monkeypatch.setattr(entrypoint, "load_socialomni_level2_samples", lambda *a, **k: samples)
    monkeypatch.setattr(
        entrypoint,
        "inspect_socialomni_dataset",
        lambda *a, **k: {
            "metadata_matches_expected_revision": False,
        },
    )
    async with _server(completion) as base_url:
        config_path = tmp_path / "judges.json"
        config_path.write_text(
            json.dumps(
                {
                    "judges": [
                        {"name": name, "model": name, "base_url": base_url}
                        for name in entrypoint.SOCIALOMNI_JUDGE_NAMES
                    ]
                }
            )
        )
        result = await entrypoint.run_socialomni(
            _config(
                level="both",
                base_url=base_url,
                judge_config=str(config_path),
                warmup=0,
            )
        )
    assert len(seen) == 7
    assert result["config"]["judges"] == [
        {"name": name, "model": name, "base_url": base_url, "max_concurrency": 1}
        for name in entrypoint.SOCIALOMNI_JUDGE_NAMES
    ]
    assert result["summary"]["elapsed_s"] > 0
    positive, negative = result["per_sample"]["level2"]
    assert positive["predicted_when"] == "NO"
    assert positive["gold_response"] == "candidate"
    assert negative["gold_response_success"] is None
    assert positive["gold_judge_scores"] == {
        name: 75 for name in entrypoint.SOCIALOMNI_JUDGE_NAMES if name != failed_judge
    }
    metrics = result["summary"]["level2"]["metrics"]
    assert metrics["judge_status"]["required_scores"] == 3
    assert metrics["judge_status"]["complete"] is (failed_judge is None)
    assert result["summary"]["status"] == ("incomplete" if failed_judge else "complete")
    if failed_judge:
        assert metrics["quality"] is None
        assert result["failures"][0]["judge"] == failed_judge
    else:
        assert not result["failures"]
        assert metrics["quality"]["qgold"] == 75
        assert metrics["quality"]["cov_plus"] == 0


@pytest.mark.asyncio
async def test_malformed_response_makes_level2_incomplete(tmp_path, monkeypatch):
    prefix = tmp_path / "prefix.mp4"
    prefix.write_bytes(b"prefix")
    sample = replace(_level2(0), gold_when="YES", video_path=str(prefix))

    async def prepared(*args):
        return prefix

    async def completion(request):
        payload = await request.json()
        if payload["max_tokens"] == LEVEL2_RESPONSE_MAX_TOKENS:
            return web.json_response({})
        assert payload["max_tokens"] == LEVEL2_WHEN_MAX_TOKENS
        return _completion("Answer: A")

    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setattr(protocol, "create_video_prefix", prepared)
    monkeypatch.setattr(entrypoint, "load_socialomni_level2_samples", lambda *a, **k: [sample])
    monkeypatch.setattr(entrypoint, "inspect_socialomni_dataset", lambda *a, **k: {})
    async with _server(completion) as base_url:
        judges = tmp_path / "judges.json"
        judges.write_text(
            json.dumps(
                {
                    "judges": [
                        {"name": name, "model": name, "base_url": base_url}
                        for name in entrypoint.SOCIALOMNI_JUDGE_NAMES
                    ]
                }
            )
        )
        result = await entrypoint.run_socialomni(
            _config(level="level2", base_url=base_url, judge_config=str(judges), warmup=0)
        )
    assert result["summary"]["status"] == "incomplete"
    assert not result["per_sample"]["level2"][0]["gold_response_success"]
    assert len(result["failures"]) == 1
    assert "invalid completion response" in result["failures"][0]["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {},
        {"choices": []},
        {"choices": [None]},
        {"choices": [{}]},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": 42}}]},
        {"choices": [{"message": {"content": [{"type": "text", "text": 42}]}}]},
    ],
)
async def test_malformed_completion_is_a_request_failure(body, monkeypatch):
    monkeypatch.setenv("no_proxy", "127.0.0.1")

    async def completion(request):
        return web.json_response(body)

    async with _server(completion) as base_url:
        async with aiohttp.ClientSession() as session:
            result = await request_chat_completion(
                session,
                api_url=f"{base_url}/v1/chat/completions",
                payload={},
                request_id="response",
            )
    assert not result.is_success
    assert "invalid completion response" in result.error


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["", None, []])
async def test_empty_completion_remains_successful(content, monkeypatch):
    monkeypatch.setenv("no_proxy", "127.0.0.1")

    async def completion(request):
        return _completion(content)

    async with _server(completion) as base_url:
        async with aiohttp.ClientSession() as session:
            result = await request_chat_completion(
                session,
                api_url=f"{base_url}/v1/chat/completions",
                payload={},
                request_id="response",
            )
    assert result.is_success
    assert result.text == ""
    assert result.error == ""


@pytest.mark.asyncio
async def test_phase_warmup_concurrency_order_and_failures(monkeypatch, capsys):
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    active, peak = 0, 0
    seen: Counter[int] = Counter()

    async def completion(request):
        nonlocal active, peak
        item = (await request.json())["item"]
        seen[item] += 1
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.005 if item else 0.015)
        active -= 1
        return _completion(str(item)) if item != 2 else web.Response(status=400, text="bad input")

    async with _server(completion) as base_url:

        async def send(session, item):
            return await request_chat_completion(
                session,
                api_url=f"{base_url}/v1/chat/completions",
                payload={"item": item},
                request_id=str(item),
            )

        results, wall_s = await run_phase(
            [0, 1, 2],
            send,
            max_concurrency=2,
            timeout_s=5,
            description="requests",
        )
    assert seen == {0: 2, 1: 2, 2: 1}
    assert peak == 2
    assert [result.request_id for result in results] == ["0", "1", "2"]
    assert [result.is_success for result in results] == [True, True, False]
    assert "HTTP 400: bad input" in results[-1].error
    assert wall_s >= max(result.latency_s for result in results)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "requests: warming up 2 requests" in captured.err
    assert "requests: 0/3" in captured.err
    assert "requests: 3/3, 1 failed" in captured.err


@pytest.mark.asyncio
async def test_prefix_failure_stays_in_denominator_and_response_uses_gold(tmp_path, monkeypatch):
    samples = [replace(_level2(i), gold_when="YES", video_path=f"/tmp/{i}.mp4") for i in range(2)]
    prefix = tmp_path / "prefix.mp4"
    prefix.write_bytes(b"prefix bytes")

    async def prepared(path, *_args):
        if path.endswith("/0.mp4"):
            raise RuntimeError("broken media")
        return prefix

    sent = []

    async def request(session, **kwargs):
        sent.append(kwargs["request_id"])
        return RequestResult(
            request_id=kwargs["request_id"],
            text="NO" if kwargs["request_id"].endswith(":when") else "candidate",
            is_success=True,
        )

    monkeypatch.setattr(protocol, "create_video_prefix", prepared)
    monkeypatch.setattr(protocol, "request_chat_completion", request)
    records, requests, _ = await protocol.run_level2_model(
        samples,
        model="qwen3-omni",
        base_url="http://localhost:8000",
        prefix_cache_dir=tmp_path,
        max_concurrency=2,
        timeout_s=5,
        warmup=0,
    )
    assert len(records) == 2
    assert not records[0]["when_success"] and not records[0]["gold_response_success"]
    assert requests[0].error == "RuntimeError: broken media"
    assert records[1]["predicted_when"] == "NO" and records[1]["gold_response"] == "candidate"
    assert sent == ["1:when", "1:response"]


def test_model_payload_embeds_video_bytes(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"\x00\xff\x80video content")
    payload = protocol.model_payload("qwen3-omni", "prompt", str(video), 8)
    video_part, text_part = payload["messages"][0]["content"]
    assert video_part["type"] == "video_url"
    media_type, encoded = video_part["video_url"]["url"].split(",", 1)
    assert media_type == "data:video/mp4;base64"
    assert base64.b64decode(encoded, validate=True) == video.read_bytes()
    assert text_part == {"type": "text", "text": "prompt"}
    assert payload["mm_processor_kwargs"] == {"use_audio_in_video": True}
    assert payload["modalities"] == ["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("directory", [False, True])
async def test_unreadable_media_is_request_failure(tmp_path, monkeypatch, directory):
    media = tmp_path if directory else tmp_path / "missing.mp4"

    async def unexpected_request(*args, **kwargs):
        pytest.fail("unreadable media must fail before an HTTP request")

    monkeypatch.setattr(protocol, "request_chat_completion", unexpected_request)
    send = protocol.make_level1_send_fn("qwen3-omni", "http://localhost:8000")
    async with aiohttp.ClientSession() as session:
        result = await send(session, _level1(str(media)))
    assert not result.is_success
    assert result.request_id == "one"
    assert result.error
    assert result.prompt_tokens == result.completion_tokens == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["prompt_tokens", "completion_tokens"])
@pytest.mark.parametrize("value", [True, -1, 1.5, "2", None])
async def test_invalid_usage_is_request_failure(monkeypatch, field, value):
    monkeypatch.setenv("no_proxy", "127.0.0.1")

    async def completion(request):
        return _completion("A", usage={field: value})

    async with _server(completion) as base_url:
        async with aiohttp.ClientSession() as session:
            result = await request_chat_completion(
                session,
                api_url=f"{base_url}/v1/chat/completions",
                payload={},
                request_id="invalid",
            )
    assert not result.is_success
    assert "invalid token usage" in result.error
    assert result.prompt_tokens == result.completion_tokens == 0


def test_judge_completeness_requires_configuration_and_every_score():
    scores = dict.fromkeys(entrypoint.SOCIALOMNI_JUDGE_NAMES, 75)
    row = {
        "sample_id": "one",
        "gold_when": "YES",
        "gold_response_success": True,
        "gold_response": "candidate",
        "gold_judge_scores": scores,
    }
    assert entrypoint._judges_complete([row], True)
    assert not entrypoint._judges_complete([row], False)
    del scores["gpt-4o"]
    assert not entrypoint._judges_complete([row], True)
