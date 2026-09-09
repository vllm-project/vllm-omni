# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import json
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
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
                        "base_url": "http://localhost:8000",
                        "api_key_env": "SECRET_VALUE" if index == 0 else None,
                        "max_concurrency": 1,
                    }
                    for index, name in enumerate(("gpt-4o", "gemini-2.5-pro", "qwen3-omni"))
                ]
            }
        ),
        encoding="utf-8",
    )
    public = [asdict(judge) for judge in load_judge_config(path)]
    assert "must-not-appear" not in json.dumps(public)
    assert public[0]["api_key_env"] == "SECRET_VALUE"


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
    samples = [replace(_level2(0), gold_when="YES"), _level2(1)]
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
            expected_path = Path("/tmp/video.mp4") if limit == 32 else tmp_path / "prefix.mp4"
            assert video_uri == expected_path.resolve().as_uri()
            assert "private reference" not in json.dumps(payload)
            assert "videos" not in payload and "use_audio_in_video" not in payload
        text = {32: "Answer: A", 8: "Answer: B", 256: "candidate", 8192: "75"}[limit]
        return _completion(text, usage={"prompt_tokens": 2, "completion_tokens": 1})

    async def prepared(*args):
        return tmp_path / "prefix.mp4"

    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setattr(protocol, "create_video_prefix", prepared)
    monkeypatch.setattr(entrypoint, "load_socialomni_level1_samples", lambda *a, **k: [_level1()])
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
async def test_phase_warmup_concurrency_order_and_failures(monkeypatch):
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
        )
    assert seen == {0: 2, 1: 2, 2: 1}
    assert peak == 2
    assert [result.request_id for result in results] == ["0", "1", "2"]
    assert [result.is_success for result in results] == [True, True, False]
    assert "HTTP 400: bad input" in results[-1].error
    assert wall_s >= max(result.latency_s for result in results)


@pytest.mark.asyncio
async def test_prefix_failure_stays_in_denominator_and_response_uses_gold(tmp_path, monkeypatch):
    samples = [replace(_level2(i), gold_when="YES", video_path=f"/tmp/{i}.mp4") for i in range(2)]

    async def prepared(path, *_args):
        if path.endswith("/0.mp4"):
            raise RuntimeError("broken media")
        return tmp_path / "prefix.mp4"

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
