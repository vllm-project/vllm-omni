# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Live camera-prefix warmup, cache reuse and cancellation through the duplex API."""

import asyncio
from collections import Counter
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from prometheus_client.parser import text_string_to_metric_families

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
from vllm_omni.clients.duplex import DuplexClient, EventCollector, SessionConfig, read_pcm16_wav

pytestmark = pytest.mark.omni

_DEPLOY = modify_stage_config(
    get_deploy_config_path("qwen3_omni_moe.yaml"),
    updates={
        "session_mode": "duplex",
        "duplex_session": {"visual_prefix_warmup": True, "max_sessions": 4},
        "stages": {
            0: {
                "enable_prefix_caching": True,
                "limit_mm_per_prompt": {"audio": 4, "image": 8},
                "max_model_len": 8192,
                "max_num_batched_tokens": 8192,
                "max_num_seqs": 4,
                "enforce_eager": True,
            },
            1: {
                "max_model_len": 8192,
                "max_num_seqs": 4,
                "gpu_memory_utilization": 0.30,
                "enforce_eager": True,
                "default_sampling_params.max_tokens": 512,
            },
            2: {"max_num_seqs": 4, "enforce_eager": True},
        },
    },
)
_SERVER = OmniServerParams(
    model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    stage_config_path=_DEPLOY,
    startup_timeout=900,
)
_AUDIO = Path(__file__).resolve().parents[2] / "assets/minicpmo_4_5/response_required_16k.wav"
_FINISHED = "vllm:request_success_total"
_HITS = "vllm:prefix_cache_hits_total"


async def _metrics(http: httpx.AsyncClient) -> Counter[tuple[str, str]]:
    response = await http.get("/metrics")
    response.raise_for_status()
    counts: Counter[tuple[str, str]] = Counter()
    for family in text_string_to_metric_families(response.text):
        for sample in family.samples:
            if sample.name in {_FINISHED, _HITS}:
                counts[sample.name, sample.labels["stage"]] += sample.value
    return counts


async def _wait_for_stage0_counter(
    http: httpx.AsyncClient, before: Counter[tuple[str, str]], metric: str
) -> Counter[tuple[str, str]]:
    for _ in range(120):
        after = await _metrics(http)
        if after[metric, "0"] > before[metric, "0"]:
            return after
        await asyncio.sleep(0.5)
    raise AssertionError(f"Stage0 {metric} did not increase: before={before}, after={after}")


async def _session(server, http, image_url, *, action):
    client = DuplexClient(
        f"ws://{server.host}:{server.port}/v1/realtime?duplex=1",
        model=server.model,
        config=SessionConfig(
            auto_response=False,
            turn_detection=None,
            temperature=0.0,
            max_output_tokens=64,
            # A fresh prefix prevents another test's cache from satisfying the reuse check.
            instructions=f"Session {uuid4().hex}. Answer briefly.",
        ),
    )
    events = EventCollector()
    async with client:
        collector = asyncio.create_task(events.consume(client))
        before = await _metrics(http)
        await client.send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": image_url}],
                },
            }
        )
        if action == "close":
            await client.close()
            await collector
        else:
            if action == "wait":
                warmed = await _wait_for_stage0_counter(http, before, _FINISHED)
                assert warmed[_FINISHED, "1"] == before[_FINISHED, "1"]
                assert warmed[_FINISHED, "2"] == before[_FINISHED, "2"]
                assert events.count("response.created") == 0
                assert events.count("response.audio.delta") == 0
            await client.append_audio(read_pcm16_wav(_AUDIO))
            await client.commit(create_response=True)
            async for response in client.responses():
                audio_bytes = 0
                async for chunk in response.audio():
                    audio_bytes += len(chunk)
                    await client.ack_playback(response.played_ms)
                await response.wait(timeout_s=30)
                assert audio_bytes > 0
                assert response.text or response.transcript
                break
            else:
                pytest.fail("committed turn produced no response")
            if action == "wait":
                await _wait_for_stage0_counter(http, warmed, _HITS)
            await client.close()
            await collector
    assert events.errors() == []
    assert events.count("session.closed") == 1


@pytest.mark.core_model
@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", [_SERVER], indirect=True)
@pytest.mark.parametrize("action", ["wait", "commit", "close"])
def test_visual_prefix_warmup(omni_server, tmp_path, action):
    image = generate_synthetic_image(384, 384, cache_dir=tmp_path, seed=42)
    image_url = f"data:image/jpeg;base64,{image['base64']}"

    async def run():
        async with httpx.AsyncClient(base_url=f"http://{omni_server.host}:{omni_server.port}") as http:
            await _session(omni_server, http, image_url, action=action)
            if action == "close":
                # Closing during preprocessing must leave the server usable by the next session.
                await _session(omni_server, http, image_url, action="wait")

    asyncio.run(asyncio.wait_for(run(), timeout=180))
