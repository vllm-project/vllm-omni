# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in M2 acceptance: scripted Qwen3-Omni duplex sessions with Mooncake KV/HS.

Two servers run the same script on the MRv2 duplex deployment: one with caching
off and one with the Mooncake backend. Every Thinker request is audited at the
real restore and Talker handoff boundaries. The script covers a cold turn,
history reuse, a cancelled reply, an injected HS read failure, history that
outgrows the prompt window, and a second session whose audio differs only in
content.

Set MOONCAKE_CONFIG_PATH to a running Store's JSON configuration. Optional
MOONCAKE_TEST_MODEL selects a local copy of the pinned checkpoint. Run with
VLLM_BATCH_INVARIANT=1: exact comparisons need batch-invariant kernels.
"""

from __future__ import annotations

import asyncio
import json
import os
import wave
from pathlib import Path
from uuid import uuid4

import numpy as np
import pybase64 as base64
import pytest
import torch
import yaml
from safetensors.torch import load_file

from tests.e2e.online_serving.helpers.minicpmo_4_5_duplex import validated_input_wav
from tests.engine.mooncake_helpers import AUDIT_ENV, FAIL_HS_READ
from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.advanced_model, pytest.mark.omni, *hardware_marks(res={"cuda": "H100"}, num_cards=2)]

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"
BLOCK_SIZE = 16
WORKER = "tests.engine.mooncake_helpers.AuditedGPUARWorker"
INSTRUCTIONS = "Answer the user's question in one short English sentence."
SAMPLE_RATE = 24000
# Session 1 replays one clip. Its prompt grows by one turn each time until the
# fifth audio commit drops the first turn from the four-audio prompt window.
SESSION_1 = ("cold", "history-hit", "cancelled", "hs-read-failure", "window-shift")
# Session 2 sends the same clip at half amplitude: identical token IDs, but
# different audio content, so a hit must stop before the first audio block.
SESSION_2 = ("changed-media",)


def _clip(scale: float = 1.0) -> bytes:
    """Checked-in 16 kHz speech, resampled to the session's 24 kHz PCM16."""
    with wave.open(str(validated_input_wav())) as source:
        pcm = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float64)
    positions = np.arange(len(pcm) * 3 // 2) * (2.0 / 3.0)
    resampled = np.interp(positions, np.arange(len(pcm)), pcm) * scale
    # A one-second silent tail, as a client streams before committing.
    return np.concatenate((resampled, np.zeros(SAMPLE_RATE))).round().astype("<i2").tobytes()


def _deploy(path: Path, *, mooncake: dict | None) -> Path:
    thinker: dict = {
        "stage_id": 0,
        "worker_cls": WORKER,
        "enable_prefix_caching": False,
        "max_model_len": 4096,
        "max_num_batched_tokens": 256,
        "block_size": BLOCK_SIZE,
        "gpu_memory_utilization": 0.85,
    }
    if mooncake is not None:
        thinker["omni_kv_config"] = {"kv_store_config": {"mooncake_config": mooncake}}
    # The residual-codebook predictor samples unless told otherwise; greedy
    # codes make the audio comparison independent of sampler state.
    talker = {"stage_id": 1, "subtalker_sampling_params": {"do_sample": False}}
    # An absolute base keeps the production YAML's relative inheritance.
    config = {"base_config": get_deploy_config_path("qwen3_omni_duplex_mrv2.yaml"), "stages": [thinker, talker]}
    path.write_text(yaml.safe_dump(config))
    return path


class Session:
    """One recorded Realtime duplex session with explicit commits."""

    def __init__(self, ws, events: list[dict]):
        self.ws = ws
        self.events = events

    async def send(self, event: dict) -> None:
        await self.ws.send(json.dumps(event))

    async def until(self, predicate, *, timeout_s: float = 300) -> None:
        async def receive() -> None:
            while not predicate():
                event = json.loads(await self.ws.recv())
                self.events.append(event)
                assert event.get("type") != "error", event

        await asyncio.wait_for(receive(), timeout=timeout_s)

    def of(self, response_id: str, event_type: str, start: int = 0) -> list[dict]:
        return [
            event
            for event in self.events[start:]
            if event["type"] == event_type
            and (event.get("response_id") or event.get("response", {}).get("id")) == response_id
        ]

    async def turn(self, pcm: bytes, *, cancel: bool = False) -> dict:
        start = len(self.events)
        chunk = SAMPLE_RATE * 2 // 5
        for offset in range(0, len(pcm), chunk):
            await self.send(
                {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm[offset : offset + chunk]).decode()}
            )
        await self.send({"type": "input_audio_buffer.commit"})
        await self.send({"type": "response.create"})
        await self.until(lambda: any(e["type"] == "response.created" for e in self.events[start:]))
        response_id = next(e["response"]["id"] for e in self.events[start:] if e["type"] == "response.created")
        if cancel:
            # Cancel once the reply is audible: the prompt has been prefilled,
            # saved and handed to Talker, and the reply is still generating.
            await self.until(lambda: self.of(response_id, "response.output_audio.delta", start))
            await self.send({"type": "response.cancel"})
        await self.until(lambda: self.of(response_id, "response.done", start))
        done = self.of(response_id, "response.done", start)[0]["response"]
        audio = b"".join(
            base64.b64decode(e["delta"]) for e in self.of(response_id, "response.output_audio.delta", start)
        )
        text = "".join(e["delta"] for e in self.of(response_id, "response.output_audio_transcript.delta", start))
        if not cancel:
            # Report the whole reply as heard, so its text joins the history
            # that the next turn's prompt is built from.
            played = done["metadata"]["playback"]["sent_ms"]
            ack_start = len(self.events)
            await self.send(
                {"type": "playback.ack", "response_id": response_id, "played_ms": played, "committed_ms": played}
            )
            await self.until(lambda: any(e["type"] == "playback.acknowledged" for e in self.events[ack_start:]))
        return {"status": done["status"], "text": text.strip(), "audio": audio}


async def _run_session(server: OmniServer, directory: Path, cases: tuple[str, ...], pcm: bytes) -> dict:
    import websockets

    results = {}
    events: list[dict] = []
    url = f"ws://{server.host}:{server.port}/v1/realtime?duplex=1"
    async with websockets.connect(url, max_size=64 * 1024 * 1024) as ws:
        session = Session(ws, events)
        audio_format = {"type": "audio/pcm", "rate": SAMPLE_RATE}
        await session.send(
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "model": server.model,
                    "instructions": INSTRUCTIONS,
                    "temperature": 0,
                    "max_output_tokens": 64,
                    "audio": {
                        "input": {"format": audio_format, "turn_detection": None},
                        "output": {"format": audio_format},
                    },
                },
            }
        )
        await session.until(lambda: any(e["type"] == "session.updated" for e in events))
        try:
            for case in cases:
                (directory / "case.txt").write_text(case)
                failure = directory / FAIL_HS_READ
                if case == "hs-read-failure":
                    failure.touch()
                try:
                    results[case] = await session.turn(pcm, cancel=case == "cancelled")
                finally:
                    failure.unlink(missing_ok=True)
        finally:
            # Release the deployment's single session slot.
            await session.send({"type": "session.close"})
            await session.until(lambda: any(e["type"] == "session.closed" for e in events), timeout_s=30)
            (directory / f"events-{cases[0]}.json").write_text(json.dumps(events, indent=1))
    return results


def _run_arm(model: str, directory: Path, mooncake: dict | None) -> dict:
    directory.mkdir()
    (directory / "case.txt").write_text("startup")
    deploy = _deploy(directory / "deploy.yaml", mooncake=mooncake)
    env = {AUDIT_ENV: str(directory), "VLLM_BATCH_INVARIANT": os.environ.get("VLLM_BATCH_INVARIANT", "1")}
    args = ["--deploy-config", str(deploy), "--init-timeout", "1800", "--stage-init-timeout", "1800"]
    with OmniServer(model, args, env_dict=env) as server:
        results = asyncio.run(_run_session(server, directory, SESSION_1, _clip()))
        results |= asyncio.run(_run_session(server, directory, SESSION_2, _clip(scale=0.5)))
    (directory / "results.json").write_text(
        json.dumps({case: {**r, "audio": len(r["audio"])} for case, r in results.items()}, indent=1)
    )
    return results


def _observations(directory: Path, case: str, kind: str) -> list[dict]:
    return [json.loads(path.read_text()) for path in directory.glob(f"{case}-{kind}-*.json")]


def _handoff(directory: Path, case: str) -> tuple[dict, dict[str, torch.Tensor]]:
    records = _observations(directory, case, "handoff")
    assert len(records) == 1, f"{case}: the prompt must reach Talker exactly once"
    return records[0], load_file(directory / records[0]["tensor_file"])


def _common_prefix(left: list[int], right: list[int]) -> int:
    return next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))


def _expected_restore(prompt: list[int], saved: list[list[int]]) -> int:
    """Rows a joint lookup can reuse: whole blocks of the longest saved prefix.

    The final prompt token is always computed, so a fully cached prompt
    reuses the blocks before it.
    """
    longest = max((_common_prefix(prompt, earlier) for earlier in saved), default=0)
    longest = min(longest, len(prompt) - 1)
    return longest - longest % BLOCK_SIZE


def test_mooncake_duplex_session_turns_and_failures(tmp_path):
    config_path = os.environ.get("MOONCAKE_CONFIG_PATH")
    if not config_path:
        pytest.skip("set MOONCAKE_CONFIG_PATH to run native Mooncake duplex acceptance")
    assert config_path is not None
    assert Path(config_path).is_file(), "MOONCAKE_CONFIG_PATH must name a Store JSON configuration"
    pytest.importorskip("mooncake", reason="Mooncake is required for native acceptance")
    from transformers import AutoTokenizer

    model = os.environ.get("MOONCAKE_TEST_MODEL", MODEL)
    audio_pad = AutoTokenizer.from_pretrained(model, revision=REVISION).convert_tokens_to_ids("<|audio_pad|>")
    mooncake = {
        "config_file": str(Path(config_path).resolve()),
        "namespace": f"m2-duplex-acceptance-{uuid4().hex}",
        "model_revision": REVISION,
    }
    baseline = _run_arm(model, tmp_path / "baseline", None)
    cached = _run_arm(model, tmp_path / "cached", mooncake)

    saved: list[list[int]] = []
    summary = {}
    for case in SESSION_1 + SESSION_2:
        expected, actual = baseline[case], cached[case]
        status = "cancelled" if case == "cancelled" else "completed"
        assert expected["status"] == actual["status"] == status, case
        if status == "completed":
            assert actual["audio"], f"{case}: no output audio"
            assert actual["text"] == expected["text"], f"{case}: reply text differs"
            assert actual["audio"] == expected["audio"], f"{case}: reply audio differs"

        # The prompt the session built, and the Thinker rows handed to Talker.
        reference, before = _handoff(tmp_path / "baseline", case)
        observed, after = _handoff(tmp_path / "cached", case)
        prompt = observed["prompt_token_ids"]
        assert prompt == reference["prompt_token_ids"], f"{case}: sessions built different prompts"
        for key in ("embedding", "hidden", "talker"):
            torch.testing.assert_close(after[key], before[key], rtol=0, atol=0, msg=f"{case}: {key} rows differ")

        captures = _observations(tmp_path / "cached", case, "capture")
        assert captures, f"{case}: production capture integration was not exercised"
        restored = [entry["inserted_rows"] for entry in captures if entry["inserted_rows"] > 0]
        injected = _observations(tmp_path / "cached", case, "injected_failure")
        if case == "hs-read-failure":
            # The prefix was found and its load started, but the failed read
            # made the request compute its whole prompt instead.
            assert injected and not restored, f"{case}: the failed load was not recomputed"
            want = 0
        elif case == "changed-media":
            first_audio = prompt.index(audio_pad)
            # Token IDs repeat session 1, but the audio does not: reuse ends
            # at the last whole block before the first audio token.
            want = first_audio - first_audio % BLOCK_SIZE
            assert _expected_restore(prompt, saved) > want, "the negative case must share token IDs past the audio"
        else:
            assert not injected
            want = _expected_restore(prompt, saved)
        assert sum(restored) == want, f"{case}: restored {restored}, expected {want}"
        assert len(restored) <= 1, f"{case}: restore must be observed at most once"
        if case == "history-hit":
            assert want > 0, "a follow-up turn must reuse the session prefix"
        summary[case] = {"prompt_tokens": len(prompt), "restored_rows": sum(restored), "text": actual["text"]}
        saved.append(prompt)
    (tmp_path / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
