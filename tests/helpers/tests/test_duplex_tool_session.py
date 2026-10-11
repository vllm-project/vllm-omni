# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tool E2E probes must verify result audio and preserve pending-call identity."""

import asyncio
import base64
from types import SimpleNamespace

import pytest

from tests.helpers.runtime import send_duplex_tool_context_request
from vllm_omni.clients import duplex

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def tool_peer(monkeypatch):
    state = SimpleNamespace(
        result_audio=True,
        interrupt_epoch=40,
        reader_error=False,
        early_ack=False,
        client=None,
        followup_text="一加一等于二",
    )

    class Client:
        session_info = {"epoch": 40}

        def __init__(self, *args, **kwargs):
            self.queue: asyncio.Queue[SimpleNamespace] = asyncio.Queue()
            self.stream_count = 0
            self.sent = []
            state.client = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            # The probe must settle both its reader and input task before exit.
            assert len(asyncio.all_tasks()) == 1

        def emit(self, event):
            self.queue.put_nowait(SimpleNamespace(raw=event))

        def response(self, response_id, *, audio=True, text="", status="completed"):
            self.emit({"type": "response.created", "response": {"id": response_id}})
            if audio:
                self.emit(
                    {
                        "type": "response.output_audio.delta",
                        "response_id": response_id,
                        "delta": base64.b64encode(b"\x10\x00" * 240).decode(),
                    }
                )
            if text:
                self.emit({"type": "response.output_audio_transcript.delta", "response_id": response_id, "delta": text})
            self.emit({"type": "response.output_item.done", "response_id": response_id, "item": {"type": "message"}})
            self.emit({"type": "response.done", "response": {"id": response_id, "status": status}})

        async def events(self):
            if state.reader_error:
                raise RuntimeError("injected reader failure")
            while True:
                yield await self.queue.get()

        async def stream_pcm(self, *args, **kwargs):
            self.stream_count += 1
            if self.stream_count == 1:
                self.response("before-tool", text="Working on it")
                self.emit(
                    {
                        "type": "response.output_item.done",
                        "item": {"type": "function_call", "name": "lookup", "call_id": "call-1", "arguments": "{}"},
                    }
                )
            elif self.stream_count == 2 and len(args[0]) > 32000 * 15:
                self.response("interrupted", status="cancelled")
                self.emit({"type": "output_audio_buffer.cleared", "response_id": "interrupted"})
                self.emit({"type": "response.listen", "response": {"metadata": {"reason": "model_interrupt"}}})
                if state.early_ack:
                    self.response("ack", text="Tell me the new question")
                    await asyncio.sleep(0.1)
                self.response("followup", text=state.followup_text)
            await asyncio.sleep(0)

        async def ack_playback(self, *args, **kwargs):
            pass

        async def send(self, event):
            self.sent.append(event)
            if event["type"] == "input.context.get":
                self.emit(
                    {
                        "type": "duplex.input.context.snapshot",
                        "event": {"type": "input.context.snapshot", "epoch": state.interrupt_epoch},
                    }
                )
            elif event["type"] == "conversation.item.create":
                self.emit({"type": "conversation.item.created", "item": event["item"]})
                self.response("after-tool", audio=state.result_audio, text="Result: 472")
            await asyncio.sleep(0)

    monkeypatch.setattr(duplex, "DuplexClient", Client)
    monkeypatch.setattr(duplex, "read_pcm16_wav", lambda path: bytes(32000))
    return state


def probe(tmp_path, **options):
    return send_duplex_tool_context_request(
        url="ws://example",
        model="model",
        session_config=None,
        input_wav=tmp_path / "question.wav",
        output_dir=tmp_path,
        expected_tool="lookup",
        tool_output={"answer": "472"},
        expected_text="472",
        timeout_s=0.5,
        **options,
    )


def test_tool_reply_requires_audio_after_result(tool_peer, tmp_path):
    tool_peer.result_audio = False
    with pytest.raises(AssertionError, match="No audio in the reply after the tool result"):
        probe(tmp_path)
    assert (tmp_path / "events.json").is_file()


def test_tool_reply_accepts_audio_after_result(tool_peer, tmp_path):
    assert probe(tmp_path)["transcript"] == "Result: 472"


@pytest.mark.parametrize("answer", ["一加一等于二", "答案是2。"])
def test_pending_tool_native_interrupt_preserves_epoch(tool_peer, tmp_path, answer):
    tool_peer.followup_text = answer
    result = probe(
        tmp_path,
        pending_interrupt_wav=tmp_path / "interrupt.wav",
        expected_interrupt_text_pattern=r"(?:一加一(?:等于|是)(?:二|2)|(?:答案|结果|結果)是(?:二|2))(?=[。.!！\s]|$)",
    )
    assert result["interrupted_response"] == "interrupted"
    assert not any(event["type"] in {"response.cancel", "output_audio_buffer.clear"} for event in tool_peer.client.sent)
    assert tool_peer.client.sent[-1]["item"]["call_id"] == "call-1"


def test_pending_tool_native_interrupt_rejects_epoch_change(tool_peer, tmp_path):
    tool_peer.interrupt_epoch = 41
    with pytest.raises(AssertionError, match="Native interrupt invalidated the pending tool epoch"):
        probe(
            tmp_path,
            pending_interrupt_wav=tmp_path / "interrupt.wav",
            expected_interrupt_text_pattern=r"(?:一加一(?:等于|是)(?:二|2)|(?:答案|结果|結果)是(?:二|2))(?=[。.!！\s]|$)",
        )
    assert not any(event["type"] == "conversation.item.create" for event in tool_peer.client.sent)


def test_pending_tool_interrupt_keeps_input_after_early_ack(tool_peer, tmp_path):
    tool_peer.early_ack = True
    result = probe(
        tmp_path,
        pending_interrupt_wav=tmp_path / "interrupt.wav",
        expected_interrupt_text_pattern=r"(?:一加一(?:等于|是)(?:二|2)|(?:答案|结果|結果)是(?:二|2))(?=[。.!！\s]|$)",
    )
    assert result["interrupted_response"] == "interrupted"
    assert tool_peer.client.sent[-1]["item"]["call_id"] == "call-1"


def test_tool_probe_observes_reader_failure(tool_peer, tmp_path):
    tool_peer.reader_error = True
    with pytest.raises(RuntimeError, match="injected reader failure"):
        probe(tmp_path)


@pytest.mark.parametrize("answer", ["答案是12。", "一加一等于三"])
def test_pending_tool_native_interrupt_rejects_wrong_answer(tool_peer, tmp_path, answer):
    tool_peer.followup_text = answer
    with pytest.raises(AssertionError, match="Timed out awaiting tool/context event"):
        probe(
            tmp_path,
            pending_interrupt_wav=tmp_path / "interrupt.wav",
            expected_interrupt_text_pattern=r"(?:一加一(?:等于|是)(?:二|2)|(?:答案|结果|結果)是(?:二|2))(?=[。.!！\s]|$)",
        )
    assert not any(event["type"] == "conversation.item.create" for event in tool_peer.client.sent)
