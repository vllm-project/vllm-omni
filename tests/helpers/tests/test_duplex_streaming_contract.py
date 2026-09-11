# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest

from tests.e2e.online_serving import run_minicpmo_realtime_duplex_soft_interrupt as driver
from tests.helpers.runtime import send_duplex_soft_interrupt_request

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("packet_counts,accepted", [([7, 1], True), ([1, 1], False)])
def test_mixed_long_short_response_packet_contract(monkeypatch, tmp_path, packet_counts, accepted):
    async def run(args):
        assert args.min_audio_deltas_per_response == 1
        return {"ok": True, "response_summaries": [{"audio_delta_count": n} for n in packet_counts]}

    monkeypatch.setattr(driver, "run_soft_interrupt", run)
    kwargs = dict(
        url="ws://test",
        model="test",
        input_wav=tmp_path / "in.wav",
        ref_audio=tmp_path / "ref.wav",
        output_dir=tmp_path,
        input_sha256="test",
    )
    if accepted:
        assert send_duplex_soft_interrupt_request(**kwargs)["ok"]
    else:
        with pytest.raises(AssertionError, match="must exercise streaming"):
            send_duplex_soft_interrupt_request(**kwargs)


@pytest.mark.parametrize(
    "native,cancelled,answer,accepted",
    [(True, True, "二", True), (False, True, "二", False), (True, False, "二", False), (True, True, "三", False)],
)
def test_native_interrupt_requires_model_action_cancel_and_correct_followup(
    tmp_path, native, cancelled, answer, accepted
):
    import json

    events: list[dict[str, object]] = [
        {"type": "response.listen"},
        {"type": "response.created", "response": {"id": "a"}},
        {"type": "response.output_audio.delta", "response_id": "a", "delta": "AAAA"},
        {"type": "response.output_audio_transcript.delta", "response_id": "a", "delta": "长回答"},
        {"type": "response.done", "response": {"id": "a", "status": "cancelled" if cancelled else "completed"}},
        {"type": "response.listen", "response": {"metadata": {"reason": "model_interrupt" if native else "vad"}}},
        {"type": "response.created", "response": {"id": "b"}},
        {"type": "response.output_audio.delta", "response_id": "b", "delta": "AAAA"},
        {"type": "response.output_audio_transcript.delta", "response_id": "b", "delta": answer},
        {"type": "response.done", "response": {"id": "b", "status": "completed"}},
        {"type": "response.listen"},
        {"type": "input_audio_buffer.committed"},
        {"type": "response.listen"},
    ]
    for i, event in enumerate(events):
        event["_client_received_at_s"] = float(i)
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (tmp_path / "result.json").write_text(json.dumps({"ok": True, "response_ids": ["a", "b"]}))
    summary = driver.summarize_artifacts(
        output_dir=tmp_path,
        validation_mode="response-required",
        min_responses=2,
        min_audio_deltas_per_response=1,
        expect_followup_response_substring="二",
        require_model_interrupt=True,
    )
    assert summary["ok"] is accepted
    if cancelled:
        legacy = driver.summarize_artifacts(
            output_dir=tmp_path,
            validation_mode="response-required",
            min_responses=2,
            min_audio_deltas_per_response=1,
            expect_followup_response_substring="二",
        )
        assert not legacy["ok"]
