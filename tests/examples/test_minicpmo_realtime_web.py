import asyncio
import base64
import importlib.util
import subprocess
import sys
import wave
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

DEMO_PATH = Path(__file__).resolve().parents[1] / "e2e/online_serving/minicpmo_realtime_duplex_scenarios.py"
MULTI_DEMO_PATH = (
    Path(__file__).resolve().parents[1] / "e2e/online_serving/run_minicpmo_realtime_duplex_multi_session.py"
)
PAIR_DEMO_PATH = Path(__file__).resolve().parents[1] / "e2e/online_serving/run_minicpmo_realtime_duplex_demo_pair.py"
SOFT_INTERRUPT_DEMO_PATH = (
    Path(__file__).resolve().parents[1] / "e2e/online_serving/run_minicpmo_realtime_duplex_soft_interrupt.py"
)


def _load_demo_module():
    spec = importlib.util.spec_from_file_location("minicpmo_realtime_duplex_demo_test", DEMO_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_multi_demo_module():
    spec = importlib.util.spec_from_file_location("minicpmo_realtime_duplex_multi_session_test", MULTI_DEMO_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_pair_demo_module():
    spec = importlib.util.spec_from_file_location("minicpmo_realtime_duplex_demo_pair_test", PAIR_DEMO_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_soft_interrupt_demo_module():
    spec = importlib.util.spec_from_file_location(
        "minicpmo_realtime_duplex_soft_interrupt_test",
        SOFT_INTERRUPT_DEMO_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_realtime_duplex_multi_session_script_is_directly_executable():
    result = subprocess.run(
        [sys.executable, str(MULTI_DEMO_PATH), "--help"],
        cwd=MULTI_DEMO_PATH.parents[3],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_realtime_duplex_demo_pair_script_is_directly_executable():
    result = subprocess.run(
        [sys.executable, str(PAIR_DEMO_PATH), "--help"],
        cwd=PAIR_DEMO_PATH.parents[3],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_realtime_duplex_soft_interrupt_script_is_directly_executable():
    result = subprocess.run(
        [sys.executable, str(SOFT_INTERRUPT_DEMO_PATH), "--help"],
        cwd=SOFT_INTERRUPT_DEMO_PATH.parents[3],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_realtime_duplex_demo_pair_requires_distinct_inputs_and_outputs(tmp_path):
    demo = _load_pair_demo_module()
    wav_a = tmp_path / "request_a.wav"
    wav_b = tmp_path / "request_b.wav"
    wav_a.write_bytes(b"same")
    wav_b.write_bytes(b"same")

    args = SimpleNamespace(
        input_wav_a=str(wav_a),
        input_wav_b=str(wav_b),
        output_dir_a=str(tmp_path / "session_a"),
        output_dir_b=str(tmp_path / "session_a"),
    )

    with pytest.raises(ValueError, match="output directories must be different"):
        demo._validate_pair_args(args)

    args.output_dir_b = str(tmp_path / "session_b")
    with pytest.raises(ValueError, match="input WAV files must have different content"):
        demo._validate_pair_args(args)


def test_realtime_duplex_demo_pair_launches_demo_processes_concurrently(tmp_path, monkeypatch):
    demo = _load_pair_demo_module()
    fake_demo = tmp_path / "fake_realtime_duplex_demo.py"
    start_log = tmp_path / "starts.log"
    fake_demo.write_text(
        "\n".join(
            [
                "import argparse, json, os, time",
                "from pathlib import Path",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--url')",
                "parser.add_argument('--model')",
                "parser.add_argument('--input-wav')",
                "parser.add_argument('--output-dir')",
                "parser.add_argument('--chunk-ms')",
                "parser.add_argument('--timeout-s')",
                "parser.add_argument('--session-id')",
                "parser.add_argument('--ref-audio')",
                "parser.add_argument('--require-audio', action='store_true')",
                "parser.add_argument('--no-realtime-pacing', action='store_true')",
                "args = parser.parse_args()",
                "if args.ref_audio != os.environ.get('EXPECTED_REF_AUDIO'):",
                "    raise SystemExit(f'unexpected ref audio: {args.ref_audio}')",
                "output = Path(args.output_dir)",
                "output.mkdir(parents=True, exist_ok=True)",
                "response_id = 'resp-' + output.name",
                "with open(os.environ['PAIR_START_LOG'], 'a', encoding='utf-8') as f:",
                "    f.write(f'{output.name} {time.time()}\\n')",
                "time.sleep(0.25)",
                "events = [",
                "    {'type': 'session.created'},",
                "    {'type': 'response.created', 'response': {'id': response_id}},",
                "    {'type': 'response.audio.delta', 'response_id': response_id, "
                "'delta': 'AAAA', 'sample_rate_hz': 24000, '_client_received_at_s': 10.1},",
                "    {'type': 'response.audio.delta', 'response_id': response_id, "
                "'delta': 'AAAA', 'sample_rate_hz': 24000, '_client_received_at_s': 10.6},",
                "    {'type': 'response.done', 'response_id': response_id},",
                "    {'type': 'session.closed'},",
                "]",
                "Path(output / 'events.jsonl').write_text(",
                "    ''.join(json.dumps(event) + '\\n' for event in events),",
                "    encoding='utf-8',",
                ")",
                "result = {'ok': True, 'response_ids': [response_id], 'audio_bytes': 1, 'output_dir': str(output)}",
                "Path(output / 'result.json').write_text(json.dumps(result), encoding='utf-8')",
                "print(json.dumps(result))",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(demo, "DEMO_PATH", fake_demo)
    monkeypatch.setenv("PAIR_START_LOG", str(start_log))
    monkeypatch.setenv("EXPECTED_REF_AUDIO", str(tmp_path / "ref.wav"))
    wav_a = tmp_path / "request_a.wav"
    wav_b = tmp_path / "request_b.wav"
    wav_a.write_bytes(b"request a")
    wav_b.write_bytes(b"request b")

    result = asyncio.run(
        demo.run_pair(
            SimpleNamespace(
                url="ws://localhost:28889/v1/realtime?duplex=1",
                model="/data/why/MiniCPM-o-4_5",
                input_wav_a=str(wav_a),
                input_wav_b=str(wav_b),
                output_dir_a=str(tmp_path / "duplex_session_a"),
                output_dir_b=str(tmp_path / "duplex_session_b"),
                ref_audio=str(tmp_path / "ref.wav"),
                summary_output=str(tmp_path / "summary.json"),
                chunk_ms=200,
                timeout_s=5.0,
                no_realtime_pacing=False,
                require_audio=True,
                min_audio_deltas_per_session=2,
            )
        )
    )

    starts = [float(line.split()[1]) for line in start_log.read_text(encoding="utf-8").splitlines()]
    assert len(starts) == 2
    assert max(starts) - min(starts) < 0.2
    assert result["ok"] is True
    assert result["input_wavs_distinct"] is True
    assert result["output_dirs_distinct"] is True
    assert result["identity_isolation_ok"] is True
    assert result["response_audio_contract_ok"] is True
    assert {session["audio_delta_count"] for session in result["sessions"]} == {2}
    for session in result["sessions"]:
        assert session["all_audio_deltas_same_response"] is True
        assert session["response_id_nonempty"] is True
        assert session["first_audio_delta_before_done"] is True
        assert len(session["audio_delta_timings"]) == 2


def test_realtime_duplex_demo_pair_rejects_false_green_audio_contract(tmp_path):
    demo = _load_pair_demo_module()
    output = tmp_path / "session"
    output.mkdir()
    response_id = "resp-a"
    events = [
        {"type": "response.created", "response": {"id": response_id}},
        {"type": "response.audio.delta", "response_id": response_id, "delta": "AAAA", "sample_rate_hz": 24000},
        {"type": "response.audio.delta", "response_id": "resp-other", "delta": "AAAA", "sample_rate_hz": 24000},
        {"type": "response.done", "response_id": response_id},
    ]
    (output / "events.jsonl").write_text(
        "".join(demo.json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    (output / "result.json").write_text(
        demo.json.dumps({"ok": True, "response_ids": [response_id], "audio_bytes": 4}),
        encoding="utf-8",
    )

    session = demo._summarize_session(
        label="a",
        input_wav="request.wav",
        output_dir=str(output),
        returncode=0,
        stdout="",
        stderr="",
    )

    assert session["ok"] is False
    assert session["all_audio_deltas_same_response"] is False
    assert session["response_audio_contract_ok"] is False


def test_realtime_duplex_demo_pair_default_requires_multiple_audio_deltas(monkeypatch):
    demo = _load_pair_demo_module()
    monkeypatch.setattr(
        demo.sys,
        "argv",
        [
            "pair.py",
            "--input-wav-a",
            "a.wav",
            "--input-wav-b",
            "b.wav",
            "--output-dir-a",
            "a",
            "--output-dir-b",
            "b",
            "--ref-audio",
            "ref.wav",
        ],
    )

    args = demo.parse_args()

    assert args.ref_audio == "ref.wav"
    assert args.min_audio_deltas_per_session == 2


def test_realtime_duplex_demo_pair_requires_ref_audio(monkeypatch):
    demo = _load_pair_demo_module()
    monkeypatch.setattr(
        demo.sys,
        "argv",
        [
            "pair.py",
            "--input-wav-a",
            "a.wav",
            "--input-wav-b",
            "b.wav",
            "--output-dir-a",
            "a",
            "--output-dir-b",
            "b",
        ],
    )

    with pytest.raises(SystemExit):
        demo.parse_args()


def test_realtime_duplex_soft_interrupt_requires_ref_audio(monkeypatch):
    demo = _load_soft_interrupt_demo_module()
    monkeypatch.setattr(
        demo.sys,
        "argv",
        [
            "soft.py",
            "--input-wav",
            "input.wav",
            "--output-dir",
            "out",
        ],
    )

    with pytest.raises(SystemExit):
        demo.parse_args()


def test_realtime_duplex_soft_interrupt_accepts_explicit_ref_audio(monkeypatch):
    demo = _load_soft_interrupt_demo_module()
    monkeypatch.setattr(
        demo.sys,
        "argv",
        [
            "soft.py",
            "--input-wav",
            "input.wav",
            "--output-dir",
            "out",
            "--ref-audio",
            "ref.wav",
        ],
    )

    args = demo.parse_args()

    assert args.ref_audio == "ref.wav"
    assert args.validation_mode == "model-policy"


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--expect-second-response-substring", "一加一等于二"],
        ["--input-sha256", "a" * 64],
    ],
)
def test_realtime_duplex_soft_interrupt_response_required_needs_bound_fixture(monkeypatch, extra_args):
    demo = _load_soft_interrupt_demo_module()
    monkeypatch.setattr(
        demo.sys,
        "argv",
        [
            "soft.py",
            "--input-wav",
            "input.wav",
            "--output-dir",
            "out",
            "--ref-audio",
            "ref.wav",
            "--validation-mode",
            "response-required",
            *extra_args,
        ],
    )

    with pytest.raises(SystemExit):
        demo.parse_args()


def test_realtime_duplex_soft_interrupt_rejects_input_checksum_mismatch(tmp_path):
    demo = _load_soft_interrupt_demo_module()
    input_wav = tmp_path / "input.wav"
    input_wav.write_bytes(b"two-turn fixture")

    with pytest.raises(ValueError, match="input WAV SHA256 mismatch"):
        demo._validate_input_sha256(input_wav, "0" * 64)


def test_realtime_duplex_soft_interrupt_accepts_multi_delta_handoff_sequence(tmp_path):
    demo = _load_soft_interrupt_demo_module()
    output = tmp_path / "soft_interrupt"
    output.mkdir()
    first_response_id = "resp-first"
    second_response_id = "resp-second"
    events = [
        {"type": "response.listen", "_client_received_at_s": 1.0},
        {"type": "response.created", "response": {"id": first_response_id}, "_client_received_at_s": 2.0},
        {"type": "response.speak", "response_id": first_response_id, "_client_received_at_s": 2.0},
        {
            "type": "response.audio.delta",
            "response_id": first_response_id,
            "delta": "AAAA",
            "_client_received_at_s": 2.1,
        },
        {
            "type": "response.audio_transcript.delta",
            "response_id": first_response_id,
            "delta": "中国古代四大发明",
            "_client_received_at_s": 2.1,
        },
        {
            "type": "response.audio.delta",
            "response_id": first_response_id,
            "delta": "AAAA",
            "_client_received_at_s": 2.6,
        },
        {
            "type": "response.audio_transcript.delta",
            "response_id": first_response_id,
            "delta": "是造纸术。",
            "_client_received_at_s": 2.6,
        },
        {"type": "response.done", "response_id": first_response_id, "_client_received_at_s": 3.0},
        {"type": "response.listen", "_client_received_at_s": 3.2},
        {"type": "response.created", "response": {"id": second_response_id}, "_client_received_at_s": 4.0},
        {"type": "response.speak", "response_id": second_response_id, "_client_received_at_s": 4.0},
        {
            "type": "response.audio.delta",
            "response_id": second_response_id,
            "delta": "AAAA",
            "_client_received_at_s": 4.1,
        },
        {
            "type": "response.audio_transcript.delta",
            "response_id": second_response_id,
            "delta": "一加一等于",
            "_client_received_at_s": 4.1,
        },
        {
            "type": "response.audio.delta",
            "response_id": second_response_id,
            "delta": "AAAA",
            "_client_received_at_s": 4.6,
        },
        {
            "type": "response.audio_transcript.delta",
            "response_id": second_response_id,
            "delta": "二。",
            "_client_received_at_s": 4.6,
        },
        {"type": "response.done", "response_id": second_response_id, "_client_received_at_s": 5.0},
        {"type": "response.listen", "_client_received_at_s": 5.2},
        {"type": "input_audio_buffer.committed", "_client_received_at_s": 6.0},
        {"type": "response.listen", "_client_received_at_s": 6.1},
    ]
    (output / "events.jsonl").write_text(
        "".join(demo.json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    (output / "result.json").write_text(
        demo.json.dumps({"ok": True, "response_ids": [first_response_id, second_response_id]}),
        encoding="utf-8",
    )

    summary = demo.summarize_artifacts(
        output_dir=output,
        validation_mode="response-required",
        min_responses=2,
        min_audio_deltas_per_response=2,
        expect_second_response_substring="一加一等于二",
    )

    assert summary["ok"] is True
    assert summary["second_response_before_final_commit"] is True
    assert summary["final_listen_after_commit"] is True
    assert summary["response_audio_contract_ok"] is True
    assert summary["response_summaries"][0]["audio_delta_count"] == 2
    assert summary["response_summaries"][1]["transcript"] == "一加一等于二。"


def test_realtime_duplex_soft_interrupt_model_policy_accepts_single_response(tmp_path):
    demo = _load_soft_interrupt_demo_module()
    output = tmp_path / "model_policy"
    output.mkdir()
    response_id = "resp-only"
    events = [
        {"type": "response.listen", "_client_received_at_s": 1.0},
        {"type": "response.created", "response": {"id": response_id}, "_client_received_at_s": 2.0},
        {"type": "response.audio.delta", "response_id": response_id, "delta": "AAAA", "_client_received_at_s": 2.1},
        {"type": "response.audio.delta", "response_id": response_id, "delta": "AAAA", "_client_received_at_s": 2.15},
        {"type": "response.done", "response_id": response_id, "_client_received_at_s": 2.2},
        {"type": "response.listen", "_client_received_at_s": 2.3},
        {"type": "input_audio_buffer.committed", "_client_received_at_s": 3.0},
        {"type": "response.listen", "_client_received_at_s": 3.1},
    ]
    (output / "events.jsonl").write_text(
        "".join(demo.json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    (output / "result.json").write_text(
        demo.json.dumps({"ok": True, "response_ids": [response_id]}),
        encoding="utf-8",
    )

    summary = demo.summarize_artifacts(
        output_dir=output,
        validation_mode="model-policy",
        min_responses=2,
        min_audio_deltas_per_response=2,
        expect_second_response_substring=None,
    )

    assert summary["ok"] is True
    assert summary["validation_mode"] == "model-policy"
    assert summary["enough_responses"] is True
    assert summary["response_before_final_commit"] is True
    assert summary["listen_after_response_before_commit"] is True


def test_realtime_duplex_soft_interrupt_response_required_rejects_single_response(tmp_path):
    demo = _load_soft_interrupt_demo_module()
    output = tmp_path / "response_required"
    output.mkdir()
    response_id = "resp-only"
    events = [
        {"type": "response.listen", "_client_received_at_s": 1.0},
        {"type": "response.created", "response": {"id": response_id}, "_client_received_at_s": 2.0},
        {"type": "response.audio.delta", "response_id": response_id, "delta": "AAAA", "_client_received_at_s": 2.1},
        {"type": "response.audio.delta", "response_id": response_id, "delta": "AAAA", "_client_received_at_s": 2.15},
        {"type": "response.done", "response_id": response_id, "_client_received_at_s": 2.2},
        {"type": "response.listen", "_client_received_at_s": 2.3},
        {"type": "input_audio_buffer.committed", "_client_received_at_s": 3.0},
        {"type": "response.listen", "_client_received_at_s": 3.1},
    ]
    (output / "events.jsonl").write_text(
        "".join(demo.json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    (output / "result.json").write_text(
        demo.json.dumps({"ok": True, "response_ids": [response_id]}),
        encoding="utf-8",
    )

    summary = demo.summarize_artifacts(
        output_dir=output,
        validation_mode="response-required",
        min_responses=2,
        min_audio_deltas_per_response=2,
        expect_second_response_substring="一加一等于二",
    )

    assert summary["ok"] is False
    assert summary["enough_responses"] is False


def test_realtime_duplex_soft_interrupt_checks_expected_text_in_second_response(tmp_path):
    demo = _load_soft_interrupt_demo_module()
    output = tmp_path / "second_response_text"
    output.mkdir()
    first_response_id = "resp-first"
    second_response_id = "resp-second"
    events = [
        {"type": "response.listen", "_client_received_at_s": 1.0},
        {"type": "response.created", "response": {"id": first_response_id}, "_client_received_at_s": 2.0},
        {
            "type": "response.audio.delta",
            "response_id": first_response_id,
            "delta": "AAAA",
            "_client_received_at_s": 2.1,
        },
        {
            "type": "response.audio.delta",
            "response_id": first_response_id,
            "delta": "AAAA",
            "_client_received_at_s": 2.2,
        },
        {
            "type": "response.audio_transcript.delta",
            "response_id": first_response_id,
            "delta": "一加一等于二",
            "_client_received_at_s": 2.2,
        },
        {"type": "response.done", "response_id": first_response_id, "_client_received_at_s": 2.3},
        {"type": "response.listen", "_client_received_at_s": 2.4},
        {"type": "response.created", "response": {"id": second_response_id}, "_client_received_at_s": 3.0},
        {
            "type": "response.audio.delta",
            "response_id": second_response_id,
            "delta": "AAAA",
            "_client_received_at_s": 3.1,
        },
        {
            "type": "response.audio.delta",
            "response_id": second_response_id,
            "delta": "AAAA",
            "_client_received_at_s": 3.2,
        },
        {
            "type": "response.audio_transcript.delta",
            "response_id": second_response_id,
            "delta": "不知道",
            "_client_received_at_s": 3.2,
        },
        {"type": "response.done", "response_id": second_response_id, "_client_received_at_s": 3.3},
        {"type": "response.listen", "_client_received_at_s": 3.4},
        {"type": "input_audio_buffer.committed", "_client_received_at_s": 4.0},
        {"type": "response.listen", "_client_received_at_s": 4.1},
    ]
    (output / "events.jsonl").write_text(
        "".join(demo.json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    (output / "result.json").write_text(
        demo.json.dumps({"ok": True, "response_ids": [first_response_id, second_response_id]}),
        encoding="utf-8",
    )

    summary = demo.summarize_artifacts(
        output_dir=output,
        validation_mode="response-required",
        min_responses=2,
        min_audio_deltas_per_response=2,
        expect_second_response_substring="一加一等于二",
    )

    assert summary["ok"] is False
    assert summary["second_response_transcript_expectation_ok"] is False


def test_realtime_duplex_multi_session_resume_url_disables_autostart():
    demo = _load_multi_demo_module()

    url = demo._with_resume_mode("ws://localhost:8113/v1/realtime?duplex=1&model=openbmb%2FMiniCPM-o-4_5")

    query = parse_qs(urlsplit(url).query)
    assert query["model"] == ["openbmb/MiniCPM-o-4_5"]
    assert query["resume"] == ["1"]


def test_realtime_duplex_multi_session_rejects_cross_session_response_identity():
    demo = _load_multi_demo_module()

    assert demo._validate_identity_isolation(
        [
            {"completed_response_ids": ["resp-a"]},
            {"completed_response_ids": ["resp-b"]},
        ]
    )
    assert not demo._validate_identity_isolation(
        [
            {"completed_response_ids": ["resp-shared"]},
            {"completed_response_ids": ["resp-shared"]},
        ]
    )


def test_realtime_duplex_multi_session_reads_nested_terminal_identity():
    demo = _load_multi_demo_module()

    assert demo._event_session_id({"type": "session.heartbeat_ack", "session_id": "sid"}) == "sid"
    assert (
        demo._event_session_id({"type": "session.closed", "event": {"type": "session.closed", "session_id": "sid"}})
        == "sid"
    )


def test_realtime_duplex_demo_resolves_distinct_turn_inputs():
    demo = _load_demo_module()
    primary = Path("first.wav")

    assert demo._turn_input_paths(primary, [], turns=3) == [primary, primary, primary]
    assert demo._turn_input_paths(primary, ["second.wav", "third.wav"], turns=3) == [
        primary,
        Path("second.wav"),
        Path("third.wav"),
    ]
    with pytest.raises(ValueError, match="one --turn-input-wav"):
        demo._turn_input_paths(primary, ["second.wav"], turns=3)


def test_realtime_duplex_demo_explicitly_enables_native_runtime_before_connect():
    demo = _load_demo_module()

    url = demo._url_with_model(
        "ws://localhost:8099/v1/realtime?duplex=1",
        "openbmb/MiniCPM-o-4_5",
    )

    query = parse_qs(urlsplit(url).query)
    assert query["minicpmo45_native_duplex"] == ["1"]


def test_realtime_duplex_demo_explicit_session_id_reaches_autostart_query():
    demo = _load_demo_module()

    url = demo._url_with_model(
        "ws://localhost:8099/v1/realtime?duplex=1",
        "openbmb/MiniCPM-o-4_5",
        session_id="reopen-e2e",
    )

    query = parse_qs(urlsplit(url).query)
    assert query["session_id"] == ["reopen-e2e"]


def test_realtime_duplex_demo_session_update_uses_explicit_session_id():
    demo = _load_demo_module()

    event = demo._session_update_event(
        SimpleNamespace(
            model="openbmb/MiniCPM-o-4_5",
            output_audio_format="pcm16",
            short_ack_ms=1200,
            session_id="reopen-e2e",
        )
    )

    assert event["type"] == "session.update"
    assert "session_id" not in event
    assert event["session"]["session_id"] == "reopen-e2e"
    assert event["session"]["extra_body"]["minicpmo45_native_duplex"] is True


def test_realtime_duplex_demo_response_required_uses_deterministic_sampling():
    demo = _load_demo_module()

    event = demo._session_update_event(
        SimpleNamespace(
            model="openbmb/MiniCPM-o-4_5",
            output_audio_format="pcm16",
            short_ack_ms=1200,
            session_id=None,
            validation_mode="response-required",
            temperature=None,
        )
    )

    assert event["session"]["temperature"] == 0.0


def test_realtime_duplex_demo_model_policy_preserves_default_sampling():
    demo = _load_demo_module()

    event = demo._session_update_event(
        SimpleNamespace(
            model="openbmb/MiniCPM-o-4_5",
            output_audio_format="pcm16",
            short_ack_ms=1200,
            session_id=None,
            validation_mode="model-policy",
            temperature=None,
        )
    )

    assert "temperature" not in event["session"]


def test_realtime_duplex_demo_resolves_explicit_turn_durations():
    demo = _load_demo_module()

    assert demo._turn_durations([], turns=3, first_turn_ms=3000) == [3000, 1200, 1200]
    assert demo._turn_durations([0, 900, 1500], turns=3, first_turn_ms=3000) == [None, 900, 1500]
    with pytest.raises(ValueError, match="one --turn-duration-ms"):
        demo._turn_durations([0, 900], turns=3, first_turn_ms=3000)
    with pytest.raises(ValueError, match="non-negative"):
        demo._turn_durations([0, -1, 900], turns=3, first_turn_ms=3000)


def test_realtime_duplex_demo_resolves_transcript_labels_for_requested_turns():
    demo = _load_demo_module()

    assert demo._turn_transcripts("first", turns=1) == ["first"]
    assert demo._turn_transcripts("first", turns=4) == ["first", "继续", "再说一次", "turn-4"]


def test_realtime_duplex_demo_reads_response_playback_cursor():
    demo = _load_demo_module()
    state = demo.DemoState()
    state.add(
        {
            "type": "response.done",
            "response_id": "resp-1",
            "response": {
                "id": "resp-1",
                "metadata": {
                    "playback": {
                        "sent_ms": 27920,
                        "played_ms": 0,
                    }
                },
            },
        }
    )

    assert state.response_playback_sent_ms("resp-1") == 27920


def test_realtime_duplex_demo_partitions_timing_by_response_identity():
    demo = _load_demo_module()
    state = demo.DemoState()
    for response_id, created_at_s, audio_at_s, token_count in (
        ("resp-1", 10.0, 10.1, 3),
        ("resp-2", 20.0, 20.2, 5),
    ):
        state.add(
            {"type": "response.created", "response": {"id": response_id}},
            received_at_s=created_at_s,
        )
        state.add(
            {
                "type": "response.audio.delta",
                "response_id": response_id,
                "delta": base64.b64encode(b"audio").decode("ascii"),
                "metadata": {
                    "audio_duration_ms": 80,
                    "vllm_omni": {
                        "stage_metrics": {
                            "0": {
                                "num_tokens_out": token_count,
                                "vllm_itls_ms": [10.0],
                            }
                        }
                    },
                },
            },
            received_at_s=audio_at_s,
        )

    timings = state.response_timing_summaries()

    assert timings["resp-1"]["stage0_tokens"]["output_token_count"] == 3
    assert timings["resp-1"]["audio_output"]["response_created_to_first_audio_ms"] == 100.0
    assert timings["resp-1"]["audio_output"]["commit_to_first_audio_ms"] is None
    assert timings["resp-2"]["stage0_tokens"]["output_token_count"] == 5
    assert timings["resp-2"]["audio_output"]["response_created_to_first_audio_ms"] == 200.0
    assert timings["resp-2"]["audio_output"]["commit_to_first_audio_ms"] is None


def test_realtime_duplex_demo_model_policy_accepts_one_listen_per_streamed_turn():
    demo = _load_demo_module()
    state = demo.DemoState()
    state.add({"type": "session.created"})
    for turn in range(3):
        state.add({"type": "input_audio_buffer.speech_started", "turn": turn})
        state.add(
            {
                "type": "response.listen",
                "response": {
                    "status": "listening",
                    "metadata": {"model_listen": True},
                },
            }
        )
        state.add({"type": "input_audio_buffer.committed", "turn": turn})

    assert state.model_policy_event_order_ok(expected_turns=3)


def test_realtime_duplex_demo_model_policy_accepts_continuous_input_without_commits():
    demo = _load_demo_module()
    state = demo.DemoState()
    state.add({"type": "session.created"})
    state.add({"type": "input_audio_buffer.speech_started"})
    for _ in range(3):
        state.add(
            {
                "type": "response.listen",
                "response": {"metadata": {"model_listen": True}},
            }
        )

    assert not state.model_policy_event_order_ok(expected_turns=3)
    assert state.model_policy_event_order_ok(expected_turns=3, require_input_commit=False)


def test_realtime_duplex_demo_accepts_overlap_unit_terminating_continuous_response():
    demo = _load_demo_module()
    state = demo.DemoState()
    state.add({"type": "response.created", "response_id": "resp-1"})
    state.add({"type": "response.done", "response_id": "resp-1"})

    assert demo._continuous_overlap_terminal_is_outcome(
        state,
        validation_mode="model-policy",
        model_unit_ready_while_active=True,
        before_created=1,
        before_model_listen=0,
    )
    assert not demo._continuous_overlap_terminal_is_outcome(
        state,
        validation_mode="response-required",
        model_unit_ready_while_active=True,
        before_created=1,
        before_model_listen=0,
    )


def test_realtime_duplex_demo_full_turn_duration_does_not_slice_audio():
    demo = _load_demo_module()
    pcm16 = b"\x01\x00" * (demo.PCM16_SAMPLE_RATE * 2)

    assert demo._select_turn_audio(pcm16, None) == pcm16
    assert len(demo._select_turn_audio(pcm16, 1000)) == demo.PCM16_SAMPLE_RATE * demo.PCM16_BYTES_PER_SAMPLE


def test_realtime_duplex_demo_distinct_inputs_compare_audio_content():
    demo = _load_demo_module()
    paths = [Path("first.wav"), Path("second.wav"), Path("third.wav")]

    assert demo._turn_inputs_are_distinct(paths, [b"first", b"second", b"third"]) is True
    assert demo._turn_inputs_are_distinct(paths, [b"same", b"same", b"third"]) is False
    assert demo._turn_inputs_are_distinct([paths[0], paths[0]], [b"first", b"second"]) is False


def _add_response_transcript(state, response_id, *, transcript, audio=True):
    state.add(
        {
            "type": "response.audio.delta",
            "response_id": response_id,
            "delta": "YQ==" if audio else "",
        }
    )
    if transcript:
        state.add(
            {
                "type": "response.audio_transcript.delta",
                "response_id": response_id,
                "delta": transcript,
            }
        )
        state.add(
            {
                "type": "response.audio_transcript.done",
                "response_id": response_id,
                "transcript": transcript,
            }
        )
    state.add(
        {
            "type": "response.done",
            "response_id": response_id,
            "response": {"id": response_id, "status": "completed"},
        }
    )


def test_realtime_duplex_demo_gate_checks_delta_done_and_turn_independence():
    demo = _load_demo_module()
    state = demo.DemoState()
    _add_response_transcript(
        state,
        "resp-1",
        transcript="第一轮回答",
    )
    _add_response_transcript(
        state,
        "resp-2",
        transcript="第二轮回答",
    )

    result = demo._evaluate_transcript_integrity(
        state,
        ["resp-1", "resp-2"],
        expected_empty_response_ids=set(),
        require_cross_turn_independence=True,
    )

    assert result["transcript_delta_done_ok"] is True
    assert result["cross_turn_independent_ok"] is True


def test_realtime_duplex_demo_speak_gate_accepts_one_decision_event_per_response():
    demo = _load_demo_module()
    state = demo.DemoState()
    state.add({"type": "response.created", "response": {"id": "resp-1"}})
    state.add(
        {
            "type": "response.speak",
            "response_id": "resp-1",
            "metadata": {"model_speak": True},
        }
    )

    assert demo._evaluate_response_speak_contract(state)["response_speak_contract_ok"] is True


def test_realtime_duplex_demo_speak_gate_rejects_duplicate_event():
    demo = _load_demo_module()
    state = demo.DemoState()
    state.add({"type": "response.created", "response": {"id": "resp-1"}})
    state.add({"type": "response.speak", "response_id": "resp-1"})
    state.add({"type": "response.speak", "response_id": "resp-1"})

    result = demo._evaluate_response_speak_contract(state)

    assert result["response_speak_contract_ok"] is False
    assert result["duplicate_response_speak_ids"] == ["resp-1"]


def test_realtime_duplex_demo_speak_gate_rejects_text_channel():
    demo = _load_demo_module()
    state = demo.DemoState()
    state.add({"type": "response.created", "response": {"id": "resp-1"}})
    state.add(
        {
            "type": "response.speak",
            "response_id": "resp-1",
            "text": "must be emitted through response.audio_transcript.delta",
        }
    )

    result = demo._evaluate_response_speak_contract(state)

    assert result["response_speak_contract_ok"] is False
    assert result["text_bearing_response_speak_count"] == 1


def test_realtime_duplex_demo_gate_rejects_cross_turn_tail_reuse():
    demo = _load_demo_module()
    state = demo.DemoState()
    _add_response_transcript(state, "resp-1", transcript="第一轮回答")
    _add_response_transcript(state, "resp-2", transcript="第二轮仍然带着第一轮回答")

    result = demo._evaluate_transcript_integrity(
        state,
        ["resp-1", "resp-2"],
        expected_empty_response_ids=set(),
        require_cross_turn_independence=True,
    )

    assert result["cross_turn_independent_ok"] is False


def test_realtime_duplex_demo_gate_rejects_terminal_only_previous_tail():
    demo = _load_demo_module()
    state = demo.DemoState()
    _add_response_transcript(state, "resp-1", transcript="上一轮回答结尾是的吗？")
    _add_response_transcript(state, "resp-2", transcript="的吗？")

    result = demo._evaluate_transcript_integrity(
        state,
        ["resp-1", "resp-2"],
        expected_empty_response_ids=set(),
        require_cross_turn_independence=True,
    )

    assert result["cross_turn_independent_ok"] is False


def test_realtime_duplex_demo_gate_rejects_delta_done_mismatch():
    demo = _load_demo_module()
    state = demo.DemoState()
    _add_response_transcript(state, "resp-1", transcript="delta文本")
    done = next(event for event in state.events if event.get("type") == "response.audio_transcript.done")
    done["transcript"] = "另一个done文本"

    result = demo._evaluate_transcript_integrity(
        state,
        ["resp-1"],
        expected_empty_response_ids=set(),
        require_cross_turn_independence=False,
    )

    assert result["transcript_delta_done_ok"] is False


def test_realtime_duplex_demo_gate_rejects_terminal_only_stale_tail():
    demo = _load_demo_module()
    state = demo.DemoState()
    _add_response_transcript(
        state,
        "resp-empty",
        transcript="的吗？",
        audio=False,
    )

    result = demo._evaluate_transcript_integrity(
        state,
        ["resp-empty"],
        expected_empty_response_ids={"resp-empty"},
        require_cross_turn_independence=False,
    )

    assert result["empty_turns_ok"] is False


def test_realtime_duplex_demo_gate_rejects_audio_without_transcript():
    demo = _load_demo_module()
    state = demo.DemoState()
    _add_response_transcript(
        state,
        "resp-audio-no-text",
        transcript="",
        audio=True,
    )

    result = demo._evaluate_transcript_integrity(
        state,
        ["resp-audio-no-text"],
        expected_empty_response_ids=set(),
        require_cross_turn_independence=False,
    )

    assert result["nonempty_audio_has_transcript_ok"] is False


def test_realtime_duplex_demo_response_required_rejects_unselected_audio_only_response():
    demo = _load_demo_module()
    state = demo.DemoState()
    _add_response_transcript(state, "resp-audio-only", transcript="", audio=True)
    _add_response_transcript(state, "resp-with-text", transcript="你好", audio=True)

    assert (
        demo._all_audio_responses_have_transcript(
            state,
            ["resp-audio-only", "resp-with-text"],
        )
        is False
    )


def test_realtime_duplex_demo_gate_rejects_incomplete_model_turn_sentence():
    demo = _load_demo_module()
    state = demo.DemoState()
    _add_response_transcript(state, "resp-1", transcript="哎，不是说好不")

    result = demo._evaluate_transcript_integrity(
        state,
        ["resp-1"],
        expected_empty_response_ids=set(),
        require_cross_turn_independence=False,
        require_terminal_punctuation=True,
    )

    assert result["terminal_punctuation_ok"] is False


def test_realtime_duplex_demo_gate_accepts_short_interjection_without_punctuation():
    demo = _load_demo_module()
    state = demo.DemoState()
    _add_response_transcript(state, "resp-1", transcript="哈哈")

    result = demo._evaluate_transcript_integrity(
        state,
        ["resp-1"],
        expected_empty_response_ids=set(),
        require_cross_turn_independence=False,
        require_terminal_punctuation=True,
    )

    assert result["terminal_punctuation_ok"] is True


def test_realtime_duplex_demo_response_required_skips_audio_only_model_turn(monkeypatch):
    demo = _load_demo_module()
    state = demo.DemoState()

    async def fake_send_pcm16(*args, **kwargs):
        state.add({"type": "response.created", "response": {"id": "resp-audio-only"}})
        _add_response_transcript(state, "resp-audio-only", transcript="", audio=True)
        state.add({"type": "response.created", "response": {"id": "resp-with-text"}})
        _add_response_transcript(state, "resp-with-text", transcript="你好", audio=True)

    class FakeWebSocket:
        async def send(self, payload):
            return None

    monkeypatch.setattr(demo, "_send_pcm16", fake_send_pcm16)

    response_id, outcome = asyncio.run(
        demo._send_clean_turn(
            FakeWebSocket(),
            state,
            b"\x00\x00",
            transcript="fixture",
            duration_ms=None,
            chunk_ms=200,
            timeout_s=0.1,
            require_audio=True,
            validation_mode="response-required",
        )
    )

    assert response_id == "resp-with-text"
    assert outcome == "speak"


def test_realtime_duplex_demo_no_hint_realtime_turn_omits_transcript_hint(monkeypatch):
    demo = _load_demo_module()
    state = demo.DemoState()
    captured = {}

    async def fake_send_pcm16(*args, **kwargs):
        del args
        captured.update(kwargs)
        state.add(
            {
                "type": "response.listen",
                "response": {"metadata": {"model_listen": True}},
            }
        )

    class FakeWebSocket:
        async def send(self, payload):
            del payload

    monkeypatch.setattr(demo, "_send_pcm16", fake_send_pcm16)

    response_id, outcome = asyncio.run(
        demo._send_clean_turn(
            FakeWebSocket(),
            state,
            b"\x00\x00",
            transcript="turn label only",
            duration_ms=None,
            chunk_ms=200,
            timeout_s=0.1,
            require_audio=False,
            validation_mode="model-policy",
            send_transcript_hint=False,
            realtime_input=True,
            model_policy_settle_s=0.01,
        )
    )

    assert response_id is None
    assert outcome == "listen"
    assert captured["hints"] == {}
    assert captured["realtime_delay"] is True


def test_realtime_duplex_demo_continuous_input_omits_browser_commit(monkeypatch):
    demo = _load_demo_module()
    state = demo.DemoState()

    async def fake_send_pcm16(*args, **kwargs):
        del args, kwargs
        state.add({"type": "response.created", "response": {"id": "resp-continuous"}})
        _add_response_transcript(state, "resp-continuous", transcript="连续输入", audio=True)

    class FakeWebSocket:
        def __init__(self):
            self.messages = []

        async def send(self, payload):
            self.messages.append(demo.json.loads(payload))

    monkeypatch.setattr(demo, "_send_pcm16", fake_send_pcm16)
    ws = FakeWebSocket()

    response_id, outcome = asyncio.run(
        demo._send_clean_turn(
            ws,
            state,
            b"\x00\x00",
            transcript="fixture",
            duration_ms=None,
            chunk_ms=200,
            timeout_s=0.1,
            require_audio=True,
            validation_mode="response-required",
            commit_input=False,
        )
    )

    assert response_id == "resp-continuous"
    assert outcome == "speak"
    assert not any(message.get("type") == "input_audio_buffer.commit" for message in ws.messages)


def test_realtime_duplex_demo_no_hint_gate_does_not_require_transcription_event():
    demo = _load_demo_module()

    assert demo._input_transcription_ok(0, transcript_hints_enabled=False)
    assert not demo._input_transcription_ok(0, transcript_hints_enabled=True)
    assert demo._input_transcription_ok(3, transcript_hints_enabled=True)


def test_realtime_duplex_demo_gate_rejects_server_errors():
    demo = _load_demo_module()
    state = demo.DemoState()
    state.add({"type": "error", "code": "runtime_signal_failed", "error": "signal failed"})

    errors = demo._unexpected_error_events(state)

    assert errors == [{"type": "error", "code": "runtime_signal_failed", "error": "signal failed"}]


def test_realtime_duplex_demo_model_policy_waits_for_speak_after_intermediate_listen(monkeypatch):
    demo = _load_demo_module()
    state = demo.DemoState()

    async def fake_send_pcm16(*args, **kwargs):
        del args, kwargs
        state.add(
            {
                "type": "response.listen",
                "response": {"metadata": {"model_listen": True}},
            }
        )

        async def delayed_speak():
            await asyncio.sleep(0.01)
            state.add({"type": "response.created", "response": {"id": "resp-delayed"}})
            _add_response_transcript(state, "resp-delayed", transcript="延迟回答", audio=False)

        asyncio.create_task(delayed_speak())

    class FakeWebSocket:
        async def send(self, payload):
            del payload

    monkeypatch.setattr(demo, "_send_pcm16", fake_send_pcm16)

    response_id, outcome = asyncio.run(
        demo._send_clean_turn(
            FakeWebSocket(),
            state,
            b"\x00\x00",
            transcript="fixture",
            duration_ms=None,
            chunk_ms=200,
            timeout_s=0.2,
            require_audio=False,
            validation_mode="model-policy",
            model_policy_settle_s=0.05,
        )
    )

    assert response_id == "resp-delayed"
    assert outcome == "speak"


def test_realtime_duplex_demo_playback_gate_covers_unassigned_completed_response():
    demo = _load_demo_module()
    state = demo.DemoState()
    for response_id in ("resp-assigned", "resp-unassigned"):
        state.add({"type": "response.created", "response": {"id": response_id}})
        state.add(
            {
                "type": "response.done",
                "response": {
                    "id": response_id,
                    "metadata": {"playback": {"sent_ms": 1200}},
                },
            }
        )
    state.add(
        {
            "type": "playback.acknowledged",
            "event": {
                "item_id": "item_resp-assigned",
                "history_committed": True,
            },
        }
    )

    assert not demo._all_response_playback_history_committed(
        state,
        state.completed_response_ids(),
    )


def test_realtime_duplex_demo_acks_unassigned_completed_response():
    demo = _load_demo_module()
    state = demo.DemoState()
    for response_id in ("resp-assigned", "resp-unassigned"):
        state.add({"type": "response.created", "response": {"id": response_id}})
        state.add(
            {
                "type": "response.done",
                "response": {
                    "id": response_id,
                    "metadata": {"playback": {"sent_ms": 1200}},
                },
            }
        )
    state.add(
        {
            "type": "playback.acknowledged",
            "event": {
                "item_id": "item_resp-assigned",
                "history_committed": True,
            },
        }
    )

    class FakeWebSocket:
        def __init__(self):
            self.messages = []

        async def send(self, payload):
            message = demo.json.loads(payload)
            self.messages.append(message)
            state.add(
                {
                    "type": "playback.acknowledged",
                    "event": {
                        "item_id": message["item_id"],
                        "history_committed": True,
                    },
                }
            )

    ws = FakeWebSocket()
    asyncio.run(demo._ack_all_completed_response_playback(ws, state, timeout_s=0.1))

    assert [message["response_id"] for message in ws.messages] == ["resp-unassigned"]
    assert demo._all_response_playback_history_committed(state, state.completed_response_ids())


def test_realtime_duplex_demo_model_policy_does_not_assume_one_response_per_physical_input():
    demo = _load_demo_module()

    assert demo._response_cardinality_ok(
        ["resp-1", "resp-2", "resp-3", "resp-4"],
        expected_turns=3,
        validation_mode="model-policy",
    )
    assert not demo._response_cardinality_ok(
        ["resp-1", "resp-2", "resp-3", "resp-4"],
        expected_turns=3,
        validation_mode="response-required",
    )


def test_realtime_duplex_demo_only_requires_cross_turn_independence_for_response_required():
    demo = _load_demo_module()

    assert not demo._requires_cross_turn_independence(
        validation_mode="model-policy",
        distinct_inputs_required=True,
    )
    assert demo._requires_cross_turn_independence(
        validation_mode="response-required",
        distinct_inputs_required=True,
    )
    assert not demo._requires_cross_turn_independence(
        validation_mode="response-required",
        distinct_inputs_required=False,
    )


def test_realtime_duplex_demo_drains_late_model_response_before_close():
    demo = _load_demo_module()
    state = demo.DemoState()
    state.add({"type": "response.created", "response": {"id": "resp-first"}})
    state.add({"type": "response.done", "response": {"id": "resp-first"}})

    class FakeWebSocket:
        def __init__(self):
            self.messages = []

        async def send(self, payload):
            self.messages.append(demo.json.loads(payload))

    async def run_fixture():
        async def finish_late_response():
            await asyncio.sleep(0.01)
            state.add({"type": "response.created", "response": {"id": "resp-late"}})
            await asyncio.sleep(0.01)
            state.add({"type": "response.done", "response": {"id": "resp-late"}})

        late_response = asyncio.create_task(finish_late_response())
        ws = FakeWebSocket()
        await demo._drain_model_policy_responses(
            ws,
            state,
            timeout_s=0.2,
            settle_s=0.03,
        )
        await late_response
        return ws

    ws = asyncio.run(run_fixture())

    assert state.completed_response_ids() == ["resp-first", "resp-late"]
    assert not [message for message in ws.messages if message.get("type") == "session.close"]


def test_realtime_duplex_demo_model_response_drain_times_out_with_active_ids():
    demo = _load_demo_module()
    state = demo.DemoState()
    state.add({"type": "response.created", "response": {"id": "resp-active"}})

    class FakeWebSocket:
        async def send(self, payload):
            del payload

    with pytest.raises(TimeoutError, match="resp-active"):
        asyncio.run(
            demo._drain_model_policy_responses(
                FakeWebSocket(),
                state,
                timeout_s=0.03,
                settle_s=0.01,
            )
        )


def test_realtime_duplex_demo_waits_at_each_model_unit_and_stops_after_speak():
    demo = _load_demo_module()

    class FakeWebSocket:
        def __init__(self):
            self.messages = []

        async def send(self, payload):
            self.messages.append(demo.json.loads(payload))

    async def run_fixture():
        ws = FakeWebSocket()
        model_unit_message_counts = []

        async def on_model_unit_ready():
            model_unit_message_counts.append(len(ws.messages))
            await asyncio.sleep(0)
            return len(model_unit_message_counts) < 2

        await demo._send_pcm16(
            ws,
            b"\x01\x00" * (demo.PCM16_SAMPLE_RATE * 3),
            chunk_ms=200,
            realtime_delay=False,
            on_model_unit_ready=on_model_unit_ready,
        )
        return ws, model_unit_message_counts

    ws, model_unit_message_counts = asyncio.run(run_fixture())

    assert model_unit_message_counts == [5, 10]
    assert len(ws.messages) == 10


def test_realtime_duplex_demo_listen_only_overlap_sends_next_turn_before_first_done(monkeypatch):
    demo = _load_demo_module()
    state = demo.DemoState()
    send_calls = 0

    async def fake_send_pcm16(*args, **kwargs):
        nonlocal send_calls
        send_calls += 1
        if send_calls == 1:
            assert len(args[1]) == 2 + demo.PCM16_SAMPLE_RATE * demo.PCM16_BYTES_PER_SAMPLE
            state.add({"type": "input_audio_buffer.speech_started", "turn": 1})
            state.add({"type": "response.created", "response": {"id": "resp-first"}})
            state.add(
                {
                    "type": "response.audio.delta",
                    "response_id": "resp-first",
                    "delta": "YQ==",
                }
            )
            assert await kwargs["on_model_unit_ready"]() is False
            return

        del args
        assert not state.response_done("resp-first")
        kwargs["on_model_unit_ready"]()
        state.add({"type": "input_audio_buffer.speech_started", "turn": 2})
        _add_response_transcript(state, "resp-first", transcript="第一轮完成", audio=False)
        state.add({"type": "response.created", "response": {"id": "resp-second"}})
        _add_response_transcript(state, "resp-second", transcript="第二轮完成", audio=True)

    class FakeWebSocket:
        def __init__(self):
            self.messages = []

        async def send(self, payload):
            self.messages.append(payload)

    monkeypatch.setattr(demo, "_send_pcm16", fake_send_pcm16)
    ws = FakeWebSocket()

    response_ids, outcomes, overlap_ok = asyncio.run(
        demo._send_listen_only_overlap_pair(
            ws,
            state,
            b"\x00\x00",
            b"\x01\x00",
            transcripts=("first", "second"),
            durations_ms=(None, None),
            chunk_ms=200,
            timeout_s=0.2,
            send_transcript_hint=False,
            realtime_input=True,
            model_policy_settle_s=0.01,
            validation_mode="response-required",
            silence_ms=1000,
        )
    )

    assert response_ids == ["resp-first", "resp-second"]
    assert outcomes == ["speak", "speak"]
    assert overlap_ok is True
    assert sum(demo.json.loads(message)["type"] == "input_audio_buffer.commit" for message in ws.messages) == 2


def test_realtime_duplex_demo_playback_ack_identifies_and_commits_response():
    demo = _load_demo_module()
    state = demo.DemoState()
    response_id = "resp-overlap-second"
    state.add(
        {
            "type": "response.done",
            "response": {
                "id": response_id,
                "metadata": {"playback": {"sent_ms": 1200}},
            },
        }
    )

    class FakeWebSocket:
        def __init__(self):
            self.messages = []

        async def send(self, payload):
            message = demo.json.loads(payload)
            self.messages.append(message)
            state.add(
                {
                    "type": "playback.acknowledged",
                    "event": {
                        "item_id": message.get("item_id"),
                        "history_committed": message.get("item_id") == f"item_{response_id}",
                    },
                }
            )

    ws = FakeWebSocket()
    asyncio.run(
        demo._ack_response_playback(
            ws,
            state,
            response_id,
            timeout_s=0.1,
            label="overlap second",
        )
    )

    assert ws.messages == [
        {
            "type": "playback.ack",
            "response_id": response_id,
            "item_id": f"item_{response_id}",
            "played_ms": 1200,
            "committed_ms": 1200,
        }
    ]
    assert state.playback_history_committed_count == 1


def test_realtime_duplex_demo_writes_audio_per_response(tmp_path):
    demo = _load_demo_module()
    state = demo.DemoState()
    for response_id, payload in (("resp-1", b"\x01\x00"), ("resp-2", b"\x02\x00")):
        state.add({"type": "response.created", "response": {"id": response_id}})
        state.add(
            {
                "type": "response.audio.delta",
                "response_id": response_id,
                "delta": demo.base64.b64encode(payload).decode(),
                "sample_rate_hz": 24000,
            }
        )

    demo._write_demo_artifacts(state, tmp_path, output_audio_format="pcm16")

    for index, expected in enumerate((b"\x01\x00", b"\x02\x00"), start=1):
        with wave.open(str(tmp_path / f"response_{index:02d}.wav"), "rb") as wf:
            assert wf.getframerate() == 24000
            assert wf.readframes(wf.getnframes()) == expected
