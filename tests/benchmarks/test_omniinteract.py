# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import subprocess
import tarfile
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncInput
from vllm.benchmarks.serve import TaskType

from vllm_omni.benchmarks import omniinteract as oi
from vllm_omni.benchmarks import serve as benchmark_serve
from vllm_omni.benchmarks.data_modules import omniinteract_dataset as data
from vllm_omni.benchmarks.metrics.metrics import calculate_metrics
from vllm_omni.benchmarks.patch import patch as benchmark_patch
from vllm_omni.entrypoints.cli.benchmark.cli_args import preprocess_serve_args
from vllm_omni.entrypoints.cli.benchmark.serve import OmniBenchmarkServingSubcommand
from vllm_omni.experimental.fullduplex.client import RealtimeEventCollector
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]
MODEL = "openbmb/MiniCPM-o-4_5"


def _case(tmp_path: Path, *, subset: str = "1q1a", name: str = "video.mp4") -> data.OmniInteractCase:
    video, annotation = tmp_path / name, tmp_path / f"{Path(name).stem}.json"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.touch()
    annotation.write_text("{}")
    return data.OmniInteractCase(subset, name, video, annotation, "multi_turn")


def _collector(*events: tuple[dict[str, object], float]) -> RealtimeEventCollector:
    collector = RealtimeEventCollector()
    for event, received_at in events:
        collector.add(event, received_at_s=received_at)
    return collector


def _created(response_id: str = "r1") -> dict[str, object]:
    return {"type": "response.created", "response": {"id": response_id}}


def _done(response_id: str = "r1", status: str = "completed") -> dict[str, object]:
    return {"type": "response.done", "response": {"id": response_id, "status": status}}


def _audio(
    response_id: str = "r1",
    *,
    samples: int = 2400,
    rate: int | None = 24_000,
    value: int = 1,
) -> dict[str, object]:
    event: dict[str, object] = {
        "type": "response.audio.delta",
        "response_id": response_id,
        "format": "pcm16",
        "delta": base64.b64encode(bytes((value, 0)) * samples).decode(),
    }
    if rate is not None:
        event["sample_rate_hz"] = rate
    return event


def _text(response_id: str = "r1", value: str = "hello") -> dict[str, object]:
    return {"type": "response.audio_transcript.delta", "response_id": response_id, "delta": value}


def _listen(*, buffering: bool = False) -> dict[str, object]:
    return {
        "type": "response.listen",
        "response": {"metadata": {"model_listen": True, "buffering": buffering}},
    }


def _session_options(tmp_path: Path) -> data.OmniInteractSessionOptions:
    return data.OmniInteractSessionOptions(tmp_path / "artifacts", 10.0, 5.0, "reference.wav", True)


def _sample(case: data.OmniInteractCase, options: data.OmniInteractSessionOptions):
    return data.OmniInteractSampleRequest(
        prompt="",
        prompt_len=0,
        expected_output_len=0,
        multi_modal_data=None,
        request_id="measured-0",
        omniinteract_case=case,
        omniinteract_options=options,
    )


def _write_dataset(root: Path) -> Path:
    data_root = root / "data"
    for subset in ("1q1a", "1q1a_math"):
        subset_root = data_root / subset
        (subset_root / "videos").mkdir(parents=True)
        (subset_root / "annotations").mkdir()
        (subset_root / "videos" / f"{subset}.mp4").touch()
        (subset_root / "annotations" / f"{subset}.json").write_text("{}")
        (subset_root / "video_json_map.json").write_text(
            json.dumps(
                {
                    "entries": [
                        {
                            "video": f"videos/{subset}.mp4",
                            "annotation": f"annotations/{subset}.json",
                            "scene_type": "multi_turn",
                        }
                    ]
                }
            )
        )
    one_to_many = data_root / "1qna"
    (one_to_many / "videos_bench" / "nested").mkdir(parents=True)
    (one_to_many / "annotations" / "nested").mkdir(parents=True)
    (one_to_many / "videos_bench" / "nested" / "guide.mp4").touch()
    (one_to_many / "annotations" / "nested" / "guide.json").write_text("{}")
    return data_root


def _successful_output(
    tmp_path: Path,
    *,
    audio_time: float = 10.0,
    status: str = "completed",
) -> tuple[data.OmniInteractCase, RealtimeEventCollector, oi.OmniInteractCaseResult]:
    case = _case(tmp_path)
    collector = _collector(
        (_created(), 9.9),
        (_audio(), audio_time),
        (_text(), audio_time + 0.01),
        (_done(status=status), audio_time + 0.1),
    )
    result = oi.OmniInteractCaseResult(case.subset, str(case.video_path), "")
    return case, collector, result


def _write_success(
    root: Path,
    case: data.OmniInteractCase,
    collector: RealtimeEventCollector,
    result: oi.OmniInteractCaseResult,
    **kwargs,
):
    require_response = kwargs.pop("require_response", True)
    return oi.prepare_success_artifacts(
        root,
        case,
        collector,
        stream_start=10.0,
        video_duration_s=1.0,
        require_response=require_response,
        result=result,
        **kwargs,
    )


def test_dataset_discovers_official_layouts_and_total_selection(tmp_path: Path):
    root = _write_dataset(tmp_path)
    all_cases = data.discover_omniinteract_cases(root, data.OMNIINTERACT_SUBSETS, num_prompts=0, disable_shuffle=True)
    selected = data.discover_omniinteract_cases(root, data.OMNIINTERACT_SUBSETS, num_prompts=2, disable_shuffle=True)
    assert [case.subset for case in all_cases] == ["1q1a", "1q1a_math", "1qna"]
    assert all_cases[-1].video_rel == "videos_bench/nested/guide.mp4"
    assert [case.subset for case in selected] == ["1q1a", "1q1a_math"]


def _archive(path: Path, member: str) -> None:
    with tarfile.open(path, "w") as archive:
        content = b"{}"
        item = tarfile.TarInfo(member)
        item.size = len(content)
        archive.addfile(item, io.BytesIO(content))


def test_archive_is_safe_and_atomically_shared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    bad = tmp_path / "bad.tar"
    _archive(bad, "../escape")
    with pytest.raises(ValueError, match="Unsafe path"):
        data._extract_archive(bad, tmp_path / "bad-cache")

    good = tmp_path / "good.tar"
    _archive(good, "data/1q1a/video_json_map.json")
    target = tmp_path / "cache"
    barrier, safe_extract = threading.Barrier(2), data._safe_extract
    monkeypatch.setattr(data, "_safe_extract", lambda *args: (barrier.wait(), safe_extract(*args)))
    with ThreadPoolExecutor(2) as executor:
        roots = list(executor.map(lambda _: data._extract_archive(good, target), range(2)))
    assert roots[0] == roots[1]
    assert (roots[0] / "1q1a" / "video_json_map.json").is_file()
    assert not list(target.glob(".tmp-*"))


def test_hub_archive_uses_vllm_filesystem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "source.tar"
    _archive(source, "data/1q1a/video_json_map.json")

    class FS:
        def get_file(self, remote: str, local: str) -> None:
            assert remote == "datasets/org/repo/data.tar.gz"
            Path(local).write_bytes(source.read_bytes())

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setattr(data, "hf_fs", lambda: FS())
    assert (data.resolve_omniinteract_root(None, "org/repo") / "1q1a").is_dir()


def test_media_commands_are_bounded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls: list[float] = []
    commands: list[list[str]] = []
    bounded_run = oi._run_media_command

    def run(command, *, timeout_s, text=False):
        calls.append(timeout_s)
        commands.append(command)
        if command[0] == "ffprobe":
            return subprocess.CompletedProcess(command, 0, "1.0", "")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(oi, "_run_media_command", run)
    duration, pcm, frames = oi.prepare_media(tmp_path / "video.mp4", 1.0, timeout_s=3.0, max_duration_s=1.0)
    assert (duration, len(pcm), frames, calls) == (1.0, 32_000, [None], [3.0, 3.0, 3.0])
    assert commands[1][commands[1].index("-t") + 1] == "1.0"
    assert commands[1][commands[1].index("-fs") + 1] == "32000"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired("x", 3)))
    with pytest.raises(TimeoutError, match="timed out"):
        bounded_run(["ffmpeg"], timeout_s=3.0)


def test_media_duration_is_bounded_before_decode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls = 0

    def run(command, *, timeout_s, text=False):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(command, 0, "601.0", "")

    monkeypatch.setattr(oi, "_run_media_command", run)
    with pytest.raises(ValueError, match="Invalid video duration"):
        oi.prepare_media(tmp_path / "video.mp4", 1.0, timeout_s=3.0, max_duration_s=600.0)
    assert calls == 1


def test_final_commit_keeps_a_partial_model_unit():
    """Exact units retain a sample for the server's correlated final flush."""

    events = [{"session": {"capabilities": {"chunk_period_ms": 1000}}}]
    exact_unit = b"\0\0" * 16_000
    assert oi._ensure_final_commit_tail(exact_unit, events) == exact_unit[:-2]
    assert oi._ensure_final_commit_tail(exact_unit + b"\0\0", events) == exact_unit + b"\0\0"


@pytest.mark.parametrize(
    ("events", "match"),
    [
        (((_done(), 0.0),), "without response.created"),
        (((_created(), 0.0), (_created(), 0.1)), "duplicate response.created"),
        (((_created(), 0.0), (_done(), 0.1), (_done(), 0.2)), "duplicate response.done"),
        (
            (
                (_created(), 0.0),
                ({"type": "response.done", "response": {"id": "r1", "status_details": {"type": "failed"}}}, 0.1),
            ),
            "reports failure",
        ),
    ],
)
def test_response_ledger_rejects_identity_errors(events, match: str):
    with pytest.raises(ValueError, match=match):
        oi.response_ledger(_collector(*events))


class _CompletionClient:
    def __init__(self, collector: RealtimeEventCollector):
        self.events, self.acks = collector, []

    def raise_if_reader_stopped(self) -> None:
        return None

    async def send_playback_ack(self, response_id: str, played_ms: int) -> None:
        self.acks.append((response_id, played_ms))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tail", "succeeds"),
    [
        (((_created(), 1.1), (_done(), 1.2)), True),
        (((_listen(), 1.1),), True),
        (((_listen(buffering=True), 1.1),), False),
        ((({"type": "session.closed", "reason": "timeout"}, 1.1),), False),
    ],
)
async def test_completion_requires_a_final_model_decision(tail, succeeds: bool):
    events = _collector(({"type": "input_audio_buffer.committed"}, 1.0), *tail)
    coroutine = oi.wait_for_session_completion(
        _CompletionClient(events), oi._Playback(), commit_from=0, timeout_s=0.08, settle_s=0.0
    )
    if succeeds:
        assert await coroutine == 0
    else:
        with pytest.raises((TimeoutError, RuntimeError)):
            await coroutine


@pytest.mark.asyncio
async def test_deferred_commit_ignores_the_prior_response_terminal():
    events = _collector(
        (_created("old"), 0.0),
        ({"type": "input_audio_buffer.committed", "event": {"overlap_deferred": True}}, 1.0),
        (_done("old"), 1.1),
        (_listen(), 1.2),
    )
    assert (
        await oi.wait_for_session_completion(
            _CompletionClient(events), oi._Playback(), commit_from=1, timeout_s=0.1, settle_s=0.0
        )
        == 1
    )


@pytest.mark.asyncio
async def test_playback_acks_only_after_serial_audio_drain():
    collector = _collector((_audio(samples=2400), 10.0), (_done(), 10.01))
    client, playback = _CompletionClient(collector), oi._Playback()
    await playback.acknowledge(client, now=10.05)
    assert client.acks == []
    await playback.acknowledge(client, now=10.1)
    assert client.acks == [("r1", 100)]


@pytest.mark.parametrize(
    ("event", "match"),
    [
        (_audio(rate=16_000), "24000 Hz"),
        ({**_audio(), "format": "opus"}, "pcm16"),
        ({**_audio(), "delta": "!"}, "base64"),
    ],
)
def test_playback_rejects_non_official_audio(event: dict[str, object], match: str):
    playback = oi._Playback()
    with pytest.raises(ValueError, match=match):
        playback.ingest(_collector((event, 1.0)))


def test_artifacts_publish_official_bundle_and_sparse_deferred_state(tmp_path: Path):
    case, collector, result = _successful_output(tmp_path)
    summary = _write_success(
        tmp_path / "out",
        case,
        collector,
        result,
    )
    context = result._artifact_context
    assert result.success and result.eligible_for_official_eval
    assert summary["scene_type"] == "multi_turn"
    assert context and context.spans
    assert all("delta" not in event for event in context.events if event["type"] == "response.audio.delta")
    oi.publish_deferred_case_artifacts(tmp_path / "out", case, result)
    directory = oi._output_dir(tmp_path / "out", case)
    assert all((directory / name).is_file() for name in oi.SUCCESS_ARTIFACTS)
    with wave.open(str(directory / "output.wav")) as output:
        assert (output.getframerate(), output.getnframes()) == (24_000, 24_000)
    transcript = json.loads((directory / "wav_transcript.json").read_text())
    assert transcript["text"] == "hello" and transcript["chunks"][0]["timestamp"] == [0.0, 0.1]
    assert "serialized playback queue" in transcript["timestamp_semantics"]
    assert result._artifact_context is None


@pytest.mark.parametrize(
    ("audio_time", "status", "reason"),
    [(10.95, "completed", "audio_clipped"), (10.0, "cancelled", "cancelled_response")],
)
def test_ineligible_outputs_are_excluded_from_manifest(tmp_path: Path, audio_time: float, status: str, reason: str):
    case, collector, result = _successful_output(tmp_path, audio_time=audio_time, status=status)
    _write_success(tmp_path / "out", case, collector, result, require_response=False)
    assert reason in result.official_eval_ineligible_reasons
    oi.write_batch_artifacts(tmp_path / "out", [case], [result])
    assert (tmp_path / "out" / "official_eval_manifest.jsonl").read_text() == ""


def test_artifacts_require_complete_audio_and_transcript_response(tmp_path: Path):
    case, _, result = _successful_output(tmp_path)
    collector = _collector((_created(), 10.0), (_audio(), 10.0), (_done(), 10.1))
    with pytest.raises(ValueError, match="audio and transcript"):
        _write_success(tmp_path, case, collector, result)


def test_publication_failure_removes_partial_success_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    case, collector, result = _successful_output(tmp_path)
    _write_success(tmp_path, case, collector, result)
    original = oi._atomic_write_json

    def fail(path: Path, value: object) -> None:
        if path.name == "events.json":
            raise OSError("disk full")
        original(path, value)

    monkeypatch.setattr(oi, "_atomic_write_json", fail)
    with pytest.raises(OSError, match="disk full"):
        oi.publish_deferred_case_artifacts(tmp_path, case, result)
    directory = oi._output_dir(tmp_path, case)
    assert not any((directory / name).exists() for name in oi.SUCCESS_ARTIFACTS)
    assert not result.success and "artifact_write_failed" in result.official_eval_ineligible_reasons


def test_atomic_write_preserves_destination_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    destination = tmp_path / "output.wav"
    destination.write_bytes(b"old")
    monkeypatch.setattr(oi, "write_pcm16_wav", lambda *a, **k: (_ for _ in ()).throw(OSError("write")))
    with pytest.raises(OSError):
        oi._atomic_write_wav(destination, b"\0\0", 24_000)
    assert destination.read_bytes() == b"old"
    assert not list(tmp_path.glob(".*.tmp"))


class _RealtimeClient:
    instances: list[_RealtimeClient] = []

    def __init__(self, url: str, **kwargs):
        self.url, self.events, self.acks, self.configure_kwargs = url, RealtimeEventCollector(), [], {}
        self.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def configure(self, model: str, **kwargs) -> None:
        self.configure_kwargs = kwargs
        self.events.add({"type": "session.created", "session": {"capabilities": {"chunk_period_ms": 1000}}})

    async def send(self, event: dict[str, object]) -> None:
        return None

    async def commit(self) -> None:
        now = time.monotonic()
        for event, offset in (
            ({"type": "input_audio_buffer.committed"}, 0.0),
            (_created(), 0.001),
            (_audio(samples=24), 0.002),
            (_text(value="answer"), 0.003),
            (_done(), 0.004),
        ):
            self.events.add(event, received_at_s=now + offset)

    async def send_playback_ack(self, response_id: str, played_ms: int) -> None:
        self.acks.append((response_id, played_ms))

    async def close_session(self, **kwargs) -> None:
        self.events.add({"type": "session.closed"})

    def raise_if_reader_stopped(self) -> None:
        return None


@pytest.mark.asyncio
async def test_public_runner_executes_one_prepared_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    case = _case(tmp_path)
    config = oi.OmniInteractBenchmarkConfig(
        model=MODEL,
        output_root=tmp_path / "out",
        ref_audio="ref.wav",
        require_response=True,
        extra_body={"custom": "value"},
    )
    prepared = data.OmniInteractPreparedInput(1.0, b"\0\0\0\0", ("frame",), "data:audio/wav;base64,ref")
    monkeypatch.setattr(oi, "RealtimeDuplexClient", _RealtimeClient)
    monkeypatch.setattr(oi, "_COMPLETION_SETTLE_S", 0.0)
    result = await oi.run_omniinteract_case(
        case,
        config,
        request_index=0,
        prepared_input=prepared,
    )
    assert result.success and result.transcript == "answer" and result._artifact_context is not None
    assert not config.output_root.exists()
    assert "autostart=0" in _RealtimeClient.instances[-1].url
    assert _RealtimeClient.instances[-1].configure_kwargs["extra_body"] == {"custom": "value"}
    assert _RealtimeClient.instances[-1].acks == [("r1", 1)]


def test_standard_sample_loading_prepares_media_before_timing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_dataset(tmp_path)
    ref = tmp_path / "ref.wav"
    ref.touch()
    parser = TrackingArgumentParser()
    OmniBenchmarkServingSubcommand.add_cli_args(parser)
    args = parser.parse_args(
        [
            "--backend",
            "openai-realtime-duplex",
            "--dataset-name",
            "omniinteract",
            "--dataset-path",
            str(tmp_path),
            "--model",
            MODEL,
            "--endpoint",
            "/v1/realtime",
            "--num-prompts",
            "0",
            "--omniinteract-ref-audio",
            str(ref),
        ]
    )
    preprocess_serve_args(args)
    assert args.endpoint == "/v1/realtime"
    monkeypatch.setattr(benchmark_patch, "reference_audio_data_url", lambda _: "data:audio/wav;base64,ref")
    monkeypatch.setattr(benchmark_patch, "prepare_media", lambda *a, **k: (1.0, b"pcm", ["frame"]))
    samples = benchmark_patch.get_samples(args, None)
    assert len(samples) == args.num_prompts == 3
    assert all(sample.omniinteract_prepared_input.video_frames == ("frame",) for sample in samples)


def _request(case: data.OmniInteractCase, options: data.OmniInteractSessionOptions) -> RequestFuncInput:
    request = RequestFuncInput(
        model=MODEL,
        model_name=MODEL,
        prompt="",
        api_url="http://server:8000/v1/realtime",
        prompt_len=0,
        output_len=0,
        logprobs=None,
        multi_modal_content=None,
        ignore_eos=False,
        request_id="measured-0",
    )
    request.omniinteract_case, request.omniinteract_options = case, options
    return request


@pytest.mark.asyncio
async def test_adapter_requires_and_forwards_exact_prepared_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    case, options = _case(tmp_path), _session_options(tmp_path)
    request = _request(case, options)
    output = await benchmark_patch.async_request_openai_realtime_duplex(request, session=None)
    assert not output.success and "prepared before benchmark request timing" in output.error

    prepared = data.OmniInteractPreparedInput(1.0, b"pcm", ("frame",), "data:audio/wav;base64,ref")
    request.omniinteract_prepared_input = prepared
    request.extra_headers = {"X-Proxy": "test"}
    request.extra_body = {"custom": "value"}
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    seen = {}

    async def run(_case_arg, config, **kwargs):
        seen["headers"] = config.extra_headers
        seen["extra_body"] = config.extra_body
        seen.update(kwargs)
        return oi.OmniInteractCaseResult(
            case.subset,
            str(case.video_path),
            str(options.output_root),
            success=True,
            transcript="timing metadata missing",
            output_tokens=0,
            duplex_request_metrics=[{"request_metrics": {"ttft_ms": 1.0}}],
            duplex_session_metrics={"mean_ttft_ms": 10.0},
        )

    monkeypatch.setattr(benchmark_patch, "run_omniinteract_case", run)
    output = await benchmark_patch.async_request_openai_realtime_duplex(request, session=None)
    assert output.success and seen["prepared_input"] is prepared
    assert seen["headers"] == {"Authorization": "Bearer secret", "X-Proxy": "test", "x-request-id": "measured-0"}
    assert seen["extra_body"] == {"custom": "value"}
    assert output.itl == []
    assert output.tpot_measured is False
    assert seen["capture_artifacts"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_metrics", "expected_itl", "expected_text_latency"),
    [
        (
            [
                {"stage0_tokens": {"output_token_count": 3, "itls_ms": [10.0, 20.0]}},
                {"stage0_tokens": {"output_token_count": 2, "itls_ms": [30.0]}},
            ],
            [0.01, 0.02, 0.03],
            0.16,
        ),
        (
            [
                {"stage0_tokens": {"output_token_count": 3, "tpot_ms": 10.0}},
                {"stage0_tokens": {"output_token_count": 2, "tpot_ms": 40.0}},
            ],
            [],
            0.18,
        ),
        (
            [
                {"stage0_tokens": {"output_token_count": 3, "tpot_ms": 0.0}},
                {"stage0_tokens": {"output_token_count": 2, "tpot_ms": 0.0}},
            ],
            [],
            0.1,
        ),
    ],
)
async def test_adapter_reports_exact_or_weighted_token_timing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request_metrics: list[dict[str, object]],
    expected_itl: list[float],
    expected_text_latency: float,
):
    case, options = _case(tmp_path), _session_options(tmp_path)
    request = _request(case, options)
    request.omniinteract_prepared_input = data.OmniInteractPreparedInput(
        1.0, b"pcm", ("frame",), "data:audio/wav;base64,ref"
    )

    async def run(*args, **kwargs):
        return oi.OmniInteractCaseResult(
            case.subset,
            str(case.video_path),
            str(options.output_root),
            success=True,
            output_tokens=5,
            duplex_request_metrics=request_metrics,
            duplex_session_metrics={"mean_ttft_ms": 100.0},
        )

    monkeypatch.setattr(benchmark_patch, "run_omniinteract_case", run)
    output = await benchmark_patch.async_request_openai_realtime_duplex(request, session=None)

    assert output.success
    assert output.itl == expected_itl
    assert output.text_latency == pytest.approx(expected_text_latency)
    assert output.tpot_measured is True


def test_batch_finalization_publishes_only_measured_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    case, options = _case(tmp_path), _session_options(tmp_path)
    options.output_root.mkdir()
    sample = _sample(case, options)
    result = oi.OmniInteractCaseResult(
        case.subset, str(case.video_path), str(options.output_root), success=True, eligible_for_official_eval=True
    )
    output = benchmark_patch.MixRequestFuncOutput(success=True)
    output.omniinteract_case_result = result
    published = []
    monkeypatch.setattr(benchmark_patch, "publish_deferred_case_artifacts", lambda *args: published.append(args))
    summary = benchmark_patch._finalize_omniinteract_batch([sample], [output])
    assert summary and summary["total"] == summary["success"] == summary["eligible_for_official_eval"] == 1
    assert summary["failed"] == summary["successful_but_ineligible"] == summary["audio_clipped_bytes"] == 0
    assert published == [(options.output_root, case, result)]


def test_case_artifact_failures_do_not_discard_serving_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    case, options = _case(tmp_path), _session_options(tmp_path)
    sample = _sample(case, options)
    result = oi.OmniInteractCaseResult(
        case.subset, str(case.video_path), str(options.output_root), success=True, eligible_for_official_eval=True
    )
    output = benchmark_patch.MixRequestFuncOutput(success=True, error="inference warning")
    output.omniinteract_case_result = result
    monkeypatch.setattr(
        benchmark_patch,
        "publish_deferred_case_artifacts",
        lambda *args: (_ for _ in ()).throw(OSError("case disk full")),
    )
    monkeypatch.setattr(
        benchmark_patch,
        "write_failure_artifacts",
        lambda *args: (_ for _ in ()).throw(OSError("failure disk full")),
    )
    monkeypatch.setattr(benchmark_patch, "write_omniinteract_batch_artifacts", lambda *args: None)

    summary = benchmark_patch._finalize_omniinteract_batch([sample], [output])
    metrics, _ = calculate_metrics(
        [],
        [output],
        1.0,
        None,
        [50.0],
        {},
        TaskType.GENERATION,
        [],
        None,
        float("inf"),
        1.0,
    )

    assert summary and summary["failed"] == 1
    assert summary["artifacts_complete"] is False
    assert len(summary["artifact_errors"]) == 2
    assert output.success and metrics.completed == 1
    assert output.error.startswith("inference warning\n")
    assert "Artifact publication failed: case disk full" in output.error


def test_batch_artifact_failure_removes_partial_batch_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    case, options = _case(tmp_path), _session_options(tmp_path)
    sample = _sample(case, options)
    result = oi.OmniInteractCaseResult(
        case.subset, str(case.video_path), str(options.output_root), success=True, eligible_for_official_eval=True
    )
    output = benchmark_patch.MixRequestFuncOutput(success=True)
    output.omniinteract_case_result = result
    monkeypatch.setattr(benchmark_patch, "publish_deferred_case_artifacts", lambda *args: None)

    def fail_batch(root, *args):
        root.mkdir(parents=True, exist_ok=True)
        (root / "batch_summary.json").write_text("partial")
        raise OSError("batch disk full")

    monkeypatch.setattr(benchmark_patch, "write_omniinteract_batch_artifacts", fail_batch)

    summary = benchmark_patch._finalize_omniinteract_batch([sample], [output])

    assert summary and summary["success"] == 1 and summary["artifacts_complete"] is False
    assert summary["artifact_errors"] == ["Batch artifact publication failed: batch disk full"]
    assert output.success
    assert not (options.output_root / "batch_summary.json").exists()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("omniinteract_ref_audio", None, "ref-audio"),
        ("endpoint", "/v1/completions", "v1/realtime"),
        ("max_concurrency", 0, "positive"),
        ("skip_tokenizer_init", True, "skip-tokenizer-init"),
    ],
)
def test_cli_rejects_incompatible_omniinteract_options(field: str, value, match: str):
    args = argparse.Namespace(
        dataset_name="omniinteract",
        backend="openai-realtime-duplex",
        endpoint="/v1/realtime",
        omniinteract_ref_audio="ref.wav",
        max_concurrency=1,
        skip_tokenizer_init=False,
    )
    setattr(args, field, value)
    with pytest.raises(ValueError, match=match):
        preprocess_serve_args(args)


def test_whole_benchmark_lock_serializes_a_shared_output_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    entered, release = threading.Event(), threading.Event()
    calls: list[int] = []

    async def fake_main_async(args):
        calls.append(len(calls))
        if len(calls) == 1:
            entered.set()
            await asyncio.to_thread(release.wait)
        return {"total": 1}

    monkeypatch.setattr(benchmark_serve, "main_async", fake_main_async)
    args = argparse.Namespace(dataset_name="omniinteract", omniinteract_output_dir=tmp_path, extra_body=None)
    with ThreadPoolExecutor(2) as executor:
        first = executor.submit(benchmark_serve.main, args)
        assert entered.wait(1)
        second = executor.submit(benchmark_serve.main, args)
        time.sleep(0.05)
        assert calls == [0]
        release.set()
        assert first.result() == second.result() == {"total": 1}
    assert calls == [0, 1]
