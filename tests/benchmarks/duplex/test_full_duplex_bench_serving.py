# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU coverage for Full-Duplex-Bench discovery and native replay."""

import json
import wave
from pathlib import Path

import pytest
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncInput

from vllm_omni.benchmarks.duplex import full_duplex_bench_serving as adapter
from vllm_omni.benchmarks.duplex import omni_duplex_eval_runner as runner
from vllm_omni.benchmarks.duplex.full_duplex_bench_dataset import discover_full_duplex_bench
from vllm_omni.benchmarks.patch import patch
from vllm_omni.entrypoints.cli.benchmark.cli_args import preprocess_serve_args
from vllm_omni.entrypoints.cli.benchmark.serve import OmniBenchmarkServingSubcommand
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]


def _wav(path: Path, *, seconds: float = 0.02) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\0\0" * round(16_000 * seconds))


def _v15_sample(root: Path, subset: str, name: str, *, valid_metadata: bool = True) -> None:
    sample = root / "v1.5" / subset / name
    _wav(sample / "input.wav")
    _wav(sample / "clean_input.wav")
    timestamps = [0.001, 0.01] if valid_metadata else [0.02, 0.01]
    (sample / "metadata.json").write_text(json.dumps({"timestamps": timestamps}), encoding="utf-8")


def _v10_sample(root: Path, subset: str, name: str) -> None:
    sample = root / "v1.0" / subset / name
    _wav(sample / "input.wav")
    if subset == "synthetic_pause_handling":
        (sample / "pause.json").write_text(json.dumps([{"timestamp": [0.001, 0.002]}]), encoding="utf-8")


def _args(tmp_path: Path, *extra: str):
    parser = TrackingArgumentParser()
    OmniBenchmarkServingSubcommand.add_cli_args(parser)
    return parser.parse_args(
        [
            "--model",
            "mock",
            "--backend",
            "openai-realtime-duplex",
            "--dataset-name",
            "full-duplex-bench",
            "--dataset-path",
            str(tmp_path / "dataset"),
            "--endpoint",
            "/v1/realtime",
            "--result-dir",
            str(tmp_path / "responses"),
            "--disable-shuffle",
            *extra,
        ]
    )


def test_v15_discovery_uses_natural_order_and_exact_ids(tmp_path: Path):
    for name in ("10", "2", "1"):
        _v15_sample(tmp_path, "user_interruption", name)
    samples = discover_full_duplex_bench(
        tmp_path,
        version="1.5",
        subsets=("user_interruption",),
        max_per_subset=2,
    )
    assert [sample.id for sample in samples] == ["user_interruption/1", "user_interruption/2"]
    assert [[audio.variant for audio in sample.inputs] for sample in samples] == [
        ["input", "clean_input"],
        ["input", "clean_input"],
    ]

    selected = discover_full_duplex_bench(
        tmp_path,
        version="1.5",
        subsets=("user_interruption",),
        sample_ids=("user_interruption/10",),
    )
    assert [sample.id for sample in selected] == ["user_interruption/10"]


def test_v10_discovery_accepts_parent_or_version_root(tmp_path: Path):
    _v10_sample(tmp_path, "synthetic_pause_handling", "1")
    parent = discover_full_duplex_bench(
        tmp_path,
        version="1.0",
        subsets=("synthetic_pause_handling",),
    )
    direct = discover_full_duplex_bench(
        tmp_path / "v1.0",
        version="1.0",
        subsets=("synthetic_pause_handling",),
    )
    assert parent == direct
    assert parent[0].inputs[0].variant == "input"


def test_v10_serving_replays_one_session_per_sample(tmp_path: Path):
    _v10_sample(tmp_path / "dataset", "synthetic_pause_handling", "1")
    args = _args(
        tmp_path,
        "--fdb-version",
        "1.0",
        "--fdb-subsets",
        "synthetic_pause_handling",
        "--num-prompts",
        "1",
    )
    preprocess_serve_args(args)
    requests = patch.get_samples(args, None)
    assert len(requests) == args.num_prompts == 1
    assert requests[0].source_id == "synthetic_pause_handling/1"
    assert requests[0].variant == "input"


def test_discovery_rejects_invalid_v15_metadata(tmp_path: Path):
    _v15_sample(tmp_path, "background_speech", "1", valid_metadata=False)
    with pytest.raises(ValueError, match=r"invalid Full-Duplex-Bench v1\.5 sample background_speech/1"):
        discover_full_duplex_bench(tmp_path, version="1.5", subsets=("background_speech",))


def test_v15_serving_selection_preserves_complete_pairs(tmp_path: Path):
    for name in ("1", "2", "3"):
        _v15_sample(tmp_path / "dataset", "user_interruption", name)
    args = _args(
        tmp_path,
        "--fdb-version",
        "1.5",
        "--fdb-subsets",
        "user_interruption",
        "--num-prompts",
        "4",
    )
    preprocess_serve_args(args)
    requests = patch.get_samples(args, None)
    assert len(requests) == args.num_prompts == 4
    assert [(request.source_id, request.variant) for request in requests] == [
        ("user_interruption/1", "input"),
        ("user_interruption/1", "clean_input"),
        ("user_interruption/2", "input"),
        ("user_interruption/2", "clean_input"),
    ]
    assert len({request.request_id for request in requests}) == 4
    assert requests[0].response_root.parent == tmp_path / "responses"
    assert args.max_concurrency == 1


def test_sglang_backend_is_accepted_for_full_duplex_bench(tmp_path: Path):
    args = _args(tmp_path, "--backend", "sglang-realtime-duplex")
    preprocess_serve_args(args)
    assert args.backend == "sglang-realtime-duplex"


def test_sglang_backend_rejects_server_vad(tmp_path: Path):
    args = _args(
        tmp_path,
        "--backend",
        "sglang-realtime-duplex",
        "--fdb-turn-detection",
        "server_vad",
    )
    with pytest.raises(ValueError, match="requires --fdb-turn-detection none"):
        preprocess_serve_args(args)


@pytest.mark.parametrize("count", ["1", "3"])
def test_v15_rejects_odd_num_prompts(tmp_path: Path, count: str):
    args = _args(tmp_path, "--num-prompts", count)
    with pytest.raises(ValueError, match="must be even"):
        preprocess_serve_args(args)


def test_prepared_audio_has_no_video_frames(tmp_path: Path):
    _v15_sample(tmp_path / "dataset", "user_backchannel", "1")
    args = _args(tmp_path, "--fdb-subsets", "user_backchannel")
    preprocess_serve_args(args)
    request = patch.get_samples(args, None)[0]
    prepared = adapter.prepare_full_duplex_bench_sample(request)
    assert prepared.frames == ()
    assert prepared.duration == pytest.approx(0.02)
    assert len(prepared.pcm) == 640


@pytest.mark.asyncio
async def test_native_replay_maps_metrics_and_publishes_artifacts(tmp_path: Path, monkeypatch):
    _v15_sample(tmp_path / "dataset", "talking_to_other", "1")
    args = _args(tmp_path, "--fdb-subsets", "talking_to_other")
    preprocess_serve_args(args)
    sample = patch.get_samples(args, None)[0]
    request = RequestFuncInput(
        model="mock",
        model_name="served-model",
        prompt="",
        api_url="http://127.0.0.1:8000/v1/realtime",
        prompt_len=0,
        output_len=0,
        request_id=sample.request_id,
    )
    adapter.attach_full_duplex_bench(sample, request)
    continuity = {
        "response_id": "response-1",
        "audio_underrun_s": 0.4,
        "audio_max_single_underrun_s": 0.3,
        "total_underrun_s": 0.4,
        "audio_continuity_ok": False,
        "audio_underrun_event_count": 2,
        "audio_duration_s": 0.6,
        "audio_underrun_ratio": 0.4,
        "audio_underrun_events_per_minute": 200.0,
        "audio_chunk_count": 3,
        "audio_continuity_threshold_s": 0.1,
    }
    output_pcm = b"\x01\x00" * 24_000

    async def generate_sample(dataset_sample, **kwargs):
        assert kwargs["prepared"].frames == ()
        return runner.GenerateSampleResult(
            output=sample.response_root / dataset_sample.split / f"{dataset_sample.id}.json",
            request_metrics=[
                {
                    "stage0_tokens": {"output_token_count": 2, "itls_ms": [10.0]},
                    "audio_continuity": continuity,
                }
            ],
            session_metrics={"ttft_ms": {"mean": 125.0}, "ttfp_ms": {"mean": 250.0}, "rtf": {"mean": 0.5}},
            timed_sentences=[{"sentence": "Hello", "start": 0.1, "end": 0.2}],
            metadata={"response_done": True},
            output_tokens=2,
            audio_bytes=48_000,
            audio_pcm=output_pcm,
        )

    monkeypatch.setattr(runner, "generate_sample", generate_sample)
    output = await patch._async_request_duplex_eval(request, pbar=None)
    assert output.success
    assert output.ttft == pytest.approx(0.125)
    assert output.audio_ttfp == pytest.approx(0.25)
    assert output.audio_rtf == pytest.approx(0.5)
    assert output.audio_continuity_samples == [continuity]
    metrics, _ = patch.calculate_metrics(
        input_requests=[sample],
        outputs=[output],
        dur_s=1.0,
        tokenizer=None,
        selected_percentiles=[50, 99],
        goodput_config_dict={},
        task_type=patch.TaskType.GENERATION,
        selected_percentile_metrics=["audio_underrun"],
        max_concurrency=1,
        request_rate=float("inf"),
        benchmark_duration=1.0,
    )
    assert metrics.num_audio_continuity_samples == 1
    assert metrics.total_audio_underrun_events == 2
    result = output.duplex_eval_result
    assert result.metadata["dataset"] == "full-duplex-bench"
    assert result.metadata["source_id"] == "talking_to_other/1"
    assert result.metadata["variant"] == "input"

    summary = adapter.finalize_full_duplex_bench([sample], [output])
    assert summary is not None
    assert summary["published"] == summary["sessions"] == 1
    assert result.output.is_file()
    with wave.open(str(result.output.with_suffix(".wav")), "rb") as output_audio:
        assert output_audio.getnchannels() == 1
        assert output_audio.getsampwidth() == 2
        assert output_audio.getframerate() == result.audio_sample_rate
        assert output_audio.readframes(output_audio.getnframes()) == output_pcm
    assert (sample.response_root / "duplex_metrics.json").is_file()
