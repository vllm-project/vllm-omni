# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "benchmarks" / "tts"))

import bench_tts  # noqa: E402

_CONFIG = _REPO_ROOT / "benchmarks" / "tts" / "model_configs.yaml"
pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]


def test_llama_omni2_uses_openai_chat_omni_backend() -> None:
    config = bench_tts.load_model_configs(_CONFIG)["ICTNLP/LLaMA-Omni2-0.5B"]

    command = bench_tts.build_bench_args(
        host="127.0.0.1",
        port=8000,
        model="ICTNLP/LLaMA-Omni2-0.5B",
        task="speech_to_speech",
        model_cfg=config,
        locale="en",
        num_prompts=8,
        concurrency=4,
        dataset_path="/data/fixed.jsonl",
        wer_eval=False,
        output_dir="/tmp/results",
        result_filename="run.json",
        extra_cli_args=[],
    )

    assert command[command.index("--backend") + 1] == "openai-chat-omni"
    assert command[command.index("--endpoint") + 1] == "/v1/chat/completions"
    assert command[command.index("--dataset-name") + 1] == "llama-omni2-s2s"
    assert json.loads(command[command.index("--extra-body") + 1]) == {
        "modalities": ["text", "audio"],
        "stream": True,
    }


class _Tokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


def test_llama_omni2_dataset_builds_fixed_audio_chat_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_omni.benchmarks.data_modules import llama_omni2_s2s_dataset

    monkeypatch.setattr(
        llama_omni2_s2s_dataset,
        "get_cached_tokenizer",
        lambda tokenizer: tokenizer,
    )

    first_audio = tmp_path / "first.wav"
    second_audio = tmp_path / "second.wav"
    first_audio.write_bytes(b"RIFF-first")
    second_audio.write_bytes(b"RIFF-second")
    dataset_path = tmp_path / "fixed.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "sample-0001",
                        "audio": str(first_audio),
                        "text": "Respond to the first speaker.",
                    }
                ),
                json.dumps(
                    {
                        "id": "sample-0002",
                        "audio": str(second_audio),
                        "text": "Respond to the second speaker.",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    requests = llama_omni2_s2s_dataset.LlamaOmni2S2SDataset(
        dataset_path=str(dataset_path),
        disable_shuffle=True,
    ).sample(
        tokenizer=_Tokenizer(),
        num_requests=2,
        output_len=128,
        request_id_prefix="bench-",
        no_oversample=True,
    )

    assert [request.request_id for request in requests] == [
        "bench-sample-0001",
        "bench-sample-0002",
    ]
    assert requests[0].expected_output_len == 128
    assert requests[0].prompt == "Respond to the first speaker."
    messages = requests[0].llama_omni2_chat_messages
    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0]["type"] == "audio_url"
    assert messages[0]["content"][0]["audio_url"]["url"].startswith("data:audio/wav;base64,")
    assert messages[0]["content"][1] == {
        "type": "text",
        "text": "Respond to the first speaker.",
    }


def test_llama_omni2_dataset_rejects_relative_audio_paths(tmp_path: Path) -> None:
    from vllm_omni.benchmarks.data_modules.llama_omni2_s2s_dataset import (
        LlamaOmni2S2SDataset,
    )

    dataset_path = tmp_path / "fixed.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "sample-0001",
                "audio": "relative.wav",
                "text": "Respond to the speaker.",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="absolute"):
        LlamaOmni2S2SDataset(dataset_path=str(dataset_path))


def _write_result(
    path: Path,
    *,
    ttfp_ms: float,
    rtf: float,
    throughput: float,
    num_prompts: int = 8,
    completed: int = 8,
    failed: int = 0,
    total_audio_duration_s: float = 12.0,
    median_audio_duration_s: float = 1.5,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "num_prompts": num_prompts,
                "completed": completed,
                "failed": failed,
                "median_audio_ttfp_ms": ttfp_ms,
                "median_audio_rtf": rtf,
                "median_audio_duration_s": median_audio_duration_s,
                "total_audio_duration_s": total_audio_duration_s,
                "audio_throughput": throughput,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_three_run_summary_reports_dispersion_and_relative_change(
    tmp_path: Path,
) -> None:
    import summarize_llama_omni2_runs as summarize

    before = [
        _write_result(
            tmp_path / f"before-{index}.json",
            ttfp_ms=ttfp,
            rtf=rtf,
            throughput=throughput,
        )
        for index, (ttfp, rtf, throughput) in enumerate(
            [
                (100.0, 0.50, 1.0),
                (110.0, 0.55, 1.1),
                (90.0, 0.45, 0.9),
            ],
            start=1,
        )
    ]
    after = [
        _write_result(
            tmp_path / f"after-{index}.json",
            ttfp_ms=ttfp,
            rtf=rtf,
            throughput=throughput,
        )
        for index, (ttfp, rtf, throughput) in enumerate(
            [
                (102.0, 0.46, 1.2),
                (104.0, 0.44, 1.3),
                (100.0, 0.45, 1.1),
            ],
            start=1,
        )
    ]

    summary = summarize.summarize_comparison(
        before_paths=before,
        after_paths=after,
        label="c4",
    )

    ttfp = summary["metrics"]["median_audio_ttfp_ms"]
    assert ttfp["before"]["median"] == 100.0
    assert ttfp["before"]["stdev"] == pytest.approx(statistics.stdev([100.0, 110.0, 90.0]))
    assert ttfp["after"] == {
        "median": 102.0,
        "stdev": 2.0,
        "minimum": 100.0,
        "maximum": 104.0,
    }
    assert ttfp["relative_change_percent"] == pytest.approx(2.0)
    assert summary["metrics"]["audio_throughput"]["relative_change_percent"] == pytest.approx(20.0)


def test_three_run_summary_rejects_failed_requests(tmp_path: Path) -> None:
    import summarize_llama_omni2_runs as summarize

    paths = [
        _write_result(
            tmp_path / f"run-{index}.json",
            ttfp_ms=100.0,
            rtf=0.5,
            throughput=1.0,
            completed=7 if index == 2 else 8,
            failed=1 if index == 2 else 0,
        )
        for index in range(1, 4)
    ]

    with pytest.raises(ValueError, match=r"run-2\.json.*failed=1"):
        summarize.summarize_comparison(
            before_paths=paths,
            after_paths=paths,
            label="c4",
        )


def test_three_run_summary_rejects_zero_audio_output(tmp_path: Path) -> None:
    import summarize_llama_omni2_runs as summarize

    paths = [
        _write_result(
            tmp_path / f"run-{index}.json",
            ttfp_ms=100.0,
            rtf=0.5,
            throughput=1.0,
            total_audio_duration_s=0.0 if index == 3 else 12.0,
        )
        for index in range(1, 4)
    ]

    with pytest.raises(
        ValueError,
        match=r"run-3\.json.*positive total_audio_duration_s",
    ):
        summarize.summarize_comparison(
            before_paths=paths,
            after_paths=paths,
            label="c8",
        )


def test_gate_rejects_c1_latency_regression_over_five_percent() -> None:
    import summarize_llama_omni2_runs as summarize

    comparisons = {
        "c1": {
            "metrics": {
                "median_audio_ttfp_ms": {"relative_change_percent": 5.1},
                "median_audio_rtf": {"relative_change_percent": 0.0},
                "audio_throughput": {"relative_change_percent": 0.0},
            }
        }
    }

    passed, reasons = summarize.evaluate_gate(comparisons)

    assert passed is False
    assert any("c1" in reason and "TTFP" in reason for reason in reasons)


def test_gate_accepts_when_c8_throughput_improves_ten_percent() -> None:
    import summarize_llama_omni2_runs as summarize

    comparisons = {
        "c1": {
            "metrics": {
                "median_audio_ttfp_ms": {"relative_change_percent": 2.0},
                "median_audio_rtf": {"relative_change_percent": 1.0},
                "audio_throughput": {"relative_change_percent": -1.0},
            }
        },
        "c4": {
            "metrics": {
                "median_audio_ttfp_ms": {"relative_change_percent": 20.0},
                "median_audio_rtf": {"relative_change_percent": -4.0},
                "audio_throughput": {"relative_change_percent": 8.0},
            }
        },
        "c8": {
            "metrics": {
                "median_audio_ttfp_ms": {"relative_change_percent": 30.0},
                "median_audio_rtf": {"relative_change_percent": -7.0},
                "audio_throughput": {"relative_change_percent": 10.0},
            }
        },
    }

    passed, reasons = summarize.evaluate_gate(comparisons)

    assert passed is True
    assert reasons == []


def test_gate_accepts_computed_ten_percent_boundary(tmp_path: Path) -> None:
    import summarize_llama_omni2_runs as summarize

    before = [
        _write_result(
            tmp_path / f"before-{index}.json",
            ttfp_ms=300.0,
            rtf=1.0,
            throughput=3.0,
        )
        for index in range(3)
    ]
    after = [
        _write_result(
            tmp_path / f"after-{index}.json",
            ttfp_ms=300.0,
            rtf=1.0,
            throughput=3.3,
        )
        for index in range(3)
    ]
    comparisons = {
        "c8": summarize.summarize_comparison(
            before_paths=before,
            after_paths=after,
            label="c8",
        )
    }

    passed, reasons = summarize.evaluate_gate(comparisons)

    assert passed is True
    assert reasons == []
