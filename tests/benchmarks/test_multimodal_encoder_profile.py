from argparse import Namespace
from pathlib import Path

import pytest

from benchmarks.profiling.benchmark_multimodal_encoder import (
    MemorySample,
    _latency_summary,
    build_payload,
    summarize_memory,
    update_stage_metrics,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_latency_summary_includes_sample_variation() -> None:
    summary = _latency_summary([1.0, 3.0])

    assert summary == {
        "mean": 2.0,
        "stdev": pytest.approx(2**0.5),
        "median": 2.0,
        "p90": 3.0,
        "p99": 3.0,
        "min": 1.0,
        "max": 3.0,
    }


def test_build_payload_embeds_repeated_image_assets(tmp_path: Path) -> None:
    image = tmp_path / "tiny.png"
    image.write_bytes(b"not-a-real-png")
    args = Namespace(
        modality="image",
        asset=[str(image), str(image)],
        prompt="What is shown?",
        model="test-model",
        max_tokens=8,
    )

    payload = build_payload(args)

    content = payload["messages"][1]["content"]
    assert [item["type"] for item in content] == ["image_url", "image_url", "text"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert payload["modalities"] == ["text"]
    assert payload["return_token_ids"] is True
    assert payload["return_stage_metrics"] is True


def test_update_stage_metrics_keeps_latest_snapshot_per_stage() -> None:
    stage_metrics = {"0": {"num_tokens_out": 1}}

    update_stage_metrics(
        stage_metrics,
        {
            "metrics": {
                "stage_metrics": {
                    "0": {"num_tokens_out": 2, "stage_name": "thinker"},
                    1: {"num_tokens_out": 3, "stage_name": "talker"},
                    "invalid": "ignored",
                }
            }
        },
    )

    assert stage_metrics == {
        "0": {"num_tokens_out": 2, "stage_name": "thinker"},
        "1": {"num_tokens_out": 3, "stage_name": "talker"},
    }


def test_summarize_memory_splits_baseline_and_request_peak() -> None:
    samples = [
        MemorySample(1.0, {"gpu-0": 100.0}, {"gpu-0": {"10": 80.0, "12": 2.0}}),
        MemorySample(2.0, {"gpu-0": 110.0}, {"gpu-0": {"10": 90.0}}),
        MemorySample(3.0, {"gpu-0": 145.0}, {"gpu-0": {"10": 125.0, "11": 5.0}}),
        MemorySample(4.0, {"gpu-0": 130.0}, {"gpu-0": {"10": 110.0}}),
    ]

    summary = summarize_memory(samples, request_started_s=2.5, request_ended_s=4.0)

    assert summary["gpu-0"] == {
        "baseline_memory_mib": 110.0,
        "request_peak_memory_mib": 145.0,
        "request_peak_delta_mib": 35.0,
        "process_baseline_memory_mib": {"10": 90.0},
        "process_peak_memory_mib": {"10": 125.0, "11": 5.0},
        "process_peak_delta_mib": {"10": 35.0, "11": None},
    }
