import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _load_html_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "tools" / "nightly" / "generate_nightly_perf_html.py"
    spec = importlib.util.spec_from_file_location("generate_nightly_perf_html", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generate_html_report_with_perf_templates(tmp_path: Path):
    module = _load_html_module()
    repo_root = Path(__file__).resolve().parents[2]
    perf_scripts_dir = repo_root / "tests" / "dfx" / "perf" / "scripts"

    omni_template_path = perf_scripts_dir / "result_omni_template.json"
    diffusion_template_path = perf_scripts_dir / "diffusion_result_template.json"

    omni_record = json.loads(omni_template_path.read_text(encoding="utf-8"))
    diffusion_records = json.loads(diffusion_template_path.read_text(encoding="utf-8"))
    diffusion_records[0]["endpoint"] = "/v1/videos"
    diffusion_records[0]["result"]["endpoint"] = "/v1/videos"

    input_dir = tmp_path / "input"
    diffusion_input_dir = tmp_path / "diffusion_input"
    input_dir.mkdir()
    diffusion_input_dir.mkdir()

    omni_result_file = input_dir / "result_test_perf_random_1_4_in2500_out900_20260415-185642.json"
    diffusion_result_file = diffusion_input_dir / "diffusion_result_qwen_image_edit_20260415-193200.json"
    omni_result_file.write_text(json.dumps(omni_record, ensure_ascii=False, indent=2), encoding="utf-8")
    diffusion_result_file.write_text(json.dumps(diffusion_records, ensure_ascii=False, indent=2), encoding="utf-8")

    output_file = tmp_path / "nightly_perf_v2.html"
    module.generate_html_report(
        input_dir=str(input_dir),
        diffusion_input_dir=str(diffusion_input_dir),
        output_file=str(output_file),
    )

    assert output_file.exists()
    html = output_file.read_text(encoding="utf-8")
    assert "Nightly Performance Report" in html
    assert "Omni records <strong>1</strong>" in html
    assert f"Diffusion records <strong>{len(diffusion_records)}</strong>" in html
    assert "const DIFF_DATA =" in html
    assert '"endpoint": "/v1/videos"' in html


def test_get_sanitized_data_replaces_nonfinite_floats_with_none():
    module = _load_html_module()
    records = [
        {
            "mean_tpot_ms": float("nan"),
            "latency_mean": float("inf"),
            "latency_p99": 19_800.0,
            "completed": 8,
            "backend": "openai-chat-omni",
            "note": None,
        }
    ]

    sanitized = module.get_sanitized_data(records)

    assert sanitized[0]["mean_tpot_ms"] is None
    assert sanitized[0]["latency_mean"] is None
    assert sanitized[0]["latency_p99"] == pytest.approx(19_800.0)
    assert sanitized[0]["completed"] == 8
    assert sanitized[0]["backend"] == "openai-chat-omni"
    assert sanitized[0]["note"] is None
    # the input records are left untouched
    assert records[0]["mean_tpot_ms"] != records[0]["mean_tpot_ms"]
    # json.dumps with allow_nan=False is a strict serializer: it must reject
    # bare NaN/Infinity literals, so this only passes on sanitized data.
    json.dumps(sanitized, allow_nan=False)


def test_get_sanitized_data_handles_nested_containers_and_tuples():
    module = _load_html_module()
    records = [
        {
            "result": {
                "nested": {"latency_mean": float("inf")},
                "values": [1.5, float("nan")],
                "shape": (float("nan"), 3),
            }
        }
    ]

    sanitized = module.get_sanitized_data(records)

    result = sanitized[0]["result"]
    assert result["nested"]["latency_mean"] is None
    assert result["values"] == [1.5, None]
    assert result["shape"] == (None, 3)
    json.dumps(sanitized, allow_nan=False)


def test_generate_html_report_sanitizes_nonfinite_metric_values(tmp_path: Path):
    module = _load_html_module()
    input_dir = tmp_path / "input"
    diffusion_input_dir = tmp_path / "diffusion_input"
    input_dir.mkdir()
    diffusion_input_dir.mkdir()

    # Unmeasured metrics (e.g. TPOT with no multi-token samples, see #6693) are
    # stored as NaN and vLLM's json.dump persists them as bare ``NaN`` literals.
    omni_record = {
        "date": "20260830-130405",
        "backend": "openai-chat-omni",
        "model_id": "Qwen/Qwen3-Omni-Flash",
        "test_name": "qwen3_omni_chat",
        "dataset_name": "daily",
        "num_prompts": 8,
        "max_concurrency": 4,
        "duration": 60.0,
        "completed": 8,
        "failed": 0,
        "request_throughput": 0.13,
        "output_throughput": 6.6,
        "total_token_throughput": 19.8,
        "mean_ttft_ms": 251.9,
        "p99_ttft_ms": 402.3,
        "mean_tpot_ms": float("nan"),
        "median_tpot_ms": float("nan"),
        "p99_tpot_ms": float("nan"),
        "mean_itl_ms": float("nan"),
        "mean_e2el_ms": 3324.0,
        "p99_e2el_ms": 5100.7,
    }
    omni_result_file = input_dir / "result_test_qwen3_omni_chat_daily_8_4_20260830-130405.json"
    omni_result_file.write_text(json.dumps(omni_record), encoding="utf-8")
    assert "NaN" in omni_result_file.read_text(encoding="utf-8")

    diffusion_record = {
        "date": "20260830-130405",
        "test_name": "qwen_image_edit",
        "model": "Qwen/Qwen-Image-Edit",
        "endpoint": "/v1/images/edits",
        "dataset": "seed-edit",
        "task": "image-edit",
        "duration": 120.0,
        "throughput_qps": 0.08,
        "latency_mean": float("inf"),
        "latency_median": 12_500.0,
        "latency_p50": 12_500.0,
        "latency_p99": 19_800.0,
        "completed_requests": 10,
        "failed_requests": 0,
        "slo_attainment_rate": 1.0,
        "result": {"endpoint": "/v1/images/edits", "nested": {"latency_mean": float("inf")}},
    }
    diffusion_result_file = diffusion_input_dir / "diffusion_result_qwen_image_edit_20260830-130405.json"
    diffusion_result_file.write_text(json.dumps([diffusion_record]), encoding="utf-8")
    assert "Infinity" in diffusion_result_file.read_text(encoding="utf-8")

    output_file = tmp_path / "nightly_perf_v2.html"
    module.generate_html_report(
        input_dir=str(input_dir),
        diffusion_input_dir=str(diffusion_input_dir),
        output_file=str(output_file),
    )

    assert output_file.exists()
    html = output_file.read_text(encoding="utf-8")

    # the embedded data lines must not carry bare non-finite literals; the
    # JS template itself legitimately uses Infinity, so scope to data lines.
    data_lines = [
        line for line in html.splitlines() if line.startswith("const OMNI_DATA") or line.startswith("const DIFF_DATA")
    ]
    assert data_lines, "embedded data lines not found"
    for line in data_lines:
        assert "NaN" not in line
        assert "Infinity" not in line
