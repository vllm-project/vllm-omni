from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.accuracy.text_to_image.gbench import main as gbench_main
from tests.e2e.accuracy.helpers import infer_model_label, reset_artifact_dir
from tests.helpers.mark import hardware_test


@pytest.mark.advanced_model
@pytest.mark.benchmark
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_gebench_h100_smoke(
    gebench_accuracy_servers,
    accuracy_artifact_root: Path,
    gebench_dataset_root: Path,
    accuracy_workers: int,
) -> None:
    model_label = infer_model_label(gebench_accuracy_servers.generate_params.model).lower()
    output_root = reset_artifact_dir(accuracy_artifact_root / f"gebench_{model_label}")

    with gebench_accuracy_servers.generate_server() as generate_server:
        for data_type in ("type3", "type4"):
            assert (
                gbench_main(
                    [
                        "generate",
                        "--dataset-root",
                        str(gebench_dataset_root),
                        "--output-root",
                        str(output_root),
                        "--base-url",
                        f"http://{generate_server.host}:{generate_server.port}",
                        "--model",
                        generate_server.model,
                        "--data-type",
                        data_type,
                        "--width",
                        "768",
                        "--height",
                        "576",
                        "--output-compression",
                        "98",
                        "--num-inference-steps",
                        "8",
                        "--workers",
                        str(accuracy_workers),
                    ]
                )
                == 0
            )

    with gebench_accuracy_servers.judge_server() as judge_server:
        for data_type in ("type3", "type4"):
            assert (
                gbench_main(
                    [
                        "evaluate",
                        "--dataset-root",
                        str(gebench_dataset_root),
                        "--output-root",
                        str(output_root),
                        "--data-type",
                        data_type,
                        "--judge-base-url",
                        f"http://{judge_server.host}:{judge_server.port}",
                        "--judge-model",
                        judge_server.model,
                        "--judge-api-key",
                        "EMPTY",
                        "--workers",
                        str(accuracy_workers),
                    ]
                )
                == 0
            )

    assert gbench_main(["summarize", "--output-root", str(output_root)]) == 0

    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert "generation" in summary
    assert "evaluation" in summary

    for data_type in ("type3", "type4"):
        assert data_type in summary["generation"]["by_type"]
        assert summary["generation"]["by_type"][data_type]["count"] > 0
        assert data_type in summary["evaluation"]["by_type"]
        assert summary["evaluation"]["by_type"][data_type]["count"] > 0

    assert summary["evaluation"]["overall_mean"] >= 0.45
    assert summary["evaluation"]["by_type"]["type3"]["overall_mean"] >= 0.45
    assert summary["evaluation"]["by_type"]["type4"]["overall_mean"] >= 0.45
