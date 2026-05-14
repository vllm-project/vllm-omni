from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.accuracy.image_to_image.gedit_bench import GROUPS
from benchmarks.accuracy.image_to_image.gedit_bench import main as gedit_main
from tests.e2e.accuracy.helpers import infer_model_label, reset_artifact_dir
from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]


@pytest.mark.benchmark
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.skip(reason="#3257")
def test_gedit_bench_h100_smoke(
    gedit_accuracy_servers,
    accuracy_artifact_root: Path,
    gedit_dataset_root: Path,
    gedit_samples_per_group: int,
    accuracy_workers: int,
) -> None:
    model_label = infer_model_label(gedit_accuracy_servers.generate_params.model).lower()
    output_root = reset_artifact_dir(accuracy_artifact_root / f"gedit_results_{model_label}")
    score_root = reset_artifact_dir(accuracy_artifact_root / f"gedit_scores_{model_label}")
    model_name = model_label

    with gedit_accuracy_servers.generate_server() as generate_server:
        assert (
            gedit_main(
                [
                    "generate",
                    "--dataset-ref",
                    str(gedit_dataset_root),
                    "--output-root",
                    str(output_root),
                    "--base-url",
                    f"http://{generate_server.host}:{generate_server.port}",
                    "--model",
                    generate_server.model,
                    "--model-name",
                    model_name,
                    "--task-type",
                    "all",
                    "--instruction-language",
                    "all",
                    "--samples-per-group",
                    str(gedit_samples_per_group),
                    "--workers",
                    str(accuracy_workers),
                ]
            )
            == 0
        )

    with gedit_accuracy_servers.judge_server() as judge_server:
        assert (
            gedit_main(
                [
                    "evaluate",
                    "--dataset-ref",
                    str(gedit_dataset_root),
                    "--output-root",
                    str(output_root),
                    "--model-name",
                    model_name,
                    "--save-dir",
                    str(score_root),
                    "--task-type",
                    "all",
                    "--instruction-language",
                    "all",
                    "--judge-base-url",
                    f"http://{judge_server.host}:{judge_server.port}",
                    "--judge-model",
                    judge_server.model,
                    "--judge-api-key",
                    "EMPTY",
                    "--samples-per-group",
                    str(gedit_samples_per_group),
                    "--workers",
                    str(accuracy_workers),
                ]
            )
            == 0
        )

    csv_path = score_root / f"{model_name}_all_all_vie_score.csv"
    assert gedit_main(["summarize", "--csv-path", str(csv_path), "--language", "all"]) == 0

    summary_path = score_root / f"{model_name}_all_all_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert set(summary["languages"]) == {"en", "cn"}

    for language in ["en", "cn"]:
        language_summary = summary["languages"][language]
        assert language_summary["overall"]["count"] is not None
        assert language_summary["intersection"]["count"] is not None
        assert language_summary["overall"]["Q_SC"] is not None
        assert language_summary["overall"]["Q_PQ"] is not None
        assert language_summary["overall"]["Q_O"] is not None

        for group in GROUPS:
            group_summary = language_summary["by_group"][group]
            assert set(group_summary) == {"count", "Q_SC", "Q_PQ", "Q_O"}

    assert summary["languages"]["en"]["overall"]["Q_SC"] >= 6.95
    assert summary["languages"]["en"]["overall"]["Q_PQ"] >= 5.8
    assert summary["languages"]["en"]["overall"]["Q_O"] >= 6.15
    assert summary["languages"]["cn"]["overall"]["Q_SC"] >= 6.9
    assert summary["languages"]["cn"]["overall"]["Q_PQ"] >= 5.7
    assert summary["languages"]["cn"]["overall"]["Q_O"] >= 6.1
