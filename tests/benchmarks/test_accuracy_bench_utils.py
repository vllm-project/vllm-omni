# ruff: noqa: E402, I001
import math
import sys
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.accuracy.common import VllmOmniImageClient
from benchmarks.accuracy.image_to_image.gedit_bench import (
    GROUPS as GEDIT_GROUPS,
    GEditBenchEvaluator,
    GEditBenchRunner,
    _load_gedit_dataset,
    _resolve_gedit_split,
    infer_model_name,
    resolve_model_name,
    select_balanced_gedit_rows,
    parse_score_payload,
    summarize_generated_records as summarize_gedit_generated_records,
    summarize_gedit_rows,
    summarize_gedit_rows_with_backbone,
)
from benchmarks.accuracy.text_to_image.gbench import (
    _expand_sample_path,
    _trajectory_judge_payload,
    _write_json_with_timestamp,
    LocalJudgeClient,
    GEBenchEvaluator,
    TYPE_TO_FOLDER,
    select_balanced_gebench_samples,
    summarize_generated_records as summarize_gebench_generated_records,
    summarize_gebench_results,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_summarize_gebench_generated_records_groups_by_type():
    records = [
        {"data_type": "type1", "sample_name": "english_phone/folder_1", "output_path": "a.png"},
        {"data_type": "type1", "sample_name": "english_phone/folder_2", "output_path": "b.png"},
        {"data_type": "type2", "sample_name": "english_phone/folder_3", "output_path": "c.png"},
    ]

    summary = summarize_gebench_generated_records(records)

    assert summary["count"] == 3
    assert summary["by_type"]["type1"]["count"] == 2
    assert summary["by_type"]["type2"]["count"] == 1
    assert "samples" not in summary["by_type"]["type1"]


def test_summarize_gebench_results_computes_type_and_global_means():
    results = [
        {"data_type": "type1", "overall": 0.8, "scores": {"goal": 5, "logic": 4}},
        {"data_type": "type1", "overall": 0.6, "scores": {"goal": 3, "logic": 4}},
        {"data_type": "type2", "overall": 0.5, "scores": {"goal": 2, "logic": 3}},
    ]

    summary = summarize_gebench_results(results)

    assert math.isclose(summary["overall_mean"], (0.8 + 0.6 + 0.5) / 3)
    assert math.isclose(summary["by_type"]["type1"]["overall_mean"], 0.7)
    assert math.isclose(summary["by_type"]["type2"]["overall_mean"], 0.5)
    assert math.isclose(summary["by_type"]["type1"]["score_means"]["goal"], 4.0)


def test_write_json_with_timestamp_writes_stable_and_timestamped_files(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "benchmarks.accuracy.text_to_image.gbench._utc_timestamp",
        lambda: "20260325T130000Z",
    )

    timestamped_path = _write_json_with_timestamp(tmp_path / "summary.json", {"ok": True})

    assert (tmp_path / "summary.json").exists()
    assert timestamped_path == tmp_path / "summary_20260325T130000Z.json"
    assert timestamped_path.exists()


def test_select_balanced_gebench_samples_limits_each_type_independently():
    sample_paths_by_type = {
        "type1": [Path(f"/tmp/type1_{idx}") for idx in range(12)],
        "type2": [Path(f"/tmp/type2_{idx}") for idx in range(8)],
        "type3": [Path(f"/tmp/type3_{idx}") for idx in range(15)],
    }

    selected = select_balanced_gebench_samples(sample_paths_by_type, samples_per_type=10)

    assert len(selected["type1"]) == 10
    assert len(selected["type2"]) == 8
    assert len(selected["type3"]) == 10
    assert selected["type1"][0].name == "type1_0"
    assert selected["type3"][-1].name == "type3_9"


def test_expand_sample_path_flattens_json_list_samples(tmp_path: Path):
    sample_path = tmp_path / "trajectories.json"
    sample_path.write_text(
        """
[
  {"id": "sample_a", "lang_device": "english_phone", "instruction": "do a"},
  {"id": "sample_b", "lang_device": "english_phone", "instruction": "do b"}
]
""".strip(),
        encoding="utf-8",
    )

    specs = _expand_sample_path(sample_path)

    assert len(specs) == 2
    assert specs[0].sample_name == "sample_a"
    assert specs[1].sample_name == "sample_b"
    assert specs[0].lang_device == "english_phone"


def test_gebench_evaluate_skips_missing_output_folder(tmp_path: Path):
    dataset_type_root = tmp_path / TYPE_TO_FOLDER["type3"] / "english_phone"
    sample_dir = dataset_type_root / "sample_a"
    sample_dir.mkdir(parents=True)
    (sample_dir / "meta_data.json").write_text("{}", encoding="utf-8")

    judge = LocalJudgeClient(base_url="http://127.0.0.1:8094", api_key="EMPTY", model="judge")
    evaluator = GEBenchEvaluator(dataset_root=tmp_path, output_root=tmp_path / "outputs", judge=judge)

    payload = evaluator.evaluate(data_type="type3")

    assert payload["results"] == []
    assert payload["summary"]["count"] == 0


def test_local_judge_client_retries_when_first_response_is_not_json(monkeypatch):
    responses = iter(
        [
            "The image looks like a GUI screenshot with several controls.",
            '{"goal": 4, "logic": 4, "cons": 5, "ui": 4, "qual": 4, "reasoning": "mostly correct"}',
        ]
    )

    def fake_request_text(self, prompt, images):
        return next(responses)

    monkeypatch.setattr(LocalJudgeClient, "_request_text", fake_request_text)

    judge = LocalJudgeClient(base_url="http://127.0.0.1:8094", api_key="EMPTY", model="judge")
    result = judge.evaluate(prompt="Evaluate this GUI trajectory.", images=[Image.new("RGB", (2, 2), color="white")])

    assert result["goal"] == 4
    assert result["cons"] == 5


def test_local_judge_client_returns_zero_scores_when_retry_is_still_invalid(monkeypatch):
    responses = iter(
        [
            "not json",
            "still not json",
        ]
    )

    def fake_request_text(self, prompt, images):
        return next(responses)

    monkeypatch.setattr(LocalJudgeClient, "_request_text", fake_request_text)

    judge = LocalJudgeClient(base_url="http://127.0.0.1:8094", api_key="EMPTY", model="judge")
    result = judge.evaluate(prompt="Evaluate this GUI trajectory.", images=[Image.new("RGB", (2, 2), color="white")])

    assert result["goal"] == 0
    assert result["logic"] == 0
    assert result["cons"] == 0
    assert result["ui"] == 0
    assert result["qual"] == 0
    assert result["reasoning"] == "still not json"


def test_trajectory_judge_payload_collapses_six_frames_into_single_storyboard():
    frames = [Image.new("RGB", (8, 6), color=(idx * 10, idx * 10, idx * 10)) for idx in range(6)]

    prompt_suffix, judge_images = _trajectory_judge_payload(frames)

    assert "frame0" in prompt_suffix
    assert len(judge_images) == 1
    assert judge_images[0].size == (24, 12)


def test_image_edit_client_uses_openai_image_edit_endpoint(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [
                    {
                        "b64_json": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aY0cAAAAASUVORK5CYII="
                    }
                ]
            }

    def fake_post(url, data=None, files=None, headers=None, timeout=None, **kwargs):
        captured["url"] = url
        captured["data"] = data
        captured["files"] = files
        captured["headers"] = headers
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("benchmarks.accuracy.common.requests.post", fake_post)

    client = VllmOmniImageClient(base_url="http://127.0.0.1:8093", api_key="EMPTY")
    image = Image.new("RGB", (2, 2), color="white")
    output = client.generate_image_edit(
        model="Qwen/Qwen-Image-Edit",
        prompt="edit this image",
        images=image,
        width=512,
        height=512,
    )

    assert output.size == (1, 1)
    assert captured["url"] == "http://127.0.0.1:8093/v1/images/edits"
    assert captured["data"]["prompt"] == "edit this image"
    assert captured["data"]["size"] == "512x512"
    assert captured["files"][0][0] == "image"


def test_text_to_image_client_forwards_output_compression(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [
                    {
                        "b64_json": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aY0cAAAAASUVORK5CYII="
                    }
                ]
            }

    def fake_post(url, json=None, headers=None, timeout=None, **kwargs):
        captured["url"] = url
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr("benchmarks.accuracy.common.requests.post", fake_post)

    client = VllmOmniImageClient(base_url="http://127.0.0.1:8093", api_key="EMPTY")
    output = client.generate_text_to_image(
        model="Qwen/Qwen-Image",
        prompt="generate a gui",
        width=768,
        height=576,
        num_inference_steps=8,
        output_compression=98,
    )

    assert output.size == (1, 1)
    assert captured["url"] == "http://127.0.0.1:8093/v1/images/generations"
    assert captured["json"]["size"] == "768x576"
    assert captured["json"]["num_inference_steps"] == 8
    assert captured["json"]["output_compression"] == 98


def test_parse_score_payload_handles_raw_json_and_delimited_json():
    raw = '{"score": [7, 8], "reasoning": "ok"}'
    wrapped = 'prefix ||V^=^V|| {"score": [6], "reasoning": "fine"} ||V^=^V|| suffix'

    assert parse_score_payload(raw)["score"] == [7, 8]
    assert parse_score_payload(wrapped)["score"] == [6]


def test_summarize_gedit_generated_records_groups_by_task_and_language():
    records = []
    for group in GEDIT_GROUPS[:2]:
        records.append(
            {
                "task_type": group,
                "instruction_language": "en",
                "key": f"{group}_en",
                "output_path": f"{group}_en.png",
            }
        )
        records.append(
            {
                "task_type": group,
                "instruction_language": "cn",
                "key": f"{group}_cn",
                "output_path": f"{group}_cn.png",
            }
        )

    summary = summarize_gedit_generated_records(records)

    assert summary["count"] == 4
    assert summary["by_task"][GEDIT_GROUPS[0]]["count"] == 2
    assert summary["by_language"]["en"]["count"] == 2
    assert summary["by_language"]["cn"]["samples"] == [
        f"{GEDIT_GROUPS[0]}_cn",
        f"{GEDIT_GROUPS[1]}_cn",
    ]


def test_select_balanced_gedit_rows_limits_each_group_independently():
    rows = []
    for idx in range(12):
        rows.append(
            {
                "task_type": "background_change",
                "instruction_language": "en",
                "key": f"background_change_{idx}",
            }
        )
    for idx in range(7):
        rows.append(
            {
                "task_type": "color_alter",
                "instruction_language": "en",
                "key": f"color_alter_{idx}",
            }
        )

    selected = select_balanced_gedit_rows(
        rows,
        task_type="all",
        instruction_language="en",
        samples_per_group=10,
    )

    selected_background = [row for row in selected if row["task_type"] == "background_change"]
    selected_color = [row for row in selected if row["task_type"] == "color_alter"]

    assert len(selected_background) == 10
    assert len(selected_color) == 7
    assert selected_background[0]["key"] == "background_change_0"
    assert selected_background[-1]["key"] == "background_change_9"


def test_select_balanced_gedit_rows_balances_languages_when_all_requested():
    rows = []
    for idx in range(10):
        rows.append(
            {
                "task_type": "background_change",
                "instruction_language": "cn",
                "key": f"background_change_cn_{idx}",
            }
        )
    for idx in range(10):
        rows.append(
            {
                "task_type": "background_change",
                "instruction_language": "en",
                "key": f"background_change_en_{idx}",
            }
        )

    selected = select_balanced_gedit_rows(
        rows,
        task_type="all",
        instruction_language="all",
        samples_per_group=10,
    )

    selected_background = [row for row in selected if row["task_type"] == "background_change"]

    assert len(selected_background) == 10
    assert sum(1 for row in selected_background if row["instruction_language"] == "en") == 5
    assert sum(1 for row in selected_background if row["instruction_language"] == "cn") == 5


def test_infer_model_name_uses_last_path_segment():
    assert infer_model_name("/workspace/models/Qwen/Qwen-Image-Edit") == "Qwen-Image-Edit"


def test_resolve_model_name_prefers_explicit_value_then_model_then_output_root(tmp_path: Path):
    assert (
        resolve_model_name(
            model_name="explicit_name",
            model="/workspace/models/Qwen/Qwen-Image-Edit",
        )
        == "explicit_name"
    )
    assert (
        resolve_model_name(
            model_name=None,
            model="/workspace/models/Qwen/Qwen-Image-Edit",
        )
        == "Qwen-Image-Edit"
    )

    output_root = tmp_path / "results"
    (output_root / "qwen_image_edit").mkdir(parents=True)
    assert resolve_model_name(model_name=None, output_root=output_root) == "qwen_image_edit"


def test_resolve_gedit_split_accepts_dataset_dict_like_input():
    train_rows = [{"key": "a"}]
    dataset = {"train": train_rows}

    assert _resolve_gedit_split(dataset) == train_rows


def test_resolve_gedit_split_accepts_dataset_like_input():
    rows = [{"key": "a"}]

    assert _resolve_gedit_split(rows) == rows


def test_load_gedit_dataset_uses_load_from_disk_for_saved_dataset(monkeypatch, tmp_path: Path):
    (tmp_path / "state.json").write_text("{}", encoding="utf-8")
    (tmp_path / "dataset_info.json").write_text("{}", encoding="utf-8")
    captured = {}

    def fake_load_dataset(path):
        captured["load_dataset"] = path
        return "load_dataset"

    def fake_load_from_disk(path):
        captured["load_from_disk"] = path
        return "load_from_disk"

    monkeypatch.setattr(
        "benchmarks.accuracy.image_to_image.gedit_bench._require_datasets",
        lambda: (fake_load_dataset, fake_load_from_disk),
    )

    result = _load_gedit_dataset(str(tmp_path))

    assert result == "load_from_disk"
    assert captured["load_from_disk"] == str(tmp_path)
    assert "load_dataset" not in captured


def test_load_gedit_dataset_uses_load_dataset_for_local_snapshot_path(monkeypatch, tmp_path: Path):
    (tmp_path / "README.md").write_text("dataset repo snapshot", encoding="utf-8")
    captured = {}

    def fake_load_dataset(path):
        captured["load_dataset"] = path
        return "load_dataset"

    def fake_load_from_disk(path):
        captured["load_from_disk"] = path
        return "load_from_disk"

    monkeypatch.setattr(
        "benchmarks.accuracy.image_to_image.gedit_bench._require_datasets",
        lambda: (fake_load_dataset, fake_load_from_disk),
    )

    result = _load_gedit_dataset(str(tmp_path))

    assert result == "load_dataset"
    assert captured["load_dataset"] == str(tmp_path)
    assert "load_from_disk" not in captured


def test_gedit_runner_generate_skips_failed_samples(monkeypatch, tmp_path: Path):
    rows = [
        {"key": "ok", "task_type": "background_change", "instruction_language": "en"},
        {"key": "bad", "task_type": "background_change", "instruction_language": "en"},
    ]

    monkeypatch.setattr("benchmarks.accuracy.image_to_image.gedit_bench._load_gedit_dataset", lambda ref: rows)
    runner = GEditBenchRunner(
        dataset_ref="dataset",
        output_root=tmp_path,
        base_url="http://127.0.0.1:8093",
        model="model",
    )

    def fake_generate_one(self, model_name, item):
        if item["key"] == "bad":
            raise RuntimeError("boom")
        return {
            "key": item["key"],
            "task_type": item["task_type"],
            "instruction_language": item["instruction_language"],
        }

    monkeypatch.setattr(GEditBenchRunner, "_generate_one", fake_generate_one)

    outputs = runner.generate(model_name="demo", workers=1)

    assert outputs == [{"key": "ok", "task_type": "background_change", "instruction_language": "en"}]


def test_gedit_runner_uses_tqdm_progress(monkeypatch, tmp_path: Path):
    rows = [
        {"key": "one", "task_type": "background_change", "instruction_language": "en"},
        {"key": "two", "task_type": "background_change", "instruction_language": "en"},
    ]
    updates = []

    monkeypatch.setattr("benchmarks.accuracy.image_to_image.gedit_bench._load_gedit_dataset", lambda ref: rows)
    runner = GEditBenchRunner(
        dataset_ref="dataset",
        output_root=tmp_path,
        base_url="http://127.0.0.1:8093",
        model="model",
    )

    def fake_generate_one(self, model_name, item):
        return {
            "key": item["key"],
            "task_type": item["task_type"],
            "instruction_language": item["instruction_language"],
        }

    monkeypatch.setattr(GEditBenchRunner, "_generate_one", fake_generate_one)

    class FakeTqdm:
        def __init__(self, total, desc, unit):
            self.total = total
            self.desc = desc
            self.unit = unit

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, value):
            updates.append(value)

    monkeypatch.setattr("benchmarks.accuracy.image_to_image.gedit_bench.tqdm", FakeTqdm)

    runner.generate(model_name="demo", workers=1)

    assert updates == [1, 1]


def test_gedit_evaluator_skips_failed_samples(monkeypatch, tmp_path: Path):
    rows = [
        {"key": "ok", "task_type": "background_change", "instruction_language": "en"},
        {"key": "bad", "task_type": "background_change", "instruction_language": "en"},
    ]

    monkeypatch.setattr("benchmarks.accuracy.image_to_image.gedit_bench._load_gedit_dataset", lambda ref: rows)
    evaluator = GEditBenchEvaluator(dataset_ref="dataset", output_root=tmp_path / "results", scorer=object())

    def fake_evaluate_one(self, model_name, item):
        if item["key"] == "bad":
            raise RuntimeError("boom")
        return {
            "key": item["key"],
            "task_type": item["task_type"],
            "edited_image": "ok.png",
            "instruction": "edit",
            "semantics_score": 8.0,
            "quality_score": 7.0,
            "overall_score": math.sqrt(56.0),
            "intersection_exist": True,
            "instruction_language": item["instruction_language"],
        }

    monkeypatch.setattr(GEditBenchEvaluator, "_evaluate_one", fake_evaluate_one)
    monkeypatch.setattr(
        "benchmarks.accuracy.image_to_image.gedit_bench._utc_timestamp",
        lambda: "20260325T120000Z",
    )

    payload = evaluator.evaluate(
        model_name="demo",
        save_dir=tmp_path / "scores",
        instruction_language="en",
        workers=1,
    )

    assert len(payload["results"]) == 1
    assert payload["results"][0]["key"] == "ok"
    assert payload["summary"]["overall"]["count"] == 1
    assert Path(payload["csv_path"]).name == "demo_all_en_vie_score.csv"
    assert Path(payload["summary_path"]).name == "demo_all_en_summary.json"
    assert Path(payload["timestamped_csv_path"]).name == "demo_all_en_vie_score_20260325T120000Z.csv"
    assert Path(payload["timestamped_summary_path"]).name == "demo_all_en_summary_20260325T120000Z.json"
    assert Path(payload["timestamped_csv_path"]).exists()
    assert Path(payload["timestamped_summary_path"]).exists()


def test_summarize_gedit_rows_computes_group_and_intersection_means():
    rows = []
    for group in GEDIT_GROUPS:
        rows.append(
            {
                "task_type": group,
                "instruction_language": "en",
                "semantics_score": 8.0,
                "quality_score": 9.0,
                "intersection_exist": True,
            }
        )
        rows.append(
            {
                "task_type": group,
                "instruction_language": "en",
                "semantics_score": 6.0,
                "quality_score": 4.0,
                "intersection_exist": False,
            }
        )

    summary = summarize_gedit_rows(rows, language="en")

    expected_overall = (math.sqrt(8.0 * 9.0) + math.sqrt(6.0 * 4.0)) / 2
    assert math.isclose(summary["overall"]["Q_SC"], 7.0)
    assert math.isclose(summary["overall"]["Q_PQ"], 6.5)
    assert math.isclose(summary["overall"]["Q_O"], expected_overall)
    assert math.isclose(summary["intersection"]["Q_SC"], 8.0)


def test_summarize_gedit_rows_uses_macro_average_across_groups():
    rows = []
    for idx in range(10):
        rows.append(
            {
                "task_type": "background_change",
                "instruction_language": "en",
                "semantics_score": 10.0,
                "quality_score": 10.0,
                "intersection_exist": True,
            }
        )
    for group in GEDIT_GROUPS[1:]:
        rows.append(
            {
                "task_type": group,
                "instruction_language": "en",
                "semantics_score": 1.0,
                "quality_score": 1.0,
                "intersection_exist": True,
            }
        )

    summary = summarize_gedit_rows_with_backbone(rows, language="en")

    expected_macro = (10.0 + 10.0 * 1.0) / 11
    assert math.isclose(summary["overall"]["Q_SC"], expected_macro)
    assert math.isclose(summary["overall"]["Q_O"], expected_macro)
    assert math.isclose(summary["by_group"]["background_change"]["Q_SC"], 10.0)


def test_summarize_gedit_rows_with_all_language_splits_en_and_cn():
    rows = []
    for group in GEDIT_GROUPS:
        rows.append(
            {
                "task_type": group,
                "instruction_language": "en",
                "semantics_score": 8.0,
                "quality_score": 6.0,
                "intersection_exist": True,
            }
        )
        rows.append(
            {
                "task_type": group,
                "instruction_language": "cn",
                "semantics_score": 4.0,
                "quality_score": 2.0,
                "intersection_exist": True,
            }
        )

    summary = summarize_gedit_rows_with_backbone(rows, language="all")

    assert set(summary["languages"]) == {"en", "cn"}
    assert math.isclose(summary["languages"]["en"]["overall"]["Q_SC"], 8.0)
    assert math.isclose(summary["languages"]["en"]["overall"]["Q_PQ"], 6.0)
    assert math.isclose(summary["languages"]["cn"]["overall"]["Q_O"], math.sqrt(8.0))
