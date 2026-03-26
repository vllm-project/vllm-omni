from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image
from tqdm.auto import tqdm

from benchmarks.accuracy.common import (
    VllmOmniImageClient,
    build_openai_url,
    extract_json_object,
    write_json,
)

GROUPS = [
    "background_change",
    "color_alter",
    "material_alter",
    "motion_change",
    "ps_human",
    "style_change",
    "subject-add",
    "subject-remove",
    "subject-replace",
    "text_change",
    "tone_transfer",
]
DEFAULT_SAMPLES_PER_GROUP = 10
logger = logging.getLogger(__name__)


def infer_model_name(model: str) -> str:
    normalized = model.rstrip("/\\")
    name = Path(normalized).name
    return name or normalized


def resolve_model_name(*, model_name: str | None, model: str | None = None, output_root: Path | None = None) -> str:
    if model_name:
        return model_name
    if model:
        return infer_model_name(model)
    if output_root is not None:
        candidates = sorted(path.name for path in output_root.iterdir() if path.is_dir())
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise ValueError(f"Could not infer model-name from empty output root: {output_root}")
        raise ValueError(
            f"Could not infer model-name from output root {output_root}; multiple candidates found: {candidates}"
        )
    raise ValueError("model-name is required when it cannot be inferred from model or output-root")


def parse_score_payload(raw_text: str) -> dict[str, Any]:
    try:
        parsed = extract_json_object(raw_text)
    except Exception:
        stripped = raw_text.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            parsed = {"score": json.loads(stripped), "reasoning": ""}
        else:
            raise
    score = parsed.get("score", [])
    if not isinstance(score, list):
        score = [score]
    parsed["score"] = [int(value) for value in score]
    return parsed


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def summarize_generated_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[record["task_type"]].append(record)
        by_language[record["instruction_language"]].append(record)

    return {
        "count": len(records),
        "by_task": {
            task_type: {
                "count": len(rows),
                "samples": sorted(row["key"] for row in rows),
            }
            for task_type, rows in sorted(by_task.items())
        },
        "by_language": {
            language: {
                "count": len(rows),
                "samples": sorted(row["key"] for row in rows),
            }
            for language, rows in sorted(by_language.items())
        },
    }


def select_balanced_gedit_rows(
    rows: list[dict[str, Any]],
    *,
    task_type: str = "all",
    instruction_language: str = "all",
    samples_per_group: int | None,
) -> list[dict[str, Any]]:
    filtered_rows = []
    for row in rows:
        if task_type != "all" and row["task_type"] != task_type:
            continue
        if instruction_language != "all" and row["instruction_language"] != instruction_language:
            continue
        filtered_rows.append(row)

    if samples_per_group is None:
        return filtered_rows

    def _select_rows_for_group(group_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if instruction_language != "all":
            return group_rows[:samples_per_group]

        rows_by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group_rows:
            rows_by_language[str(row.get("instruction_language", "")).strip()].append(row)

        ordered_languages = ["en", "cn"]
        per_language_quota = samples_per_group // max(len(ordered_languages), 1)
        remainder = samples_per_group % max(len(ordered_languages), 1)

        selected_rows: list[dict[str, Any]] = []
        leftovers: list[dict[str, Any]] = []
        for index, language in enumerate(ordered_languages):
            language_rows = rows_by_language.get(language, [])
            quota = per_language_quota + (1 if index < remainder else 0)
            selected_rows.extend(language_rows[:quota])
            leftovers.extend(language_rows[quota:])

        if len(selected_rows) < samples_per_group:
            selected_rows.extend(leftovers[: samples_per_group - len(selected_rows)])

        return selected_rows

    if task_type != "all":
        return _select_rows_for_group(filtered_rows)

    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in filtered_rows:
        grouped_rows[row["task_type"]].append(row)

    selected: list[dict[str, Any]] = []
    for group in GROUPS:
        selected.extend(_select_rows_for_group(grouped_rows.get(group, [])))
    return selected


def summarize_gedit_rows(rows: list[dict[str, Any]], language: str = "all") -> dict[str, Any]:
    return summarize_gedit_rows_with_backbone(rows, language=language)


def _mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _to_q_metrics(section: dict[str, float | None]) -> dict[str, float | None]:
    return {
        "count": section.get("count"),
        "Q_SC": section.get("avg_semantics"),
        "Q_PQ": section.get("avg_quality"),
        "Q_O": section.get("avg_overall"),
    }


def _summarize_gedit_rows_single_language(
    rows: list[dict[str, Any]],
    *,
    language: str,
) -> dict[str, Any]:
    filtered_rows = [row for row in rows if str(row.get("instruction_language", "")).strip() == language]

    def _to_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes"}

    per_group: dict[str, dict[str, float | None]] = {}
    intersection_group: dict[str, dict[str, float | None]] = {}

    for group in GROUPS:
        group_rows = [row for row in filtered_rows if row.get("task_type") == group]
        semantics = [float(row["semantics_score"]) for row in group_rows]
        quality = [float(row["quality_score"]) for row in group_rows]
        overall = [math.sqrt(float(row["semantics_score"]) * float(row["quality_score"])) for row in group_rows]

        per_group[group] = {
            "count": len(group_rows),
            "avg_semantics": _mean_or_none(semantics),
            "avg_quality": _mean_or_none(quality),
            "avg_overall": _mean_or_none(overall),
        }

        intersection_rows = [row for row in group_rows if _to_bool(row.get("intersection_exist", False))]
        intersection_semantics = [float(row["semantics_score"]) for row in intersection_rows]
        intersection_quality = [float(row["quality_score"]) for row in intersection_rows]
        intersection_overall = [
            math.sqrt(float(row["semantics_score"]) * float(row["quality_score"])) for row in intersection_rows
        ]
        intersection_group[group] = {
            "count": len(intersection_rows),
            "avg_semantics": _mean_or_none(intersection_semantics),
            "avg_quality": _mean_or_none(intersection_quality),
            "avg_overall": _mean_or_none(intersection_overall),
        }

    overall_section = {
        "count": len(filtered_rows),
        "avg_semantics": _mean_or_none(
            [score["avg_semantics"] for score in per_group.values() if score["avg_semantics"] is not None]
        ),
        "avg_quality": _mean_or_none(
            [score["avg_quality"] for score in per_group.values() if score["avg_quality"] is not None]
        ),
        "avg_overall": _mean_or_none(
            [score["avg_overall"] for score in per_group.values() if score["avg_overall"] is not None]
        ),
    }
    intersection_section = {
        "count": sum(score["count"] for score in intersection_group.values()),
        "avg_semantics": _mean_or_none(
            [score["avg_semantics"] for score in intersection_group.values() if score["avg_semantics"] is not None]
        ),
        "avg_quality": _mean_or_none(
            [score["avg_quality"] for score in intersection_group.values() if score["avg_quality"] is not None]
        ),
        "avg_overall": _mean_or_none(
            [score["avg_overall"] for score in intersection_group.values() if score["avg_overall"] is not None]
        ),
    }

    return {
        "language": language,
        "by_group": {group: _to_q_metrics(section) for group, section in per_group.items()},
        "overall": _to_q_metrics(overall_section),
        "intersection": _to_q_metrics(intersection_section),
    }


def summarize_gedit_rows_with_backbone(
    rows: list[dict[str, Any]],
    *,
    language: str = "all",
) -> dict[str, Any]:
    if language == "all":
        return {
            "language": "all",
            "languages": {
                single_language: _summarize_gedit_rows_single_language(
                    rows,
                    language=single_language,
                )
                for single_language in ["en", "cn"]
            },
        }

    return _summarize_gedit_rows_single_language(rows, language=language)


def _require_datasets():
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError as exc:
        raise ImportError("GEdit-Bench requires the optional `datasets` package.") from exc
    return load_dataset, load_from_disk


def _load_gedit_dataset(dataset_ref: str):
    load_dataset, load_from_disk = _require_datasets()
    dataset_path = Path(dataset_ref)
    if dataset_path.exists():
        if (dataset_path / "state.json").exists() and (dataset_path / "dataset_info.json").exists():
            return load_from_disk(str(dataset_path))
        return load_dataset(str(dataset_path))
    return load_dataset(dataset_ref)


def _resolve_gedit_split(dataset_obj: Any) -> Any:
    if isinstance(dataset_obj, dict):
        if "train" in dataset_obj:
            return dataset_obj["train"]
        return dataset_obj
    try:
        return dataset_obj["train"]
    except Exception:
        return dataset_obj


def _to_pil_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict) and "bytes" in value:
        image = Image.open(BytesIO(value["bytes"]))
        image.load()
        return image.convert("RGB")
    raise TypeError(f"Unsupported image payload type: {type(value)!r}")


class LocalVIEScorer:
    def __init__(self, *, base_url: str, api_key: str, model: str, timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.sc_prompt = (
            "You are evaluating image editing quality.\n"
            "Two images are provided: the source image and the edited image.\n"
            'Return JSON only in the format {"score": [edit_success, content_preservation], '
            '"reasoning": "..."}.\n'
            "Each score must be an integer from 0 to 10.\n"
            "Editing instruction: <instruction>"
        )
        self.pq_prompt = (
            "You are evaluating image quality.\n"
            'Return JSON only in the format {"score": [naturalness, artifact_free], "reasoning": "..."}.\n'
            "Each score must be an integer from 0 to 10."
        )

    def _request(self, prompt: str, images: list[Image.Image]) -> dict[str, Any]:
        from benchmarks.accuracy.common import pil_to_data_url

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in images:
            content.append({"type": "image_url", "image_url": {"url": pil_to_data_url(image)}})

        response = requests.post(
            build_openai_url(self.base_url, "/chat/completions"),
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0,
            },
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        message_content = response.json()["choices"][0]["message"]["content"]
        if isinstance(message_content, list):
            text = "\n".join(part.get("text", "") for part in message_content if part.get("type") == "text")
        else:
            text = str(message_content)
        return parse_score_payload(text)

    def evaluate(self, source_image: Image.Image, edited_image: Image.Image, instruction: str) -> dict[str, float]:
        sc_payload = self._request(self.sc_prompt.replace("<instruction>", instruction), [source_image, edited_image])
        pq_payload = self._request(self.pq_prompt, [edited_image])
        semantics = float(min(sc_payload["score"])) if sc_payload["score"] else 0.0
        quality = float(min(pq_payload["score"])) if pq_payload["score"] else 0.0
        overall = math.sqrt(semantics * quality)
        return {
            "semantics_score": semantics,
            "quality_score": quality,
            "overall_score": overall,
        }


class GEditBenchRunner:
    def __init__(
        self,
        *,
        dataset_ref: str,
        output_root: Path,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float | None = None,
        seed: int | None = 42,
    ):
        self.dataset_ref = dataset_ref
        self.output_root = output_root
        self.model = model
        self.width = width
        self.height = height
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.seed = seed
        self.client = VllmOmniImageClient(base_url=base_url, api_key=api_key)

    def generate(
        self,
        *,
        model_name: str,
        task_type: str = "all",
        instruction_language: str = "all",
        workers: int = 1,
        max_samples: int | None = None,
        samples_per_group: int | None = None,
    ) -> list[dict[str, Any]]:
        dataset = _resolve_gedit_split(_load_gedit_dataset(self.dataset_ref))
        rows = select_balanced_gedit_rows(
            list(dataset),
            task_type=task_type,
            instruction_language=instruction_language,
            samples_per_group=samples_per_group,
        )
        if max_samples is not None:
            rows = rows[:max_samples]

        outputs: list[dict[str, Any]] = []
        total = len(rows)
        if workers <= 1:
            with tqdm(total=total, desc="GEdit generate", unit="sample") as progress:
                for item in rows:
                    result = self._safe_generate_one(model_name, item)
                    if result:
                        outputs.append(result)
                    progress.update(1)
            return outputs

        with tqdm(total=total, desc="GEdit generate", unit="sample") as progress:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(self._safe_generate_one, model_name, item) for item in rows]
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        outputs.append(result)
                    progress.update(1)
        return outputs

    def _safe_generate_one(self, model_name: str, item: dict[str, Any]) -> dict[str, Any] | None:
        try:
            return self._generate_one(model_name, item)
        except Exception:
            logger.exception("Failed to generate GEdit-Bench sample %s", item.get("key", "<unknown>"))
            return None

    def _generate_one(self, model_name: str, item: dict[str, Any]) -> dict[str, Any] | None:
        output_path = (
            self.output_root
            / model_name
            / "fullset"
            / item["task_type"]
            / item["instruction_language"]
            / f"{item['key']}.png"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            return {
                "task_type": item["task_type"],
                "instruction_language": item["instruction_language"],
                "key": item["key"],
                "output_path": str(output_path),
            }

        source_image = _to_pil_image(item["input_image_raw"])
        edited_image = self.client.generate_image_edit(
            model=self.model,
            prompt=item["instruction"],
            images=source_image,
            width=self.width,
            height=self.height,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            seed=self.seed,
        )
        edited_image.save(output_path)
        return {
            "task_type": item["task_type"],
            "instruction_language": item["instruction_language"],
            "key": item["key"],
            "output_path": str(output_path),
        }


class GEditBenchEvaluator:
    def __init__(self, *, dataset_ref: str, output_root: Path, scorer: LocalVIEScorer):
        self.dataset_ref = dataset_ref
        self.output_root = output_root
        self.scorer = scorer

    def evaluate(
        self,
        *,
        model_name: str,
        save_dir: Path,
        task_type: str = "all",
        instruction_language: str = "all",
        workers: int = 1,
        max_samples: int | None = None,
        samples_per_group: int | None = None,
    ) -> dict[str, Any]:
        dataset = _resolve_gedit_split(_load_gedit_dataset(self.dataset_ref))
        rows = select_balanced_gedit_rows(
            list(dataset),
            task_type=task_type,
            instruction_language=instruction_language,
            samples_per_group=samples_per_group,
        )
        if max_samples is not None:
            rows = rows[:max_samples]

        results: list[dict[str, Any]] = []
        total = len(rows)
        if workers <= 1:
            with tqdm(total=total, desc="GEdit evaluate", unit="sample") as progress:
                for item in rows:
                    result = self._safe_evaluate_one(model_name, item)
                    if result:
                        results.append(result)
                    progress.update(1)
        else:
            with tqdm(total=total, desc="GEdit evaluate", unit="sample") as progress:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [executor.submit(self._safe_evaluate_one, model_name, item) for item in rows]
                    for future in as_completed(futures):
                        result = future.result()
                        if result:
                            results.append(result)
                        progress.update(1)

        save_dir.mkdir(parents=True, exist_ok=True)
        base_name = f"{model_name}_{task_type}_{instruction_language}"
        timestamp = _utc_timestamp()
        csv_path = save_dir / f"{base_name}_vie_score.csv"
        timestamped_csv_path = save_dir / f"{base_name}_vie_score_{timestamp}.csv"

        def _write_csv(path: Path) -> None:
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "key",
                        "task_type",
                        "edited_image",
                        "instruction",
                        "semantics_score",
                        "quality_score",
                        "overall_score",
                        "intersection_exist",
                        "instruction_language",
                    ],
                )
                writer.writeheader()
                for row in results:
                    writer.writerow(row)

        _write_csv(csv_path)
        _write_csv(timestamped_csv_path)

        summary = summarize_gedit_rows_with_backbone(
            results,
            language=instruction_language,
        )
        summary_path = save_dir / f"{base_name}_summary.json"
        timestamped_summary_path = save_dir / f"{base_name}_summary_{timestamp}.json"
        write_json(summary_path, summary)
        write_json(timestamped_summary_path, summary)
        return {
            "results": results,
            "summary": summary,
            "csv_path": str(csv_path),
            "summary_path": str(summary_path),
            "timestamped_csv_path": str(timestamped_csv_path),
            "timestamped_summary_path": str(timestamped_summary_path),
        }

    def _safe_evaluate_one(self, model_name: str, item: dict[str, Any]) -> dict[str, Any] | None:
        try:
            return self._evaluate_one(model_name, item)
        except Exception:
            logger.exception("Failed to evaluate GEdit-Bench sample %s", item.get("key", "<unknown>"))
            return None

    def _evaluate_one(self, model_name: str, item: dict[str, Any]) -> dict[str, Any] | None:
        edited_image_path = (
            self.output_root
            / model_name
            / "fullset"
            / item["task_type"]
            / item["instruction_language"]
            / f"{item['key']}.png"
        )
        if not edited_image_path.exists():
            return None

        source_image = _to_pil_image(item["input_image_raw"])
        edited_image = Image.open(edited_image_path).convert("RGB")
        scores = self.scorer.evaluate(source_image, edited_image, item["instruction"])
        return {
            "key": item["key"],
            "task_type": item["task_type"],
            "edited_image": str(edited_image_path),
            "instruction": item["instruction"],
            "semantics_score": scores["semantics_score"],
            "quality_score": scores["quality_score"],
            "overall_score": scores["overall_score"],
            "intersection_exist": item.get("Intersection_exist", False),
            "instruction_language": item["instruction_language"],
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the GEdit-Bench integration against a local vLLM-Omni server.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--dataset-ref", type=str, default="stepfun-ai/GEdit-Bench")
    generate.add_argument("--output-root", type=Path, required=True)
    generate.add_argument("--base-url", type=str, required=True)
    generate.add_argument("--model", type=str, required=True)
    generate.add_argument("--model-name", type=str, default=None)
    generate.add_argument("--api-key", type=str, default="EMPTY")
    generate.add_argument("--task-type", choices=["all", *GROUPS], default="all")
    generate.add_argument("--instruction-language", choices=["all", "en", "cn"], default="all")
    generate.add_argument("--width", type=int, default=512)
    generate.add_argument("--height", type=int, default=512)
    generate.add_argument("--num-inference-steps", type=int, default=20)
    generate.add_argument("--guidance-scale", type=float, default=None)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--workers", type=int, default=1)
    generate.add_argument("--max-samples", type=int, default=None)
    generate.add_argument("--samples-per-group", type=int, default=None)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--dataset-ref", type=str, default="stepfun-ai/GEdit-Bench")
    evaluate.add_argument("--output-root", type=Path, required=True)
    evaluate.add_argument("--model-name", type=str, default=None)
    evaluate.add_argument("--save-dir", type=Path, required=True)
    evaluate.add_argument("--task-type", choices=["all", *GROUPS], default="all")
    evaluate.add_argument("--instruction-language", choices=["all", "en", "cn"], default="all")
    evaluate.add_argument("--judge-base-url", type=str, required=True)
    evaluate.add_argument("--judge-model", type=str, required=True)
    evaluate.add_argument("--judge-api-key", type=str, default="EMPTY")
    evaluate.add_argument("--workers", type=int, default=1)
    evaluate.add_argument("--max-samples", type=int, default=None)
    evaluate.add_argument("--samples-per-group", type=int, default=None)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--csv-path", type=Path, required=True)
    summarize.add_argument("--language", choices=["all", "en", "cn"], default="all")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate":
        model_name = resolve_model_name(model_name=args.model_name, model=args.model)
        runner = GEditBenchRunner(
            dataset_ref=args.dataset_ref,
            output_root=args.output_root,
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            width=args.width,
            height=args.height,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
        records = runner.generate(
            model_name=model_name,
            task_type=args.task_type,
            instruction_language=args.instruction_language,
            workers=args.workers,
            max_samples=args.max_samples,
            samples_per_group=args.samples_per_group,
        )
        payload = {"records": records, "summary": summarize_generated_records(records)}
        write_json(args.output_root / model_name / "generation_manifest.json", payload)
        return 0

    if args.command == "evaluate":
        model_name = resolve_model_name(model_name=args.model_name, output_root=args.output_root)
        scorer = LocalVIEScorer(
            base_url=args.judge_base_url,
            api_key=args.judge_api_key,
            model=args.judge_model,
        )
        evaluator = GEditBenchEvaluator(dataset_ref=args.dataset_ref, output_root=args.output_root, scorer=scorer)
        evaluator.evaluate(
            model_name=model_name,
            save_dir=args.save_dir,
            task_type=args.task_type,
            instruction_language=args.instruction_language,
            workers=args.workers,
            max_samples=args.max_samples,
            samples_per_group=args.samples_per_group,
        )
        return 0

    if args.command == "summarize":
        with args.csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        summary = summarize_gedit_rows_with_backbone(rows, language=args.language)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 1
