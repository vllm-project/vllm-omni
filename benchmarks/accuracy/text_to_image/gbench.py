from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from benchmarks.accuracy.common import (
    VllmOmniImageClient,
    build_openai_url,
    ensure_dir,
    extract_json_object,
    find_first_image,
    load_json,
    pil_to_data_url,
    save_image,
    write_json,
)

TYPE_TO_FOLDER = {
    "type1": "01_single_step",
    "type2": "02_multi_step",
    "type3": "03_trajectory_text_fictionalapp",
    "type4": "04_trajectory_text_realapp",
    "type5": "05_grounding_data",
}
SCORE_KEYS = ("goal", "logic", "cons", "ui", "qual")
DEFAULT_SAMPLES_PER_TYPE = 10


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json_with_timestamp(path: Path, payload: dict[str, Any]) -> Path:
    write_json(path, payload)
    timestamped_path = path.with_name(f"{path.stem}_{_utc_timestamp()}{path.suffix}")
    write_json(timestamped_path, payload)
    return timestamped_path


@dataclass(frozen=True)
class GEBenchSampleSpec:
    sample_path: Path
    metadata: dict[str, Any]
    sample_name: str
    lang_device: str


def summarize_generated_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_type[record["data_type"]].append(record)

    return {
        "count": len(records),
        "by_type": {
            data_type: {
                "count": len(rows),
            }
            for data_type, rows in sorted(by_type.items())
        },
    }


def summarize_gebench_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_type[result["data_type"]].append(result)

    summary: dict[str, Any] = {
        "count": len(results),
        "overall_mean": statistics.fmean(r["overall"] for r in results) if results else 0.0,
        "by_type": {},
    }
    for data_type, rows in by_type.items():
        score_means: dict[str, float] = {}
        all_score_keys = {key for row in rows for key in row.get("scores", {}).keys()}
        for score_key in all_score_keys:
            values = [row["scores"][score_key] for row in rows if score_key in row.get("scores", {})]
            score_means[score_key] = statistics.fmean(values) if values else 0.0
        overall_mean = statistics.fmean(row["overall"] for row in rows)
        summary["by_type"][data_type] = {
            "count": len(rows),
            "overall_mean": overall_mean,
            "overall_mean_100": overall_mean * 100.0,
            "score_means": score_means,
        }
    return summary


def select_balanced_gebench_samples(
    sample_paths_by_type: dict[str, list[Any]],
    *,
    samples_per_type: int | None,
) -> dict[str, list[Any]]:
    if samples_per_type is None:
        return {data_type: list(paths) for data_type, paths in sample_paths_by_type.items()}
    return {data_type: list(paths)[:samples_per_type] for data_type, paths in sample_paths_by_type.items()}


def collect_gebench_generation_summary(output_root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for data_type, folder_name in TYPE_TO_FOLDER.items():
        type_root = output_root / folder_name
        if not type_root.exists():
            continue
        for lang_dir in sorted(path for path in type_root.iterdir() if path.is_dir()):
            for sample_dir in sorted(path for path in lang_dir.iterdir() if path.is_dir()):
                expected = sample_dir / "frame5.png" if data_type in {"type2", "type3", "type4"} else None
                if expected is None:
                    expected = find_first_image(sample_dir)
                elif not expected.exists():
                    expected = None
                if expected is None:
                    continue
                records.append(
                    {
                        "data_type": data_type,
                        "sample_name": f"{lang_dir.name}/{sample_dir.name}",
                        "output_path": str(expected),
                    }
                )
    return summarize_generated_records(records)


def _normalize_score_key(key: str) -> str:
    mapping = {
        "goal": "goal",
        "logic": "logic",
        "cons": "cons",
        "consistency": "cons",
        "ui": "ui",
        "qual": "qual",
        "quality": "qual",
    }
    return mapping.get(key.lower(), key.lower())


def _normalize_scores(raw_scores: dict[str, Any]) -> dict[str, int]:
    scores: dict[str, int] = {}
    for key, value in raw_scores.items():
        normalized = _normalize_score_key(key)
        if normalized not in SCORE_KEYS:
            continue
        scalar = value.get("s", 0) if isinstance(value, dict) else value
        try:
            scores[normalized] = int(scalar)
        except (TypeError, ValueError):
            scores[normalized] = 0
    for key in SCORE_KEYS:
        scores.setdefault(key, 0)
    return scores


def _compute_overall(scores: dict[str, int]) -> float:
    return sum(scores.values()) / (len(SCORE_KEYS) * 5.0)


def _iter_sample_paths(dataset_root: Path, data_type: str) -> list[Path]:
    data_dir = dataset_root / TYPE_TO_FOLDER[data_type]
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_dir}")

    samples: list[Path] = []
    for lang_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        for child in sorted(lang_dir.iterdir()):
            if child.is_dir():
                samples.append(child)
            elif child.suffix.lower() == ".json":
                samples.append(child)
    return samples


def _sample_name_from_metadata(metadata: dict[str, Any], sample_path: Path, item_index: int | None = None) -> str:
    sample_id = metadata.get("id") or metadata.get("sample_id") or metadata.get("name")
    if sample_id:
        return str(sample_id)
    if item_index is not None:
        return f"{sample_path.stem}_{item_index:04d}"
    return sample_path.stem if sample_path.is_file() else sample_path.name


def _expand_sample_path(sample_path: Path) -> list[GEBenchSampleSpec]:
    if sample_path.is_dir():
        metadata = _load_metadata(sample_path)
        return [
            GEBenchSampleSpec(
                sample_path=sample_path,
                metadata=metadata,
                sample_name=_sample_name(sample_path),
                lang_device=_lang_device(sample_path, metadata),
            )
        ]

    payload = load_json(sample_path)
    if isinstance(payload, dict):
        return [
            GEBenchSampleSpec(
                sample_path=sample_path,
                metadata=payload,
                sample_name=_sample_name_from_metadata(payload, sample_path),
                lang_device=_lang_device(sample_path, payload),
            )
        ]

    if isinstance(payload, list):
        specs: list[GEBenchSampleSpec] = []
        for item_index, item in enumerate(payload):
            if not isinstance(item, dict):
                continue
            specs.append(
                GEBenchSampleSpec(
                    sample_path=sample_path,
                    metadata=item,
                    sample_name=_sample_name_from_metadata(item, sample_path, item_index),
                    lang_device=_lang_device(sample_path, item),
                )
            )
        return specs

    raise TypeError(f"Unsupported metadata payload type for sample {sample_path}: {type(payload)!r}")


def _iter_sample_specs(dataset_root: Path, data_type: str) -> list[GEBenchSampleSpec]:
    sample_specs: list[GEBenchSampleSpec] = []
    for sample_path in _iter_sample_paths(dataset_root, data_type):
        sample_specs.extend(_expand_sample_path(sample_path))
    return sample_specs


def _load_metadata(sample_path: Path) -> dict[str, Any]:
    if sample_path.is_file():
        return load_json(sample_path)
    for candidate in ("meta_data.json", "metadata.json"):
        meta_path = sample_path / candidate
        if meta_path.exists():
            return load_json(meta_path)
    raise FileNotFoundError(f"Metadata not found for sample: {sample_path}")


def _sample_name(sample_path: Path) -> str:
    return sample_path.stem if sample_path.is_file() else sample_path.name


def _lang_device(sample_path: Path, metadata: dict[str, Any]) -> str:
    return str(metadata.get("lang_device") or sample_path.parent.name)


def _resolve_referenced_image(
    *,
    metadata: dict[str, Any],
    sample_path: Path,
    dataset_root: Path,
    data_type: str,
) -> Image.Image | None:
    for key in ("image", "input_image", "initial_image", "reference_image"):
        image_ref = metadata.get(key)
        if not image_ref:
            continue
        candidate = dataset_root / TYPE_TO_FOLDER[data_type] / str(image_ref)
        if candidate.exists():
            image = Image.open(candidate)
            image.load()
            return image.convert("RGB")
    if sample_path.is_dir():
        local_image = find_first_image(sample_path)
        if local_image:
            image = Image.open(local_image)
            image.load()
            return image.convert("RGB")
    return None


def _trajectory_steps(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("trajectory", "steps", "frames"):
        value = metadata.get(key)
        if isinstance(value, list):
            return [step for step in value if isinstance(step, dict)]
    extracted: list[dict[str, Any]] = []
    for index in range(1, 6):
        value = metadata.get(f"step{index}") or metadata.get(str(index))
        if isinstance(value, dict):
            extracted.append(value)
    return extracted


def _text_or_default(value: Any, default: str = "") -> str:
    return str(value).strip() if value is not None else default


def _type1_prompt(metadata: dict[str, Any]) -> str:
    caption = _text_or_default(metadata.get("caption") or metadata.get("instruction"), "Transform the reference GUI.")
    return (
        "Using the reference GUI screenshot, generate the next GUI state after the requested interaction.\n\n"
        f"Requested change:\n{caption}\n\n"
        "Requirements:\n"
        "- Preserve layout, visual identity, and unrelated regions.\n"
        "- Only apply the requested state change.\n"
        "- Keep all text and controls readable.\n"
    )


def _type2_prompt(goal: str, step_num: int) -> str:
    return (
        "Generate the next GUI state for a multi-step task.\n\n"
        f"Overall goal: {goal}\n"
        f"Current progress step: {step_num}/5\n\n"
        "Requirements:\n"
        "- The change should be incremental and plausible.\n"
        "- Preserve layout and visual identity.\n"
        "- Make text/buttons readable.\n"
    )


def _type34_initial_prompt(metadata: dict[str, Any], first_step: dict[str, Any]) -> str:
    app_name = _text_or_default(metadata.get("app_name"), "App")
    final_goal = _text_or_default(metadata.get("final_goal") or metadata.get("instruction"), "Complete the task.")
    visual_description = _text_or_default(
        metadata.get("visual_description") or first_step.get("visual_description") or first_step.get("description"),
        "A clean product-quality app home screen.",
    )
    return (
        "Generate the first GUI frame for a task trajectory.\n\n"
        f"App name: {app_name}\n"
        f"Final goal: {final_goal}\n"
        f"Visual description:\n{visual_description}\n\n"
        "Requirements:\n"
        "- Generate a production-looking UI screenshot only.\n"
        "- Keep the layout coherent and readable.\n"
    )


def _type34_next_prompt(step_num: int, step_info: dict[str, Any]) -> str:
    action = _text_or_default(step_info.get("action") or step_info.get("instruction"), "Continue the task.")
    visual_description = _text_or_default(
        step_info.get("visual_description") or step_info.get("description"),
        "Reflect the expected next GUI state.",
    )
    return (
        "Using the previous frame as reference, generate the next GUI frame.\n\n"
        f"Step {step_num} action: {action}\n"
        f"Expected visual state:\n{visual_description}\n\n"
        "Requirements:\n"
        "- Only change UI regions affected by this action.\n"
        "- Preserve persistent bars, layout, and style.\n"
        "- Keep text and icons readable.\n"
    )


def _type5_prompt(metadata: dict[str, Any]) -> str:
    grounding = metadata.get("grounding") or {}
    explanation = _text_or_default(
        metadata.get("grounding_explanation") or grounding.get("effect") or grounding.get("description"),
        "Predict the immediate GUI reaction to the indicated target.",
    )
    return (
        "Using the reference GUI screenshot, predict the immediate GUI state after the grounded interaction.\n\n"
        f"Expected effect: {explanation}\n"
        f"Grounding metadata: {json.dumps(grounding, ensure_ascii=False)}\n\n"
        "Requirements:\n"
        "- Apply only the interaction-triggered change.\n"
        "- Preserve unrelated regions.\n"
        "- Keep the UI realistic and readable.\n"
    )


def _make_storyboard_image(
    frames: list[Image.Image],
    *,
    columns: int = 3,
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    if not frames:
        raise ValueError("Expected at least one frame to build a storyboard image.")

    normalized = [frame.convert("RGB") for frame in frames]
    frame_width = max(frame.width for frame in normalized)
    frame_height = max(frame.height for frame in normalized)
    rows = (len(normalized) + columns - 1) // columns
    storyboard = Image.new("RGB", (frame_width * columns, frame_height * rows), color=background_color)

    for index, frame in enumerate(normalized):
        x_offset = (index % columns) * frame_width
        y_offset = (index // columns) * frame_height
        if frame.size != (frame_width, frame_height):
            frame = frame.resize((frame_width, frame_height))
        storyboard.paste(frame, (x_offset, y_offset))
    return storyboard


def _trajectory_judge_payload(frames: list[Image.Image]) -> tuple[str, list[Image.Image]]:
    storyboard = _make_storyboard_image(frames, columns=3)
    prompt_suffix = (
        "The attached image is a storyboard containing six frames arranged left-to-right, "
        "top-to-bottom as frame0, frame1, frame2, frame3, frame4, frame5."
    )
    return prompt_suffix, [storyboard]


class LocalJudgeClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _build_scoring_prompt(self, task_prompt: str) -> str:
        return (
            "You are an expert evaluator for GUI image editing and GUI trajectory generation.\n"
            "Evaluate whether the generated image(s) satisfy the task.\n\n"
            "Score these five dimensions from 0 to 5:\n"
            "- goal: whether the user goal is completed correctly\n"
            "- logic: whether the transition/state change is logically correct\n"
            "- cons: whether unrelated regions remain consistent\n"
            "- ui: whether the UI layout/components remain realistic and coherent\n"
            "- qual: whether the images are visually clear and artifact-free\n\n"
            "Return JSON only. Do not add any prose outside JSON.\n"
            "Use exactly this schema:\n"
            "{\n"
            '  "goal": 0,\n'
            '  "logic": 0,\n'
            '  "cons": 0,\n'
            '  "ui": 0,\n'
            '  "qual": 0,\n'
            '  "reasoning": "short explanation"\n'
            "}\n\n"
            "Scoring task:\n"
            f"{task_prompt}"
        )

    def _request_text(self, prompt: str, images: list[Image.Image]) -> str:
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
            return "\n".join(part.get("text", "") for part in message_content if part.get("type") == "text")
        return str(message_content)

    def evaluate(self, *, prompt: str, images: list[Image.Image]) -> dict[str, Any]:
        primary_prompt = self._build_scoring_prompt(prompt)
        text = self._request_text(primary_prompt, images)
        try:
            return extract_json_object(text)
        except ValueError:
            retry_prompt = (
                self._build_scoring_prompt(prompt) + "\n\nYour previous response was not valid JSON. "
                "Return only the JSON object with integer scores."
            )
            retry_text = self._request_text(retry_prompt, images)
            try:
                return extract_json_object(retry_text)
            except ValueError:
                return {
                    "goal": 0,
                    "logic": 0,
                    "cons": 0,
                    "ui": 0,
                    "qual": 0,
                    "reasoning": retry_text.strip() or text.strip() or "Judge response was not valid JSON.",
                }


class GEBenchRunner:
    def __init__(
        self,
        *,
        dataset_root: Path,
        output_root: Path,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        width: int = 768,
        height: int = 576,
        num_inference_steps: int = 8,
        output_compression: int | None = 98,
        guidance_scale: float | None = None,
        seed: int | None = 42,
    ):
        self.dataset_root = dataset_root
        self.output_root = output_root
        self.model = model
        self.width = width
        self.height = height
        self.num_inference_steps = num_inference_steps
        self.output_compression = output_compression
        self.guidance_scale = guidance_scale
        self.seed = seed
        self.client = VllmOmniImageClient(base_url=base_url, api_key=api_key)

    def generate(
        self,
        *,
        data_type: str,
        workers: int = 1,
        max_samples: int | None = None,
        samples_per_type: int | None = None,
    ) -> list[dict[str, Any]]:
        sample_specs = select_balanced_gebench_samples(
            {data_type: _iter_sample_specs(self.dataset_root, data_type)},
            samples_per_type=samples_per_type,
        )[data_type]
        if max_samples is not None:
            sample_specs = sample_specs[:max_samples]

        results: list[dict[str, Any]] = []
        if workers <= 1:
            for sample_spec in sample_specs:
                result = self._generate_one(data_type, sample_spec)
                if result:
                    results.append(result)
            return results

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self._generate_one, data_type, sample_spec) for sample_spec in sample_specs]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
        return results

    def _generate_one(self, data_type: str, sample_spec: GEBenchSampleSpec) -> dict[str, Any] | None:
        sample_path = sample_spec.sample_path
        metadata = sample_spec.metadata
        lang_device = sample_spec.lang_device
        sample_name = sample_spec.sample_name
        output_dir = ensure_dir(self.output_root / TYPE_TO_FOLDER[data_type] / lang_device / sample_name)

        if data_type == "type1":
            output_path = output_dir / "generated.png"
            if output_path.exists():
                return {
                    "data_type": data_type,
                    "sample_name": f"{lang_device}/{sample_name}",
                    "output_path": str(output_path),
                }
            source = _resolve_referenced_image(
                metadata=metadata, sample_path=sample_path, dataset_root=self.dataset_root, data_type=data_type
            )
            if source is None:
                return None
            generated = self.client.generate_image_edit(
                model=self.model,
                prompt=_type1_prompt(metadata),
                images=source,
                width=self.width,
                height=self.height,
                num_inference_steps=self.num_inference_steps,
                output_compression=self.output_compression,
                guidance_scale=self.guidance_scale,
                seed=self.seed,
            )
            save_image(output_path, generated)
            return {
                "data_type": data_type,
                "sample_name": f"{lang_device}/{sample_name}",
                "output_path": str(output_path),
            }

        if data_type == "type2":
            goal = _text_or_default(metadata.get("question") or metadata.get("caption"), "Complete the task.")
            source = _resolve_referenced_image(
                metadata=metadata, sample_path=sample_path, dataset_root=self.dataset_root, data_type=data_type
            )
            if source is None:
                return None
            frame0_path = output_dir / "frame0.png"
            if not frame0_path.exists():
                save_image(frame0_path, source)
            previous = source
            for step_num in range(1, 6):
                frame_path = output_dir / f"frame{step_num}.png"
                if frame_path.exists():
                    previous = Image.open(frame_path).convert("RGB")
                    continue
                generated = self.client.generate_image_edit(
                    model=self.model,
                    prompt=_type2_prompt(goal, step_num),
                    images=previous,
                    width=self.width,
                    height=self.height,
                    num_inference_steps=self.num_inference_steps,
                    output_compression=self.output_compression,
                    guidance_scale=self.guidance_scale,
                    seed=self.seed,
                )
                save_image(frame_path, generated)
                previous = generated
            output_path = output_dir / "frame5.png"
            return {
                "data_type": data_type,
                "sample_name": f"{lang_device}/{sample_name}",
                "output_path": str(output_path),
            }

        if data_type in {"type3", "type4"}:
            steps = _trajectory_steps(metadata)
            frame0_path = output_dir / "frame0.png"
            if frame0_path.exists():
                previous = Image.open(frame0_path).convert("RGB")
            else:
                previous = self.client.generate_text_to_image(
                    model=self.model,
                    prompt=_type34_initial_prompt(metadata, steps[0] if steps else {}),
                    width=self.width,
                    height=self.height,
                    num_inference_steps=self.num_inference_steps,
                    output_compression=self.output_compression,
                    guidance_scale=self.guidance_scale,
                    seed=self.seed,
                )
                save_image(frame0_path, previous)

            for step_num in range(1, 6):
                frame_path = output_dir / f"frame{step_num}.png"
                if frame_path.exists():
                    previous = Image.open(frame_path).convert("RGB")
                    continue
                step_info = steps[step_num - 1] if step_num - 1 < len(steps) else {}
                generated = self.client.generate_image_edit(
                    model=self.model,
                    prompt=_type34_next_prompt(step_num, step_info),
                    images=previous,
                    width=self.width,
                    height=self.height,
                    num_inference_steps=self.num_inference_steps,
                    output_compression=self.output_compression,
                    guidance_scale=self.guidance_scale,
                    seed=self.seed,
                )
                save_image(frame_path, generated)
                previous = generated
            output_path = output_dir / "frame5.png"
            return {
                "data_type": data_type,
                "sample_name": f"{lang_device}/{sample_name}",
                "output_path": str(output_path),
            }

        if data_type == "type5":
            output_path = output_dir / "generated.png"
            if output_path.exists():
                return {
                    "data_type": data_type,
                    "sample_name": f"{lang_device}/{sample_name}",
                    "output_path": str(output_path),
                }
            source = _resolve_referenced_image(
                metadata=metadata, sample_path=sample_path, dataset_root=self.dataset_root, data_type=data_type
            )
            if source is None:
                return None
            generated = self.client.generate_image_edit(
                model=self.model,
                prompt=_type5_prompt(metadata),
                images=source,
                width=self.width,
                height=self.height,
                num_inference_steps=self.num_inference_steps,
                output_compression=self.output_compression,
                guidance_scale=self.guidance_scale,
                seed=self.seed,
            )
            save_image(output_path, generated)
            return {
                "data_type": data_type,
                "sample_name": f"{lang_device}/{sample_name}",
                "output_path": str(output_path),
            }

        raise ValueError(f"Unsupported data type: {data_type}")


class GEBenchEvaluator:
    def __init__(self, *, dataset_root: Path, output_root: Path, judge: LocalJudgeClient):
        self.dataset_root = dataset_root
        self.output_root = output_root
        self.judge = judge

    def evaluate(
        self,
        *,
        data_type: str,
        workers: int = 1,
        max_samples: int | None = None,
        samples_per_type: int | None = None,
    ) -> dict[str, Any]:
        output_type_dir = self.output_root / TYPE_TO_FOLDER[data_type]
        sample_specs_by_name = {
            (spec.lang_device, spec.sample_name): spec for spec in _iter_sample_specs(self.dataset_root, data_type)
        }
        if not output_type_dir.exists():
            payload = {"data_type": data_type, "results": [], "summary": summarize_gebench_results([])}
            write_json(self.output_root / "evaluations" / f"{data_type}.json", payload)
            return payload
        sample_dirs = [
            sample_dir
            for lang_dir in sorted(path for path in output_type_dir.iterdir() if path.is_dir())
            for sample_dir in sorted(path for path in lang_dir.iterdir() if path.is_dir())
            if (lang_dir.name, sample_dir.name) in sample_specs_by_name
        ]
        sample_dirs = select_balanced_gebench_samples(
            {data_type: sample_dirs},
            samples_per_type=samples_per_type,
        )[data_type]
        if max_samples is not None:
            sample_dirs = sample_dirs[:max_samples]
        results: list[dict[str, Any]] = []
        if workers <= 1:
            for sample_dir in sample_dirs:
                result = self._evaluate_one(
                    data_type,
                    sample_dir,
                    sample_specs_by_name[(sample_dir.parent.name, sample_dir.name)],
                )
                if result:
                    results.append(result)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        self._evaluate_one,
                        data_type,
                        sample_dir,
                        sample_specs_by_name[(sample_dir.parent.name, sample_dir.name)],
                    )
                    for sample_dir in sample_dirs
                ]
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        results.append(result)

        payload = {"data_type": data_type, "results": results, "summary": summarize_gebench_results(results)}
        write_json(self.output_root / "evaluations" / f"{data_type}.json", payload)
        return payload

    def _evaluate_one(self, data_type: str, sample_dir: Path, sample_spec: GEBenchSampleSpec) -> dict[str, Any] | None:
        lang_device = sample_dir.parent.name
        sample_name = sample_dir.name
        dataset_sample = sample_spec.sample_path
        metadata = sample_spec.metadata

        if data_type == "type1":
            source = _resolve_referenced_image(
                metadata=metadata, sample_path=dataset_sample, dataset_root=self.dataset_root, data_type=data_type
            )
            generated_path = find_first_image(sample_dir)
            if source is None or generated_path is None:
                return None
            generated = Image.open(generated_path).convert("RGB")
            raw_scores = self.judge.evaluate(prompt=_type1_prompt(metadata), images=[source, generated])
        elif data_type == "type2":
            frames = [Image.open(sample_dir / f"frame{i}.png").convert("RGB") for i in range(6)]
            goal = _text_or_default(metadata.get("question") or metadata.get("caption"), "Complete the task.")
            prompt_suffix, judge_images = _trajectory_judge_payload(frames)
            raw_scores = self.judge.evaluate(
                prompt=f"Evaluate a six-frame GUI trajectory.\nTask: {goal}\n{prompt_suffix}",
                images=judge_images,
            )
        elif data_type in {"type3", "type4"}:
            frames = [Image.open(sample_dir / f"frame{i}.png").convert("RGB") for i in range(6)]
            instruction = _text_or_default(metadata.get("instruction") or metadata.get("caption"), "Complete the task.")
            prompt_suffix, judge_images = _trajectory_judge_payload(frames)
            raw_scores = self.judge.evaluate(
                prompt=f"Evaluate a six-frame GUI trajectory.\nInstruction: {instruction}\n{prompt_suffix}",
                images=judge_images,
            )
        elif data_type == "type5":
            source = _resolve_referenced_image(
                metadata=metadata, sample_path=dataset_sample, dataset_root=self.dataset_root, data_type=data_type
            )
            generated_path = find_first_image(sample_dir)
            if source is None or generated_path is None:
                return None
            generated = Image.open(generated_path).convert("RGB")
            raw_scores = self.judge.evaluate(prompt=_type5_prompt(metadata), images=[source, generated])
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        scores = _normalize_scores(raw_scores)
        return {
            "sample_name": f"{lang_device}/{sample_name}",
            "data_type": data_type,
            "scores": scores,
            "overall": _compute_overall(scores),
            "raw_scores": raw_scores,
        }


def _data_types_arg(value: str) -> list[str]:
    return list(TYPE_TO_FOLDER.keys()) if value == "all" else [value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local GEBench generation and scoring against vLLM-Omni.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--dataset-root", type=Path, required=True)
    generate.add_argument("--output-root", type=Path, required=True)
    generate.add_argument("--base-url", type=str, required=True)
    generate.add_argument("--model", type=str, required=True)
    generate.add_argument("--data-type", choices=["all", *TYPE_TO_FOLDER.keys()], default="all")
    generate.add_argument("--api-key", type=str, default="EMPTY")
    generate.add_argument("--width", type=int, default=768)
    generate.add_argument("--height", type=int, default=576)
    generate.add_argument("--num-inference-steps", type=int, default=8)
    generate.add_argument("--output-compression", type=int, default=98)
    generate.add_argument("--guidance-scale", type=float, default=None)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--workers", type=int, default=1)
    generate.add_argument("--max-samples", type=int, default=None)
    generate.add_argument("--samples-per-type", type=int, default=None)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--dataset-root", type=Path, required=True)
    evaluate.add_argument("--output-root", type=Path, required=True)
    evaluate.add_argument("--data-type", choices=["all", *TYPE_TO_FOLDER.keys()], default="all")
    evaluate.add_argument("--judge-base-url", type=str, required=True)
    evaluate.add_argument("--judge-model", type=str, required=True)
    evaluate.add_argument("--judge-api-key", type=str, default="EMPTY")
    evaluate.add_argument("--workers", type=int, default=1)
    evaluate.add_argument("--max-samples", type=int, default=None)
    evaluate.add_argument("--samples-per-type", type=int, default=None)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--output-root", type=Path, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate":
        runner = GEBenchRunner(
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            width=args.width,
            height=args.height,
            num_inference_steps=args.num_inference_steps,
            output_compression=args.output_compression,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
        records: list[dict[str, Any]] = []
        for data_type in _data_types_arg(args.data_type):
            records.extend(
                runner.generate(
                    data_type=data_type,
                    workers=args.workers,
                    max_samples=args.max_samples,
                    samples_per_type=args.samples_per_type,
                )
            )
        payload = {"records": records, "summary": summarize_generated_records(records)}
        write_json(args.output_root / "generation_manifest.json", payload)
        return 0

    if args.command == "evaluate":
        judge = LocalJudgeClient(
            base_url=args.judge_base_url,
            api_key=args.judge_api_key,
            model=args.judge_model,
        )
        evaluator = GEBenchEvaluator(dataset_root=args.dataset_root, output_root=args.output_root, judge=judge)
        combined_results: list[dict[str, Any]] = []
        for data_type in _data_types_arg(args.data_type):
            payload = evaluator.evaluate(
                data_type=data_type,
                workers=args.workers,
                max_samples=args.max_samples,
                samples_per_type=args.samples_per_type,
            )
            combined_results.extend(payload["results"])
        _write_json_with_timestamp(
            args.output_root / "evaluations" / "summary.json",
            {"summary": summarize_gebench_results(combined_results)},
        )
        return 0

    if args.command == "summarize":
        generation_summary = collect_gebench_generation_summary(args.output_root)
        evaluation_dir = args.output_root / "evaluations"
        result_records: list[dict[str, Any]] = []
        if evaluation_dir.exists():
            for file_path in sorted(evaluation_dir.glob("type*.json")):
                payload = load_json(file_path)
                result_records.extend(payload.get("results", []))
        payload: dict[str, Any] = {"generation": generation_summary}
        if result_records:
            payload["evaluation"] = summarize_gebench_results(result_records)
        _write_json_with_timestamp(args.output_root / "summary.json", payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 1
