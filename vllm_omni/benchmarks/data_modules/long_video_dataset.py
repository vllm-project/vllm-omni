# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Local official annotations for continuous visual-session QA.

OVO uses the annotated prefix; Video-MME uses the full video. Each question
starts a fresh session so earlier answers cannot change later questions.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoQuestion:
    id: str
    dataset: str
    task: str
    video: Path
    question: str
    options: tuple[str, ...]
    answer: str
    end_s: float | None
    annotation_sha256: str

    @property
    def prompt(self) -> str:
        options = "\n".join(f"{chr(65 + i)}. {text}" for i, text in enumerate(self.options))
        return f"{self.question}\n{options}\nAnswer with only the letter of the correct option (A, B, C, or D)."


def load_video_questions(
    annotation: Path, media_root: Path, *, dataset: str, ids: list[str] | None = None
) -> list[VideoQuestion]:
    """Read official OVO JSON or Video-MME JSON/parquet, without downloading media."""
    digest = hashlib.sha256(annotation.read_bytes()).hexdigest()
    if annotation.suffix == ".parquet":
        import pyarrow.parquet as pq

        rows = pq.read_table(annotation).to_pylist()
    else:
        rows = json.loads(annotation.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Expected an annotation array")
    root = media_root.resolve()
    selected: list[VideoQuestion] = []
    wanted = set(ids) if ids else None
    seen: set[str] = set()
    for row in rows:
        if dataset == "ovo-bench":
            # Forward responding has a different protocol and scoring rule.
            # Explicit selection of such an ID fails below instead of silently
            # treating it as a multiple-choice retrospective question.
            if row["task"] not in {"EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD"}:
                continue
            questions = [row]
            video = root / str(row["video"])
            end = float(row["realtime"])
            if not math.isfinite(end) or end <= 0:
                raise ValueError(f"Invalid OVO prefix duration for {row['id']}")
        elif dataset == "videomme-streaming":
            questions = row.get("questions", [row])
            video = root / f"{row['videoID']}.mp4"
            end = None
        else:
            raise ValueError(f"Unknown visual-session dataset: {dataset}")
        for question in questions:
            identity = str(question["id"] if dataset == "ovo-bench" else question["question_id"])
            if wanted is not None and identity not in wanted:
                continue
            if identity in seen:
                raise ValueError(f"Duplicate question ID: {identity}")
            seen.add(identity)
            options = tuple(str(option) for option in question["options"])
            if len(options) != 4:
                raise ValueError(f"Expected four options for {identity}")
            if dataset == "ovo-bench":
                index = question["gt"]
                if type(index) is not int or not 0 <= index < 4:
                    raise ValueError(f"Invalid answer index for {identity}")
                answer = chr(65 + index)
            else:
                options = tuple(
                    text[3:] if text.startswith(f"{chr(65 + i)}. ") else text for i, text in enumerate(options)
                )
                answer = question["answer"]
                if answer not in {"A", "B", "C", "D"}:
                    raise ValueError(f"Invalid answer for {identity}")
            path = video.resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"Video must exist inside media root: {video}")
            selected.append(
                VideoQuestion(
                    identity,
                    dataset,
                    str(question.get("task", question.get("task_type", ""))),
                    path,
                    str(question["question"]),
                    options,
                    answer,
                    end,
                    digest,
                )
            )
    if wanted is not None and wanted - seen:
        raise ValueError(f"Missing or unsupported question IDs: {sorted(wanted - seen)}")
    if not selected:
        raise ValueError("No supported visual-session questions selected")
    return selected
