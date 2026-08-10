"""Video-MME dataset loader for vLLM-Omni bench serve.

Video-MME (lmms-lab/Video-MME) is a full-spectrum video MLLM benchmark with
900 videos / 2,700 MCQ pairs across short / medium / long durations.

This loader follows OpenBMB OmniEvalKit's MiniCPM-o recipe
(``o_e_Kit/configs/generation_configs.json``):

- ``videomme`` (default): frames only, ``max_frames=96``, ``load_av=false``
- ``videomme_short``-style: interleaved AV at 1fps via ``minicpm-interleave``

Prompt template and option formatting match OmniEvalKit ``minicpmo.py``; frame
timestamps match OmniEvalKit ``_sample_video_frame_indices``.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

try:
    from vllm.benchmarks.datasets import BenchmarkDataset, SampleRequest
except ImportError:
    from vllm.benchmarks.datasets import HuggingFaceDataset as BenchmarkDataset
    from vllm.benchmarks.datasets import SampleRequest
from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.hf import get_cached_tokenizer

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

from vllm_omni.benchmarks.data_modules.daily_omni_dataset import (
    MINICPM_OMNI_SYSTEM_TEXT,
    _ListDatasetIterator,
    _numpy_to_wav_bytes,
    _pil_to_jpeg_bytes,
    _uniform_sample_indices,
)

logger = logging.getLogger(__name__)

_MINICPM_AUDIO_SR = 16000

# OmniEvalKit MiniCPM ``videomme`` / ``videomme_short`` defaults.
VIDEOMME_DEFAULT_MAX_FRAMES = 96
VIDEOMME_SHORT_MAX_FRAMES = 64
VIDEOMME_DEFAULT_HF_REPO = "lmms-lab/Video-MME"

#: Decode forward to reach a target this close, seek to the preceding keyframe beyond it.
_SEEK_THRESHOLD_S = 3.0

#: How often ``sample`` reports progress while warming a cold frame cache.
_SAMPLE_PROGRESS_INTERVAL_S = 30.0

VideoMMEPackMode = Literal["minicpm-frames", "minicpm-interleave", "video_url"]
VideoMMEDurationFilter = Literal["all", "short", "medium", "long"]

# OmniEvalKit ``generation_configs.json`` → ``videomme.user_prompt`` (``{media}`` stripped;
# media parts are prepended as OpenAI content parts, matching ``build_content``).
VIDEOMME_USER_PROMPT_TEMPLATE = (
    "Carefully read the following question and select the letter corresponding to the "
    "correct answer.Highlight the applicable choices without giving explanations.\n"
    "{question}\n"
    "Options:\n"
    "{options}\n"
    "Please select the correct answer from the options above. Only respond with the letter."
)

_OPTION_LETTERS = "ABCDEFGHIJKL"
_OPTION_PREFIX_RE = re.compile(r"^([A-L])\s*[.、:：)]\s*(.*)$")

_VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".avi")


def _iter_video_files(root: Path) -> Iterator[Path]:
    """Yield video files under ``root``, skipping dot-dirs such as the frame cache."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if os.path.splitext(name)[1].lower() in _VIDEO_SUFFIXES:
                yield Path(dirpath) / name


def _resolve_hf_cache_snapshot(path: Path) -> Path:
    """Map a HF hub cache dir (``datasets--org--name``) to its checked-out snapshot dir."""
    snapshots = path / "snapshots"
    if not snapshots.is_dir():
        return path
    ref = path / "refs" / "main"
    if ref.is_file():
        try:
            target = snapshots / ref.read_text(encoding="utf-8").strip()
        except OSError:
            target = snapshots
        if target.is_dir():
            return target
    revisions = [p for p in snapshots.iterdir() if p.is_dir()]
    if revisions:
        return max(revisions, key=lambda p: p.stat().st_mtime)
    return path


def resolve_videomme_local_root(dataset_path: str | None) -> Path | None:
    """Return a local Video-MME root if ``dataset_path`` points at an on-disk mirror.

    Hub ids such as ``lmms-lab/Video-MME`` return ``None`` (they are not directories).
    """
    raw = (dataset_path or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_dir():
        return None
    return _resolve_hf_cache_snapshot(path.resolve())


def ensure_videomme_hub_root(repo_id: str) -> Path:
    """Download the Video-MME dataset snapshot from Hugging Face and return its root.

    Mirrors :func:`vllm_omni.benchmarks.data_modules.seed_tts_dataset.resolve_seed_tts_root`:
    Hub ids trigger ``snapshot_download``; callers then extract videos via
    :func:`ensure_videomme_videos_extracted`.

    Raises:
        ImportError: if ``huggingface_hub`` is not installed.
        ValueError: if ``repo_id`` is empty.
    """
    rid = (repo_id or "").strip()
    if not rid:
        raise ValueError("repo_id is required to download Video-MME from Hugging Face")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "Install huggingface_hub to download Video-MME from the Hub, or pass a local "
            "--dataset-path / --videomme-video-dir with parquet + extracted videos."
        ) from e

    # Pull QA parquet + video/subtitle archives; skip unrelated repo files.
    cache = snapshot_download(
        repo_id=rid,
        repo_type="dataset",
        allow_patterns=[
            "videomme/**",
            "*.parquet",
            "videos_chunked_*.zip",
            "subtitle.zip",
            "subtitle/**",
            "video/**",
            "videos/**",
        ],
    )
    root = _resolve_hf_cache_snapshot(Path(cache).resolve())
    logger.info("Video-MME Hub snapshot ready at %s (repo=%s)", root, rid)
    return root


def resolve_videomme_root(dataset_path: str | None) -> Path:
    """Return a Video-MME root containing parquet and (or zip) video assets.

    * Existing local directories (absolute or relative) are used as-is.
    * Otherwise ``dataset_path`` is treated as a Hugging Face dataset id and
      downloaded via :func:`ensure_videomme_hub_root` (default
      ``lmms-lab/Video-MME``).
    """
    local = resolve_videomme_local_root(dataset_path)
    if local is not None:
        return local
    rid = (dataset_path or "").strip() or VIDEOMME_DEFAULT_HF_REPO
    return ensure_videomme_hub_root(rid)


def videomme_local_parquet(root: Path) -> Path | None:
    for candidate in (
        root / "videomme" / "test-00000-of-00001.parquet",
        root / "test-00000-of-00001.parquet",
    ):
        if candidate.is_file():
            return candidate
    return None


def videomme_local_video_dir(root: Path) -> Path | None:
    """Return a directory already holding extracted videos, flat or nested.

    Unzipping ``videos_chunked_*.zip`` in place yields ``videos/videos_chunked_NN/data/*.mp4``,
    so probing only for ``video/*.mp4`` would miss it and trigger a redundant re-extract of
    the whole ~95GB archive set. Lookups below walk the tree, so any depth is acceptable.
    """
    for name in ("video", "videos", "data"):
        candidate = root / name
        if candidate.is_dir() and next(_iter_video_files(candidate), None) is not None:
            return candidate
    return None


def videomme_local_subtitle_dir(root: Path) -> Path | None:
    candidate = root / "subtitle"
    return candidate if candidate.is_dir() else None


def _unzip_member_flat(zf: zipfile.ZipFile, member: str, dest_dir: Path) -> None:
    """Extract a zip member into ``dest_dir`` using only its basename (VLMEvalKit layout)."""
    name = os.path.basename(member)
    if not name:
        return
    dest = dest_dir / name
    if dest.is_file() and dest.stat().st_size > 0:
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as src, open(dest, "wb") as out:
        out.write(src.read())


def ensure_videomme_videos_extracted(root: Path) -> Path:
    """Ensure ``video/*.mp4`` exists under ``root`` by unzipping ``videos_chunked_*.zip``."""
    video_dir = root / "video"
    marker = root / ".videomme_videos_extracted"
    if marker.is_file() and video_dir.is_dir() and next(video_dir.glob("*.mp4"), None) is not None:
        return video_dir

    # Honour a tree someone unzipped by hand rather than re-extracting ~95GB alongside it.
    existing = videomme_local_video_dir(root)
    if existing is not None and existing != video_dir:
        return existing

    zips = [z for z in sorted(root.glob("videos_chunked_*.zip")) if z.is_file()]
    if not zips and not video_dir.is_dir():
        raise FileNotFoundError(f"No Video-MME videos under {root}: expected video/*.mp4 or videos_chunked_*.zip")

    video_dir.mkdir(parents=True, exist_ok=True)
    for zp in zips:
        logger.info("Extracting Video-MME videos from %s -> %s", zp, video_dir)
        with zipfile.ZipFile(zp, "r") as zf:
            for member in zf.namelist():
                if member.lower().endswith((".mp4", ".mkv", ".webm", ".avi")):
                    _unzip_member_flat(zf, member, video_dir)

    marker.write_text("ok", encoding="utf-8")
    return video_dir


def ensure_videomme_subtitles_extracted(root: Path) -> Path | None:
    """Ensure ``subtitle/*.srt`` exists under ``root`` when ``subtitle.zip`` is present."""
    subtitle_dir = root / "subtitle"
    if subtitle_dir.is_dir() and next(subtitle_dir.glob("*.srt"), None) is not None:
        return subtitle_dir

    zip_path = root / "subtitle.zip"
    if not zip_path.is_file():
        return subtitle_dir if subtitle_dir.is_dir() else None

    subtitle_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting Video-MME subtitles from %s -> %s", zip_path, subtitle_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if member.lower().endswith(".srt"):
                _unzip_member_flat(zf, member, subtitle_dir)
    return subtitle_dir


def _probe_video_duration(video_path: Path) -> float:
    """Container duration in seconds, without decoding the stream.

    Frame-counting fallbacks are deliberately avoided: Video-MME long videos run up to
    an hour, and ``stream.frames == 0`` containers would otherwise force a full decode
    just to learn the duration.
    """
    import av

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        duration: float | None = None
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = container.duration / av.time_base
        elif stream.frames and stream.average_rate:
            duration = stream.frames / float(stream.average_rate)

    if not duration or duration <= 0:
        raise ValueError(f"Could not determine duration of {video_path}")
    return duration


def _decode_frames_at_timestamps(
    video_path: Path,
    timestamps: list[float],
    *,
    seek_threshold_s: float = _SEEK_THRESHOLD_S,
) -> list[Any]:
    """Decode one RGB ``PIL.Image`` at (or just after) each ascending timestamp.

    A single forward pass over the stream — the Daily-Omni approach — is not affordable
    here: Video-MME samples points spread across the whole video, so the last target sits
    near the end and every frame of an hour-long file would be decoded. Targets within
    ``seek_threshold_s`` of the decoder position are reached by decoding forward (cheaper
    than a keyframe round-trip at 1fps spacing); farther ones seek to the preceding
    keyframe first.
    """
    import av

    if not timestamps:
        return []

    images: list[Any] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = float(stream.time_base) if stream.time_base else 0.0
        start_s = float(stream.start_time * stream.time_base) if stream.start_time else 0.0

        decoder = None
        frame = None
        frame_time = -1.0
        image = None

        for ts in timestamps:
            if decoder is None or ts < frame_time or (ts - frame_time) > seek_threshold_s:
                if time_base > 0:
                    container.seek(int((ts + start_s) / time_base), stream=stream, backward=True)
                else:
                    container.seek(0, stream=stream, backward=True)
                decoder = container.decode(stream)
                frame, image, frame_time = None, None, -1.0

            while frame is None or frame_time < ts:
                nxt = next(decoder, None)
                if nxt is None:
                    break
                frame, image = nxt, None
                if nxt.pts is not None and time_base > 0:
                    frame_time = float(nxt.pts) * time_base - start_s

            if frame is None:
                break
            if image is None:
                image = frame.to_image()
            images.append(image)

    if not images:
        raise ValueError(f"Decoded no frames from {video_path} at timestamps {timestamps[:4]}")
    # Container duration can overshoot the real stream; repeat the last frame (decord clamping).
    images.extend([images[-1]] * (len(timestamps) - len(images)))
    return images


@dataclass
class VideoMMESampleRequest(SampleRequest):
    """``SampleRequest`` with Video-MME gold labels for post-run accuracy scoring."""

    videomme_gold_answer: str = ""
    videomme_video_id: str = ""
    videomme_question_id: str = ""
    videomme_duration: str = ""
    videomme_domain: str = ""
    videomme_sub_category: str = ""
    videomme_task_type: str = ""
    omni_extra_body: dict[str, Any] | None = None
    omni_chat_messages: list[dict[str, Any]] | None = None
    omni_chat_mm_position: Literal["first", "last"] = "first"


class VideoMMEDataset(BenchmarkDataset):
    """Video-MME MCQ dataset for Omni bench serve (MiniCPM-oriented packing)."""

    SUPPORTED_DATASET_PATHS: set[str] = {VIDEOMME_DEFAULT_HF_REPO}
    DEFAULT_HF_DATASET_ID = VIDEOMME_DEFAULT_HF_REPO
    IS_MULTIMODAL = True
    DEFAULT_OUTPUT_LEN = 128

    def __init__(
        self,
        dataset_path: str | None = None,
        dataset_split: str = "test",
        random_seed: int = 0,
        video_dir: str | None = None,
        subtitle_dir: str | None = None,
        parquet_path: str | None = None,
        pack_mode: VideoMMEPackMode = "minicpm-frames",
        max_frames: int | None = None,
        duration_filter: VideoMMEDurationFilter = "all",
        use_subtitle: bool = False,
        inline_local_video: bool = False,
        trust_remote_code: bool = False,
        dataset_subset: str | None = None,
        no_stream: bool = False,
        **kwargs,
    ) -> None:
        if pack_mode not in ("minicpm-frames", "minicpm-interleave", "video_url"):
            raise ValueError(
                f"pack_mode must be 'minicpm-frames', 'minicpm-interleave', or 'video_url', got {pack_mode!r}"
            )
        if duration_filter not in ("all", "short", "medium", "long"):
            raise ValueError(f"duration_filter must be all|short|medium|long, got {duration_filter!r}")
        if parquet_path is None and dataset_path is None:
            raise ValueError("Either 'parquet_path' (local) or 'dataset_path' (HF id / local mirror) must be provided.")

        self.parquet_path = Path(parquet_path) if parquet_path else None
        self.dataset_path = dataset_path
        self.dataset_split = dataset_split
        self.dataset_subset = dataset_subset
        self._hf_streaming = not no_stream
        self.video_dir = Path(video_dir) if video_dir else None
        self.subtitle_dir = Path(subtitle_dir) if subtitle_dir else None
        self.pack_mode: VideoMMEPackMode = pack_mode
        self.duration_filter: VideoMMEDurationFilter = duration_filter
        self.use_subtitle = use_subtitle
        self.inline_local_video = inline_local_video
        self.trust_remote_code = trust_remote_code
        self.max_frames = int(
            max_frames
            if max_frames is not None
            else (VIDEOMME_SHORT_MAX_FRAMES if pack_mode == "minicpm-interleave" else VIDEOMME_DEFAULT_MAX_FRAMES)
        )
        #: In-process memo of content parts; the on-disk frame cache survives across runs.
        self._frame_cache: dict[str, list[dict[str, Any]]] = {}
        self._video_index: dict[str, Path] | None = None

        super().__init__(
            dataset_path=dataset_path if self.parquet_path is None else None,
            random_seed=random_seed,
            **kwargs,
        )
        self.load_data()
        logger.info(
            "Loaded Video-MME: source=%s, pack_mode=%s, max_frames=%d, duration_filter=%s, "
            "use_subtitle=%s, video_dir=%s",
            str(self.parquet_path) if self.parquet_path else f"{dataset_path}/{dataset_split}",
            pack_mode,
            self.max_frames,
            duration_filter,
            use_subtitle,
            self.video_dir,
        )

    # ------------------------------------------------------------------ loading

    def load_data(self) -> None:
        if self.parquet_path is not None:
            self._load_from_parquet(self.parquet_path)
            return

        local_root = resolve_videomme_local_root(self.dataset_path)
        if local_root is not None:
            local_pq = videomme_local_parquet(local_root)
            if local_pq is not None:
                if self.video_dir is None:
                    try:
                        self.video_dir = ensure_videomme_videos_extracted(local_root)
                    except FileNotFoundError:
                        self.video_dir = videomme_local_video_dir(local_root)
                if self.subtitle_dir is None:
                    self.subtitle_dir = ensure_videomme_subtitles_extracted(local_root)
                self._load_from_parquet(local_pq)
                return

        self._load_from_huggingface()

    def _load_from_parquet(self, path: Path) -> None:
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("pandas is required to load Video-MME parquet") from e
        if not path.is_file():
            raise FileNotFoundError(f"Video-MME parquet not found: {path}")
        # aarch64 pyarrow 25 can segfault on dict-encoded pages; duckdb reader is safe.
        try:
            import duckdb

            cur = duckdb.connect().execute(f"SELECT * FROM read_parquet('{path}')")
            df = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
        except ImportError:
            df = pd.read_parquet(path)
        self._set_rows([row.to_dict() for _, row in df.iterrows()])

    def _load_from_huggingface(self) -> None:
        if load_dataset is None:
            raise ImportError(
                "datasets library is required for HuggingFace Video-MME loading. "
                "Install with: pip install datasets, or pass --videomme-parquet."
            )
        load_kw: dict[str, Any] = {
            "split": self.dataset_split,
            "streaming": self._hf_streaming,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.dataset_subset is not None:
            load_kw["name"] = self.dataset_subset
        ds = load_dataset(self.dataset_path, **load_kw)
        self._set_rows([self._coerce_row(item) for item in ds])

    def _set_rows(self, rows: list[dict[str, Any]]) -> None:
        """Apply the duration filter and shuffle, then expose rows as ``self.data``."""
        if self.duration_filter != "all":
            want = self.duration_filter.lower()
            rows = [r for r in rows if str(r.get("duration") or "").strip().lower() == want]
        if not getattr(self, "disable_shuffle", False) and self.random_seed is not None:
            import random

            rows = rows[:]
            random.Random(self.random_seed).shuffle(rows)
        self.data = _ListDatasetIterator(rows)

    @staticmethod
    def _coerce_row(item: Any) -> dict[str, Any]:
        if isinstance(item, dict):
            return item
        if hasattr(item, "as_py"):
            return dict(item.as_py())
        try:
            return dict(item)
        except (TypeError, ValueError):
            return {k: item[k] for k in item}  # type: ignore[misc]

    @staticmethod
    def _normalize_fields(row: dict[str, Any]) -> dict[str, Any]:
        # The parquet hands back a numpy array, whose truthiness raises, so an ``or`` chain
        # over the alias keys is not usable here.
        options: Any = []
        for key in ("options", "candidates", "choices"):
            value = row.get(key)
            if hasattr(value, "tolist"):
                value = value.tolist()
            if value is not None and len(value) > 0:
                options = value
                break
        return {
            "video_id": str(row.get("videoID") or row.get("video_id") or row.get("video") or "").strip(),
            "question_id": str(row.get("question_id") or "").strip(),
            "question": str(row.get("question") or "").strip(),
            "options": [str(x) for x in list(options)],
            "answer": str(row.get("answer") or "").strip(),
            "duration": str(row.get("duration") or "").strip(),
            "domain": str(row.get("domain") or "").strip(),
            "sub_category": str(row.get("sub_category") or "").strip(),
            "task_type": str(row.get("task_type") or "").strip(),
        }

    # ------------------------------------------------------------------ sampling

    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        output_len: int | None = None,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs,
    ) -> list[SampleRequest]:
        if output_len is None:
            output_len = self.DEFAULT_OUTPUT_LEN

        sampled: list[SampleRequest] = []
        cached_tokenizer = get_cached_tokenizer(tokenizer)
        # A cold frame cache costs a few seconds per video, so the whole set takes tens of
        # minutes before the first request goes out; report progress to keep that legible.
        started = time.monotonic()
        last_report = started
        try:
            total = min(num_requests, len(self.data))
        except TypeError:  # HF streaming datasets are not sized
            total = num_requests
        for seen, item in enumerate(self.data, start=1):
            if len(sampled) >= num_requests:
                break
            req = self._create_sample_request(
                self._coerce_row(item), cached_tokenizer, output_len, request_id_prefix, len(sampled)
            )
            if req:
                sampled.append(req)
            now = time.monotonic()
            if now - last_report >= _SAMPLE_PROGRESS_INTERVAL_S:
                logger.info(
                    "Video-MME preparing requests: %d/%d rows, %d ready, %.0fs elapsed",
                    seen,
                    total,
                    len(sampled),
                    now - started,
                )
                last_report = now

        logger.info("Created %d sample requests from Video-MME in %.0fs", len(sampled), time.monotonic() - started)
        self.maybe_oversample_requests(sampled, num_requests, request_id_prefix, no_oversample)
        return sampled

    def _create_sample_request(
        self,
        qa_item: dict[str, Any],
        tokenizer: TokenizerLike,
        output_len: int,
        request_id_prefix: str,
        index: int,
    ) -> SampleRequest | None:
        fields = self._normalize_fields(qa_item)
        if not fields["video_id"] or not fields["question"]:
            logger.warning("Skipping Video-MME item without videoID/question: %r", fields["video_id"])
            return None

        mm_payload, omni_extra = self._compose_multimodal(fields["video_id"])
        if not mm_payload:
            return None

        user_text = self._build_user_prompt(fields)
        return VideoMMESampleRequest(
            prompt=user_text,
            prompt_len=len(tokenizer.encode(user_text)),
            expected_output_len=output_len,
            multi_modal_data=None,
            request_id=f"{request_id_prefix}{index}",
            videomme_gold_answer=fields["answer"],
            videomme_video_id=fields["video_id"],
            videomme_question_id=fields["question_id"],
            videomme_duration=fields["duration"],
            videomme_domain=fields["domain"],
            videomme_sub_category=fields["sub_category"],
            videomme_task_type=fields["task_type"],
            omni_extra_body=omni_extra,
            omni_chat_messages=self._build_openai_messages(mm_payload, user_text),
            omni_chat_mm_position="first",
        )

    # ------------------------------------------------------------------ prompt

    @staticmethod
    def _format_options(options: list[str]) -> str:
        """Match OmniEvalKit ``_build_options_prompt`` (``A. content`` per line).

        The Hub parquet already ships options as ``"A. content"``. Existing labels are
        dropped before relabelling so the prompt never reads ``A. A. content``, but only
        when *every* option carries one — otherwise contents such as ``A.M. shift`` would
        be truncated.
        """
        if not options:
            return ""
        matches = [_OPTION_PREFIX_RE.match(str(o).strip()) for o in options]
        relabel = all(m is not None and m.group(1) == key for m, key in zip(matches, _OPTION_LETTERS))
        lines = []
        for key, raw, m in zip(_OPTION_LETTERS, options, matches):
            content = m.group(2).strip() if (relabel and m is not None) else str(raw).strip()
            lines.append(f"{key}. {content}")
        return "\n".join(lines)

    def _build_user_prompt(self, fields: dict[str, Any]) -> str:
        prompt = VIDEOMME_USER_PROMPT_TEMPLATE.format(
            question=fields["question"],
            options=self._format_options(fields["options"]),
        )
        if self.use_subtitle:
            subs = self._load_subtitle_text(fields["video_id"])
            if subs:
                prompt = f"This video's subtitles are listed below:\n{subs}\n{prompt}"
        return prompt

    def _load_subtitle_text(self, video_id: str) -> str:
        if not self.subtitle_dir:
            return ""
        path = self.subtitle_dir / f"{video_id}.srt"
        if not path.is_file():
            return ""
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return ""
        lines = [
            s.replace("\\N", " ")
            for s in (line.strip() for line in raw.splitlines())
            if s and not s.isdigit() and "-->" not in s
        ]
        return "\n".join(lines)

    def _build_openai_messages(
        self,
        mm_payload: dict[str, Any] | list[dict[str, Any]],
        user_text: str,
    ) -> list[dict[str, Any]]:
        mm_list = mm_payload if isinstance(mm_payload, list) else [mm_payload]
        messages: list[dict[str, Any]] = []
        if MINICPM_OMNI_SYSTEM_TEXT:
            messages.append({"role": "system", "content": [{"type": "text", "text": MINICPM_OMNI_SYSTEM_TEXT}]})
        messages.append({"role": "user", "content": [*mm_list, {"type": "text", "text": user_text}]})
        return messages

    # ------------------------------------------------------------------ media

    def _video_path_index(self) -> dict[str, Path]:
        """Lazily map ``videoID -> path`` so nested layouts resolve after the flat probe misses."""
        if self._video_index is None:
            index: dict[str, Path] = {}
            if self.video_dir is not None:
                for path in _iter_video_files(self.video_dir):
                    index.setdefault(path.stem, path)
            logger.info("Indexed %d Video-MME video files under %s", len(index), self.video_dir)
            self._video_index = index
        return self._video_index

    def _resolve_local_video_path(self, video_id: str) -> Path | None:
        if not self.video_dir or not video_id:
            return None
        for p in (
            self.video_dir / f"{video_id}.mp4",
            self.video_dir / f"{video_id}.MP4",
            self.video_dir / f"{video_id}.mkv",
            self.video_dir / video_id / f"{video_id}.mp4",
        ):
            if p.is_file():
                return p
        return self._video_path_index().get(video_id)

    def _media_part(self, path: Path, *, typ: str, mime: str) -> dict[str, Any]:
        path = path.expanduser().resolve()
        if self.inline_local_video:
            b64 = base64.b64encode(path.read_bytes()).decode("ascii")
            return {"type": typ, typ: {"url": f"data:{mime};base64,{b64}"}}
        return {"type": typ, typ: {"url": path.as_uri()}}

    def _compose_multimodal(
        self,
        video_id: str,
    ) -> tuple[dict[str, Any] | list[dict[str, Any]] | None, dict[str, Any] | None]:
        # OmniEvalKit ``minicpmo.py`` uses max_slice_nums=1 / use_image_id=False whenever the
        # sample carries a video path: 9 slices per frame would blow past max_model_len and
        # <image_id>N</image_id> would label each frame as a separate picture.
        extra: dict[str, Any] = {
            "mm_processor_kwargs": {
                "use_audio_in_video": False,
                "max_slice_nums": 1,
                "use_image_id": False,
            }
        }

        if self.pack_mode == "video_url":
            video_path = self._resolve_local_video_path(video_id)
            if video_path is None:
                logger.warning("Video-MME video not found for video_id=%r under %s", video_id, self.video_dir)
                return None, None
            return self._media_part(video_path, typ="video_url", mime="video/mp4"), extra

        parts = self._get_minicpm_frame_parts(video_id, include_audio=self.pack_mode == "minicpm-interleave")
        if not parts:
            return None, None
        return parts, extra

    def _frame_cache_dir(self, video_id: str, *, include_audio: bool) -> Path:
        assert self.video_dir is not None
        tag = f"f{self.max_frames}{'_av' if include_audio else ''}"
        return self.video_dir / ".minicpm_videomme_frames" / video_id / tag

    def _cached_parts_from_disk(self, cache_dir: Path, *, include_audio: bool) -> list[dict[str, Any]] | None:
        """Rebuild content parts from a previous run's frame dump, or ``None`` on a miss."""
        manifest = cache_dir / "manifest.json"
        if not manifest.is_file():
            return None
        try:
            count = int(json.loads(manifest.read_text(encoding="utf-8"))["count"])
        except (OSError, ValueError, KeyError, TypeError):
            return None

        parts: list[dict[str, Any]] = []
        for i in range(count):
            frame_path = cache_dir / f"frame_{i:04d}.jpg"
            if not frame_path.is_file():
                return None
            parts.append({"type": "image_url", "image_url": {"url": frame_path.as_uri()}})
            if include_audio:
                audio_path = cache_dir / f"audio_{i:04d}.wav"
                if not audio_path.is_file():
                    return None
                parts.append({"type": "audio_url", "audio_url": {"url": audio_path.as_uri()}})
        return parts or None

    def _get_minicpm_frame_parts(
        self,
        video_id: str,
        *,
        include_audio: bool,
    ) -> list[dict[str, Any]] | None:
        cache_key = f"{video_id}|audio={int(include_audio)}|frames={self.max_frames}"
        cached = self._frame_cache.get(cache_key)
        if cached is not None:
            return cached

        # Inline base64 must not be persisted or retained: 900 videos would pin GBs of RSS.
        cache_dir = None if self.inline_local_video else self._frame_cache_dir(video_id, include_audio=include_audio)
        if cache_dir is not None:
            disk_parts = self._cached_parts_from_disk(cache_dir, include_audio=include_audio)
            if disk_parts is not None:
                self._frame_cache[cache_key] = disk_parts
                return disk_parts

        video_path = self._resolve_local_video_path(video_id)
        if video_path is None:
            logger.warning(
                "Video-MME MiniCPM pack requires local video under video_dir=%s for video_id=%r",
                self.video_dir,
                video_id,
            )
            return None

        try:
            frames, audio_segments = self._extract_frames_and_audio(
                video_path,
                include_audio=include_audio,
                max_num_frames=self.max_frames,
            )
        except Exception:
            logger.exception("Failed Video-MME frame extract for video_id=%r path=%s", video_id, video_path)
            return None

        if not frames:
            logger.warning("No frames extracted for video_id=%r", video_id)
            return None
        if include_audio and len(audio_segments) != len(frames):
            n = min(len(frames), len(audio_segments))
            frames, audio_segments = frames[:n], audio_segments[:n]

        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

        parts: list[dict[str, Any]] = []
        for i, frame in enumerate(frames):
            parts.append(
                self._emit_media(cache_dir, f"frame_{i:04d}.jpg", _pil_to_jpeg_bytes(frame), "image_url", "image/jpeg")
            )
            if include_audio:
                parts.append(
                    self._emit_media(
                        cache_dir,
                        f"audio_{i:04d}.wav",
                        _numpy_to_wav_bytes(audio_segments[i]),
                        "audio_url",
                        "audio/wav",
                    )
                )

        if cache_dir is not None:
            (cache_dir / "manifest.json").write_text(json.dumps({"count": len(frames)}), encoding="utf-8")
            self._frame_cache[cache_key] = parts
        logger.debug("Video-MME packed video_id=%r frames=%d parts=%d", video_id, len(frames), len(parts))
        return parts

    @staticmethod
    def _emit_media(cache_dir: Path | None, name: str, payload: bytes, typ: str, mime: str) -> dict[str, Any]:
        """Write ``payload`` to the frame cache and return a file URL part, or inline base64."""
        if cache_dir is None:
            b64 = base64.b64encode(payload).decode("ascii")
            return {"type": typ, typ: {"url": f"data:{mime};base64,{b64}"}}
        path = cache_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            path.write_bytes(payload)
        return {"type": typ, typ: {"url": path.as_uri()}}

    @staticmethod
    def _sample_timestamps(duration: float, max_num_frames: int) -> list[float]:
        """Port of OmniEvalKit ``_sample_video_frame_indices`` timestamps."""
        if duration > max_num_frames:
            grid = [round(i * 0.1, 1) for i in range(int(duration / 0.1))]
            return [grid[i] for i in _uniform_sample_indices(len(grid), max_num_frames)]
        # OmniEvalKit uses int(duration); clamp to >=1 so sub-second clips still yield a frame.
        return [float(i) for i in range(max(1, int(duration)))]

    @classmethod
    def _extract_frames_and_audio(
        cls,
        video_path: Path,
        *,
        include_audio: bool,
        max_num_frames: int,
    ) -> tuple[list[Any], list[Any]]:
        import numpy as np
        from vllm.multimodal.media.audio import load_audio

        duration = _probe_video_duration(video_path)
        timestamps = cls._sample_timestamps(duration, max_num_frames)
        frames = _decode_frames_at_timestamps(video_path, timestamps)

        audio_segments: list[Any] = []
        if include_audio:
            audio_np, sr = load_audio(str(video_path), sr=_MINICPM_AUDIO_SR, mono=True)
            for i, start_time in enumerate(timestamps):
                end_time = timestamps[i + 1] if i < len(timestamps) - 1 else duration
                segment = audio_np[int(start_time * sr) : int(end_time * sr)]
                if i == len(timestamps) - 1 and len(segment) < 1600:
                    segment = np.concatenate([segment, np.zeros(1600 - len(segment), dtype=segment.dtype)])
                if len(segment) == 0:
                    segment = np.zeros(1600, dtype=np.float32)
                audio_segments.append(segment)

        return frames, audio_segments
