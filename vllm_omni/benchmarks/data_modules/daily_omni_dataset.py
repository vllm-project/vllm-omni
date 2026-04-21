"""Daily-Omni Dataset loader for benchmark.

Daily-Omni is an audio-visual reasoning benchmark with 684 videos
and 1,197 multiple-choice QA pairs across 6 major task types.

Dataset source: https://huggingface.co/datasets/liarliar/Daily-Omni

Supports loading QA metadata from:
- Local JSON file (``qa_json_path``): recommended for offline/air-gapped environments
- HuggingFace datasets (``dataset_path``): legacy online mode

The videos must be separately downloaded and extracted from Videos.tar.

Why ``BenchmarkDataset`` instead of ``HuggingFaceDataset``?
    vLLM's ``HuggingFaceDataset`` is a thin wrapper whose ``__init__`` always ends by calling
    ``load_data()`` → ``datasets.load_dataset(...)`` with a required Hub id and split. That
    contract fits "Hub-only" benches, but Daily-Omni also needs **offline QA metadata** from a
    local ``qa.json`` without touching the network. Subclassing ``HuggingFaceDataset`` would
    mean fighting the parent constructor (fake ``dataset_path``, reordering ``load_data``, or
    duplicating half the parent) and would still imply ``datasets`` is always relevant.

    This class therefore inherits only ``BenchmarkDataset`` (minimal: ``dataset_path``,
    ``random_seed``, ``self.data``) and implements **two explicit loaders**:
    ``_load_from_local_json`` (default path for air-gapped runs) and ``_load_from_huggingface``
    (optional legacy path for users who prefer ``datasets`` + Hub cache). The latter is **not**
    inheritance; it is the same Hub rows as before, factored into a helper so one class can
    serve both deployment modes without mandatory ``datasets`` when using ``qa_json_path``.

Usage:
    from vllm_omni.benchmarks.data_modules.daily_omni_dataset import DailyOmniDataset

    # Local JSON mode (recommended)
    dataset = DailyOmniDataset(
        qa_json_path="/path/to/qa.json",
        video_dir="/path/to/Videos",
        random_seed=42,
    )

    # HuggingFace mode (legacy, requires network)
    dataset = DailyOmniDataset(
        dataset_path="liarliar/Daily-Omni",
        dataset_split="train",
        random_seed=42,
    )
    requests = dataset.sample(
        tokenizer=tokenizer,
        num_requests=100,
        output_len=256,
    )
"""

import base64
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

try:
    from vllm.benchmarks.datasets import BenchmarkDataset, SampleRequest
except ImportError:
    # Fallback: if BenchmarkDataset not available, use base class from same module
    from vllm.benchmarks.datasets import HuggingFaceDataset as BenchmarkDataset
    from vllm.benchmarks.datasets import SampleRequest
from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.hf import get_cached_tokenizer

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

logger = logging.getLogger(__name__)


class _ListDatasetIterator:
    """Simple iterator wrapper around a list to mimic HuggingFace streaming dataset behavior."""

    def __init__(self, data: list[dict[str, Any]]) -> None:
        self._data = data
        self._index = 0

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self) -> dict[str, Any]:
        if self._index >= len(self._data):
            raise StopIteration
        item = self._data[self._index]
        self._index += 1
        return item

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        return self._data[idx]


# Aligns with Lliar-liar/Daily-Omni CLI ``--input_mode`` (test_model/*/testmodel.py).
DailyOmniInputMode = Literal["all", "visual", "audio"]

# ``build_conversation()`` in Daily-Omni ``test_model/Qwen2.5-Omni/testmodel.py`` (verbatim).
DAILY_OMNI_SYSTEM_TEXT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


@dataclass
class DailyOmniSampleRequest(SampleRequest):
    """``SampleRequest`` with Daily-Omni gold labels for post-run accuracy scoring."""

    daily_omni_gold_answer: str = ""
    daily_omni_video_id: str = ""
    daily_omni_task_type: str = ""
    #: Official qa.json ``video_duration`` (e.g. ``30s``, ``60s``) for leaderboard-style breakdown.
    daily_omni_video_duration: str = ""
    #: Official ``video_category`` (YouTube-style category string) for per-category accuracy.
    daily_omni_video_category: str = ""
    #: Extra JSON fields merged into chat-completions ``extra_body`` (e.g. ``mm_processor_kwargs``).
    omni_extra_body: dict[str, Any] | None = None
    #: Full OpenAI ``messages`` (system + user) mirroring upstream Daily-Omni conversation.
    omni_chat_messages: list[dict[str, Any]] | None = None
    #: Used only when ``omni_chat_messages`` is None (non-Daily-Omni-style requests).
    omni_chat_mm_position: Literal["first", "last"] = "last"


class DailyOmniDataset(BenchmarkDataset):
    """Daily-Omni audio-visual QA dataset for benchmarking.

    Inherits ``BenchmarkDataset`` only (not ``HuggingFaceDataset``): see module docstring for why
    Hub loading lives in ``_load_from_huggingface`` instead of subclassing the HF base class.

    The dataset includes:
    - 684 videos from daily life scenarios (available in Videos.tar)
    - 1,197 multiple-choice QA pairs in qa.json
    - 6 major task categories

    QA metadata can be loaded from:
    - Local JSON file (``qa_json_path``): recommended for offline/air-gapped environments
    - HuggingFace datasets (``dataset_path``): legacy online mode

    The videos must be separately downloaded and extracted from Videos.tar.

    Args:
        qa_json_path: Path to local qa.json file (offline mode, preferred). When provided,
            ``dataset_path`` and ``dataset_split`` are ignored.
        dataset_path: HuggingFace dataset path (e.g., "liarliar/Daily-Omni"). Used only if
            ``qa_json_path`` is not provided (legacy online mode).
        dataset_split: Dataset split to use (default: "train"). Used only in online mode.
        random_seed: Random seed for shuffling
        video_dir: Directory containing extracted video files (default: None)
        input_mode: Which modalities to send, matching upstream Daily-Omni ``--input_mode``:
            ``all`` — video + WAV (default; official audio-visual protocol);
            ``visual`` — video only;
            ``audio`` — extracted WAV only (requires ``{video_id}/{video_id}_audio.wav`` under ``video_dir``).
        max_duration_seconds: Reserved for future ffprobe-based filtering; currently **not applied**
            when building requests (metadata ``video_duration`` is still passed through for eval).
        dataset_subset: Optional HuggingFace subset name (``load_dataset(..., name=...)``); used by bench
            ``--hf-subset`` / patch.
        no_stream: If True, load the Hub split non-streaming (matches bench ``--no-stream``).
        inline_local_video: If True, embed local MP4 as ``data:video/mp4;base64,...`` in requests so
            the API server does not need ``--allowed-local-media-path`` (large JSON; use for small runs).
            When ``input_mode`` is ``audio`` or ``all``, local WAV is embedded the same way
            (``data:audio/wav;base64,...``).
        trust_remote_code: Whether to trust remote code when loading HuggingFace dataset
            (online mode only).
    """

    SUPPORTED_DATASET_PATHS: set[str] = {
        "liarliar/Daily-Omni",
    }
    #: Default Hub id for synthetic video URLs when ``qa_json_path`` is used (``dataset_path`` None).
    DEFAULT_HF_DATASET_ID = "liarliar/Daily-Omni"
    IS_MULTIMODAL = True
    DEFAULT_OUTPUT_LEN = 256

    def __init__(
        self,
        qa_json_path: str | None = None,
        dataset_path: str | None = None,
        dataset_split: str = "train",
        random_seed: int = 0,
        video_dir: str | None = None,
        input_mode: DailyOmniInputMode = "all",
        inline_local_video: bool = False,
        trust_remote_code: bool = False,
        max_duration_seconds: float | None = None,
        dataset_subset: str | None = None,
        no_stream: bool = False,
        **kwargs,
    ) -> None:
        if input_mode not in ("all", "visual", "audio"):
            raise ValueError(f"input_mode must be 'all', 'visual', or 'audio', got {input_mode!r}")

        # Validate arguments: need either local JSON or HF path
        if qa_json_path is None and dataset_path is None:
            raise ValueError(
                "Either 'qa_json_path' (local JSON) or 'dataset_path' (HuggingFace) must be provided. "
                "For offline/air-gapped environments, download qa.json and use qa_json_path."
            )

        # Store configuration
        self.qa_json_path = Path(qa_json_path) if qa_json_path else None
        self.dataset_path = dataset_path
        self.dataset_split = dataset_split
        self.dataset_subset = dataset_subset
        #: Match vLLM ``HuggingFaceDataset`` / bench CLI ``--no-stream``.
        self._hf_streaming = not no_stream
        self.video_dir = Path(video_dir) if video_dir else None
        self.inline_local_video = inline_local_video
        self.input_mode: DailyOmniInputMode = input_mode
        self.max_duration_seconds = max_duration_seconds
        self.trust_remote_code = trust_remote_code

        #: In-process cache of ffprobe durations only (no disk persistence).
        self._video_durations: dict[str, float] = {}

        # Initialize parent BenchmarkDataset
        super().__init__(
            dataset_path=dataset_path if qa_json_path is None else None,
            random_seed=random_seed,
            **kwargs,
        )

        # Load data based on mode
        self.load_data()

        # Verify dataset info
        logger.info(
            "Loaded Daily-Omni dataset: mode=%s, source=%s, random_seed=%d, input_mode=%s, max_duration=%s",
            "local_json" if self.qa_json_path else "huggingface",
            str(self.qa_json_path) if self.qa_json_path else f"{dataset_path}/{dataset_split}",
            random_seed,
            input_mode,
            f"{max_duration_seconds}s" if max_duration_seconds else "unlimited",
        )

    def load_data(self) -> None:
        """Populate ``self.data`` from either local JSON or the Hub.

        See module docstring: we do not subclass ``HuggingFaceDataset`` because Daily-Omni needs
        a first-class offline path; Hub loading is an optional branch implemented below.
        """
        if self.qa_json_path is not None:
            self._load_from_local_json()
        else:
            self._load_from_huggingface()

    def _load_from_local_json(self) -> None:
        """Load QA data from local JSON file."""
        if not self.qa_json_path.exists():
            raise FileNotFoundError(f"QA JSON file not found: {self.qa_json_path}")

        with open(self.qa_json_path, encoding="utf-8") as f:
            data = json.load(f)

        # Support both list format and dict with "train"/"test" splits
        if isinstance(data, dict):
            # Try to get the requested split, fallback to first available
            split_data = data.get(self.dataset_split)
            if split_data is None:
                available = list(data.keys())
                if available:
                    logger.warning(
                        "Split '%s' not found in %s, using '%s' instead",
                        self.dataset_split,
                        self.qa_json_path,
                        available[0],
                    )
                    split_data = data[available[0]]
                else:
                    split_data = []
            data = split_data

        if not isinstance(data, list):
            raise ValueError(f"Expected list of QA items in JSON, got {type(data).__name__}")

        # Shuffle if requested
        if not getattr(self, "disable_shuffle", False) and self.random_seed is not None:
            import random

            rng = random.Random(self.random_seed)
            shuffled = data[:]
            rng.shuffle(shuffled)
            data = shuffled

        # Create an iterator-like wrapper for compatibility
        self.data = _ListDatasetIterator(data)

    def _load_from_huggingface(self) -> None:
        """Load QA rows via ``datasets.load_dataset`` (legacy / convenience path).

        Kept for backward compatibility: callers can still pass ``dataset_path=liarliar/Daily-Omni``
        and get the same parquet-backed rows as the Hub dataset card, with streaming (or
        non-streaming if ``no_stream=True``) and shuffle.

        This is intentionally **not** implemented by subclassing ``HuggingFaceDataset``: that base
        always runs Hub ``load_dataset`` from its constructor and expects a Hub id as the primary
        API; Daily-Omni instead chooses the source in ``load_data()`` (JSON vs Hub) while sharing
        one ``sample()`` / request-building implementation for both.
        """
        if load_dataset is None:
            raise ImportError(
                "datasets library is required for HuggingFace mode. "
                "Install with: pip install datasets, or use local JSON mode instead."
            )

        ds = load_dataset(
            self.dataset_path,
            name=self.dataset_subset,
            split=self.dataset_split,
            streaming=self._hf_streaming,
            trust_remote_code=self.trust_remote_code,
        )
        if not getattr(self, "disable_shuffle", False):
            ds = ds.shuffle(seed=self.random_seed)
        self.data = ds

    def get_task_statistics(self) -> dict[str, int]:
        """Get distribution of task types in the dataset.

        Returns:
            Dict mapping task type to count
        """
        stats: dict[str, int] = {}
        for item in self.data:
            row = self._coerce_row(item)
            fields = self._normalize_qa_fields(row)
            task_type = fields["task_type"] or "unknown"
            stats[task_type] = stats.get(task_type, 0) + 1
        return stats

    @staticmethod
    def _coerce_row(item: Any) -> dict[str, Any]:
        """Turn a dataset row into a plain dict (Arrow / Mapping)."""
        if isinstance(item, dict):
            return item
        if hasattr(item, "as_py"):
            return dict(item.as_py())  # pyarrow Row
        try:
            return dict(item)
        except (TypeError, ValueError):
            return {k: item[k] for k in item}  # type: ignore[misc]

    @staticmethod
    def _normalize_qa_fields(row: dict[str, Any]) -> dict[str, Any]:
        """Map official Daily-Omni qa.json / Hub schema to internal fields.

        Official fields (see liarliar/Daily-Omni ``qa.json``): ``Question``, ``Choice`` (list),
        ``Answer``, ``video_id``, ``Type``, ``video_duration`` (``30s`` / ``60s``), ``video_category``,
        plus other category columns. Legacy aliases (lowercase / older loaders) are still accepted.
        """
        out: dict[str, Any] = {}

        out["question"] = str(row.get("Question") or row.get("question") or "").strip()
        vid = row.get("video_id") if row.get("video_id") is not None else row.get("video")
        out["video_id"] = str(vid).strip() if vid is not None else ""
        out["task_type"] = str(row.get("Type") or row.get("task_type") or row.get("type") or "").strip()
        vc = row.get("video_category") if row.get("video_category") is not None else row.get("videoCategory")
        out["video_category"] = str(vc).strip() if vc is not None else ""
        vd = row.get("video_duration") if row.get("video_duration") is not None else row.get("videoDuration")
        out["video_duration"] = str(vd).strip() if vd is not None else ""
        out["answer"] = str(row.get("Answer") or row.get("answer") or "").strip()
        vu = row.get("video_url") if row.get("video_url") is not None else row.get("Video_URL")
        out["video_url"] = str(vu).strip() if vu is not None and str(vu).strip() else None

        choice = row.get("Choice")
        if choice is None:
            choice = row.get("options") or row.get("choice")
        out["choice"] = choice

        return out

    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        output_len: int | None = None,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs,
    ) -> list[SampleRequest]:
        """Sample requests from Daily-Omni dataset.

        Args:
            tokenizer: Tokenizer for computing prompt length
            num_requests: Number of requests to sample
            output_len: Target output length in tokens (default: 256)
            request_id_prefix: Prefix for request IDs
            no_oversample: If True, do not oversample if fewer examples available
            **kwargs: Additional arguments (ignored)

        Returns:
            List of SampleRequest objects with video URLs and prompts
        """
        if output_len is None:
            output_len = self.DEFAULT_OUTPUT_LEN

        sampled_requests: list[SampleRequest] = []
        ind = 0
        cached_tokenizer = get_cached_tokenizer(tokenizer)

        # Iterate over shuffled dataset
        for item in self.data:
            if len(sampled_requests) >= num_requests:
                break

            request = self._create_sample_request(
                self._coerce_row(item), cached_tokenizer, output_len, request_id_prefix, ind
            )
            if request:
                sampled_requests.append(request)
                ind += 1

        logger.info("Created %d sample requests from Daily-Omni dataset", len(sampled_requests))

        # Handle oversampling if needed
        self.maybe_oversample_requests(sampled_requests, num_requests, request_id_prefix, no_oversample)

        return sampled_requests

    def _create_sample_request(
        self,
        qa_item: dict[str, Any],
        tokenizer: TokenizerLike,
        output_len: int,
        request_id_prefix: str,
        index: int,
    ) -> SampleRequest | None:
        """Create a SampleRequest from a QA item.

        Args:
            qa_item: QA pair from the dataset
            tokenizer: Tokenizer
            output_len: Target output length
            request_id_prefix: Prefix for request ID
            index: Request index

        Returns:
            SampleRequest or None if invalid
        """
        fields = self._normalize_qa_fields(qa_item)
        video_id = fields["video_id"]
        question = fields["question"]
        choice = fields["choice"]
        task_type = fields["task_type"]
        video_url = fields["video_url"]
        video_duration = fields.get("video_duration") or ""
        video_category = fields.get("video_category") or ""

        if not video_id and not video_url:
            logger.warning("Skipping item: no video_id / video_url")
            return None

        if not question:
            logger.warning("Skipping item: no question found")
            return None

        # Official layout after extracting Videos.tar (see Lliar-liar/Daily-Omni test_model):
        #   {video_base_dir}/{video_id}/{video_id}_video.mp4
        mm_payload, omni_extra, mm_pos = self._compose_daily_omni_multimodal(video_id, video_url)
        if not mm_payload:
            return None

        messages = self._build_daily_omni_openai_messages(mm_payload, question, choice)
        user_text = self._official_daily_omni_user_prompt(question, choice)
        # Text-only length estimate (same as before: no MM token count in bench).
        prompt_len = len(tokenizer.encode(f"{DAILY_OMNI_SYSTEM_TEXT}\n{user_text}"))

        return DailyOmniSampleRequest(
            prompt=user_text,
            prompt_len=prompt_len,
            expected_output_len=output_len,
            multi_modal_data=None,
            request_id=f"{request_id_prefix}{index}",
            daily_omni_gold_answer=fields["answer"],
            daily_omni_video_id=video_id,
            daily_omni_task_type=task_type,
            daily_omni_video_duration=video_duration,
            daily_omni_video_category=video_category,
            omni_extra_body=omni_extra,
            omni_chat_messages=messages,
            omni_chat_mm_position=mm_pos,
        )

    @staticmethod
    def _official_video_relpath(video_id: str) -> str:
        """Relative path inside extracted ``Videos/`` per upstream Daily-Omni scripts."""
        return f"{video_id}/{video_id}_video.mp4"

    @staticmethod
    def _official_audio_relpath(video_id: str) -> str:
        """Relative path for extracted WAV per upstream ``get_audio_path``."""
        return f"{video_id}/{video_id}_audio.wav"

    def _resolve_local_video_path(self, video_id: str) -> Path | None:
        """Pick an existing file under ``video_dir`` (official layout + flat fallback)."""
        if not self.video_dir or not video_id:
            return None

        candidates = [
            self.video_dir / self._official_video_relpath(video_id),
            self.video_dir / f"{video_id}.mp4",  # flat layout (custom mirrors / outdated docs)
        ]
        seen: set[Path] = set()
        for p in candidates:
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            if p.exists():
                return p
        return None

    def _resolve_local_audio_path(self, video_id: str) -> Path | None:
        """Pick an existing WAV under ``video_dir`` (official layout + flat fallback)."""
        if not self.video_dir or not video_id:
            return None
        candidates = [
            self.video_dir / self._official_audio_relpath(video_id),
            self.video_dir / f"{video_id}.wav",
        ]
        seen: set[Path] = set()
        for p in candidates:
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            if p.exists():
                return p
        return None

    def _local_file_to_video_url_payload(self, video_path: Path) -> dict[str, Any]:
        """Build OpenAI-style video_url part for a resolved local file.

        vLLM rejects ``file://`` unless the server was started with
        ``--allowed-local-media-path`` set to a directory that **contains** the file
        (typically the extracted ``Videos`` root). Use ``inline_local_video=True`` to
        send base64 data URLs instead (no server path allowlist; larger requests).
        """
        path = video_path.expanduser().resolve()
        if self.inline_local_video:
            raw = path.read_bytes()
            b64 = base64.b64encode(raw).decode("ascii")
            return {
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{b64}"},
            }
        return {
            "type": "video_url",
            "video_url": {"url": path.as_uri()},
        }

    def _local_file_to_audio_url_payload(self, audio_path: Path) -> dict[str, Any]:
        """Build OpenAI-style ``audio_url`` part for a resolved local WAV file."""
        path = audio_path.expanduser().resolve()
        if self.inline_local_video:
            raw = path.read_bytes()
            b64 = base64.b64encode(raw).decode("ascii")
            return {
                "type": "audio_url",
                "audio_url": {"url": f"data:audio/wav;base64,{b64}"},
            }
        return {
            "type": "audio_url",
            "audio_url": {"url": path.as_uri()},
        }

    def _get_video_content(
        self,
        video_id: str,
        video_url: str | None,
    ) -> dict[str, Any] | None:
        """Resolve video for OpenAI-style ``video_url`` content.

        Upstream uses ``get_video_path(video_id, base) -> base/video_id/video_id_video.mp4``.
        The Hub repo only publishes ``Videos.tar``; use ``--daily-omni-video-dir`` pointing
        at the extracted ``Videos`` folder (parent of per-``video_id`` subdirs).

        For ``file://`` URLs, start ``vllm serve`` with e.g.
        ``--allowed-local-media-path /same/path/as/daily-omni-video-dir``.
        """
        if video_url:
            url = video_url
            if not url.startswith(("http://", "https://", "file://")):
                url = f"https://{url.lstrip('/')}"
            return {"type": "video_url", "video_url": {"url": url}}

        if self.video_dir and video_id:
            video_path = self._resolve_local_video_path(video_id)
            if video_path is not None:
                return self._local_file_to_video_url_payload(video_path)
            logger.warning(
                "Video not found under video_dir=%s for video_id=%r (expected %s or %s)",
                self.video_dir,
                video_id,
                self._official_video_relpath(video_id),
                f"{video_id}.mp4",
            )

        if video_id:
            repo = self.dataset_path or self.DEFAULT_HF_DATASET_ID
            rel = self._official_video_relpath(video_id)
            hf_video_url = f"https://huggingface.co/datasets/{repo}/resolve/main/Videos/{rel}"
            logger.debug(
                "Using HF video URL (likely 404 — Hub ships Videos.tar only): %s",
                hf_video_url,
            )
            return {"type": "video_url", "video_url": {"url": hf_video_url}}

        logger.error("Could not determine video source for video_id=%r", video_id)
        return None

    def _get_audio_content(self, video_id: str) -> dict[str, Any] | None:
        """Resolve extracted WAV for OpenAI-style ``audio_url`` (local files only)."""
        if not self.video_dir or not video_id:
            logger.warning(
                "Daily-Omni input_mode %r requires --daily-omni-video-dir with %s",
                self.input_mode,
                self._official_audio_relpath(video_id),
            )
            return None
        audio_path = self._resolve_local_audio_path(video_id)
        if audio_path is not None:
            return self._local_file_to_audio_url_payload(audio_path)
        logger.warning(
            "Audio not found under video_dir=%s for video_id=%r (expected %s or %s)",
            self.video_dir,
            video_id,
            self._official_audio_relpath(video_id),
            f"{video_id}.wav",
        )
        return None

    def _compose_daily_omni_multimodal(
        self,
        video_id: str,
        video_url: str | None,
    ) -> tuple[dict[str, Any] | list[dict[str, Any]] | None, dict[str, Any] | None, Literal["first", "last"]]:
        """Build ``multi_modal_data`` and request extras for the active ``input_mode``.

        Mirrors upstream Daily-Omni: separate video + WAV with ``use_audio_in_video=False``.
        """
        extra: dict[str, Any] = {"mm_processor_kwargs": {"use_audio_in_video": False}}
        mode = self.input_mode

        if mode == "visual":
            v = self._get_video_content(video_id, video_url)
            return v, extra, "last"

        if mode == "audio":
            a = self._get_audio_content(video_id)
            return a, extra, "first"

        v = self._get_video_content(video_id, video_url)
        a = self._get_audio_content(video_id)
        if not v or not a:
            return None, None, "first"
        return [v, a], extra, "first"

    @staticmethod
    def _media_desc_for_official_prompt(mode: DailyOmniInputMode) -> str:
        """``media_desc`` in upstream ``build_conversation``."""
        if mode == "audio":
            return "given audio"
        if mode == "all":
            return "given video and audio together"
        return "given video"

    @staticmethod
    def _choices_repr_for_official_prompt(choice: Any) -> str:
        """Format ``Choice`` from qa.json for the model (one option per line when possible).

        Using ``str(list)`` embeds Python list brackets and quotes, which is poor for MCQ
        reading; lists/tuples are joined with newlines instead. Other shapes fall back to
        ``str(choice)`` for parity with exotic upstream payloads.
        """
        if choice is None:
            return ""
        if isinstance(choice, (list, tuple)):
            lines = [str(x).strip() for x in choice if str(x).strip()]
            return "\n".join(lines)
        if isinstance(choice, dict):
            return "\n".join(f"{k}. {v}" for k, v in choice.items())
        return str(choice)

    def _official_daily_omni_user_prompt(self, question: str, choice: Any) -> str:
        """User text block from Daily-Omni ``build_conversation`` (after media parts)."""
        task_prompt = self._media_desc_for_official_prompt(self.input_mode)
        choices = self._choices_repr_for_official_prompt(choice)
        # Single f-string with explicit newlines avoids accidental implicit concatenation
        # gluing sentences (e.g. ``...media_desc.Select...``) when editing.
        return (
            "Your task is to accurately answer multiple-choice questions "
            f"based on the {task_prompt}.\n"
            "Select the single most accurate answer from the given choices.\n"
            f"Question: {question}\n"
            f"Choices: {choices}\n"
            "Your answer should be a capital letter representing your choice: "
            "A, B, C, or D. Don't generate any other text.\n"
        )

    def _build_daily_omni_openai_messages(
        self,
        mm_payload: dict[str, Any] | list[dict[str, Any]],
        question: str,
        choice: Any,
    ) -> list[dict[str, Any]]:
        """Map upstream conversation to OpenAI Chat Completions ``messages`` (video_url / audio_url parts)."""
        user_text = self._official_daily_omni_user_prompt(question, choice)
        mm_list: list[dict[str, Any]] = mm_payload if isinstance(mm_payload, list) else [mm_payload]
        user_content: list[dict[str, Any]] = [*mm_list, {"type": "text", "text": user_text}]
        return [
            {"role": "system", "content": [{"type": "text", "text": DAILY_OMNI_SYSTEM_TEXT}]},
            {"role": "user", "content": user_content},
        ]

    def sample_by_task_type(
        self,
        tokenizer: TokenizerLike,
        task_type: str,
        num_samples: int,
        output_len: int | None = None,
        request_id_prefix: str = "",
        **kwargs,
    ) -> list[SampleRequest]:
        """Sample requests filtered by task type.

        Args:
            tokenizer: Tokenizer
            task_type: Task type to filter by
            num_samples: Number of samples
            output_len: Target output length
            request_id_prefix: Prefix for request IDs
            **kwargs: Additional sampling arguments

        Returns:
            List of SampleRequest objects matching the task type
        """
        if output_len is None:
            output_len = self.DEFAULT_OUTPUT_LEN

        filtered = [
            item for item in self.data if self._normalize_qa_fields(self._coerce_row(item))["task_type"] == task_type
        ]

        available = len(filtered)
        if available < num_samples:
            logger.warning(
                "Only %d samples available for task type '%s', requested %d",
                available,
                task_type,
                num_samples,
            )
            num_samples = available

        sampled_requests: list[SampleRequest] = []
        cached_tokenizer = get_cached_tokenizer(tokenizer)

        for i, item in enumerate(filtered[:num_samples]):
            request = self._create_sample_request(item, cached_tokenizer, output_len, request_id_prefix, i)
            if request:
                sampled_requests.append(request)

        return sampled_requests

    def __repr__(self) -> str:
        return (
            f"DailyOmniDataset("
            f"dataset_path={self.dataset_path!r}, "
            f"dataset_split={self.dataset_split!r}, "
            f"video_dir={self.video_dir!r}, "
            f"input_mode={self.input_mode!r}, "
            f"inline_local_video={self.inline_local_video!r}, "
            f"max_duration_seconds={self.max_duration_seconds}, "
            f"random_seed={self.random_seed}"
            f")"
        )


def load_daily_omni_dataset(
    qa_json_path: str | None = None,
    dataset_path: str | None = None,
    dataset_split: str = "train",
    random_seed: int = 0,
    video_dir: str | None = None,
    input_mode: DailyOmniInputMode = "all",
    max_duration_seconds: float | None = None,
    dataset_subset: str | None = None,
    no_stream: bool = False,
    **kwargs,
) -> DailyOmniDataset:
    """Convenience function to load Daily-Omni dataset.

    Args:
        qa_json_path: Path to local qa.json file (recommended for offline/air-gapped environments).
            When provided, ``dataset_path`` is ignored.
        dataset_path: HuggingFace dataset path (default: liarliar/Daily-Omni). Used only if
            ``qa_json_path`` is not provided (legacy online mode).
        dataset_split: Dataset split to use (default: "train")
        random_seed: Random seed for shuffling
        video_dir: Directory containing extracted ``Videos/`` tree (MP4 and, for ``all``/``audio``, WAV)
        input_mode: ``visual`` | ``audio`` | ``all`` (same semantics as upstream Daily-Omni)
        max_duration_seconds: Maximum video duration in seconds (e.g., 30 for 30s subset, 60 for 60s subset);
            uses ffprobe on local files under ``video_dir`` (in-memory cache only for this process).
        **kwargs: Additional arguments passed to DailyOmniDataset

    Returns:
        DailyOmniDataset instance

    Example:
        >>> from vllm_omni.benchmarks.data_modules.daily_omni_dataset import load_daily_omni_dataset

        # Local JSON mode (recommended for offline)
        >>> dataset = load_daily_omni_dataset(
        ...     qa_json_path="/path/to/qa.json",
        ...     video_dir="/path/to/Daily-Omni/Videos",
        ...     random_seed=42,
        ...     max_duration_seconds=30,
        ... )

        # HuggingFace mode (legacy online)
        >>> dataset = load_daily_omni_dataset(
        ...     dataset_path="liarliar/Daily-Omni",
        ...     video_dir="/path/to/Daily-Omni/Videos",
        ...     random_seed=42,
        ... )
        >>> requests = dataset.sample(tokenizer, num_requests=100)
    """
    return DailyOmniDataset(
        qa_json_path=qa_json_path,
        dataset_path=dataset_path,
        dataset_split=dataset_split,
        random_seed=random_seed,
        video_dir=video_dir,
        input_mode=input_mode,
        max_duration_seconds=max_duration_seconds,
        dataset_subset=dataset_subset,
        no_stream=no_stream,
        **kwargs,
    )


def get_daily_omni_statistics(
    qa_json_path: str | None = None,
    dataset_path: str | None = DailyOmniDataset.DEFAULT_HF_DATASET_ID,
    dataset_split: str = "train",
) -> dict[str, Any]:
    """Get statistics about the Daily-Omni dataset.

    Args:
        qa_json_path: Path to local qa.json file (recommended for offline/air-gapped environments).
            When provided, ``dataset_path`` is ignored.
        dataset_path: HuggingFace dataset path. Defaults to ``DailyOmniDataset.DEFAULT_HF_DATASET_ID``
            when ``qa_json_path`` is omitted. Pass ``None`` only together with ``qa_json_path``.
        dataset_split: Dataset split to use (default: "train")

    Returns:
        Statistics dict with task type distribution and other info

    Example:
        >>> from vllm_omni.benchmarks.data_modules.daily_omni_dataset import get_daily_omni_statistics

        # Local JSON mode
        >>> stats = get_daily_omni_statistics(qa_json_path="/path/to/qa.json")

        # HuggingFace mode
        >>> stats = get_daily_omni_statistics(dataset_path="liarliar/Daily-Omni")
        >>> print(f"Total QA pairs: {stats['total_qa_pairs']}")
        >>> print(f"Task distribution: {stats['task_distribution']}")
    """
    dataset = DailyOmniDataset(
        qa_json_path=qa_json_path,
        dataset_path=dataset_path,
        dataset_split=dataset_split,
    )
    task_stats = dataset.get_task_statistics()

    source = str(qa_json_path) if qa_json_path else f"{dataset_path}/{dataset_split}"
    return {
        "source": source,
        "total_qa_pairs": len(list(dataset.data)),
        "task_distribution": task_stats,
    }
