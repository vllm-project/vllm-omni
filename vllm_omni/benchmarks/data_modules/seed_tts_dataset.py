"""Seed-TTS zero-shot evaluation-style prompts for ``vllm bench serve``.

Loads rows from the `meta.lst` format used in `BytedanceSpeech/seed-tts-eval`_ (or any
HuggingFace dataset repo with the same layout)::

    utt_id|prompt_transcript|prompt_wav_relative_path|text_to_synthesize

Each benchmark request supplies target text plus ``ref_text`` / ``ref_audio`` (Qwen3-TTS ``Base`` /
voice clone), merged into the JSON body. By default ``ref_audio`` is an inline ``data:`` URL so
the server does not need ``--allowed-local-media-path``. Use ``--seed-tts-file-ref-audio`` for
``file://`` (smaller bodies; requires that flag). Use ``--backend openai-audio-speech``
(``/v1/audio/speech``) or ``--backend openai-chat-omni`` (``/v1/chat/completions`` with the same
fields on the body plus a Qwen3-Omni-style ``system`` message and the target text as ``user`` content).

.. _BytedanceSpeech/seed-tts-eval: https://github.com/BytedanceSpeech/seed-tts-eval
"""

from __future__ import annotations

import base64
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm.benchmarks.datasets import BenchmarkDataset, SampleRequest
from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.hf import get_cached_tokenizer

logger = logging.getLogger(__name__)

# Matches Qwen3-Omni serving examples (``openai_chat_completion_client_for_multimodal_generation`` /
# ``qwen3_omni/gradio_demo``) plus explicit TTS / voice-clone instructions for chat completions.
SEED_TTS_DEFAULT_OMNI_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech.\n"
    "For this request you act as a text-to-speech engine with zero-shot voice cloning: "
    "the API provides reference audio and its transcript (ref_audio, ref_text) and task_type Base. "
    "The user message is the exact text you must speak. "
    "Synthesize natural speech in the same language as that user text, "
    "matching the timbre, prosody, and speaking style of the reference audio while reading the new content clearly."
)


@dataclass
class SeedTTSSampleRequest(SampleRequest):
    """``SampleRequest`` with per-row fields merged into ``/v1/audio/speech`` JSON."""

    #: Shallow-merged into ``RequestFuncInput.extra_body`` (ref_audio, ref_text, task_type, …).
    seed_tts_speech_extra: dict[str, Any] | None = None
    seed_tts_utterance_id: str = ""
    seed_tts_locale: str = ""
    #: For ``openai-chat-omni``: becomes the chat ``system`` message (Qwen3-Omni + TTS behavior).
    seed_tts_system_prompt: str = ""
    #: Local path to reference prompt WAV (for SIM vs. synthesized PCM in ``seed_tts_eval``).
    seed_tts_ref_wav_path: str = ""


@dataclass
class _SeedTTSRow:
    utterance_id: str
    ref_text: str
    prompt_wav_rel: str
    target_text: str


def _parse_meta_line(line: str) -> _SeedTTSRow | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split("|")
    if len(parts) < 4:
        logger.warning("Skipping malformed meta.lst line (need 4 '|'-fields): %r", line[:120])
        return None
    utt_id, ref_text, wav_rel, target = parts[0], parts[1], parts[2], parts[3]
    if not target.strip():
        return None
    return _SeedTTSRow(
        utterance_id=utt_id.strip(),
        ref_text=ref_text.strip(),
        prompt_wav_rel=wav_rel.strip(),
        target_text=target.strip(),
    )


def _load_meta_rows(meta_file: Path) -> list[_SeedTTSRow]:
    text = meta_file.read_text(encoding="utf-8")
    rows: list[_SeedTTSRow] = []
    for line in text.splitlines():
        r = _parse_meta_line(line)
        if r is not None:
            rows.append(r)
    return rows


def resolve_seed_tts_root(dataset_path: str | None, *, explicit_root: str | None, locale: str = "en") -> Path:
    """Return directory containing ``{locale}/meta.lst`` and ``{locale}/prompt-wavs/``."""
    if explicit_root:
        root = Path(explicit_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"--seed-tts-root is not a directory: {root}")
        return root

    if not dataset_path:
        raise ValueError("Seed-TTS requires --dataset-path (HF repo id or local root) or --seed-tts-root.")

    p = Path(dataset_path).expanduser()
    if p.exists() and p.is_dir():
        return p.resolve()

    repo_id = dataset_path.strip()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "Install huggingface_hub to download Seed-TTS from the Hub, or clone the dataset "
            "locally and pass --dataset-path / --seed-tts-root to that directory."
        ) from e
    # Download only the requested locale subtree instead of the whole dataset
    # repo. This avoids large, flaky nightly downloads when we only need e.g.
    # ``en/meta.lst`` + ``en/prompt-wavs/**``.
    cache = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[f"{locale}/**"],
    )
    return Path(cache).resolve()


def _ref_audio_payload(wav_path: Path, *, inline: bool) -> str:
    if inline:
        raw = wav_path.read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:audio/wav;base64,{b64}"
    return wav_path.expanduser().resolve().as_uri()


class SeedTTSDataset(BenchmarkDataset):
    """Seed-TTS-style zero-shot TTS rows for throughput/latency benchmarking.

    Args:
        dataset_path: HuggingFace dataset repo id (``org/dataset``) or local directory with
            ``en/meta.lst`` (and ``zh/meta.lst`` if using zh).
        locale: ``en`` or ``zh`` — which subfolder under the root to read.
        inline_ref_audio: If True (default), embed prompt WAV as ``data:audio/wav;base64,...``
            so Qwen3-TTS / ``/v1/audio/speech`` works without server
            ``--allowed-local-media-path``. If False, use ``file://`` (smaller
            requests; server must set ``--allowed-local-media-path`` to the dataset root).
        seed_tts_root: Optional override for the root directory (same layout as HF dataset).
        system_prompt: Optional override for the chat system message when using
            ``--backend openai-chat-omni``; defaults to :data:`SEED_TTS_DEFAULT_OMNI_SYSTEM_PROMPT`.
    """

    IS_MULTIMODAL = False
    DEFAULT_OUTPUT_LEN = 2048

    def __init__(
        self,
        dataset_path: str,
        random_seed: int = 0,
        locale: str = "en",
        inline_ref_audio: bool = True,
        seed_tts_root: str | None = None,
        system_prompt: str | None = None,
        disable_shuffle: bool = False,
        **kwargs: Any,
    ) -> None:
        if locale not in ("en", "zh"):
            raise ValueError("locale must be 'en' or 'zh'")
        self.locale = locale
        self.inline_ref_audio = inline_ref_audio
        self._explicit_root = seed_tts_root
        sp = (system_prompt or "").strip()
        self._system_prompt = sp if sp else SEED_TTS_DEFAULT_OMNI_SYSTEM_PROMPT
        super().__init__(
            dataset_path=dataset_path,
            random_seed=random_seed,
            disable_shuffle=disable_shuffle,
            **kwargs,
        )
        self._root = resolve_seed_tts_root(self.dataset_path, explicit_root=self._explicit_root, locale=self.locale)
        self._rows: list[_SeedTTSRow] = []
        self.load_data()

    def load_data(self) -> None:
        meta = self._root / self.locale / "meta.lst"
        if not meta.is_file():
            raise FileNotFoundError(
                f"Seed-TTS meta not found: {meta}. "
                f"Expected layout from seed-tts-eval (e.g. {self._root}/{self.locale}/meta.lst)."
            )
        self._rows = _load_meta_rows(meta)
        if not self._rows:
            raise ValueError(f"No valid rows in {meta}")
        if not self.disable_shuffle:
            rng = random.Random(self.random_seed)
            rng.shuffle(self._rows)
        self.data = self._rows
        logger.info(
            "Loaded Seed-TTS: root=%s locale=%s rows=%d inline_ref_audio=%s",
            self._root,
            self.locale,
            len(self._rows),
            self.inline_ref_audio,
        )

    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        output_len: int | None = None,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs: Any,
    ) -> list[SampleRequest]:
        if output_len is None:
            output_len = self.DEFAULT_OUTPUT_LEN

        tok = get_cached_tokenizer(tokenizer)
        out: list[SampleRequest] = []
        for i, row in enumerate(self._rows):
            if len(out) >= num_requests:
                break
            wav_path = (self._root / self.locale / row.prompt_wav_rel).resolve()
            if not wav_path.is_file():
                logger.warning("Missing prompt wav for %s: %s", row.utterance_id, wav_path)
                continue

            target = row.target_text
            prompt_len = len(tok.encode(f"{self._system_prompt}\n{target}"))
            lang = "English" if self.locale == "en" else "Chinese"
            ref_uri = _ref_audio_payload(wav_path, inline=self.inline_ref_audio)
            speech_extra: dict[str, Any] = {
                "ref_audio": ref_uri,
                "ref_text": row.ref_text,
                "task_type": "Base",
                "language": lang,
                "max_new_tokens": output_len,
            }

            out.append(
                SeedTTSSampleRequest(
                    prompt=target,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    multi_modal_data=None,
                    request_id=f"{request_id_prefix}{i}",
                    seed_tts_speech_extra=speech_extra,
                    seed_tts_utterance_id=row.utterance_id,
                    seed_tts_locale=self.locale,
                    seed_tts_system_prompt=self._system_prompt,
                    seed_tts_ref_wav_path=str(wav_path),
                )
            )

        logger.info("Seed-TTS: built %d requests (asked %d)", len(out), num_requests)
        self.maybe_oversample_requests(out, num_requests, request_id_prefix, no_oversample)
        return out


@dataclass
class _SeedTTSDesignRow:
    utterance_id: str
    target_text: str
    voice_description: str


def _parse_design_meta_line(line: str) -> _SeedTTSDesignRow | None:
    """Parse a 5-field design meta.lst line.

    Format: ``utt_id|ref_text|wav_rel|target_text|voice_description``

    Returns None (with a warning) if the line has fewer than 5 fields or if
    voice_description is empty.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split("|")
    if len(parts) < 5:
        logger.warning("Skipping malformed design meta.lst line (need 5 '|'-fields): %r", line[:120])
        return None
    utt_id = parts[0].strip()
    target_text = parts[3].strip()
    voice_description = parts[4].strip()
    if not voice_description:
        logger.warning("Skipping design meta.lst line with empty voice_description: %r", line[:120])
        return None
    return _SeedTTSDesignRow(
        utterance_id=utt_id,
        target_text=target_text,
        voice_description=voice_description,
    )


@dataclass
class SeedTTSDesignSampleRequest(SeedTTSSampleRequest):
    """SampleRequest for voice-design TTS (no ref_audio; voice described via natural language).

    The ``seed_tts_speech_extra`` dict carries ``instructions`` (natural-language
    voice description, forwarded as-is to the Qwen3-TTS VoiceDesign endpoint) and
    ``task_type="VoiceDesign"`` instead of ``ref_audio`` / ``ref_text``.
    SIM is skipped (``seed_tts_ref_wav_path`` is empty).
    """


class SeedTTSDesignDataset(SeedTTSDataset):
    """Seed-TTS prompts for voice-design benchmarking (dataset name: ``seed-tts-design``).

    Loads a 5-field ``meta.lst``::

        utt_id|ref_text|wav_rel|target_text|voice_description

    and builds requests with ``task_type="VoiceDesign"`` and the natural-language
    ``voice_description`` column forwarded via the ``instructions`` field
    (the Qwen3-TTS VoiceDesign endpoint's expected key) instead of
    ``ref_audio`` / ``ref_text``.  Speaker-similarity (SIM) is not computed.
    """

    def load_data(self) -> None:
        # Does NOT call super().load_data() — the format is different (5 fields,
        # no wav file).  self._rows is intentionally left empty; the parent
        # sample() is fully overridden so an empty self._rows is safe.
        meta = self._root / self.locale / "meta.lst"
        if not meta.is_file():
            raise FileNotFoundError(
                f"Seed-TTS-Design meta not found: {meta}. Expected layout: {self._root}/{self.locale}/meta.lst"
            )
        text = meta.read_text(encoding="utf-8")
        design_rows: list[_SeedTTSDesignRow] = []
        for line in text.splitlines():
            r = _parse_design_meta_line(line)
            if r is not None:
                design_rows.append(r)
        if not design_rows:
            raise ValueError(f"No valid rows in {meta}")
        if not self.disable_shuffle:
            rng = random.Random(self.random_seed)
            rng.shuffle(design_rows)
        self._design_rows = design_rows
        # Keep self._rows empty — parent sample() is overridden.
        self._rows = []
        self.data = self._design_rows
        logger.info(
            "Loaded Seed-TTS-Design: root=%s locale=%s rows=%d",
            self._root,
            self.locale,
            len(self._design_rows),
        )

    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        output_len: int | None = None,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs: Any,
    ) -> list[SampleRequest]:
        if output_len is None:
            output_len = self.DEFAULT_OUTPUT_LEN

        tok = get_cached_tokenizer(tokenizer)
        lang = "English" if self.locale == "en" else "Chinese"
        out: list[SampleRequest] = []
        for i, row in enumerate(self._design_rows):
            if len(out) >= num_requests:
                break
            target = row.target_text
            prompt_len = len(tok.encode(target))
            speech_extra: dict[str, Any] = {
                "instructions": row.voice_description,
                "task_type": "VoiceDesign",
                "language": lang,
                "max_new_tokens": output_len,
            }
            out.append(
                SeedTTSDesignSampleRequest(
                    prompt=target,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    multi_modal_data=None,
                    request_id=f"{request_id_prefix}{i}",
                    seed_tts_speech_extra=speech_extra,
                    seed_tts_utterance_id=row.utterance_id,
                    seed_tts_locale=self.locale,
                    seed_tts_system_prompt=self._system_prompt,
                    seed_tts_ref_wav_path="",  # SIM skipped for voice-design
                )
            )

        logger.info(
            "Seed-TTS-Design: built %d requests (asked %d) — no ref_audio (voice design)",
            len(out),
            num_requests,
        )
        self.maybe_oversample_requests(out, num_requests, request_id_prefix, no_oversample)
        return out


def load_seed_tts_dataset(
    dataset_path: str,
    random_seed: int = 0,
    locale: str = "en",
    inline_ref_audio: bool = True,
    seed_tts_root: str | None = None,
    system_prompt: str | None = None,
    **kwargs: Any,
) -> SeedTTSDataset:
    return SeedTTSDataset(
        dataset_path=dataset_path,
        random_seed=random_seed,
        locale=locale,
        inline_ref_audio=inline_ref_audio,
        seed_tts_root=seed_tts_root,
        system_prompt=system_prompt,
        **kwargs,
    )


@dataclass
class SeedTTSTextSampleRequest(SeedTTSSampleRequest):
    """SampleRequest for default-voice TTS (no ref_audio, no ref_text).

    The voice param (e.g. ``voice: "Vivian"``) is supplied at request time via
    ``--extra-body`` in the benchmark config. SIM is skipped (empty ref_wav_path).
    WER and UTMOS are computed normally.
    """


class SeedTTSTextDataset(SeedTTSDataset):
    """Seed-TTS prompts for default-voice benchmarking (dataset name: ``seed-tts-text``).

    Loads the same ``meta.lst`` as :class:`SeedTTSDataset` but builds requests
    WITHOUT ``ref_audio`` / ``ref_text`` body fields. The named voice must be
    supplied via ``--extra-body`` in the benchmark config.
    Speaker-similarity (SIM) is not computed.
    """

    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        output_len: int | None = None,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs: Any,
    ) -> list[SampleRequest]:
        if output_len is None:
            output_len = self.DEFAULT_OUTPUT_LEN

        tok = get_cached_tokenizer(tokenizer)
        out: list[SampleRequest] = []
        for i, row in enumerate(self._rows):
            if len(out) >= num_requests:
                break
            target = row.target_text
            prompt_len = len(tok.encode(target))
            out.append(
                SeedTTSTextSampleRequest(
                    prompt=target,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    multi_modal_data=None,
                    request_id=f"{request_id_prefix}{i}",
                    seed_tts_speech_extra=None,  # voice supplied via --extra-body in config
                    seed_tts_utterance_id=row.utterance_id,
                    seed_tts_locale=self.locale,
                    seed_tts_system_prompt=self._system_prompt,
                    seed_tts_ref_wav_path="",  # empty → SIM skipped in seed_tts_eval
                )
            )

        logger.info(
            "Seed-TTS-Text: built %d requests (asked %d) — no ref_audio (default voice)",
            len(out),
            num_requests,
        )
        self.maybe_oversample_requests(out, num_requests, request_id_prefix, no_oversample)
        return out
