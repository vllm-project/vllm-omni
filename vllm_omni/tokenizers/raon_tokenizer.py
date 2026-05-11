# SPDX-License-Identifier: Apache-2.0
"""Unified Raon tokenizer utilities and chat-template builder.

Merges ``tokenizer_utils`` and ``chat_template_builder`` into a single module.
"""

from __future__ import annotations

import enum
import json
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from vllm.logger import init_logger

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# SpecialToken dataclass + token constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpecialToken:
    """Frozen container for a special token's id and surface text."""

    id: int
    text: str

    def __int__(self) -> int:
        return self.id

    def __str__(self) -> str:
        return self.text


PAD = SpecialToken(id=151679, text="<|endoftext|>")
IM_START = SpecialToken(id=151644, text="<|im_start|>")
IM_END = SpecialToken(id=151645, text="<|im_end|>")
AUDIO_START = SpecialToken(id=151669, text="<|audio_start|>")
AUDIO_END = SpecialToken(id=151670, text="<|audio_end|>")
SPEAKER_EMBEDDING_PLACEHOLDER = SpecialToken(id=151671, text="<|speaker_embedding_placeholder|>")
AUDIO_OUTPUT_PLACEHOLDER = SpecialToken(id=151675, text="<|audio_output_placeholder|>")
AUDIO_INPUT_PLACEHOLDER = SpecialToken(id=151676, text="<|audio_input_placeholder|>")
AUDIO_OUTPUT_PAD = SpecialToken(id=151677, text="<|audio_output_pad|>")
AUDIO_OUTPUT_END_PAD = SpecialToken(id=151678, text="<|audio_output_end_pad|>")

ALL_SPECIAL_TOKENS: list[SpecialToken] = [
    PAD,
    IM_START,
    IM_END,
    AUDIO_START,
    AUDIO_END,
    SPEAKER_EMBEDDING_PLACEHOLDER,
    AUDIO_OUTPUT_PLACEHOLDER,
    AUDIO_INPUT_PLACEHOLDER,
    AUDIO_OUTPUT_PAD,
    AUDIO_OUTPUT_END_PAD,
]

AUDIO_START_TOKEN: str = AUDIO_START.text
AUDIO_END_TOKEN: str = AUDIO_END.text
AUDIO_INPUT_PAD_TOKEN: str = AUDIO_INPUT_PLACEHOLDER.text
AUDIO_OUTPUT_PAD_TOKEN: str = AUDIO_OUTPUT_PLACEHOLDER.text
SPEAKER_EMBEDDING_PLACEHOLDER_TOKEN: str = SPEAKER_EMBEDDING_PLACEHOLDER.text
SPEAKER_EMBEDDING_PLACEHOLDER_ID: int = SPEAKER_EMBEDDING_PLACEHOLDER.id
LEGACY_SPEAKER_PAD_TOKEN: str = "<tts_pad>"


# ---------------------------------------------------------------------------
# Resolved special-token IDs (live-tokenizer-backed)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RaonResolvedIds:
    """Raon special-token IDs resolved against a live tokenizer.

    This keeps serving paths from depending on stale hardcoded SpecialToken ids.
    """

    audio_start: int
    audio_end: int
    audio_input_placeholder: int
    audio_output_placeholder: int
    speaker_placeholder: int
    audio_output_pad: int
    audio_output_end_pad: int


def resolve_raon_special_ids(tokenizer: Any) -> RaonResolvedIds:
    """Resolve Raon special tokens against a live tokenizer.

    Missing tokens fall back to the module defaults and log a warning so
    tokenizer/checkpoint drift stays visible.
    """

    def _resolve(surface: str, fallback: int, label: str) -> int:
        try:
            tid = tokenizer.convert_tokens_to_ids(surface)
        except Exception as exc:
            logger.warning(
                "Raon special token %s (%r) lookup failed (%s); falling back to hardcoded id %d",
                label,
                surface,
                exc,
                fallback,
            )
            return fallback
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if tid is None or (unk_id is not None and tid == unk_id):
            logger.warning(
                "Raon special token %s (%r) missing from tokenizer vocab; falling back to hardcoded id %d",
                label,
                surface,
                fallback,
            )
            return fallback
        return int(tid)

    return RaonResolvedIds(
        audio_start=_resolve(AUDIO_START.text, AUDIO_START.id, "audio_start"),
        audio_end=_resolve(AUDIO_END.text, AUDIO_END.id, "audio_end"),
        audio_input_placeholder=_resolve(
            AUDIO_INPUT_PLACEHOLDER.text,
            AUDIO_INPUT_PLACEHOLDER.id,
            "audio_input_placeholder",
        ),
        audio_output_placeholder=_resolve(
            AUDIO_OUTPUT_PLACEHOLDER.text,
            AUDIO_OUTPUT_PLACEHOLDER.id,
            "audio_output_placeholder",
        ),
        speaker_placeholder=_resolve(
            SPEAKER_EMBEDDING_PLACEHOLDER.text,
            SPEAKER_EMBEDDING_PLACEHOLDER.id,
            "speaker_placeholder",
        ),
        audio_output_pad=_resolve(
            AUDIO_OUTPUT_PAD.text,
            AUDIO_OUTPUT_PAD.id,
            "audio_output_pad",
        ),
        audio_output_end_pad=_resolve(
            AUDIO_OUTPUT_END_PAD.text,
            AUDIO_OUTPUT_END_PAD.id,
            "audio_output_end_pad",
        ),
    )


# ---------------------------------------------------------------------------
# Placeholder sequences and patterns
# ---------------------------------------------------------------------------

USER_PROMPT_MARKER: str = f"{IM_START.text}user\n"
AUDIO_PLACEHOLDER_SEQ: str = f"{AUDIO_START_TOKEN}{AUDIO_INPUT_PAD_TOKEN}{AUDIO_END_TOKEN}"
AUDIO_OUTPUT_PLACEHOLDER_SEQ: str = f"{AUDIO_START_TOKEN}{AUDIO_OUTPUT_PAD_TOKEN}{AUDIO_END_TOKEN}"
AUDIO_OUTPUT_OPEN_SEQ: str = f"{AUDIO_START_TOKEN}{AUDIO_OUTPUT_PAD_TOKEN}"
LEGACY_AUDIO_PAD_TOKEN: str = "<|audio_pad|>"
LEGACY_AUDIO_PLACEHOLDER_SEQ: str = f"{AUDIO_START_TOKEN}{LEGACY_AUDIO_PAD_TOKEN}{AUDIO_END_TOKEN}"
ALL_AUDIO_PLACEHOLDER_VARIANTS: tuple[str, ...] = (
    AUDIO_PLACEHOLDER_SEQ,
    AUDIO_OUTPUT_PLACEHOLDER_SEQ,
    AUDIO_OUTPUT_OPEN_SEQ,
    LEGACY_AUDIO_PLACEHOLDER_SEQ,
)
AUDIO_PLACEHOLDER_PATTERN: re.Pattern[str] = re.compile("|".join(re.escape(p) for p in ALL_AUDIO_PLACEHOLDER_VARIANTS))

LEGACY_SECONDARY_AUDIO_PAD_TOKEN: str = "<|secondary_audio_pad|>"

AUDIO_PLACEHOLDER_TOKENS: tuple[str, ...] = (
    AUDIO_OUTPUT_PAD_TOKEN,
    AUDIO_INPUT_PAD_TOKEN,
    LEGACY_AUDIO_PAD_TOKEN,
    LEGACY_SECONDARY_AUDIO_PAD_TOKEN,
    AUDIO_START_TOKEN,
    AUDIO_END_TOKEN,
)

_AUDIO_PLACEHOLDER_PATTERN = re.compile("|".join(re.escape(tok) for tok in AUDIO_PLACEHOLDER_TOKENS))

# ---------------------------------------------------------------------------
# Text filtering
# ---------------------------------------------------------------------------


def filter_audio_placeholder_text(text: str | None) -> str | None:
    if text is None:
        return None
    if not text:
        return text
    return _AUDIO_PLACEHOLDER_PATTERN.sub("", text)


# ---------------------------------------------------------------------------
# Tokenizer alignment
# ---------------------------------------------------------------------------


class _TokenizerWithEncode(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = ...) -> list[int]: ...

    def decode(self, token_ids: list[int], **kwargs: Any) -> str: ...


def _mk_added_token_payload(token_id: int, content: str) -> dict[str, Any]:
    return dict(
        id=token_id, content=content, single_word=False, lstrip=False, rstrip=False, normalized=False, special=True
    )


def _tokenizer_is_aligned(tokenizer: Any) -> bool:
    for token in ALL_SPECIAL_TOKENS:
        try:
            encoded = tokenizer.encode(token.text, add_special_tokens=False)
        except Exception:
            return False
        if encoded != [token.id]:
            return False
    return True


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


_AUDIO_TOKEN_OVERRIDES: dict[str, str] = {
    "audio_bos_token": AUDIO_START.text,
    "audio_eos_token": AUDIO_END.text,
    "audio_token": AUDIO_OUTPUT_PLACEHOLDER.text,
}


def _patch_tokenizer_files(tokenizer_dir: Path) -> None:
    expected_by_id = {token.id: token.text for token in ALL_SPECIAL_TOKENS}

    vocab_path = tokenizer_dir / "vocab.json"
    if vocab_path.exists():
        vocab = _read_json(vocab_path)
        for tid, txt in expected_by_id.items():
            vocab[txt] = tid
        _write_json(vocab_path, vocab)

    tokenizer_json_path = tokenizer_dir / "tokenizer.json"
    if tokenizer_json_path.exists():
        tj = _read_json(tokenizer_json_path)
        model_vocab = tj.get("model", {}).get("vocab")
        if isinstance(model_vocab, dict):
            for tid, txt in expected_by_id.items():
                model_vocab[txt] = tid
        by_id: dict[int, dict[str, Any]] = {}
        for entry in tj.get("added_tokens", []):
            by_id[int(entry["id"])] = entry
        for tid, txt in expected_by_id.items():
            entry = by_id.get(tid)
            if entry is None:
                by_id[tid] = _mk_added_token_payload(tid, txt)
            else:
                entry.update(content=txt, single_word=False, lstrip=False, rstrip=False, normalized=False, special=True)
        tj["added_tokens"] = [by_id[k] for k in sorted(by_id)]
        _write_json(tokenizer_json_path, tj)

    added_path = tokenizer_dir / "added_tokens.json"
    if added_path.exists():
        m = _read_json(added_path)
        for t in ALL_SPECIAL_TOKENS:
            m[t.text] = t.id
        _write_json(added_path, m)

    tc_path = tokenizer_dir / "tokenizer_config.json"
    if tc_path.exists():
        tc = _read_json(tc_path)
        tc["additional_special_tokens"] = [AUDIO_INPUT_PLACEHOLDER.text]
        tc.update(_AUDIO_TOKEN_OVERRIDES)
        extra = tc.get("extra_special_tokens")
        if isinstance(extra, dict):
            extra.update(_AUDIO_TOKEN_OVERRIDES)
        _write_json(tc_path, tc)

    stm_path = tokenizer_dir / "special_tokens_map.json"
    if stm_path.exists():
        stm = _read_json(stm_path)
        stm.update(_AUDIO_TOKEN_OVERRIDES)
        stm["additional_special_tokens"] = [AUDIO_INPUT_PLACEHOLDER.text]
        _write_json(stm_path, stm)


def align_tokenizer(tokenizer: Any) -> Any:
    if tokenizer is None or _tokenizer_is_aligned(tokenizer):
        return tokenizer

    tokenizer_cls = tokenizer.__class__
    logger.warning("[Raon tokenizer] tokenizer special token mapping is outdated; applying overrides.")

    with tempfile.TemporaryDirectory(prefix="raon_tokenizer_patch_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        tokenizer.save_pretrained(tmp_path)
        _patch_tokenizer_files(tmp_path)
        patched = tokenizer_cls.from_pretrained(tmp_path)

    tokenizer.__dict__.update(patched.__dict__)

    if not _tokenizer_is_aligned(tokenizer):
        raise RuntimeError("Failed to align Raon tokenizer special tokens.")
    return tokenizer


# ---------------------------------------------------------------------------
# Token resolution
# ---------------------------------------------------------------------------


def resolve_audio_input_token_id(
    tokenizer: _TokenizerWithEncode,
    expected_audio_input_token_id: int,
) -> int:
    encoded = tokenizer.encode(AUDIO_INPUT_PAD_TOKEN, add_special_tokens=False)
    if len(encoded) != 1:
        msg = f"{AUDIO_INPUT_PAD_TOKEN} must map to one token ID, got {encoded}."
        raise ValueError(msg)

    token_id = int(encoded[0])
    if token_id != expected_audio_input_token_id:
        msg = f"audio_input_token_id mismatch: config={expected_audio_input_token_id}, tokenizer={token_id}."
        raise ValueError(msg)

    return token_id


def resolve_speaker_token_id(
    tokenizer: _TokenizerWithEncode,
    expected_speaker_token_id: int | None = None,
) -> int:
    candidate_ids: list[int] = []
    if isinstance(expected_speaker_token_id, int):
        candidate_ids.append(int(expected_speaker_token_id))
    candidate_ids.append(SPEAKER_EMBEDDING_PLACEHOLDER_ID)

    seen: set[int] = set()
    for token_id in candidate_ids:
        if token_id in seen:
            continue
        seen.add(token_id)
        try:
            token_text = tokenizer.decode([token_id])
        except Exception:
            continue
        if not token_text:
            continue
        try:
            encoded = tokenizer.encode(token_text, add_special_tokens=False)
        except Exception:
            continue
        if encoded == [token_id]:
            return token_id

    for token_text in (SPEAKER_EMBEDDING_PLACEHOLDER_TOKEN, LEGACY_SPEAKER_PAD_TOKEN):
        try:
            encoded = tokenizer.encode(token_text, add_special_tokens=False)
        except Exception:
            continue
        if len(encoded) == 1 and isinstance(encoded[0], int):
            return int(encoded[0])

    raise ValueError("Unable to resolve speaker placeholder token from tokenizer.")


def resolve_speaker_token_text(
    tokenizer: _TokenizerWithEncode,
    expected_speaker_token_id: int | None = None,
) -> str:
    token_id = resolve_speaker_token_id(
        tokenizer,
        expected_speaker_token_id=expected_speaker_token_id,
    )
    token_text = tokenizer.decode([token_id])
    if not token_text:
        raise ValueError(f"Tokenizer returned empty text for speaker token id {token_id}.")
    return token_text


# ---------------------------------------------------------------------------
# Placeholder counting / injection
# ---------------------------------------------------------------------------


def count_audio_placeholders_str(text: str) -> int:
    """Count all placeholder variant sequences in a string."""
    return sum(text.count(v) for v in ALL_AUDIO_PLACEHOLDER_VARIANTS)


def normalize_placeholders_str(text: str) -> str:
    """Replace legacy ``<|audio_pad|>`` placeholders with the canonical form."""
    return text.replace(LEGACY_AUDIO_PLACEHOLDER_SEQ, AUDIO_PLACEHOLDER_SEQ)


def inject_placeholders_into_str(
    prompt: str,
    *,
    num_audios: int,
) -> str:
    """Ensure *prompt* has exactly *num_audios* placeholders, filling empty user turns."""
    if num_audios <= 0:
        return prompt

    prompt = normalize_placeholders_str(prompt)
    existing = count_audio_placeholders_str(prompt)
    missing = num_audios - existing
    if missing <= 0:
        return prompt

    marker = USER_PROMPT_MARKER
    positions: list[int] = []
    start = 0
    while True:
        idx = prompt.find(marker, start)
        if idx < 0:
            break
        positions.append(idx + len(marker))
        start = idx + len(marker)

    if not positions:
        return (AUDIO_PLACEHOLDER_SEQ * missing) + prompt

    turns_without: list[int] = []
    for i, pos in enumerate(positions):
        end = (positions[i + 1] - len(marker)) if i + 1 < len(positions) else len(prompt)
        region = prompt[pos:end]
        if count_audio_placeholders_str(region) == 0:
            turns_without.append(i)

    offset = 0
    for turn_idx in turns_without:
        if missing <= 0:
            break
        insert_at = positions[turn_idx] + offset
        prompt = prompt[:insert_at] + AUDIO_PLACEHOLDER_SEQ + prompt[insert_at:]
        offset += len(AUDIO_PLACEHOLDER_SEQ)
        missing -= 1

    if missing > 0:
        prompt = (AUDIO_PLACEHOLDER_SEQ * missing) + prompt

    return prompt


def inject_placeholders_into_token_ids(
    prompt_ids: list[int],
    *,
    num_audios: int,
    ph_ids: list[int],
    legacy_ph_ids: list[int] | None = None,
    marker_ids: list[int] | None = None,
) -> list[int]:
    """Token-ID-level equivalent of ``inject_placeholders_into_str``."""
    if num_audios <= 0:
        return prompt_ids

    def _count_subsequence(seq: list[int], pattern: list[int]) -> int:
        plen = len(pattern)
        if plen == 0 or len(seq) < plen:
            return 0
        return sum(1 for i in range(len(seq) - plen + 1) if seq[i : i + plen] == pattern)

    existing = _count_subsequence(prompt_ids, ph_ids)
    if legacy_ph_ids:
        existing += _count_subsequence(prompt_ids, legacy_ph_ids)
    missing = num_audios - existing
    if missing <= 0:
        return prompt_ids

    if marker_ids:
        mlen = len(marker_ids)
        positions: list[int] = []
        for i in range(len(prompt_ids) - mlen + 1):
            if prompt_ids[i : i + mlen] == marker_ids:
                positions.append(i + mlen)

        if positions:
            turns_without: list[int] = []
            for ti, pos in enumerate(positions):
                end = (positions[ti + 1] - mlen) if ti + 1 < len(positions) else len(prompt_ids)
                region = prompt_ids[pos:end]
                has = _count_subsequence(region, ph_ids) > 0
                if not has and legacy_ph_ids:
                    has = _count_subsequence(region, legacy_ph_ids) > 0
                if not has:
                    turns_without.append(ti)

            offset = 0
            for turn_idx in turns_without:
                if missing <= 0:
                    break
                insert_at = positions[turn_idx] + offset
                prompt_ids = prompt_ids[:insert_at] + ph_ids + prompt_ids[insert_at:]
                offset += len(ph_ids)
                missing -= 1

    if missing > 0:
        prompt_ids = (ph_ids * missing) + prompt_ids

    return prompt_ids


# ---------------------------------------------------------------------------
# Token ID normalisation
# ---------------------------------------------------------------------------


def normalize_token_ids(
    prompt_token_ids: list[int],
    *,
    special: RaonResolvedIds,
) -> list[int]:
    """Replace output-placeholder IDs with input-placeholder, ensuring the
    ``[audio_start, audio_input, audio_end]`` triple is present.

    ``special`` must be a :class:`RaonResolvedIds` resolved from the live
    tokenizer (e.g. the processor's cached ``self._ids`` or the serving
    hook's ``self._resolved_ids``) so checkpoint-drifted IDs are respected.
    """
    out_id = special.audio_output_placeholder
    in_id = special.audio_input_placeholder
    start_id = special.audio_start
    end_id = special.audio_end

    result = [in_id if tok == out_id else tok for tok in prompt_token_ids]

    expected = [start_id, in_id, end_id]
    if in_id in result:
        n = len(expected)
        found = any(result[i : i + n] == expected for i in range(len(result) - n + 1))
        if not found:
            result = expected + result

    return result


# ---------------------------------------------------------------------------
# OutputMode enum
# ---------------------------------------------------------------------------


class OutputMode(enum.Enum):
    TEXT_ONLY = "text_only"
    AUDIO_ONLY = "audio_only"
    TEXT_AND_AUDIO = "text_and_audio"


# ---------------------------------------------------------------------------
# TaskType + registry (imported from configs when available)
# ---------------------------------------------------------------------------

try:
    from vllm_omni.transformers_utils.configs.raon import (  # type: ignore[import-untyped]
        TASK_REGISTRY,
        TaskConfig,
        TaskType,
    )

    def _output_mode_for_task(task: TaskType) -> OutputMode:
        cfg: TaskConfig = TASK_REGISTRY[task]
        return OutputMode(cfg.output_mode)

    def _task_has_audio_input(task: TaskType) -> bool:
        cfg: TaskConfig = TASK_REGISTRY[task]
        return "audio" in cfg.input_modalities

except ImportError:

    class TaskType(enum.Enum):  # type: ignore[no-redef]
        TEXT_QA = "text_qa"
        STT = "stt"
        TTS = "tts"
        SPOKEN_QA = "spoken_qa"
        SPEECH_QA = "speech_qa"

    _TASK_OUTPUT_MODE: dict[TaskType, OutputMode] = {
        TaskType.TEXT_QA: OutputMode.TEXT_ONLY,
        TaskType.STT: OutputMode.TEXT_ONLY,
        TaskType.TTS: OutputMode.AUDIO_ONLY,
        TaskType.SPOKEN_QA: OutputMode.TEXT_ONLY,
        TaskType.SPEECH_QA: OutputMode.TEXT_ONLY,
    }

    _TASK_HAS_AUDIO_INPUT: set[TaskType] = {
        TaskType.STT,
        TaskType.SPOKEN_QA,
        TaskType.SPEECH_QA,
    }

    def _output_mode_for_task(task: TaskType) -> OutputMode:  # type: ignore[no-redef]
        return _TASK_OUTPUT_MODE[task]

    def _task_has_audio_input(task: TaskType) -> bool:  # type: ignore[no-redef]
        return task in _TASK_HAS_AUDIO_INPUT


# ---------------------------------------------------------------------------
# PromptResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptResult:
    """Immutable container returned by ``build_prompt``."""

    prompt_text: str
    task: TaskType
    output_mode: OutputMode
    has_audio_input: bool
    audio_count: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# RaonChatTemplateBuilder
# ---------------------------------------------------------------------------


class RaonChatTemplateBuilder:
    """Single entry-point for building Raon prompts across all task types."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def build_prompt(
        self,
        task: TaskType,
        *,
        user_content: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        audio_count: int = 0,
        instruction: str | None = None,
        system_prompt: str | None = None,
        tts_prompt_style: str = "instruction",
        prepend_speaker_token: bool = False,
        append_audio_start: bool = False,
    ) -> PromptResult:
        if messages is not None:
            chat_messages = self._build_messages_from_multi_turn(
                messages,
                task=task,
                audio_count=audio_count,
            )
            total_audio = sum(m.get("_audio_count", 0) for m in chat_messages)
        else:
            chat_messages, total_audio = self._build_messages_single_turn(
                task=task,
                user_content=user_content or "",
                audio_count=audio_count,
                instruction=instruction,
                tts_prompt_style=tts_prompt_style,
                prepend_speaker_token=prepend_speaker_token,
            )

        if system_prompt is not None:
            chat_messages = [
                {"role": "system", "content": system_prompt},
                *chat_messages,
            ]

        clean_messages = [{"role": m["role"], "content": m["content"]} for m in chat_messages]

        prompt_text = self._apply_chat_template(clean_messages)

        if task == TaskType.TTS and append_audio_start:
            if not prompt_text.endswith(AUDIO_START.text):
                prompt_text = f"{prompt_text}{AUDIO_START.text}"

        output_mode = _output_mode_for_task(task)
        has_audio = _task_has_audio_input(task) and total_audio > 0

        return PromptResult(
            prompt_text=prompt_text,
            task=task,
            output_mode=output_mode,
            has_audio_input=has_audio,
            audio_count=total_audio,
        )

    def encode_prompt(self, prompt_text: str) -> list[int]:
        if hasattr(self._tokenizer, "encode"):
            return self._tokenizer.encode(prompt_text, add_special_tokens=False)
        raise TypeError("Tokenizer does not support encode()")

    # ------------------------------------------------------------------
    # Internal: single-turn message construction
    # ------------------------------------------------------------------

    def _build_messages_single_turn(
        self,
        *,
        task: TaskType,
        user_content: str,
        audio_count: int,
        instruction: str | None,
        tts_prompt_style: str,
        prepend_speaker_token: bool,
    ) -> tuple[list[dict[str, Any]], int]:
        if task == TaskType.TEXT_QA:
            return [{"role": "user", "content": user_content, "_audio_count": 0}], 0

        if task == TaskType.TTS:
            content = self._build_tts_content(
                user_content,
                tts_prompt_style,
                prepend_speaker_token,
            )
            return [{"role": "user", "content": content, "_audio_count": 0}], 0

        if task == TaskType.STT:
            ac = max(audio_count, 1)
            content = _build_audio_content(
                text_after=instruction or "Transcribe this audio.",
                audio_count=ac,
            )
            return [{"role": "user", "content": content, "_audio_count": ac}], ac

        if task == TaskType.SPOKEN_QA:
            ac = max(audio_count, 1)
            content = _build_audio_content(text_after=None, audio_count=ac)
            return [{"role": "user", "content": content, "_audio_count": ac}], ac

        if task == TaskType.SPEECH_QA:
            ac = max(audio_count, 1)
            existing = count_audio_placeholders_str(user_content)
            needed = max(ac - existing, 0)
            content = _build_audio_content(text_after=user_content, audio_count=needed)
            return [{"role": "user", "content": content, "_audio_count": ac}], ac

        if task.value == "tts_icl":
            content = self._build_tts_content(
                user_content,
                tts_prompt_style,
                prepend_speaker_token,
            )
            return [{"role": "user", "content": content, "_audio_count": 0}], 0

        raise ValueError(f"Unsupported task: {task}")

    # ------------------------------------------------------------------
    # Internal: multi-turn message construction
    # ------------------------------------------------------------------

    def _build_messages_from_multi_turn(
        self,
        messages: list[dict[str, Any]],
        *,
        task: TaskType,
        audio_count: int,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            msg_audio = msg.get("audio_count", 0)

            content = normalize_placeholders_str(content)

            if role == "user" and _task_has_audio_input(task) and msg_audio > 0:
                existing = count_audio_placeholders_str(content)
                needed = msg_audio - existing
                if needed > 0:
                    content = (AUDIO_PLACEHOLDER_SEQ * needed) + content
                result.append(
                    {
                        "role": role,
                        "content": content,
                        "_audio_count": msg_audio,
                    }
                )
            else:
                result.append(
                    {
                        "role": role,
                        "content": content,
                        "_audio_count": 0,
                    }
                )

        return result

    # ------------------------------------------------------------------
    # Internal: content builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_tts_content(
        text: str,
        prompt_style: str,
        prepend_speaker_token: bool,
    ) -> str:
        if prompt_style == "raw":
            content = text
        else:
            content = f"Speak the following text:\n{text}"

        if prepend_speaker_token:
            speaker_token = SPEAKER_EMBEDDING_PLACEHOLDER.text
            if not content.startswith(speaker_token):
                content = f"{speaker_token}{content}"

        return content

    # ------------------------------------------------------------------
    # Internal: chat template application
    # ------------------------------------------------------------------

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        tokenizer = self._tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                result = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if isinstance(result, str) and result:
                    return result
            except Exception:
                pass

        return self._fallback_chat_template(messages)

    @staticmethod
    def _fallback_chat_template(messages: list[dict[str, str]]) -> str:
        parts: list[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"{IM_START.text}{role}\n{content}{IM_END.text}\n")
        parts.append(f"{IM_START.text}assistant\n")
        return "".join(parts)


def _build_audio_content(
    *,
    text_after: str | None,
    audio_count: int,
) -> str:
    placeholders = AUDIO_PLACEHOLDER_SEQ * audio_count
    if text_after:
        return f"{placeholders}{text_after}"
    return placeholders


__all__ = [
    "ALL_SPECIAL_TOKENS",
    "AUDIO_END",
    "AUDIO_END_TOKEN",
    "AUDIO_INPUT_PAD_TOKEN",
    "AUDIO_INPUT_PLACEHOLDER",
    "AUDIO_OUTPUT_END_PAD",
    "AUDIO_OUTPUT_PAD",
    "AUDIO_OUTPUT_PAD_TOKEN",
    "AUDIO_OUTPUT_PLACEHOLDER",
    "AUDIO_PLACEHOLDER_PATTERN",
    "AUDIO_PLACEHOLDER_SEQ",
    "AUDIO_PLACEHOLDER_TOKENS",
    "AUDIO_START",
    "AUDIO_START_TOKEN",
    "ALL_AUDIO_PLACEHOLDER_VARIANTS",
    "AUDIO_OUTPUT_OPEN_SEQ",
    "AUDIO_OUTPUT_PLACEHOLDER_SEQ",
    "IM_END",
    "IM_START",
    "LEGACY_AUDIO_PAD_TOKEN",
    "LEGACY_AUDIO_PLACEHOLDER_SEQ",
    "LEGACY_SECONDARY_AUDIO_PAD_TOKEN",
    "LEGACY_SPEAKER_PAD_TOKEN",
    "PAD",
    "SPEAKER_EMBEDDING_PLACEHOLDER",
    "SPEAKER_EMBEDDING_PLACEHOLDER_ID",
    "SPEAKER_EMBEDDING_PLACEHOLDER_TOKEN",
    "SpecialToken",
    "RaonResolvedIds",
    "USER_PROMPT_MARKER",
    "align_tokenizer",
    "filter_audio_placeholder_text",
    "resolve_audio_input_token_id",
    "resolve_raon_special_ids",
    "resolve_speaker_token_id",
    "resolve_speaker_token_text",
    "count_audio_placeholders_str",
    "inject_placeholders_into_str",
    "inject_placeholders_into_token_ids",
    "normalize_placeholders_str",
    "normalize_token_ids",
    "OutputMode",
    "PromptResult",
    "TaskType",
    "RaonChatTemplateBuilder",
]
