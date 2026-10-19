# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PersonaPlex frame/token contract and system-prompt assets.

Constants and loaders shared by every PersonaPlex serving path (the full-duplex
plugin and OpenAI Live): the Mimi frame geometry, the lockstep token constants
of the hybrid system prompt, and the bundled voice prompts and persona
tokenizer read from the checkpoint directory.

Frame layout per 80 ms tick: 17 token rows -- row 0 inner-monologue text,
rows 1-8 agent audio codebooks, rows 9-16 user audio codebooks.
"""

from __future__ import annotations

import io
import tarfile
from functools import lru_cache
from pathlib import Path
from typing import Any

# Mimi codec constants (loaders.py: SAMPLE_RATE / FRAME_RATE).
SAMPLE_RATE = 24000
FRAME_RATE = 12.5
FRAME_SIZE = int(SAMPLE_RATE / FRAME_RATE)  # 1920 samples per 80 ms frame
CHUNK_PERIOD_MS = int(round(1000 / FRAME_RATE))  # 80

# Default assistant persona shipped with PersonaPlex (offline.py:335).
DEFAULT_PERSONA = "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."
# Bundled voice prompt used when a session names none.
DEFAULT_VOICE = "NATF2.pt"

# Inner-monologue token ids that carry no displayable text (EPAD, PAD).
NON_TEXT_TOKEN_IDS = frozenset({0, 3})
ZERO_TEXT_TOKEN = 3
# Mimi token constants for the hybrid system prompt (agent silence / user sine).
SILENCE_TOKENS = (948, 243, 1178, 546, 1736, 1030, 1978, 2008)
SINE_TOKENS = (430, 1268, 381, 1611, 1095, 1495, 56, 472)
# LM initial ("beginning of stream") token ids: audio rows use the codec
# cardinality, the text row uses the text vocabulary size.
AUDIO_INITIAL_TOKEN = 2048
TEXT_INITIAL_TOKEN = 32000
TEXT_VOCAB_SIZE = 32000
AUDIO_SILENCE_FRAME_CNT = 6  # 0.5 s at 12.5 Hz


@lru_cache(maxsize=4)
def personaplex_model_dir(model_path: str) -> str:
    """Local checkpoint directory for a path or a Hugging Face repo id."""
    if Path(model_path).is_dir():
        return model_path
    from vllm_omni.transformers_utils.repo_utils import hf_api

    return hf_api().snapshot_download(model_path, allow_patterns=["tokenizer_spm_32k_3.model", "voices*"])


def wrap_with_system_tags(text: str) -> str:
    t = text.strip()
    return t if t.startswith("<system>") else f"<system> {t} <system>"


def _tokenizer_path(model_path: str) -> Path:
    path = Path(model_path) / "tokenizer_spm_32k_3.model"
    if not path.is_file():
        raise FileNotFoundError(f"PersonaPlex tokenizer not found: {path}")
    return path


def load_personaplex_tokenizer(model_path: str):
    import sentencepiece

    tokenizer = sentencepiece.SentencePieceProcessor(str(_tokenizer_path(model_path)))
    return lambda text: list(tokenizer.encode(text))


def load_personaplex_voice_state(model_path: str, voice: str) -> dict[str, Any]:
    import torch

    root = Path(model_path)
    for candidate in (root / "voices" / voice, root / voice):
        if candidate.is_file():
            state = torch.load(candidate, map_location="cpu", weights_only=True)
            if isinstance(state, dict):
                return state
            raise ValueError(f"PersonaPlex voice bundle is not a mapping: {candidate}")

    archive = root / "voices.tgz"
    if archive.is_file():
        data = _voice_archive_members(str(archive)).get(voice)
        if data is not None:
            state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
            if isinstance(state, dict):
                return state
            raise ValueError(f"PersonaPlex voice bundle {voice!r} is not a mapping")
    raise FileNotFoundError(f"PersonaPlex bundled voice prompt {voice!r} was not found under {model_path!r}")


@lru_cache(maxsize=2)
def _voice_archive_members(archive: str) -> dict[str, bytes]:
    """Every file of a ``voices.tgz`` bundle by base name, read in one pass.

    A gzip stream has no random access, so each lookup would decompress the
    archive from the start; the bundle is small (about 8 MB unpacked), so read it
    once and keep the members. The first file with a given base name wins.
    """
    members: dict[str, bytes] = {}
    with tarfile.open(archive, "r:gz") as tar:
        for item in tar:
            if not item.isfile():
                continue
            extracted = tar.extractfile(item)
            if extracted is not None:
                members.setdefault(Path(item.name).name, extracted.read())
    return members


@lru_cache(maxsize=4)
def cached_personaplex_tokenizer(model_path: str):
    return load_personaplex_tokenizer(model_path)


@lru_cache(maxsize=16)
def _cached_voice_embedding_rows(model_path: str, voice: str) -> int:
    state = load_personaplex_voice_state(model_path, voice)
    embeddings = state.get("embeddings")
    if not hasattr(embeddings, "shape") or len(embeddings.shape) < 1:
        raise ValueError(f"PersonaPlex voice prompt {voice!r} has no embeddings")
    return int(embeddings.shape[0])


def personaplex_prefill_slots(model_path: str, voice: str, persona: str) -> int:
    """Scheduler slots the first append of a session needs for the voice + persona prefill.

    The voice bundle row count and the tokenizer are cached per model path:
    they are constant, and this runs on every session open.
    """
    voice_rows = _cached_voice_embedding_rows(model_path, voice)
    persona_tokens = cached_personaplex_tokenizer(model_path)(wrap_with_system_tags(persona)) if persona else []
    return voice_rows + 2 * AUDIO_SILENCE_FRAME_CNT + len(persona_tokens)


__all__ = [
    "AUDIO_INITIAL_TOKEN",
    "AUDIO_SILENCE_FRAME_CNT",
    "CHUNK_PERIOD_MS",
    "DEFAULT_PERSONA",
    "DEFAULT_VOICE",
    "FRAME_RATE",
    "FRAME_SIZE",
    "NON_TEXT_TOKEN_IDS",
    "SAMPLE_RATE",
    "SILENCE_TOKENS",
    "SINE_TOKENS",
    "TEXT_INITIAL_TOKEN",
    "TEXT_VOCAB_SIZE",
    "ZERO_TEXT_TOKEN",
    "cached_personaplex_tokenizer",
    "load_personaplex_tokenizer",
    "load_personaplex_voice_state",
    "personaplex_model_dir",
    "personaplex_prefill_slots",
    "wrap_with_system_tags",
]
