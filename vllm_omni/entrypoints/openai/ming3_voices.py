# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-bundled Ming3 voices, persisted through the Speech API voice store."""

import asyncio
import hashlib
import io
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from starlette.datastructures import Headers, UploadFile
from vllm.logger import init_logger

from vllm_omni.model_executor.models.ming_flash_omni.reference_audio import pack_reference_audio

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech

logger = init_logger(__name__)


def voice_id(value: Any) -> str:
    """Accept an OpenAI voice name or a custom voice object {\"id\": ...}."""
    if isinstance(value, dict):
        value = value.get("id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("voice must be a nonempty string or an object with a nonempty 'id'")
    return value.strip().lower()


async def register_model_voice_presets(
    server: "OmniOpenAIServingSpeech", model: str, *, revision: str | None = None, cache_dir: str | None = None
) -> None:
    """Import the model's manifest; unchanged registrations survive restarts.

    No assets are written to the model directory. An invalid manifest fails
    startup instead of leaving the server silently missing an advertised voice.
    Runtime uploads with the same name are preserved.
    """
    root = Path(model).expanduser()
    if not root.is_dir():
        from vllm_omni.transformers_utils.repo_utils import hf_api

        root = Path(
            await asyncio.to_thread(
                hf_api().snapshot_download,
                model,
                revision=revision,
                cache_dir=cache_dir,
                allow_patterns=["voice_presets/**"],
            )
        )
    presets = root / "voice_presets"
    manifest = presets / "voice_prompts.json"
    if not manifest.is_file():
        logger.info("No Ming3 voice manifest at %s", manifest)
        return
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(entries, dict):
        raise ValueError(f"Expected a voice-name mapping in {manifest}")
    from vllm_omni.entrypoints.openai.serving_speech import _validate_speaker_name
    from vllm_omni.model_executor.models.ming_flash_omni.omni_voice import ip_voice_name

    pending = []
    names = set()
    for name, entry in entries.items():
        name = _validate_speaker_name(name)
        key = voice_id(name)
        if ip_voice_name(key) is not None:
            raise ValueError(f"Ming3 voice name {name!r} is reserved by an IP voice; choose a different name")
        if key in names:
            raise ValueError(f"Duplicate Ming3 voice name (case insensitive): {name}")
        names.add(key)
        if not isinstance(entry, dict) or not isinstance(entry.get("text"), str):
            raise ValueError(f"Ming3 voice {name!r} requires a string 'text'")
        audio = entry.get("audio")
        if not isinstance(audio, str) or not audio or Path(audio).is_absolute() or ".." in Path(audio).parts:
            raise ValueError(f"Ming3 voice {name!r} requires a relative 'audio' path")
        # HF snapshots store assets as symlinks into the shared blob cache.
        # Validate the manifest path without rejecting those cached assets.
        path = presets / audio
        content = await asyncio.to_thread(path.read_bytes)
        fingerprint = hashlib.sha256(entry["text"].encode() + b"\0" + content).hexdigest()
        pending.append((name, key, path, entry["text"], content, fingerprint))

    for name, key, path, text, content, fingerprint in pending:
        previous = server.uploaded_speakers.get(key)
        if previous:
            if not previous.get("model_preset_fingerprint"):
                logger.info("Preserving runtime-uploaded voice %r over model preset", name)
                continue
            if previous["model_preset_fingerprint"] == fingerprint:
                continue
        sample = UploadFile(io.BytesIO(content), filename=path.name, headers=Headers({"content-type": "audio/wav"}))
        try:
            await server.upload_voice(
                sample,
                "model-preset",
                name,
                ref_text=text,
                model_preset_fingerprint=fingerprint,
            )
        finally:
            await sample.close()
    logger.info("Registered Ming3 model voices from %s: %s", manifest, sorted(names))


async def resolve_stored_voice(server: "OmniOpenAIServingSpeech", value: Any) -> dict[str, Any]:
    """Resolve the same stored voice for Chat, Realtime and Talker handoffs."""
    name = voice_id(value)
    entry = server.uploaded_speakers.get(name)
    if entry is None:
        raise ValueError(f"Unknown voice {name!r}; list available voices with GET /v1/audio/voices")
    if entry.get("embedding_source") == "direct":
        embedding = server._get_uploaded_speaker_embedding(name)
        if embedding is None or len(embedding) != 192:
            raise ValueError(f"Ming3 voice {name!r} requires a 192-dimensional speaker embedding")
        return {"spk_emb": embedding, "prompt_audio_mode": "speaker_only"}
    audio = await asyncio.to_thread(server._load_uploaded_audio, name)
    if audio is None:
        raise ValueError(f"Audio for voice {name!r} is missing")
    samples, sr = audio
    text = entry.get("ref_text") or None
    return {
        "voice_name": name,
        **pack_reference_audio([(samples, sr)]),
        "prompt_text": text,
        "prompt_audio_mode": "continuation" if text else "speaker_only",
    }
