# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for the offline Audex examples in this directory.

Import style follows the repo's example convention (same-directory module,
see ``text_to_speech/qwen3_tts/tts_common.py``): scripts run as files, so
``from common import ...`` resolves via the script's own directory.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from vllm_omni import Omni
from vllm_omni.model_executor.models.audex.prompt import build_null_prompt

SAMPLE_RATE = 16_000
_DEPLOY_DIR = Path(__file__).resolve().parents[3] / "vllm_omni" / "deploy"

# Guided decoding sharpens the distribution, so the unguided temperature
# (0.1) adds excess sampling noise under CFG. Measured on the en-24 gate
# corpus at cfg 1.5: temp 0.1 -> CER 7.31%, temp 0.05 -> CER 6.87% (vs the
# unguided baseline 7.24%).
GUIDED_TEMPERATURE = 0.05


def default_deploy_config(name: str) -> str:
    """Absolute path of a deploy yaml bundled with vllm_omni."""
    return str(_DEPLOY_DIR / name)


def slugify(text: str, fallback: str = "prompt") -> str:
    slug = re.sub(r"\s+", "_", text.strip().lower())
    slug = re.sub(r"[^a-z0-9_]+", "", slug)
    return slug[:48] or fallback


def load_corpus(
    corpus_file: str | None,
    texts: list[str] | None,
    default_texts: tuple[str, ...],
) -> list[tuple[str, str]]:
    """(utt_id, text) pairs from a TSV corpus file, CLI texts, or defaults."""
    if corpus_file:
        corpus = []
        for line in Path(corpus_file).read_text().splitlines():
            if not line.strip():
                continue
            utt, text = line.split("\t", 1)
            corpus.append((utt, text.strip()))
        return corpus
    resolved = texts if texts else list(default_texts)
    return [(slugify(t), t) for t in resolved]


def load_audio_file(path: str) -> tuple[np.ndarray, int]:
    """Mono float32 samples + native sample rate (processors resample)."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio, sr


def extract_pcm(multimodal_output: dict) -> torch.Tensor:
    """Flatten one request's audio payload to a 1-D float32 CPU tensor."""
    audio = multimodal_output.get("model_outputs")
    if audio is None:
        audio = multimodal_output.get("audio")
    if audio is None:
        raise ValueError(f"no audio key in multimodal_output: {list(multimodal_output.keys())}")
    if isinstance(audio, list):
        valid = [torch.as_tensor(a).float().cpu().reshape(-1) for a in audio if a is not None]
        if not valid:
            raise ValueError("audio list is empty")
        return torch.cat(valid, dim=0) if len(valid) > 1 else valid[0]
    return torch.as_tensor(audio).float().cpu().reshape(-1)


def write_wav(path: Path, pcm: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.clip(pcm.numpy(), -1.0, 1.0) * 32767.0).astype(np.int16)
    sf.write(str(path), arr, sample_rate, format="WAV", subtype="PCM_16")


def load_audex_tokenizer(model: str, profile: str = "tts", folder: str = "checkpoint_folder_audiogen"):
    """Thinker tokenizer for CFG/TTA prompt building (resolves the snapshot)."""
    from transformers import AutoTokenizer

    from vllm_omni.model_executor.models.audex.checkpoint import ensure_audex_snapshot

    root = ensure_audex_snapshot(model, profile=profile)
    return AutoTokenizer.from_pretrained(str(Path(root) / folder), trust_remote_code=True)


def request_index(req_output) -> int:
    """Submission index from the numeric request-id prefix.

    ``Omni.generate`` may return outputs out of submission order, and a
    lexicographic sort would misplace "10_..." before "2_...".
    """
    match = re.search(r"(\d+)", str(req_output.request_id))
    return int(match.group(1)) if match else 0


def guided_stage_params(
    engine: Omni,
    cond_prompt: str,
    tokenizer,
    cfg_scale: float,
    pair_id: str,
    *,
    temperature: float | None = None,
    extra_args: dict[str, Any] | None = None,
    null_prompt_builder: Callable[[str, Any], str] = build_null_prompt,
) -> list:
    """Stage sampling params carrying the CFG pair contract for ONE request.

    Clones the engine's resolved defaults (never mutates the shared list),
    applies ``extra_args`` unconditionally (e.g. the TTA RVQ contract), and
    attaches the cond/uncond pair contract when ``cfg_scale > 1.0``. Guided
    requests must go one at a time: each needs its own pair id and a
    length-matched null prompt built from its own text.
    """
    params = copy.deepcopy(engine.resolve_sampling_params_list(None))
    stage0 = params[0]
    if temperature is not None:
        stage0.temperature = temperature
    if stage0.extra_args is None:
        stage0.extra_args = {}
    if extra_args:
        stage0.extra_args.update(extra_args)
    if cfg_scale > 1.0:
        stage0.extra_args.update(
            {
                "cfg_scale": float(cfg_scale),
                "cfg_role": "cond",
                "cfg_pair_id": pair_id,
                "cfg_null_prompt": null_prompt_builder(cond_prompt, tokenizer),
            }
        )
    return params
