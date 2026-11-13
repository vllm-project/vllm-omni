# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU prompt rows and logical lengths, following the official prompt plan.

The worker ModelState builds actual embeddings. Text normalization/tokenization
and delay construction here never invoke the AR Backbone or Codec decoder.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

from .ar_config import ZONOS3_PROMPT_KEY, Zonos3Architecture

SILENCE_FRAME = (1918, 746, 3544, 234, 1267, 2914, 3944, 2849, 326, 1337)
NV_TAGS = ("laugh", "breath", "sigh", "cough", "panting", "shouting", "crying")


def delayed_rows(codes: torch.Tensor, pad: int, *, flush: bool) -> torch.Tensor:
    length = codes.shape[0] + (codes.shape[1] - 1 if flush else 0)
    result = torch.full((length, codes.shape[1]), pad, dtype=torch.int64)
    for book in range(codes.shape[1]):
        available = min(codes.shape[0], length - book)
        if available > 0:
            result[book : book + available, book] = codes[:available, book]
    return result


def condition_ids(
    config: Zonos3Architecture, values: dict[str, float] | None, buckets: dict[str, int] | None
) -> list[int]:
    values, buckets = values or {}, buckets or {}
    unknown = (set(values) | set(buckets)) - set(config.condition_names)
    if unknown:
        raise ValueError(f"Unknown Zonos3 condition names: {sorted(unknown)}")
    output = []
    for name, count in zip(config.condition_names, config.condition_bucket_counts, strict=True):
        if buckets.get(name) is not None:
            bucket = buckets[name]
            if not isinstance(bucket, int) or not 0 <= bucket <= count:
                raise ValueError(f"Invalid condition bucket for {name}")
        elif values.get(name) is not None:
            value = float(values[name])
            edges = config.condition_edges.get(name)
            if not math.isfinite(value) or not edges:
                raise ValueError(f"A finite value and checkpoint bucket edges are required for {name}")
            bucket = bisect.bisect_right(edges, value)
        else:
            bucket = count  # the release's unknown row
        output.append(bucket)
    return output


class Zonos3PromptBuilder:
    def __init__(self, config: Zonos3Architecture, model: str, subdir: str, revision: str | None = None) -> None:
        import zyphra_textnorm

        self.config = config
        self.model, self.subdir, self.revision = model, subdir, revision
        registry = json.loads(Path(__file__).with_name("languages.json").read_text())
        self.languages = {entry["code"]: entry for entry in registry["server"]}
        self.aliases = {entry["label"].casefold(): entry["code"] for entry in registry["server"]}
        for entry in registry["spoken"]:
            if entry.get("tts") in self.languages:
                self.aliases.setdefault(entry["code"], entry["tts"])
                self.aliases.setdefault(entry["name"].casefold(), entry["tts"])
        self._normalizer = zyphra_textnorm.Normalizer(NV_TAGS)
        self._collapse = zyphra_textnorm.collapse_line_breaks
        self.tokenizer = None
        if config.text_tokenizer_spec is not None:
            spec = config.text_tokenizer_spec
            path = self._tokenizer_path(str(spec["path"]))
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != spec["sha256"]:
                raise ValueError("Zonos3 SuperBPE tokenizer checksum mismatch")
            self.tokenizer = Tokenizer.from_str(raw.decode())
            if (
                self.tokenizer.get_vocab_size(with_added_tokens=True) != spec["vocab_size"]
                or self.tokenizer.padding is not None
                or self.tokenizer.truncation is not None
            ):
                raise ValueError("Zonos3 SuperBPE artifact differs from its release specification")
        elif config.text_vocab < 448:
            raise ValueError("Zonos3 byte-tokenizer checkpoints need the 192+256 byte vocabulary")

    def _tokenizer_path(self, reference: str) -> Path:
        root = Path(self.model)
        model_dir = root if (root / "params.json").is_file() else root / self.subdir
        path = Path(reference).expanduser()
        if reference.startswith("hf://"):
            parts = reference[5:].split("/")
            repo, filename = "/".join(parts[:2]), "/".join(parts[2:])
            folder = download_weights_from_hf_specific(
                repo, None, [filename], revision=self.revision if repo == self.model else None, require_all=True
            )
            return Path(folder) / filename
        if path.is_absolute() and path.is_file():
            return path
        relative = f"tokenizer/{path.name}" if path.is_absolute() else reference
        if (model_dir / relative).is_file():
            return model_dir / relative
        if root.is_dir():
            raise FileNotFoundError(model_dir / relative)
        filename = f"{self.subdir}/{relative}"
        folder = download_weights_from_hf_specific(
            self.model, None, [filename], revision=self.revision, require_all=True
        )
        return Path(folder) / filename

    def canonical_language(self, language: str | None) -> str:
        value = (language or "en_us").casefold()
        if value == "auto":
            value = "en_us"
        value = self.aliases.get(value, value)
        if value not in self.languages:
            raise ValueError(f"Unsupported Zonos3 language: {language}")
        return value

    def text_ids(self, text: str, language: str | None, normalize: bool) -> list[int]:
        language = self.canonical_language(language)
        text = self._collapse(text)
        if normalize:
            nemo = self.languages[language].get("nemo")
            if nemo:
                text = self._normalizer.normalize(text, nemo)
        if self.tokenizer is None:
            return [2, *(byte + 192 for byte in text.encode("utf-8")), 3]
        spec = self.config.text_tokenizer_spec
        phonemes = spec.get("phonemes")

        def bpe(stretch: str) -> list[int]:
            return [token + 4 for token in self.tokenizer.encode(stretch, add_special_tokens=False).ids]

        if phonemes is None:
            return [2, *bpe(text), 3]
        opening, closing = phonemes["open_marker"], phonemes["close_marker"]
        phone_ids = {symbol: phonemes["open_id"] + 2 + i for i, symbol in enumerate(phonemes["symbols"])}
        output, position = [], 0
        while position < len(text):
            start = text.find(opening, position)
            stretch = text[position:] if start < 0 else text[position:start]
            if closing in stretch:
                raise ValueError("Unbalanced Zonos3 phoneme group")
            output.extend(bpe(stretch))
            if start < 0:
                break
            end = text.find(closing, start + 1)
            group = text[start + 1 : end] if end >= 0 else ""
            if end < 0 or not group or opening in group or set(group) - phone_ids.keys():
                raise ValueError("Invalid Zonos3 phoneme group")
            output.extend([phonemes["open_id"], *(phone_ids[symbol] for symbol in group), phonemes["close_id"]])
            position = end + 1
        return [2, *output, 3]

    def build(
        self,
        text: str,
        *,
        language: str | None = None,
        normalize: bool = True,
        speaker_embedding: torch.Tensor | None = None,
        reference_codes: torch.Tensor | None = None,
        values: dict[str, float] | None = None,
        buckets: dict[str, int] | None = None,
        silence_frames: int = 3,
    ) -> dict[str, Any]:
        c = self.config
        if not text.strip() or not isinstance(silence_frames, int) or not 0 <= silence_frames <= 30:
            raise ValueError("Zonos3 needs non-empty text and silence_prefill_frames in 0..30")
        if speaker_embedding is not None and reference_codes is not None:
            raise ValueError("Zonos3 speaker embedding and reference-prefix cloning are mutually exclusive")
        tokens = self.text_ids(text, language, normalize)
        rows = torch.full((len(tokens), 11), c.audio_pad_id, dtype=torch.int64)
        rows[:, 10] = torch.tensor(tokens)
        conditions = torch.tensor([[c.audio_pad_id] * 10 + [c.text_vocab]] * c.condition_count)
        prefix = torch.empty((0, 11), dtype=torch.int64)
        if reference_codes is not None:
            if not c.speaker_reference_prefix_enabled:
                raise ValueError("This Zonos3 checkpoint does not support reference-prefix cloning")
            if (
                reference_codes.device.type != "cpu"
                or reference_codes.dtype not in (torch.int32, torch.int64)
                or reference_codes.ndim != 2
                or reference_codes.shape[1] != 10
                or not 0 < len(reference_codes) <= 900
            ):
                raise ValueError("Zonos3 reference codes must be CPU [1..900 frames, 10 codebooks]")
            if bool(((reference_codes < 0) | (reference_codes >= c.codebook_size)).any()):
                raise ValueError("Zonos3 reference code is outside its vocabulary")
            audio = delayed_rows(reference_codes, c.audio_pad_id, flush=True)
            prefix = torch.cat((audio, torch.full((len(audio), 1), c.text_vocab)), -1)
        if speaker_embedding is not None:
            if (
                speaker_embedding.device.type != "cpu"
                or speaker_embedding.shape != (c.speaker_embedding_dim,)
                or not bool(speaker_embedding.isfinite().all())
            ):
                raise ValueError("Invalid Zonos3 speaker embedding")
            prefix = torch.tensor([[c.audio_pad_id] * 10 + [c.text_vocab]])
        condition_start = len(prefix)
        frames = torch.cat((prefix, conditions, rows))
        if silence_frames:
            if c.codebook_size != 4096 or c.audio_codec != "opus_dual_3n_600k_sched600k_r4096_step600000":
                raise ValueError("Zonos3 silence prefill requires the matched release codec")
            audio = delayed_rows(torch.tensor(SILENCE_FRAME).repeat(silence_frames, 1), c.audio_pad_id, flush=False)
            frames = torch.cat((frames, torch.cat((audio, torch.full((silence_frames, 1), c.text_vocab)), -1)))
        if len(frames) >= c.max_seqlen:
            raise ValueError("Zonos3 conditioning leaves no context for generation")
        metadata = {
            "frame_ids": frames,
            "condition_start": condition_start,
            "condition_buckets": condition_ids(c, values, buckets),
            "has_audio_prefix": bool((frames[-1, :10] != c.audio_pad_id).any()),
            "silence_prefill_frames": silence_frames,
        }
        if speaker_embedding is not None:
            metadata.update(speaker_embedding=speaker_embedding, speaker_row=0)
        return {"prompt_token_ids": [0] * len(frames), "additional_information": {ZONOS3_PROMPT_KEY: metadata}}
