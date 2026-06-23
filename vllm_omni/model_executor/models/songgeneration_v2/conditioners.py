# Licensed under the License Terms of SongGeneration (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/tencent-ailab/SongGeneration/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. The License Terms of SongGeneration permit
# academic / research / educational use only and prohibit commercial use.
# ==============================================================================
#
# Copyright (C) 2025 Tencent. All rights reserved.
# Modifications Copyright contributors to the vLLM-Omni project.
#
# Vendored from tencent-ailab/SongGeneration (codeclm/modules/conditioners.py
# and codeclm/utils/utils.py), adapted for inference-only use in vLLM-Omni.
"""Conditioner provider for SongGeneration v2 LeLM Stage 0.

Vendored from upstream ``codeclm/modules/conditioners.py``. Inference-only
classes; weight layout matches the checkpoint (``audiolm.condition_provider.*``).
"""

from __future__ import annotations

import os
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# (input1 cond emb, input2 cond emb, mask) -- same triple the fuser consumes.
ConditionType = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


# ---------------------------------------------------------------------------
# Small helpers (vendored from codeclm/utils/utils.py)
# ---------------------------------------------------------------------------


def _length_to_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    """Length vector -> right-padded boolean mask.

    ``[3, 5] -> [[1,1,1,0,0], [1,1,1,1,1]]`` (cast to int by caller).
    """
    assert lengths.dim() == 1, lengths.shape
    final_length = lengths.max().item() if not max_len else max_len
    final_length = max(final_length, 1)
    return torch.arange(final_length, device=lengths.device)[None, :] < lengths[:, None]


# ---------------------------------------------------------------------------
# ConditioningAttributes + AudioCondition (input data containers)
# ---------------------------------------------------------------------------


class AudioCondition(NamedTuple):
    """Audio side data: ``wav`` here is a discrete-token tensor for v2."""

    wav: torch.Tensor
    length: torch.Tensor
    sample_rate: list[int]
    path: list[str | None] = []
    seek_time: list[float | None] = []


@dataclass
class ConditioningAttributes:
    text: dict[str, str | None] = field(default_factory=dict)
    audio: dict[str, AudioCondition] = field(default_factory=dict)

    def __getitem__(self, item: str):  # mimic upstream sample["text"] access
        return getattr(self, item)

    @property
    def text_attributes(self) -> list[str]:
        return list(self.text.keys())

    @property
    def audio_attributes(self) -> list[str]:
        return list(self.audio.keys())

    @property
    def attributes(self) -> dict[str, list[str]]:
        return {"text": self.text_attributes, "audio": self.audio_attributes}


# ---------------------------------------------------------------------------
# Conditioner base classes
# ---------------------------------------------------------------------------


class BaseConditioner(nn.Module):
    """Base conditioner: linear or embedding projection to ``output_dim``.

    The ``input_token`` flag selects between ``nn.Embedding`` (token-ID
    input, for text/codec conditioners) and ``nn.Linear`` (continuous
    feature input, unused at inference for v2).
    """

    def __init__(self, dim: int, output_dim: int, input_token: bool = False, padding_idx: int = 0):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        if input_token:
            self.output_proj = nn.Embedding(dim, output_dim, padding_idx=padding_idx)
        else:
            self.output_proj = nn.Linear(dim, output_dim)

    def tokenize(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, inputs):  # -> ConditionType
        raise NotImplementedError


class TextConditioner(BaseConditioner):
    pass


class AudioConditioner(BaseConditioner):
    pass


def _pad_2d(x: torch.Tensor, max_len: int, pad_id: int) -> torch.Tensor:
    """Right-pad or truncate a 2D ``[B, T]`` tensor to ``T == max_len``."""
    B, T = x.shape
    pad_len = max_len - T
    if pad_len > 0:
        pad = torch.full((B, pad_len), pad_id, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=1)
    if pad_len < 0:
        return x[:, :max_len]
    return x


# ---------------------------------------------------------------------------
# Lyric tokenizer + structure embedding
# ---------------------------------------------------------------------------


class QwTokenizerConditioner(TextConditioner):
    """Qwen2 BPE for lyrics, with per-token structure-span embeddings.

    On top of the base Qwen2 vocab the upstream config adds 13 structure
    tokens (``[verse]``, ``[chorus]``, ..., ``[silence]``) plus ``"."`` --
    see upstream ``codeclm/modules/conditioners.py``. At forward time each
    token is tagged with the structure span it falls inside (offset
    ``token_id - 151645``); the ``structure_emb`` lookup adds a per-span
    embedding to the content embedding.

    The Qwen2 tokenizer is loaded lazily on first ``tokenize()`` call --
    construction reads only the vocab size from a sidecar, so the module
    can be built (with the correct embedding shapes) before the tokenizer
    artefacts are on disk.
    """

    def __init__(
        self,
        output_dim: int,
        token_path: str = "",
        max_len: int = 600,
        add_token_list: list[str] | None = None,
        version: str = "v2",
        # vocab_size override lets us build the module on ``meta`` device
        # for shape-only checks without instantiating the tokenizer.
        vocab_size: int | None = None,
    ):
        if add_token_list is None:
            add_token_list = []
        add_token_list = list(add_token_list)
        if version != "v1":
            add_token_list.append(".")

        self._token_path = token_path
        self._add_token_list = add_token_list
        self._tokenizer: Any = None  # lazy
        self._struct_token_ids: list[int] = []

        if vocab_size is None:
            # Eager tokenizer load -- the upstream path.
            from transformers import Qwen2Tokenizer

            self._tokenizer = Qwen2Tokenizer.from_pretrained(token_path)
            if add_token_list:
                self._tokenizer.add_tokens(add_token_list, special_tokens=True)
            vocab_size = len(self._tokenizer.get_vocab())
            vocab = self._tokenizer.get_vocab()
            self._struct_token_ids = [vocab[t] for t in add_token_list if t and t[0] == "[" and t[-1] == "]"]

        super().__init__(vocab_size, output_dim, input_token=True, padding_idx=151643)
        self.max_len = max_len
        self.pad_token_idx = 151643

        # Per-structure embedding row count fixed at 200 to match the
        # upstream checkpoint (``structure_emb.weight`` is ``[200, dim]``).
        self.structure_emb = nn.Embedding(200, output_dim, padding_idx=0)

    def _ensure_tokenizer(self) -> None:
        if self._tokenizer is not None:
            return
        from transformers import Qwen2Tokenizer

        self._tokenizer = Qwen2Tokenizer.from_pretrained(self._token_path)
        if self._add_token_list:
            self._tokenizer.add_tokens(self._add_token_list, special_tokens=True)
        vocab = self._tokenizer.get_vocab()
        self._struct_token_ids = [vocab[t] for t in self._add_token_list if t and t[0] == "[" and t[-1] == "]"]

    def tokenize(self, x: list[str | None]) -> dict[str, torch.Tensor]:
        self._ensure_tokenizer()
        # The upstream code always prepends ``<|im_start|>`` (id 151644) --
        # see conditioners.py:139 -- even for None inputs (CFG null).
        x = ["<|im_start|>" + xi if xi is not None else "<|im_start|>" for xi in x]
        return self._tokenizer(x, return_tensors="pt", padding=True)

    def forward(self, inputs: dict[str, torch.Tensor]) -> ConditionType:
        mask = inputs["attention_mask"]
        tokens = inputs["input_ids"]

        # Compute structure-span coverage: for each batch element, find
        # positions that are structure tokens, then label every position
        # in that span with the offset ``struct_token_id - 151645`` (so
        # ``[verse]`` -> 1, ``[chorus]`` -> 2, etc., matching the upstream
        # ``structure_emb`` row indices).
        is_sp = torch.zeros_like(tokens, dtype=torch.bool)
        for sid in self._struct_token_ids:
            is_sp |= tokens == sid

        tp_cover_range = torch.zeros_like(tokens)
        for b in range(tokens.shape[0]):
            sp_list = torch.where(is_sp[b])[0].tolist()
            sp_list.append(int(mask[b].sum().item()))
            for i, st in enumerate(sp_list[:-1]):
                tp_cover_range[b, st : sp_list[i + 1]] = tokens[b, st] - 151645

        if self.max_len is not None:
            if tokens.shape[-1] > self.max_len:
                warnings.warn(f"QwTokenizerConditioner max_len={self.max_len} exceeded; truncating lyrics.")
            device = self.output_proj.weight.device
            tokens = _pad_2d(tokens, self.max_len, self.pad_token_idx).to(device)
            mask = _pad_2d(mask, self.max_len, 0).to(device)
            tp_cover_range = _pad_2d(tp_cover_range, self.max_len, 0).to(device)

        content = self.output_proj(tokens)
        structure = self.structure_emb(tp_cover_range)
        embeds = content + structure
        # Upstream returns (e1, e2, mask) where e1 == e2 for text conditioners
        # (the same prepend tensor goes into both transformers).
        return embeds, embeds, mask


# ---------------------------------------------------------------------------
# Style text tokenizer (no structure embedding)
# ---------------------------------------------------------------------------


class QwTextConditioner(TextConditioner):
    """Qwen2 BPE for the style ``descriptions`` field.

    v2 adds 6 Musicality tags (``[Musicality-very-high]`` etc.) + ``[Pure-Music]``
    + ``.`` -- see upstream ``conf/vocab.yaml``. No structure embedding.
    """

    _V2_EXTRA_TOKENS = [
        "[Musicality-very-high]",
        "[Musicality-high]",
        "[Musicality-medium]",
        "[Musicality-low]",
        "[Musicality-very-low]",
        "[Pure-Music]",
        ".",
    ]

    def __init__(
        self,
        output_dim: int,
        token_path: str = "",
        max_len: int = 100,
        version: str = "v2",
        vocab_size: int | None = None,
    ):
        self._token_path = token_path
        self._tokenizer: Any = None
        self._extra = list(self._V2_EXTRA_TOKENS) if version != "v1" else []

        if vocab_size is None:
            from transformers import Qwen2Tokenizer

            self._tokenizer = Qwen2Tokenizer.from_pretrained(token_path)
            if self._extra:
                self._tokenizer.add_tokens(self._extra, special_tokens=True)
            vocab_size = len(self._tokenizer.get_vocab())

        super().__init__(vocab_size, output_dim, input_token=True, padding_idx=151643)
        self.max_len = max_len

    def _ensure_tokenizer(self) -> None:
        if self._tokenizer is not None:
            return
        from transformers import Qwen2Tokenizer

        self._tokenizer = Qwen2Tokenizer.from_pretrained(self._token_path)
        if self._extra:
            self._tokenizer.add_tokens(self._extra, special_tokens=True)

    def tokenize(self, x: list[str | None]) -> dict[str, torch.Tensor]:
        self._ensure_tokenizer()
        x = ["<|im_start|>" + xi if xi is not None else "<|im_start|>" for xi in x]
        return self._tokenizer(x, return_tensors="pt", padding=True)

    def forward(self, inputs: dict[str, torch.Tensor]) -> ConditionType:
        mask = inputs["attention_mask"]
        tokens = inputs["input_ids"]

        if self.max_len is not None:
            if tokens.shape[-1] > self.max_len:
                warnings.warn(f"QwTextConditioner max_len={self.max_len} exceeded; truncating style.")
            device = self.output_proj.weight.device
            tokens = _pad_2d(tokens, self.max_len, 151643).to(device)
            mask = _pad_2d(mask, self.max_len, 0).to(device)

        embeds = self.output_proj(tokens)
        return embeds, embeds, mask


# ---------------------------------------------------------------------------
# Prompt audio (3-stream discrete codec) conditioner
# ---------------------------------------------------------------------------


class QuantizedEmbeddingConditioner(AudioConditioner):
    """Embeds a 3-stream prompt-audio token tensor for the LeLM prefill.

    Input is an ``AudioCondition`` whose ``wav`` field is a LongTensor of
    shape ``[B, code_depth, T_prompt]`` carrying discrete codec indices.
    Output is two parallel ``[B, max_len, dim]`` embeddings (one for each
    transformer in the dual-AR) plus a length mask. The two outputs differ
    only by a BOS row: ``EOT_emb`` for transformer 1, ``layer2_EOT_emb``
    for transformer 2.

    See upstream ``codeclm/modules/conditioners.py`` for wiring details.
    """

    def __init__(
        self,
        dim: int,
        code_size: int,
        code_depth: int,
        max_len: int,
        **kwargs: Any,  # accept upstream extras (e.g. ``output_dim``)
    ):
        super().__init__(dim, dim, input_token=True)
        self.code_depth = code_depth
        # ``code_size + 1`` is EOS; ``code_size + 2`` rows allows another
        # special slot. ``padding_idx = code_size + 1`` matches upstream.
        self.emb = nn.ModuleList(
            [nn.Embedding(code_size + 2, dim, padding_idx=code_size + 1) for _ in range(code_depth)]
        )
        # Free-floating BOS embeddings for the two transformer streams.
        self.EOT_emb = nn.Parameter(torch.randn(1, dim))
        self.layer2_EOT_emb = nn.Parameter(torch.randn(1, dim))
        # ``output_proj`` exists on the base class; the upstream sets it
        # to None for this conditioner (no extra projection).
        self.output_proj = None  # type: ignore[assignment]
        self.max_len = max_len
        self.vocab_size = code_size

    def tokenize(self, x: AudioCondition) -> AudioCondition:
        return x

    def forward(self, x: AudioCondition) -> ConditionType:
        wav, lengths, *_ = x
        B = wav.shape[0]
        wav = wav.reshape(B, self.code_depth, -1).long()
        if wav.shape[2] < self.max_len - 1:
            wav = F.pad(wav, [0, self.max_len - 1 - wav.shape[2]], value=self.vocab_size + 1)
        else:
            wav = wav[:, :, : self.max_len - 1]

        e1 = self.emb[0](wav[:, 0])
        e1 = torch.cat((self.EOT_emb.unsqueeze(0).repeat(B, 1, 1), e1), dim=1)
        e2 = sum(self.emb[k](wav[:, k]) for k in range(1, self.code_depth))
        e2 = torch.cat((self.layer2_EOT_emb.unsqueeze(0).repeat(B, 1, 1), e2), dim=1)

        lengths = lengths + 1
        lengths = torch.clamp(lengths, max=self.max_len)
        mask = _length_to_mask(lengths, max_len=e1.shape[1]).int()
        return e1, e2, mask


# ---------------------------------------------------------------------------
# ConditionerProvider (orchestrator)
# ---------------------------------------------------------------------------


class ConditionerProvider(nn.Module):
    """Hold one BaseConditioner per attribute, dispatch tokenize/forward.

    Matches the upstream ``ModuleDict`` layout so checkpoint keys land at
    ``conditioners.<name>.<param>`` under the parent prefix.
    """

    def __init__(self, conditioners: dict[str, BaseConditioner]):
        super().__init__()
        self.conditioners = nn.ModuleDict(conditioners)

    @property
    def text_conditions(self) -> list[str]:
        return [k for k, v in self.conditioners.items() if isinstance(v, TextConditioner)]

    @property
    def audio_conditions(self) -> list[str]:
        return [k for k, v in self.conditioners.items() if isinstance(v, AudioConditioner)]

    @property
    def has_audio_condition(self) -> bool:
        return len(self.audio_conditions) > 0

    def tokenize(self, inputs: list[ConditioningAttributes]) -> dict[str, Any]:
        assert all(isinstance(x, ConditioningAttributes) for x in inputs)
        output: dict[str, Any] = {}
        text = self._collate_text(inputs)
        audios = self._collate_audios(inputs)
        for attribute, batch in chain(text.items(), audios.items()):
            output[attribute] = self.conditioners[attribute].tokenize(batch)
        return output

    def forward(self, tokenized: dict[str, Any]) -> dict[str, ConditionType]:
        output: dict[str, ConditionType] = {}
        for attribute, inputs in tokenized.items():
            e1, e2, mask = self.conditioners[attribute](inputs)
            output[attribute] = (e1, e2, mask)
        return output

    def _collate_text(self, samples: list[ConditioningAttributes]) -> dict[str, list[str | None]]:
        out: dict[str, list[str | None]] = defaultdict(list)
        for s in samples:
            for cond in self.text_conditions:
                out[cond].append(s.text.get(cond))
        return out

    def _collate_audios(self, samples: list[ConditioningAttributes]) -> dict[str, AudioCondition]:
        # For inference we always pass a pre-batched ``AudioCondition``
        # (B==1 at the sample level); when there are multiple samples
        # (CFG pair, multi-request) we stack the ``wav`` tensors along
        # batch dim. The upstream collate path through
        # ``collate(wavs[attribute])`` flattens then pads, which is
        # heavyweight; for v2 inference each ``wav`` is already the same
        # length so a simple cat suffices.
        out: dict[str, AudioCondition] = {}
        if not self.audio_conditions:
            return out
        for attribute in self.audio_conditions:
            wavs, lengths, sample_rates, paths, seek_times = [], [], [], [], []
            for s in samples:
                if attribute not in s.audio:
                    continue
                ac = s.audio[attribute]
                wavs.append(ac.wav)
                lengths.append(ac.length)
                sample_rates.extend(ac.sample_rate)
                paths.extend(ac.path)
                seek_times.extend(ac.seek_time)
            if not wavs:
                continue
            # Each ``wav`` is ``[1, code_depth, T]``; stack along batch dim.
            stacked = torch.cat([w if w.dim() == 3 else w.unsqueeze(0) for w in wavs], dim=0)
            out[attribute] = AudioCondition(
                wav=stacked,
                length=torch.cat(lengths) if lengths else torch.empty(0, dtype=torch.long),
                sample_rate=sample_rates,
                path=paths,
                seek_time=seek_times,
            )
        return out


# ---------------------------------------------------------------------------
# Classifier-free guidance dropout (inference)
# ---------------------------------------------------------------------------


class ClassifierFreeGuidanceDropoutInference(nn.Module):
    """Build the null-conditioned batch half of a CFG-paired forward.

    For each attribute in each sample, replace text with ``None``
    (or a v2-specific tag swap, see below) and replace prompt-audio
    tokens with the special padding id ``code_size + 1``. Returns a
    deep-copied list of samples so the original conditioning stays intact.

    v2-specific text behaviour (upstream ``conditioners.py:644-647``):
    if the ``type_info`` text contains ``[Musicality-very-high]``, the null
    version becomes the literal string ``"[Musicality-very-low], ."``
    instead of plain ``None``. This is the only "soft" CFG case in v2.
    """

    def __init__(self, seed: int = 1234):
        super().__init__()
        self._seed = seed  # kept for parity; deterministic at p=1

    def _get_null_wav(self, wav: torch.Tensor, code_size: int) -> torch.Tensor:
        return torch.full_like(wav, code_size + 1)

    def _drop(self, sample: ConditioningAttributes, kind: str, name: str, *, code_size: int = 16384) -> None:
        """In-place null-out of ``sample.<kind>[name]``."""
        if kind == "audio":
            ac = sample.audio[name]
            null = self._get_null_wav(ac.wav, code_size=code_size)
            sample.audio[name] = AudioCondition(
                wav=null,
                length=ac.length,
                sample_rate=ac.sample_rate,
                path=ac.path,
                seek_time=ac.seek_time,
            )
        else:
            current = sample.text.get(name)
            if name == "type_info" and current is not None and "[Musicality-very-high]" in current:
                sample.text[name] = "[Musicality-very-low], ."
            else:
                sample.text[name] = None

    def forward(
        self,
        samples: list[ConditioningAttributes],
        condition_types: list[str] = ("audio", "text"),
        customized: list[str] | None = None,  # unused at inference for v2
        code_size: int = 16384,
    ) -> list[ConditioningAttributes]:
        new_samples = deepcopy(samples)
        for sample in new_samples:
            for kind in condition_types:
                for name in list(sample.attributes[kind]):
                    self._drop(sample, kind, name, code_size=code_size)
        return new_samples


# ---------------------------------------------------------------------------
# Structure-token vocab from the upstream conf/vocab.yaml.
# The Qwen2 tokenizer registers these as
# special tokens; the QwTokenizerConditioner uses the resulting IDs to
# tag per-token structure spans in the lyric embedding.
# ---------------------------------------------------------------------------
DEFAULT_STRUCTURE_VOCAB: tuple = (
    "[verse]",
    "[chorus]",
    "[bridge]",
    "[intro-short]",
    "[intro-medium]",
    "[intro-long]",
    "[outro-short]",
    "[outro-medium]",
    "[outro-long]",
    "[inst-short]",
    "[inst-medium]",
    "[inst-long]",
    "[silence]",
)


def _resolve_token_path_in_model_args(
    model_args: dict,
    upstream_repo_path: str | None,
) -> None:
    """Join relative ``token_path`` values against the SongGeneration repo root."""
    token_path = model_args.get("token_path")
    if (
        token_path is not None
        and upstream_repo_path is not None
        and isinstance(token_path, str)
        and not os.path.isabs(token_path)
    ):
        model_args["token_path"] = os.path.join(upstream_repo_path, token_path)


def build_conditioner_provider(
    config,
    *,
    structure_vocab: list[str] | None = None,
    upstream_repo_path: str | None = None,
    version: str = "v2",
) -> ConditionerProvider:
    """Construct :class:`ConditionerProvider` from ``config.conditioners``.

    Same branching as upstream ``codeclm/models/builders.py:get_conditioner_provider``:
    each entry selects ``qt_embedding``, ``QwTokenizer``, or ``QwTextTokenizer``.
    Model kwargs are taken from the nested config block and passed through
    (``**model_args``) so future YAML fields are not dropped. vLLM-Omni-only
    additions: resolve relative ``token_path`` via ``upstream_repo_path``, and
    default ``add_token_list`` from ``structure_vocab`` when the config omits it.
    """
    if structure_vocab is None:
        structure_vocab = list(DEFAULT_STRUCTURE_VOCAB)

    cond_specs = config.conditioners or {}
    conditioners: dict = {}
    dim = int(config.dim)

    for name, spec in cond_specs.items():
        model_type = spec.get("model")
        if model_type is None:
            raise ValueError(f"conditioner {name!r} missing 'model' key")
        sub = spec.get(model_type) or {}
        model_args = dict(sub)
        _resolve_token_path_in_model_args(model_args, upstream_repo_path)

        if model_type == "qt_embedding":
            conditioners[name] = QuantizedEmbeddingConditioner(dim=dim, **model_args)
        elif model_type == "QwTokenizer":
            model_args.setdefault("add_token_list", list(structure_vocab))
            conditioners[name] = QwTokenizerConditioner(
                output_dim=dim,
                version=version,
                **model_args,
            )
        elif model_type == "QwTextTokenizer":
            conditioners[name] = QwTextConditioner(
                output_dim=dim,
                version=version,
                **model_args,
            )
        else:
            raise ValueError(f"Unknown conditioner model_type {model_type!r} for attribute {name!r}")

    return ConditionerProvider(conditioners)


__all__ = [
    "AudioCondition",
    "ConditioningAttributes",
    "ConditionType",
    "BaseConditioner",
    "TextConditioner",
    "AudioConditioner",
    "QwTokenizerConditioner",
    "QwTextConditioner",
    "QuantizedEmbeddingConditioner",
    "ConditionerProvider",
    "ClassifierFreeGuidanceDropoutInference",
    "DEFAULT_STRUCTURE_VOCAB",
    "build_conditioner_provider",
]
