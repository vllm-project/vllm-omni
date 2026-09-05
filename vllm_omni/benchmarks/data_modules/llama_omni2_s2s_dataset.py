# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pybase64 as base64
from vllm.benchmarks.datasets import BenchmarkDataset, SampleRequest
from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.hf import get_cached_tokenizer


@dataclass(frozen=True)
class _LlamaOmni2S2SRow:
    sample_id: str
    audio_path: Path
    text: str


@dataclass
class LlamaOmni2S2SSampleRequest(SampleRequest):
    llama_omni2_chat_messages: list[dict[str, Any]] | None = None


class LlamaOmni2S2SDataset(BenchmarkDataset):
    DEFAULT_OUTPUT_LEN = 2048
    _REQUIRED_KEYS = frozenset({"id", "audio", "text"})

    def __init__(
        self,
        dataset_path: str,
        random_seed: int = 0,
        disable_shuffle: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            dataset_path=dataset_path,
            random_seed=random_seed,
            disable_shuffle=disable_shuffle,
            **kwargs,
        )
        path = Path(dataset_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"LLaMA-Omni2 S2S JSONL not found: {path}")
        self._rows = self._load_rows(path)
        if not disable_shuffle:
            random.Random(random_seed).shuffle(self._rows)
        self.data = self._rows

    @classmethod
    def _load_rows(cls, path: Path) -> list[_LlamaOmni2S2SRow]:
        rows: list[_LlamaOmni2S2SRow] = []
        seen_ids: set[str] = set()
        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not raw_line.strip():
                continue
            value = json.loads(raw_line)
            if not isinstance(value, dict) or set(value) != cls._REQUIRED_KEYS:
                raise ValueError(f"LLaMA-Omni2 S2S rows require exactly id, audio, and text (line {line_number})")
            sample_id = str(value["id"]).strip()
            text = str(value["text"]).strip()
            audio_path = Path(str(value["audio"])).expanduser()
            if not sample_id or not text:
                raise ValueError(f"LLaMA-Omni2 S2S id and text must be non-empty (line {line_number})")
            if sample_id in seen_ids:
                raise ValueError(f"duplicate LLaMA-Omni2 S2S id: {sample_id!r}")
            if not audio_path.is_absolute():
                raise ValueError(f"LLaMA-Omni2 S2S audio paths must be absolute (line {line_number})")
            if not audio_path.is_file():
                raise FileNotFoundError(f"LLaMA-Omni2 S2S audio file not found: {audio_path}")
            seen_ids.add(sample_id)
            rows.append(
                _LlamaOmni2S2SRow(
                    sample_id=sample_id,
                    audio_path=audio_path,
                    text=text,
                )
            )
        if not rows:
            raise ValueError(f"No valid LLaMA-Omni2 S2S rows in {path}")
        return rows

    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        output_len: int | None = None,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs: Any,
    ) -> list[SampleRequest]:
        del kwargs
        output_len = output_len or self.DEFAULT_OUTPUT_LEN
        tokenizer = get_cached_tokenizer(tokenizer)
        requests: list[SampleRequest] = []
        for row in self._rows[:num_requests]:
            audio_b64 = base64.b64encode(row.audio_path.read_bytes()).decode("ascii")
            requests.append(
                LlamaOmni2S2SSampleRequest(
                    prompt=row.text,
                    prompt_len=len(tokenizer.encode(row.text)),
                    expected_output_len=output_len,
                    multi_modal_data=None,
                    request_id=f"{request_id_prefix}{row.sample_id}",
                    llama_omni2_chat_messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "audio_url",
                                    "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"},
                                },
                                {"type": "text", "text": row.text},
                            ],
                        }
                    ],
                )
            )
        self.maybe_oversample_requests(
            requests,
            num_requests,
            request_id_prefix,
            no_oversample,
        )
        return requests
