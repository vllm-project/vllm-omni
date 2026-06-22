# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, field

from vllm_omni.experimental.fullduplex.joyvl.decision.prompts import SYSTEM_PROMPTS


@dataclass
class SamplingConfig:
    max_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1
    presence_penalty: float = 1.5


@dataclass
class InteractionConfig:
    main_backend_url: str = "http://127.0.0.1:8061/v1"
    main_model: str = "JoyAI-VL-Interaction-Preview"
    api_key: str = "EMPTY"

    persona: str = "default"
    frame_seconds: float = 1.0
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    force_silence_before_query: bool = True

    response_dedup_threshold: float = 1.0

    enable_memory: bool = True

    summarizer_backend_url: str | None = None
    summarizer_model: str | None = None

    chunk_frames: int = 100

    mid_term_key_frames: int = 0

    long_term_every_n_chunks: int = 5
    mid_term_max_tokens: int = 4000
    long_term_max_tokens: int = 4000
    keep_qa_history: bool = True

    enable_delegation: bool = True
    delegation_backend_url: str | None = None
    delegation_model: str | None = None
    delegation_api_key: str | None = None
    delegation_max_tokens: int = 512
    delegation_kind: str = "chat"
    delegation_image_url: str | None = None
    delegation_edit_url: str | None = None
    delegation_edit_model: str | None = None

    session_timeout_seconds: float = 3600.0
    request_timeout_seconds: float = 300.0

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPTS.get(self.persona, SYSTEM_PROMPTS["default"])

    @property
    def resolved_summarizer_url(self) -> str:
        return self.summarizer_backend_url or self.main_backend_url

    @property
    def resolved_summarizer_model(self) -> str:
        return self.summarizer_model or self.main_model

    @property
    def resolved_delegation_model(self) -> str:
        return self.delegation_model or self.main_model

    @property
    def resolved_delegation_api_key(self) -> str:
        return self.delegation_api_key or self.api_key
