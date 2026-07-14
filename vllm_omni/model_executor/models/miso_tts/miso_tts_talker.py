# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Miso TTS talker: one ``generate_frame()`` per AR scheduler step → RVQ codes."""

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.miso_tts.modeling_miso_tts import (
    DEFAULT_MISO_TTS_REPO_ID,
    MISO_NUM_CODEBOOKS,
    MisoTTSModel,
    load_mimi_codec,
    load_miso_model_weights,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)

_DEFAULT_TEMPERATURE = 0.9
_DEFAULT_TOPK = 50
_DEFAULT_MAX_FRAMES = 1125


def _pick(info: dict, key: str, default):
    val = info.get(key, default)
    if isinstance(val, (list, tuple)) and val:
        return val[0]
    return val if val is not None else default


@dataclass
class _Segment:
    speaker: int
    text: str
    audio: torch.Tensor


def _llama3_text_tokenizer():
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    bos, eos = tok.bos_token, tok.eos_token
    tok._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(bos, tok.bos_token_id), (eos, tok.eos_token_id)],
    )
    return tok


def _parse_context(raw: Any, device: torch.device) -> list[_Segment]:
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[_Segment] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        audio_field = item.get("audio")
        if audio_field is None:
            continue
        wav = audio_field[0] if isinstance(audio_field, (list, tuple)) else audio_field
        if not isinstance(wav, torch.Tensor):
            import numpy as np

            wav = torch.from_numpy(np.asarray(wav, dtype=np.float32).reshape(-1))
        out.append(
            _Segment(
                speaker=int(item.get("speaker", 0)),
                text=str(item.get("text", "") or ""),
                audio=wav.reshape(-1).float().to(device),
            )
        )
    return out


def _build_prompt(
    model: MisoTTSModel,
    text_tok: Any,
    mimi: Any,
    device: torch.device,
    text: str,
    speaker: int,
    context: list[_Segment],
    max_gen_frames: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fs = model.config.audio_num_codebooks + 1
    parts_t, parts_m = [], []

    def text_seg(t: str, sp: int) -> None:
        ids = text_tok.encode(f"[{sp}] {t.lstrip()}")
        fr = torch.zeros(len(ids), fs).long()
        mk = torch.zeros(len(ids), fs).bool()
        fr[:, -1] = torch.tensor(ids)
        mk[:, -1] = True
        parts_t.append(fr.to(device))
        parts_m.append(mk.to(device))

    def audio_seg(audio: torch.Tensor) -> None:
        codes = mimi.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        codes = torch.cat([codes, torch.zeros(codes.size(0), 1, device=device)], dim=1)
        fr = torch.zeros(codes.size(1), fs).long().to(device)
        mk = torch.zeros(codes.size(1), fs).bool().to(device)
        fr[:, :-1] = codes.transpose(0, 1)
        mk[:, :-1] = True
        parts_t.append(fr)
        parts_m.append(mk)

    for seg in context:
        text_seg(seg.text, seg.speaker)
        audio_seg(seg.audio)
    text_seg(text, speaker)

    prompt = torch.cat(parts_t, dim=0).long()
    mask = torch.cat(parts_m, dim=0).bool()
    if prompt.size(0) >= 2048 - max_gen_frames:
        raise ValueError("Miso prompt too long for max_seq_len - max_generation_frames")
    pos = torch.arange(prompt.size(0), device=device).unsqueeze(0).long()
    return prompt.unsqueeze(0), mask.unsqueeze(0), pos


@dataclass
class _Session:
    curr_tokens: torch.Tensor
    curr_tokens_mask: torch.Tensor
    curr_pos: torch.Tensor
    frames_left: int
    temperature: float
    topk: int
    done: bool = False


class MisoTTSTalkerForConditionalGeneration(nn.Module):
    requires_raw_input_tokens = True
    have_multimodal_outputs = True
    has_preprocess = False
    has_postprocess = False
    enable_update_additional_information = True
    inject_omni_request_id_into_runtime_info = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.model_path = vllm_config.model_config.model
        self._model: MisoTTSModel | None = None
        self._text_tok: Any = None
        self._mimi_encode: Any = None
        self._device: torch.device | None = None
        self._lock = threading.Lock()
        self._sessions: dict[str, _Session] = {}
        self._ar_last_chunk_flags: list[bool] = []

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        with self._lock:
            if self._model is not None:
                return None
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._device = device
            dtype = (
                torch.bfloat16
                if device.type == "cuda" and torch.cuda.is_bf16_supported()
                else torch.float16
                if device.type == "cuda"
                else torch.float32
            )
            path = self.model_path or DEFAULT_MISO_TTS_REPO_ID
            self._model = load_miso_model_weights(path, device, dtype)
            # Use cache size of 1 like official implementation, and pass dtype
            self._model.setup_caches(1, dtype)
            # Explicitly cast embeddings to correct dtype to avoid RMS norm mismatch
            self._model.text_embeddings.to(dtype=dtype)
            self._model.audio_embeddings.to(dtype=dtype)
            # Use cache size of 1 like official implementation, and pass dtype
            self._model.setup_caches(1, dtype)
            # Explicitly cast embeddings to correct dtype to avoid RMS norm mismatch
            self._model.text_embeddings.to(dtype=dtype)
            self._model.audio_embeddings.to(dtype=dtype)
            self._text_tok = _llama3_text_tokenizer()
            self._mimi_encode = load_mimi_codec(device, self._model.config.audio_num_codebooks)
        for _ in weights:
            pass
        return None

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict]:
        return [{"text": "hello", "speaker": 0, "_is_dummy": True}] * num_reqs

    def _session(self, key: str, info: dict[str, Any]) -> _Session | None:
        if key in self._sessions:
            return self._sessions[key]
        model, tok, mimi = self._model, self._text_tok, self._mimi_encode
        device = self._device or torch.device("cpu")
        if model is None or tok is None or mimi is None:
            return None
        text = str(_pick(info, "text", "") or "").strip()
        if not text:
            return None
        ctx = _pick(info, "context", None)
        if ctx is not None and not isinstance(ctx, list):
            ctx = [ctx]
        max_f = max(1, int(_pick(info, "max_generation_frames", _DEFAULT_MAX_FRAMES)))
        # Only reset caches when creating a new session
        model.reset_caches()
        t, m, p = _build_prompt(
            model,
            tok,
            mimi,
            device,
            text,
            int(_pick(info, "speaker", 0)),
            _parse_context(ctx, device),
            max_f,
        )
        s = _Session(
            curr_tokens=t,
            curr_tokens_mask=m,
            curr_pos=p,
            frames_left=max_f,
            temperature=float(_pick(info, "temperature", _DEFAULT_TEMPERATURE)),
            topk=int(_pick(info, "topk", _DEFAULT_TOPK)),
        )
        self._sessions[key] = s
        return s

    def _step(self, s: _Session) -> tuple[torch.Tensor, bool]:
        model = self._model
        if model is None or s.done or s.frames_left <= 0:
            return torch.zeros(MISO_NUM_CODEBOOKS, dtype=torch.long), True
        frame = model.generate_frame(s.curr_tokens, s.curr_tokens_mask, s.curr_pos, s.temperature, s.topk)
        s.frames_left -= 1
        is_zero_frame = bool((frame == 0).all())
        
        # Match official behavior: break immediately on zero frame (EOS)
        # Don't update state after zero frame to prevent garbage generation
        if is_zero_frame:
            s.done = True
            return frame.reshape(-1).long(), True
        
        dev = frame.device
        nc = model.config.audio_num_codebooks
        # Match official implementation exactly
        s.curr_tokens = torch.cat([frame, torch.zeros(1, 1).long().to(dev)], dim=1).unsqueeze(1)
        s.curr_tokens_mask = torch.cat(
            [torch.ones_like(frame).bool(), torch.zeros(1, 1).bool().to(dev)],
            dim=1,
        ).unsqueeze(1)
        s.curr_pos = s.curr_pos[:, -1:] + 1
        s.done = s.frames_left <= 0
        return frame.reshape(-1).long(), s.done

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        n = 1 if input_ids is None else max(1, input_ids.shape[0])
        infos = runtime_additional_information or [{}]
        is_dummy_only = not runtime_additional_information or all(i.get("_is_dummy") for i in infos)
        if self._device is None:
            try:
                self._device = next(self.parameters()).device
            except StopIteration:
                self._device = torch.device("cpu")
        hidden = torch.zeros(
            n,
            int(getattr(self.config, "hidden_size", 4096)),
            device=self._device,
        )
        if is_dummy_only:
            self._ar_last_chunk_flags = [True] * len(infos)
            z = torch.zeros(MISO_NUM_CODEBOOKS, dtype=torch.long)
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs={"codes": {"audio": [z] * len(infos)}})

        if self._model is None:
            self.load_weights([])
        codes, flags = [], []
        for info in infos:
            if info.get("_is_dummy"):
                codes.append(torch.zeros(MISO_NUM_CODEBOOKS, dtype=torch.long))
                flags.append(True)
                continue
            key = str(info.get("global_request_id") or info.get("_omni_req_id") or id(info))
            sess = self._session(key, info)
            if sess is None:
                codes.append(torch.zeros(MISO_NUM_CODEBOOKS, dtype=torch.long))
                flags.append(True)
                continue
            fr, done = self._step(sess)
            codes.append(fr.cpu())
            flags.append(done)
            # Don't pop session here - let connector signal when truly finished via runtime info
        self._ar_last_chunk_flags = flags
        # Include done flag in multimodal_output so connector knows when to finish
        return OmniOutput(text_hidden_states=hidden, multimodal_outputs={"codes": {"audio": codes}, "done": flags})

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        for rid in finished_req_ids:
            self._sessions.pop(str(rid), None)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> torch.Tensor:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            hidden_states = torch.zeros((0, 1), device=self._device or "cpu")
        if hidden_states.ndim > 2:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        vocab = int(getattr(self.config, "vocab_size", 128256))
        logits = torch.zeros((hidden_states.shape[0], vocab), device=hidden_states.device)
        eos = 2 if vocab > 2 else 0
        safe = 1 if vocab > 1 and 1 != eos else 0
        flags = self._ar_last_chunk_flags
        for row in range(int(logits.shape[0])):
            is_last = flags[row] if row < len(flags) else True
            if is_last:
                logits[row, eos] = 1e6
            else:
                logits[row, eos] = -1e9
                logits[row, safe] = 1e6
        return logits

    def embed_input_ids(self, input_ids: torch.Tensor, multimodal_embeddings=None, is_multimodal=None) -> torch.Tensor:
        h = int(getattr(self.config, "hidden_size", 4096))
        return torch.zeros((input_ids.shape[0], h), device=input_ids.device)
