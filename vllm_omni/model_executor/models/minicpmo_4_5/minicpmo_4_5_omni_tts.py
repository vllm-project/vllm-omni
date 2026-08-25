# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from:
# https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/main/modeling_minicpmo.py
"""MiniCPM-o 4.5 native autoregressive Talker.

Pipeline:
  1. Receive thinker hidden_states + full token IDs via additional_information
  2. Extract tts_bos..tts_eos region
  3. Build condition: emb_text(tokens) + projector_semantic(hidden) (hidden_text_merge)
  4. Continuously generate request-aligned discrete audio-code deltas
"""

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.models.utils import maybe_prefix
from vllm.v1.sample.sampler import Sampler

from vllm_omni.experimental.fullduplex.engine.intermediate import get_tts_handoff
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

_REPETITION_WINDOW = 16
_REPETITION_PENALTY_CHUNK_SIZE = 16
_MIN_AUDIO_TOKENS = 64
_MAX_AUDIO_TOKENS = 2048
_AUDIO_TOKENS_PER_TEXT_TOKEN = 10
# Codec-token sampling happens inside the model; vLLM sampling parameters
# only choose the Talker's binary continue/stop row.
_CODEC_SEED = 42
_CODEC_TEMPERATURE = 0.8
_CODEC_TOP_K = 25
_CODEC_TOP_P = 0.85
_CODEC_REPETITION_PENALTY = 1.05
_CODEC_MIN_TOKENS = 50
_DUPLEX_CODEC_TOKENS_PER_CHUNK = 26
_BATCHED_CODEC_SAMPLING_ENV = "VLLM_OMNI_MINICPMO45_BATCHED_CODEC_SAMPLING"


def _batched_codec_sampling_enabled() -> bool:
    raw_value = os.environ.get(_BATCHED_CODEC_SAMPLING_ENV)
    if raw_value is None:
        return True
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Invalid {_BATCHED_CODEC_SAMPLING_ENV}={raw_value!r}; "
        "expected one of 1/0, true/false, yes/no, or on/off"
    )


@dataclass(slots=True)
class _PendingCodecSample:
    output_index: int
    hidden_row: torch.Tensor
    codes: torch.Tensor
    request_id: str
    step: int
    state: dict[str, Any]
    info: dict[str, Any]


def _max_audio_tokens(condition_tokens: int) -> int:
    """Bound codec generation with a conservative text-length estimate.

    EOS is masked for the first 50 steps, so a direct ``text_tokens * 10``
    limit can terminate short responses before EOS is eligible. The 2048
    ceiling matches the checkpoint's native generation default and keeps the
    sequence within the Talker's 4096-position context.
    """
    return max(
        _MIN_AUDIO_TOKENS,
        min(_MAX_AUDIO_TOKENS, condition_tokens * _AUDIO_TOKENS_PER_TEXT_TOKEN),
    )


def _restore_weight_norm_weight(weight_g: torch.Tensor, weight_v: torch.Tensor) -> torch.Tensor:
    """Materialize ``weight_norm(..., dim=0)`` checkpoint parameters."""
    return torch._weight_norm(weight_v, weight_g, dim=0)


def _apply_batched_repetition_penalty(
    logits: torch.Tensor,
    histories: Sequence[torch.Tensor],
    *,
    penalty: float,
    window_size: int,
) -> torch.Tensor:
    """Apply request-local frequency penalties to a batch of codec logits."""
    if logits.ndim != 2:
        raise ValueError(f"batched codec logits must be 2D, got shape {tuple(logits.shape)}")
    batch_size, vocab_size = logits.shape
    if len(histories) != batch_size:
        raise ValueError(f"expected {batch_size} codec histories, got {len(histories)}")
    if penalty == 1.0:
        return logits

    if batch_size == 0:
        return logits

    penalized = logits.clone()
    penalty_tensor = torch.as_tensor(penalty, device=logits.device, dtype=logits.dtype)
    for start in range(0, batch_size, _REPETITION_PENALTY_CHUNK_SIZE):
        end = min(start + _REPETITION_PENALTY_CHUNK_SIZE, batch_size)
        chunk_logits = logits[start:end]
        encoded_rows: list[torch.Tensor] = []
        for local_row, history in enumerate(histories[start:end]):
            recent = history.reshape(-1)[-window_size:].to(device=logits.device, dtype=torch.long)
            if recent.numel() > 0:
                encoded_rows.append(recent + local_row * vocab_size)
        if not encoded_rows:
            continue

        # Bound the int64 bincount workspace independently of concurrency.
        encoded = encoded_rows[0] if len(encoded_rows) == 1 else torch.cat(encoded_rows)
        frequencies = torch.bincount(
            encoded,
            minlength=(end - start) * vocab_size,
        ).reshape(end - start, vocab_size)
        alpha = torch.pow(penalty_tensor, frequencies.to(dtype=logits.dtype))
        penalized[start:end] = torch.where(
            chunk_logits < 0,
            chunk_logits * alpha,
            chunk_logits / alpha,
        )

    return penalized


def _apply_top_k_top_p(
    logits: torch.Tensor,
    *,
    top_k: int | None,
    top_p: float | None,
    min_tokens_to_keep: int = 3,
) -> torch.Tensor:
    """Apply the same candidate floors as the upstream Transformers warpers."""
    filtered = logits.clone()
    vocab_size = filtered.shape[-1]
    # MiniCPM-o's gen_logits() appends TopPLogitsWarper before
    # TopKLogitsWarper. The order is observable for fixed-seed sampling.
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=False, dim=-1)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative_probs <= (1.0 - float(top_p))
        remove[..., -min_tokens_to_keep:] = False
        remove = remove.scatter(-1, sorted_indices, remove)
        filtered.masked_fill_(remove, float("-inf"))
    if top_k is not None and top_k > 0:
        keep = min(vocab_size, max(int(top_k), min_tokens_to_keep))
        threshold = torch.topk(filtered, keep, dim=-1).values[..., -1, None]
        filtered.masked_fill_(filtered < threshold, float("-inf"))
    return filtered


class _MiniCPMTTSProjector(nn.Module):
    """Checkpoint-compatible hidden-state projector used by MiniCPMTTS."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(hidden_states)))


class MiniCPMO45OmniTTSForConditionalGeneration(nn.Module, SupportsPP):
    """Runner-owned MiniCPM-o 4.5 Talker that emits codec tokens only."""

    requires_request_sample_eligibility = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import MiniCPMOConfig

        config: MiniCPMOConfig = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        self._batch_stop_logits: torch.Tensor | None = None
        self._request_generators: dict[str, torch.Generator] = {}
        self._request_audio_states: dict[str, dict[str, Any]] = {}
        self._deferred_cleanup_ids: set[str] = set()
        self._sampler = Sampler()
        self._batched_codec_sampling = _batched_codec_sampling_enabled()

        tts_config = getattr(config, "tts_config", None)
        if tts_config is None and getattr(config, "model_type", None) == "minicpmtts":
            tts_config = config
        if tts_config is not None:
            self._tts_config = tts_config
            self._tts_bos_id = getattr(tts_config, "audio_bos_token_id", 151687)
            self._text_eos_id = getattr(tts_config, "text_eos_token_id", 151692)
            self._num_audio_tokens = getattr(tts_config, "num_audio_tokens", 6562)
            self._hidden_size = getattr(tts_config, "hidden_size", 768)
            self._normalize = getattr(tts_config, "normalize_projected_hidden", True)
            self._codec_seed = int(getattr(tts_config, "seed", _CODEC_SEED))
            self._codec_temperature = float(getattr(tts_config, "temperature", _CODEC_TEMPERATURE))
            self._codec_top_k = int(getattr(tts_config, "top_k", _CODEC_TOP_K))
            self._codec_top_p = float(getattr(tts_config, "top_p", _CODEC_TOP_P))
            self._codec_repetition_penalty = float(getattr(tts_config, "repetition_penalty", _CODEC_REPETITION_PENALTY))
            self._codec_min_tokens = int(getattr(tts_config, "min_new_tokens", _CODEC_MIN_TOKENS))
        else:
            self._tts_config = None

        self.has_preprocess = True
        self.has_postprocess = False
        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {
            ("audio_codes", "current"),
            ("audio_codes", "accumulated"),
        }
        self._init_native_talker(prefix)

    def _init_native_talker(self, prefix: str) -> None:
        if self._tts_config is None:
            raise ValueError("MiniCPM-o continuous Talker requires tts_config")
        cfg = self._tts_config
        if int(getattr(cfg, "num_vq", 1)) != 1:
            raise ValueError(
                "MiniCPM-o continuous Talker currently requires num_vq=1; "
                f"checkpoint reports {getattr(cfg, 'num_vq', None)}"
            )
        llama_config = LlamaConfig(
            vocab_size=32000,
            hidden_size=int(cfg.hidden_size),
            intermediate_size=int(cfg.intermediate_size),
            num_hidden_layers=int(cfg.num_hidden_layers),
            num_attention_heads=int(cfg.num_attention_heads),
            num_key_value_heads=int(cfg.num_key_value_heads),
            hidden_act=getattr(cfg, "hidden_act", "silu"),
            max_position_embeddings=int(cfg.max_position_embeddings),
            rms_norm_eps=float(getattr(cfg, "rms_norm_eps", 1e-6)),
            tie_word_embeddings=False,
        )
        talker_config = self.vllm_config.with_hf_config(llama_config, architectures=["LlamaForCausalLM"])
        talker_config.model_config.hf_text_config = llama_config
        self.tts_model = LlamaModel(
            vllm_config=talker_config,
            prefix=maybe_prefix(prefix, "tts_obj.model"),
        )
        self.emb_text = nn.Embedding(int(cfg.num_text_tokens), int(cfg.hidden_size))
        self.projector_semantic = _MiniCPMTTSProjector(int(cfg.llm_dim), int(cfg.hidden_size))
        self.emb_code = nn.ModuleList(
            [nn.Embedding(int(cfg.num_audio_tokens), int(cfg.hidden_size)) for _ in range(int(cfg.num_vq))]
        )
        self.head_code = nn.ModuleList(
            [nn.Linear(int(cfg.hidden_size), int(cfg.num_audio_tokens), bias=False) for _ in range(int(cfg.num_vq))]
        )
        self.make_empty_intermediate_tensors = self.tts_model.make_empty_intermediate_tensors

    def _boundary_embeddings(self) -> torch.Tensor:
        """Embed the ``<text_eos><audio_bos>`` tail every condition ends with."""
        ids = torch.tensor(
            [self._text_eos_id, self._tts_bos_id],
            device=self.emb_text.weight.device,
            dtype=torch.long,
        )
        return self.emb_text(ids)

    def _build_condition_embeddings(
        self,
        tts_token_ids: torch.Tensor,
        tts_hidden_states: torch.Tensor,
        *,
        native_duplex: bool = False,
    ) -> torch.Tensor:
        if tts_token_ids.numel() == 0 or tts_hidden_states.numel() == 0:
            # The thinker can legally emit an empty speech segment (<|tts_bos|>
            # immediately followed by a boundary token) when it decides not to
            # speak. Condition on the boundary tokens alone, which matches the
            # 2-token scheduler prompt the stage bridge builds for an empty
            # handoff.
            return self._boundary_embeddings()
        device = self.emb_text.weight.device
        dtype = self.emb_text.weight.dtype
        token_ids = tts_token_ids.to(device=device, dtype=torch.long).reshape(-1)
        hidden = tts_hidden_states.to(device=device, dtype=dtype)
        if hidden.shape[0] != token_ids.shape[0] and token_ids.shape[0] != 1:
            raise ValueError(
                "MiniCPM-o Talker condition length mismatch: "
                f"token_ids={token_ids.shape[0]} hidden_states={hidden.shape[0]}"
            )
        text_embeds = self.emb_text(token_ids)
        hidden_embeds = self.projector_semantic(hidden)
        if self._normalize:
            hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)
        audio_bos = self.emb_text(torch.tensor([self._tts_bos_id], device=device, dtype=torch.long))
        condition = text_embeds + hidden_embeds
        if native_duplex:
            # Match MiniCPMTTS.generate_chunk's streaming condition.
            return torch.cat([condition, audio_bos], dim=0)
        return torch.cat([condition, self._boundary_embeddings()], dim=0)

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Build request-local prefill/decode embeddings for the vLLM runner."""
        del input_embeds
        span_len = int(input_ids.shape[0])
        is_prefill = bool(info_dict.get("_omni_is_prefill", False))
        state = info_dict.get("audio_state")
        first_call = not isinstance(state, dict)

        if is_prefill or first_call:
            token_ids, hidden_states = get_tts_handoff(info_dict)
            # Accept the structured tensor path and the rollback-compatible
            # legacy list path before building the Talker condition.
            if isinstance(token_ids, (list, tuple)):
                token_ids = torch.as_tensor(token_ids, dtype=torch.long)
            if isinstance(hidden_states, (list, tuple)):
                hidden_states = torch.as_tensor(hidden_states, dtype=torch.float32)
            if not isinstance(token_ids, torch.Tensor) or not isinstance(hidden_states, torch.Tensor):
                available = sorted(key for key in info_dict if not key.startswith("_"))
                raise ValueError(
                    "MiniCPM-o Talker requires tensor tts_token_ids and "
                    "tts_hidden_states conditioning; "
                    f"received token_ids={type(token_ids).__name__}, "
                    f"hidden_states={type(hidden_states).__name__}, "
                    f"available_keys={available}"
                )
            # An empty condition means the thinker chose not to speak: finish the
            # request up front so it emits zero audio codes instead of killing
            # the stage engine.
            empty_condition = token_ids.numel() == 0 or hidden_states.numel() == 0
            if empty_condition:
                logger.warning_once(
                    "MiniCPM-o Talker received an empty condition (request %s); this request produces no audio.",
                    info_dict.get("request_id"),
                )
            native_duplex = bool(info_dict.get("native_duplex", False))
            full_embeds = self._build_condition_embeddings(
                token_ids,
                hidden_states,
                native_duplex=native_duplex,
            )
            offset = int(info_dict.get("_omni_num_computed_tokens", 0))
            request_id = str(info_dict.get("request_id", "0"))
            meta = info_dict.get("meta")
            # The handoff rebuilds only the tail-aligned Talker condition.
            # Materialize zero-token embeddings for any scheduler prompt
            # prefix so chunked prefill can slice from a non-zero offset.
            prompt_len = info_dict.get("_omni_prompt_len")
            target_len = int(prompt_len) if prompt_len is not None else offset + span_len
            prefix_len = target_len - full_embeds.shape[0]
            if prefix_len > 0:
                placeholder_ids = torch.zeros(
                    prefix_len,
                    dtype=torch.long,
                    device=self.emb_text.weight.device,
                )
                full_embeds = torch.cat([self.emb_text(placeholder_ids), full_embeds], dim=0)
            embeds = full_embeds[offset : offset + span_len]
            if embeds.shape[0] != span_len:
                raise ValueError(
                    "MiniCPM-o Talker prefill span exceeds condition: "
                    f"request_id={info_dict.get('request_id')} offset={offset} "
                    f"span={span_len} condition={full_embeds.shape[0]} "
                    f"tts_ids={token_ids.shape[0]} tts_hidden={hidden_states.shape[0]} "
                    f"prompt_len={info_dict.get('_omni_prompt_len')}"
                )
            duplex_boundary = isinstance(meta, dict) and (
                bool(meta.get("turn_start", False)) or bool(meta.get("turn_end", False))
            )
            if native_duplex:
                max_tokens = _DUPLEX_CODEC_TOKENS_PER_CHUNK
                min_tokens = 0 if duplex_boundary else _DUPLEX_CODEC_TOKENS_PER_CHUNK
            else:
                max_tokens = _max_audio_tokens(int(token_ids.numel()))
                min_tokens = self._codec_min_tokens
            state = {
                "step": 0,
                "max_tokens": max_tokens,
                "min_tokens": min_tokens,
                "finished": empty_condition,
            }
            request_states = getattr(self, "_request_audio_states", None)
            if request_states is None:
                request_states = {}
                self._request_audio_states = request_states
            request_states[request_id] = state
            empty_codes = torch.empty(0, dtype=torch.long, device=embeds.device)
            return (
                input_ids,
                embeds,
                {
                    "audio_state": state,
                    "audio_codes": {
                        "current": empty_codes,
                        "accumulated": empty_codes,
                    },
                },
            )

        current = (info_dict.get("audio_codes", {}) or {}).get("current")
        if not isinstance(current, torch.Tensor) or current.numel() != 1:
            if state.get("finished"):
                # A request that finished before sampling any code can still be
                # scheduled for decode steps while sampling min_tokens masks the
                # stop token. make_omni_output ignores its hidden states, so any
                # shape-correct embedding will do.
                weight = self.emb_code[0].weight
                return input_ids, weight.new_zeros((span_len, weight.shape[1])), {}
            raise RuntimeError("MiniCPM-o Talker decode is missing the previous request-local audio code")
        code = current.to(device=self.emb_code[0].weight.device, dtype=torch.long).reshape(1)
        embeds = self.emb_code[0](code)
        return input_ids, embeds, {}

    def _request_generator(self, request_id: str, device: torch.device) -> torch.Generator:
        generator = self._request_generators.get(request_id)
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self._codec_seed)
            self._request_generators[request_id] = generator
        return generator

    def _sample_audio_codes(
        self,
        hidden_states: torch.Tensor,
        histories: Sequence[torch.Tensor],
        request_ids: Sequence[str],
        steps: Sequence[int],
    ) -> torch.Tensor:
        batch_size = int(hidden_states.shape[0])
        if not (len(histories) == len(request_ids) == len(steps) == batch_size):
            raise ValueError(
                "MiniCPM-o batched codec sampling requires one history, request id, "
                f"and step per hidden row, got batch={batch_size}, "
                f"histories={len(histories)}, request_ids={len(request_ids)}, "
                f"steps={len(steps)}"
            )
        if batch_size == 0:
            return torch.empty(0, dtype=torch.long, device=hidden_states.device)

        logits = self.head_code[0](hidden_states).float() / self._codec_temperature
        eos_id = self._num_audio_tokens - 1
        logits = _apply_batched_repetition_penalty(
            logits,
            histories,
            penalty=self._codec_repetition_penalty,
            window_size=_REPETITION_WINDOW,
        )
        request_states = getattr(self, "_request_audio_states", {})
        mask_eos_values: list[bool] = []
        for request_id, step in zip(request_ids, steps, strict=True):
            state = request_states.get(request_id)
            min_tokens = (
                int(state.get("min_tokens", self._codec_min_tokens))
                if isinstance(state, dict)
                else self._codec_min_tokens
            )
            mask_eos_values.append(step < min_tokens)
        mask_eos = torch.tensor(
            mask_eos_values,
            dtype=torch.bool,
            device=logits.device,
        )
        logits[:, eos_id].masked_fill_(mask_eos, float("-inf"))
        logits = _apply_top_k_top_p(
            logits,
            top_k=self._codec_top_k,
            top_p=self._codec_top_p,
            min_tokens_to_keep=3,
        )
        probabilities = torch.softmax(logits, dim=-1)
        # torch.multinomial accepts one generator for the entire call. Keep it
        # per row so compaction, request ordering, or another request finishing
        # cannot perturb request-local RNG streams; the expensive projection
        # and logit transforms above remain fully batched.
        return torch.cat(
            [
                torch.multinomial(
                    probabilities[row : row + 1],
                    num_samples=1,
                    generator=self._request_generator(request_id, probabilities.device),
                )
                for row, request_id in enumerate(request_ids)
            ],
            dim=0,
        ).reshape(-1)

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | OmniOutput,
        **kwargs: Any,
    ) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        hidden = model_outputs
        infos = kwargs.get("model_intermediate_buffer") or []
        spans = kwargs.get("request_token_spans")
        if spans is None or len(spans) != len(infos):
            raise RuntimeError("MiniCPM-o continuous Talker requires one request_token_span per request")
        sample_eligible = kwargs.get("request_sample_eligible")
        if sample_eligible is None:
            sample_eligible = [True] * len(infos)
        if len(sample_eligible) != len(infos):
            raise RuntimeError(
                f"MiniCPM-o continuous Talker received {len(sample_eligible)} sampling flags for {len(infos)} requests"
            )
        emit_duplex_metadata = any(isinstance(info, dict) and info.get("native_duplex") is True for info in infos)

        # Rows default to continue. Only previously finished or newly terminal
        # requests stop; prefill/ineligible rows stay aligned as False.
        stop_flags = [False] * len(infos)
        native_duplex_flags: list[torch.Tensor] = []
        duplex_epochs: list[torch.Tensor] = []
        duplex_turn_ids: list[torch.Tensor] = []
        segment_texts_utf8: list[torch.Tensor] = []
        turn_end_flags: list[torch.Tensor] = []
        empty_delta = hidden.new_empty((0, 1), dtype=torch.long)
        codec_deltas = [empty_delta for _ in infos]
        terminal_flags = [torch.tensor(False, dtype=torch.bool) for _ in infos]
        pending_samples: list[_PendingCodecSample] = []
        for index, info in enumerate(infos):
            info_dict = info if isinstance(info, dict) else {}
            native_duplex = info_dict.get("native_duplex") is True
            if emit_duplex_metadata:
                duplex_info = info_dict.get("duplex")
                if not isinstance(duplex_info, dict):
                    duplex_info = {}
                epoch = duplex_info.get("epoch", -1)
                turn_id = duplex_info.get("turn_id", -1)
                if native_duplex and not all(
                    isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in (epoch, turn_id)
                ):
                    raise RuntimeError(
                        "MiniCPM-o native duplex Talker requires non-negative integer "
                        f"epoch and turn_id, got epoch={epoch!r}, turn_id={turn_id!r}"
                    )
                meta_info = info_dict.get("meta")
                if not isinstance(meta_info, dict):
                    meta_info = {}
                segment_text = meta_info.get("native_duplex_segment_text", "") if native_duplex else ""
                if not isinstance(segment_text, str):
                    segment_text = ""
                turn_eos_id = meta_info.get("turn_eos_token_id")
                ids_info = info_dict.get("ids")
                tts_ids = ids_info.get("tts") if native_duplex and isinstance(ids_info, dict) else None
                if isinstance(tts_ids, torch.Tensor):
                    contains_turn_eos = isinstance(turn_eos_id, int) and bool(
                        torch.any(tts_ids.reshape(-1) == turn_eos_id).item()
                    )
                elif isinstance(tts_ids, (list, tuple)):
                    contains_turn_eos = isinstance(turn_eos_id, int) and turn_eos_id in tts_ids
                else:
                    contains_turn_eos = False
                native_duplex_flags.append(torch.tensor(native_duplex, dtype=torch.bool))
                duplex_epochs.append(torch.tensor(epoch if isinstance(epoch, int) else -1, dtype=torch.long))
                duplex_turn_ids.append(torch.tensor(turn_id if isinstance(turn_id, int) else -1, dtype=torch.long))
                segment_texts_utf8.append(
                    torch.tensor(
                        list(segment_text.encode("utf-8")),
                        dtype=torch.uint8,
                    )
                )
                turn_end_flags.append(torch.tensor(native_duplex and contains_turn_eos, dtype=torch.bool))

            if not isinstance(info, dict):
                continue
            start, end = spans[index]
            end = min(int(end), int(hidden.shape[0]))
            if int(start) >= end:
                continue
            request_id = str(info.get("request_id", index))
            request_states = getattr(self, "_request_audio_states", None)
            if request_states is None:
                request_states = {}
                self._request_audio_states = request_states
            state = request_states.get(request_id)
            if not isinstance(state, dict):
                state = dict(info.get("audio_state", {}) or {})
                request_states[request_id] = state
            if state.get("finished"):
                stop_flags[index] = True
                continue
            if not sample_eligible[index]:
                # vLLM computes a logit row for incomplete chunked prefills but
                # discards its sampled token. Advancing codec/RNG state here
                # would make output depend on prefill chunking and compaction.
                continue
            codes = state.get("codes")
            if not isinstance(codes, torch.Tensor):
                codes = (info.get("audio_codes", {}) or {}).get("accumulated")
            if not isinstance(codes, torch.Tensor):
                codes = torch.empty(0, dtype=torch.long, device=hidden.device)
            else:
                codes = codes.to(device=hidden.device, dtype=torch.long).reshape(-1)
            step = int(state.get("step", 0))
            pending_samples.append(
                _PendingCodecSample(
                    output_index=index,
                    hidden_row=hidden[end - 1 : end],
                    codes=codes,
                    request_id=request_id,
                    step=step,
                    state=state,
                    info=info,
                )
            )

        if pending_samples:
            active_hidden_rows = [pending.hidden_row for pending in pending_samples]
            if getattr(self, "_batched_codec_sampling", True):
                active_hidden = (
                    active_hidden_rows[0]
                    if len(active_hidden_rows) == 1
                    else torch.cat(active_hidden_rows, dim=0)
                )
                sampled_batch = self._sample_audio_codes(
                    active_hidden,
                    [pending.codes for pending in pending_samples],
                    [pending.request_id for pending in pending_samples],
                    [pending.step for pending in pending_samples],
                )
                # One batched device-to-host synchronization replaces one
                # .item() synchronization per request.
                sampled_ids = sampled_batch.detach().to(device="cpu").tolist()
            else:
                # Rollback-compatible reference path: project, filter, sample,
                # and synchronize each request independently.
                sampled_rows: list[torch.Tensor] = []
                sampled_ids = []
                for pending in pending_samples:
                    sampled = self._sample_audio_codes(
                        pending.hidden_row,
                        [pending.codes],
                        [pending.request_id],
                        [pending.step],
                    )[0]
                    sampled_rows.append(sampled)
                    sampled_ids.append(int(sampled.item()))
                sampled_batch = torch.stack(sampled_rows)
        else:
            sampled_batch = hidden.new_empty((0,), dtype=torch.long)
            sampled_ids = []

        for row, pending in enumerate(pending_samples):
            sampled = sampled_batch[row].reshape(())
            sampled_id = int(sampled_ids[row])
            codes = pending.codes
            state = pending.state
            info = pending.info
            is_eos = sampled_id == self._num_audio_tokens - 1
            state["step"] = int(state.get("step", 0)) + 1
            reached_limit = int(state["step"]) >= int(state.get("max_tokens", 2048))
            finished = is_eos or reached_limit
            state["finished"] = finished
            # MiniCPMTTS.generate_chunk consumes the boundary sample but
            # returns only codes that were fed into the retained KV state.
            if not is_eos and not reached_limit:
                codes = torch.cat([codes[-(_REPETITION_WINDOW - 1) :], sampled.reshape(1)])
                delta = sampled.reshape(1, 1)
            else:
                delta = empty_delta
            state["codes"] = codes
            info["audio_state"] = state
            info["audio_codes"] = {
                "current": sampled.reshape(1),
                "accumulated": codes,
            }
            codec_deltas[pending.output_index] = delta
            terminal_flags[pending.output_index] = torch.tensor(finished, dtype=torch.bool)
            stop_flags[pending.output_index] = finished

        self._batch_stop_logits = (
            hidden.new_tensor(
                [[float("-inf"), 0.0] if finished else [0.0, float("-inf")] for finished in stop_flags]
            )
            if stop_flags
            else hidden.new_empty((0, 2))
        )
        # Lists are deliberate: the runner routes element i to request i,
        # preserving compaction alignment while emitting only this step's code.
        meta_outputs = {"finished": terminal_flags}
        if emit_duplex_metadata:
            meta_outputs.update(
                {
                    "native_duplex": native_duplex_flags,
                    "duplex_epoch": duplex_epochs,
                    "duplex_turn_id": duplex_turn_ids,
                    "llm_output_text_utf8": segment_texts_utf8,
                    "turn_end": turn_end_flags,
                }
            )
        multimodal_outputs: dict[str, Any] = {
            "codes": {"audio": codec_deltas},
            "meta": meta_outputs,
        }
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs=multimodal_outputs,
        )

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        self._deferred_cleanup_ids.update(str(req_id) for req_id in finished_req_ids)

    def _flush_deferred_cleanup(self) -> None:
        request_audio_states = getattr(self, "_request_audio_states", {})
        for request_id in self._deferred_cleanup_ids:
            self._request_generators.pop(request_id, None)
            request_audio_states.pop(request_id, None)
        self._deferred_cleanup_ids.clear()

    def _dummy_hidden_states(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        """Shape-correct zero tensor for vllm KV cache profiling.

        vllm's gpu_model_runner._dummy_run takes forward()'s return value as
        ``hidden_states`` and does ``hidden_states[logit_indices_device]``;
        returning None on the dummy path crashes with
        ``TypeError: 'NoneType' object is not subscriptable``.
        """
        for ref in (input_ids, positions, inputs_embeds):
            if isinstance(ref, torch.Tensor):
                num_tokens = int(ref.shape[0]) if ref.ndim >= 1 else 1
                device = ref.device
                break
        else:
            num_tokens = 1
            device = current_omni_platform.get_torch_device()
        hidden_size = int(getattr(self, "_hidden_size", 768) or 768)
        return torch.zeros((num_tokens, hidden_size), device=device, dtype=torch.bfloat16)

    def forward(
        self,
        input_ids=None,
        positions=None,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        self._flush_deferred_cleanup()
        if input_ids is None and inputs_embeds is None:
            return self._dummy_hidden_states(input_ids, positions, inputs_embeds)
        return self.tts_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states, *args, **kwargs):
        if not isinstance(hidden_states, torch.Tensor):
            return None
        if self._batch_stop_logits is None:
            return torch.zeros(
                hidden_states.shape[0],
                2,
                device=hidden_states.device,
                dtype=torch.float32,
            )
        logits = self._batch_stop_logits
        self._batch_stop_logits = None
        return logits

    def sample(self, logits, sampling_metadata):
        return self._sampler(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        return self._load_native_weights(weights)

    def _load_native_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded: set[str] = set()
        backbone_weights: list[tuple[str, torch.Tensor]] = []
        direct_params = dict(self.named_parameters())
        head_g = head_v = None

        for name, tensor in weights:
            if not name.startswith("tts."):
                continue
            stripped = name[len("tts.") :]
            if stripped.startswith("model."):
                backbone_weights.append((stripped[len("model.") :], tensor))
                continue
            if stripped == "head_code.0.parametrizations.weight.original0":
                head_g = tensor
                continue
            if stripped == "head_code.0.parametrizations.weight.original1":
                head_v = tensor
                continue
            target = stripped
            parameter = direct_params.get(target)
            if parameter is None:
                continue
            parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
            loaded.add(target)

        for name in self.tts_model.load_weights(backbone_weights):
            loaded.add(f"tts_model.{name}")

        if head_g is None or head_v is None:
            raise ValueError("MiniCPM-o checkpoint is missing weight-norm Talker head parameters")
        restored = _restore_weight_norm_weight(head_g, head_v)
        self.head_code[0].weight.data.copy_(
            restored.to(
                device=self.head_code[0].weight.device,
                dtype=self.head_code[0].weight.dtype,
            )
        )
        loaded.add("head_code.0.weight")
        return loaded

    def get_input_embeddings(self, input_ids, multimodal_embeddings=None, **kwargs):
        if hasattr(self, "emb_text") and self.emb_text is not None:
            return self.emb_text(input_ids)
        return torch.zeros(input_ids.shape[0], 1)

    def embed_input_ids(self, input_ids, **kwargs):
        return self.get_input_embeddings(input_ids, **kwargs)
