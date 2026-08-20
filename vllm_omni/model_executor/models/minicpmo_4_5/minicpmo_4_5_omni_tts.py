# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from:
# https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/main/modeling_minicpmo.py
"""MiniCPM-o 4.5 native autoregressive Talker.

Pipeline:
  1. Receive thinker hidden_states + full token IDs via additional_information
  2. Extract tts_bos..tts_eos region
  3. Build condition: emb_text(tokens) + projector_semantic(hidden) (hidden_text_merge)
  4. Project last hidden through head_code; vLLM Sampler picks the codec id
  5. Next decode embeds that id with emb_code and emits it to Code2Wav
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
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

_REPETITION_PENALTY_CHUNK_SIZE = 16
# Native duplex Talker must finish after one MiniCPMTTS.generate_chunk:
# 25 codec frames (``codec_chunk_frames``) plus the terminating sample.
# Without this, the single-vocab Sampler keeps the stage-1 request alive
# until codec EOS / 4096 and Thinker never starts the next model turn.
_DUPLEX_CODEC_TOKENS_PER_CHUNK = 26


def _native_duplex_chunk_budget(meta: Mapping[str, Any] | None) -> tuple[int, int]:
    """Return ``(max_tokens, min_tokens)`` for one native-duplex Talker request."""
    boundary = isinstance(meta, Mapping) and (bool(meta.get("turn_start")) or bool(meta.get("turn_end")))
    ceiling = _DUPLEX_CODEC_TOKENS_PER_CHUNK
    return ceiling, 0 if boundary else ceiling


def blank_scheduler_prompt_for_penalties(
    prompt_token_ids: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """Return a penalty prompt whose every position is the pad id (``vocab_size``).

    No Talker prompt position is codec history: prefill conditioning arrives as
    embeddings from ``preprocess`` and decode embeds sampled ids with
    ``emb_code``. The scheduler ids are placeholders (``llm2tts`` fills them
    with ``0``, or with thinker token ids on the non-handoff path), so scoring
    them would tax unrelated codec tokens.
    """
    return torch.full_like(prompt_token_ids, int(vocab_size))


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

        # Bound the int64 bincount workspace independently of request concurrency.
        encoded = encoded_rows[0] if len(encoded_rows) == 1 else torch.cat(encoded_rows)
        frequencies = torch.bincount(
            encoded,
            minlength=(end - start) * vocab_size,
        ).reshape(end - start, vocab_size)
        alpha = torch.pow(penalty_tensor, frequencies.to(dtype=logits.dtype))
        penalized[start:end] = torch.where(chunk_logits < 0, chunk_logits * alpha, chunk_logits / alpha)

    return penalized


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
        self._force_eos_rows: list[bool] | None = None
        self._mask_eos_rows: list[bool] | None = None
        self._request_audio_states: dict[str, dict[str, Any]] = {}
        self._deferred_cleanup_ids: set[str] = set()

        tts_config = getattr(config, "tts_config", None)
        if tts_config is None and getattr(config, "model_type", None) == "minicpmtts":
            tts_config = config
        if tts_config is not None:
            self._tts_config = tts_config
            self._tts_bos_id = getattr(tts_config, "audio_bos_token_id", 151687)
            self._text_eos_id = getattr(tts_config, "text_eos_token_id", 151692)
            self._num_audio_tokens = getattr(tts_config, "num_audio_tokens", 6562)
            self._codec_eos_id = int(getattr(tts_config, "eos_token_id", self._num_audio_tokens - 1))
            self._hidden_size = getattr(tts_config, "hidden_size", 768)
            self._normalize = getattr(tts_config, "normalize_projected_hidden", True)
        else:
            self._tts_config = None
            self._codec_eos_id = 0

        self.has_preprocess = True
        self.has_postprocess = False
        # Same-step codes travel through make_omni_output from the previous
        # sampled id (decode preprocess embeds that id via emb_code).
        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {("codes", "audio")}
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
            # Cross-process stage transport serializes CPU tensors as lists.
            # Normalize both local tensor handoffs and transported payloads
            # before validating/building the Talker condition.
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
            meta = info_dict.get("meta")
            full_embeds = self._build_condition_embeddings(
                token_ids,
                hidden_states,
                native_duplex=native_duplex,
            )
            offset = int(info_dict.get("_omni_num_computed_tokens", 0))
            request_id = str(info_dict.get("request_id", "0"))
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
            state: dict[str, Any] = {"finished": empty_condition, "step": 0}
            if native_duplex:
                max_tokens, min_tokens = _native_duplex_chunk_budget(meta if isinstance(meta, Mapping) else None)
                state["max_tokens"] = max_tokens
                state["min_tokens"] = min_tokens
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
                    # Prefill has no previous codec id. vLLM samples the first
                    # code after this forward; the next decode emits it.
                    "codes": {"audio": empty_codes},
                },
            )

        request_id = str(info_dict.get("request_id", "0"))
        stored = self._request_audio_states.get(request_id)
        if isinstance(stored, dict):
            state = stored
        if isinstance(state, dict) and state.get("finished"):
            # An empty speech segment can still be scheduled until EOS is
            # eligible. The sampler is forced to EOS; any shape-correct
            # embedding is enough for these leftover decode rows.
            weight = self.emb_code[0].weight
            empty_codes = torch.empty(0, dtype=torch.long, device=weight.device)
            return input_ids, weight.new_zeros((span_len, weight.shape[1])), {"codes": {"audio": empty_codes}}

        # Decode: vLLM's previous sampled codec id is this step's input.
        # Embed it with the codec table (not the 32k Llama embed_tokens) and
        # hand the same id to make_omni_output so Code2Wav sees it this step.
        code = input_ids.to(device=self.emb_code[0].weight.device, dtype=torch.long).reshape(-1)[-1:]
        embeds = self.emb_code[0](code)
        if int(code.item()) == int(self._codec_eos_id):
            if isinstance(state, dict):
                state["finished"] = True
            elif stored is None:
                self._request_audio_states[request_id] = {"finished": True, "step": 0}
            delta = torch.empty(0, dtype=torch.long, device=code.device)
        else:
            delta = code.reshape(1, 1)
        return input_ids, embeds, {"codes": {"audio": delta}}

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
        emit_duplex_metadata = any(isinstance(info, dict) and info.get("native_duplex") is True for info in infos)

        native_duplex_flags: list[torch.Tensor] = []
        duplex_epochs: list[torch.Tensor] = []
        duplex_turn_ids: list[torch.Tensor] = []
        segment_texts_utf8: list[torch.Tensor] = []
        turn_end_flags: list[torch.Tensor] = []
        empty_delta = hidden.new_empty((0, 1), dtype=torch.long)
        codec_deltas = [empty_delta for _ in infos]
        terminal_flags = [torch.tensor(False, dtype=torch.bool) for _ in infos]
        force_eos_rows = [False] * len(infos)
        mask_eos_rows = [False] * len(infos)
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
            request_id = str(info.get("request_id", index))
            state = self._request_audio_states.get(request_id)
            if not isinstance(state, dict):
                state = dict(info.get("audio_state", {}) or {})
                self._request_audio_states[request_id] = state
            empty_speech = bool(state.get("finished"))
            codes = info.get("codes", {})
            audio = codes.get("audio") if isinstance(codes, Mapping) else None
            if isinstance(audio, torch.Tensor) and audio.numel() > 0:
                codec_deltas[index] = audio.to(device=hidden.device, dtype=torch.long).reshape(-1, 1)
                if state.get("max_tokens") is not None:
                    state["step"] = int(state.get("step", 0)) + 1
            max_tokens = state.get("max_tokens")
            min_tokens = state.get("min_tokens")
            step = int(state.get("step", 0))
            # 26 samples include the terminating EOS, so force it after 25
            # forwarded frames — the same cadence as generate_chunk.
            hit_chunk_limit = max_tokens is not None and step >= int(max_tokens) - 1
            chunk_done = empty_speech or hit_chunk_limit
            if hit_chunk_limit:
                # The next sampled id is forced EOS and will finish the
                # request; mark the chunk done on this step so the data
                # plane does not wait for a follow-up empty decode.
                state["finished"] = True
            force_eos_rows[index] = chunk_done
            mask_eos_rows[index] = not force_eos_rows[index] and min_tokens is not None and step < int(min_tokens)
            terminal_flags[index] = torch.tensor(chunk_done, dtype=torch.bool)

        # Empty-speech and finished duplex chunks must sample codec EOS so
        # the scheduler releases the request; mid-chunk rows mask EOS until
        # the 26-token generate_chunk budget is met.
        self._force_eos_rows = force_eos_rows
        self._mask_eos_rows = mask_eos_rows
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
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={
                "codes": {"audio": codec_deltas},
                "meta": meta_outputs,
            },
        )

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        self._deferred_cleanup_ids.update(str(req_id) for req_id in finished_req_ids)

    def _flush_deferred_cleanup(self) -> None:
        request_audio_states = getattr(self, "_request_audio_states", {})
        for request_id in self._deferred_cleanup_ids:
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
        if hidden_states.numel() == 0:
            return hidden_states.new_empty((0, int(self._num_audio_tokens)))
        logits = self.head_code[0](hidden_states).float()
        force_eos = self._force_eos_rows
        mask_eos = self._mask_eos_rows
        self._force_eos_rows = None
        self._mask_eos_rows = None
        need_force = bool(force_eos and len(force_eos) == logits.shape[0] and any(force_eos))
        need_mask = bool(mask_eos and len(mask_eos) == logits.shape[0] and any(mask_eos))
        if need_force or need_mask:
            logits = logits.clone()
            eos_id = int(self._codec_eos_id)
            if need_force:
                assert force_eos is not None
                forced = torch.tensor(force_eos, dtype=torch.bool, device=logits.device)
                logits[forced] = float("-inf")
                logits[forced, eos_id] = 0.0
            if need_mask:
                assert mask_eos is not None
                masked = torch.tensor(mask_eos, dtype=torch.bool, device=logits.device)
                logits[masked, eos_id] = float("-inf")
        return logits

    def sample(self, logits, sampling_metadata):
        prompt_ids = getattr(sampling_metadata, "prompt_token_ids", None)
        if isinstance(logits, torch.Tensor) and isinstance(prompt_ids, torch.Tensor):
            # Copy rather than mutate: the runner may hand us the input batch's
            # own persistent SamplingMetadata.
            sampling_metadata = replace(
                sampling_metadata,
                prompt_token_ids=blank_scheduler_prompt_for_penalties(prompt_ids, logits.shape[-1]),
            )
        return Sampler()(logits, sampling_metadata)

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
        del multimodal_embeddings
        # Decode tokens live in the codec table. Prefill overwrites these
        # embeddings in preprocess with emb_text + projected thinker hidden.
        return self.emb_code[0](input_ids)

    def embed_input_ids(self, input_ids, **kwargs):
        return self.get_input_embeddings(input_ids, **kwargs)
