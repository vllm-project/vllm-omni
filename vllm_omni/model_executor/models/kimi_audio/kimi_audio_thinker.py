# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-Audio fused thinker: extends upstream vLLM's
``KimiAudioForConditionalGeneration`` (Whisper-large-v3 + VQ-Adaptor +
Qwen2-7B, ASR-only) with the MIMO audio-out branch (6 Qwen2 decoder layers
+ RMS norm + shared-vocab linear head) hooked at layer
``kimia_mimo_transformer_from_layer_index`` of the main LLM.

The upstream model is single-stream (ASR). The real Kimi-Audio model is
dual-stream: at every position the model sees ``embed(audio_stream) +
embed(text_stream)``. This module rebuilds the dual-stream embedding
from the single stream vLLM hands us, runs the MIMO branch with
temperature/top-k sampling, enforces the ``kimia_mimo_audiodelaytokens``
lag on the audio head, and feeds the previously-sampled audio token back
through the embedding on subsequent decode steps."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm.model_executor.models.kimi_audio import (
    KimiAudioForConditionalGeneration as _UpstreamKimiAudio,
)
from vllm.model_executor.models.utils import WeightsMapper
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)

# Token IDs for Kimi-Audio special tokens (from the shipped tokenizer_config).
_KIMIA_MSG_END = 151645  # <|im_msg_end|>
_KIMIA_MEDIA_BEGIN = 151661  # <|im_media_begin|>
_KIMIA_MEDIA_END = 151663  # <|im_media_end|>
_KIMIA_TEXT_BLANK = 151666  # <|im_kimia_text_blank|>
_KIMIA_TEXT_EOS = 151667  # <|im_kimia_text_eos|>
_KIMIA_USER_MSG_START = 151670  # <|im_kimia_user_msg_start|>
_KIMIA_ASSISTANT_MSG_START = 151671  # <|im_kimia_assistant_msg_start|>
_KIMIA_SPEECH_CT = 151675  # <|im_kimia_speech_ct_id|>
_KIMIA_SPEECH_CTD = 151676  # <|im_kimia_speech_ctd_id|>

# Tokens that belong on the AUDIO stream; at those positions the TEXT stream
# carries ``kimia_text_blank``. Everything else (regular tokenizer ids, and
# the text-only ``kimia_text_eos``) stays on the TEXT stream with the AUDIO
# stream holding ``kimia_text_blank``. This exactly mirrors
# ``prompt_manager.tokenize_message`` — verified against the reference dump.
_AUDIO_STREAM_TOKENS = frozenset(
    {
        _KIMIA_USER_MSG_START,
        _KIMIA_ASSISTANT_MSG_START,
        _KIMIA_MEDIA_BEGIN,
        _KIMIA_MEDIA_END,
        _KIMIA_MSG_END,
        _KIMIA_SPEECH_CT,
        _KIMIA_SPEECH_CTD,
    }
)

# Audio codes live in [KIMIA_TOKEN_OFFSET, vocab_size); the MIMO head's
# logits for the text vocabulary are discarded before sampling.
KIMIA_TOKEN_OFFSET = 152064


def _build_mimo_layers(config) -> tuple[nn.ModuleList, nn.Module]:
    """Build the MIMO decoder layers with HF's Qwen2DecoderLayer. We skip
    vLLM's optimized layer because it's tied to paged KV cache; the MIMO
    branch is small enough that eager attention is fine.

    Returns the layer stack plus a ``Qwen2RotaryEmbedding`` — since
    transformers>=4.45 moved RoPE out of the attention layer, we must
    precompute ``(cos, sin)`` and pass it to each layer as
    ``position_embeddings``."""
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding

    # transformers 5 moved ``rope_theta`` into ``rope_parameters``. Older
    # 4.x configs expose it as a flat attribute; support both.
    if hasattr(config, "rope_parameters"):
        rope_theta = config.rope_parameters.get("rope_theta", 10000.0)
    else:
        rope_theta = config.rope_theta

    qwen2_cfg = Qwen2Config(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.kimia_mimo_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=rope_theta,
        hidden_act=config.hidden_act,
        attention_dropout=0.0,
        # sdpa avoids the O(N^2) attention matrix eager would materialize
        # at the 8192-token dummy profile run.
        attn_implementation="sdpa",
    )
    layers = nn.ModuleList([Qwen2DecoderLayer(qwen2_cfg, layer_idx=i) for i in range(config.kimia_mimo_layers)])
    rotary_emb = Qwen2RotaryEmbedding(config=qwen2_cfg)
    return layers, rotary_emb


def _build_mimo_norm(config) -> nn.Module:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    return Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


_HF_TO_VLLM_MAPPER = WeightsMapper(
    orig_to_new_prefix={
        "model.encoder.": "audio_tower.",
        "model.vq_adaptor.layers.0.": "multi_modal_projector.vq_adaptor_layers_0.",
        "model.vq_adaptor.layers.3.": "multi_modal_projector.vq_adaptor_layers_3.",
        "model.vq_adaptor.layers.4.": "multi_modal_projector.vq_adaptor_layers_4.",
        "model.layers.": "language_model.model.layers.",
        "model.embed_tokens.": "language_model.model.embed_tokens.",
        "model.norm.": "language_model.model.norm.",
        "lm_head.": "language_model.lm_head.",
        "model.mimo_layers.": "mimo_layers.",
        "model.mimo_norm.": "mimo_norm.",
        "mimo_output.": "mimo_output.",
    },
    # Whisper HF uses .fc1/.fc2; vLLM's WhisperEncoder expects .mlp.fc1/.mlp.fc2.
    orig_to_new_substr={
        ".fc1.": ".mlp.fc1.",
        ".fc2.": ".mlp.fc2.",
    },
)


class KimiAudioFusedThinker(_UpstreamKimiAudio):
    hf_to_vllm_mapper = _HF_TO_VLLM_MAPPER

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config

        # The dual-stream feedback loop (``self._req_state`` /
        # ``self._mimo_capture_slot``) carries per-request state on the
        # MODULE, not on ``sampling_metadata``. That's safe only with one
        # in-flight request per stage. Reject larger ``max_num_seqs``
        # loudly at init rather than producing silently-corrupted audio
        # under concurrent decode. Multi-request support would mean
        # threading the vLLM request id through ``embed_input_ids`` /
        # ``forward`` and keying state on it.
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        if max_num_seqs != 1:
            raise ValueError(
                f"KimiAudioFusedThinker requires max_num_seqs=1 on stage 0 "
                f"(got {max_num_seqs}). The MIMO branch's per-request state is "
                f"kept on the module instance and would race under concurrent "
                f"requests. See _req_state / _mimo_capture_slot in "
                f"kimi_audio_thinker.py."
            )

        self.model_path = vllm_config.model_config.model

        self._generate_audio = bool(getattr(config, "kimia_generate_audio", False))

        # kimi-audio-7b uses 6; finetunes may differ.
        self._audio_delay = int(getattr(config, "kimia_mimo_audiodelaytokens", 6))

        # Defaults match the kimia README. Override via ``hf_overrides``.
        self._audio_temperature = float(getattr(config, "kimia_audio_temperature", 0.8))
        self._audio_top_k = int(getattr(config, "kimia_audio_top_k", 10))

        self._req_state: dict[str, Any] = self._fresh_req_state()

        if self._generate_audio:
            self.mimo_layers, self.mimo_rotary_emb = _build_mimo_layers(config)
            self.mimo_norm = _build_mimo_norm(config)
            # Head shares the global vocab with the text head; audio IDs
            # occupy [KIMIA_TOKEN_OFFSET, vocab_size).
            self.mimo_output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

            self._mimo_branch_layer_idx = int(config.kimia_mimo_transformer_from_layer_index)
            self._mimo_capture_slot: list[torch.Tensor | None] = [None]

            target_layer = self.language_model.model.layers[self._mimo_branch_layer_idx]
            target_layer.register_forward_hook(self._capture_layer21_output)
            logger.info(
                "KimiAudioFusedThinker: MIMO branch hooked at language_model.model.layers[%d] "
                "(mimo_layers=%d, audio_delay=%d, audio_temperature=%.2f, audio_top_k=%d)",
                self._mimo_branch_layer_idx,
                len(self.mimo_layers),
                self._audio_delay,
                self._audio_temperature,
                self._audio_top_k,
            )
        else:
            logger.info(
                "KimiAudioFusedThinker: kimia_generate_audio=False; MIMO audio-out branch not built. Text-out only."
            )

    @staticmethod
    def _fresh_req_state() -> dict[str, Any]:
        return {
            "decode_step": 0,
            "last_audio_token": None,
            "text_eos_seen": False,
            # Set once the MIMO audio head samples msg_end/media_end;
            # read by compute_logits to boost vLLM's text EOS and stop
            # generation — mirrors upstream "output_type=both" termination.
            "audio_eod_seen": False,
            # Per-layer KV cache for the MIMO branch. Mirrors upstream's
            # past_key_values[idx + len(self.layers)] — without this the
            # MIMO layers only see the current token's hidden state at
            # decode time and can't track audio-stream context, so the
            # audio head never emits media_end/msg_end and generation
            # runs to max_tokens. Lazily created in _mimo_sample_at_indices.
            "mimo_cache": None,
        }

    def _capture_layer21_output(self, module, args, output) -> None:
        # vLLM's Qwen2DecoderLayer returns (hidden_states, residual) with
        # the residual connection deferred to the next layer's
        # input_layernorm. The logical "post-layer" hidden state that HF's
        # MoonshotKimiaModel.forward clones into ``mimo_hidden_states`` is
        # actually hidden_states + residual — grabbing just ``output[0]``
        # feeds the MIMO branch post-MLP pre-residual activations, which
        # aren't in the manifold the MIMO layers were trained to consume.
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            self._mimo_capture_slot[0] = output[0] + output[1]
        else:
            self._mimo_capture_slot[0] = output[0] if isinstance(output, tuple) else output

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: tuple[torch.Tensor, ...] | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """vLLM hands a single ``input_ids`` tensor; Kimi-Audio is
        dual-stream (audio + text) and sums the two embeddings. Split rule
        mirrors ``prompt_manager.tokenize_message``: audio-control tokens go
        to the audio stream, everything else to the text stream, and the
        "other" stream carries ``kimia_text_blank`` at that position."""
        embed_tokens = self.language_model.model.embed_tokens

        # Prefill detection at batch_size=1: prefill prompts contain the
        # role marker (kimia_user_msg_start) which can't appear in decode.
        is_prefill = bool(self._contains_any(input_ids, _AUDIO_STREAM_TOKENS))

        if is_prefill:
            audio_ids, text_ids = self._split_prefill(input_ids)
            # Reset per-request state on every prefill — including
            # audio-input prefill, otherwise QA reuses stale decode_step /
            # audio_eod_seen across requests.
            self._req_state = self._fresh_req_state()
        else:
            audio_ids, text_ids = self._build_decode_streams(input_ids)

        audio_emb = embed_tokens(audio_ids)
        text_emb = embed_tokens(text_ids)

        # Audio-in fusion at media-content positions: ``audio_emb =
        # (audio_emb + whisper) * sqrt(2)``. Mirrors upstream HF
        # ``MoonshotKimiaModel.forward``.
        if (
            multimodal_embeddings is not None
            and len(multimodal_embeddings) > 0
            and is_multimodal is not None
            and is_multimodal.any()
        ):
            whisper_embeds = multimodal_embeddings[0]
            if isinstance(whisper_embeds, (list, tuple)):
                whisper_embeds = torch.cat(list(whisper_embeds), dim=0)
            elif whisper_embeds.dim() == 3:
                whisper_embeds = whisper_embeds.reshape(-1, whisper_embeds.shape[-1])

            num_to_use = min(whisper_embeds.shape[0], int(is_multimodal.sum().item()))
            mm_positions = is_multimodal.nonzero(as_tuple=True)[0]
            mm_mask = torch.zeros_like(is_multimodal)
            mm_mask[mm_positions[:num_to_use]] = True

            used_whisper = whisper_embeds[:num_to_use].to(dtype=audio_emb.dtype)
            audio_emb[mm_mask] = (audio_emb[mm_mask] + used_whisper) * (2.0**0.5)

        return audio_emb + text_emb

    @staticmethod
    def _contains_any(input_ids: torch.Tensor, token_set: frozenset[int]) -> bool:
        # Fast membership test without a Python loop over the tensor.
        flat = input_ids.reshape(-1)
        hit = torch.zeros_like(flat, dtype=torch.bool)
        for tid in token_set:
            hit |= flat == tid
        return bool(hit.any().item())

    @staticmethod
    def _split_prefill(input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Audio stream: control tokens AND discrete codec IDs
        # (>= KIMIA_TOKEN_OFFSET, GLM-4-Voice tokens for input audio).
        # Text stream: everything else.
        is_audio = input_ids >= KIMIA_TOKEN_OFFSET
        for tid in _AUDIO_STREAM_TOKENS:
            is_audio |= input_ids == tid

        audio_ids = torch.where(is_audio, input_ids, torch.full_like(input_ids, _KIMIA_TEXT_BLANK))
        text_ids = torch.where(is_audio, torch.full_like(input_ids, _KIMIA_TEXT_BLANK), input_ids)
        return audio_ids, text_ids

    def _build_decode_streams(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """For a decode step, ``input_ids`` is the text token vLLM just
        sampled. The audio stream input is the audio token the MIMO branch
        sampled at the previous step (or ``kimia_text_blank`` while the
        audio head is still in the ``kimia_mimo_audiodelaytokens`` lag)."""
        state = self._req_state
        step = int(state["decode_step"])

        # Audio stream: cached previous-step sample, or blank during lag.
        last_audio = state["last_audio_token"]
        if step < self._audio_delay or last_audio is None:
            audio_tok = _KIMIA_TEXT_BLANK
        else:
            audio_tok = int(last_audio)
        audio_ids = torch.full_like(input_ids, audio_tok)

        # Off-by-one: pass kimia_text_eos through on the step it's first
        # observed; substitute blanks only on subsequent steps. Replacing
        # on the same step we latch causes the MIMO branch to diverge.
        if state["text_eos_seen"]:
            text_ids = torch.full_like(input_ids, _KIMIA_TEXT_BLANK)
        else:
            text_ids = input_ids
            if input_ids.numel() == 1 and int(input_ids.item()) == _KIMIA_TEXT_EOS:
                state["text_eos_seen"] = True

        return audio_ids, text_ids

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors | OmniOutput:
        if self._generate_audio:
            # Clear stale capture before the hook fires inside super().forward.
            self._mimo_capture_slot[0] = None

        # Drop runner-only kwargs (logits_index, sampler, ...) — the
        # upstream ASR-only forward takes only input_ids/positions/etc.
        text_hidden = super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        if not self._generate_audio:
            return text_hidden

        capture = self._mimo_capture_slot[0]
        if capture is None:
            logger.warning(
                "MIMO branch enabled but capture buffer is empty. "
                "Layer-21 hook did not fire (CUDA Graph capture? Compiled "
                "forward path?). Falling back to text-only output."
            )
            return text_hidden
        self._mimo_capture_slot[0] = None

        logits_index = kwargs.get("logits_index")
        audio_token_ids = self._mimo_sample_at_indices(
            capture=capture,
            positions=positions,
            logits_index=logits_index,
        )
        return OmniOutput(
            text_hidden_states=text_hidden,
            multimodal_outputs={"audio_tokens": audio_token_ids},
        )

    def _mimo_sample_at_indices(
        self,
        capture: torch.Tensor,
        positions: torch.Tensor | None,
        logits_index: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the MIMO branch, then sample the audio head with temp/top-k
        at the sampling positions only. Applies the audio-delay mask by
        forcing the sampled token to ``kimia_text_blank`` for the first
        ``kimia_mimo_audiodelaytokens`` steps of generation.

        The MIMO layers maintain their own KV cache in
        ``_req_state["mimo_cache"]`` (a ``DynamicCache``), mirroring
        upstream's ``past_key_values[idx + len(self.layers)]``. Without
        this, decode-step attention has no audio-stream history to condition
        on and the audio head never emits ``media_end``."""
        from transformers.cache_utils import DynamicCache

        # vLLM flattens activations to 2-D [N, hidden]; HF's Qwen2DecoderLayer
        # expects [B, T, hidden] — treat the flattened tensor as one batch.
        hidden_3d = capture.unsqueeze(0) if capture.dim() == 2 else capture
        seq_len = hidden_3d.shape[1]

        # Absolute position ids (prefill: 0..N-1; decode: past_len .. past_len+seq_len-1).
        # Use vLLM's ``positions`` when its shape lines up with the capture;
        # the hook sometimes fires on a slightly padded activation, so fall
        # back to a fresh arange anchored on the cache length.
        cache = self._req_state["mimo_cache"]
        if cache is None:
            cache = DynamicCache()
            self._req_state["mimo_cache"] = cache
        past_len = cache.get_seq_length(layer_idx=0)

        if positions is not None and positions.reshape(-1).shape[0] == seq_len:
            position_ids = positions.reshape(1, -1).to(torch.long)
        else:
            position_ids = torch.arange(
                past_len, past_len + seq_len, device=capture.device, dtype=torch.long
            ).unsqueeze(0)

        cache_position = torch.arange(past_len, past_len + seq_len, device=capture.device, dtype=torch.long)

        # Prefill (seq_len > 1) needs a causal mask over the current block
        # plus all zeros for past-cache columns. Decode (seq_len == 1) can
        # attend to every cached position unconditionally, so a None mask
        # is correct and also lets HF's attention take its faster path.
        if seq_len > 1:
            kv_len = past_len + seq_len
            attn_mask = torch.zeros((1, 1, seq_len, kv_len), device=capture.device, dtype=hidden_3d.dtype)
            causal_block = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=capture.device, dtype=hidden_3d.dtype),
                diagonal=1,
            )
            attn_mask[:, :, :, past_len : past_len + seq_len] = causal_block.unsqueeze(0).unsqueeze(0)
        else:
            attn_mask = None

        position_embeddings = self.mimo_rotary_emb(hidden_3d, position_ids)

        h = hidden_3d
        for layer in self.mimo_layers:
            layer_out = layer(
                hidden_states=h,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            h = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        h = self.mimo_norm(h)  # [1, T, hidden]

        # Restrict to the sampling positions. vLLM passes logits_index as a
        # 1-D index into the flattened [N, hidden] activation. At batch=1 the
        # mapping is straight indexing into the T dim.
        if logits_index is None:
            sample_idx = torch.tensor([seq_len - 1], device=capture.device, dtype=torch.long)
        else:
            sample_idx = logits_index.reshape(-1).to(capture.device).to(torch.long)

        h_sample = h[0, sample_idx, :]  # [num_sample, hidden]
        audio_logits_full = self.mimo_output(h_sample)
        # Sample from the FULL vocab (mirrors reference: the audio head
        # can emit eod tokens like media_end/msg_end to signal end-of-audio;
        # downstream ``_extract_audio_tokens`` drops anything below
        # KIMIA_TOKEN_OFFSET, so the filter on the output side is what
        # keeps non-code tokens out of the code2wav stage.
        audio_codes = self._sample_topk(audio_logits_full).to(torch.long)

        # Audio-delay mask: first ``kimia_mimo_audiodelaytokens`` generation
        # steps emit ``kimia_text_blank`` on the audio stream. Batch=1
        # assumption: all sampling positions in this call belong to the
        # same request, so one step counter is enough.
        step = int(self._req_state["decode_step"])
        if step < self._audio_delay:
            audio_codes = torch.full_like(audio_codes, _KIMIA_TEXT_BLANK)
        else:
            last_id = int(audio_codes.reshape(-1)[-1].item())
            if last_id == _KIMIA_MSG_END or last_id == _KIMIA_MEDIA_END:
                self._req_state["audio_eod_seen"] = True

        # Cache the last-sampled token for the next decode step's audio
        # stream, and advance the step counter.
        self._req_state["last_audio_token"] = int(audio_codes.reshape(-1)[-1].item())
        self._req_state["decode_step"] = step + 1

        return audio_codes.reshape(-1)

    def _sample_topk(self, logits: torch.Tensor) -> torch.Tensor:
        """Temperature + top-k sampling on a [..., vocab] logits tensor,
        matching ``KimiASampler.sample_audio_logits``. No repetition
        penalty yet — left as a follow-up; the audio head repetition
        patterns aren't triggered at the default prompt length."""
        if self._audio_temperature < 1e-6:
            return logits.argmax(dim=-1)

        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        logprobs = logprobs / self._audio_temperature
        probs = torch.exp(logprobs)

        if self._audio_top_k > 0:
            top_k_probs, top_k_indices = torch.topk(probs, self._audio_top_k, dim=-1)
            sampled = torch.multinomial(top_k_probs, num_samples=1).squeeze(-1)
            return top_k_indices.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from vllm.model_executor.models.utils import AutoWeightsLoader

        # Skip the whisper decoder (model.*), the detokenizer
        # (audio_decoder.*), and MIMO keys when the branch is disabled.
        skip_prefixes = ["model.", "audio_decoder."]
        if not self._generate_audio:
            skip_prefixes.extend(["mimo_layers.", "mimo_norm.", "mimo_output."])

        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
