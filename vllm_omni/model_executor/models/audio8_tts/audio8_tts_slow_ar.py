# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audio8 TTS Preview -- Slow AR model (Stage 0).

Text (+ optional reference voice) -> one semantic token per codec frame; the
frame's remaining 9 codebooks are produced by the nested Fast AR.

Backbone is vLLM's ``Qwen2Model`` (qkv bias, no q/k norm), with three
adjustments: interleaved (GPT-J) RoPE, multi-codebook input embedding at
semantic positions (summed, unlike Fish Speech's ``1/sqrt(Q+1)`` rescale), and
logits masked to the semantic range + ``<|im_end|>`` then Repetition-Aware
sampled.

Streaming contract is delta: each decode step appends one frame of
``num_codebooks`` codes to ``codes.audio``.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.utils.speaker_cache import get_speaker_cache

from .audio8_tts_fast_ar import Audio8TTSFastAR
from .codec_utils import encode_reference_audio_codes
from .configuration_audio8_tts import (
    Audio8TTSConfig,
    Audio8TTSFastARConfig,
    Audio8TTSSlowARConfig,
)
from .prompt_utils import build_voice_clone_prompt_ids
from .sampling import SAMPLING_EPS, ras_sample_batch

logger = init_logger(__name__)

#: Defaults from ``generation_config.json`` of the released checkpoint.
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 0.9


def _remap_audio8_tts_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    q_size: int,
    kv_size: int,
    fast_q_size: int,
    fast_kv_size: int,
) -> Iterable[tuple[str, torch.Tensor]]:
    """Rename Audio8 checkpoint tensors to vLLM / Qwen2 names.

    The checkpoint is flat (no ``text_model.`` prefix). Slow AR tensors move
    under ``model.``, Fast AR tensors under ``fast_ar.``, and ``wqkv`` splits
    into ``{q,k,v}_proj``. See the body for the exact mapping table.
    """

    def rewrite_block(suffix: str) -> str:
        for old, new in (
            (".attention.wo.", ".self_attn.o_proj."),
            (".attention.q_norm.", ".self_attn.q_norm."),
            (".attention.k_norm.", ".self_attn.k_norm."),
            (".attention_norm.", ".input_layernorm."),
            (".feed_forward.w1.", ".mlp.gate_proj."),
            (".feed_forward.w3.", ".mlp.up_proj."),
            (".feed_forward.w2.", ".mlp.down_proj."),
            (".ffn_norm.", ".post_attention_layernorm."),
        ):
            suffix = suffix.replace(old, new)
        return suffix

    for name, tensor in weights:
        # RoPE tables are recomputed by vLLM.
        if name.endswith(("freqs_cis", "fast_freqs_cis")) or "rotary_emb.inv_freq" in name:
            continue

        if name.startswith("fast_layers."):
            suffix = name[len("fast_layers.") :]
            if ".attention.wqkv." in suffix:
                layer_prefix, _, param = suffix.partition(".attention.wqkv.")
                yield f"fast_ar.layers.{layer_prefix}.self_attn.q_proj.{param}", tensor[:fast_q_size]
                yield (
                    f"fast_ar.layers.{layer_prefix}.self_attn.k_proj.{param}",
                    tensor[fast_q_size : fast_q_size + fast_kv_size],
                )
                yield (
                    f"fast_ar.layers.{layer_prefix}.self_attn.v_proj.{param}",
                    tensor[fast_q_size + fast_kv_size :],
                )
                continue
            yield f"fast_ar.layers.{rewrite_block(suffix)}", tensor
            continue

        if name.startswith(("fast_embeddings.", "fast_output.", "fast_norm.", "fast_project_in.")):
            yield f"fast_ar.{name}", tensor
            continue

        if name.startswith("layers."):
            suffix = name[len("layers.") :]
            if ".attention.wqkv." in suffix:
                layer_prefix, _, param = suffix.partition(".attention.wqkv.")
                yield f"model.layers.{layer_prefix}.self_attn.q_proj.{param}", tensor[:q_size]
                yield (
                    f"model.layers.{layer_prefix}.self_attn.k_proj.{param}",
                    tensor[q_size : q_size + kv_size],
                )
                yield f"model.layers.{layer_prefix}.self_attn.v_proj.{param}", tensor[q_size + kv_size :]
                continue
            yield f"model.layers.{rewrite_block(suffix)}", tensor
            continue

        if name == "embeddings.weight":
            yield "model.embed_tokens.weight", tensor
            continue
        if name == "norm.weight":
            yield "model.norm.weight", tensor
            continue

        yield name, tensor


class Audio8TTSSlowARForConditionalGeneration(nn.Module):
    """Stage 0: text -> semantic tokens + residual codec codes."""

    prefer_model_sampler = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        config: Audio8TTSConfig = vllm_config.model_config.hf_config  # type: ignore[assignment]
        self.config = config
        self.text_config: Audio8TTSSlowARConfig = config.get_text_config()
        self.fast_ar_config: Audio8TTSFastARConfig = config.fast_ar_config

        self._semantic_begin_id = int(self.text_config.semantic_begin_id)
        self._semantic_end_id = int(self.text_config.semantic_end_id)
        self._num_semantic_ids = self._semantic_end_id - self._semantic_begin_id + 1
        self._eos_token_id = int(self.text_config.eos_token_id)
        self._pad_token_id = int(self.text_config.pad_token_id)
        self._codebook_size = int(self.text_config.codebook_size)
        self._num_codebooks = int(self.text_config.num_codebooks)
        self._ras_window_size = int(config.ras_window_size)
        self._ras_temperature = float(config.ras_temperature)
        self._ras_top_p = float(config.ras_top_p)

        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = True
        self.mtp_hidden_size = int(self.text_config.hidden_size)
        self.talker_mtp_output_key = ("codes", "audio")
        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {("hidden_states", "last")}
        self.talker_mtp_graph_safe = True

        self.model = Qwen2Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self._fix_rope_style()

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.text_config.vocab_size,
                self.text_config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(self.text_config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        # Summed multi-codebook input embedding table.
        self.codebook_embeddings = nn.Embedding(
            self._codebook_size * self._num_codebooks,
            self.text_config.hidden_size,
        )

        # Fast AR shares the parent's VllmConfig but must not share the
        # compilation context: see the Fish Speech note on why this is
        # copy.copy and not dataclasses.replace (pydantic validators re-run and
        # reject an already-rebound compilation backend).
        fast_ar_compilation = copy.copy(vllm_config.compilation_config)
        fast_ar_compilation.static_forward_context = {}
        self._fast_ar_vllm_config = copy.copy(vllm_config)
        self._fast_ar_vllm_config.compilation_config = fast_ar_compilation
        from vllm.config.vllm import set_current_vllm_config

        with set_current_vllm_config(self._fast_ar_vllm_config):
            self.fast_ar = Audio8TTSFastAR(
                vllm_config=self._fast_ar_vllm_config,
                config=self.fast_ar_config,
                slow_ar_config=self.text_config,
                prefix="fast_ar",
            )

        # Constant logits mask: semantic codes plus <|im_end|>. Safe as a
        # non-persistent buffer under vLLM (module built on the target device),
        # unlike under ``PreTrainedModel.from_pretrained`` on transformers >= 5,
        # which meta-inits and then empty_like's every non-persistent buffer.
        vocab = int(self.text_config.vocab_size)
        allowed = torch.zeros((vocab,), dtype=torch.bool)
        allowed[self._semantic_begin_id : min(self._semantic_end_id + 1, vocab)] = True
        if self._eos_token_id < vocab:
            allowed[self._eos_token_id] = True
        self.register_buffer("_semantic_allowed_mask", allowed, persistent=False)

        self._speaker_cache = get_speaker_cache()
        self._tokenizer = None

    def _fix_rope_style(self) -> None:
        """Rebuild RoPE as interleaved (GPT-J); vLLM's Qwen2 defaults to NeoX.

        This also replaces the reference implementation's ``freqs_cis`` table
        entirely: ``get_rope`` computes its cos/sin cache in the constructor, so
        there is no checkpoint-absent buffer for a loader to leave uninitialised.
        """
        from vllm.model_executor.layers.rotary_embedding import get_rope

        rope_parameters = dict(self.text_config.rope_parameters or {})
        rope_parameters.setdefault("rope_type", "default")
        for layer in self.model.layers:
            attn = layer.self_attn
            attn.rotary_emb = get_rope(
                head_size=attn.head_dim,
                max_position=self.text_config.max_position_embeddings,
                is_neox_style=False,
                rope_parameters=rope_parameters,
            )
        logger.info("Audio8 TTS: switched %d layers to interleaved RoPE", len(self.model.layers))

    # -------------------- vLLM hooks --------------------

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is None:
            return None
        return logits.masked_fill(~self._semantic_allowed_mask, float("-inf"))

    # -------------------- Repetition-Aware Sampling --------------------

    def _compact_semantic_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Slice the allowed vocabulary into ``[B, num_semantic_ids + 1]``.

        Local ids ``0..num_semantic_ids-1`` are semantic codes; the last column
        is ``<|im_end|>``. Sorting 4097 columns instead of the full 155776
        vocabulary is the difference between a cheap and an expensive AR step,
        and is exactly equivalent because everything else is already ``-inf``.
        """
        semantic = logits[:, self._semantic_begin_id : self._semantic_end_id + 1]
        eos = logits[:, self._eos_token_id : self._eos_token_id + 1]
        return torch.cat((semantic, eos), dim=-1)

    def _local_to_token_id(self, local_ids: torch.Tensor) -> torch.Tensor:
        is_eos = local_ids >= self._num_semantic_ids
        return torch.where(is_eos, torch.full_like(local_ids, self._eos_token_id), local_ids + self._semantic_begin_id)

    def _recent_local_ids(self, output_token_ids: list[list[int]], num_reqs: int, device: torch.device):
        """Build the ``[B, W]`` RAS window in compact id space.

        Reads the host-side decoded-token history the AR runner already
        materialised, so this costs one small H2D copy and never syncs the GPU.
        Returns ``None`` when no request has history yet.
        """
        window = self._ras_window_size
        if window <= 0:
            return None
        rows: list[list[int]] = []
        any_history = False
        for req_idx in range(num_reqs):
            history = output_token_ids[req_idx] if req_idx < len(output_token_ids) else []
            recent = [int(token) for token in history[-window:]]
            local = [
                token - self._semantic_begin_id if self._semantic_begin_id <= token <= self._semantic_end_id else -1
                for token in recent
            ]
            any_history = any_history or bool(local)
            rows.append([-1] * (window - len(local)) + local)
        if not any_history:
            return None
        return torch.tensor(rows, dtype=torch.long, device=device)

    def sample(self, logits: torch.Tensor, sampling_metadata: Any) -> SamplerOutput | None:
        """Sample the semantic token with the reference filter order + RAS.

        Returns ``None`` to fall back to vLLM's sampler when the request mix
        needs features RAS does not model (logprobs, penalties, bad words).
        """
        if logits is None or logits.numel() == 0:
            return None
        if sampling_metadata.max_num_logprobs is not None:
            return None
        if not sampling_metadata.no_penalties:
            return None
        if sampling_metadata.bad_words_token_ids:
            return None

        logits = logits.to(torch.float32)
        if sampling_metadata.allowed_token_ids_mask is not None:
            num_reqs = int(logits.shape[0])
            logits.masked_fill_(sampling_metadata.allowed_token_ids_mask[:num_reqs], float("-inf"))
        for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
            logits = processor.apply(logits)

        compact = self._compact_semantic_logits(logits)
        num_reqs = int(compact.shape[0])
        device = compact.device

        temperature = sampling_metadata.temperature
        if sampling_metadata.all_greedy or temperature is None:
            local_ids = compact.argmax(dim=-1)
        else:
            temperature = temperature[:num_reqs]
            top_p = DEFAULT_TOP_P if sampling_metadata.top_p is None else sampling_metadata.top_p[:num_reqs]
            top_k = DEFAULT_TOP_K if sampling_metadata.top_k is None else sampling_metadata.top_k[:num_reqs]
            recent = self._recent_local_ids(sampling_metadata.output_token_ids, num_reqs, device)
            local_ids = ras_sample_batch(
                compact,
                recent,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                ras_temperature=self._ras_temperature,
                ras_top_p=self._ras_top_p,
                num_semantic_ids=self._num_semantic_ids,
                generators=sampling_metadata.generators,
                num_reqs=num_reqs,
            )

        token_ids = self._local_to_token_id(local_ids).to(dtype=torch.int32)
        return SamplerOutput(sampled_token_ids=token_ids.unsqueeze(-1), logprobs_tensors=None)

    # -------------------- Omni multimodal plumbing --------------------

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        hidden = model_outputs
        info_dicts = kwargs.get("model_intermediate_buffer")
        if info_dicts is None:
            info_dicts = kwargs.get("runtime_additional_information") or []

        audio_codes_list: list[torch.Tensor] = []
        for info in info_dicts:
            if not isinstance(info, dict):
                continue
            codes = info.get("codes", {}).get("audio")
            if isinstance(codes, torch.Tensor):
                audio_codes_list.append(codes)

        if not audio_codes_list:
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})

        audio_codes = torch.cat(audio_codes_list, dim=0)
        hidden = hidden[: int(audio_codes.shape[0])]
        return OmniOutput(text_hidden_states=hidden, multimodal_outputs={"audio_codes": audio_codes})

    # -------------------- preprocess / postprocess --------------------

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        additional_information = info_dict.get("additional_information")
        if isinstance(additional_information, dict):
            merged: dict[str, Any] = {k: v for k, v in info_dict.items() if k != "additional_information"}
            for key, value in additional_information.items():
                merged.setdefault(key, value)
            info_dict = merged

        span_len = int(input_ids.shape[0])
        if span_len <= 0:
            embeds = input_embeds if input_embeds is not None else self.embed_input_ids(input_ids)
            return input_ids, embeds, {}

        if span_len > 1:
            return self._preprocess_prefill(input_ids, info_dict, span_len)
        return self._preprocess_decode(input_ids, info_dict)

    def _prefill_chunk(
        self,
        prompt_embeds_buf: torch.Tensor,
        start: int,
        span_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Slice ``span_len`` rows out of the pinned CPU prompt-embed buffer."""
        total = int(prompt_embeds_buf.shape[0])
        begin = max(0, min(start, total))
        end = max(0, min(start + span_len, total))
        take = prompt_embeds_buf[begin:end]
        if int(take.shape[0]) < span_len:
            pad_rows = span_len - int(take.shape[0])
            pad_embed = self.embed_input_ids(
                torch.tensor([self._pad_token_id], device=device, dtype=torch.long)
            ).reshape(1, -1)
            take = torch.cat([take.to(device=device, dtype=torch.bfloat16), pad_embed.expand(pad_rows, -1)], dim=0)
            return take.to(dtype=torch.bfloat16)
        return take.to(device=device, dtype=torch.bfloat16, non_blocking=True)

    def _preprocess_prefill(
        self,
        input_ids: torch.Tensor,
        info_dict: dict[str, Any],
        span_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        device = input_ids.device
        embed = info_dict.get("embed", {})
        prompt_embeds_buf = embed.get("prefill")
        is_first_chunk = not isinstance(prompt_embeds_buf, torch.Tensor) or prompt_embeds_buf.ndim != 2

        if is_first_chunk:
            if bool(info_dict.get("audio8_structured_voice_clone", False)):
                prompt_embeds = self._build_voice_clone_prefill_embeds(info_dict)
            else:
                prompt_embeds = self.embed_input_ids(input_ids.reshape(1, -1).to(torch.long)).squeeze(0)
            prompt_embeds_buf = prompt_embeds.detach().to("cpu", dtype=torch.bfloat16).contiguous()
            if not prompt_embeds_buf.is_pinned():
                prompt_embeds_buf = prompt_embeds_buf.pin_memory()
            offset = 0
        else:
            offset = int(info_dict.get("meta", {}).get("prefill_offset", 0) or 0)

        total_prompt_len = int(prompt_embeds_buf.shape[0])
        chunk = self._prefill_chunk(prompt_embeds_buf, offset, span_len, device)
        next_offset = min(offset + span_len, total_prompt_len)

        # No codes are emitted during prefill; a zero frame per position keeps
        # the codes buffer aligned with the hidden-state span.
        zeros = torch.zeros((chunk.shape[0], self._num_codebooks), device=device, dtype=torch.long)
        info_update = {
            "embed": {"prefill": prompt_embeds_buf if next_offset < total_prompt_len else None},
            "meta": {"prefill_offset": next_offset},
            "codes": {"audio": zeros},
        }
        return torch.full_like(input_ids, self._pad_token_id), chunk, info_update

    def _preprocess_decode(
        self,
        input_ids: torch.Tensor,
        info_dict: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        device = input_ids.device
        token_embed = self.embed_input_ids(input_ids.reshape(1, 1).to(torch.long)).to(
            device=device, dtype=torch.bfloat16
        )
        last_hidden = info_dict.get("hidden_states", {}).get("last")
        if not isinstance(last_hidden, torch.Tensor):
            # First decode step right after prefill: no Fast AR input yet.
            logger.warning("Audio8 TTS preprocess: hidden_states.last missing; emitting text-only embedding")
            return input_ids, token_embed.reshape(1, -1), {}

        # Codebook embeddings are added in talker_mtp, using this step's Fast AR
        # output; adding them here would use the previous step's codes.
        info_update = {
            "mtp_inputs": (
                last_hidden.to(device=device, dtype=torch.bfloat16).reshape(1, -1),
                torch.zeros(1, self.text_config.hidden_size, device=device, dtype=torch.bfloat16),
            ),
        }
        return input_ids, token_embed.reshape(1, -1), info_update

    def postprocess(self, hidden_states: torch.Tensor, **_: Any) -> dict[str, Any]:
        if hidden_states.numel() == 0:
            return {}
        return {"hidden_states": {"last": hidden_states[-1, :].detach().contiguous()}}

    # -------------------- voice cloning --------------------

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        return self._tokenizer

    def _codec_kwargs(self) -> dict[str, int]:
        return {
            "post_n_layer": int(self.config.codec_post_n_layer),
            "post_n_head": int(self.config.codec_post_n_head),
            "post_n_local_heads": int(self.config.codec_post_n_local_heads),
            "post_intermediate_size": int(self.config.codec_post_intermediate_size),
        }

    def _build_voice_clone_prefill_embeds(self, info_dict: dict[str, Any]) -> torch.Tensor:
        """Encode the reference audio, build the clone prompt, embed the codes."""
        text = info_dict.get("text")
        ref_text = info_dict.get("ref_text")
        if not isinstance(text, str) or not isinstance(ref_text, str):
            raise ValueError("Audio8 TTS voice cloning requires string 'text' and 'ref_text'")

        device = self.codebook_embeddings.weight.device
        cache_key = None
        voice_name = info_dict.get("voice_name")
        if isinstance(voice_name, str) and voice_name:
            cache_key = self._speaker_cache.make_cache_key(
                voice_name,
                model_type="audio8_tts",
                created_at=int(info_dict.get("voice_created_at") or 0),
            )
            cached = self._speaker_cache.get(cache_key)
            if cached is not None:
                logger.debug("Speaker cache HIT for Audio8 TTS speaker '%s'", voice_name)
                return self._embed_voice_clone_prompt(
                    text, ref_text, cached["ref_codes_fq"].to(device=device, dtype=torch.long)
                )

        ref_audio_sr = info_dict.get("ref_audio_sr")
        if not isinstance(ref_audio_sr, int):
            raise ValueError("Audio8 TTS voice cloning requires integer 'ref_audio_sr'")
        ref_audio_wav = info_dict.get("ref_audio_wav")
        if ref_audio_wav is None:
            raise ValueError("Audio8 TTS voice cloning requires 'ref_audio_wav'")
        if not isinstance(ref_audio_wav, torch.Tensor):
            ref_audio_wav = torch.from_numpy(np.asarray(ref_audio_wav, dtype=np.float32))

        ref_codes_fq = encode_reference_audio_codes(
            self.model_path,
            ref_audio_wav,
            ref_audio_sr,
            device=device,
            **self._codec_kwargs(),
        )
        if cache_key is not None:
            self._speaker_cache.put(cache_key, {"ref_codes_fq": ref_codes_fq.detach().cpu()})
            logger.debug("Speaker cache STORE for Audio8 TTS speaker '%s'", voice_name)
        return self._embed_voice_clone_prompt(text, ref_text, ref_codes_fq)

    def _embed_voice_clone_prompt(
        self,
        text: str,
        ref_text: str,
        ref_codes_fq: torch.Tensor,
    ) -> torch.Tensor:
        """Embed the clone prompt, summing codebooks over the reference frames.

        Args:
            ref_codes_fq: ``[frames, num_codebooks]`` reference codec codes.
        """
        device = self.codebook_embeddings.weight.device
        ref_codes_fq = ref_codes_fq.to(device=device, dtype=torch.long)
        semantic_token_ids = (ref_codes_fq[:, 0] + self._semantic_begin_id).tolist()
        prompt_ids, ref_start, _, _ = build_voice_clone_prompt_ids(
            self._get_tokenizer(), text, ref_text, semantic_token_ids
        )
        prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        embeds = self.embed_input_ids(prompt.unsqueeze(0)).squeeze(0).to(dtype=torch.bfloat16)

        frames = int(ref_codes_fq.shape[0])
        if frames <= 0:
            return embeds
        codebooks = min(int(ref_codes_fq.shape[1]), self._num_codebooks)
        offsets = (torch.arange(codebooks, device=device, dtype=torch.long) * self._codebook_size).unsqueeze(0)
        codes = ref_codes_fq[:, :codebooks].clamp(min=0, max=self._codebook_size - 1) + offsets
        codebook_sum = self.codebook_embeddings(codes).sum(dim=1).to(dtype=embeds.dtype)

        result = embeds.clone()
        # Audio8 TTS sums the embeddings directly -- no 1/sqrt(Q+1) rescale.
        result[ref_start : ref_start + frames] += codebook_sum
        return result.to(dtype=torch.bfloat16)

    # -------------------- GPU-side Fast AR fast path --------------------

    @torch.inference_mode()
    def talker_mtp(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        text_step: torch.Tensor,
        seed: int | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the Fast AR and fold its codes into this step's input embedding.

        Returns:
            ``(inputs_embeds, audio_codes)`` where ``audio_codes`` is
            ``[B, num_codebooks]``.
        """
        del text_step
        bsz = int(input_ids.shape[0])
        device = input_embeds.device

        token_ids = input_ids.reshape(bsz).to(dtype=torch.long, device=device)
        past_hidden = last_talker_hidden.reshape(bsz, -1).to(dtype=torch.bfloat16, device=device)

        temperature = kwargs.get("temperature")
        temperature = DEFAULT_TEMPERATURE if temperature is None else float(temperature)
        do_sample = kwargs.get("do_sample")
        generator = kwargs.get("generator")
        if generator is None and seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))

        _top_k = kwargs.get("top_k")
        _top_p = kwargs.get("top_p")
        audio_codes = self.fast_ar(
            slow_ar_hidden=past_hidden,
            semantic_token_id=token_ids,
            do_sample=(temperature >= SAMPLING_EPS) if do_sample is None else bool(do_sample),
            temperature=temperature,
            top_k=int(DEFAULT_TOP_K if _top_k is None else _top_k),
            top_p=float(DEFAULT_TOP_P if _top_p is None else _top_p),
            generator=generator,
        )

        embeds = input_embeds.reshape(bsz, -1)
        offsets = (torch.arange(self._num_codebooks, device=device, dtype=torch.long) * self._codebook_size).unsqueeze(
            0
        )
        codebook_sum = self.codebook_embeddings(audio_codes + offsets).sum(dim=1).to(dtype=embeds.dtype)
        is_semantic = (token_ids >= self._semantic_begin_id) & (token_ids <= self._semantic_end_id)
        embeds = torch.where(is_semantic.unsqueeze(-1), embeds + codebook_sum, embeds)
        return embeds, audio_codes.to(dtype=torch.long)

    # -------------------- prompt length estimation --------------------

    @staticmethod
    def estimate_prompt_len_from_additional_information(
        additional_information: dict[str, Any] | None,
        **kwargs: Any,
    ) -> int:
        """Upper-bound the text-only prompt length for placeholder allocation."""
        del kwargs
        info = additional_information or {}
        text = info.get("text", "")
        if isinstance(text, list):
            text = text[0] if text else ""
        # ~1 token per character is a safe bound for CJK; +64 covers the
        # chat-template scaffolding.
        return max(2, len(str(text)) + 64)

    # -------------------- weight loading --------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        text_config = self.text_config
        fast_config = self.fast_ar_config
        remapped = _remap_audio8_tts_weights(
            weights,
            q_size=text_config.num_attention_heads * text_config.head_dim,
            kv_size=text_config.num_key_value_heads * text_config.head_dim,
            fast_q_size=fast_config.num_attention_heads * fast_config.head_dim,
            fast_kv_size=fast_config.num_key_value_heads * fast_config.head_dim,
        )

        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        unexpected: list[str] = []

        for name, loaded_weight in remapped:
            if name == "model.embed_tokens.weight" and self.text_config.tie_word_embeddings:
                lm_key = "lm_head.weight"
                if lm_key in params_dict:
                    param = params_dict[lm_key]
                    loader = getattr(param, "weight_loader", default_weight_loader)
                    loader(param, loaded_weight)
                    loaded_params.add(lm_key)

            handled = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                loader = getattr(param, "weight_loader", default_weight_loader)
                if loader == default_weight_loader:
                    loader(param, loaded_weight)
                else:
                    loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped)
                handled = True
                break
            if handled:
                continue

            if name in params_dict:
                param = params_dict[name]
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, loaded_weight)
                loaded_params.add(name)
            else:
                unexpected.append(name)

        if unexpected:
            raise ValueError(
                f"Audio8 TTS Slow AR received {len(unexpected)} unmapped checkpoint tensors, "
                f"e.g. {sorted(unexpected)[:5]}. The weight remapper is out of sync with the checkpoint."
            )
        missing = sorted(set(params_dict) - loaded_params)
        if missing:
            raise ValueError(f"Audio8 TTS Slow AR is missing weights for {missing[:5]} ({len(missing)} total)")
        logger.info("Loaded %d weights for Audio8TTSSlowARForConditionalGeneration", len(loaded_params))

        # The reference precomputes RoPE in bf16; keeping fp32 cos/sin here
        # shifts the logits enough to trigger early EOS (same failure mode as
        # Fish Speech).
        truncated = 0
        for module in self.modules():
            cache = getattr(module, "cos_sin_cache", None)
            if isinstance(cache, torch.Tensor):
                module.cos_sin_cache = cache.to(torch.bfloat16).to(cache.dtype)
                truncated += 1
        if truncated:
            logger.info("Audio8 TTS: truncated %d RoPE cos_sin_cache buffers to bf16", truncated)

        return loaded_params


__all__ = ["Audio8TTSSlowARForConditionalGeneration"]
