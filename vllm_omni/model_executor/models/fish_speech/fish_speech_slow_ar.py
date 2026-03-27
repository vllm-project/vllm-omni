"""Fish Speech S2 Pro -- Slow AR model (Stage 0).

Uses vLLM's ``Qwen3Model`` as the transformer backbone.  Adds:
  - Multi-codebook input embedding (text + summed codebook embeddings at
    semantic-token positions).
  - Semantic logit masking.
  - Nested Fast AR for residual codebook prediction (``talker_mtp``).
  - ``preprocess`` / ``postprocess`` hooks for vLLM-omni's AR scheduler.

Analogous to ``Qwen3TTSTalkerForConditionalGeneration`` in qwen3_tts.
"""

from __future__ import annotations

import dataclasses
import math
import os
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
from vllm.model_executor.models.qwen3 import Qwen3Model
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .configuration_fish_speech import FishSpeechConfig, FishSpeechFastARConfig, FishSpeechSlowARConfig
from .dac_encoder import _load_dac_codec, encode_reference_audio
from .fish_speech_fast_ar import FishSpeechFastAR

logger = init_logger(__name__)


def _remap_fish_speech_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    n_head: int,
    n_local_heads: int,
    head_dim: int,
    fast_n_head: int,
    fast_n_local_heads: int,
    fast_head_dim: int,
) -> Iterable[tuple[str, torch.Tensor]]:
    """Transform Fish Speech HF weight names/values to Qwen3-compatible format.

    Key transformations:
      - ``wqkv`` → split into ``q_proj``, ``k_proj``, ``v_proj``
      - ``wo`` → ``o_proj``
      - ``w1`` → ``gate_proj``, ``w3`` → ``up_proj``, ``w2`` → ``down_proj``
      - ``attention_norm`` → ``input_layernorm``
      - ``ffn_norm`` → ``post_attention_layernorm``
      - ``text_model.model.embeddings`` → ``model.embed_tokens``
      - ``audio_decoder.*`` → ``fast_ar.*`` with similar transforms
    """
    q_size_text = n_head * head_dim
    kv_size_text = n_local_heads * head_dim
    q_size_fast = fast_n_head * fast_head_dim
    kv_size_fast = fast_n_local_heads * fast_head_dim

    for name, tensor in weights:
        # --- Text model (Slow AR) ---
        if name.startswith("text_model.model."):
            suffix = name[len("text_model.model.") :]

            # Embeddings
            if suffix == "embeddings.weight":
                yield "model.embed_tokens.weight", tensor
                continue

            # Norm
            if suffix == "norm.weight":
                yield "model.norm.weight", tensor
                continue

            # Layer weights
            if suffix.startswith("layers."):
                # layers.{N}.attention.wqkv.weight → split into q/k/v
                if ".attention.wqkv.weight" in suffix:
                    layer_prefix = suffix.split(".attention.wqkv.weight")[0]
                    q = tensor[:q_size_text, :]
                    k = tensor[q_size_text : q_size_text + kv_size_text, :]
                    v = tensor[q_size_text + kv_size_text :, :]
                    yield f"model.{layer_prefix}.self_attn.q_proj.weight", q
                    yield f"model.{layer_prefix}.self_attn.k_proj.weight", k
                    yield f"model.{layer_prefix}.self_attn.v_proj.weight", v
                    continue

                new_suffix = suffix
                new_suffix = new_suffix.replace(".attention.wo.", ".self_attn.o_proj.")
                new_suffix = new_suffix.replace(".attention.q_norm.", ".self_attn.q_norm.")
                new_suffix = new_suffix.replace(".attention.k_norm.", ".self_attn.k_norm.")
                new_suffix = new_suffix.replace(".attention_norm.", ".input_layernorm.")
                new_suffix = new_suffix.replace(".feed_forward.w1.", ".mlp.gate_proj.")
                new_suffix = new_suffix.replace(".feed_forward.w3.", ".mlp.up_proj.")
                new_suffix = new_suffix.replace(".feed_forward.w2.", ".mlp.down_proj.")
                new_suffix = new_suffix.replace(".ffn_norm.", ".post_attention_layernorm.")
                yield f"model.{new_suffix}", tensor
                continue

            # Fallback for any other text_model.model.* weights
            yield f"model.{suffix}", tensor
            continue

        # --- Audio decoder (Fast AR) ---
        if name.startswith("audio_decoder."):
            suffix = name[len("audio_decoder.") :]

            # Codebook embeddings (belongs to the main model, not Fast AR).
            if suffix == "codebook_embeddings.weight":
                yield "codebook_embeddings.weight", tensor
                continue

            # Fast AR embeddings, output, norm.
            if suffix == "embeddings.weight":
                yield "fast_ar.fast_embeddings.weight", tensor
                continue
            if suffix == "output.weight":
                yield "fast_ar.fast_output.weight", tensor
                continue
            if suffix == "norm.weight":
                yield "fast_ar.fast_norm.weight", tensor
                continue

            # Fast AR projection in.
            if suffix.startswith("fast_project_in."):
                yield f"fast_ar.fast_project_in.{suffix[len('fast_project_in.') :]}", tensor
                continue

            # Fast AR layer weights.
            if suffix.startswith("layers."):
                if ".attention.wqkv.weight" in suffix:
                    layer_prefix = suffix.split(".attention.wqkv.weight")[0]
                    q = tensor[:q_size_fast, :]
                    k = tensor[q_size_fast : q_size_fast + kv_size_fast, :]
                    v = tensor[q_size_fast + kv_size_fast :, :]
                    yield f"fast_ar.model.{layer_prefix}.self_attn.q_proj.weight", q
                    yield f"fast_ar.model.{layer_prefix}.self_attn.k_proj.weight", k
                    yield f"fast_ar.model.{layer_prefix}.self_attn.v_proj.weight", v
                    continue

                new_suffix = suffix
                new_suffix = new_suffix.replace(".attention.wo.", ".self_attn.o_proj.")
                new_suffix = new_suffix.replace(".attention.q_norm.", ".self_attn.q_norm.")
                new_suffix = new_suffix.replace(".attention.k_norm.", ".self_attn.k_norm.")
                new_suffix = new_suffix.replace(".attention_norm.", ".input_layernorm.")
                new_suffix = new_suffix.replace(".feed_forward.w1.", ".mlp.gate_proj.")
                new_suffix = new_suffix.replace(".feed_forward.w3.", ".mlp.up_proj.")
                new_suffix = new_suffix.replace(".feed_forward.w2.", ".mlp.down_proj.")
                new_suffix = new_suffix.replace(".ffn_norm.", ".post_attention_layernorm.")
                yield f"fast_ar.model.{new_suffix}", tensor
                continue

            yield f"fast_ar.{suffix}", tensor
            continue

        # Pass through any other weights.
        yield name, tensor


class FishSpeechSlowARForConditionalGeneration(nn.Module):
    """vLLM-AR Slow AR model for Fish Speech S2 Pro.

    Stage 0: text → semantic tokens (+ residual codebook codes via Fast AR).
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        config: FishSpeechConfig = vllm_config.model_config.hf_config  # type: ignore[assignment]
        self.config = config
        self.text_config: FishSpeechSlowARConfig = config.text_config
        self.fast_ar_config: FishSpeechFastARConfig = config.audio_decoder_config

        self._semantic_begin_id = int(config.semantic_start_token_id)
        self._semantic_end_id = int(config.semantic_end_token_id)
        self._audio_pad_token_id = int(config.audio_pad_token_id)
        self._codebook_size = int(self.text_config.codebook_size)
        self._num_codebooks = int(self.text_config.num_codebooks)

        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = True
        self.mtp_hidden_size = int(self.text_config.hidden_size)
        self.talker_mtp_output_key = "audio_codes"
        self.gpu_resident_buffer_keys: set[str] = {"last_slow_ar_hidden"}

        # Qwen3 transformer backbone.
        self.model = Qwen3Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))

        # Fish Speech uses interleaved (GPT-J) RoPE, not NeoX style.
        # vLLM's Qwen3Attention defaults to NeoX (is_neox_style=True).
        # Replace with interleaved RoPE to match training.
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

        # Multi-codebook embedding table: codebook_size * num_codebooks entries.
        self.codebook_embeddings = nn.Embedding(
            self._codebook_size * self._num_codebooks,
            self.text_config.hidden_size,
        )

        # Fast AR (residual codebook predictor).
        predictor_compilation = dataclasses.replace(vllm_config.compilation_config)
        predictor_compilation.static_forward_context = {}
        self._fast_ar_vllm_config = dataclasses.replace(vllm_config, compilation_config=predictor_compilation)
        from vllm.config.vllm import set_current_vllm_config as _set_cfg

        with _set_cfg(self._fast_ar_vllm_config):
            self.fast_ar = FishSpeechFastAR(
                vllm_config=self._fast_ar_vllm_config,
                config=self.fast_ar_config,
                slow_ar_config=self.text_config,
                prefix="fast_ar",
            )

        # Constant logit mask: allow only semantic tokens + im_end.
        vocab = int(self.text_config.vocab_size)
        semantic_mask = torch.zeros((vocab,), dtype=torch.bool)
        lo = self._semantic_begin_id
        hi = min(self._semantic_end_id + 1, vocab)
        if hi > lo:
            semantic_mask[lo:hi] = True
        # Also allow <|im_end|> (token 151645 in Qwen3 tokeniser).
        im_end_id = 151645
        if im_end_id < vocab:
            semantic_mask[im_end_id] = True
        self.register_buffer("_semantic_allowed_mask", semantic_mask, persistent=False)

        # Tokeniser (lazy).
        self._tokenizer = None

    def _fix_rope_style(self) -> None:
        """Replace NeoX-style RoPE with interleaved (GPT-J) style.

        Fish Speech was trained with interleaved RoPE (complex-number pairs),
        but vLLM's Qwen3Attention defaults to NeoX style.  We rebuild the
        rotary embedding with ``is_neox_style=False`` for each attention layer.
        """
        from vllm.model_executor.layers.rotary_embedding import get_rope

        for layer in self.model.layers:
            attn = layer.self_attn
            # Extract parameters from the existing RoPE to rebuild it.
            head_dim = attn.head_dim
            max_position = self.text_config.max_position_embeddings
            rope_params = getattr(self.text_config, "rope_scaling", None) or {}
            rope_params.setdefault("rope_theta", getattr(self.text_config, "rope_theta", 1000000.0))
            attn.rotary_emb = get_rope(
                head_size=head_dim,
                max_position=max_position,
                is_neox_style=False,
                rope_parameters=rope_params,
            )
        logger.info("Fixed RoPE style to interleaved (GPT-J) for %d layers", len(self.model.layers))

    # -------------------- vLLM required hooks --------------------

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

        # Mask to semantic tokens + im_end only.
        logits = logits.masked_fill(~self._semantic_allowed_mask, float("-inf"))
        return logits

    # -------------------- Omni multimodal output plumbing --------------------

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
            ac = info.get("audio_codes")
            if isinstance(ac, torch.Tensor):
                audio_codes_list.append(ac)

        if not audio_codes_list:
            logger.debug("make_omni_output: no audio_codes found in info_dicts (len=%d)", len(info_dicts))
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})

        audio_codes = torch.cat(audio_codes_list, dim=0)
        span_len = int(audio_codes.shape[0])
        hidden = hidden[:span_len]
        mm: dict[str, torch.Tensor] = {"audio_codes": audio_codes}
        return OmniOutput(text_hidden_states=hidden, multimodal_outputs=mm)

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
            for k, v in additional_information.items():
                merged.setdefault(k, v)
            info_dict = merged

        span_len = int(input_ids.shape[0])
        if span_len <= 0:
            return input_ids, input_embeds if input_embeds is not None else self.embed_input_ids(input_ids), {}

        if span_len > 1:
            # --- Prefill ---
            prompt_embeds_buf = info_dict.get("slow_ar_prompt_embeds")
            is_first_prefill = not isinstance(prompt_embeds_buf, torch.Tensor) or prompt_embeds_buf.ndim != 2
            dev = input_ids.device

            if is_first_prefill:
                if bool(info_dict.get("fish_structured_voice_clone", False)):
                    prompt_embeds = self._build_structured_voice_clone_prefill_embeds(info_dict)
                else:
                    prompt_embeds = self._build_prefill_embeds(input_ids, info_dict)
                prompt_embeds_buf = prompt_embeds.detach().to("cpu").contiguous()
                if not prompt_embeds_buf.is_pinned():
                    prompt_embeds_buf = prompt_embeds_buf.pin_memory()
                total_prompt_len = int(prompt_embeds_buf.shape[0])
                next_offset = min(span_len, total_prompt_len)

                info_update: dict[str, Any] = {
                    "slow_ar_prompt_embeds": prompt_embeds_buf if next_offset < total_prompt_len else None,
                    "prefill_offset": next_offset,
                }

                take = prompt_embeds_buf[:span_len]
                if int(take.shape[0]) < span_len:
                    pad_n = span_len - int(take.shape[0])
                    pad_embed = self.embed_input_ids(
                        torch.tensor([self._audio_pad_token_id], device=dev, dtype=torch.long)
                    ).reshape(1, -1)
                    take = torch.cat([take, pad_embed.expand(pad_n, -1)], dim=0)
                prompt_embeds = take.to(device=dev, dtype=torch.bfloat16, non_blocking=True)

                zeros = torch.zeros(
                    (prompt_embeds.shape[0], self._num_codebooks),
                    device=dev,
                    dtype=torch.long,
                )
                info_update["audio_codes"] = zeros

                input_ids_out = input_ids.clone()
                input_ids_out[:] = self._audio_pad_token_id
                return input_ids_out, prompt_embeds, info_update

            else:
                # Subsequent prefill chunk.
                offset = int(info_dict.get("prefill_offset", 0) or 0)
                total_prompt_len = int(prompt_embeds_buf.shape[0])
                s = max(0, min(offset, total_prompt_len))
                e = max(0, min(offset + span_len, total_prompt_len))
                take = prompt_embeds_buf[s:e]
                if int(take.shape[0]) < span_len:
                    pad_n = span_len - int(take.shape[0])
                    pad_embed = self.embed_input_ids(
                        torch.tensor([self._audio_pad_token_id], device=dev, dtype=torch.long)
                    ).reshape(1, -1)
                    take = torch.cat([take, pad_embed.expand(pad_n, -1)], dim=0)
                prompt_embeds = take.to(device=dev, dtype=torch.bfloat16, non_blocking=True)
                next_offset = offset + span_len

                zeros = torch.zeros((prompt_embeds.shape[0], self._num_codebooks), device=dev, dtype=torch.long)
                return (
                    input_ids.clone().fill_(self._audio_pad_token_id),
                    prompt_embeds,
                    {
                        "slow_ar_prompt_embeds": prompt_embeds_buf if next_offset < total_prompt_len else None,
                        "prefill_offset": next_offset,
                        "audio_codes": zeros,
                    },
                )

        # --- Decode: span_len == 1 ---
        dev = input_ids.device

        last_hidden = info_dict.get("last_slow_ar_hidden")
        if not isinstance(last_hidden, torch.Tensor):
            # First decode step after prefill -- just embed the token directly.
            logger.warning(
                "preprocess decode: last_slow_ar_hidden not found (keys=%s), "
                "returning plain embed (mtp_inputs will NOT be set)",
                list(info_dict.keys()),
            )
            embeds = self.embed_input_ids(input_ids.reshape(1, 1).to(torch.long)).reshape(1, -1)
            return input_ids, embeds.to(dtype=torch.bfloat16), {}

        token_embed = self.embed_input_ids(input_ids.reshape(1, 1).to(torch.long)).to(
            device=dev, dtype=torch.bfloat16
        )  # [1, 1, H]

        # Codebook embeddings are added by talker_mtp (using the CURRENT step's
        # FastAR hidden state) instead of here (which would use the PREVIOUS step's
        # codes, one step too old).  Preprocess always returns plain text embed.
        inputs_embeds_out = token_embed.reshape(1, -1)

        info_update = {
            "mtp_inputs": (
                last_hidden.to(device=dev, dtype=torch.bfloat16).reshape(1, -1),
                torch.zeros(1, self.text_config.hidden_size, device=dev, dtype=torch.bfloat16),
            ),
        }
        return input_ids, inputs_embeds_out, info_update

    def postprocess(self, hidden_states: torch.Tensor, **_: Any) -> dict[str, Any]:
        if hidden_states.numel() == 0:
            logger.debug("postprocess: empty hidden_states")
            return {}
        last = hidden_states[-1, :].detach().contiguous()
        logger.debug("postprocess: saved last_slow_ar_hidden shape=%s", tuple(last.shape))
        return {"last_slow_ar_hidden": last}

    # -------------------- prompt construction --------------------

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        return self._tokenizer

    def _build_prefill_embeds(
        self,
        input_ids: torch.Tensor,
        info_dict: dict[str, Any],
    ) -> torch.Tensor:
        """Build prefill embeddings, adding codebook embeddings at semantic positions.

        For text-only prefill (no reference audio), this is just embed_input_ids.
        For voice cloning, reference codes are embedded with codebook offsets.
        """
        dev = input_ids.device
        # Basic text embeddings.
        base_embeds = self.embed_input_ids(input_ids.reshape(1, -1).to(torch.long))  # [1, T, H]

        # Check for reference codebook codes (for voice cloning).
        ref_codes = info_dict.get("ref_codes")
        if not isinstance(ref_codes, torch.Tensor) or ref_codes.numel() == 0:
            return base_embeds.squeeze(0).to(dtype=torch.bfloat16)

        # ref_codes: [T_ref, num_codebooks] -- codebook codes for reference audio positions.
        ref_codes = ref_codes.to(device=dev, dtype=torch.long)
        ref_positions = info_dict.get("ref_positions")
        if not isinstance(ref_positions, torch.Tensor):
            return base_embeds.squeeze(0).to(dtype=torch.bfloat16)

        ref_positions = ref_positions.to(device=dev, dtype=torch.long).reshape(-1)
        seq_len = int(input_ids.shape[0])
        codebook_sum = torch.zeros_like(base_embeds)  # [1, T, H]

        for pos_idx in range(int(ref_positions.shape[0])):
            pos = int(ref_positions[pos_idx].item())
            if pos < 0 or pos >= seq_len:
                continue
            for cb_idx in range(min(int(ref_codes.shape[1]), self._num_codebooks)):
                code = ref_codes[pos_idx, cb_idx].clamp(min=0)
                code_with_offset = code + cb_idx * self._codebook_size
                emb = self.codebook_embeddings(code_with_offset.unsqueeze(0))
                codebook_sum[0, pos, :] += emb.squeeze(0).to(dtype=base_embeds.dtype)

        result = base_embeds + codebook_sum
        return result.squeeze(0).to(dtype=torch.bfloat16)

    def _build_structured_voice_clone_prefill_embeds(self, info_dict: dict[str, Any]) -> torch.Tensor:
        tokenizer = self._get_tokenizer()
        ref_text = info_dict.get("ref_text")
        text = info_dict.get("text")
        ref_audio_path = info_dict.get("ref_audio_path")
        ref_audio_sr = info_dict.get("ref_audio_sr")
        if not isinstance(ref_text, str) or not isinstance(text, str):
            raise ValueError("Fish Speech structured voice clone requires string text and ref_text")
        if not isinstance(ref_audio_path, str) or not ref_audio_path:
            raise ValueError("Fish Speech structured voice clone requires ref_audio_path")
        if not isinstance(ref_audio_sr, int):
            raise ValueError("Fish Speech structured voice clone requires integer ref_audio_sr")

        ref_audio_wav = np.load(ref_audio_path)
        os.remove(ref_audio_path)

        semantic_token_ids = encode_reference_audio(
            self.model_path,
            ref_audio_wav,
            ref_audio_sr,
            device=self.codebook_embeddings.weight.device,
        )
        audio_start_id = tokenizer.encode("<|audio_start|>", add_special_tokens=False)
        audio_end_id = tokenizer.encode("<|audio_end|>", add_special_tokens=False)
        prefix_ids = tokenizer.encode(f"<|speaker:0|>{ref_text}", add_special_tokens=False)
        im_start = tokenizer.encode("<|im_start|>", add_special_tokens=False)
        im_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        system_tag = tokenizer.encode("system\n", add_special_tokens=False)
        newline = tokenizer.encode("\n", add_special_tokens=False)
        system_ids = (
            im_start + system_tag + prefix_ids + audio_start_id + semantic_token_ids + audio_end_id + im_end + newline
        )
        user_text = f"<|speaker:0|>{text}"
        user_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=True,
            add_generation_prompt=True,
        )
        voice_token_id = tokenizer.encode("<|voice|>", add_special_tokens=False)
        prompt_ids = torch.tensor(
            system_ids + user_ids + voice_token_id,
            dtype=torch.long,
            device=self.codebook_embeddings.weight.device,
        )
        return self.embed_input_ids(prompt_ids.unsqueeze(0)).squeeze(0).to(dtype=torch.bfloat16)

    # -------------------- GPU-side MTP fast-path --------------------

    @torch.inference_mode()
    def talker_mtp(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        text_step: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """GPU fast-path: run Fast AR to predict residual codebook codes.

        Returns (inputs_embeds, audio_codes).

        The embedding is: text_embed(token) + sum(codebook_embed(code_i + i * codebook_size))
        where codes come from FastAR(last_talker_hidden).

        This matches the reference Fish Speech inference flow:
        - At step t, the Slow AR embedding includes codes from FastAR(hidden_{t-1})
        - last_talker_hidden IS hidden_{t-1} (from postprocess of the previous step)
        - Preprocess provides the plain text embed; we add codebook embeddings here
        """
        bsz = int(input_ids.shape[0])
        dev = input_embeds.device

        input_ids = input_ids.reshape(bsz, 1).to(dtype=torch.long, device=dev)
        past_hidden = last_talker_hidden.reshape(bsz, -1).to(dtype=torch.bfloat16, device=dev)

        # Run Fast AR to predict all num_codebooks codes.
        audio_codes = self.fast_ar(
            slow_ar_hidden=past_hidden,
            semantic_token_id=input_ids.reshape(bsz),
            do_sample=True,
            temperature=0.8,
            top_k=30,
            top_p=0.9,
        )  # [B, num_codebooks]

        # Add codebook embeddings to the input embedding (from preprocess).
        # This ensures the Slow AR sees codes from FastAR(hidden_{t-1}).
        inputs_embeds_out = input_embeds.reshape(bsz, -1).clone()

        semantic_mask = (input_ids[:, 0] >= self._semantic_begin_id) & (input_ids[:, 0] <= self._semantic_end_id)
        if semantic_mask.any():
            semantic_codes = audio_codes[semantic_mask].clamp(min=0)
            offsets = (
                torch.arange(self._num_codebooks, device=dev, dtype=semantic_codes.dtype) * self._codebook_size
            ).unsqueeze(0)
            codebook_sum = self.codebook_embeddings(semantic_codes + offsets).sum(dim=1).to(dtype=torch.bfloat16)

            # Normalize by sqrt(num_codebooks + 1) as in the reference model
            # (scale_codebook_embeddings=True for fish_qwen3_omni).
            inputs_embeds_out[semantic_mask] = (inputs_embeds_out[semantic_mask] + codebook_sum) / math.sqrt(
                self._num_codebooks + 1
            )

        return inputs_embeds_out, audio_codes.to(dtype=torch.long)

    # -------------------- Prompt length estimation --------------------

    @staticmethod
    def estimate_prompt_len_from_additional_information(
        additional_information: dict[str, Any] | None,
        **kwargs: Any,
    ) -> int:
        """Estimate prompt length for placeholder allocation."""
        info = additional_information or {}
        text = info.get("text", [""])[0] if isinstance(info.get("text"), list) else info.get("text", "")
        # Conservative estimate: tokenize text length + overhead.
        return max(2, len(str(text)) // 2 + 64)

    # -------------------- Weight loading --------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with Fish Speech → Qwen3 format transformation.

        Transforms weight names (wqkv → q/k/v split, w1/w2/w3 → gate/up/down,
        etc.) and routes to the correct sub-modules.
        """
        n_head = self.text_config.num_attention_heads
        n_local_heads = self.text_config.num_key_value_heads
        head_dim = self.text_config.head_dim
        fast_n_head = self.fast_ar_config.num_attention_heads
        fast_n_local_heads = self.fast_ar_config.num_key_value_heads
        fast_head_dim = self.fast_ar_config.head_dim

        remapped = _remap_fish_speech_weights(
            weights,
            n_head,
            n_local_heads,
            head_dim,
            fast_n_head,
            fast_n_local_heads,
            fast_head_dim,
        )

        # Qwen3Model uses stacked_params_mapping for q/k/v → qkv_proj
        # and gate/up → gate_up_proj.  Feed the remapped weights through
        # the standard Qwen3 loading path.
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in remapped:
            if "rotary_emb.inv_freq" in name:
                continue

            # Handle tied embeddings → lm_head.
            if name == "model.embed_tokens.weight" and self.text_config.tie_word_embeddings:
                # Also load into lm_head if present.
                lm_key = "lm_head.weight"
                if lm_key in params_dict:
                    p = params_dict[lm_key]
                    wl = getattr(p, "weight_loader", default_weight_loader)
                    wl(p, loaded_weight)
                    loaded_params.add(lm_key)

            # Try stacked params mapping (q/k/v → qkv_proj, gate/up → gate_up_proj).
            handled = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped)
                handled = True
                break

            if handled:
                continue

            # Direct parameter mapping.
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        logger.info("Loaded %d weights for FishSpeechSlowARForConditionalGeneration", len(loaded_params))

        # Truncate RoPE cos/sin caches to bf16 precision to match training.
        # Without this, f32 RoPE values cause logit divergence and premature EOS.
        truncated = 0
        for module in self.modules():
            if hasattr(module, "cos_sin_cache") and isinstance(module.cos_sin_cache, torch.Tensor):
                cache = module.cos_sin_cache
                module.cos_sin_cache = cache.to(torch.bfloat16).to(cache.dtype)
                truncated += 1
        if truncated:
            logger.info("Truncated %d RoPE cos_sin_cache buffers to bf16 precision", truncated)

        try:
            self.fast_ar.warmup_compile(
                device=self.codebook_embeddings.weight.device,
                dtype=torch.bfloat16,
                batch_sizes=(1,),
            )
        except Exception as exc:
            logger.warning("Fish Speech Fast AR compile warmup failed: %s", exc)

        codec_device = self.codebook_embeddings.weight.device
        _load_dac_codec(
            self.model_path,
            device=codec_device,
            dtype=torch.float32,
        )

        return loaded_params
