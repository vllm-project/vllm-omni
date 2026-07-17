"""Moshi TTS Talker — Stage 0 model for vLLM-Omni.

A text-to-speech analogue of ``MoshiMainTransformerForConditionalGeneration``.
Follows the qwen3_tts talker schema: ``has_preprocess`` + ``has_postprocess`` +
``have_multimodal_outputs``, with a GPU ``talker_mtp`` fast-path for the depth
decoder.

Differences from S2S:
  - **Single output stream**: all ``n_q`` codebooks come from the depth decoder;
    there is no user-audio input stream.
  - **State machine driven text**: the text channel is *forced* each step based
    on the DSM state machine (see ``tts_state_machine.py``), not sampled freely.
  - **Audio delay**: audio codes are emitted ``audio_delay_steps`` after the
    corresponding text is consumed.
  - **Speaker conditioning**: audio prefix or cross-attention, depending on the checkpoint.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix
from vllm.sequence import IntermediateTensors

from vllm_omni.data_entry_keys import OmniPayload
from vllm_omni.model_executor.models.output_templates import OmniOutput

from .configuration_moshi import MoshiDepthConfig, MoshiMainConfig, MoshiVLLMConfig
from .delay_pattern import DelayPattern
from .moshi_cross_attention import MoshiCrossAttentionBlock, MoshiTransformerLayer
from .moshi_depth_decoder import MoshiDepthDecoder, _MoshiRMSNorm
from .tts_state_machine import State, StateMachine, TokenIds, script_to_entries
from .weight_remapping import filter_depth_decoder_weights, remap_moshi_weights

logger = init_logger(__name__)

import os as _os

# Set MOSHI_TTS_DBG=1 to emit per-step traces aligned with the moshi/
# reference implementation for side-by-side diffing.
_TTS_DBG = _os.environ.get("MOSHI_TTS_DBG", "0") == "1"
_TTS_DBG_FWD_STEP = 0  # global step counter for _tts_dbg_print_input (matches lm.py)


def _tts_dbg_print_state(tag: str, step: int, sampled: int, forced: int, state: Any) -> None:
    if not _TTS_DBG:
        return
    print(
        f"[TTS-DBG/{tag} step={step:4d} batch=0] "
        f"sampled={sampled:5d} forced={forced:5d} "
        f"q={len(state.queued):2d} fp={state.forced_padding} rp={state.remaining_padding} "
        f"end={state.end_step} entries={len(state.entries):2d}",
        flush=True,
    )


def _tts_dbg_print_consume(
    tag: str,
    step: int,
    *,
    sampled: int,
    rp_before: int,
    fp_before: int,
    queued_before: int,
    entries_before: int,
    new_word_id: int,
    end_step_set: bool,
) -> None:
    if not _TTS_DBG:
        return
    sampled_new_word = sampled == new_word_id and queued_before == 0 and fp_before == 0 and rp_before > 0
    cause = "sampled-new_word" if sampled_new_word else "remaining-padding-exhausted"
    drained = "drained" if end_step_set else "word"
    print(
        f"[TTS-DBG/{tag} CONSUME step={step:4d} batch=0] "
        f"cause={cause:28s} kind={drained} "
        f"sampled={sampled:5d} rp_before={rp_before} fp_before={fp_before} "
        f"queued_before={queued_before} entries_before={entries_before}",
        flush=True,
    )


def _tts_dbg_print_codes(tag: str, step: int, codes: list | torch.Tensor) -> None:
    if not _TTS_DBG:
        return
    if isinstance(codes, torch.Tensor):
        codes = codes.reshape(-1).tolist()
    print(f"[TTS-DBG/{tag} step={step:4d} batch=0] codes[:8]={codes[:8]}", flush=True)


def _tts_dbg_print_input(tag: str, t: torch.Tensor, *, has_cross: bool) -> None:
    global _TTS_DBG_FWD_STEP
    if not _TTS_DBG:
        return
    print(
        f"[TTS-DBG/{tag} input_] step={_TTS_DBG_FWD_STEP:4d} "
        f"shape={tuple(t.shape)} dtype={t.dtype} "
        f"cross={'yes' if has_cross else 'no '}\n{t}",
        flush=True,
    )
    _TTS_DBG_FWD_STEP += 1


def _sin_embedding(ctx: torch.Tensor, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal positional embedding matching moshi/modules/transformer.py:create_sin_embedding.

    Adds position information to the cross-attention context before it is used
    as K/V, as required when ``fuser.cross_attention_pos_emb`` is True.

    Args:
        ctx: ``[B, S, D]`` projected speaker context.
    Returns:
        Positional embedding of the same shape.
    """
    B, S, D = ctx.shape
    half = D // 2
    pos = torch.arange(S, device=ctx.device, dtype=torch.float32).view(1, -1, 1)
    freq = torch.arange(half, device=ctx.device, dtype=torch.float32).view(1, 1, -1)
    phase = pos / (max_period ** (freq / max(half - 1, 1)))
    emb = torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)  # [1, S, D]
    return emb.to(ctx.dtype).expand(B, -1, -1)


class MoshiTTSTalkerForConditionalGeneration(nn.Module):
    """Stage 0: text-driven DSM TTS talker.

    Each decode step:
      1. ``preprocess`` runs the state machine on the sampled text token to
         decide what the LM should see next. When ``second_stream_ahead > 0``
         the token is muxed as ``(second + 1) * card + main``; ``embed_input_ids``
         demuxes it using the two pre-multiplied embedding tables.
      2. ``forward`` runs the main transformer.
      3. ``compute_logits`` emits text logits over ``text_card + 1``.
      4. The sampler picks a text token (new_word / pad only — the padding
         bonus can be applied here).
      5. ``postprocess`` runs the depth decoder to emit all ``n_q`` codebooks
         and stores the last hidden state for the next step.
      6. ``talker_mtp`` is a pass-through that returns the embed built in
         ``preprocess`` and the audio codes computed in ``postprocess``.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        hf_config = vllm_config.model_config.hf_config
        if isinstance(hf_config, MoshiVLLMConfig):
            config = hf_config
        elif isinstance(hf_config, dict):
            config = MoshiVLLMConfig.from_converted_config(hf_config)
        else:
            config = MoshiVLLMConfig.from_hf_config(hf_config)
        self.config = config
        self.main_config: MoshiMainConfig = config.main_config
        self.depth_config: MoshiDepthConfig = config.depth_config

        self._num_codebooks = int(self.main_config.num_codebooks)
        self._n_q = self._num_codebooks
        self._audio_vocab_size = int(self.main_config.audio_vocab_size)
        self._text_card = int(self.main_config.vocab_size) - 1
        # Some models have a small decision head rather than a full text-vocab head
        # (e.g. tts-0.75b-en-public has 5 output classes). lm_head_vocab_size in
        # the converted config captures the actual weight shape when it differs.
        _lm_head_vocab_override = getattr(vllm_config.model_config.hf_config, "lm_head_vocab_size", None)
        self._lm_head_vocab_size = int(_lm_head_vocab_override) if _lm_head_vocab_override else self._text_card

        raw_tts = getattr(hf_config, "tts_config", None) or {}
        if isinstance(raw_tts, dict):
            audio_delay_s = float(raw_tts.get("audio_delay", 0.0))
            second_stream_ahead = int(raw_tts.get("second_stream_ahead", 0))
        else:
            audio_delay_s = 0.0
            second_stream_ahead = 0
        frame_rate = 12.5
        aec = getattr(hf_config, "audio_encoder_config", None)
        if aec is not None:
            fr = getattr(aec, "_frame_rate", None) or getattr(aec, "frame_rate", None)
            if fr:
                frame_rate = float(fr)
        self._frame_rate = frame_rate
        self._audio_delay_steps = int(round(audio_delay_s * frame_rate))

        self._second_stream_ahead = second_stream_ahead

        # --- vLLM-Omni scheduler flags
        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = True
        self.mtp_hidden_size = self.main_config.hidden_size
        self.talker_mtp_output_key = ("codes", "audio")
        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {
            ("hidden_states", "last"),
            ("codes", "audio"),
        }

        original_hf_config = vllm_config.model_config.hf_config
        vllm_config.model_config.hf_config = self.main_config
        try:
            self.model = LlamaModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        finally:
            vllm_config.model_config.hf_config = original_hf_config

        # Replace vllm's RMSNorm with _MoshiRMSNorm when the checkpoint uses
        # rms_norm_f32: vllm casts back to bfloat16 before the weight multiply
        # while moshi/'s rms_norm_f32 keeps everything in float32 through the
        # weight multiply, matching the original model's numerical behaviour.
        if getattr(self.main_config, "norm", None) == "rms_norm_f32":
            from vllm.model_executor.layers.layernorm import RMSNorm as VllmRMSNorm

            eps = getattr(self.main_config, "rms_norm_eps", 1e-8)
            replaced = 0
            for module in self.model.modules():
                for name, child in list(module.named_children()):
                    if isinstance(child, VllmRMSNorm):
                        f32_norm = _MoshiRMSNorm(child.weight.shape[0], eps=eps)
                        f32_norm.weight = child.weight
                        setattr(module, name, f32_norm)
                        replaced += 1
            logger.info("rms_norm_f32: replaced %d VllmRMSNorm → _MoshiRMSNorm (eps=%g)", replaced, eps)

        # moshi/ uses interleaved (GPT-J / is_neox_style=False) RoPE.
        # vllm's LlamaModel defaults to is_neox_style=True, so replace each
        # rotary_emb with a fresh RotaryEmbedding(is_neox_style=False).
        # Creating a new instance (rather than patching the attribute) ensures
        # the CUDA kernel receives is_neox_style=False from the first call.
        rope_theta = float(getattr(self.main_config, "rope_theta", 10000.0))
        rope_patched = 0
        for layer in self.model.layers:
            attn = layer.self_attn
            if hasattr(attn, "rotary_emb"):
                old = attn.rotary_emb
                attn.rotary_emb = RotaryEmbedding(
                    head_size=old.head_size,
                    rotary_dim=old.head_size,
                    max_position_embeddings=old.max_position_embeddings,
                    base=rope_theta,
                    is_neox_style=False,
                    dtype=torch.bfloat16,
                )
                rope_patched += 1
        logger.info("RoPE: replaced %d RotaryEmbedding → is_neox_style=False (GPT-J / interleaved)", rope_patched)

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self._lm_head_vocab_size,
                self.main_config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(self._lm_head_vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        self.audio_embed_tokens = nn.ModuleList(
            [nn.Embedding(self._audio_vocab_size + 1, self.main_config.hidden_size) for _ in range(self._num_codebooks)]
        )

        if self._second_stream_ahead > 0:
            self.embed_tokens_second_stream: nn.Embedding | None = nn.Embedding(
                self._text_card + 1, self.main_config.hidden_size
            )
        else:
            self.embed_tokens_second_stream = None

        # --- Per-codebook delay (independent of audio_delay_steps) ---
        # Kyutai delays: [text_delay, cb0..cbN-1]. Skip text delay at index 0.
        delays = getattr(config, "delays", None)
        if delays and len(delays) > self._num_codebooks:
            codebook_delays = delays[1 : self._num_codebooks + 1]
        elif delays:
            codebook_delays = delays[: self._num_codebooks]
        else:
            codebook_delays = None
        self.delay_pattern = DelayPattern(
            num_codebooks=self._num_codebooks,
            delays=codebook_delays,
            bos_token_id=self._audio_vocab_size,
            pad_token_id=self._audio_vocab_size,
        )

        self.depth_decoder = MoshiDepthDecoder(self.depth_config)
        self.depth_decoder.enable_compile()

        self._token_ids = TokenIds(card=self._text_card + 1)
        self._machine = StateMachine(
            token_ids=self._token_ids,
            second_stream_ahead=self._second_stream_ahead,
            max_padding=8,
            initial_padding=2,
        )

        self._cond_control_embed: torch.Tensor | None = None
        self._cfg_embeds: dict[str, torch.Tensor] = {}
        cond_cfg = getattr(hf_config, "conditioners", None)
        self._has_control_cond = bool(cond_cfg) and isinstance(cond_cfg, dict) and "control" in cond_cfg
        self._has_cfg_cond = bool(cond_cfg) and isinstance(cond_cfg, dict) and "cfg" in cond_cfg
        raw_tts = getattr(hf_config, "tts_config", None) or {}
        self._default_cfg_coef: str = str(raw_tts.get("cfg_coef", "2.0"))
        fuser_cfg = getattr(hf_config, "fuser", None) or {}
        self._cross_attn_pos_emb: bool = bool(fuser_cfg.get("cross_attention_pos_emb", False))
        self._cross_attn_pos_emb_scale: float = float(fuser_cfg.get("cross_attention_pos_emb_scale", 1.0))
        self._cross_attn_wrappers: list[MoshiTransformerLayer] = []
        if config.cross_attention:
            num_heads = self.main_config.num_attention_heads
            head_dim = self.main_config.head_dim
            hidden_size = self.main_config.hidden_size
            for i, layer in enumerate(self.model.layers):
                block = MoshiCrossAttentionBlock(hidden_size, num_heads, head_dim)
                wrapper = MoshiTransformerLayer(layer, block)
                self.model.layers[i] = wrapper
                self._cross_attn_wrappers.append(wrapper)
            self.cross_attn_output_proj = nn.Linear(config.cross_attention_dim, hidden_size, bias=False)
            self.register_buffer(
                "learnt_padding",
                torch.zeros(1, 1, hidden_size),
            )
            logger.info(
                "Cross-attention enabled: %d layers, speaker_dim=%d → hidden=%d",
                len(self._cross_attn_wrappers),
                config.cross_attention_dim,
                hidden_size,
            )
        self._cross_attn_ctx: dict[str, torch.Tensor | None] = {}

        self._states: dict[str, State] = {}
        self._prefixes: dict[str, torch.Tensor | None] = {}
        self._prefix_num_frames: dict[str, int] = {}
        self.__dict__["_mimi_encoder_stage"] = None
        try:
            from vllm_omni.utils.speaker_cache import get_speaker_cache

            self._speaker_cache = get_speaker_cache()
        except ImportError:
            self._speaker_cache = None
        self._last_machine_step: dict[str, int] = {}
        self._last_forced_token: dict[str, int] = {}
        self._current_model_step: dict[str, int] = {}
        self._last_audio_codes: dict[str, torch.Tensor] = {}

        self._stop_token_id: int = self._token_ids.main
        self._force_stop_next: bool = False
        self._active_req_id: str = "default"

    # -------------------- vLLM required hooks --------------------

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if self._second_stream_ahead == 0 or self.embed_tokens_second_stream is None:
            return self.model.embed_input_ids(input_ids)
        # Demux: token = (second + 1) * card + main  (state machine encoding)
        card = self._token_ids.card
        main_ids = input_ids % card
        second_ids = (input_ids // card) - 1
        first_embed = self.model.embed_input_ids(main_ids)
        second_zero = (second_ids < 0)[..., None]
        second_ids = second_ids.clamp(min=0)
        second_embed = self.embed_tokens_second_stream(second_ids)
        return first_embed + torch.where(second_zero, torch.zeros_like(second_embed), second_embed)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor | IntermediateTensors:
        if self._cross_attn_wrappers:
            ctx = self._cross_attn_ctx.get(self._active_req_id)
            if _TTS_DBG:
                _step_dbg = getattr(self, "_dbg_step_idx", "?")
                if ctx is None:
                    print(f"[MOSHI:cross_ctx] step={_step_dbg} ctx=None (cross-attn DISABLED)", flush=True)
                else:
                    print(
                        f"[MOSHI:cross_ctx] step={_step_dbg} shape={tuple(ctx.shape)} norm={ctx.float().norm().item():.4f}",
                        flush=True,
                    )
            for wrapper in self._cross_attn_wrappers:
                wrapper.cross_attention_src = ctx
        if not (_TTS_DBG and inputs_embeds is not None and hasattr(self.model, "layers")):
            return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        _step = getattr(self, "_dbg_step_idx", "?")
        _handles = []
        for _li, _layer in enumerate(self.model.layers):

            def _norm1_hook(m, inp, out, li=_li, s=_step):
                print(f"[MOSHI:norm1_out]    step={s} layer={li} norm1={out}", flush=True)
                if isinstance(out, torch.Tensor):
                    print(f"[MOSHI:norm1_max]    step={s} layer={li} max={out.abs().max().item():.6f}", flush=True)

            _handles.append(_layer.input_layernorm.register_forward_hook(_norm1_hook))
            _qkv = getattr(_layer.self_attn, "qkv_proj", None)
            if _qkv is not None:

                def _in_proj_hook(m, inp, out, li=_li, s=_step):
                    print(f"[MOSHI:in_proj_out]  step={s} layer={li} projected={out}", flush=True)

                _handles.append(_qkv.register_forward_hook(_in_proj_hook))
            _rope = getattr(_layer.self_attn, "rotary_emb", None)
            if _rope is not None:

                def _rope_hook(m, inp, out, li=_li, s=_step):
                    if isinstance(out, (tuple, list)) and len(out) >= 2:
                        print(f"[MOSHI:rope_q]       step={s} layer={li} q={out[0]}", flush=True)
                        print(f"[MOSHI:rope_k]       step={s} layer={li} k={out[1]}", flush=True)

                _handles.append(_rope.register_forward_hook(_rope_hook))

            def _attn_hook(m, inp, out, li=_li, s=_step):
                _h = out[0] if isinstance(out, (tuple, list)) else out
                print(f"[MOSHI:attn_out]     step={s} layer={li} attn={_h}", flush=True)

            _handles.append(_layer.self_attn.register_forward_hook(_attn_hook))

            def _norm2_hook(m, inp, out, li=_li, s=_step):
                print(f"[MOSHI:norm2_out]    step={s} layer={li} norm2={out}", flush=True)

            _handles.append(_layer.post_attention_layernorm.register_forward_hook(_norm2_hook))

            def _ff_hook(m, inp, out, li=_li, s=_step):
                _h = out[0] if isinstance(out, (tuple, list)) else out
                print(f"[MOSHI:ff_update]    step={s} layer={li} ff={_h}", flush=True)

            _handles.append(_layer.mlp.register_forward_hook(_ff_hook))

            def _layer_hook(m, inp, out, li=_li, s=_step):
                _h = out[0] if isinstance(out, (tuple, list)) else out
                print(f"[MOSHI:layer_hidden] step={s} layer={li} h={_h}", flush=True)

            _handles.append(_layer.register_forward_hook(_layer_hook))
        result = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        for _h in _handles:
            _h.remove()
        return result

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        if _TTS_DBG and isinstance(hidden_states, torch.Tensor):
            _tts_dbg_print_input("dst_out", hidden_states, has_cross=bool(self._cross_attn_wrappers))
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is None:
            return None
        if _TTS_DBG:
            _step = getattr(self, "_dbg_step_idx", "?")
            print(f"[MOSHI:txt_logits] step={_step} logits={logits}", flush=True)

        # TTS sampling is constrained to {new_word, pad}. When the state
        # machine is done + audio_delay has elapsed, ``preprocess`` flips
        # ``_force_stop_next`` and we emit ``_stop_token_id`` so vLLM's
        # ``stop_token_ids`` sampling-param terminates the request.
        mask = torch.full_like(logits, float("-inf"))
        if self._force_stop_next:
            mask[..., self._stop_token_id] = 1.0
        else:
            mask[..., self._token_ids.new_word] = logits[..., self._token_ids.new_word]
            mask[..., self._token_ids.pad] = logits[..., self._token_ids.pad]
        return mask

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        hidden = model_outputs
        info_dicts = kwargs.get("model_intermediate_buffer")
        if info_dicts is None:
            info_dicts = kwargs.get("runtime_additional_information") or []
        if "runtime_additional_information" in kwargs and "model_intermediate_buffer" not in kwargs:
            logger.warning_once("runtime_additional_information is deprecated, use model_intermediate_buffer")

        audio_codes_list: list[torch.Tensor] = []
        for info in info_dicts:
            if not isinstance(info, dict):
                continue
            meta = info.get("meta") or {}
            codes = info.get("codes") or {}
            # Skip Mimi during the audio delay window and the prefix override window.
            tts_step = meta.get("tts_step")
            T_frames_info = int(meta.get("prefix_num_frames") or 0)
            skip_until = self._audio_delay_steps + T_frames_info
            if tts_step is not None and int(tts_step) <= skip_until:
                continue
            ac = codes.get("audio")
            if isinstance(ac, torch.Tensor):
                audio_codes_list.append(ac)

        if not audio_codes_list:
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})

        audio_codes = torch.cat(audio_codes_list, dim=0)
        span_len = int(audio_codes.shape[0])
        hidden = hidden[:span_len]
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={"codes": {"audio": audio_codes}},
        )

    def _get_cond_sum(
        self, device: torch.device, dtype: torch.dtype, cfg_coef: str | None = None
    ) -> torch.Tensor | None:
        """Return the (control + cfg) sum embedding if available.

        Args:
            cfg_coef: CFG distillation coefficient as a string (e.g. ``"2.0"``).
                Selects which pre-computed cfg LUT embedding to use.  Falls back
                to ``_default_cfg_coef`` (``"2.0"``), then to the closest available
                value in ``_cfg_embeds``.
        """
        parts: list[torch.Tensor] = []
        if self._cond_control_embed is not None:
            parts.append(self._cond_control_embed.to(device=device, dtype=dtype))
        if self._cfg_embeds:
            key = cfg_coef if cfg_coef is not None else self._default_cfg_coef
            cfg_emb = self._cfg_embeds.get(key, next(iter(self._cfg_embeds.values())))
            parts.append(cfg_emb.to(device=device, dtype=dtype))
        if not parts:
            return None
        return torch.stack(parts).sum(0).view(1, -1)

    def _script_from_info(self, info_dict: dict[str, Any]) -> list[str]:
        """Extract the list of utterances (speaker turns) from additional_information."""
        turns = info_dict.get("turns")
        if isinstance(turns, list) and turns:
            return [str(t) for t in turns]
        text = info_dict.get("text")
        if isinstance(text, list) and text:
            return [str(text[0])]
        if isinstance(text, str) and text:
            return [text]
        raise ValueError("MoshiTTS requires additional_information.text or .turns")

    def _tokenize(self, word: str) -> list[int]:
        """Tokenize a word with the SentencePiece tokenizer shipped with the model."""
        sp = self._get_tokenizer()
        return list(sp.encode(word))

    def _get_tokenizer(self):
        sp = getattr(self, "_sp", None)
        if sp is not None:
            return sp
        import glob
        import os

        import sentencepiece

        candidates = glob.glob(os.path.join(self.model_path, "*.model"))
        if not candidates:
            raise FileNotFoundError(f"No SentencePiece .model file found in {self.model_path}")
        sp = sentencepiece.SentencePieceProcessor()
        sp.Load(candidates[0])
        self._sp = sp
        return sp

    def _ensure_state(self, req_id: str, info_dict: dict[str, Any]) -> State:
        state = self._states.get(req_id)
        if state is not None:
            return state
        script = self._script_from_info(info_dict)
        entries = script_to_entries(
            tokenize=self._tokenize,
            token_ids=self._token_ids,
            frame_rate=self._frame_rate,
            script=script,
            multi_speaker=len(script) > 1,
            padding_between=1,
        )
        state = self._machine.new_state(entries)
        self._states[req_id] = state

        # Initialize the audio prefix for this request, if provided.
        # Expected format in additional_information:
        #   prefix_audio_codes: flat int list of length num_codebooks * T
        #   prefix_num_frames:  T (int or one-element list)
        # Shape encoded as [num_codebooks, T] in row-major order.
        # Speaker cache: avoid re-encoding the same voice WAV on repeated requests.
        cache_key = None
        if self._speaker_cache is not None:
            prefix_wav_key = info_dict.get("prefix_wav_key")
            if isinstance(prefix_wav_key, list) and prefix_wav_key:
                prefix_wav_key = prefix_wav_key[0]
            if isinstance(prefix_wav_key, str) and prefix_wav_key.strip():
                created_at = info_dict.get("voice_created_at", [0])
                if isinstance(created_at, list):
                    created_at = created_at[0] if created_at else 0
                cache_key = self._speaker_cache.make_cache_key(
                    prefix_wav_key.strip(), model_type="moshi_tts_prefix", created_at=int(created_at)
                )
                cached = self._speaker_cache.get(cache_key)
                if cached is not None:
                    info_dict = dict(info_dict)
                    info_dict["prefix_audio_codes"] = cached["codes"]
                    info_dict["prefix_num_frames"] = cached["T"]
                    cache_key = None

        prefix_tensor, T = self._build_prefix_from_info(info_dict)

        if cache_key is not None and prefix_tensor is not None and self._speaker_cache is not None:
            self._speaker_cache.put(
                cache_key,
                {
                    "codes": info_dict.get("_encoded_prefix_codes"),
                    "T": T,
                },
            )

        self._prefixes[req_id] = prefix_tensor
        self._prefix_num_frames[req_id] = T
        # Cross-attention context is built lazily on first _ensure_state call;
        # device/dtype are resolved in preprocess just after this returns.
        return state

    def _load_speaker_embedding(self, source: str) -> torch.Tensor:
        """Load a speaker embedding from a local path or http(s) URL.

        Accepts ``.npy``, ``.safetensors``/``.sft`` or ``.pt``/``.pth`` files.  Returns a float32 tensor
        of shape ``[S, D]``.
        """
        import numpy as np

        def _parse(buf: bytes, name: str) -> np.ndarray:
            if name.endswith(".npy"):
                return np.load(io.BytesIO(buf)).astype("float32")
            if name.endswith((".pt", ".pth")):
                obj = torch.load(io.BytesIO(buf), map_location="cpu")
                if not isinstance(obj, torch.Tensor):
                    raise ValueError(f"Expected a Tensor in {name!r}, got {type(obj)}")
                return obj.float().numpy()
            if name.endswith((".safetensors", ".sft")):
                from safetensors.torch import load as st_load

                tensors = st_load(buf)
                if "speaker_wavs" not in tensors:
                    raise ValueError(f"No 'speaker_wavs' key in {name!r}. Available keys: {list(tensors.keys())}")
                emb = tensors["speaker_wavs"].float()  # [1, D, S] per kyutai convention
                if emb.dim() == 3:
                    emb = emb.squeeze(0)  # [D, S]
                    emb = emb.T  # [S, D]  — match reference's emb.transpose(1,2)
                return emb.numpy()
            raise ValueError(f"Unsupported format {name!r}. Use .safetensors, .npy or .pt")

        if source.startswith(("http://", "https://")):
            import httpx

            resp = httpx.get(source, follow_redirects=True, timeout=60.0)
            resp.raise_for_status()
            arr = _parse(resp.content, source.split("?")[0])
        else:
            with open(source, "rb") as fh:
                arr = _parse(fh.read(), source)

        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError(f"Speaker embedding must be 2-D [S, D], got {arr.shape}")
        return torch.from_numpy(arr)

    def _build_cross_attn_ctx(
        self, info_dict: dict[str, Any], req_id: str, dev: torch.device, dtype: torch.dtype
    ) -> None:
        """Build and cache the cross-attention context for this request.

        Accepts a speaker embedding via ``additional_information``:
          - ``speaker_embedding``: local path, http(s) URL, pre-loaded
            ``[S, D]`` tensor, or nested list.

        Falls back to the learned null-speaker padding when not provided.
        """
        if not self._cross_attn_wrappers:
            return
        spk = info_dict.get("speaker_embedding")
        if spk is not None:
            try:
                if isinstance(spk, str):
                    spk = self._load_speaker_embedding(spk)
                elif isinstance(spk, list):
                    import numpy as np

                    spk = torch.from_numpy(np.asarray(spk, dtype="float32"))
                if isinstance(spk, torch.Tensor):
                    if spk.ndim == 2:
                        spk = spk.unsqueeze(0)  # [1, S, D]
                    spk = spk.to(device=dev, dtype=dtype)
                    real = torch.nn.functional.linear(spk, self.cross_attn_output_proj.weight)
                    _max_speakers = 5  # TTSModel.max_speakers default
                    S = real.shape[1]
                    pad_frames = (_max_speakers - 1) * S
                    pad = self.learnt_padding.to(device=dev, dtype=dtype).expand(1, pad_frames, -1)
                    ctx = torch.cat([real, pad], dim=1)  # [1, max_speakers*S, hidden]
                    if self._cross_attn_pos_emb:
                        ctx = ctx + self._cross_attn_pos_emb_scale * _sin_embedding(ctx)
                else:
                    raise TypeError(f"Unexpected speaker_embedding type: {type(spk)}")
            except Exception as exc:
                logger.warning("Failed to load speaker_embedding, using null speaker: %s", exc)
                ctx = self.learnt_padding.to(device=dev, dtype=dtype)
        else:
            ctx = self.learnt_padding.to(device=dev, dtype=dtype)
        self._cross_attn_ctx[req_id] = ctx

    def _get_encoder_stage(self):
        """Lazily instantiate a MoshiMimiEncoder for voice-prefix encoding."""
        if self.__dict__["_mimi_encoder_stage"] is not None:
            return self.__dict__["_mimi_encoder_stage"]
        try:
            from .moshi_mimi_encoder import MoshiMimiEncoder

            stage = MoshiMimiEncoder(vllm_config=self.vllm_config)
            self.__dict__["_mimi_encoder_stage"] = stage
        except Exception as exc:
            logger.warning("Could not instantiate MoshiMimiEncoder for prefix: %s", exc)
        return self.__dict__["_mimi_encoder_stage"]

    def _is_url(self, s: str) -> bool:
        try:
            u = urlparse(s)
            return u.scheme in ("http", "https", "file")
        except Exception:
            return False

    def _is_probably_base64(self, s: str) -> bool:
        return s.startswith("data:audio") or ("/" not in s and "\\" not in s and len(s) > 256)

    def _decode_base64_to_wav_bytes(self, b64: str) -> bytes:
        if "," in b64 and b64.strip().startswith("data:"):
            b64 = b64.split(",", 1)[1]
        return base64.b64decode(b64)

    def _load_audio_to_np(self, source: str) -> tuple[Any, int]:
        """Load audio from a local path, http(s)/file URL, or base64 data URI."""
        import numpy as np
        import soundfile as sf

        if self._is_url(source):
            from vllm.multimodal.media import MediaConnector

            connector = MediaConnector(allowed_local_media_path="/")
            audio, sr = connector.fetch_audio(source)
        elif self._is_probably_base64(source):
            wav_bytes = self._decode_base64_to_wav_bytes(source)
            audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
        else:
            from vllm.multimodal.media.audio import load_audio

            audio, sr = load_audio(source, sr=None, mono=True)

        if isinstance(audio, np.ndarray) and audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        return np.asarray(audio, dtype=np.float32), int(sr)

    def _encode_wav_as_prefix(self, wav_np, sr: int) -> tuple[list[int], int]:
        """Encode a waveform into flat codebook-major audio prefix codes via
        MoshiMimiEncoder.  Returns ``(flat_codes, T)`` or ``([], 0)`` on failure.
        """
        encoder = self._get_encoder_stage()
        if encoder is None:
            return [], 0
        try:
            out = encoder.forward(
                runtime_additional_information=[
                    {
                        "raw_audio": wav_np,
                        "raw_audio_sample_rate": sr,
                    }
                ]
            )
            frames = out.multimodal_outputs["user_audio_frames"]  # [T, K]
            if frames.numel() == 0:
                return [], 0
            T = int(frames.shape[0])
            return frames.T.reshape(-1).tolist(), T  # codebook-major [K*T]
        except Exception as exc:
            logger.warning("Mimi encoding failed for prefix WAV: %s", exc)
            return [], 0

    def _build_prefix_from_info(self, info_dict: dict[str, Any]) -> tuple[torch.Tensor | None, int]:
        """Encode an audio prefix into a per-step delayed override tensor.

        Accepts either:
        - ``prefix_audio_codes`` + ``prefix_num_frames``: pre-encoded flat int list.
        - ``prefix_wav``: ``[[wav_samples, sr]]`` list (pre-loaded by the serving
          layer) or a local path / http(s)/file URL / base64 URI (offline inference).
          The audio is Mimi-encoded on first call; results are cached via the
          speaker cache when ``prefix_wav_key`` is also present.

        Returns ``(tensor, T)`` where ``tensor`` has shape
        ``[T + max_total_delay, num_codebooks]`` (for audio code override) and
        ``T`` is the raw prefix frame count (for state-machine bypass duration,
        matching moshi/'s text_prefixes deque length = T not T + max_delay).
        """
        flat = info_dict.get("prefix_audio_codes")
        tfield = info_dict.get("prefix_num_frames")
        if isinstance(flat, list) and isinstance(tfield, list) and tfield:
            tfield = tfield[0]

        # Fall back to WAV encoding when no pre-encoded codes are present
        if not (isinstance(flat, list) and flat):
            prefix_wav = info_dict.get("prefix_wav")
            if isinstance(prefix_wav, list) and prefix_wav:
                item = prefix_wav[0] if isinstance(prefix_wav[0], (list, tuple)) else prefix_wav
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    import numpy as np

                    wav_data, sr_val = item
                    flat, tfield = self._encode_wav_as_prefix(np.asarray(wav_data, dtype="float32"), int(sr_val))
                elif isinstance(prefix_wav[0], str) and prefix_wav[0].strip():
                    try:
                        wav_np, sr = self._load_audio_to_np(prefix_wav[0].strip())
                        flat, tfield = self._encode_wav_as_prefix(wav_np, sr)
                    except Exception as exc:
                        logger.warning("Failed to load prefix WAV: %s", exc)
            elif isinstance(prefix_wav, str) and prefix_wav.strip():
                try:
                    wav_np, sr = self._load_audio_to_np(prefix_wav.strip())
                    flat, tfield = self._encode_wav_as_prefix(wav_np, sr)
                except Exception as exc:
                    logger.warning("Failed to load prefix WAV: %s", exc)
            # Stash for the speaker-cache store path in _ensure_state
            if flat:
                info_dict["_encoded_prefix_codes"] = flat

        if not (isinstance(flat, list) and flat) or not isinstance(tfield, int):
            return None, 0
        K = self._num_codebooks
        T = int(tfield)
        if len(flat) != K * T:
            logger.warning(
                "prefix_audio_codes length %d != num_codebooks*T = %d*%d; ignoring prefix.",
                len(flat),
                K,
                T,
            )
            return None, 0
        codes = torch.tensor(flat, dtype=torch.long).reshape(K, T)
        per_cb_delays = self.delay_pattern.delays[:K]
        total_delays = [int(d) + int(self._audio_delay_steps) for d in per_cb_delays]
        max_delay = max(total_delays) if total_delays else 0
        out = torch.full((K, T + max_delay), -2, dtype=torch.long)  # -2 = ungenerated
        for k, d in enumerate(total_delays):
            out[k, d : d + T] = codes[k]
        return out.t().contiguous(), T  # [T + max_delay, K], T

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
            return (
                input_ids,
                input_embeds if input_embeds is not None else self.embed_input_ids(input_ids),
                {},
            )

        dev = input_ids.device
        dtype = torch.bfloat16

        req_id = str(info_dict.get("request_id") or info_dict.get("external_req_id") or "default")
        self._active_req_id = req_id
        state = self._ensure_state(req_id, info_dict)
        if req_id not in self._cross_attn_ctx:
            self._build_cross_attn_ctx(info_dict, req_id, dev, dtype)

        cfg_coef = info_dict.get("cfg_coef")
        if isinstance(cfg_coef, (int, float)):
            cfg_coef = str(cfg_coef)
        cond_sum = self._get_cond_sum(dev, dtype, cfg_coef=cfg_coef)

        is_decode = span_len == 1 and isinstance((info_dict.get("hidden_states") or {}).get("last"), torch.Tensor)

        if not is_decode:
            self._force_stop_next = False
            self._current_model_step[req_id] = 0
            K = self._num_codebooks

            init_text = torch.full((span_len,), self._text_card, dtype=torch.long, device=dev)
            text_embed = self.embed_input_ids(init_text.reshape(1, span_len)).reshape(span_len, -1).to(dtype=dtype)

            sos_code = torch.full((span_len,), self._audio_vocab_size, dtype=torch.long, device=dev)
            audio_embed_sum = torch.zeros((span_len, self.main_config.hidden_size), device=dev, dtype=dtype)
            for cb in range(K):
                audio_embed_sum = audio_embed_sum + self.audio_embed_tokens[cb](sos_code).to(dtype=dtype)

            embed = (text_embed + audio_embed_sum).to(dtype=dtype)
            if cond_sum is not None:
                embed = embed + cond_sum

            zeros = torch.zeros((span_len, K), device=dev, dtype=torch.long)
            info_update: OmniPayload = {
                "codes": {"audio": zeros},
                "meta": {"tts_step": 0},
            }
            if span_len == 1:
                H = self.main_config.hidden_size
                info_update["mtp_inputs"] = (
                    torch.zeros(1, H, device=dev, dtype=dtype),
                    torch.zeros(1, H, device=dev, dtype=dtype),
                )
            return input_ids, embed, info_update

        last_hidden = (info_dict.get("hidden_states") or {}).get("last")
        assert isinstance(last_hidden, torch.Tensor)

        tts_step = int((info_dict.get("meta") or {}).get("tts_step", 0) or 0)
        self._current_model_step[req_id] = tts_step + 1

        if _TTS_DBG:
            self._dbg_step_idx = tts_step
            if tts_step == 0:
                print(
                    f"[MOSHI:delays] delay_steps={self._audio_delay_steps} "
                    f"delays={list(self.delay_pattern.delays[:8])}",
                    flush=True,
                )
                if hasattr(self.model, "layers") and self.model.layers:
                    _attn0 = self.model.layers[0].self_attn
                    _qkv = getattr(_attn0, "qkv_proj", None)
                    if _qkv is not None and hasattr(_qkv, "weight"):
                        _w = _qkv.weight
                        print(
                            f"[MOSHI:in_proj_w] shape={tuple(_w.shape)} w[:2,:4]={_w[:2, :4].tolist()}",
                            flush=True,
                        )

        T_frames = self._prefix_num_frames.get(req_id, 0)
        in_prefix = T_frames > 0 and tts_step < T_frames

        if in_prefix:
            # Kyutai feeds ``zero_idx = -1`` for text during the prefix, which
            # ScaledEmbedding short-circuits to zero. vLLM's embedding can't
            # take -1, so emit a zero text contribution directly.
            text_embed = torch.zeros((1, self.main_config.hidden_size), device=dev, dtype=dtype)
            self._force_stop_next = False
        else:
            # The sampled text token from the previous step dictates the state
            # machine transition. ``input_ids`` at decode is the sampled token.
            sampled = int(input_ids.reshape(-1)[0].item())
            if self._last_machine_step.get(req_id) == tts_step:
                # Already advanced at this step. Skip the
                # state machine so it isn't double-advanced.
                forced_input = self._token_ids.pad
            else:
                if _TTS_DBG:
                    _rp_before = state.remaining_padding
                    _fp_before = state.forced_padding
                    _queued_before = len(state.queued)
                    _entries_before = len(state.entries)
                    _end_step_before = state.end_step
                forced_input = self._machine.process(tts_step, state, sampled)
                self._last_machine_step[req_id] = tts_step
                self._last_forced_token[req_id] = forced_input
                if _TTS_DBG and _entries_before > len(state.entries):
                    _tts_dbg_print_consume(
                        "dst ",
                        tts_step,
                        sampled=sampled,
                        rp_before=_rp_before,
                        fp_before=_fp_before,
                        queued_before=_queued_before,
                        entries_before=_entries_before,
                        new_word_id=self._token_ids.new_word,
                        end_step_set=(_end_step_before is None and state.end_step is not None),
                    )
            if _TTS_DBG:
                _forced_main = forced_input % self._token_ids.card if self._second_stream_ahead > 0 else forced_input
                _tts_dbg_print_state("dst ", tts_step, sampled, _forced_main, state)

            if state.end_step is not None and tts_step >= state.end_step + self._audio_delay_steps:
                forced_input = self._token_ids.pad
                self._force_stop_next = True
            else:
                self._force_stop_next = False

            text_ids = torch.tensor([forced_input], dtype=torch.long, device=dev).reshape(1, 1)
            text_embed = self.embed_input_ids(text_ids).reshape(1, -1).to(dtype=dtype)

        current_model_step = self._current_model_step.get(req_id, 0)
        delays_pp = self.delay_pattern.delays
        sos_code_pp = torch.full((1,), self._audio_vocab_size, dtype=torch.long, device=dev)
        prev_codes_pp = self._last_audio_codes.get(req_id)
        audio_embed_sum = torch.zeros((1, self.main_config.hidden_size), device=dev, dtype=dtype)
        for cb in range(self._num_codebooks):
            per_cb_delay = int(delays_pp[cb])
            if current_model_step <= per_cb_delay:
                cb_emb = self.audio_embed_tokens[cb](sos_code_pp)
                audio_embed_sum = audio_embed_sum + cb_emb.to(dtype=dtype)
            elif current_model_step > per_cb_delay + self._audio_delay_steps:
                if prev_codes_pp is not None:
                    cb_emb = self.audio_embed_tokens[cb](prev_codes_pp[:, cb])
                    audio_embed_sum = audio_embed_sum + cb_emb.to(dtype=dtype)

        embed = text_embed + audio_embed_sum
        if cond_sum is not None:
            embed = embed + cond_sum

        if _TTS_DBG:
            _step_dbg = tts_step
            _audio_ids_debug = prev_codes_pp.tolist() if prev_codes_pp is not None else None
            print(f"[MOSHI:audio_ids]    step={_step_dbg} ids={_audio_ids_debug}", flush=True)
            for _cb in range(min(4, self._num_codebooks)):
                _per_cb_delay = int(delays_pp[_cb])
                if current_model_step <= _per_cb_delay:
                    _cb_code = self._audio_vocab_size
                    _cb_emb = self.audio_embed_tokens[_cb](sos_code_pp).to(dtype=dtype)
                elif current_model_step > _per_cb_delay + self._audio_delay_steps and prev_codes_pp is not None:
                    _cb_code = int(prev_codes_pp[0, _cb].item())
                    _cb_emb = self.audio_embed_tokens[_cb](prev_codes_pp[:, _cb]).to(dtype=dtype)
                else:
                    _cb_code = 0
                    _cb_emb = torch.zeros(1, self.main_config.hidden_size, device=dev, dtype=dtype)
                print(f"[MOSHI:cb_embed]     step={_step_dbg} cb={_cb} code={_cb_code} emb={_cb_emb}", flush=True)
            print(f"[MOSHI:text_embed]   step={_step_dbg} text_emb={text_embed}", flush=True)
            print(f"[MOSHI:audio_embed]  step={_step_dbg} audio_emb_sum={audio_embed_sum}", flush=True)
            print(f"[MOSHI:input_embeds] step={_step_dbg} input_={embed}", flush=True)
            _tts_dbg_print_input("dst ", embed.reshape(1, -1), has_cross=bool(self._cross_attn_wrappers))

        H = self.main_config.hidden_size
        text_step = torch.zeros(1, H, device=dev, dtype=dtype)
        text_step[0, 0] = float(tts_step)

        info_update: OmniPayload = {
            "mtp_inputs": (
                last_hidden.to(device=dev, dtype=dtype).reshape(1, -1),
                text_step,
            ),
            "meta": {"tts_step": tts_step + 1, "prefix_num_frames": T_frames},
        }
        return input_ids, embed.reshape(1, -1), info_update

    def postprocess(self, hidden_states: torch.Tensor, **_: Any) -> dict[str, Any]:
        if hidden_states.numel() == 0:
            return {}
        last = hidden_states[-1, :].detach().contiguous()

        req_id = self._active_req_id
        bsz = 1
        K = self._num_codebooks
        dev = last.device
        current_model_step = self._current_model_step.get(req_id, 0)
        in_audio_delay = current_model_step < self._audio_delay_steps

        if _TTS_DBG:
            _step = getattr(self, "_dbg_step_idx", "?")
            print(f"[MOSHI:depth_used] step={_step} in_audio_delay={in_audio_delay}", flush=True)

        if in_audio_delay:
            audio_codes = torch.zeros((bsz, K), dtype=torch.long, device=dev)
        else:
            forced_tok = self._last_forced_token.get(req_id)
            if forced_tok is not None:
                # Pass the full muxed token; the depth decoder's _text_embed()
                # demuxes it into main + second stream contributions internally.
                text_token_id = torch.tensor([forced_tok] * bsz, dtype=torch.long, device=dev)
            else:
                text_token_id = torch.zeros((bsz,), dtype=torch.long, device=dev)

            if _TTS_DBG and getattr(self, "_dbg_step_idx", -1) == 16:
                print(f"[MOSHI:hidden_at_depformer] step=16 main_hidden={last.reshape(bsz, -1)}", flush=True)
                print(f"[MOSHI:text_tok_at_depformer] step=16 text_token={text_token_id}", flush=True)
                print(f"[MOSHI:forced_at_depformer]  step=16 text_token_after_hook={text_token_id}", flush=True)
            if _TTS_DBG:
                self.depth_decoder._depth_dbg_step = getattr(self, "_dbg_step_idx", -1)

            audio_codes = self.depth_decoder(
                main_hidden=last.reshape(bsz, -1).to(dtype=torch.bfloat16),
                text_token_id=text_token_id,
                do_sample=True,
                temperature=0.6,
                top_k=250,
            )

            if _TTS_DBG:
                self.depth_decoder._depth_dbg_step = -1

            prefix = self._prefixes.get(req_id)
            decode_step = current_model_step - 1
            if prefix is not None and 0 <= decode_step < int(prefix.shape[0]):
                prefix_row = prefix[decode_step].to(device=dev)
                override_mask = (prefix_row != -2).unsqueeze(0).expand(bsz, -1)
                audio_codes = torch.where(override_mask, prefix_row.unsqueeze(0).expand(bsz, -1), audio_codes)

        if _TTS_DBG and not in_audio_delay:
            _step = getattr(self, "_dbg_step_idx", 0)
            _tts_dbg_print_codes("dst ", _step if isinstance(_step, int) else 0, audio_codes[0])

        self._last_audio_codes[req_id] = audio_codes.detach()
        return {"hidden_states": {"last": last}}

    def talker_mtp(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        text_step: torch.Tensor,
        do_sample: bool | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pass-through: the full composite embed is built in preprocess."""
        bsz = int(input_ids.shape[0])
        dev = input_embeds.device
        K = self._num_codebooks

        audio_codes = self._last_audio_codes.get(
            self._active_req_id,
            torch.zeros((bsz, K), dtype=torch.long, device=dev),
        )

        return input_embeds.reshape(bsz, -1), audio_codes.to(dtype=torch.long)

    @staticmethod
    def estimate_prompt_len_from_additional_information(
        additional_information: dict[str, Any] | None,
        **kwargs: Any,
    ) -> int:
        # Single-token prefill is sufficient; decode loop drives generation.
        return 1

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from an HF Moshi-format TTS checkpoint.

        Reuses ``remap_moshi_weights`` for the main transformer; loads
        conditioner LUTs if present and caches the fixed control/cfg
        embeddings used by ``_get_cond_sum``.
        """
        ffn_dim = self.main_config.intermediate_size * 2
        all_weights = list(weights)

        cond_weights: dict[str, torch.Tensor] = {}
        model_weights: list[tuple[str, torch.Tensor]] = []
        for name, t in all_weights:
            if name.startswith("condition_provider.") or name.startswith("conditioners."):
                cond_weights[name] = t
            else:
                model_weights.append((name, t))

        remapped = list(remap_moshi_weights(iter(model_weights), ffn_dim=ffn_dim))

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
            if name.startswith("audio_encoder.") or name.startswith("depth_decoder."):
                continue

            if name == "model.embed_tokens_second_stream.weight":
                if self.embed_tokens_second_stream is not None:
                    default_weight_loader(self.embed_tokens_second_stream.weight, loaded_weight)
                    loaded_params.add("embed_tokens_second_stream.weight")
                continue

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

            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        depth_weights = list(filter_depth_decoder_weights(iter(remapped)))
        depth_loaded = self.depth_decoder.load_weights(iter(depth_weights))
        loaded_params.update(f"depth_decoder.{n}" for n in depth_loaded)

        # Sum-conditioner LUTs. Pre-compute projected embeddings for all valid
        # cfg values so any per-request cfg_coef can be served without re-computation.
        self._cond_control_embed = self._lookup_lut_cond(cond_weights, "control", token="ok")
        hf_cfg = self.vllm_config.model_config.hf_config
        cfg_lut = (getattr(hf_cfg, "conditioners", None) or {}).get("cfg") or {}
        possible_cfg = (cfg_lut.get("lut") or {}).get("possible_values") or ["1.0", "2.0"]
        for tok in possible_cfg:
            emb = self._lookup_lut_cond(cond_weights, "cfg", token=str(tok))
            if emb is not None:
                self._cfg_embeds[str(tok)] = emb
        if self._cfg_embeds:
            logger.info(
                "cfg LUT embeddings loaded for: %s (default=%s)", sorted(self._cfg_embeds), self._default_cfg_coef
            )

        # Speaker-wavs conditioner weights for cross-attention.
        if self._cross_attn_wrappers:
            spk_prefix = "condition_provider.conditioners.speaker_wavs."
            proj_key = f"{spk_prefix}output_proj.weight"
            pad_key = f"{spk_prefix}learnt_padding"
            if proj_key in cond_weights:
                default_weight_loader(self.cross_attn_output_proj.weight, cond_weights[proj_key])
                loaded_params.add("cross_attn_output_proj.weight")
            if pad_key in cond_weights:
                pad = cond_weights[pad_key].squeeze(0)  # [1, H] or [1, 1, H] → [1, H]
                if pad.ndim == 3:
                    pad = pad.squeeze(0)
                self.learnt_padding.copy_(pad.view(1, 1, -1))
                loaded_params.add("learnt_padding")

        all_params = set(dict(self.named_parameters(remove_duplicate=False)).keys())
        missing = all_params - loaded_params
        logger.info(
            "Loaded %d/%d weights for MoshiTTSTalkerForConditionalGeneration",
            len(loaded_params),
            len(all_params),
        )
        if missing:
            logger.warning("Missing params (random init): %s", sorted(missing)[:30])

        if self._cross_attn_wrappers:
            all_tensors = {**dict(self.named_parameters(remove_duplicate=False)), **dict(self.named_buffers())}
            for key in (
                "model.layers.0.cross_attn.q_proj.weight",
                "model.layers.0.cross_attn_layernorm.weight",
                "cross_attn_output_proj.weight",
                "learnt_padding",
            ):
                p = all_tensors.get(key)
                if p is not None:
                    logger.info("cross-attn diag %s: shape=%s norm=%.4f", key, tuple(p.shape), p.float().norm().item())
                else:
                    logger.warning("cross-attn diag %s: NOT FOUND", key)

        return loaded_params

    def _lookup_lut_cond(
        self,
        cond_weights: dict[str, torch.Tensor],
        name: str,
        token: str,
    ) -> torch.Tensor | None:
        """Compute the projected LUT embedding for a single fixed token value.

        The source layout is roughly:
          conditioners.<name>.embed.weight : [n_bins+1, dim]
          conditioners.<name>.output_proj.weight : [output_dim, dim]
        """
        hf_cfg = self.vllm_config.model_config.hf_config
        cond_cfg_dict = getattr(hf_cfg, "conditioners", None) or {}
        spec = cond_cfg_dict.get(name)
        if not isinstance(spec, dict):
            return None
        lut = spec.get("lut", {})
        possible = list(lut.get("possible_values") or [])
        if token not in possible:
            logger.warning("LUT cond %s: token %r not in possible_values", name, token)
            return None
        idx = possible.index(token)

        emb_w: torch.Tensor | None = None
        proj_w: torch.Tensor | None = None
        for k, v in cond_weights.items():
            if k.endswith(f".{name}.embed.weight") or k.endswith(f"{name}.embed.weight"):
                emb_w = v
            elif k.endswith(f".{name}.output_proj.weight") or k.endswith(f"{name}.output_proj.weight"):
                proj_w = v
        if emb_w is None:
            logger.debug("LUT cond %s: embed weight not found in checkpoint", name)
            return None

        vec = emb_w[idx]
        if proj_w is not None:
            vec = proj_w @ vec
        vec = vec.reshape(-1)
        if vec.shape[0] != self.main_config.hidden_size:
            logger.warning(
                "LUT cond %s: projected dim %d != hidden_size %d; skipping",
                name,
                vec.shape[0],
                self.main_config.hidden_size,
            )
            return None
        return vec.detach()


__all__ = ["MoshiTTSTalkerForConditionalGeneration"]
