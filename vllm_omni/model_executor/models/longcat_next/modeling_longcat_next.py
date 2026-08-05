"""LongCat-Next thinker stage — native vLLM port.

The 74B MoE+MLA backbone runs on vLLM's native FlashNgramModel, so TP, paged
KV cache and fused MoE all apply. On top of that:

- n-gram embedding fusion runs per-request in the omni runner's preprocess
  hook (vllm-omni forces the V1 runner, so it can't live in an MRV2
  ModelState like upstream); left context lives on the model, cleaned up
  via on_requests_finished.
- multimodal understanding: remote-code visual/audio tokenizers produce code
  embeddings merged at the mm placeholder positions via vLLM's standard
  is_multimodal mask path.
- lm_head is 131125-wide (text + special tokens only).
- audio generation uses talker_mtp: preprocess emits mtp_inputs once
  <longcat_audiogen_start> is seen, the runner calls talker_mtp() per decode
  step, which runs the 8-level audio_head (rank 0 + broadcast, since the
  checkpoint hardcodes cuda:0) and accumulates codes into
  model_intermediate_buffer["codes"]["audio"]. EOS suppression / forced
  tokens live in compute_logits to work on the async-scheduling path.
"""

import os
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.longcat_flash import FlashConfig
from vllm.model_executor.models.longcat_flash_ngram import FlashNgramModel, NgramEmbedding
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    _merge_multimodal_embeddings,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .longcat_next_processor import (
    LongcatNextDummyInputsBuilder,
    LongcatNextMultiModalProcessor,
    LongcatNextProcessingInfo,
)
from .longcat_next_utils import (
    AUDIO_PAD_TOKEN_ID,
    AUDIOGEN_END_TOKEN_ID,
    AUDIOGEN_START_TOKEN_ID,
    AUDIOTEXT_PAD_TOKEN_ID,
    AUDIOTEXT_START_TOKEN_ID,
    IMG_END_TOKEN_ID,
    IMG_NEWLINE_TOKEN_ID,
    IMG_PAD_TOKEN_ID,
    IMG_START_TOKEN_ID,
    get_remote_attr,
    load_remote_hf_config,
)

logger = init_logger(__name__)

_DEFAULT_PAD_TOKEN_ID = 3  # generation_config.json; config.json omits it

# CFG twin request_id suffix (see expand_longcat_cfg_prompts); used to pair
# the unconditional stream with its parent for per-step logit combination.
_CFG_VISUAL_SUFFIX = "__cfg_visual"
_DEFAULT_CFG_SCALE = 3.0  # generation_config.json custom_params.cfg_scale


@MULTIMODAL_REGISTRY.register_processor(
    LongcatNextMultiModalProcessor,
    info=LongcatNextProcessingInfo,
    dummy_inputs=LongcatNextDummyInputsBuilder,
)
class LongcatNextForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):
    supports_multimodal = True
    merge_by_field_config = True

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # Omni runner hook: per-request preprocess computes the fused n-gram (+mm)
    # inputs_embeds for each span. See gpu_model_runner._preprocess.
    has_preprocess = True
    has_postprocess = True
    have_multimodal_outputs = True
    talker_mtp_accepts_req_infos = True
    # Shares one backbone for text and audio: only emits "mtp_inputs" while
    # in audio-gen mode, so the runner must skip talker decode on text-only
    # steps (gpu_model_runner._talker_mtp_forward).
    omits_talker_mtp_inputs_when_idle = True
    # n-gram hashing needs raw token ids, not embeddings -- upstream's
    # _prepare_mm_inputs otherwise sets input_ids=None on the mm path.
    requires_raw_input_tokens = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<longcat_img_start><longcat_img_pad><longcat_img_end>"
        if modality.startswith("audio"):
            return "<longcat_audio_start><longcat_audio_pad><longcat_audio_end>"
        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        model_config = vllm_config.model_config
        self.model_stage = getattr(model_config, "model_stage", "thinker")
        if self.model_stage != "thinker":
            raise ValueError(f"Unsupported model_stage: {self.model_stage}")
        self.model_path: str = model_config.model

        hf = model_config.hf_config
        # FlashModel/FlashConfig field alignment: checkpoint names the MLP
        # width ffn_hidden_size and omits pad_token_id.
        hf.intermediate_size = getattr(hf, "ffn_hidden_size", getattr(hf, "intermediate_size", None))
        if getattr(hf, "pad_token_id", None) is None:
            hf.pad_token_id = _DEFAULT_PAD_TOKEN_ID
        # With quantization: fp8, FP8/Marlin would also quantize MLA's
        # kv_b_proj, but FlashDecoderLayer's weight-loading fixup unflattens
        # it into w_kc/w_vc assuming an unpacked bf16 layout -- excluding
        # self_attn keeps MLA in native bf16 and only quantizes the MoE
        # experts (the actual memory driver).
        disable_quant_module = list(getattr(hf, "disable_quant_module", []) or [])
        if "self_attn" not in disable_quant_module:
            disable_quant_module.append("self_attn")
        hf.disable_quant_module = disable_quant_module
        self.config = hf
        self.quant_config = vllm_config.quant_config

        # --- native TP-sharded backbone -------------------------------------
        # n-gram tables hash with text_vocab_size (131072), not the full
        # 282624 vocab -- the native NgramEmbedding uses config.vocab_size,
        # which would allocate a ~135 GB table, so attach a correctly-sized
        # one instead of the base class's default.
        self.text_vocab_hash_size = int(getattr(hf, "text_vocab_size", 131072))
        ngram_ratio = hf.ngram_vocab_size_ratio
        hf.ngram_vocab_size_ratio = None
        try:
            self.model = FlashNgramModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        finally:
            hf.ngram_vocab_size_ratio = ngram_ratio
        if get_pp_group().is_first_rank:
            ngram_cfg = FlashConfig(**hf.__dict__)
            ngram_cfg.vocab_size = self.text_vocab_hash_size
            self.model.ngram_embeddings = NgramEmbedding(ngram_cfg, self.model.embed_tokens)

        self.text_vocab_size = int(getattr(hf, "text_vocab_plus_multimodal_special_token_size", 131125))
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.text_vocab_size,
                hf.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(self.text_vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        # --- multimodal encoders + depth heads (remote code, replicated) ---
        remote_config = load_remote_hf_config(self.model_path)
        visual_tok_cls = get_remote_attr(self.model_path, "modular_longcat_next_visual", "LongcatNextVisualTokenizer")
        audio_tok_cls = get_remote_attr(self.model_path, "modular_longcat_next_audio", "LongcatNextAudioTokenizer")
        head_cls = get_remote_attr(self.model_path, "modular_longcat_next", "CasualDepthTransformerHead")
        self.visual_tokenizer = visual_tok_cls(remote_config)
        self.audio_tokenizer = audio_tok_cls(remote_config)
        vc, ac = remote_config.visual_config, remote_config.audio_config
        self.visual_head = head_cls(
            hidden_size=hf.hidden_size,
            codebook_sizes=vc.vq_config.codebook_sizes,
            transformer_layer_num=vc.image_head_transformer_layers,
            transformer_dim=vc.image_head_transformer_dims,
            transformer_ffn_scale=vc.image_head_transformer_ffn_scale,
        )
        self.audio_head = head_cls(
            hidden_size=hf.hidden_size,
            codebook_sizes=ac.vq_config.codebook_sizes,
            transformer_layer_num=ac.audio_head_transformer_layers,
            transformer_dim=ac.audio_head_transformer_dims,
            transformer_ffn_scale=ac.audio_head_transformer_ffn_scale,
        )

        # Cache codebook sizes off the *remote* config: self.config's
        # audio_config/visual_config are plain dicts, so this would raise
        # AttributeError there.
        self.audio_codebook_sizes = list(ac.vq_config.codebook_sizes)
        self.visual_codebook_sizes = list(vc.vq_config.codebook_sizes)

        # Per-level cumulative code-id offsets, mirroring the HF buffers.
        visual_offsets = torch.cumsum(
            torch.tensor([hf.visual_offset] + list(vc.vq_config.codebook_sizes[:-1]), dtype=torch.long),
            dim=0,
        )
        audio_offsets = torch.cumsum(
            torch.tensor([hf.audio_offset] + list(ac.vq_config.codebook_sizes[:-1]), dtype=torch.long),
            dim=0,
        )
        self.register_buffer("visual_offset_vals", visual_offsets, persistent=False)
        self.register_buffer("audio_offset_vals", audio_offsets, persistent=False)

        self.dtype = getattr(model_config, "dtype", torch.bfloat16)
        if not isinstance(self.dtype, torch.dtype):
            self.dtype = torch.bfloat16

        # --- per-request n-gram left context (omni V1 preprocess state) ---
        ngram = self.model.ngram_embeddings
        self._ngram_n = int(getattr(hf, "emb_neighbor_num", 4))
        self._ctx_len = self._ngram_n - 1
        self._eos_id = int(getattr(hf, "eos_token_id", 2))
        self._ngram_ctx: dict[str, torch.Tensor] = {}
        self._ngram_disabled = int(os.environ.get("LONGCAT_NGRAM_DISABLE", "0")) != 0
        self._max_ctx_entries = 4 * max(int(getattr(vllm_config.scheduler_config, "max_num_seqs", 64)), 64)
        assert ngram is not None, "LongCat-Next requires ngram embeddings"

        # --- audio generation state (talker_mtp) ---
        self._audio_gen: dict[str, dict[str, Any]] = {}
        self._audio_delay_default = 0
        max_audio_seconds = getattr(ac, "max_audio_seconds", 30)
        # 25 fps, matches reference. LONGCAT_MAX_GEN caps it for bounded
        # validation runs (a full 30 s chunk is ~750 decode steps, which can
        # outlast a short job walltime); unset it for full-length audio.
        self.max_gen = int(os.environ.get("LONGCAT_MAX_GEN") or max_audio_seconds * 25)

        # --- image generation state (talker_mtp dispatches to this too) ---
        self._visual_gen: dict[str, dict[str, Any]] = {}
        # The checkpoint declares no default grid size; token_h/token_w are
        # normally caller-supplied per request. This 37x37 fallback only
        # applies when a request omits them.
        self._default_token_w = 37
        self._default_token_h = 37
        # _replicated_audio_code_embedding is NOT pre-declared here (e.g.
        # `= None`): register_buffer raises KeyError if the name is already a
        # plain instance attribute, so it must stay absent from __dict__
        # until the first real register_buffer call.

    # ------------------------------------------------------------------ #
    # embeddings
    # ------------------------------------------------------------------ #

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        inputs_embeds = self.model.embed_input_ids(input_ids)
        if multimodal_embeddings is not None and len(multimodal_embeddings) > 0:
            assert is_multimodal is not None
            inputs_embeds = _merge_multimodal_embeddings(
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                is_multimodal=is_multimodal,
            )
        return inputs_embeds

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.embed_input_ids(input_ids, multimodal_embeddings, **kwargs)

    def _code_embeddings(self, code_ids: torch.Tensor) -> torch.Tensor:
        """[n, num_levels] offset code ids -> [n, hidden] summed embeddings."""
        return self.model.embed_tokens(code_ids).sum(dim=1)

    def _encode_images(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> list[torch.Tensor]:
        # pixel_values is every image's patches flat-concatenated on dim 0;
        # the vision encoder consumes the whole batch in one call, so split
        # its output back into per-image chunks via grid_thw.
        device = self.visual_offset_vals.device
        grid_thw = grid_thw.reshape(-1, 3).to(device)
        # merge_size isn't a visual_config field -- read it off the
        # constructed encoder module, which defaults it to 2.
        merge_size = getattr(self.visual_tokenizer.visual_model, "merge_size", 2)
        sizes = (grid_thw.prod(dim=-1) // (merge_size**2)).tolist()

        with torch.inference_mode():
            ids = self.visual_tokenizer.encode(pixel_values.to(device=device, dtype=self.dtype), grid_thw)
            ids = ids.long() + self.visual_offset_vals
            emb = self._code_embeddings(ids)
            emb = self.visual_tokenizer.visual_embedding_layer(emb.to(self.dtype))
        return list(emb.split([int(s) for s in sizes], dim=0))

    def _encode_audios(
        self,
        features: torch.Tensor,
        encoder_lengths: torch.Tensor,
        bridge_lengths: torch.Tensor,
    ) -> list[torch.Tensor]:
        # features is every clip's chunks flat-concatenated on dim 0; the
        # audio encoder consumes the whole batch in one call, so split its
        # output per bridge_length.
        device = self.audio_offset_vals.device
        bridge_lengths = bridge_lengths.reshape(-1).to(device)

        with torch.inference_mode():
            ids = self.audio_tokenizer.encode(
                features.to(device=device, dtype=self.dtype),
                encoder_lengths.reshape(-1).to(device),
                bridge_lengths,
            )
            ids = ids.long() + self.audio_offset_vals
            emb = self._code_embeddings(ids)
        return list(emb.split([int(s) for s in bridge_lengths.tolist()], dim=0))

    def embed_multimodal(self, **kwargs: Any) -> tuple[torch.Tensor, ...]:
        # Must be named embed_multimodal, not get_multimodal_embeddings: the
        # latter falls through to SupportsMultiModal's stub (implicit None),
        # which breaks engine profiling.
        embeds: list[torch.Tensor] = []

        pixel_values = kwargs.get("pixel_values")
        if pixel_values is not None:
            grid_thw = kwargs.get("image_grid_thw")
            embeds.extend(self._encode_images(pixel_values, grid_thw))

        audio_features = kwargs.get("audio_features")
        if audio_features is not None:
            enc = kwargs.get("audio_encoder_lengths")
            bridge = kwargs.get("audio_bridge_lengths")
            embeds.extend(self._encode_audios(audio_features, enc, bridge))

        return tuple(embeds)

    # ------------------------------------------------------------------ #
    # per-request n-gram fusion (omni preprocess hook)
    # ------------------------------------------------------------------ #

    def _neg_eos(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.where(tokens == self._eos_id, -tokens, tokens)

    def _span_oe_ids(self, request_id: str, span_ids: torch.Tensor, fresh: bool) -> torch.Tensor:
        """Global n-gram ids [span, num_embedders] for one request span."""
        ngram = self.model.ngram_embeddings
        device = span_ids.device
        span = span_ids.shape[0]

        ctx = self._ngram_ctx.get(request_id)
        if fresh or ctx is None:
            ctx = torch.full((self._ctx_len,), -1, dtype=torch.int32, device=device)

        cur = self._neg_eos(span_ids.to(torch.int32))
        table = torch.cat([ctx, cur]).unsqueeze(0).contiguous()
        qsl = torch.tensor([0, span], dtype=torch.int32, device=device)
        row_indices = torch.zeros(1, dtype=torch.int64, device=device)
        column_starts = torch.full((1,), self._ctx_len, dtype=torch.int32, device=device)
        oe_ids = torch.empty(span, ngram.num_embedders, dtype=torch.int32, device=device)
        ops.ngram_compute_n_gram_ids(
            self._ngram_n,
            ngram.k,
            ngram.ne_weights,
            ngram.ne_mods,
            ngram.exclusive_sizes,
            qsl,
            table,
            row_indices,
            column_starts,
            oe_ids,
        )

        # Roll the context forward and bound the state dict.
        self._ngram_ctx[request_id] = table[0, -self._ctx_len :].clone()
        if len(self._ngram_ctx) > self._max_ctx_entries:
            for stale in list(self._ngram_ctx)[: len(self._ngram_ctx) - self._max_ctx_entries]:
                self._ngram_ctx.pop(stale, None)

        return oe_ids.long()

    def _advance_audio_gen(
        self,
        request_id: str,
        last_token: int,
        device: torch.device,
        dtype: torch.dtype,
        decode_eligible: bool = True,
    ) -> dict[str, Any]:
        """Update audio-gen state based on the last emitted visible token.
        Returns update_dict entries for the runner.

        ``decode_eligible`` must mirror the runner's own talker_mtp dispatch
        gate (span_len == 1 and not is_prefill): if <longcat_audiogen_start>
        lands as the last token of a multi-token prefill chunk, the runner
        won't call talker_mtp that step even though preprocess() sees the
        token. Advancing gen_step unconditionally would then desync it from
        the actual number of talker_mtp calls, silently dropping a frame.
        Gating the advance on decode_eligible keeps them 1:1.
        """
        update: dict[str, Any] = {}
        if last_token == AUDIOGEN_START_TOKEN_ID:
            self._audio_gen[request_id] = {
                "gen_step": 0,
                "audio_start": False,
                "text_end": False,
                "delay": self._audio_delay_default,
                "ext_id": AUDIOTEXT_PAD_TOKEN_ID,
                "terminal": False,
            }
        elif last_token == AUDIOGEN_END_TOKEN_ID:
            self._audio_gen.pop(request_id, None)

        state = self._audio_gen.get(request_id)
        if state is not None and not state["terminal"] and decode_eligible:
            # 0-based index of this step, captured before advancing (the
            # reference compares its pre-increment value against delay).
            gen_step = state["gen_step"]
            state["gen_step"] = gen_step + 1

            # First AUDIOTEXT_PAD sampled marks end of the spoken transcript;
            # from then on compute_logits pins the text stream to pad.
            if not state["text_end"] and last_token == AUDIOTEXT_PAD_TOKEN_ID:
                state["text_end"] = True
                state["delay"] = min(state["delay"], gen_step)

            # Audio enables at the END of step == delay (codes already
            # discarded that step), so the first real frame is delay+1 --
            # hence '>' not '>='.
            delay = state["delay"]
            state["ext_id"] = AUDIOTEXT_START_TOKEN_ID if gen_step == delay else AUDIOTEXT_PAD_TOKEN_ID
            state["audio_start"] = gen_step > delay
            # Emit mtp_inputs so the runner calls talker_mtp this step;
            # last_hidden comes from the previous forward, zeros if new.
            last_hidden = state.get("last_hidden")
            if last_hidden is None:
                last_hidden = torch.zeros(1, self.config.hidden_size, device=device, dtype=dtype)
            text_step = torch.zeros(1, self.config.hidden_size, device=device, dtype=dtype)
            update["mtp_inputs"] = (last_hidden, text_step)
        return update

    def _advance_visual_gen(
        self,
        request_id: str,
        last_token: int,
        device: torch.device,
        dtype: torch.dtype,
        token_w: int | None = None,
        token_h: int | None = None,
        cfg_scale: float | None = None,
        decode_eligible: bool = True,
    ) -> dict[str, Any]:
        """Update image-gen state based on the last emitted visible token.

        ``decode_eligible`` mirrors the runner's talker_mtp dispatch gate --
        same rationale as _advance_audio_gen's docstring, to keep gen_step
        1:1 with actual talker_mtp calls.

        Unlike audio, the reference's image state machine has no delay/
        text_end loop-back -- it just tracks a 2D grid: every
        (token_w + 1)-th step is a row boundary (forced IMAGE_NEWLINE, pixel
        discarded); every other step is a real pixel (forced IMAGE_PAD).
        Termination is the grid bound (gen_step reaches token_h*(token_w+1)),
        forcing IMAGE_END -- the reference's own end-of-image sentinel is
        masked out of the head output, so without this bound generation
        would only stop when max_new_tokens runs out.
        """
        update: dict[str, Any] = {}
        if last_token == IMG_START_TOKEN_ID:
            # token_w/token_h should come from additional_information (and
            # match the prompt's anyres prefix); missing them means the
            # model generates without knowing its canvas.
            is_cfg_twin = request_id.endswith(_CFG_VISUAL_SUFFIX)
            if token_w is None or token_h is None:
                # The uncond CFG twin has no additional_information of its
                # own; inherit the parent's canvas/cfg_scale so both streams
                # decode the same grid.
                parent_id = request_id[: -len(_CFG_VISUAL_SUFFIX)]
                parent_state = self._visual_gen.get(parent_id)
                if is_cfg_twin and parent_state is not None:
                    token_w = parent_state.get("token_w")
                    token_h = parent_state.get("token_h")
                    cfg_scale = parent_state.get("cfg_scale", cfg_scale)
                if (token_w is None or token_h is None) and not is_cfg_twin:
                    logger.warning(
                        "[longcat-image] req=%s entered image gen WITHOUT "
                        "token_w/token_h; falling back to %sx%s. Add "
                        "additional_information={'token_w': ..., 'token_h': ...} "
                        "and the <longcat_img_token_size>{h} {w}</longcat_img_token_size> "
                        "prompt prefix to match the reference behavior.",
                        request_id,
                        self._default_token_w,
                        self._default_token_h,
                    )
            self._visual_gen[request_id] = {
                "gen_step": 0,
                "token_w": token_w or self._default_token_w,
                "token_h": token_h or self._default_token_h,
                "cfg_scale": cfg_scale if cfg_scale is not None else _DEFAULT_CFG_SCALE,
                "ext_id": IMG_PAD_TOKEN_ID,
                "terminal": False,
            }
        elif last_token == IMG_END_TOKEN_ID:
            self._visual_gen.pop(request_id, None)

        state = self._visual_gen.get(request_id)
        if state is not None and not state["terminal"] and decode_eligible:
            gen_step = state["gen_step"]
            state["gen_step"] = gen_step + 1
            token_w = state["token_w"]
            grid_end = state["token_h"] * (token_w + 1)
            if state["gen_step"] >= grid_end:
                # Grid complete; this step would be a trailing row-boundary
                # newline, so close the image instead (visible IMG_END is
                # forced in compute_logits; the state pops next step).
                state["terminal"] = True
                state["ext_id"] = IMG_END_TOKEN_ID
            else:
                is_row_boundary = state["gen_step"] % (token_w + 1) == 0
                state["ext_id"] = IMG_NEWLINE_TOKEN_ID if is_row_boundary else IMG_PAD_TOKEN_ID
            last_hidden = state.get("last_hidden")
            if last_hidden is None:
                last_hidden = torch.zeros(1, self.config.hidden_size, device=device, dtype=dtype)
            text_step = torch.zeros(1, self.config.hidden_size, device=device, dtype=dtype)
            update["mtp_inputs"] = (last_hidden, text_step)
        return update

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        request_id = str(info.get("request_id", "default"))
        num_computed = int(info.get("_omni_num_computed_tokens", 0))

        # Special tokens (incl. mm pad markers) are hash-segment boundaries
        # and take pure word embeddings, no n-gram fusion -- a negative
        # entry in the kernel's table convention.
        ignored = input_ids >= self.text_vocab_hash_size
        boundary = ignored | (input_ids == 0)
        table_ids = torch.where(boundary, torch.full_like(input_ids, -1), input_ids)

        # LONGCAT_NGRAM_DISABLE=1: A/B bypass for isolating whether a
        # divergence is in n-gram fusion vs. attention/MLA.
        if self._ngram_disabled:
            out = self.model.embed_tokens(input_ids)
        else:
            oe_ids = self._span_oe_ids(request_id, table_ids, fresh=num_computed == 0)
            fused = self.model.ngram_embeddings.embed_batched(input_ids, oe_ids)
            word = self.model.embed_tokens(input_ids)
            out = torch.where(ignored.unsqueeze(-1), word, fused)

        # Placeholder positions carry the multimodal embeddings already merged
        # into input_embeds by the runner's standard mm path.
        pad_mask = (input_ids == IMG_PAD_TOKEN_ID) | (input_ids == AUDIO_PAD_TOKEN_ID)
        if input_embeds is not None and pad_mask.any():
            out = torch.where(pad_mask.unsqueeze(-1), input_embeds.to(out.dtype), out)

        # Advance audio/image-gen state off the LAST token of this span. A
        # request is in at most one of _audio_gen/_visual_gen at a time, so
        # both calls are safe unconditionally. decode_eligible must mirror
        # the runner's talker_mtp dispatch gate -- see _advance_audio_gen.
        update_dict: dict[str, Any] = {}
        last_token = int(input_ids[-1])
        is_prefill = bool(info.get("_omni_is_prefill", False))
        decode_eligible = (not is_prefill) and int(input_ids.shape[0]) == 1
        update_dict.update(
            self._advance_audio_gen(
                request_id,
                last_token,
                device=input_ids.device,
                dtype=self.dtype,
                decode_eligible=decode_eligible,
            )
        )
        additional_information = info.get("additional_information")
        if not isinstance(additional_information, dict):
            additional_information = info
        token_w = additional_information.get("token_w")
        token_h = additional_information.get("token_h")
        cfg_scale = additional_information.get("cfg_scale")
        update_dict.update(
            self._advance_visual_gen(
                request_id,
                last_token,
                device=input_ids.device,
                dtype=self.dtype,
                token_w=token_w,
                token_h=token_h,
                cfg_scale=cfg_scale,
                decode_eligible=decode_eligible,
            )
        )

        return input_ids, out.to(self.dtype), update_dict

    def on_requests_finished(self, finished_req_ids: Any) -> None:
        for req_id in finished_req_ids:
            req_id = str(req_id)
            self._ngram_ctx.pop(req_id, None)
            self._audio_gen.pop(req_id, None)
            self._visual_gen.pop(req_id, None)

    # ------------------------------------------------------------------ #
    # forward / logits
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    # ------------------------------------------------------------------ #
    # audio generation (talker_mtp + state machine)
    # ------------------------------------------------------------------ #

    def make_omni_output(self, model_output: torch.Tensor, **kwargs: Any) -> OmniOutput:
        """Stash hidden states for the next step and emit this step's codes.

        Must return an OmniOutput, not a bare tensor: the runner's
        extract_multimodal_outputs only reads multimodal_outputs off an
        OmniOutput instance, so a tensor return silently drops every frame
        talker_mtp produces. _preprocess (which runs talker_mtp) precedes
        _model_forward (which calls this) within a step, so this step's
        codes are already in model_intermediate_buffer; the output processor
        concatenates per-step rows into the final [T, 8] tensor.
        """
        # Non-last pipeline ranks forward IntermediateTensors, which carry no
        # hidden states to stash and no codes to emit.
        if not isinstance(model_output, torch.Tensor):
            return model_output

        info_dicts = kwargs.get("model_intermediate_buffer")
        if info_dicts is None:
            info_dicts = kwargs.get("runtime_additional_information") or []

        # Stash each active request's own last-position hidden state, using
        # request_token_spans to pick the right row per request in a
        # multi-request batch (rather than everyone inheriting the last row).
        spans = kwargs.get("request_token_spans")
        active_ids = list(self._audio_gen.keys()) + list(self._visual_gen.keys())
        if active_ids and isinstance(spans, list) and len(spans) == len(info_dicts):
            for idx, info in enumerate(info_dicts):
                if not isinstance(info, dict):
                    continue
                req_id = info.get("req_id")
                if req_id is None:
                    continue
                state = self._audio_gen.get(req_id) or self._visual_gen.get(req_id)
                if state is None:
                    continue
                start, end = spans[idx]
                row = model_output[start:end]
                if row.numel():
                    state["last_hidden"] = row[-1:].detach().clone()
        elif active_ids:
            # Fallback for callers without request_token_spans (e.g. tests).
            for req_id in active_ids:
                state = self._audio_gen.get(req_id) or self._visual_gen.get(req_id)
                state["last_hidden"] = model_output[-1:].detach().clone()

        # The runner's talker_mtp output key doesn't carry a per-row modality
        # tag, so each frame is routed by looking up which of
        # _audio_gen/_visual_gen its req_id belongs to -- this is what lets a
        # batch mixing audio-gen and image-gen requests route correctly. The
        # fallback only applies to a batch that isn't already mixed.
        fallback_modality = "visual" if (self._visual_gen and not self._audio_gen) else "audio"

        frames_by_modality: dict[str, list[torch.Tensor]] = {"audio": [], "visual": []}
        for info in info_dicts:
            if not isinstance(info, dict):
                continue
            codes = info.get("codes")
            if not isinstance(codes, dict):
                continue
            # Pop (not just read): leaving it would re-emit the same frame on
            # every later step where talker_mtp didn't run, duplicating audio.
            frame = codes.pop("audio", None)
            if not isinstance(frame, torch.Tensor) or frame.numel() == 0:
                continue
            frame = frame.reshape(1, -1) if frame.dim() == 1 else frame
            # talker_mtp marks discarded rows (pre-audio_start, row-boundary,
            # chunk/image-end sentinel) with all -1 to stay batch-aligned;
            # those must not reach the decoder.
            frame = frame[(frame >= 0).all(dim=1)]
            if frame.numel() == 0:
                continue
            req_id = info.get("req_id")
            if req_id in self._visual_gen:
                modality = "visual"
            elif req_id in self._audio_gen:
                modality = "audio"
            else:
                modality = fallback_modality
            frames_by_modality[modality].append(frame)

        codes_out_by_modality: dict[str, torch.Tensor] = {}
        for modality, frames in frames_by_modality.items():
            if not frames:
                continue
            codes_out = torch.cat(frames, dim=0)
            codes_out_by_modality[modality] = codes_out
        if not codes_out_by_modality:
            return OmniOutput(text_hidden_states=model_output, multimodal_outputs={})
        return OmniOutput(
            text_hidden_states=model_output,
            multimodal_outputs={"codes": codes_out_by_modality},
        )

    def postprocess(
        self,
        model_output: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Post-forward hook — stash hidden state for talker_mtp conditioning.

        Falls back to make_omni_output for the thinker-only pipeline where
        postprocess is filtered out (no downstream stage). This hook provides
        per-request hidden state alignment via update_dict.
        """
        return {}

    def _ensure_replicated_audio_code_embedding(self, device: torch.device) -> torch.Tensor:
        existing = getattr(self, "_replicated_audio_code_embedding", None)
        if existing is not None:
            return existing
        # Replicated (non-TP) copy of the audio-code embedding rows, so
        # audio_head's rank-0-only code path avoids VocabParallelEmbedding's
        # TP all-reduce. All ranks participate in this embed_tokens call
        # (matched collective) before each stores its own local copy.
        base = int(self.audio_offset_vals[0].item())
        total = sum(self.audio_codebook_sizes)
        audio_id_range = torch.arange(base, base + total, device=device, dtype=torch.long)
        emb = self.model.embed_tokens(audio_id_range).detach().float()
        self.register_buffer(
            "_replicated_audio_code_embedding", emb.to(device=device, dtype=self.dtype), persistent=False
        )
        del audio_id_range
        return self._replicated_audio_code_embedding

    def _ensure_audio_code_embed_module(self, device: torch.device) -> nn.Module:
        """Callable embedding over the audio-code rows, for audio_head.

        CasualDepthTransformerHead.forward calls .to("cuda:0") then invokes
        this argument directly, so it needs a real callable nn.Module (not
        None). Indices are relative to AUDIO_OFFSET to stay a compact table
        instead of the full TP-sharded embed_tokens.
        """
        holder = self.__dict__.setdefault("_audio_embed_holder", {})
        module = holder.get("module")
        if module is not None:
            return module
        rows = self._ensure_replicated_audio_code_embedding(device)
        module = nn.Embedding(rows.shape[0], rows.shape[1], _weight=rows.detach().clone())
        module = module.to(device=device, dtype=self.dtype)
        module.requires_grad_(False)
        holder["module"] = module
        return module

    def _ensure_replicated_visual_code_embedding(self, device: torch.device) -> torch.Tensor:
        """Visual-code analog of _ensure_replicated_audio_code_embedding.
        Used only for the depth-head's own internal embedding argument --
        NOT the outer next-step feedback embedding, which reuses
        _code_embeddings + visual_embedding_layer (same as _encode_images).
        """
        existing = getattr(self, "_replicated_visual_code_embedding", None)
        if existing is not None:
            return existing
        base = int(self.visual_offset_vals[0].item())
        total = sum(self.visual_codebook_sizes)
        visual_id_range = torch.arange(base, base + total, device=device, dtype=torch.long)
        emb = self.model.embed_tokens(visual_id_range).detach().float()
        self.register_buffer(
            "_replicated_visual_code_embedding", emb.to(device=device, dtype=self.dtype), persistent=False
        )
        del visual_id_range
        return self._replicated_visual_code_embedding

    def _ensure_visual_code_embed_module(self, device: torch.device) -> nn.Module:
        """Callable embedding over the visual-code rows, for visual_head.
        Same rationale as _ensure_audio_code_embed_module."""
        holder = self.__dict__.setdefault("_visual_embed_holder", {})
        module = holder.get("module")
        if module is not None:
            return module
        rows = self._ensure_replicated_visual_code_embedding(device)
        module = nn.Embedding(rows.shape[0], rows.shape[1], _weight=rows.detach().clone())
        module = module.to(device=device, dtype=self.dtype)
        module.requires_grad_(False)
        holder["module"] = module
        return module

    def _sample_audio_code(
        self,
        logits: torch.Tensor,
        do_sample: bool = True,
        temperature: float = 0.5,
        top_k: int = 5,
        top_p: float = 0.85,
        repetition_penalty: float = 1.0,
        past_codes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample one audio code from logits, matching generation_config
        defaults. None sampling keys coalesce to defaults (the runner passes
        explicit None when unset). repetition_penalty/past_codes implement
        the reference's code-level penalty on raw logits, before
        temperature; no-op when penalty==1.0 or no history.
        """
        if do_sample is None:
            do_sample = True
        if temperature is None:
            temperature = 0.5
        if top_k is None:
            top_k = 5
        if top_p is None:
            top_p = 0.85
        if repetition_penalty is None:
            repetition_penalty = 1.0
        if past_codes is not None and past_codes.numel() and repetition_penalty != 1.0:
            past = past_codes.reshape(-1)
            scores = logits[past]
            scores = torch.where(scores < 0, scores * repetition_penalty, scores / repetition_penalty)
            logits = logits.clone()
            logits[past] = scores.to(logits.dtype)
        if do_sample and temperature > 0:
            logits = logits / temperature
            if top_k > 0:
                top_k = min(top_k, logits.shape[-1])
                threshold = logits.topk(top_k).values[..., -1, None]
                logits[logits < threshold] = float("-inf")
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)
        return logits.argmax(dim=-1)

    def _sample_depth_head(
        self,
        head: nn.Module,
        code_embed: nn.Module,
        offset_vals: torch.Tensor,
        num_levels: int,
        last_hidden: torch.Tensor,
        rank: int,
        tp_group: Any,
        device: torch.device,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float = 1.0,
        past_codes: torch.Tensor | None = None,
        mask_sentinel: bool = False,
        mask_audio_sentinels: bool = False,
    ) -> torch.Tensor:
        """Shared rank-0+broadcast depth-head sampling loop, used by both
        audio_head and visual_head (same checkpoint class, hardcoded
        cuda:0). Returns the broadcast codes_row [num_levels].

        mask_sentinel zeroes the level-0 end-of-image class for the visual
        head so the grid bound in _advance_visual_gen is the sole
        terminator. mask_audio_sentinels zeroes each level >= 1's own
        sentinel class -- the depth head emits a +1 sentinel per level, but
        only level-0's is the real chunk-end marker; a non-zero-level
        sentinel would OOB the VQ codebook gather at decode time.
        """
        codes_row = torch.zeros(num_levels, dtype=torch.long, device=device)
        if rank == 0:
            sampled_codes = []
            base = offset_vals[0]
            hid = last_hidden.to(dtype=self.dtype)
            # Accumulator relative to the modality's own offset; filling one
            # slot per iteration makes the depth loop autoregressive.
            cum_ids = torch.zeros(1, num_levels, dtype=torch.long, device=device)
            for level in range(num_levels):
                logits = head(hid, cum_ids.to(hid.device), code_embed, level)
                level_logits = logits[0]
                sentinel_idx = None
                if mask_sentinel and level == 0:
                    sentinel_idx = self.visual_codebook_sizes[0]
                elif mask_audio_sentinels and level >= 1:
                    sentinel_idx = self.audio_codebook_sizes[level]
                if sentinel_idx is not None and sentinel_idx < level_logits.shape[-1]:
                    level_logits = level_logits.clone()
                    level_logits[sentinel_idx] = float("-inf")
                code = self._sample_audio_code(
                    level_logits,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    past_codes=None if past_codes is None else past_codes[:, level],
                )
                sampled_codes.append(int(code))
                cum_ids[0, level] = code + offset_vals[level] - base
            codes_row = torch.tensor(sampled_codes, device=device, dtype=torch.long)
        tp_group.broadcast(codes_row, src=0)
        return codes_row

    def _sample_cfg_visual_codes(
        self,
        code_embed: nn.Module,
        offset_vals: torch.Tensor,
        num_levels: int,
        cond_hidden: torch.Tensor,
        uncond_hidden: torch.Tensor,
        rank: int,
        tp_group: Any,
        device: torch.device,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float = 1.0,
        past_codes: torch.Tensor | None = None,
        cfg_scale: float = _DEFAULT_CFG_SCALE,
    ) -> torch.Tensor:
        """Sample visual codes with classifier-free guidance from a cond/uncond
        pair. Mirrors _sample_depth_head's rank-0+broadcast structure, but at
        every level combines cond/uncond logits as
        ``cfg_scale * (cond - uncond) + uncond`` (cond from the parent's
        hidden state, uncond from the twin's separate-KV-cache stream). Both
        streams share one sampled code per level to stay locked in sequence.
        """
        codes_row = torch.zeros(num_levels, dtype=torch.long, device=device)
        if rank == 0:
            sampled_codes = []
            base = offset_vals[0]
            cond_hid = cond_hidden.to(dtype=self.dtype)
            uncond_hid = uncond_hidden.to(dtype=self.dtype)
            head = self.visual_head
            cum_ids = torch.zeros(1, num_levels, dtype=torch.long, device=device)
            for level in range(num_levels):
                cond_logits = head(cond_hid, cum_ids.to(cond_hid.device), code_embed, level)[0]
                uncond_logits = head(uncond_hid, cum_ids.to(uncond_hid.device), code_embed, level)[0]
                combined = cfg_scale * (cond_logits - uncond_logits) + uncond_logits
                sentinel_idx = None
                if level == 0:
                    # Masked so the image can never self-terminate early --
                    # the grid bound in _advance_visual_gen is the terminator.
                    sentinel_idx = self.visual_codebook_sizes[0]
                if sentinel_idx is not None and sentinel_idx < combined.shape[-1]:
                    combined = combined.clone()
                    combined[sentinel_idx] = float("-inf")
                code = self._sample_audio_code(
                    combined,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    past_codes=None if past_codes is None else past_codes[:, level],
                )
                sampled_codes.append(int(code))
                cum_ids[0, level] = code + offset_vals[level] - base
            codes_row = torch.tensor(sampled_codes, device=device, dtype=torch.long)
        tp_group.broadcast(codes_row, src=0)
        return codes_row

    def talker_mtp(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        text_step: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Per-step audio code generation via audio_head (rank 0 + broadcast).

        Mirrors the reference's get_multimodal_logits_and_ids: runs the 8-level
        CasualDepthTransformerHead on rank 0 only (workaround for its hardcoded
        ``cuda:0``), broadcasts the sampled codes, then builds the 3-stream
        next-step embedding (ext_id_emb + text_emb + audio_embs, matching the
        reference's input_processor.py::get_audio_embeddings).
        """
        batch_size = input_ids.shape[0]
        if batch_size == 0:
            return inputs_embeds, None

        rank = get_tp_group().rank
        device = input_ids.device
        tp_group = get_tp_group()

        req_ids = kwargs.get("req_ids", [])
        codebook_sizes = self.audio_codebook_sizes
        num_levels = len(codebook_sizes)

        # Sampling params from subtalker_sampling_params; keys are explicit
        # None when unset (not absent), so coalesce below to the checkpoint's
        # per-modality generation_config defaults. Modality-specific keys win
        # over the generic fallback.
        do_sample = kwargs.get("do_sample", True)
        temperature = kwargs.get("temperature", 0.5)
        audio_top_k = kwargs.get("audio_top_k", kwargs.get("top_k", 5))
        audio_top_p = kwargs.get("audio_top_p", kwargs.get("top_p", 0.85))
        visual_top_k = kwargs.get("visual_top_k", kwargs.get("top_k", 1024))
        visual_top_p = kwargs.get("visual_top_p", kwargs.get("top_p", 0.75))
        audio_rep_penalty = kwargs.get("audio_repetition_penalty", kwargs.get("repetition_penalty", 1.3))
        visual_rep_penalty = kwargs.get("visual_repetition_penalty", kwargs.get("repetition_penalty", 1.0))
        if do_sample is None:
            do_sample = True
        if temperature is None:
            temperature = 0.5
        if audio_top_k is None:
            audio_top_k = 5
        if audio_top_p is None:
            audio_top_p = 0.85
        if visual_top_k is None:
            visual_top_k = 1024
        if visual_top_p is None:
            visual_top_p = 0.75
        if audio_rep_penalty is None:
            audio_rep_penalty = 1.3
        if visual_rep_penalty is None:
            visual_rep_penalty = 1.0
        # Diagnostic override for testing whether a repetition penalty
        # interrupts visual CFG collapse; the reference's own default is 1.0.
        _visual_rep_override = os.environ.get("LONGCAT_VISUAL_REP_PENALTY")
        if _visual_rep_override:
            visual_rep_penalty = float(_visual_rep_override)

        # Per-row results
        all_codes = torch.full((batch_size, num_levels), -1, dtype=torch.long, device=device)
        all_embeds = inputs_embeds.clone()

        # Materialise both modalities' code embedding tables on EVERY rank,
        # unconditionally: self.model.embed_tokens all-reduces across the TP
        # group, so building either lazily inside a rank-0-only block would
        # leave other ranks out of that collective and deadlock.
        audio_code_embed = self._ensure_audio_code_embed_module(device)
        visual_code_embed = self._ensure_visual_code_embed_module(device)

        # Visual CFG twin pairing: the engine admits one unconditional
        # companion per image request as f"{parent_id}__cfg_visual", sharing
        # the batch with its own KV cache/hidden state. When both are in
        # visual-gen state this step, combine their depth-head logits per
        # the CFG formula. All ranks derive the same pairing.
        row_of: dict[str, int] = {}
        for i, rid in enumerate(req_ids):
            if i < batch_size:
                row_of[rid] = i
        cfg_parent_of_twin: dict[str, str] = {}
        for rid in req_ids:
            if rid.endswith(_CFG_VISUAL_SUFFIX):
                cfg_parent_of_twin[rid[: -len(_CFG_VISUAL_SUFFIX)]] = rid

        # Rows whose all_codes/all_embeds were written by a parent's combined
        # CFG iteration (both the parent and its twin rows).
        cfg_handled_rows: set[int] = set()

        for row in range(batch_size):
            req_id = req_ids[row] if row < len(req_ids) else f"row_{row}"
            audio_state = self._audio_gen.get(req_id)
            visual_state = self._visual_gen.get(req_id)
            last_hidden = (
                last_talker_hidden[row : row + 1] if row < last_talker_hidden.shape[0] else last_talker_hidden[-1:]
            )

            if audio_state is not None:
                state = audio_state
                gen_step = state.get("gen_step", 0)
                audio_start = state.get("audio_start", False)
                terminal = state.get("terminal", False)
                ext_id = state.get("ext_id", AUDIOTEXT_PAD_TOKEN_ID)

                if terminal:
                    all_codes[row] = -1
                    continue

                past_codes = state.get("past_codes")
                past_codes_t = torch.stack(past_codes).to(device) if past_codes else None
                codes_row = self._sample_depth_head(
                    self.audio_head,
                    audio_code_embed,
                    self.audio_offset_vals,
                    num_levels,
                    last_hidden,
                    rank,
                    tp_group,
                    device,
                    do_sample,
                    temperature,
                    audio_top_k,
                    audio_top_p,
                    audio_rep_penalty,
                    past_codes_t,
                    mask_audio_sentinels=True,
                )

                deoff = int(codes_row[0].item())
                eoc_terminal = audio_start and (deoff >= codebook_sizes[0])
                max_gen_terminal = audio_start and gen_step >= self.max_gen
                is_terminal = eoc_terminal or max_gen_terminal

                # A terminal row becomes an explicit chunk-end marker row
                # (level-0 = codebook_sizes[0], same value _split_chunks
                # splits on) instead of a -1 discard sentinel -- -1 rows are
                # stripped before reaching the decoder, so without a real
                # boundary marker, a request that produces multiple audio
                # segments (max_gen can force a close and let generation
                # resume) would have its segments' codes concatenate with no
                # surviving boundary, overflowing the audio decoder's
                # fixed-size positional embedding table.
                frame_kept = audio_start and not is_terminal
                if is_terminal:
                    boundary_row = torch.zeros_like(codes_row)
                    boundary_row[0] = codebook_sizes[0]
                    all_codes[row] = boundary_row
                else:
                    all_codes[row] = codes_row if frame_kept else torch.full_like(codes_row, -1)

                if frame_kept:
                    # Repetition-penalty history: kept frames only.
                    state.setdefault("past_codes", []).append(codes_row.detach().cpu())

                # Build the 3-stream next-step embedding.
                # Stream 1: ext_id (audiotext_start/pad/audiogen_end).
                ext_tok = torch.tensor([ext_id], device=device, dtype=torch.long)
                ext_emb = self.model.embed_tokens(ext_tok)
                if ext_id == AUDIOTEXT_PAD_TOKEN_ID:
                    ext_emb.zero_()

                # Stream 2: visible text token, zeroed when it's audiotext_pad
                # (keyed off the token value so it's correct even for a pad
                # sampled before text_end is set).
                text_tok = input_ids[row : row + 1]
                text_emb = self.model.embed_tokens(text_tok)
                if int(text_tok.item()) == AUDIOTEXT_PAD_TOKEN_ID:
                    text_emb.zero_()

                # Stream 3: audio code embeddings, zeroed for invalid rows --
                # code 0 doubles as the clamped-invalid value.
                audio_emb = torch.zeros_like(ext_emb)
                if frame_kept and deoff != 0:
                    replicated_emb = self._ensure_replicated_audio_code_embedding(device)
                    offset_codes = codes_row + self.audio_offset_vals[:num_levels].to(device)
                    row_embs = []
                    for level in range(num_levels):
                        idx = (offset_codes[level] - self.audio_offset_vals[0]).item()
                        if 0 <= idx < replicated_emb.shape[0]:
                            row_embs.append(replicated_emb[idx : idx + 1])
                    if row_embs:
                        audio_emb = torch.cat(row_embs, dim=0).sum(dim=0, keepdim=True)

                # Sum the 3 streams
                next_emb = ext_emb + text_emb + audio_emb
                all_embeds[row : row + 1] = next_emb.to(dtype=self.dtype)

                if is_terminal:
                    state["terminal"] = True
                    # Force the closing tag on the terminal step itself, so
                    # compute_logits' terminal branch emits
                    # <longcat_audiogen_end> instead of leaving EOS unbanned
                    # with no forced closure (which let the model end the
                    # whole request instead of just this audio segment).
                    state["ext_id"] = AUDIOGEN_END_TOKEN_ID

            elif visual_state is not None:
                state = visual_state
                gen_step = state.get("gen_step", 0)
                terminal = state.get("terminal", False)
                ext_id = state.get("ext_id", IMG_PAD_TOKEN_ID)
                is_row_boundary = ext_id == IMG_NEWLINE_TOKEN_ID

                if terminal:
                    all_codes[row] = -1
                    continue

                visual_codebook_sizes = self.visual_codebook_sizes
                visual_num_levels = len(visual_codebook_sizes)

                # This row's codes/embedding were already written by its
                # parent's combined CFG iteration.
                if row in cfg_handled_rows:
                    continue

                # Resolve the (parent, twin) pair this row belongs to, if
                # any; only combine when both are present and non-terminal,
                # otherwise fall back to independent sampling.
                parent_req_id: str | None = None
                twin_req_id: str | None = None
                if req_id in cfg_parent_of_twin:
                    parent_req_id, twin_req_id = req_id, cfg_parent_of_twin[req_id]
                elif req_id.endswith(_CFG_VISUAL_SUFFIX):
                    parent_req_id, twin_req_id = req_id[: -len(_CFG_VISUAL_SUFFIX)], req_id

                p_row: int | None = None
                t_row: int | None = None
                if parent_req_id is not None and twin_req_id is not None:
                    p_row = row_of.get(parent_req_id)
                    t_row = row_of.get(twin_req_id)
                p_state = self._visual_gen.get(parent_req_id) if parent_req_id is not None else None
                t_state = self._visual_gen.get(twin_req_id) if twin_req_id is not None else None
                both_visual_active = (
                    parent_req_id is not None
                    and p_row is not None
                    and t_row is not None
                    and p_state is not None
                    and t_state is not None
                    and not bool(p_state.get("terminal", False))
                    and not bool(t_state.get("terminal", False))
                )

                # This row belongs to a cond/uncond pair but both streams
                # aren't active this step (parent/twin are independently
                # scheduled, not a hard same-step guarantee), so it falls
                # back to independent sampling. Log unconditionally since
                # this only fires on a real desync, not every step.
                if parent_req_id is not None and not both_visual_active:
                    logger.warning(
                        "[longcat-image] CFG desync: req=%s parent=%s(row=%s,state=%s,terminal=%s) "
                        "twin=%s(row=%s,state=%s,terminal=%s) -- falling back to independent "
                        "(non-CFG) sampling for this step",
                        req_id,
                        parent_req_id,
                        p_row,
                        p_state is not None,
                        bool(p_state.get("terminal", False)) if p_state is not None else None,
                        twin_req_id,
                        t_row,
                        t_state is not None,
                        bool(t_state.get("terminal", False)) if t_state is not None else None,
                    )

                if both_visual_active and row == p_row:
                    # Combined CFG path (drives both streams).
                    p_last_hidden = (
                        last_talker_hidden[p_row : p_row + 1]
                        if p_row < last_talker_hidden.shape[0]
                        else last_talker_hidden[-1:]
                    )
                    t_last_hidden = (
                        last_talker_hidden[t_row : t_row + 1]
                        if t_row < last_talker_hidden.shape[0]
                        else last_talker_hidden[-1:]
                    )
                    p_past = p_state.get("past_codes")
                    p_past_t = torch.stack(p_past).to(device) if p_past else None
                    cfg_scale = float(p_state.get("cfg_scale", _DEFAULT_CFG_SCALE))
                    codes_row = self._sample_cfg_visual_codes(
                        visual_code_embed,
                        self.visual_offset_vals,
                        visual_num_levels,
                        p_last_hidden,
                        t_last_hidden,
                        rank,
                        tp_group,
                        device,
                        do_sample,
                        temperature,
                        visual_top_k,
                        visual_top_p,
                        visual_rep_penalty,
                        p_past_t,
                        cfg_scale,
                    )

                    deoff = int(codes_row[0].item())
                    eoc_terminal = deoff >= visual_codebook_sizes[0]
                    frame_kept = not is_row_boundary and not eoc_terminal
                    is_terminal = eoc_terminal
                    if frame_kept:
                        p_state.setdefault("past_codes", []).append(codes_row.detach().cpu())

                    # Both streams share the combined sample; mirror the
                    # twin's grid state onto the parent's so both terminate
                    # on the same step and see the same forced visible token.
                    for _sync_key in ("gen_step", "ext_id", "token_w", "token_h"):
                        if _sync_key in p_state:
                            t_state[_sync_key] = p_state[_sync_key]
                    t_state["terminal"] = bool(is_terminal or p_state.get("terminal", False))

                    for rr in (p_row, t_row):
                        all_codes[rr] = codes_row if frame_kept else torch.full_like(codes_row, -1)

                    # Next-step embedding built once from the shared codes
                    # and applied to both rows.
                    text_tok = input_ids[p_row : p_row + 1]
                    text_emb = self.model.embed_tokens(text_tok)
                    if frame_kept and deoff != 0:
                        offset_codes = (codes_row + self.visual_offset_vals[:visual_num_levels].to(device)).unsqueeze(0)
                        vision_emb = self._code_embeddings(offset_codes)
                        vision_emb = self.visual_tokenizer.visual_embedding_layer(vision_emb.to(self.dtype))
                        next_emb = vision_emb
                    else:
                        next_emb = text_emb
                    for rr in (p_row, t_row):
                        all_embeds[rr : rr + 1] = next_emb.to(dtype=self.dtype)

                    if is_terminal:
                        p_state["terminal"] = True
                    cfg_handled_rows.update({p_row, t_row})
                    continue

                if both_visual_active and row == t_row:
                    # Twin processed before its parent in this batch: defer to
                    # the parent's combined iteration, which writes this row.
                    cfg_handled_rows.add(t_row)
                    continue

                past_codes = state.get("past_codes")
                past_codes_t = torch.stack(past_codes).to(device) if past_codes else None
                codes_row = self._sample_depth_head(
                    self.visual_head,
                    visual_code_embed,
                    self.visual_offset_vals,
                    visual_num_levels,
                    last_hidden,
                    rank,
                    tp_group,
                    device,
                    do_sample,
                    temperature,
                    visual_top_k,
                    visual_top_p,
                    visual_rep_penalty,
                    past_codes_t,
                    mask_sentinel=True,
                )

                # The level-0 sentinel is masked in _sample_depth_head so this
                # never fires; the grid bound in _advance_visual_gen is the
                # real terminator. Defensive guard for any non-masked path.
                deoff = int(codes_row[0].item())
                eoc_terminal = deoff >= visual_codebook_sizes[0]

                # A row-boundary (newline) step never carries a real pixel.
                frame_kept = not is_row_boundary and not eoc_terminal
                all_codes[row] = codes_row if frame_kept else torch.full_like(codes_row, -1)

                is_terminal = eoc_terminal
                if frame_kept:
                    state.setdefault("past_codes", []).append(codes_row.detach().cpu())

                # Unlike audio's 3-way sum, this is a masked replace: the
                # visible token (IMAGE_PAD/IMAGE_NEWLINE) keeps its normal
                # embedding at a newline, or is fully replaced by the vision
                # embedding at a real pixel. No "ext" stream here.
                text_tok = input_ids[row : row + 1]
                text_emb = self.model.embed_tokens(text_tok)

                if frame_kept and deoff != 0:
                    # Reuses the understanding-direction path (_encode_images).
                    offset_codes = (codes_row + self.visual_offset_vals[:visual_num_levels].to(device)).unsqueeze(0)
                    vision_emb = self._code_embeddings(offset_codes)
                    vision_emb = self.visual_tokenizer.visual_embedding_layer(vision_emb.to(self.dtype))
                    next_emb = vision_emb
                else:
                    next_emb = text_emb
                all_embeds[row : row + 1] = next_emb.to(dtype=self.dtype)

                if is_terminal:
                    state["terminal"] = True

            else:
                continue

        # Return this step's codes, not an accumulation -- the output
        # processor concatenates per-step rows into the final [T, 8] tensor;
        # returning the running total would grow the result quadratically.
        return all_embeds, all_codes

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is None or not (self._audio_gen or self._visual_gen):
            return logits
        # Suppress EOS and force visible tokens during audio/image-gen. Skip
        # in prefill (logits has many rows but at most 1 active-gen request)
        # to avoid misaligning row 0. A request is in at most one of the two
        # dicts, so dict union order gives an unambiguous row->request map.
        req_ids = list(self._audio_gen.keys()) + list(self._visual_gen.keys())
        if not req_ids:
            return logits
        num_logits = logits.shape[0]
        if num_logits != len(req_ids):
            return logits
        for row in range(num_logits):
            req_id = req_ids[row]
            audio_state = self._audio_gen.get(req_id)
            visual_state = self._visual_gen.get(req_id)
            if audio_state is not None:
                if audio_state.get("terminal"):
                    # Force the closing tag instead of leaving EOS unbanned
                    # with no replacement, which would end the whole request
                    # instead of just closing this audio segment.
                    if self._eos_id < logits.shape[-1]:
                        logits[row, self._eos_id] = float("-inf")
                    forced_id = audio_state.get("ext_id", AUDIOGEN_END_TOKEN_ID)
                    logits[row, :] = float("-inf")
                    logits[row, forced_id] = 0.0
                    continue
                # Ban EOS so it never terminates generation during audio
                if self._eos_id < logits.shape[-1]:
                    logits[row, self._eos_id] = float("-inf")
                if audio_state.get("text_end"):
                    logits[row, :] = float("-inf")
                    logits[row, AUDIOTEXT_PAD_TOKEN_ID] = 0.0
            elif visual_state is not None:
                # Image gen has no unforced phase: every GEN_IMAGE_STAGE step
                # is forced (IMAGE_PAD/NEWLINE/END). The terminal grid-end step
                # carries ext_id=IMAGE_END, closing the image with
                # <longcat_img_end>, which pops the state next step.
                if self._eos_id < logits.shape[-1]:
                    logits[row, self._eos_id] = float("-inf")
                forced_id = visual_state.get("ext_id", IMG_PAD_TOKEN_ID)
                logits[row, :] = float("-inf")
                logits[row, forced_id] = 0.0
        return logits

    def get_expert_mapping(self):
        return self.model.get_expert_mapping()

    # ------------------------------------------------------------------ #
    # weights
    # ------------------------------------------------------------------ #

    _SIDE_MODULE_PREFIXES = (
        ("model.visual_tokenizer.", "visual_tokenizer"),
        ("model.audio_tokenizer.", "audio_tokenizer"),
        ("visual_head.", "visual_head"),
        ("audio_head.", "audio_head"),
    )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        side_state: dict[str, dict[str, torch.Tensor]] = {attr: {} for _, attr in self._SIDE_MODULE_PREFIXES}

        def split(weights):
            for name, tensor in weights:
                for ckpt_prefix, attr in self._SIDE_MODULE_PREFIXES:
                    if name.startswith(ckpt_prefix):
                        side_state[attr][name[len(ckpt_prefix) :]] = tensor
                        break
                else:
                    yield name, tensor

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["model.mtp."],
            skip_substrs=["visual_tokenizer", "audio_tokenizer", "visual_head", "audio_head"],
        )
        loaded = loader.load_weights(split(weights))

        # Reapply the LongCat-specific MLA LoRA input scales here,
        # unconditionally, after AutoWeightsLoader has fully finished.
        #
        # The vllm fork's own FlashModel.load_weights also applies this
        # scale, guarded by a permanent "already scaled" flag -- but it gets
        # invoked more than once per model instance, and the guard trips on
        # an earlier pass whose weights contain no self_attn keys at all
        # (scaling the un-loaded default-init value), before the real data
        # lands via a plain copy_ later and silently wipes that scaling out,
        # with the guard never re-triggering. Applying the scale exactly
        # once here, after the loader is fully done and gated by our own
        # (distinct) guard, is correct regardless of how many internal
        # passes happened upstream, with no vLLM changes needed.
        model = getattr(self, "model", None)
        layers = getattr(model, "layers", None)
        if layers is not None:
            hf = getattr(self.config, "hidden_size", None) or 3072
            q_lora = int(getattr(self.config, "q_lora_rank", 0) or 0)
            kv_lora = int(getattr(self.config, "kv_lora_rank", 0) or 0)
            do_q = bool(getattr(self.config, "mla_scale_q_lora", False)) and q_lora > 0
            do_kv = bool(getattr(self.config, "mla_scale_kv_lora", False)) and kv_lora > 0
            scale_q = (hf / q_lora) ** 0.5 if do_q else None
            scale_kv = (hf / kv_lora) ** 0.5 if do_kv else None
            for layer in layers:
                attns = getattr(layer, "self_attn", None)
                if attns is None:
                    continue
                for attn in attns:
                    if scale_q is not None and not getattr(attn, "_omni_mla_q_scaled", False):
                        attn.q_a_layernorm.weight.data.mul_(scale_q)
                        attn._omni_mla_q_scaled = True
                    if scale_kv is not None and not getattr(attn, "_omni_mla_kv_scaled", False):
                        attn.kv_a_layernorm.weight.data.mul_(scale_kv)
                        attn._omni_mla_kv_scaled = True

        device = next(self.model.parameters()).device
        for ckpt_prefix, attr in self._SIDE_MODULE_PREFIXES:
            module: nn.Module = getattr(self, attr)
            state = side_state[attr]
            if not state:
                logger.warning("No checkpoint weights found for %s (%s*)", attr, ckpt_prefix)
                continue
            missing, unexpected = module.load_state_dict(state, strict=False)
            if missing:
                logger.warning("%s: %d missing keys (e.g. %s)", attr, len(missing), missing[:3])
            if unexpected:
                logger.warning("%s: %d unexpected keys (e.g. %s)", attr, len(unexpected), unexpected[:3])
            module.to(device=device, dtype=self.dtype)
            module.eval()
            loaded.update(f"{ckpt_prefix}{k}" for k in state)

        # Mark remote-code submodules as loaded (weights come from this
        # method's own state dict above, not vLLM's tracked loader), so
        # track_weights_loading doesn't raise for "uninitialized" weights.
        _skip_substrs = ("visual_tokenizer", "audio_tokenizer", "visual_head", "audio_head")
        for name, _ in self.named_parameters():
            if any(s in name for s in _skip_substrs):
                loaded.add(name)

        return loaded
