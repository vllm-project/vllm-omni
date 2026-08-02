"""LongCat-Next thinker stage — native vLLM port.

The 74B MoE+MLA backbone runs on vLLM's native ``FlashNgramModel``
(``vllm/model_executor/models/longcat_flash_ngram.py``), so tensor parallelism,
paged KV cache and the fused MoE kernels all apply. On top of the backbone:

- **n-gram embedding fusion** — upstream's ``LongcatFlashNgramForCausalLM``
  computes it in an MRV2 ``ModelState``, but vllm-omni forces the V1 runner.
  Here it runs in the omni runner's per-request ``preprocess`` hook, which
  provides ``request_id`` + per-span token ids each step; per-request left
  context (last n-1 tokens, EOS-negated) lives on the model and is cleaned up
  via ``on_requests_finished``. Ids are computed with the same
  ``ngram_compute_n_gram_ids`` CUDA kernel the native model state uses.
- **multimodal understanding** — the checkpoint's remote-code visual/audio
  tokenizers (encoders + VQ bridges) run per rank; code embeddings are
  ``embed_tokens(ids).sum(levels)`` (+ the visual bridge), merged at
  ``<longcat_img_pad>`` / ``<longcat_audio_pad>`` positions through vLLM's
  standard is_multimodal mask path. Placeholder positions keep the pure
  multimodal embedding (no n-gram fusion), matching the HF forward.
- **lm_head** is 131125-wide (text + special tokens) per the checkpoint.
- **Audio generation** (speech synthesis) uses ``talker_mtp``: when the model
  emits ``<longcat_audiogen_start>``, ``preprocess`` emits ``mtp_inputs`` so
  the runner calls ``talker_mtp()`` each decode step. ``talker_mtp`` runs the
  checkpoint's 8-level ``audio_head`` (rank 0 only + broadcast to avoid TP
  deadlock from the checkpoint's hardcoded ``cuda:0``) and accumulates
  per-frame codes into ``model_intermediate_buffer["codes"]["audio"]``.
  Visible-token control (EOS suppression, audiotext_pad forcing) lives in
  ``compute_logits`` to work on the async-scheduling path.
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
    AUDIOGEN_END_TOKEN_ID,
    AUDIOGEN_START_TOKEN_ID,
    AUDIO_PAD_TOKEN_ID,
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

# Visual CFG twin-stream suffix: the companion request admitted by the
# engine's CFG expansion carries request_id = f"{parent_id}__cfg_visual"
# (see stage_input_processors/longcat_next.py::expand_longcat_cfg_prompts).
# The model uses it to pair the unconditional stream with its parent so it can
# combine the two depth-head logit streams per step.
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
    # This model shares one backbone for text and audio: it emits "mtp_inputs"
    # only while in audio-generation mode, so the runner must skip the talker
    # decode on text-only steps (gpu_model_runner._talker_mtp_forward).
    omits_talker_mtp_inputs_when_idle = True
    # preprocess() needs raw token ids (n-gram hashing operates on token ids,
    # not embeddings) even on the supports_mm_inputs path, where upstream
    # vLLM's _prepare_mm_inputs otherwise sets input_ids=None and passes only
    # embeddings (vllm/v1/worker/gpu_model_runner.py::_prepare_mm_inputs).
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
        # FlashModel/FlashConfig field alignment: the checkpoint names the MLP
        # width ffn_hidden_size and omits pad_token_id (generation_config has 3).
        hf.intermediate_size = getattr(hf, "ffn_hidden_size", getattr(hf, "intermediate_size", None))
        if getattr(hf, "pad_token_id", None) is None:
            hf.pad_token_id = _DEFAULT_PAD_TOKEN_ID
        # With deploy-config `quantization: fp8` (needed to fit the ~68GB MoE
        # experts on 40GB-class GPUs), FP8/Marlin online quantization applies
        # to every ColumnParallelLinear/RowParallelLinear by default —
        # including MLA's kv_b_proj. FlashDecoderLayer's native weight-loading
        # fixup (longcat_flash.py) unflattens/splits kv_b_proj.weight into
        # w_kc/w_vc assuming an unpacked bf16 layout; Marlin's packed FP8
        # layout breaks that reshape. FlashDecoderLayer already supports
        # excluding modules from quantization via `disable_quant_module`
        # (checked for "self_attn" and "mlps"); keep MLA attention in native
        # bf16 and let the MoE experts (the actual memory driver) quantize.
        disable_quant_module = list(getattr(hf, "disable_quant_module", []) or [])
        if "self_attn" not in disable_quant_module:
            disable_quant_module.append("self_attn")
        hf.disable_quant_module = disable_quant_module
        self.config = hf
        self.quant_config = vllm_config.quant_config

        # --- native TP-sharded backbone -------------------------------------
        # LongCat-Next sizes/hashes the n-gram tables with text_vocab_size
        # (131072), not the full 282624 vocab (the checkpoint's
        # modeling_longcat_ngram.py literally comments out the vocab_size
        # variant). The native NgramEmbedding uses config.vocab_size, which
        # here would allocate a ~135 GB table and break every hash — so skip
        # the base class's ngram build and attach a correctly-sized one.
        self.text_vocab_hash_size = int(getattr(hf, "text_vocab_size", 131072))
        ngram_ratio = hf.ngram_vocab_size_ratio
        hf.ngram_vocab_size_ratio = None
        try:
            self.model = FlashNgramModel(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
            )
        finally:
            hf.ngram_vocab_size_ratio = ngram_ratio
        if get_pp_group().is_first_rank:
            ngram_cfg = FlashConfig(**hf.__dict__)
            ngram_cfg.vocab_size = self.text_vocab_hash_size
            self.model.ngram_embeddings = NgramEmbedding(ngram_cfg, self.model.embed_tokens)

        self.text_vocab_size = int(
            getattr(hf, "text_vocab_plus_multimodal_special_token_size", 131125)
        )
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
        visual_tok_cls = get_remote_attr(
            self.model_path, "modular_longcat_next_visual", "LongcatNextVisualTokenizer"
        )
        audio_tok_cls = get_remote_attr(
            self.model_path, "modular_longcat_next_audio", "LongcatNextAudioTokenizer"
        )
        head_cls = get_remote_attr(
            self.model_path, "modular_longcat_next", "CasualDepthTransformerHead"
        )
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

        # Cache the audio codebook sizes off the *remote* config. self.config
        # is vllm-omni's registered shim, whose audio_config/visual_config are
        # plain dicts (see load_remote_hf_config's docstring), so
        # self.config.audio_config.vq_config would raise AttributeError at
        # decode time -- only the remote config carries real sub-config objects.
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
        self._ngram_audited = False
        self._ngram_disabled = int(os.environ.get("LONGCAT_NGRAM_DISABLE", "0")) != 0
        self._logits_audited = False
        self._max_ctx_entries = 4 * max(
            int(getattr(vllm_config.scheduler_config, "max_num_seqs", 64)), 64
        )
        assert ngram is not None, "LongCat-Next requires ngram embeddings"

        # --- audio generation state (talker_mtp) ---
        self._audio_gen: dict[str, dict[str, Any]] = {}
        self._audio_delay_default = 0
        max_audio_seconds = getattr(ac, "max_audio_seconds", 30)
        # 25 fps, matches reference. LONGCAT_MAX_GEN caps it for bounded
        # validation runs (a full 30 s chunk is ~750 decode steps, which can
        # outlast a short job walltime); unset it for full-length audio.
        self.max_gen = int(os.environ.get("LONGCAT_MAX_GEN") or max_audio_seconds * 25)
        self._audio_debug = int(os.environ.get("LONGCAT_AUDIO_DEBUG", "0")) != 0

        # --- image generation state (talker_mtp dispatches to this too) ---
        self._visual_gen: dict[str, dict[str, Any]] = {}
        # generation_config.json's image_generation_config is None -- the
        # checkpoint does not declare a default grid size at all (confirmed:
        # `json.load(...)['image_generation_config']` is None). The reference
        # reads token_h/token_w per REQUEST (req.input_extra_infos[0]), so
        # this is a caller-supplied parameter, not an internal constant.
        # This fallback (37x37, matching the image DECODER stage's own
        # generation_config-default grid) only applies if a request doesn't
        # supply one via sampling_params.extra_args; real callers should
        # supply token_w explicitly once this path is GPU-verified.
        self._default_token_w = 37
        self._default_token_h = 37
        # NOT pre-declared here (e.g. `= None`): nn.Module.register_buffer
        # raises KeyError("attribute already exists") if the name is already
        # a plain instance attribute, even if its value is None. It must stay
        # entirely absent from __dict__/_buffers until the first real
        # register_buffer call in _ensure_replicated_audio_code_embedding.
        # Debug tallies: every frame sampled by talker_mtp should be either
        # kept or explicitly discarded, and every kept frame should be emitted
        # by make_omni_output exactly once. Divergence between these three
        # numbers localises a drop to a specific boundary.
        self._dbg_sampled = 0
        self._dbg_kept = 0
        self._dbg_emitted = 0

    @staticmethod
    def _dbg_step(step: int) -> bool:
        """Log the first few steps in full, then sample, to bound log size."""
        return step < 12 or step % 100 == 0

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

    def _encode_images(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> list[torch.Tensor]:
        # `pixel_values` is a single flat [total_patches, D] tensor holding
        # every image's patches concatenated on dim 0 (per
        # MultiModalFieldConfig.flat_from_sizes); `grid_thw` is [num_images, 3].
        # The vision encoder consumes/produces the whole flat batch in one
        # call, so we split its *output* back into per-image chunks rather
        # than looping the encoder per image.
        device = self.visual_offset_vals.device
        grid_thw = grid_thw.reshape(-1, 3).to(device)
        # Not a `visual_config` field (checked: absent from config.json under
        # both "merge_size" and "spatial_merge_size") — `VisualEncoder.__init__`
        # reads `config.merge_size` with a `2` fallback and stores it as
        # `self.merge_size`; read the constructed module's own attribute.
        merge_size = getattr(self.visual_tokenizer.visual_model, "merge_size", 2)
        sizes = (grid_thw.prod(dim=-1) // (merge_size ** 2)).tolist()

        with torch.inference_mode():
            ids = self.visual_tokenizer.encode(
                pixel_values.to(device=device, dtype=self.dtype), grid_thw
            )
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
        # `features` is [num_chunks, mel_bins, T] — every audio clip's chunks
        # concatenated on dim 0 (per flat_from_sizes); `encoder_lengths` /
        # `bridge_lengths` are [num_chunks]. The audio encoder consumes the
        # whole flat batch in one call; split its output per bridge_length.
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
        # SupportsMultiModal's abstract method in this vLLM tree is
        # `embed_multimodal` (called directly by gpu_model_runner.py, incl.
        # profile_run's dummy-batch pass) — an earlier `get_multimodal_embeddings`
        # name here was never invoked, silently falling through to the
        # interface's `...`-bodied stub (implicit `None` return), which broke
        # engine profiling with "Expected multimodal embeddings to be a
        # list/tuple ... but got NoneType".
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

    def _span_oe_ids(
        self, request_id: str, span_ids: torch.Tensor, fresh: bool
    ) -> torch.Tensor:
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
        self._ngram_ctx[request_id] = table[0, -self._ctx_len:].clone()
        if len(self._ngram_ctx) > self._max_ctx_entries:
            for stale in list(self._ngram_ctx)[: len(self._ngram_ctx) - self._max_ctx_entries]:
                self._ngram_ctx.pop(stale, None)

        # n-gram audit: the reference runs the SAME kernel over the full
        # prompt in one call; this port streams it span-by-span with a rolling
        # _ctx_len left context. A mismatch in that streaming (fresh ctx
        # init, _neg_eos sign convention, or span boundaries) would shift the
        # fused embedding for every token and produce "coherent but generic"
        # text even with identical input_ids. Log the first span's computed
        # oe_ids (token x num_embedders) so a pod run can diff them against
        # the reference kernel's output for the same prompt.
        if self._audio_debug and fresh and not self._ngram_audited:
            self._ngram_audited = True
            logger.info(
                "[longcat-ngram] req=%s first-span oe_ids[:4]=%s ngram_n=%d ctx_len=%d "
                "num_embedders=%d",
                request_id, oe_ids[:4].tolist(), self._ngram_n, self._ctx_len,
                int(oe_ids.shape[1]),
            )
        return oe_ids.long()

    def _advance_audio_gen(
        self, request_id: str, last_token: int, device: torch.device, dtype: torch.dtype,
        decode_eligible: bool = True,
    ) -> dict[str, Any]:
        """Update audio-gen state based on the last emitted visible token.
        Returns update_dict entries for the runner.

        ``decode_eligible`` must mirror the runner's own talker_mtp dispatch
        gate (``span_len == 1 and not is_prefill``, gpu_model_runner.py
        ~line 1744/1767): when ``<longcat_audiogen_start>`` lands as the
        LAST token of a multi-token prefill chunk (e.g. it was written
        literally into the prompt, as every debug script does), preprocess()
        still runs and still sees last_token==AUDIOGEN_START_TOKEN_ID, but
        the runner will NOT call talker_mtp for this same step (prefill
        steps never go through the decode/talker_mtp path). Unconditionally
        advancing gen_step here (as an earlier version did) desyncs the
        state machine's step counter from the number of talker_mtp calls
        that actually happen: gen_step race ahead by 1 for a step whose
        code was never sampled, permanently losing that frame with no error
        -- silently producing one fewer real frame than the reference
        expects for the rest of the request. Guarding the whole advance
        block on decode_eligible defers gen_step's first 0->1 transition to
        the first REAL decode step, keeping it 1:1 with talker_mtp
        invocations. The freshly created state's default ext_id
        (AUDIOTEXT_PAD_TOKEN_ID) is already the correct forced token for
        this ineligible step, so skipping the advance costs nothing.
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
            if self._audio_debug:
                logger.info("[longcat-audio] req=%s audio_gen created (delay=%d)",
                            request_id, self._audio_delay_default)
        elif last_token == AUDIOGEN_END_TOKEN_ID:
            self._audio_gen.pop(request_id, None)
            if self._audio_debug:
                logger.info("[longcat-audio] req=%s audio_gen ended", request_id)

        state = self._audio_gen.get(request_id)
        if state is not None and not state["terminal"] and decode_eligible:
            # 0-based index of *this* step. The reference captures gen_step
            # before advancing the state machine (output_processor.py:218) and
            # compares that pre-increment value against delay, so every
            # comparison below uses it rather than the incremented counter.
            gen_step = state["gen_step"]
            state["gen_step"] = gen_step + 1

            # The first AUDIOTEXT_PAD sampled as the visible token marks the
            # end of the spoken transcript (reference output_processor.py:
            # 233-237). From then on compute_logits pins the text stream to
            # pad, and for a deferred (inf) delay this is also where audio
            # starts.
            if not state["text_end"] and last_token == AUDIOTEXT_PAD_TOKEN_ID:
                state["text_end"] = True
                state["delay"] = min(state["delay"], gen_step)
                if self._audio_debug:
                    logger.info(
                        "[longcat-audio] req=%s TEXT_END at gen_step=%d (delay->%s)",
                        request_id, gen_step, state["delay"],
                    )

            # ext stream + audio gating (reference output_processor.py:242-251).
            # The reference enables audio at the *end* of step == delay, after
            # that step's codes were already discarded, so the first real frame
            # lands on delay+1 — hence the strict '>' rather than '>='.
            delay = state["delay"]
            state["ext_id"] = (
                AUDIOTEXT_START_TOKEN_ID if gen_step == delay else AUDIOTEXT_PAD_TOKEN_ID
            )
            state["audio_start"] = gen_step > delay
            if self._audio_debug and self._dbg_step(gen_step):
                logger.info(
                    "[longcat-audio] req=%s advance step=%d last_token=%d "
                    "ext_id=%d audio_start=%s text_end=%s delay=%s",
                    request_id, gen_step, last_token, state["ext_id"],
                    state["audio_start"], state["text_end"], delay,
                )
            # Emit mtp_inputs so the runner calls talker_mtp this step.
            # last_hidden from previous forward (stashed by make_omni_output),
            # or zeros if this is the first step after audio_gen creation.
            last_hidden = state.get("last_hidden")
            if last_hidden is None:
                last_hidden = torch.zeros(
                    1, self.config.hidden_size, device=device, dtype=dtype
                )
            text_step = torch.zeros(
                1, self.config.hidden_size, device=device, dtype=dtype
            )
            update["mtp_inputs"] = (last_hidden, text_step)
        return update

    def _advance_visual_gen(
        self, request_id: str, last_token: int, device: torch.device, dtype: torch.dtype,
        token_w: int | None = None, token_h: int | None = None,
        cfg_scale: float | None = None,
        decode_eligible: bool = True,
    ) -> dict[str, Any]:
        """Update image-gen state based on the last emitted visible token.

        ``decode_eligible`` mirrors the runner's talker_mtp dispatch gate
        (``span_len == 1 and not is_prefill``): see _advance_audio_gen's
        docstring for the full rationale -- the identical bug applies here.
        When ``<longcat_img_start>`` is the last token of a multi-token
        prefill chunk (true whenever it's written into the prompt itself,
        as longcat_next_debug_quality.py's run_image does), advancing
        gen_step unconditionally raced the step counter one step ahead of
        the number of talker_mtp calls the runner actually makes, silently
        dropping the FIRST pixel's code (observed: 1368 kept codes vs the
        expected 37x37=1369, crashing the reference image decoder's
        positions_2d assert). Gating the advance on decode_eligible defers
        gen_step's 0->1 transition to the first real decode step, which is
        exactly 1:1 with talker_mtp invocations.

        Mirrors _advance_audio_gen's role, but the reference's image state
        machine (state_machine.py's GenImageStageStage) is a simpler, single
        continuous stage -- no delay/text_end/NEXT_*_STAGE loop-back like
        audio has. It tracks a 2D grid instead: every (token_w + 1)-th step
        is a row boundary, forced to IMAGE_NEWLINE with its sampled pixel
        discarded (output_processor.py:204-216); every other step is a real
        pixel, forced to IMAGE_PAD.

        Termination is the grid bound: once gen_step reaches
        token_h * (token_w + 1), all rows are emitted, so the step is marked
        terminal and its visible token forced to IMAGE_END (in compute_logits).
        The reference instead terminates on ``multi_ids[0] ==
        IMAGE_END_TOKEN_ID`` (state_machine.py:84), but that is unreachable
        because the level-0 end-of-image sentinel class is masked out of the
        head output (output_processor.py:312), so a request only ends when the
        caller's max_new_tokens runs out -- the source of the image overrun.
        """
        update: dict[str, Any] = {}
        if last_token == IMG_START_TOKEN_ID:
            # The checkpoint's visual_generation_config defaults to a 37x37
            # token grid, but the real canvas is supplied per-request via
            # additional_information (token_w/token_h) AND via the
            # "<longcat_img_token_size>{h} {w}</longcat_img_token_size>"
            # anyres prefix in the prompt. A missing token_w/token_h means
            # the caller also omitted the prefix, so the model is generating
            # without knowing its canvas -- the #1 cause of image content
            # drifting from the description.
            is_cfg_twin = request_id.endswith(_CFG_VISUAL_SUFFIX)
            if token_w is None or token_h is None:
                # The uncond CFG twin has no additional_information of its
                # own; inherit the parent's canvas (and cfg_scale) so both
                # streams decode the same grid. If the parent's state is not
                # visible yet (batch ordering), the defaults below apply and
                # talker_mtp's CFG sync corrects the twin on the first
                # combined step anyway.
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
                        request_id, self._default_token_w, self._default_token_h,
                    )
            self._visual_gen[request_id] = {
                "gen_step": 0,
                "token_w": token_w or self._default_token_w,
                "token_h": token_h or self._default_token_h,
                "cfg_scale": cfg_scale if cfg_scale is not None else _DEFAULT_CFG_SCALE,
                "ext_id": IMG_PAD_TOKEN_ID,
                "terminal": False,
            }
            if self._audio_debug:
                logger.info("[longcat-image] req=%s visual_gen created (token_w=%s, token_h=%s)",
                            request_id, self._visual_gen[request_id]["token_w"],
                            self._visual_gen[request_id]["token_h"])
        elif last_token == IMG_END_TOKEN_ID:
            self._visual_gen.pop(request_id, None)
            if self._audio_debug:
                logger.info("[longcat-image] req=%s visual_gen ended", request_id)

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
                if self._audio_debug:
                    logger.info(
                        "[longcat-image] req=%s advance step=%d GRID-END terminal "
                        "(token_h=%d token_w=%d grid_end=%d)",
                        request_id, gen_step, state["token_h"], token_w, grid_end,
                    )
            else:
                is_row_boundary = state["gen_step"] % (token_w + 1) == 0
                state["ext_id"] = IMG_NEWLINE_TOKEN_ID if is_row_boundary else IMG_PAD_TOKEN_ID
                if self._audio_debug and self._dbg_step(gen_step):
                    logger.info(
                        "[longcat-image] req=%s advance step=%d last_token=%d ext_id=%d row_boundary=%s",
                        request_id, gen_step, last_token, state["ext_id"], is_row_boundary,
                    )
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

        # Tokenization audit: the reference HF path tokenizes text prompts
        # with add_special_tokens=True (HF default), which prepends BOS when
        # the tokenizer config defines one. vLLM-omni's text-only path may
        # differ, and a 1-token shift changes every n-gram hash context --
        # producing "coherent but generic" text (observed: reference answers
        # properly, port emits filler). Log the first span's raw ids once per
        # request so a pod run can compare against the reference's tokenizer
        # output directly.
        if self._audio_debug and num_computed == 0:
            logger.info(
                "[longcat-text] req=%s first-span input_ids[:16]=%s span=%d ignored_first=%d",
                request_id, input_ids[:16].tolist(), int(input_ids.shape[0]),
                int((input_ids[0] >= self.text_vocab_hash_size).item()) if input_ids.numel() else -1,
            )

        # oe_ignored ids (all special tokens >= text_vocab_size, incl. the
        # mm pad markers) are hash-segment boundaries and take *pure* word
        # embeddings — no n-gram fusion, per the HF NgramEmbedding forward.
        # In the kernel's table convention a boundary is a negative entry.
        ignored = input_ids >= self.text_vocab_hash_size
        boundary = ignored | (input_ids == 0)
        table_ids = torch.where(boundary, torch.full_like(input_ids, -1), input_ids)

        # A/B bypass (LONGCAT_NGRAM_DISABLE=1): pure word embeddings for every
        # token, no n-gram fusion. If the port-with-ngram matches the
        # reference-with-ngram and only the *with-ngram* variant drifts, the
        # divergence is in this fused path (hashing or streaming), not in
        # attention/MLA. Compare against a reference run with its own ngram
        # disabled; matching outputs there isolates the fused path as the bug.
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

        # Audio/image-gen state advance: check the LAST token of this span.
        # A request is in at most one of _audio_gen/_visual_gen at a time
        # (the reference's state machine treats GEN_AUDIO_STAGE/
        # GEN_IMAGE_STAGE as mutually exclusive), so both advance calls are
        # safe to run unconditionally -- each only creates/updates its own
        # state dict when it sees its own start/end marker.
        #
        # decode_eligible mirrors the runner's own talker_mtp dispatch gate
        # (gpu_model_runner.py: `span_len == 1 and not is_prefill`, checked
        # both before batching into decode_batch_items and again when
        # popping "mtp_inputs"). It must be threaded into both advance calls
        # so gen_step only advances on a step where the runner will actually
        # invoke talker_mtp -- see _advance_audio_gen's docstring for why an
        # unconditional advance silently drops a frame whenever
        # <longcat_audiogen_start>/<longcat_img_start> lands as the last
        # token of a multi-token prefill chunk (e.g. written into the
        # prompt itself).
        update_dict: dict[str, Any] = {}
        last_token = int(input_ids[-1])
        is_prefill = bool(info.get("_omni_is_prefill", False))
        decode_eligible = (not is_prefill) and int(input_ids.shape[0]) == 1
        update_dict.update(self._advance_audio_gen(
            request_id, last_token, device=input_ids.device, dtype=self.dtype,
            decode_eligible=decode_eligible,
        ))
        additional_information = info.get("additional_information")
        if not isinstance(additional_information, dict):
            additional_information = info
        token_w = additional_information.get("token_w")
        token_h = additional_information.get("token_h")
        cfg_scale = additional_information.get("cfg_scale")
        update_dict.update(self._advance_visual_gen(
            request_id, last_token, device=input_ids.device, dtype=self.dtype,
            token_w=token_w, token_h=token_h, cfg_scale=cfg_scale,
            decode_eligible=decode_eligible,
        ))

        return input_ids, out.to(self.dtype), update_dict

    def on_requests_finished(self, finished_req_ids: Any) -> None:
        for req_id in finished_req_ids:
            req_id = str(req_id)
            self._ngram_ctx.pop(req_id, None)
            state = self._audio_gen.pop(req_id, None)
            if self._audio_debug:
                # Final tally per request. sampled == kept + discarded, and
                # emitted should equal kept; any mismatch points at the
                # boundary that dropped frames.
                logger.info(
                    "[longcat-audio] req=%s FINISHED audio_gen=%s gen_step=%s "
                    "terminal=%s | sampled=%d kept=%d emitted=%d",
                    req_id, state is not None,
                    (state or {}).get("gen_step"), (state or {}).get("terminal"),
                    self._dbg_sampled, self._dbg_kept, self._dbg_emitted,
                )
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

        model_output is the backbone's hidden states [num_tokens, hidden_size].

        Returning an ``OmniOutput`` (rather than the bare tensor) is what makes
        audio codes reach the request at all: the runner's
        ``extract_multimodal_outputs`` reads ``multimodal_outputs`` only off an
        ``OmniOutput`` instance and yields ``{}`` for a plain tensor, so a
        tensor return silently drops every frame talker_mtp produces.

        Ordering: ``_preprocess`` (which runs talker_mtp) precedes
        ``_model_forward`` (which calls this) within a step, so the codes
        sampled this step are already in ``model_intermediate_buffer`` and go
        out on this same step's output. The output processor concatenates the
        per-step rows (CONCAT_DIM0 for the thinker stage's ``latent`` modality)
        into the final [T, 8] tensor the audio decoder consumes.
        """
        # Non-last pipeline ranks forward IntermediateTensors, which carry no
        # hidden states to stash and no codes to emit.
        if not isinstance(model_output, torch.Tensor):
            return model_output

        info_dicts = kwargs.get("model_intermediate_buffer")
        if info_dicts is None:
            info_dicts = kwargs.get("runtime_additional_information") or []

        # Stash each active request's own last-position hidden state. During
        # decode model_output is flat [num_tokens, hidden]; the runner passes
        # request_token_spans (aligned 1:1 with info_dicts, each carrying its
        # req_id) so a multi-request batch gets the right row instead of
        # everyone inheriting model_output[-1:] from the last request.
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

        # The runner's talker_mtp_output_key is a single fixed ("codes",
        # "audio") destination (gpu_model_runner.py) regardless of which
        # modality's branch inside talker_mtp actually produced a row's
        # codes -- there is no per-row modality tag on that wire. What DOES
        # identify each row's request is the info dict it arrived in: every
        # entry gathered by _gather_runtime_additional_information carries
        # its own "req_id" (added there specifically to support this), so
        # each frame is routed by looking up which of _audio_gen/_visual_gen
        # that request actually belongs to, rather than guessing one
        # modality for the whole step's batch. This is what makes a batch
        # mixing an audio-gen request and an image-gen request route
        # correctly instead of one modality silently overwriting the other's
        # key. The fallback (no usable req_id, e.g. older callers/tests) only
        # applies to a batch that is not already mixed.
        fallback_modality = "visual" if (self._visual_gen and not self._audio_gen) else "audio"

        frames_by_modality: dict[str, list[torch.Tensor]] = {"audio": [], "visual": []}
        for info in info_dicts:
            if not isinstance(info, dict):
                continue
            codes = info.get("codes")
            if not isinstance(codes, dict):
                continue
            # Consumed, not just read: the buffer entry persists across steps,
            # so leaving it would re-emit the same frame on every later step
            # where talker_mtp did not run (e.g. after terminal) and duplicate
            # audio.
            frame = codes.pop("audio", None)
            if not isinstance(frame, torch.Tensor) or frame.numel() == 0:
                continue
            frame = frame.reshape(1, -1) if frame.dim() == 1 else frame
            # talker_mtp marks discarded frames (pre-audio_start/row-boundary,
            # chunk/image-end sentinel, rows of requests not generating) with
            # an all -1 row so the returned tensor stays batch-aligned; those
            # are not real codes and must not reach the decoder.
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
            if self._audio_debug:
                self._dbg_emitted += int(codes_out.shape[0])
                if self._dbg_step(self._dbg_emitted):
                    logger.info(
                        "[longcat-%s] make_omni_output emitting %s row(s) "
                        "(emitted_total=%d vs kept_total=%d) first_row=%s",
                        "image" if modality == "visual" else "audio",
                        list(codes_out.shape), self._dbg_emitted, self._dbg_kept,
                        codes_out[0].tolist(),
                    )
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
        # Materialize a replicated (non-TP) copy of the audio-code embedding
        # rows [AUDIO_OFFSET, AUDIO_OFFSET + sum(codebook_sizes)). This avoids
        # the TP all-reduce in VocabParallelEmbedding when audio_head indexes
        # embeddings inside the rank-0-only code path.
        base = int(self.audio_offset_vals[0].item())
        total = sum(self.audio_codebook_sizes)
        audio_id_range = torch.arange(
            base, base + total, device=device, dtype=torch.long,
        )
        # All ranks participate in this embed_tokens call (matched collective).
        emb = self.model.embed_tokens(audio_id_range).detach().float()
        # Now each rank has the same embedding; store a local copy.
        self.register_buffer(
            "_replicated_audio_code_embedding", emb.to(device=device, dtype=self.dtype), persistent=False
        )
        del audio_id_range
        return self._replicated_audio_code_embedding

    def _ensure_audio_code_embed_module(self, device: torch.device) -> nn.Module:
        """Callable embedding over the audio-code rows, for audio_head.

        The checkpoint's ``CasualDepthTransformerHead.forward(x, visual_tokens,
        visual_emb_layers, level)`` does ``visual_emb_layers.to("cuda:0")`` and
        then ``visual_emb_layers(visual_tokens[..., i])`` -- i.e. it wants a
        single callable ``nn.Module`` (it is the *checkpoint's* variant; the
        SGLang reference's ``image_head.py`` instead indexes a list per level).
        Passing ``None`` raises AttributeError on the ``.to()``.

        Indices are relative to AUDIO_OFFSET so this stays a compact table
        rather than the full vocab, which also keeps the hardcoded ``cuda:0``
        move off the TP-sharded ``embed_tokens``.
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

        Used only as the depth-head's OWN internal ``visual_emb_layers``
        argument (see _ensure_visual_code_embed_module) -- NOT the outer
        next-step feedback embedding, which reuses the already-proven
        _code_embeddings + visual_embedding_layer path (same one image
        *understanding* uses in _encode_images).
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

        Same rationale as _ensure_audio_code_embed_module: the checkpoint's
        CasualDepthTransformerHead (the same class backs both audio_head and
        visual_head) calls this argument's ``.to()`` and invokes it directly,
        so it needs a real nn.Module, not None.
        """
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
        """Sample one audio code from logits, matching generation_config defaults.

        ``None`` sampling keys are coalesced to the defaults (the runner passes
        explicit ``None`` when subtalker_sampling_params is unset; ``None``
        would otherwise fall through to greedy argmax or raise on temperature).

        ``repetition_penalty``/``past_codes`` implement the reference's
        code-level penalty (output_processor.py:369-397): each code already
        sampled for this level is penalised (score<0 * penalty, else /penalty)
        on the raw logits, before temperature. No-op when penalty==1.0 or the
        level has no history.
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
                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
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
        """Shared rank-0+broadcast depth-head sampling loop.

        Used by both audio_head and visual_head -- same checkpoint class
        (CasualDepthTransformerHead), same hardcoded-cuda:0 workaround, same
        autoregressive per-level accumulator. Returns the broadcast
        codes_row [num_levels], already visible on every rank.

        ``repetition_penalty``/``past_codes`` are forwarded per level to
        _sample_audio_code (each level penalises its own history, mirroring
        the reference's ``past_multi_ids[:, :, i]``). ``mask_sentinel`` zeroes
        the level-0 end-of-image class for the visual head only (reference
        output_processor.py:312), so the image can never self-terminate early
        and the grid bound in _advance_visual_gen is the sole terminator.

        ``mask_audio_sentinels`` zeroes each level >= 1's own sentinel class
        (index ``codebook_sizes[i]``). The depth head is
        ``nn.Linear(transformer_dim, vq_size + 1)``, so EVERY level has a
        ``+1`` sentinel class, but only level-0's is the real chunk-end marker
        (the reference splits chunks on ``audio_ids[:, 0] == codebook_sizes[0]``
        and its ``decode_wave_vocoder2`` truncates there). A non-zero-level
        sentinel would leak into a kept frame and OOB the VQ codebook gather
        (``codebook.embed[indices]``) at decode time -- the exact GPU assert
        seen once audio-termination let generation run long. Masking levels
        >= 1 keeps level-0's chunk-end sentinel available while guaranteeing
        no out-of-range code is ever emitted.
        """
        codes_row = torch.zeros(num_levels, dtype=torch.long, device=device)
        if rank == 0:
            sampled_codes = []
            base = offset_vals[0]
            hid = last_hidden.to(dtype=self.dtype)
            # Accumulator fed into the head, in indices relative to the
            # modality's own offset (see _ensure_*_code_embed_module). Level
            # L's logits read hidden_states[:, L], built from the cumulative
            # embedding of levels 0..L-1 -- filling one slot per iteration is
            # what makes the depth loop autoregressive.
            cum_ids = torch.zeros(1, num_levels, dtype=torch.long, device=device)
            for level in range(num_levels):
                logits = head(hid, cum_ids.to(hid.device), code_embed, level)
                level_logits = logits[0]
                sentinel_idx = None
                if mask_sentinel and level == 0:
                    # codebook_sizes[0] is the end-of-image class (OmniImageHead),
                    # masked per the reference so it can never be sampled.
                    sentinel_idx = self.visual_codebook_sizes[0]
                elif mask_audio_sentinels and level >= 1:
                    # Non-zero-level sentinels are meaningless and OOB the VQ
                    # codebook at decode; only level-0's is the chunk-end marker.
                    sentinel_idx = self.audio_codebook_sizes[level]
                if sentinel_idx is not None and sentinel_idx < level_logits.shape[-1]:
                    level_logits = level_logits.clone()
                    level_logits[sentinel_idx] = float("-inf")
                code = self._sample_audio_code(
                    level_logits, do_sample=do_sample, temperature=temperature,
                    top_k=top_k, top_p=top_p,
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
        """Sample visual codes with classifier-free guidance from a cond/uncond pair.

        Mirrors _sample_depth_head's rank-0+broadcast structure (the
        CasualDepthTransformerHead's hardcoded cuda:0 workaround), but drives
        the autoregressive depth loop with the reference's CFG combination: at
        every level,

            combined = cfg_scale * (cond_logits - uncond_logits) + uncond_logits

        where cond_logits come from the parent's hidden state and uncond_logits
        from the twin's (separate KV cache, so the uncond stream never attends
        to the user instruction). Both streams share one sampled code per
        level, keeping the two streams locked to the same pixel sequence.
        Returns the broadcast codes_row [num_levels] visible on every rank.
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
                    # Level-0 end-of-image class, masked per the reference
                    # (output_processor.py:312) so the image can never
                    # self-terminate early; the grid bound is the terminator.
                    sentinel_idx = self.visual_codebook_sizes[0]
                if sentinel_idx is not None and sentinel_idx < combined.shape[-1]:
                    combined = combined.clone()
                    combined[sentinel_idx] = float("-inf")
                code = self._sample_audio_code(
                    combined, do_sample=do_sample, temperature=temperature,
                    top_k=top_k, top_p=top_p,
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

        # Sampling params from the runner's subtalker_sampling_params.
        # _talker_mtp_forward passes explicit None for every key when
        # subtalker_sampling_params is unset, so `dict.get(key, default)`
        # yields None (key present), which would fall through to greedy argmax
        # or raise. Coalesce None to the checkpoint's generation_config
        # defaults, per modality (audio top_k=5/top_p=0.85/rep=1.3; visual
        # top_k=1024/top_p=0.75/rep=1.0). Modality-specific keys
        # (audio_top_k/visual_top_k, ...) win; the generic key applies to
        # both only as a fallback, so an audio-tuned value can't silently
        # override the visual default.
        do_sample = kwargs.get("do_sample", True)
        temperature = kwargs.get("temperature", 0.5)
        audio_top_k = kwargs.get("audio_top_k", kwargs.get("top_k", 5))
        audio_top_p = kwargs.get("audio_top_p", kwargs.get("top_p", 0.85))
        visual_top_k = kwargs.get("visual_top_k", kwargs.get("top_k", 1024))
        visual_top_p = kwargs.get("visual_top_p", kwargs.get("top_p", 0.75))
        audio_rep_penalty = kwargs.get(
            "audio_repetition_penalty", kwargs.get("repetition_penalty", 1.3)
        )
        visual_rep_penalty = kwargs.get(
            "visual_repetition_penalty", kwargs.get("repetition_penalty", 1.0)
        )
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

        # Per-row results
        all_codes = torch.full((batch_size, num_levels), -1, dtype=torch.long, device=device)
        all_embeds = inputs_embeds.clone()

        # Materialise BOTH modalities' code embedding tables on EVERY rank,
        # before the per-row loop and outside any rank-0 guard. Both are
        # built from self.model.embed_tokens -- a VocabParallelEmbedding
        # whose forward all-reduces across the TP group -- so building
        # either lazily inside a rank-0-only sampling block leaves ranks
        # 1..N-1 out of that collective and deadlocks the group (observed
        # for audio: RPC TimeoutError with a c10d::Work stack, zero mtp log
        # lines). Every rank must reach both unconditionally, regardless of
        # which modality any given row in this batch is actually using.
        audio_code_embed = self._ensure_audio_code_embed_module(device)
        visual_code_embed = self._ensure_visual_code_embed_module(device)

        # Visual CFG twin pairing. The engine's prompt expansion admits one
        # unconditional companion per image request with
        # request_id = f"{parent_id}__cfg_visual" (see
        # expand_longcat_cfg_prompts). The two streams share the same batch
        # (admitted with affinity), each with its OWN KV cache / hidden state,
        # so when both are in visual-gen state in this step we can combine
        # their depth-head logits per the reference's CFG formula. All ranks
        # derive the same pairing from the same req_ids/states, keeping the
        # rank-0 broadcast counts aligned.
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
                last_talker_hidden[row:row+1] if row < last_talker_hidden.shape[0] else last_talker_hidden[-1:]
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
                past_codes_t = (
                    torch.stack(past_codes).to(device) if past_codes else None
                )
                codes_row = self._sample_depth_head(
                    self.audio_head, audio_code_embed, self.audio_offset_vals, num_levels,
                    last_hidden, rank, tp_group, device, do_sample, temperature,
                    audio_top_k, audio_top_p, audio_rep_penalty, past_codes_t,
                    mask_audio_sentinels=True,
                )

                # Detect terminal conditions (reference state_machine.py:97-106)
                deoff = int(codes_row[0].item())
                # Condition 1: end-of-chunk marker — model emitted chunk_end_code
                eoc_terminal = audio_start and (deoff >= codebook_sizes[0])
                # Condition 2: max_gen safety cap — force-end to prevent infinite gen
                max_gen_terminal = audio_start and gen_step >= self.max_gen
                is_terminal = eoc_terminal or max_gen_terminal

                # Every terminal row becomes an explicit chunk-end MARKER row
                # (level-0 = codebook_sizes[0], the same value _split_chunks
                # in modeling_longcat_next_audio_decoder.py splits on; other
                # levels are never decoded for a boundary row, so 0 is a safe
                # placeholder) instead of the -1 discard sentinel.
                #
                # -1 rows are stripped entirely by _extract_codes_from_output
                # before reaching the decoder -- fine for a single natural
                # eoc in isolation, but our earlier fix (forcing a clean
                # <longcat_audiogen_end> close on BOTH eoc_terminal AND
                # max_gen_terminal, instead of leaving the row unconstrained
                # and letting the whole request end on real EOS) lets one
                # request legitimately produce multiple audio segments -- the
                # model can resume and re-enter <longcat_audiogen_start>
                # after a max_gen-forced close. Previously a max_gen close
                # kept the row as an ordinary real frame (reference
                # behavior, which never continues past max_gen) and an eoc
                # close discarded it as -1 -- neither survives as a boundary
                # marker once _extract_codes_from_output strips negatives, so
                # multiple segments' codes concatenate with NO surviving
                # boundary between them. LongcatNextAudioDecoder's
                # _split_chunks then sees one giant merged chunk instead of
                # several bounded ones and overflows the checkpoint's
                # fixed-size positional embedding table (observed:
                # `tensor a (5731) != tensor b (3000)` in
                # modular_longcat_next_audio.py's audio_decoder.forward).
                frame_kept = audio_start and not is_terminal
                if is_terminal:
                    boundary_row = torch.zeros_like(codes_row)
                    boundary_row[0] = codebook_sizes[0]
                    all_codes[row] = boundary_row
                else:
                    all_codes[row] = codes_row if frame_kept else torch.full_like(codes_row, -1)

                self._dbg_sampled += 1
                if frame_kept:
                    self._dbg_kept += 1
                    # Repetition-penalty history: kept frames only (the
                    # reference filters discarded rows, output_processor.py:349).
                    state.setdefault("past_codes", []).append(codes_row.detach().cpu())
                if self._audio_debug and (self._dbg_step(gen_step) or is_terminal):
                    logger.info(
                        "[longcat-audio] req=%s mtp step=%d codes0=%d kept=%s "
                        "(audio_start=%s eoc=%s max_gen=%s) codes=%s sampled=%d kept_total=%d",
                        req_id, gen_step, deoff, frame_kept, audio_start,
                        eoc_terminal, max_gen_terminal, codes_row.tolist(),
                        self._dbg_sampled, self._dbg_kept,
                    )
                if is_terminal and self._audio_debug:
                    logger.info(
                        "[longcat-audio] req=%s TERMINAL at step=%d reason=%s "
                        "(total sampled=%d kept=%d)",
                        req_id, gen_step,
                        "chunk_end" if eoc_terminal else "max_gen",
                        self._dbg_sampled, self._dbg_kept,
                    )

                # Build 3-stream next-step embedding (reference's get_audio_embeddings)
                # Stream 1: ext_id embedding — audiotext_start/pad/audiogen_end
                ext_tok = torch.tensor([ext_id], device=device, dtype=torch.long)
                ext_emb = self.model.embed_tokens(ext_tok)
                if ext_id == AUDIOTEXT_PAD_TOKEN_ID:
                    ext_emb.zero_()

                # Stream 2: visible text token embedding — masked to 0 when the
                # token itself is audiotext_pad. The reference keys this off the
                # token value (input_ids_mask, input_processor.py:126,134), not off
                # the text_end flag: once text_end is set compute_logits pins the
                # token to pad anyway, so the two agree, but matching on the value
                # keeps this correct even for a pad sampled before text_end.
                text_tok = input_ids[row:row+1]
                text_emb = self.model.embed_tokens(text_tok)
                if int(text_tok.item()) == AUDIOTEXT_PAD_TOKEN_ID:
                    text_emb.zero_()

                # Stream 3: audio code embeddings — masked to 0 for invalid rows.
                # The reference additionally drops rows whose level-0 code is 0 or
                # the chunk-end sentinel (multi_ids_row_mask,
                # input_processor.py:129-132,145); code 0 doubles as its
                # clamped-invalid value, so an embedding is only summed in for a
                # kept frame with a non-zero level-0 code.
                audio_emb = torch.zeros_like(ext_emb)
                if frame_kept and deoff != 0:
                    replicated_emb = self._ensure_replicated_audio_code_embedding(device)
                    offset_codes = codes_row + self.audio_offset_vals[:num_levels].to(device)
                    row_embs = []
                    for level in range(num_levels):
                        idx = (offset_codes[level] - self.audio_offset_vals[0]).item()
                        if 0 <= idx < replicated_emb.shape[0]:
                            row_embs.append(replicated_emb[idx:idx+1])
                    if row_embs:
                        audio_emb = torch.cat(row_embs, dim=0).sum(dim=0, keepdim=True)

                # Sum the 3 streams
                next_emb = ext_emb + text_emb + audio_emb
                all_embeds[row:row+1] = next_emb.to(dtype=self.dtype)

                if is_terminal:
                    state["terminal"] = True
                    # Force the closing tag on the terminal step itself,
                    # mirroring _advance_visual_gen's IMG_END forcing
                    # (grid-bound termination). Without this, compute_logits
                    # used to `continue` entirely once terminal -- fully
                    # unbanning EOS with no forced closure -- so the model
                    # was free to (and observed to) end the WHOLE request via
                    # real EOS within a few tokens of chunk_end/max_gen,
                    # instead of just closing this audio segment and
                    # resuming normal generation. Setting ext_id here means
                    # compute_logits' terminal branch (below) forces
                    # <longcat_audiogen_end> for exactly this one step; once
                    # the model emits it, _advance_audio_gen's
                    # AUDIOGEN_END_TOKEN_ID branch pops the state next step,
                    # returning full freedom only after a clean close.
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

                # Resolve the (parent, twin) pair this visual row belongs to,
                # if any. Only combine when BOTH streams are present in this
                # batch AND both are non-terminal visual-gen rows; otherwise
                # fall back to independent sampling (identical to the
                # no-CFG path).
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
                    and p_row is not None and t_row is not None
                    and p_state is not None and t_state is not None
                    and not bool(p_state.get("terminal", False))
                    and not bool(t_state.get("terminal", False))
                )

                # Diagnostic for CFG desync: this row belongs to a cond/uncond
                # pair, but the two streams aren't both active this step, so
                # the code below falls back to independent (non-CFG) sampling
                # for whichever side IS present. That fallback is intentional
                # (the parent/twin are two independently-scheduled engine
                # requests linked only by affinity to the same replica, not a
                # hard same-step scheduling guarantee like the reference's
                # own group_size=2 batching contract -- see
                # expand_longcat_cfg_prompts's docstring), but it was
                # previously silent: nothing distinguished "CFG combined
                # every step" from "CFG never actually combined" short of
                # noticing degraded image quality. Logged unconditionally
                # (not gated behind _audio_debug) since it only fires on an
                # actual desync, not every step -- a real anomaly signal for
                # any CFG-enabled request, not per-step noise.
                if parent_req_id is not None and not both_visual_active:
                    logger.warning(
                        "[longcat-image] CFG desync: req=%s parent=%s(row=%s,state=%s,terminal=%s) "
                        "twin=%s(row=%s,state=%s,terminal=%s) -- falling back to independent "
                        "(non-CFG) sampling for this step",
                        req_id,
                        parent_req_id, p_row, p_state is not None,
                        bool(p_state.get("terminal", False)) if p_state is not None else None,
                        twin_req_id, t_row, t_state is not None,
                        bool(t_state.get("terminal", False)) if t_state is not None else None,
                    )

                if both_visual_active and row == p_row:
                    # ---- combined CFG path (drives both streams) ----
                    p_last_hidden = (
                        last_talker_hidden[p_row:p_row + 1]
                        if p_row < last_talker_hidden.shape[0] else last_talker_hidden[-1:]
                    )
                    t_last_hidden = (
                        last_talker_hidden[t_row:t_row + 1]
                        if t_row < last_talker_hidden.shape[0] else last_talker_hidden[-1:]
                    )
                    p_past = p_state.get("past_codes")
                    p_past_t = torch.stack(p_past).to(device) if p_past else None
                    cfg_scale = float(p_state.get("cfg_scale", _DEFAULT_CFG_SCALE))
                    codes_row = self._sample_cfg_visual_codes(
                        visual_code_embed, self.visual_offset_vals, visual_num_levels,
                        p_last_hidden, t_last_hidden, rank, tp_group, device,
                        do_sample, temperature, visual_top_k, visual_top_p,
                        visual_rep_penalty, p_past_t, cfg_scale,
                    )

                    deoff = int(codes_row[0].item())
                    eoc_terminal = deoff >= visual_codebook_sizes[0]
                    frame_kept = not is_row_boundary and not eoc_terminal
                    is_terminal = eoc_terminal
                    self._dbg_sampled += 1
                    if frame_kept:
                        self._dbg_kept += 1
                        p_state.setdefault("past_codes", []).append(codes_row.detach().cpu())
                    if self._audio_debug and (self._dbg_step(gen_step) or is_terminal):
                        logger.info(
                            "[longcat-image] req=%s mtp step=%d CFG codes0=%d kept=%s "
                            "(row_boundary=%s eoc=%s) codes=%s sampled=%d kept_total=%d",
                            req_id, gen_step, deoff, frame_kept, is_row_boundary,
                            eoc_terminal, codes_row.tolist(), self._dbg_sampled, self._dbg_kept,
                        )

                    # Both streams share the combined sample (lockstep) and the
                    # twin's grid state mirrors the parent's so both terminate
                    # on the same step and compute_logits forces the same
                    # visible token on both rows. Keys present on the parent's
                    # state are copied over (defensive: production states
                    # always carry all of these, but a desynced/older state
                    # must not crash the sync).
                    for _sync_key in ("gen_step", "ext_id", "token_w", "token_h"):
                        if _sync_key in p_state:
                            t_state[_sync_key] = p_state[_sync_key]
                    t_state["terminal"] = bool(is_terminal or p_state.get("terminal", False))

                    for rr in (p_row, t_row):
                        all_codes[rr] = codes_row if frame_kept else torch.full_like(codes_row, -1)

                    # 2-stream next-step embedding, built once from the shared
                    # codes and applied to both rows (the twin's visible token
                    # is forced to the same value as the parent's, so reusing
                    # the parent's text token keeps both streams aligned).
                    text_tok = input_ids[p_row:p_row + 1]
                    text_emb = self.model.embed_tokens(text_tok)
                    if frame_kept and deoff != 0:
                        offset_codes = (codes_row + self.visual_offset_vals[:visual_num_levels].to(device)).unsqueeze(0)
                        vision_emb = self._code_embeddings(offset_codes)
                        vision_emb = self.visual_tokenizer.visual_embedding_layer(vision_emb.to(self.dtype))
                        next_emb = vision_emb
                    else:
                        next_emb = text_emb
                    for rr in (p_row, t_row):
                        all_embeds[rr:rr + 1] = next_emb.to(dtype=self.dtype)

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
                past_codes_t = (
                    torch.stack(past_codes).to(device) if past_codes else None
                )
                codes_row = self._sample_depth_head(
                    self.visual_head, visual_code_embed, self.visual_offset_vals, visual_num_levels,
                    last_hidden, rank, tp_group, device, do_sample, temperature,
                    visual_top_k, visual_top_p, visual_rep_penalty, past_codes_t,
                    mask_sentinel=True,
                )

                # The level-0 end-of-image sentinel class (codebook_sizes[0])
                # is masked in _sample_depth_head (reference output_processor.py:312)
                # and so never fires here; the deterministic terminator is the
                # grid bound in _advance_visual_gen. This check is a defensive
                # guard for any non-masked path.
                deoff = int(codes_row[0].item())
                eoc_terminal = deoff >= visual_codebook_sizes[0]

                # A row-boundary (newline) step never carries a real pixel,
                # regardless of what was sampled -- mirrors output_processor.py
                # discarding tmp_multi_ids at row boundaries (filled -999997).
                frame_kept = not is_row_boundary and not eoc_terminal
                all_codes[row] = codes_row if frame_kept else torch.full_like(codes_row, -1)

                is_terminal = eoc_terminal
                self._dbg_sampled += 1
                if frame_kept:
                    self._dbg_kept += 1
                    state.setdefault("past_codes", []).append(codes_row.detach().cpu())
                if self._audio_debug and (self._dbg_step(gen_step) or is_terminal):
                    logger.info(
                        "[longcat-image] req=%s mtp step=%d codes0=%d kept=%s "
                        "(row_boundary=%s eoc=%s) codes=%s sampled=%d kept_total=%d",
                        req_id, gen_step, deoff, frame_kept, is_row_boundary,
                        eoc_terminal, codes_row.tolist(), self._dbg_sampled, self._dbg_kept,
                    )
                if is_terminal and self._audio_debug:
                    logger.info(
                        "[longcat-image] req=%s TERMINAL at step=%d (total sampled=%d kept=%d)",
                        req_id, gen_step, self._dbg_sampled, self._dbg_kept,
                    )

                # 2-stream next-step embedding (reference's
                # get_visual_embed_given_tokens / get_multimodal_embed): unlike
                # audio's 3-way SUM, this is a MASKED REPLACE -- the visible
                # text token (IMAGE_PAD or IMAGE_NEWLINE) gets its normal
                # embedding at a newline position, or gets fully replaced by
                # the vision embedding at a real-pixel (IMAGE_PAD) position.
                # No "ext" stream: IMAGE_PAD/IMAGE_NEWLINE simply ARE the
                # visible token, not a side-channel like audiotext_start/pad.
                text_tok = input_ids[row:row+1]
                text_emb = self.model.embed_tokens(text_tok)

                if frame_kept and deoff != 0:
                    # Reuses the already-proven understanding-direction path
                    # (_encode_images): embed_tokens per level summed, then
                    # the visual_embedding_layer bridge refinement. Zero-code
                    # (deoff == 0) is the clamped-invalid marker, same
                    # convention as audio's `deoff != 0` guard.
                    offset_codes = (codes_row + self.visual_offset_vals[:num_levels].to(device)).unsqueeze(0)
                    vision_emb = self._code_embeddings(offset_codes)
                    vision_emb = self.visual_tokenizer.visual_embedding_layer(vision_emb.to(self.dtype))
                    next_emb = vision_emb
                else:
                    next_emb = text_emb
                all_embeds[row:row+1] = next_emb.to(dtype=self.dtype)

                if is_terminal:
                    state["terminal"] = True

            else:
                continue

        # Return *this step's* codes (not an accumulation): the runner stores
        # them under ``codes.audio`` per request, make_omni_output emits them
        # on this step's OmniOutput, and the output processor concatenates the
        # per-step rows into the final [T, 8] tensor. Returning the running
        # accumulation instead would re-send every earlier frame each step and
        # grow the result quadratically.
        return all_embeds, all_codes

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        # Logits audit: capture the first single-token decode step's top-k for
        # a pure-text request (no audio/visual gen active). All the cheap
        # checks pass (tokenization, ngram, MLA/RoPE config, weights), so the
        # divergence must live in the forward pass or the sampler. Dumping the
        # raw pre-sampling logits lets a pod run diff them directly against
        # the reference HF model's logits for the identical input_ids -- the
        # two possibilities are (a) logits differ -> forward-pass bug, or (b)
        # logits match but samples diverge -> sampler-param application.
        if (
            self._audio_debug
            and not self._logits_audited
            and not (self._audio_gen or self._visual_gen)
            and logits is not None
            and logits.shape[0] == 1
        ):
            self._logits_audited = True
            topk_vals, topk_ids = torch.topk(logits[0], 10)
            logger.info(
                "[longcat-logits] first decode step top10 ids=%s vals=%s "
                "eos=%s logits_range=[%.3f, %.3f]",
                topk_ids.tolist(),
                [round(float(v), 3) for v in topk_vals],
                self._eos_id,
                float(logits[0].min()), float(logits[0].max()),
            )
        if logits is None or not (self._audio_gen or self._visual_gen):
            return logits
        # During audio/image-gen mode, suppress EOS and force visible tokens
        # per the reference's parallel model.
        # Row-to-request alignment: during decode (1 token/request), logits
        # rows == len(_audio_gen)+len(_visual_gen) == 1 in single-request
        # mode. During prefill logits has many rows but at most 1 active-gen
        # request; skip forcing in prefill to avoid misaligning row 0 with
        # the wrong position. A request is in at most one of the two dicts
        # (mutually exclusive per the reference's state machine), so simple
        # dict union order is a safe, unambiguous row->request mapping.
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
                    # Force the closing tag instead of leaving this row
                    # totally unconstrained: an earlier version `continue`d
                    # here, unbanning EOS with no forced replacement, which
                    # let the model end the WHOLE request via real EOS
                    # within a few tokens of chunk_end/max_gen instead of
                    # just closing this audio segment (observed: audio
                    # truncating to a handful of frames, finish_reason=stop,
                    # with generation ending only ~30 visible tokens after
                    # <longcat_audiogen_start>). talker_mtp sets
                    # ext_id=AUDIOGEN_END_TOKEN_ID the moment it marks
                    # terminal (see the is_terminal branch above), so this
                    # mirrors the visual branch's IMAGE_END forcing below.
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
        side_state: dict[str, dict[str, torch.Tensor]] = {
            attr: {} for _, attr in self._SIDE_MODULE_PREFIXES
        }

        def split(weights):
            for name, tensor in weights:
                for ckpt_prefix, attr in self._SIDE_MODULE_PREFIXES:
                    if name.startswith(ckpt_prefix):
                        side_state[attr][name[len(ckpt_prefix):]] = tensor
                        break
                else:
                    yield name, tensor

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["model.mtp."],
            skip_substrs=["visual_tokenizer", "audio_tokenizer", "visual_head", "audio_head"],
        )
        loaded = loader.load_weights(split(weights))

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

        # Mark remote-code submodules as loaded — their weights come from the
        # checkpoint's own lazy loading path, not from the thinker's sharded
        # weight iter. Without this, vLLM's track_weights_loading raises:
        #   ValueError: Following weights were not initialized from checkpoint
        _skip_substrs = ("visual_tokenizer", "audio_tokenizer", "visual_head", "audio_head")
        for name, _ in self.named_parameters():
            if any(s in name for s in _skip_substrs):
                loaded.add(name)

        return loaded
