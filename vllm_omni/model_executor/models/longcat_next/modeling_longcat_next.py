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
- **lm_head** is 131125-wide (text + special tokens) per the checkpoint;
  DiNA code generation via the depth heads is not wired into AR sampling yet
  (the heads are still loaded for downstream use — see build.md).
"""

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
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

from .longcat_next_processor import (
    LongcatNextDummyInputsBuilder,
    LongcatNextMultiModalProcessor,
    LongcatNextProcessingInfo,
)
from .longcat_next_utils import (
    AUDIO_PAD_TOKEN_ID,
    IMG_PAD_TOKEN_ID,
    get_remote_attr,
    load_remote_hf_config,
)

logger = init_logger(__name__)

_DEFAULT_PAD_TOKEN_ID = 3  # generation_config.json; config.json omits it


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
    has_postprocess = False
    have_multimodal_outputs = False
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
        self._max_ctx_entries = 4 * max(
            int(getattr(vllm_config.scheduler_config, "max_num_seqs", 64)), 64
        )
        assert ngram is not None, "LongCat-Next requires ngram embeddings"

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
        return oe_ids.long()

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        request_id = str(info.get("request_id", "default"))
        num_computed = int(info.get("_omni_num_computed_tokens", 0))

        # oe_ignored ids (all special tokens >= text_vocab_size, incl. the
        # mm pad markers) are hash-segment boundaries and take *pure* word
        # embeddings — no n-gram fusion, per the HF NgramEmbedding forward.
        # In the kernel's table convention a boundary is a negative entry.
        ignored = input_ids >= self.text_vocab_hash_size
        boundary = ignored | (input_ids == 0)
        table_ids = torch.where(boundary, torch.full_like(input_ids, -1), input_ids)

        oe_ids = self._span_oe_ids(request_id, table_ids, fresh=num_computed == 0)
        fused = self.model.ngram_embeddings.embed_batched(input_ids, oe_ids)
        word = self.model.embed_tokens(input_ids)
        out = torch.where(ignored.unsqueeze(-1), word, fused)

        # Placeholder positions carry the multimodal embeddings already merged
        # into input_embeds by the runner's standard mm path.
        pad_mask = (input_ids == IMG_PAD_TOKEN_ID) | (input_ids == AUDIO_PAD_TOKEN_ID)
        if input_embeds is not None and pad_mask.any():
            out = torch.where(pad_mask.unsqueeze(-1), input_embeds.to(out.dtype), out)

        return input_ids, out.to(self.dtype), {}

    def on_requests_finished(self, finished_req_ids: Any) -> None:
        for req_id in finished_req_ids:
            self._ngram_ctx.pop(str(req_id), None)

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

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        # 131125-wide text+special logits; multimodal code ids are produced by
        # the depth heads, never by the text sampler.
        return self.logits_processor(self.lm_head, hidden_states)

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

        return loaded
