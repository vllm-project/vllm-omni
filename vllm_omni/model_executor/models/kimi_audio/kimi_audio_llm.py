# Copyright 2025 vLLM-Omni Team
"""Stage 0: Kimi Audio LLM with bifurcation for dual output (text + audio)."""

import os
import zlib
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.model_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsTranscription
from vllm.model_executor.models.kimi_audio import (
    KimiAudioDummyInputsBuilder,
    KimiAudioProcessingInfo,
    KimiAudioWhisperEncoder,
)
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata

# Import custom processor for Kimi Audio
from vllm_omni.model_executor.models.kimi_audio.constants import (
    KIMI_AUDIO_AUDIO_EOD_TOKEN_IDS,
    KIMI_AUDIO_BLANK_TOKEN_ID,
    KIMI_AUDIO_DELAY,
    KIMI_AUDIO_EOS_TOKEN_ID,
    KIMI_AUDIO_TEXT_EOS_TOKEN_ID,
    KIMI_AUDIO_TOKEN_OFFSET,
)
from vllm_omni.model_executor.models.kimi_audio.custom_processor import (
    CustomKimiAudioMultiModalProcessor,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


# Monkey-patch the Kimi tokenizer to fix initialization order
# This must be done before the tokenizer is loaded
def _patch_kimi_tokenizer():
    """Patch Kimi tokenizer to initialize _special_tokens_map before setting special tokens."""
    try:
        import sys

        # Try to import the tokenizer module
        for module_name in list(sys.modules.keys()):
            if "tokenization_kimia" in module_name:
                module = sys.modules[module_name]
                if hasattr(module, "TikTokenTokenizer"):
                    # Get the original __init__
                    original_init = module.TikTokenTokenizer.__init__

                    # Create a wrapper that initializes _special_tokens_map first
                    def patched_init(self, vocab_file, **kwargs):
                        # Initialize _special_tokens_map before calling original __init__
                        self._special_tokens_map = {}
                        return original_init(self, vocab_file, **kwargs)

                    # Replace the __init__
                    module.TikTokenTokenizer.__init__ = patched_init
                    break
    except Exception:
        # If patching fails, continue anyway
        pass


# Apply the patch at module load time
_patch_kimi_tokenizer()


@MULTIMODAL_REGISTRY.register_processor(
    CustomKimiAudioMultiModalProcessor,
    info=KimiAudioProcessingInfo,
    dummy_inputs=KimiAudioDummyInputsBuilder,
)
class KimiAudioLLMForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsTranscription):
    """Stage 0: Shared backbone → bifurcation → text + audio logits.

    Architecture (matches MoonshotKimiaForCausalLM):
    - Layers 0-21: Shared backbone (from Qwen2)
    - Bifurcation at layer 21: Clone hidden states
    - Audio path: full backbone (layers 0-27) -> lm_head (audio logits)
    - Text path: shared layers 0-21 -> 6 MIMO layers -> mimo_norm -> mimo_output (text logits)
    """

    # Mark as generative model so vllm's runner validation passes
    is_text_generation_model = True
    # Mark as producing multimodal outputs (audio logits for Stage 1)
    have_multimodal_outputs = True

    # Dual streaming extension points (from HiggsAudioV2 pattern)
    prefer_model_sampler = True  # Use custom sampling for dual streams
    has_postprocess = True  # Enable per-request state sync
    postprocess_uses_hidden_states = True
    postprocess_uses_multimodal_outputs = True
    postprocess_uses_req_infos = True

    # Required by SupportsTranscription
    supported_languages = {
        "zh": "Chinese",
        "en": "English",
        "ja": "Japanese",
        "ko": "Korean",
        "de": "German",
        "fr": "French",
        "es": "Spanish",
        "it": "Italian",
        "pt": "Portuguese",
        "ru": "Russian",
        "ar": "Arabic",
    }

    @classmethod
    def get_generation_prompt(cls, stt_params) -> str:
        """Delegate to the upstream KimiAudio transcription prompt builder.

        This is required by the SupportsTranscription interface so that the
        chat-completions entry point can construct the same ASR prompt as
        upstream vLLM when the request contains audio input only.
        """
        from vllm.model_executor.models.kimi_audio import KimiAudioForConditionalGeneration

        return KimiAudioForConditionalGeneration.get_generation_prompt(stt_params)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config

        # Import upstream vllm components
        from vllm.model_executor.models.kimi_audio import (
            KimiAudioMultiModalProjector,
        )

        # Load Whisper encoder weights from the whisper-large-v3 subfolder.
        # This mirrors upstream KimiAudioForConditionalGeneration.
        self.secondary_weights = [
            DefaultModelLoader.Source(
                model_or_path=vllm_config.model_config.model,
                subfolder="whisper-large-v3",
                revision=vllm_config.model_config.revision,
            )
        ]

        with self._mark_tower_model(vllm_config, "audio"):
            # Use custom Whisper encoder that matches reference implementation.
            # Prefix "audio_tower" matches upstream vLLM's weight mapping so the
            # whisper-large-v3 encoder weights load correctly.
            self.audio_tower = KimiAudioWhisperEncoder(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "audio_tower"),
            )

            # Audio input processing — aligned with upstream vllm 0.23:
            # Whisper-Large-v3 outputs [B, T, 1280]. 4-frame concat via reshape
            # produces [B, T//4, 5120], which feeds directly into the VQ Adaptor
            # (first layer is Linear[5120→3584]). No intermediate projection.
            self.multi_modal_projector = KimiAudioMultiModalProjector(
                whisper_dim=5120,  # = whisper_d_model (1280) × 4-frame concat
                llm_dim=self.config.hidden_size,
                prefix=maybe_prefix(prefix, "multi_modal_projector"),
            )

        # Back-compat attribute (some older code paths reference this)
        self.whisper_projection = None

        # Qwen2 backbone (layers 0-27)
        # Use "language_model" prefix to match upstream vLLM's WeightsMapper
        self.model = init_vllm_registered_model(
            vllm_config.with_hf_config(self.config, architectures=["Qwen2ForCausalLM"]),
            prefix=maybe_prefix(prefix, "language_model"),
        )

        # NEW: MIMO layers (6 layers) - audio-specific transformer layers.
        # These reuse the Qwen2 decoder layer structure but replace the two
        # RowParallelLinear reductions (o_proj / down_proj) with an exact
        # all-gather + full-GEMM path. The audio branch is numerically sensitive
        # (hot residual stream + competitive audio-token softmax), so the bf16
        # all-reduce rounding compounds through the audio feedback loop and
        # collapses generation under TP>1. qkv/gate_up/attention stay sharded so
        # the KV-cache layout matches the text layers. See replicated_qwen2.py.
        from vllm_omni.model_executor.models.kimi_audio.replicated_qwen2 import (
            ExactQwen2DecoderLayer,
            ReplicatedQwen2DecoderLayer,
        )

        # Get cache_config and quant_config from vllm_config for proper KV caching
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        # GATED backbone-exact fix for TP>1 audio divergence.
        # The shared backbone (layers 0-21) feeds BOTH the text and audio
        # branches at the layer-21 bifurcation. Its RowParallelLinear
        # o_proj/down_proj all-reduce introduces a ~1% bf16 rounding difference
        # vs TP=1 that the numerically-sensitive audio (mimo) branch amplifies
        # into total collapse. Replacing those layers with ExactQwen2DecoderLayer
        # (all-gather + full replicated GEMM) removes that reduction.
        #
        # Gated by KIMI_AUDIO_EXACT_BACKBONE (default on). It is a strict no-op
        # at TP=1 (the all-gather short-circuits and a full ReplicatedLinear
        # GEMM equals the single TP=1 GEMM), so the working single-GPU text
        # path is untouched.
        #
        # Coverage is controlled by KIMI_AUDIO_EXACT_LAYERS:
        #   "backbone" (default) -> replace layers 0-21 only.
        #   "all"                -> replace every layer (0-27), so the text head
        #                           also matches TP=1. Measured to matter because
        #                           layers 22-27 shard the text stream, and the
        #                           divergent text is fed back into the audio
        #                           branch's attention context, re-seeding the
        #                           collapse at the audio-feedback boundary.
        # NOTE: even "all" leaves the Whisper encoder tower sharded (its
        # RowParallelLinear out_proj/fc2 all-reduce keeps the layer-21 output
        # ~1.7% off TP=1), which can still perturb the hyper-sensitive audio
        # branch. See kimi-tp2-mimo-forward-divergence memory.
        _exact_backbone = os.environ.get("KIMI_AUDIO_EXACT_BACKBONE", "1") == "1"
        _tp_size = get_tensor_model_parallel_world_size()

        # GATED attention-replication fix (option A) for TP>1 audio collapse.
        # By elimination, once every RowParallelLinear reduction is exact the only
        # remaining cross-rank difference is the attention kernel's per-head
        # geometry: TP=1 runs all 4 KV heads on one rank, TP=2 runs 2 per rank, so
        # the flash kernel tiles/reduces differently and the layer-21 bifurcation
        # shifts ~1.5% — which the audio-token feedback loop amplifies into total
        # collapse. ReplicatedQwen2DecoderLayer reconstructs the FULL head set on
        # every rank (all-gather q/k/v -> rope -> full-head attention), so each
        # rank runs the identical kernel config as TP=1. It replaces ALL backbone
        # layers (uniform full-head KV-cache spec) and supersedes the backbone-
        # exact linear fix (its MLP is already the exact all-gather path).
        #
        # Gated by KIMI_AUDIO_REPLICATE_ATTN (default OFF — it replicates the KV
        # cache and attention FLOPs, trading TP efficiency for TP=1 equivalence).
        # Strict no-op at TP=1. See kimi-tp2-mimo-forward-divergence memory.
        _replicate_attn = os.environ.get("KIMI_AUDIO_REPLICATE_ATTN", "0") == "1"
        if _replicate_attn and _tp_size > 1:
            _n_total = len(self.model.model.layers)
            for _i in range(_n_total):
                _old = self.model.model.layers[_i]
                # Pop the old Attention's static_forward_context entry first so the
                # replacement keeps the same layer_name (same KV-cache slot) without
                # a "Duplicate layer name" error.
                _old_attn_name = _old.self_attn.attn.layer_name
                _layer_prefix = _old_attn_name[: -len(".self_attn.attn")]
                vllm_config.compilation_config.static_forward_context.pop(_old_attn_name, None)
                self.model.model.layers[_i] = ReplicatedQwen2DecoderLayer(
                    config=self.config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=_layer_prefix,
                )
            logger.info(
                "[REPLICATE-ATTN] Replaced all %d backbone layers with "
                "ReplicatedQwen2DecoderLayer (full-head attention, TP=%d)",
                _n_total,
                _tp_size,
            )

        if _exact_backbone and _tp_size > 1 and not _replicate_attn:
            _exact_layers_mode = os.environ.get("KIMI_AUDIO_EXACT_LAYERS", "backbone")
            _n_total = len(self.model.model.layers)
            _n_shared = _n_total if _exact_layers_mode == "all" else min(22, _n_total)
            for _i in range(_n_shared):
                _old = self.model.model.layers[_i]
                # Reuse the exact original prefix so the replacement Attention
                # keeps the same layer_name -> same KV-cache slot and the same
                # extract_layer_index() result. The old layer's Attention already
                # registered this layer_name in static_forward_context; pop it
                # first so the replacement does not hit "Duplicate layer name".
                _old_attn_name = _old.self_attn.attn.layer_name
                _layer_prefix = _old_attn_name[: -len(".self_attn.attn")]
                vllm_config.compilation_config.static_forward_context.pop(_old_attn_name, None)
                self.model.model.layers[_i] = ExactQwen2DecoderLayer(
                    config=self.config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=_layer_prefix,
                )
            logger.info(
                "[EXACT-BACKBONE] Replaced layers 0-%d/%d with ExactQwen2DecoderLayer (mode=%s, TP=%d)",
                _n_shared - 1,
                _n_total - 1,
                _exact_layers_mode,
                _tp_size,
            )

        # GATED whisper-tower exact fix for TP>1 audio divergence.
        # The Whisper encoder produces the audio *input* features that seed the
        # LLM embedding and the layer-21 bifurcation. Its RowParallelLinear
        # reductions (self_attn.out_proj / mlp.fc2) all-reduce two bf16 partial
        # sums, which rounds differently than TP=1's single fused GEMM and makes
        # the audio features ~1.7% different under TP>1. That difference shifts
        # the layer-21 output (~1.5061 vs TP=1's 1.5316) and — amplified by the
        # audio-token feedback loop — collapses S2S audio even when every
        # transformer layer is exact. Replacing out_proj/fc2 with an exact
        # all-gather + full replicated GEMM removes that reduction, so the audio
        # features match TP=1 bit-for-bit.
        #
        # Gated by KIMI_AUDIO_EXACT_WHISPER (default on). Strict no-op at TP=1
        # (the all-gather short-circuits; a full ReplicatedLinear GEMM equals the
        # single TP=1 GEMM). qkv_proj / fc1 / attention stay sharded (column
        # parallel, no reduction); encoder attention has no KV cache, so nothing
        # registers in static_forward_context and no duplicate-name pop is needed.
        _exact_whisper = os.environ.get("KIMI_AUDIO_EXACT_WHISPER", "1") == "1"
        if _exact_whisper and _tp_size > 1:
            from vllm_omni.model_executor.models.kimi_audio.replicated_whisper import (
                ExactWhisperEncoderLayer,
                ReplicatedWhisperEncoderLayer,
            )

            # When attention replication is on, use the full-head whisper attention
            # so the audio features are bit-identical to TP=1 (the sharded whisper
            # attention is the remaining input-path divergence source).
            _whisper_cls = ReplicatedWhisperEncoderLayer if _replicate_attn else ExactWhisperEncoderLayer

            # Recover each encoder layer's prefix from the live module names so
            # the replacement keeps the same param-name structure that
            # KimiAudioWhisperEncoder.load_weights() matches (name-based loading).
            _layer_prefix_by_idx: dict[int, str] = {}
            for _mod_name, _ in self.audio_tower.named_modules():
                if _mod_name.endswith(".self_attn"):
                    _idx = int(_mod_name.split(".")[-2])
                    _layer_prefix_by_idx[_idx] = _mod_name[: -len(".self_attn")]
            _n_whisper = len(self.audio_tower.layers)
            for _i in range(_n_whisper):
                _old = self.audio_tower.layers[_i]
                # Skip pipeline-parallel placeholder layers (not present at TP-only).
                if _old.__class__.__name__ == "PPMissingLayer":
                    continue
                self.audio_tower.layers[_i] = _whisper_cls(
                    embed_dim=_old.embed_dim,
                    num_heads=_old.self_attn.total_num_heads,
                    num_heads_local=_old.self_attn.num_heads,
                    num_kv_heads_local=_old.self_attn.num_kv_heads,
                    head_dim=_old.self_attn.head_dim,
                    scaling=_old.self_attn.scaling,
                    ffn_dim=_old.mlp.fc1.output_size,
                    activation_fn=_old.mlp.activation_fn,  # reuse stateless module
                    quant_config=quant_config,
                    prefix=_layer_prefix_by_idx.get(_i, f"{maybe_prefix(prefix, 'audio_tower')}.layers.{_i}"),
                )
            logger.info(
                "[EXACT-WHISPER] Replaced %d/%d Whisper encoder layers with %s (TP=%d)",
                _n_whisper,
                _n_whisper,
                _whisper_cls.__name__,
                _tp_size,
            )

        # Gated whisper-encoder instrumentation (TP-INDEPENDENT — fires under TP=1
        # and TP>1 alike so dumps are directly comparable across configs). Two hooks
        # on encoder layer 0: a pre-hook captures the conv-stem output (layer-0
        # input) and a forward hook captures layer-0's output. Comparing both across
        # TP bisects the residual audio-feature divergence:
        #   - convstem matches but layer0-out differs  → seed is layer-0 attn/MLP
        #   - both match, but final whisper out differs → divergence accumulates L1-31
        #   - convstem differs                        → conv stem / positional path
        # (conv stem has no TP-sharded ops, so it should match given identical mel.)
        # The pre-hook also logs type(module).__name__ once to prove which layer
        # class executes in the feature path.
        if os.environ.get("KIMI_EMB_DUMP") and os.environ.get("KIMI_MIMO_DEBUG") == "1":
            _conv_state = {"logged": False}

            def _conv_pre_hook(_module, _args):
                if torch.cuda.is_current_stream_capturing() or get_tensor_model_parallel_rank() != 0:
                    return
                _x = _args[0] if isinstance(_args, tuple) and _args else None
                if not _conv_state["logged"]:
                    _conv_state["logged"] = True
                    logger.info(
                        "[EXACT-WHISPER] layer0 EXEC class=%s convstem=%s",
                        type(_module).__name__,
                        tuple(_x.shape) if torch.is_tensor(_x) else type(_x),
                    )
                if torch.is_tensor(_x):
                    torch.save(
                        _x.detach().float().cpu(),
                        f"{os.environ['KIMI_EMB_DUMP']}/convstem_n{_x.shape[1]}_tp{get_tensor_model_parallel_world_size()}.pt",
                    )

            def _layer0_out_hook(_module, _args, _out):
                if torch.cuda.is_current_stream_capturing() or get_tensor_model_parallel_rank() != 0:
                    return
                if torch.is_tensor(_out):
                    torch.save(
                        _out.detach().float().cpu(),
                        f"{os.environ['KIMI_EMB_DUMP']}/layer0out_n{_out.shape[1]}_tp{get_tensor_model_parallel_world_size()}.pt",
                    )

            def _sub_out_hook(_tag):
                def _hook(_module, _args, _out):
                    if torch.cuda.is_current_stream_capturing() or get_tensor_model_parallel_rank() != 0:
                        return
                    _t = _out[0] if isinstance(_out, tuple) else _out
                    if torch.is_tensor(_t):
                        torch.save(
                            _t.detach().float().cpu(),
                            f"{os.environ['KIMI_EMB_DUMP']}/{_tag}_n{_t.shape[1]}_tp{get_tensor_model_parallel_world_size()}.pt",
                        )

                return _hook

            if len(self.audio_tower.layers) > 0:
                _l0 = self.audio_tower.layers[0]
                _l0.register_forward_pre_hook(_conv_pre_hook)
                _l0.register_forward_hook(_layer0_out_hook)
                # Bisect WITHIN layer 0: attention output vs MLP output. Layer 0 =
                # LN->attn->+res->LN->mlp->+res; LN/res are elementwise (can't
                # diverge), so exactly one of attn/mlp seeds the 1.56e-2 layer-0 diff.
                if hasattr(_l0, "self_attn"):
                    _l0.self_attn.register_forward_hook(_sub_out_hook("layer0attn"))
                    # Deeper: capture the (q,k,v) entering SDPA to distinguish a
                    # sharded-qkv-GEMM divergence (q/k/v differ across TP) from an
                    # SDPA-kernel divergence (q/k/v identical, output differs). Both
                    # WhisperAttention and ReplicatedWhisperAttention call
                    # self.attn(q, k, v), so a pre-hook on .attn captures (q,k,v).
                    _inner_attn = getattr(_l0.self_attn, "attn", None)
                    if _inner_attn is not None:

                        def _qkv_pre_hook(_module, _args):
                            if torch.cuda.is_current_stream_capturing() or get_tensor_model_parallel_rank() != 0:
                                return
                            if len(_args) >= 3 and all(torch.is_tensor(t) for t in _args[:3]):
                                _q, _k, _v = _args[0], _args[1], _args[2]
                                torch.save(
                                    {
                                        "q": _q.detach().float().cpu(),
                                        "k": _k.detach().float().cpu(),
                                        "v": _v.detach().float().cpu(),
                                    },
                                    f"{os.environ['KIMI_EMB_DUMP']}/layer0qkv_n{_q.shape[-2]}_tp{get_tensor_model_parallel_world_size()}.pt",
                                )

                        _inner_attn.register_forward_pre_hook(_qkv_pre_hook)
                if hasattr(_l0, "mlp"):
                    _l0.mlp.register_forward_hook(_sub_out_hook("layer0mlp"))

        # mimo (audio) branch: use the same attention-replicated layer class as the
        # backbone when KIMI_AUDIO_REPLICATE_ATTN is on, so the mimo KV-cache spec
        # also matches TP=1's full-head geometry. Otherwise fall back to the
        # linear-exact layer (its attention stays sharded).
        _mimo_layer_cls = ReplicatedQwen2DecoderLayer if (_replicate_attn and _tp_size > 1) else ExactQwen2DecoderLayer
        self.mimo_layers = nn.ModuleList(
            [
                _mimo_layer_cls(
                    config=self.config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, f"mimo_layers.{i}"),
                )
                for i in range(self.config.kimia_mimo_layers)  # 6 layers
            ]
        )

        # NEW: Audio output head
        from vllm.model_executor.layers.layernorm import RMSNorm

        self.mimo_norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

        # Audio output projection (supports tensor parallelism)
        # Tied with lm_head - same weights
        self.mimo_output = ColumnParallelLinear(
            self.config.hidden_size,
            self.config.vocab_size,  # 168448 - full vocab including text and audio
            gather_output=True,  # Gather across TP ranks
            bias=False,  # No bias, matching checkpoint
        )

        # Text logits processor
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size,
            scale=1.0,
        )

        # Dual streaming state (per-slot management following HiggsAudioV2 pattern)
        # These are lazily initialized in sample() to avoid issues with distributed setup
        # self._audio_state: dict[int, dict[str, Any]] = {}
        # self._slot_output_len: dict[int, int] = {}
        # self._text_stream_finished: dict[int, bool] = {}
        self._pending_audio_logits: torch.Tensor | None = None

        # Logits-head dump state (KIMI_LOGIT_DUMP=1 + KIMI_EMB_DUMP=<dir>). A
        # monotonic per-step counter shared between forward() (audio head) and
        # compute_logits() (text head) so the two per-step files can be matched.
        self._logit_dump_step: int = 0
        self._last_logit_dump_step: int = 0

        # Special tokens (from tokenizer)
        self._audio_delay: int = KIMI_AUDIO_DELAY  # First 6 audio tokens are BLANK
        self._blank_token_id: int = KIMI_AUDIO_BLANK_TOKEN_ID  # <|im_kimia_text_blank|>
        self._text_eos_id: int = KIMI_AUDIO_TEXT_EOS_TOKEN_ID  # <|im_kimia_text_eos|>
        self._engine_eos_id: int = KIMI_AUDIO_EOS_TOKEN_ID  # [EOS] recognized by vLLM
        self._audio_eod_ids: set[int] = KIMI_AUDIO_AUDIO_EOD_TOKEN_IDS  # audio stream EOS markers
        self._token_offset: int = KIMI_AUDIO_TOKEN_OFFSET  # Audio tokens start here

        # Audio tokenizer (GLM-4) removed — upstream-aligned architecture doesn't
        # use discrete audio tokens for input comprehension. GLM-4 is only needed
        # for voice cloning (tokenizing reference audio), which is a separate path.

    def _log_tensor_stats(self, name: str, tensor: torch.Tensor) -> None:
        """Log tensor statistics (only in eager mode, not during CUDA graph capture).

        Args:
            name: Label for the tensor (e.g., "After layer 21")
            tensor: Tensor to log statistics for
        """
        if not torch.cuda.is_current_stream_capturing():
            try:
                mean_val = tensor.mean().item()
                std_val = tensor.std().item()
                logger.debug("%s: mean=%.4f, std=%.4f", name, mean_val, std_val)
            except Exception as e:
                logger.debug("%s: error logging stats: %s", name, e)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        multimodal_embeddings: torch.Tensor | list[torch.Tensor] | None = None,
        inputs_embeds: torch.Tensor | None = None,
        additional_information: dict | None = None,
        runtime_additional_information: list | dict | None = None,
        **kwargs,
    ) -> OmniOutput:
        """Forward pass — upstream-aligned.

        Bifurcation at layer 21: clone hidden states for text path (layers 22-27)
        and audio path (MIMO layers).

        Input comprehension: multimodal fusion is handled by embed_input_ids
        (BLANK text embeddings + Whisper continuous features × √2). No discrete
        audio token insertion here.
        """
        # Reset per-slot audio state for any newly scheduled request *before*
        # embeddings are computed.  Stale state from a previous slot occupant
        # (e.g. a finished TTS request) would otherwise corrupt the prefill of
        # the next request (e.g. ASR).
        if input_ids is not None and input_ids.dim() >= 2:
            num_reqs_for_reset = input_ids.shape[0]
        elif isinstance(runtime_additional_information, list):
            num_reqs_for_reset = len(runtime_additional_information)
        elif input_ids is not None:
            num_reqs_for_reset = 1
        else:
            num_reqs_for_reset = 0
        self._reset_audio_state_for_new_requests(runtime_additional_information, num_reqs_for_reset)

        # DEBUG: log positions / input_ids for newly scheduled slots so we can
        # detect stale positional encodings after a previous slot occupant.
        if not torch.cuda.is_current_stream_capturing():
            logger.debug(
                "[forward] debug new-step input_ids=%s positions=%s runtime_type=%s runtime_len=%s",
                input_ids.shape if input_ids is not None else None,
                positions.shape if positions is not None else None,
                type(runtime_additional_information).__name__ if runtime_additional_information is not None else None,
                len(runtime_additional_information) if isinstance(runtime_additional_information, list) else None,
            )
            if input_ids is not None and positions is not None and isinstance(runtime_additional_information, list):
                for batch_i, info in enumerate(runtime_additional_information):
                    if isinstance(info, dict) and info.get("generated_len", -1) == 0:
                        logger.debug(
                            "[forward] NEW slot=%d positions min=%s max=%s input_ids_first=%s input_ids_last=%s",
                            batch_i,
                            positions.min().item() if positions.numel() else None,
                            positions.max().item() if positions.numel() else None,
                            input_ids[0, :5].tolist() if input_ids.dim() == 2 else input_ids[:5].tolist(),
                            input_ids[0, -5:].tolist() if input_ids.dim() == 2 else input_ids[-5:].tolist(),
                        )

        # 1. Get inputs_embeds from vllm's pipeline (which calls embed_input_ids).
        if inputs_embeds is not None:
            inputs_embeds_fused = inputs_embeds
        else:
            inputs_embeds_fused = self.embed_input_ids(
                input_ids,
                multimodal_embeddings,
                runtime_additional_information=runtime_additional_information,
            )

        # Backbone-INPUT dump: the fused embeddings (BLANK text + Whisper features
        # x sqrt2) feeding layer 0. Comparing TP=1 vs TP=2 here isolates whether the
        # audio divergence is seeded UPSTREAM of the backbone (whisper/projector
        # sharding baked in at prefill) or INSIDE the backbone. Only meaningful at
        # prefill (n>2), where the audio features are spliced in.
        if os.environ.get("KIMI_MIMO_DEBUG") == "1" and not torch.cuda.is_current_stream_capturing():
            if inputs_embeds_fused.shape[0] > 2:
                with torch.no_grad():
                    _ief = inputs_embeds_fused.float()
                    logger.info(
                        "[MIMO-EMB] inputs_embeds n=%d [mean=%.6f std=%.6f absmax=%.4f sum=%.4f]",
                        inputs_embeds_fused.shape[0],
                        _ief.mean().item(),
                        _ief.std().item(),
                        _ief.abs().max().item(),
                        _ief.sum().item(),
                    )
                    # Optional element-wise dump (rank 0 only) for cross-TP diffing.
                    _dump = os.environ.get("KIMI_EMB_DUMP")
                    if _dump and get_tensor_model_parallel_rank() == 0:
                        torch.save(
                            {
                                "emb": inputs_embeds_fused.detach().float().cpu(),
                                "n": inputs_embeds_fused.shape[0],
                                "tp": get_tensor_model_parallel_world_size(),
                            },
                            f"{_dump}/emb_n{inputs_embeds_fused.shape[0]}_tp{get_tensor_model_parallel_world_size()}.pt",
                        )

        # 2. Forward through layers 0-21 (first 22 layers)
        hidden_states, residual = self._forward_to_layer_21(input_ids, positions, inputs_embeds_fused)

        # 3. BIFURCATION: Clone for main and MIMO paths.
        #    The reference architecture (MoonshotKimiaForCausalLM) uses:
        #      - main transformer output + lm_head  -> audio logits
        #      - MIMO transformer output + mimo_output -> text logits
        main_hidden = hidden_states.clone()
        main_residual = residual.clone() if residual is not None else None

        mimo_hidden = hidden_states.clone()
        mimo_residual = residual.clone() if residual is not None else None

        # Bifurcation-input (layer-21 output) dump: the shared-backbone result
        # feeding BOTH the main and mimo branches. Comparing TP=1 vs TP=2 here
        # isolates whether the audio divergence is seeded upstream (backbone
        # all_reduce) or inside the mimo layers themselves.
        if os.environ.get("KIMI_MIMO_DEBUG") == "1" and not torch.cuda.is_current_stream_capturing():
            with torch.no_grad():
                _l21 = hidden_states + residual if residual is not None else hidden_states
                _l21f = _l21.float()
                # Log the last-token slice always (decode + prefill share the same
                # final position) so prefill vs decode can be compared 1:1. shape[0]
                # distinguishes prefill (>2) from decode (<=2).
                logger.info(
                    "[MIMO-IN] l21 n=%d [mean=%.4f std=%.4f absmax=%.3f] last[mean=%.4f std=%.4f absmax=%.3f]",
                    hidden_states.shape[0],
                    _l21f.mean().item(),
                    _l21f.std().item(),
                    _l21f.abs().max().item(),
                    _l21f[-1].mean().item(),
                    _l21f[-1].std().item(),
                    _l21f[-1].abs().max().item(),
                )

        # 4. Main path: layers 22-27 (remaining 6 layers of main backbone)
        for layer in self.model.model.layers[22:]:
            main_hidden, main_residual = layer(positions, main_hidden, main_residual)

        # Apply final norm for main path
        # Handle residual connection: add residual to hidden_states before norm
        if main_residual is not None:
            main_hidden = main_hidden + main_residual
        main_hidden = self.model.model.norm(main_hidden)

        # 5. MIMO path: MIMO layers (6 text-specific layers)
        _mimo_dbg = os.environ.get("KIMI_MIMO_DEBUG") == "1" and not torch.cuda.is_current_stream_capturing()
        for _li, layer in enumerate(self.mimo_layers):
            mimo_hidden, mimo_residual = layer(positions, mimo_hidden, mimo_residual)
            # Per-layer running-hidden dump (gated) to localize TP=1-vs-TP=2 divergence.
            if _mimo_dbg and mimo_hidden.shape[0] <= 2:
                with torch.no_grad():
                    _run = mimo_hidden + mimo_residual if mimo_residual is not None else mimo_hidden
                    _r = _run.float()
                    logger.info(
                        "[MIMO-L%d] run[mean=%.4f std=%.4f absmax=%.3f]",
                        _li,
                        _r.mean().item(),
                        _r.std().item(),
                        _r.abs().max().item(),
                    )

        # Apply final norm for MIMO path
        if mimo_residual is not None:
            mimo_hidden = mimo_hidden + mimo_residual
        mimo_hidden = self.mimo_norm(mimo_hidden)

        # 6. Compute AUDIO logits from the MIMO (audio) branch output via
        #    mimo_output. Empirically, for this checkpoint, mimo_output is the
        #    AUDIO head and lm_head is the TEXT head (verified 2026-07-11:
        #    lm_head(main_hidden) argmax for "Say hello..." = 9707 "Hello",
        #    whereas mimo_output(*) only emits audio-token ids >= 152064). This
        #    is the OPPOSITE of the reference modeling_kimia.py comments, so we
        #    route by measured behavior rather than by name.
        audio_logits = self.logits_processor(self.mimo_output, mimo_hidden)

        # ColumnParallelLinear may return a tuple on some TP layouts; keep the
        # tensor when available.
        if isinstance(audio_logits, tuple):
            audio_logits = audio_logits[0]

        if audio_logits is None:
            audio_logits = torch.empty(
                mimo_hidden.shape[0],
                self.config.vocab_size,
                device=mimo_hidden.device,
                dtype=mimo_hidden.dtype,
            )

        # 7. Store audio logits for later retrieval by sample()
        self._pending_audio_logits = audio_logits

        # Logits-head dump (TP-comparable): capture the AUDIO head's raw input
        # (mimo_hidden) and raw output (audio_logits) at the last position, plus
        # the TEXT head's input (main_hidden). compute_logits() dumps the text
        # head's raw output under the same step index. Comparing TP=1 vs TP=2
        # bisects whether the residual divergence is IN the vocab-sharded head
        # GEMMs (input matches, output differs) or upstream (input differs).
        if (
            os.environ.get("KIMI_EMB_DUMP")
            and os.environ.get("KIMI_LOGIT_DUMP") == "1"
            and not torch.cuda.is_current_stream_capturing()
            and get_tensor_model_parallel_rank() == 0
            and audio_logits.shape[0] <= 2
        ):
            with torch.no_grad():
                _step = self._logit_dump_step
                self._logit_dump_step += 1
                self._last_logit_dump_step = _step
                torch.save(
                    {
                        "step": _step,
                        "mimo_hidden_last": mimo_hidden[-1:].detach().float().cpu(),
                        "main_hidden_last": main_hidden[-1:].detach().float().cpu(),
                        "audio_logits_last": audio_logits[-1:].detach().float().cpu(),
                    },
                    f"{os.environ['KIMI_EMB_DUMP']}/logits_audio_step{_step}_tp{get_tensor_model_parallel_world_size()}.pt",
                )

        # Optional MIMO-branch health dump (decode steps only) for TP debugging.
        # Enable with KIMI_MIMO_DEBUG=1. Compares the audio branch (mimo_hidden)
        # against the text branch (main_hidden) and reports the top audio token.
        if (
            os.environ.get("KIMI_MIMO_DEBUG") == "1"
            and not torch.cuda.is_current_stream_capturing()
            and mimo_hidden.shape[0] <= 2
        ):
            with torch.no_grad():
                mh = mimo_hidden.float()
                ah = main_hidden.float()
                last = audio_logits[-1].float()
                tok_off = self._token_offset
                audio_range = last[tok_off:]
                eod_max = (
                    max(float(last[e].item()) for e in self._audio_eod_ids) if self._audio_eod_ids else float("nan")
                )
                logger.info(
                    "[MIMO-DBG] mh[mean=%.4f std=%.4f absmax=%.3f] mainh[mean=%.4f std=%.4f] "
                    "| top_audio_tok=%d top_audio_logit=%.3f | best_eod_logit=%.3f",
                    mh.mean().item(),
                    mh.std().item(),
                    mh.abs().max().item(),
                    ah.mean().item(),
                    ah.std().item(),
                    int(audio_range.argmax().item()) + tok_off,
                    float(audio_range.max().item()),
                    eod_max,
                )

        # 8. Determine per-request output modality from runtime metadata.
        #    task_type=["tts"] means we must generate audio until the audio
        #    stream emits an EOD token; otherwise stop when the text stream
        #    emits its EOS token.
        num_reqs = audio_logits.shape[0]
        self._ensure_audio_state(num_reqs, audio_logits.device)

        # Per-request state is reset when the scheduler reuses a slot for a new
        # request (see _reset_audio_state_for_new_requests); we only refresh task
        # metadata here so each forward step has the correct output_type /
        # max_audio_tokens.  State is keyed by request ID, with a transient
        # slot->req_id mapping for steps that do not carry runtime metadata.
        runtime_infos = kwargs.get("runtime_additional_information") or kwargs.get("model_intermediate_buffer")
        if not torch.cuda.is_current_stream_capturing():
            logger.debug(
                "[forward] num_reqs=%d runtime_infos type=%s len=%s keys_first=%s",
                num_reqs,
                type(runtime_infos).__name__,
                len(runtime_infos) if isinstance(runtime_infos, list) else "n/a",
                list(runtime_infos[0].keys()) if isinstance(runtime_infos, list) and runtime_infos else "n/a",
            )

        def _req_id_for(info: Any, slot: int) -> str:
            if isinstance(info, dict):
                return info.get("req_id") or f"__slot_{slot}"
            return f"__slot_{slot}"

        def _output_type_for(info: Any) -> str:
            task_type = info.get("task_type") if isinstance(info, dict) else None
            return "audio" if (isinstance(task_type, (list, tuple)) and "tts" in task_type) else "text"

        def _max_audio_tokens_for(info: Any) -> int | None:
            max_audio_tokens = info.get("max_audio_tokens") if isinstance(info, dict) else None
            if isinstance(max_audio_tokens, (list, tuple)) and max_audio_tokens:
                max_audio_tokens = max_audio_tokens[0]
            return int(max_audio_tokens) if max_audio_tokens is not None else None

        def _target_text_token_ids_for(info: Any) -> list[int] | None:
            target = info.get("target_text_token_ids") if isinstance(info, dict) else None
            if isinstance(target, (list, tuple)) and target:
                target = target[0]
            if isinstance(target, list) and all(isinstance(t, int) for t in target):
                return target
            return None

        def _audio_sampling_params_for(info: Any) -> tuple[float, int]:
            temperature = info.get("audio_temperature") if isinstance(info, dict) else None
            top_k = info.get("audio_top_k") if isinstance(info, dict) else None
            if isinstance(temperature, (list, tuple)) and temperature:
                temperature = temperature[0]
            if isinstance(top_k, (list, tuple)) and top_k:
                top_k = top_k[0]
            try:
                temperature = float(temperature) if temperature is not None else 0.8
            except (TypeError, ValueError):
                temperature = 0.8
            try:
                top_k = int(top_k) if top_k is not None else 10
            except (TypeError, ValueError):
                top_k = 10
            return temperature, top_k

        if isinstance(runtime_infos, list) and runtime_infos:
            if len(runtime_infos) == 1 and num_reqs > 1:
                # Prefill case: runtime info is per-request, but audio_logits has
                # one row per scheduled token.  Broadcast the single request's
                # metadata to all token positions.
                runtime_info = runtime_infos[0]
                output_type = _output_type_for(runtime_info)
                max_audio_tokens = _max_audio_tokens_for(runtime_info)
                target_text_token_ids = _target_text_token_ids_for(runtime_info)
                audio_temperature, audio_top_k = _audio_sampling_params_for(runtime_info)
                req_id = _req_id_for(runtime_info, 0)
                self._audio_state.setdefault(req_id, self._default_audio_state())
                for batch_i in range(num_reqs):
                    self._slot_to_req_id[batch_i] = req_id
                    self._audio_state[req_id]["output_type"] = output_type
                    self._audio_state[req_id]["req_id"] = req_id
                    if max_audio_tokens is not None:
                        self._audio_state[req_id]["max_audio_tokens"] = max_audio_tokens
                    if target_text_token_ids is not None:
                        self._audio_state[req_id]["target_text_token_ids"] = target_text_token_ids
                    self._audio_state[req_id]["audio_temperature"] = audio_temperature
                    self._audio_state[req_id]["audio_top_k"] = audio_top_k
                logger.debug(
                    "[forward] prefill broadcast: task_type=%s -> output_type=%s "
                    "max_audio_tokens=%s audio_temp=%s audio_top_k=%s for %d slots gen_len=%s",
                    runtime_info.get("task_type") if isinstance(runtime_info, dict) else None,
                    output_type,
                    max_audio_tokens,
                    audio_temperature,
                    audio_top_k,
                    num_reqs,
                    runtime_info.get("generated_len") if isinstance(runtime_info, dict) else None,
                )
            else:
                for batch_i, runtime_info in enumerate(runtime_infos[:num_reqs]):
                    req_id = _req_id_for(runtime_info, batch_i)
                    self._slot_to_req_id[batch_i] = req_id
                    state = self._audio_state.setdefault(req_id, self._default_audio_state())
                    output_type = _output_type_for(runtime_info)
                    state["output_type"] = output_type
                    state["req_id"] = req_id
                    max_audio_tokens = _max_audio_tokens_for(runtime_info)
                    if max_audio_tokens is not None:
                        state["max_audio_tokens"] = max_audio_tokens
                    target_text_token_ids = _target_text_token_ids_for(runtime_info)
                    if target_text_token_ids is not None:
                        state["target_text_token_ids"] = target_text_token_ids
                    audio_temperature, audio_top_k = _audio_sampling_params_for(runtime_info)
                    state["audio_temperature"] = audio_temperature
                    state["audio_top_k"] = audio_top_k
                    logger.debug(
                        "[forward] slot=%d req_id=%s task_type=%s -> "
                        "output_type=%s max_audio_tokens=%s audio_temp=%s "
                        "audio_top_k=%s gen_len=%s",
                        batch_i,
                        req_id,
                        runtime_info.get("task_type") if isinstance(runtime_info, dict) else None,
                        output_type,
                        state.get("max_audio_tokens"),
                        audio_temperature,
                        audio_top_k,
                        runtime_info.get("generated_len") if isinstance(runtime_info, dict) else None,
                    )

        # 9. Return the MAIN hidden states as the primary output so the runner's
        #    compute_logits() projects them with lm_head (the TEXT head for this
        #    checkpoint) to produce the TEXT logits, and return the AUDIO logits
        #    (mimo_output on the MIMO output) for the audio sampler in
        #    make_omni_output().
        return main_hidden, audio_logits

    def _default_audio_state(self) -> dict[str, Any]:
        """Return a fresh per-slot audio state dictionary."""
        return {
            "generation_step": 0,
            "audio_out_ids": None,
            "tokens_after_text": 0,
            "output_type": "text",
            "audio_finished": False,
            "text_finished": False,
            "req_id": None,
            "audio_temperature": 0.0,
            "audio_top_k": 5,
        }

    def _ensure_audio_state(self, num_reqs: int, device: torch.device) -> None:
        """Initialize per-request audio state containers if needed.

        State is keyed by request ID rather than slot index so that concurrent
        requests cannot collide when the scheduler places them in the same batch
        position across different steps.
        """
        del num_reqs, device  # state is demand-allocated by request ID
        if not hasattr(self, "_audio_state"):
            self._audio_state = {}
        if not hasattr(self, "_slot_to_req_id"):
            self._slot_to_req_id = {}

    def _reset_audio_state_for_new_requests(
        self,
        runtime_infos: list | dict | None,
        num_reqs: int,
    ) -> None:
        """Reset audio state for newly scheduled requests before embeddings.

        The runner tags every request with ``generated_len``.  A value of 0
        means this request just started, so any stale state left by a previous
        occupant of the same slot must be cleared before ``embed_input_ids``
        uses it.  State is keyed by request ID, so reset is by ``req_id`` rather
        than slot index.
        """
        if not hasattr(self, "_audio_state"):
            self._audio_state = {}
        if not hasattr(self, "_slot_to_req_id"):
            self._slot_to_req_id = {}

        # Drop slot mappings that are no longer part of the current batch.
        for slot in list(self._slot_to_req_id.keys()):
            if isinstance(slot, int) and (slot < 0 or slot >= num_reqs):
                self._slot_to_req_id.pop(slot, None)

        if not isinstance(runtime_infos, list) or not runtime_infos:
            return

        for batch_i in range(min(num_reqs, len(runtime_infos))):
            info = runtime_infos[batch_i]
            if not isinstance(info, dict):
                continue
            req_id = info.get("req_id") or f"__slot_{batch_i}"
            if info.get("generated_len", -1) == 0:
                self._audio_state[req_id] = self._default_audio_state()
                logger.debug(
                    "[forward] reset audio state for req_id=%s slot=%d",
                    req_id,
                    batch_i,
                )

    def _forward_to_layer_21(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward through layers 0-21 (first 22 layers) and return hidden states and residual."""
        hidden_states = inputs_embeds
        residual = None

        # Forward through layers 0-21 (first 22 layers)
        for layer in self.model.model.layers[:22]:  # Layers 0-21
            hidden_states, residual = layer(positions, hidden_states, residual)

        # Ensure residual has correct shape (squeeze extra dimensions if present)
        if residual is not None and residual.dim() > hidden_states.dim():
            residual = residual.view(hidden_states.shape)

        return hidden_states, residual

    def embed_multimodal(self, **kwargs: object) -> list[torch.Tensor] | None:
        """Process audio input and return multimodal embeddings.

        This method is called by vLLM's multimodal processing pipeline.
        It processes audio inputs through the Whisper encoder and VQAdaptor.
        """
        logger.debug("[embed_multimodal] Called with kwargs keys: %s", list(kwargs.keys()))

        # Parse audio input from kwargs
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            logger.debug("[embed_multimodal] No audio input found, returning []")
            return []

        logger.debug("[embed_multimodal] audio_input keys: %s", list(audio_input.keys()))

        # Process audio through Whisper encoder and VQAdaptor
        audio_embeds = self._process_audio_input(audio_input)
        logger.debug(
            "[embed_multimodal] audio_embeds shape: %s, dtype: %s", tuple(audio_embeds.shape), audio_embeds.dtype
        )

        # Return as list of 2D tensors, one per batch item
        if audio_embeds.dim() == 3:
            # Unbind batch dimension: [B, T, D] -> list of B tensors [T, D]
            result = list(audio_embeds.unbind(dim=0))
            logger.debug(
                "[embed_multimodal] Returning list of %d tensors, first shape: %s",
                len(result),
                tuple(result[0].shape) if result else "empty",
            )
            return result
        else:
            # Single sample: [T, D] -> wrap in list
            logger.debug("[embed_multimodal] Returning single-item list, shape: %s", tuple(audio_embeds.shape))
            return [audio_embeds]

    def _parse_and_validate_audio_input(self, **kwargs: object) -> dict | None:
        """Parse audio input — upstream-aligned.

        Just extracts whisper_input_features from kwargs. No discrete
        tokenization (GLM-4 not used for input comprehension).
        """
        whisper_features = kwargs.get("whisper_input_features", None)
        feature_attention_mask = kwargs.get("feature_attention_mask", None)
        audio_real_frame_counts = kwargs.get("audio_real_frame_counts", None)

        if whisper_features is None:
            return None

        logger.debug(
            "_parse_and_validate_audio_input: whisper_features shape=%s",
            tuple(whisper_features.shape) if hasattr(whisper_features, "shape") else type(whisper_features),
        )

        return {
            "whisper_input_features": whisper_features,
            "feature_attention_mask": feature_attention_mask,
            "audio_real_frame_counts": audio_real_frame_counts,
        }

    def _process_audio_input(self, audio_input: dict) -> torch.Tensor:
        """Process audio input — aligned with upstream vllm 0.23.

        Pipeline:
          Whisper-Large-v3 encoder → [B, T, 1280]
          4-frame concat via reshape → [B, T//4, 5120]
          VQ Adaptor → [B, T//4, 3584]

        The VQ Adaptor's first layer is Linear[5120→3584], so the 4-frame
        concat (not a learned projection) produces the 5120-dim input.

        The HF processor pads all inputs to the model's max audio length
        (3000 mel frames for Whisper-Large-v3).  To avoid feeding silence
        through the encoder and corrupting the real audio positions, we
        truncate ``input_features`` to the real per-item frame count given
        by ``audio_real_frame_counts`` before running the audio tower.
        """
        input_features = audio_input["whisper_input_features"]
        feature_attention_mask = audio_input.get("feature_attention_mask", None)
        audio_real_frame_counts = audio_input.get("audio_real_frame_counts", None)

        logger.debug(
            "[_process_audio_input] input_features shape: %s, mask: %s, real_counts: %s",
            input_features.shape if hasattr(input_features, "shape") else "list",
            feature_attention_mask.shape if hasattr(feature_attention_mask, "shape") else None,
            audio_real_frame_counts.shape if hasattr(audio_real_frame_counts, "shape") else None,
        )

        # Truncate padded mel frames to the real audio length.
        # input_features: [B, 128, padded_T]
        # audio_real_frame_counts: [B] (per-item real frame counts)
        real_counts_list: list[int] | None = None
        if audio_real_frame_counts is not None and hasattr(audio_real_frame_counts, "tolist"):
            real_counts_list = [int(x) for x in audio_real_frame_counts.tolist()]
        elif feature_attention_mask is not None and input_features.dim() == 3 and feature_attention_mask.dim() == 2:
            real_counts_list = [int(feature_attention_mask[b].sum().item()) for b in range(input_features.shape[0])]

        if real_counts_list is not None and input_features.dim() == 3:
            B = input_features.shape[0]
            truncated_features = []
            for b in range(B):
                real_frames = real_counts_list[b]
                if 0 < real_frames < input_features.shape[-1]:
                    truncated_features.append(input_features[b, :, :real_frames])
                else:
                    truncated_features.append(input_features[b])
            input_features = truncated_features
            logger.debug(
                "[_process_audio_input] Truncated to real lengths: %s, first shape: %s",
                real_counts_list,
                tuple(input_features[0].shape) if input_features else None,
            )
        elif input_features.dim() == 3:
            # Fall back to upstream list-of-tensors behaviour.
            input_features = list(input_features.unbind(dim=0))

        # Run through Whisper encoder
        audio_features = self.audio_tower(input_features)
        # audio_features: [B, T, 1280]
        logger.debug("[_process_audio_input] After encoder: %s", audio_features.shape)

        # Gated whisper-tower OUTPUT dump (rank 0): bisects whether the cross-TP
        # audio-feature divergence is seeded INSIDE the whisper encoder or
        # downstream (reshape/projector). Compare whisper_n{T}_tp{1,2}.pt.
        _wdump = os.environ.get("KIMI_EMB_DUMP")
        if (
            _wdump
            and os.environ.get("KIMI_MIMO_DEBUG") == "1"
            and not torch.cuda.is_current_stream_capturing()
            and get_tensor_model_parallel_rank() == 0
        ):
            with torch.no_grad():
                _mel = (
                    [t.detach().float().cpu() for t in input_features]
                    if isinstance(input_features, list)
                    else input_features.detach().float().cpu()
                )
                torch.save(
                    {
                        "whisper": audio_features.detach().float().cpu(),
                        "mel": _mel,
                        "tp": get_tensor_model_parallel_world_size(),
                    },
                    f"{_wdump}/whisper_n{audio_features.shape[1]}_tp{get_tensor_model_parallel_world_size()}.pt",
                )

        # 4-frame concat via reshape: [B, T, 1280] → [B, T//4, 5120]
        B, T, D = audio_features.shape
        if T % 4 != 0:
            pad_len = 4 - (T % 4)
            audio_features = torch.nn.functional.pad(audio_features, (0, 0, 0, pad_len))
            T = audio_features.shape[1]
            logger.debug("[_process_audio_input] Padded to: %s", audio_features.shape)

        audio_features = audio_features.reshape(B, T // 4, D * 4)
        logger.debug("[_process_audio_input] After reshape: %s", audio_features.shape)

        # Project to LLM dimension: [B, T//4, 5120] → [B, T//4, 3584]
        audio_embeds = self.multi_modal_projector(audio_features)

        logger.debug(
            "[ASR-DEBUG] _process_audio_input output audio_embeds shape=%s mean=%s std=%s",
            audio_embeds.shape,
            audio_embeds.mean().item(),
            audio_embeds.std().item(),
        )
        logger.debug("[_process_audio_input] Final audio_embeds shape: %s", audio_embeds.shape)
        return audio_embeds

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: torch.Tensor | None = None,
        is_multimodal: torch.Tensor | None = None,
        runtime_additional_information: list | dict | None = None,
    ) -> torch.Tensor:
        """Embed input IDs with dual-stream fusion.

        Fusion formula (from reference implementation):
        - For continuous whisper features: (text_emb + whisper_emb) × √2
        - For discrete audio tokens: text_emb + audio_emb (simple addition)

        For TTS requests the audio stream markers (role starts, continuation
        tokens, message ends) are supplied through
        ``runtime_additional_information`` and added to the text-stream
        embeddings during the prefill, matching the reference dual-stream
        prompt layout.
        """
        # 1. Embed text tokens
        logger.debug(
            "[embed_input_ids] called input_ids=%s mm=%s is_mm=%s runtime=%s",
            input_ids.shape if input_ids is not None else None,
            multimodal_embeddings is not None,
            is_multimodal is not None,
            runtime_additional_information is not None,
        )
        text_emb = self.model.model.embed_tokens(input_ids)

        # 2. Handle whisper continuous features with √2 scaling
        if multimodal_embeddings is not None:
            logger.debug(
                "[ASR-DEBUG] embed_input_ids mm=%s is_mm=%s",
                multimodal_embeddings.shape if hasattr(multimodal_embeddings, "shape") else type(multimodal_embeddings),
                is_multimodal.shape if is_multimodal is not None else None,
            )
            if is_multimodal is not None and not torch.cuda.is_current_stream_capturing():
                logger.debug("[ASR-DEBUG] is_multimodal sum=%s", is_multimodal.sum().item())
            logger.debug("input_ids shape: %s", input_ids.shape)
            logger.debug("multimodal_embeddings type: %s", type(multimodal_embeddings))
            if isinstance(multimodal_embeddings, list):
                logger.debug("multimodal_embeddings is list with %s items", len(multimodal_embeddings))
                for i, item in enumerate(multimodal_embeddings):
                    logger.debug("  Item %s: type=%s", i, type(item))
                    if hasattr(item, "shape"):
                        logger.debug("shape=%s, dtype=%s", item.shape, item.dtype)
                    elif item is None:
                        logger.debug("value=None")
                    else:
                        logger.debug("value=%s", item)
            else:
                logger.debug("multimodal_embeddings shape: %s", multimodal_embeddings.shape)
            if is_multimodal is not None:
                logger.debug("is_multimodal shape: %s", is_multimodal.shape)
                # CUDA graph compatible logging
                if not torch.cuda.is_current_stream_capturing():
                    logger.debug("is_multimodal sum: %s", is_multimodal.sum().item())

            # Ensure multimodal_embeddings is a tensor (not a list)
            if isinstance(multimodal_embeddings, list):
                # DEBUG: Log before filtering
                logger.debug("BEFORE FILTER: multimodal_embeddings has %d items", len(multimodal_embeddings))
                for i, item in enumerate(multimodal_embeddings):
                    if item is None:
                        logger.debug("  Item %d: None", i)
                    elif isinstance(item, torch.Tensor):
                        logger.debug("  Item %d: Tensor shape=%s", i, item.shape)
                    else:
                        logger.debug("  Item %d: type=%s", i, type(item))

                # Filter out None items and check if list is empty
                multimodal_embeddings = [
                    item for item in multimodal_embeddings if item is not None and isinstance(item, torch.Tensor)
                ]
                if len(multimodal_embeddings) == 0:
                    # No valid embeddings, skip multimodal fusion
                    logger.debug("No valid multimodal embeddings after filtering")
                    multimodal_embeddings = None
                else:
                    # Stack list items: [N, seq_len, hidden_size] where N is number of audio items
                    multimodal_embeddings = torch.stack(multimodal_embeddings)

            # Only process multimodal embeddings if we have valid tensors
            if multimodal_embeddings is not None:
                if is_multimodal is not None:
                    # is_multimodal marks which positions in the sequence should use audio features
                    # multimodal_embeddings shape: [num_audio_items, audio_seq_len, hidden_size]
                    # We need to place audio features at positions where is_multimodal is True

                    # For now, assume single audio item (most common case)
                    if multimodal_embeddings.shape[0] == 1:
                        audio_features = multimodal_embeddings[0]  # [audio_seq_len, hidden_size]
                        # CUDA graph compatible: keep as tensor
                        num_audio_positions = is_multimodal.sum()

                        # CUDA graph compatible: convert to int for slicing.
                        # During capture, assume the counts match and use the
                        # full embedding length to avoid data-dependent slicing.
                        if not torch.cuda.is_current_stream_capturing():
                            num_audio_int = int(num_audio_positions.item())
                        else:
                            num_audio_int = audio_features.shape[0]

                        # The BLANK count computed by the processor may differ
                        # from the exact encoder/reshape length by a few frames
                        # (padding/stride rounding).  Fuse at the first
                        # min(...) BLANK positions instead of giving up.
                        num_to_use = min(audio_features.shape[0], num_audio_int)
                        if num_to_use <= 0:
                            logger.debug(
                                "embed_input_ids: slot=0 NO multimodal fusion audio_features=%d positions=%d",
                                audio_features.shape[0],
                                num_audio_int,
                            )
                        else:
                            logger.debug(
                                "embed_input_ids: slot=0 fusing audio_features=%d positions=%d using=%d",
                                audio_features.shape[0],
                                num_audio_int,
                                num_to_use,
                            )
                            if not torch.cuda.is_current_stream_capturing():
                                if audio_features.shape[0] != num_audio_int:
                                    logger.debug(
                                        "embed_input_ids: audio length mismatch "
                                        "audio_features=%d positions=%d; fusing first %d",
                                        audio_features.shape[0],
                                        num_audio_int,
                                        num_to_use,
                                    )
                                else:
                                    logger.debug(
                                        "embed_input_ids: fusing audio_features=%d positions=%d",
                                        audio_features.shape[0],
                                        num_audio_int,
                                    )

                            multimodal_positions = torch.where(is_multimodal)[0][:num_to_use]
                            used_audio_features = audio_features[:num_to_use]

                            # text_emb at BLANK positions + projected audio features,
                            # scaled by √2 (matches upstream vllm KimiAudioForCausalLM)
                            result_emb = text_emb.clone()
                            text_at_mm = result_emb[multimodal_positions]
                            combined_emb = (text_at_mm + used_audio_features) * (2**0.5)
                            result_emb[multimodal_positions] = combined_emb
                            text_emb = result_emb
                    else:
                        # Multiple audio items - not yet implemented
                        raise NotImplementedError("Multiple audio items not yet supported")
                else:
                    # No mask provided, apply √2 scaling to all
                    text_emb = (text_emb + multimodal_embeddings) * (2**0.5)

        # 2.5 Add prompt audio-stream markers for TTS requests.
        #     The TTS adapter supplies the audio stream separately because vLLM
        #     only accepts a single input_ids sequence.  During the prefill we
        #     add the audio-token embeddings at each position to recover the
        #     reference dual-stream embedding: embed(text_stream[i]) +
        #     embed(audio_stream[i]).
        audio_stream = None
        if runtime_additional_information is not None:
            logger.debug(
                "[embed_input_ids] runtime_additional_information type=%s",
                type(runtime_additional_information),
            )
            if isinstance(runtime_additional_information, list) and runtime_additional_information:
                info = runtime_additional_information[0]
                logger.debug(
                    "[embed_input_ids] first info type=%s keys=%s",
                    type(info),
                    list(info.keys()) if isinstance(info, dict) else "n/a",
                )
                if isinstance(info, dict):
                    audio_stream = info.get("audio_stream")
            elif isinstance(runtime_additional_information, dict):
                logger.debug(
                    "[embed_input_ids] dict keys=%s",
                    list(runtime_additional_information.keys()),
                )
                audio_stream = runtime_additional_information.get("audio_stream")

        if audio_stream is not None:
            # audio_stream is batched: [[...]]; take the first batch item.
            if isinstance(audio_stream, (list, tuple)) and audio_stream and isinstance(audio_stream[0], (list, tuple)):
                audio_stream = audio_stream[0]
            if isinstance(audio_stream, torch.Tensor):
                audio_stream = audio_stream.tolist()
            if audio_stream:
                flat_input_ids = input_ids.view(-1)
                num_tokens = flat_input_ids.shape[0]
                stream_len = len(audio_stream)
                if num_tokens == stream_len:
                    audio_stream_ids = torch.tensor(
                        audio_stream,
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    audio_marker_emb = self.model.model.embed_tokens(audio_stream_ids)
                    # Flatten text_emb to [num_tokens, hidden_size] so the addition
                    # works regardless of whether input_ids was [num_tokens] or
                    # [batch, num_tokens].
                    orig_shape = text_emb.shape
                    text_emb_flat = text_emb.view(-1, text_emb.shape[-1])
                    text_emb_flat = text_emb_flat + audio_marker_emb.to(dtype=text_emb_flat.dtype)
                    text_emb = text_emb_flat.view(orig_shape)
                    logger.debug(
                        "[embed_input_ids] fused audio_stream markers for %d positions",
                        num_tokens,
                    )
                else:
                    logger.warning(
                        "[embed_input_ids] audio_stream length mismatch: "
                        "input_ids=%d audio_stream=%d; skipping marker fusion",
                        num_tokens,
                        stream_len,
                    )

        # 3. Embed audio tokens from per-request state (dual token stream fusion)
        # For each request, add the delayed audio token embedding from previous
        # steps.  The audio stream lags the text stream by ``_audio_delay`` tokens,
        # so we prepend that many BLANK placeholders to the generated audio sequence
        # before taking the last token for feedback.
        state = getattr(self, "_audio_state", None)
        slot_to_req_id = getattr(self, "_slot_to_req_id", {})
        if not state or not slot_to_req_id:
            # No audio state yet (first step or no audio generation)
            inputs_embeds = text_emb
            logger.debug("embed_input_ids: No audio state, using text_emb only")
        else:
            num_tokens = text_emb.shape[0]

            logger.debug(
                "embed_input_ids: Audio state exists, applying dual stream fusion for %d tokens",
                num_tokens,
            )

            # Determine batch row indices for each token position.
            # In vLLM, input_ids is typically [num_tokens] for prefill or
            # [num_reqs, 1] for decode.
            if input_ids.dim() == 1:
                # Prefill or single request: all tokens map to slot 0
                batch_row_indices = [0] * num_tokens
            else:
                # Decode: each row is one request with 1 token
                num_reqs = input_ids.shape[0]
                batch_row_indices = list(range(num_reqs))

            # Start with text embeddings
            inputs_embeds = text_emb.clone()

            # BLANK prefix used to implement the audio delay for feedback.
            blank_prefix = torch.full(
                (1, self._audio_delay),
                self._blank_token_id,
                device=text_emb.device,
                dtype=torch.long,
            )

            # For each position, add the delayed audio embedding from the
            # corresponding request's state.
            audio_tokens_added = 0
            for pos in range(num_tokens):
                batch_i = batch_row_indices[pos] if pos < len(batch_row_indices) else 0
                req_id = slot_to_req_id.get(batch_i)
                if req_id is None:
                    continue

                req_state = state.get(req_id)
                if req_state is None:
                    continue

                audio_out_ids = req_state.get("audio_out_ids")
                if audio_out_ids is None or audio_out_ids.numel() == 0:
                    continue

                # Prepend BLANK delay tokens so the feedback lags by 6 steps.
                delayed_ids = torch.cat([blank_prefix, audio_out_ids], dim=-1)
                last_audio_token = delayed_ids[:, -1:]  # [1, 1] GLOBAL token id

                # Embed the audio token via the (vocab-parallel) embedding MODULE,
                # never the raw sharded ``.weight``. Under TP>1 ``embed_tokens.weight``
                # holds only this rank's vocab shard (rank0=[0,V/2), rank1=[V/2,V)),
                # and every audio token id (>= KIMI_AUDIO_TOKEN_OFFSET=152064) lives
                # in the upper shard. Direct indexing plus a clamp to the LOCAL shard
                # size silently maps all of them to the shard boundary, so each rank
                # feeds back a constant, wrong embedding (rank0 -> token V/2-1, rank1
                # -> token V-1). The mimo branch then never emits EOD and S2S audio
                # over-generates to the cap. The module forward performs the shard-
                # aware lookup + all-reduce and returns the correct full embedding of
                # the GLOBAL id, identical on every rank (TP=1: weight is full, so the
                # module path is equivalent to the old direct index). Clamp to the
                # GLOBAL vocab size, not the local shard size.
                last_audio_token_id = torch.clamp(last_audio_token.reshape(-1), 0, self.config.vocab_size - 1)
                audio_emb = self.model.model.embed_tokens(last_audio_token_id)  # [1, hidden]
                inputs_embeds[pos] = inputs_embeds[pos] + audio_emb[0].to(dtype=inputs_embeds.dtype)
                audio_tokens_added += 1

            logger.debug(
                "embed_input_ids: Added %d audio token embeddings",
                audio_tokens_added,
            )

        return inputs_embeds

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute text logits via lm_head on the main (text) hidden states."""
        # For this checkpoint lm_head is the TEXT head and forward() returns the
        # MAIN hidden states as the primary output, so ``hidden_states`` here is
        # the main output. (Reference modeling_kimia.py names the two heads the
        # opposite way; we route by measured behavior — see forward() note.)
        logits = self.logits_processor(self.model.lm_head, hidden_states)

        # Logits-head dump (TP-comparable): the TEXT head's raw input
        # (hidden_states == main_hidden last position) and raw output BEFORE the
        # audio-token/EOD masking below, under the step index set by forward().
        # Matched with logits_audio_step{N} to bisect head vs upstream divergence.
        if (
            os.environ.get("KIMI_EMB_DUMP")
            and os.environ.get("KIMI_LOGIT_DUMP") == "1"
            and not torch.cuda.is_current_stream_capturing()
            and get_tensor_model_parallel_rank() == 0
            and logits is not None
            and logits.shape[0] <= 2
        ):
            with torch.no_grad():
                _step = self._last_logit_dump_step
                torch.save(
                    {
                        "step": _step,
                        "text_hidden_last": hidden_states[-1:].detach().float().cpu(),
                        "text_logits_last_raw": logits[-1:].detach().float().cpu(),
                    },
                    f"{os.environ['KIMI_EMB_DUMP']}/logits_text_step{_step}_tp{get_tensor_model_parallel_world_size()}.pt",
                )

        # CRITICAL: Mask out audio tokens from text logits
        # Kimi Audio uses a unified vocabulary where:
        # - Text tokens: [0, KIMI_AUDIO_TOKEN_OFFSET - 1]
        # - Audio tokens: [KIMI_AUDIO_TOKEN_OFFSET, vocab_size)
        # We need to prevent the text sampler from selecting audio tokens
        if logits is not None and logits.shape[-1] > KIMI_AUDIO_TOKEN_OFFSET:
            # Set audio token logits to -inf so they won't be sampled
            logits[:, KIMI_AUDIO_TOKEN_OFFSET:] = -float("inf")

        # Audio EOD markers (e.g. <|im_msg_end|>, <|im_media_end|>) live in the
        # text vocabulary.  If the text stream samples them during TTS, vLLM's
        # stop-token logic aborts generation after one step.  Mask them so the
        # text stream keeps producing real tokens/audio-conditioning tokens.
        if logits is not None and self._audio_eod_ids:
            audio_eod_ids = list(self._audio_eod_ids)
            logits[:, audio_eod_ids] = -float("inf")

        logger.debug(
            "compute_logits: hidden_states shape=%s, logits shape=%s, hidden_mean=%.4f, hidden_std=%.4f",
            hidden_states.shape,
            logits.shape if logits is not None else "None",
            hidden_states.mean(),
            hidden_states.std(),
        )
        if logits is not None:
            logger.debug(
                "compute_logits: logits_mean=%.4f, logits_std=%.4f, logits_max=%.4f",
                logits.mean(),
                logits.std(),
                logits.max(),
            )
            # Log top 5 tokens for debugging
            top5 = torch.topk(logits[0], 5)
            logger.debug(
                "[ASR-DEBUG] compute_logits top5 token_ids=%s, top5 logits=%s",
                top5.indices.tolist(),
                top5.values.tolist(),
            )

        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        """Custom sampler for dual token streaming.

        Text tokens drive the engine.  Audio tokens are sampled earlier in
        make_omni_output(); here we only decide when to stop the engine:

        * text output: stop when the text stream emits ``kimia_text_eos``.
        * audio output: continue after text EOS (feed BLANK text tokens) until
          the audio stream emits an EOD marker (``im_media_end``/``im_msg_end``).
        """
        if not hasattr(self, "_audio_state"):
            self._audio_state = {}

        sampler = getattr(self, "_stock_sampler", None)
        if sampler is None:
            from vllm.v1.sample.sampler import Sampler

            sampler = Sampler()
            self._stock_sampler = sampler

        num_reqs = logits.shape[0]
        batch_row_indices = list(range(num_reqs))
        slot_to_req_id = getattr(self, "_slot_to_req_id", {})

        # Slot reuse is now handled in forward() before embeddings are computed,
        # using the runner-provided generated_len.  We no longer pop state here
        # because doing so would discard output_type set by forward() on the
        # first step of a new request.

        # Sample the text token from the (audio-masked) text logits.
        text_sampler_output = sampler(logits=logits, sampling_metadata=sampling_metadata)
        text_tokens = text_sampler_output.sampled_token_ids  # [num_reqs, 1]

        for batch_i in batch_row_indices:
            req_id = slot_to_req_id.get(batch_i)
            if req_id is None:
                # No runtime metadata for this slot; treat as text-only.
                continue
            state = self._audio_state.setdefault(req_id, self._default_audio_state())
            output_type = state.get("output_type", "text")
            audio_finished = state.get("audio_finished", False)
            logger.debug(
                "[sample] req_id=%s slot=%d output_type=%s gen_step=%s "
                "audio_finished=%s text_token=%s target_len=%s target_pos=%s",
                req_id,
                batch_i,
                output_type,
                state.get("generation_step", 0),
                audio_finished,
                text_tokens[batch_i].item() if text_tokens.numel() > batch_i else None,
                len(state.get("target_text_token_ids", [])) if state.get("target_text_token_ids") else 0,
                state.get("target_text_position", 0),
            )

            if text_tokens.numel() <= batch_i:
                continue
            text_token_id = text_tokens[batch_i]

            if output_type == "audio":
                if not torch.cuda.is_current_stream_capturing():
                    logger.debug(
                        "[sample] req_id=%s slot=%d text_token_sampled=%s output_type=%s audio_finished=%s",
                        req_id,
                        batch_i,
                        text_token_id.item() if text_token_id.numel() else None,
                        output_type,
                        audio_finished,
                    )
                if audio_finished:
                    # Audio stream emitted EOD this step; tell the engine to stop.
                    text_tokens[batch_i] = self._engine_eos_id
                else:
                    target = state.get("target_text_token_ids") or []
                    if target:
                        # Teacher-forced TTS (future TTS-capable checkpoint): the
                        # text stream is driven by the supplied transcript rather
                        # than sampled.  Mirrors the reference ``audio-text``
                        # training format; padded with BLANK to the audio length.
                        pos = state.get("target_text_position", 0)
                        if pos < len(target):
                            text_tokens[batch_i] = int(target[pos])
                            state["target_text_position"] = pos + 1
                        else:
                            # Transcript exhausted: pad the text stream with
                            # BLANK while the audio stream completes.
                            state["text_finished"] = True
                            state["tokens_after_text"] = state.get("tokens_after_text", 0) + 1
                            text_tokens[batch_i] = self._blank_token_id
                    else:
                        # S2S / reference ``output_type="both"``: the text stream
                        # is generated freely and fed back each decode step.  On
                        # text EOS, pad BLANK while the audio stream completes;
                        # the engine stops when the audio stream emits EOD (the
                        # ``audio_finished`` branch above).
                        if state.get("text_finished"):
                            text_tokens[batch_i] = self._blank_token_id
                            state["tokens_after_text"] = state.get("tokens_after_text", 0) + 1
                        elif text_token_id.numel() and int(text_token_id.item()) == self._text_eos_id:
                            state["text_finished"] = True
                            text_tokens[batch_i] = self._blank_token_id
                        # else: keep the sampled text token so it feeds back as
                        # the next text-stream input (dual-stream feedback).

                    max_audio_tokens = state.get("max_audio_tokens", 200)
                    if state.get("tokens_after_text", 0) >= max_audio_tokens:
                        text_tokens[batch_i] = self._engine_eos_id
                        if not torch.cuda.is_current_stream_capturing():
                            logger.debug(
                                "[sample] req_id=%s slot=%d audio length cap "
                                "reached tokens_after_text=%d >= %d, forcing EOS",
                                req_id,
                                batch_i,
                                state["tokens_after_text"],
                                max_audio_tokens,
                            )
                if not torch.cuda.is_current_stream_capturing():
                    logger.info(
                        "[sample] req_id=%s slot=%d text_token_out=%s "
                        "target_pos=%d target_len=%d tokens_after_text=%d "
                        "text_finished=%s audio_finished=%s gen_step=%s",
                        req_id,
                        batch_i,
                        text_tokens[batch_i].item(),
                        state.get("target_text_position", 0),
                        len(state.get("target_text_token_ids", [])),
                        state.get("tokens_after_text", 0),
                        state.get("text_finished", False),
                        audio_finished,
                        state.get("generation_step", 0),
                    )
            else:
                # Text-only output: stop when the text stream reaches its EOS.
                if text_token_id.numel() and int(text_token_id.item()) == self._text_eos_id:
                    text_tokens[batch_i] = self._engine_eos_id
                # Audio tokens are forced to BLANK in make_omni_output().

            state["generation_step"] = state.get("generation_step", 0) + 1

        # Sync the last audio tokens from make_omni_output's per-slot accumulation.
        last_audio_tokens = []
        for batch_i in batch_row_indices:
            req_id = slot_to_req_id.get(batch_i)
            if req_id is None:
                continue
            req_state = self._audio_state.get(req_id)
            if req_state and req_state.get("audio_out_ids") is not None:
                last_audio_tokens.append(req_state["audio_out_ids"][:, -1:])

        if last_audio_tokens:
            self._last_audio_tokens = torch.cat(last_audio_tokens, dim=0)
        else:
            self._last_audio_tokens = None

        return text_sampler_output

    def _sample_audio_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample one audio token from masked audio logits.

        Matches the reference KimiAudio sampler: apply temperature to log-probs,
        optionally restrict to the top-k tokens, then multinomial sample.  When
        temperature is near zero we fall back to greedy argmax.

        ``generator`` must be seeded identically on every tensor-parallel rank.
        The audio token is fed back as the next step's input, so if two ranks
        draw different tokens their KV caches diverge and the all-reduce in the
        next forward sums incompatible partial results, degrading the audio
        stream to noise that never emits EOD.  Seeding from rank-invariant data
        (request id + generation step) keeps all ranks in lockstep.
        """
        if temperature > 1e-6:
            logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
            logprobs = logprobs / temperature
            probs = torch.exp(logprobs)
            if top_k > 0:
                k = min(top_k, probs.shape[-1])
                top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
                sampled_idx = torch.multinomial(top_k_probs, num_samples=1, generator=generator)
                return top_k_indices.gather(-1, sampled_idx)
            return torch.multinomial(probs, num_samples=1, generator=generator)
        return torch.argmax(logits, dim=-1, keepdim=True)

    @staticmethod
    def _audio_sampling_generator(req_id: str, step: int, device: torch.device) -> torch.Generator:
        """Build a per-(request, step) CUDA generator seeded from rank-invariant data.

        ``req_id`` is broadcast by the scheduler and ``step`` is a counter that
        advances in lockstep across TP ranks, so every rank derives the same
        seed and draws the same audio token.  Python's built-in ``hash`` is
        salted per process, so use ``zlib.crc32`` for a deterministic digest.
        """
        seed = (zlib.crc32(str(req_id).encode("utf-8")) + (step + 1) * 0x9E3779B1) & 0x7FFFFFFFFFFFFFFF
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return generator

    def make_omni_output(
        self,
        model_outputs,  # Can be OmniOutput, dict, or tuple (when CUDA graphs are enabled)
        **kwargs,
    ) -> OmniOutput:
        """Package dual-stream output into OmniOutput.

        Audio tokens are sampled here, *before* text sampling, from the audio
        logits produced by the current forward pass.  This ensures the async
        chunk transport sends Stage 1 the audio token for the current step
        rather than the previous step.

        For text-only outputs (e.g. pure chat / ASR) we force the audio token
        to BLANK and do not emit any multimodal payload, so Stage 1 stays idle.
        For audio outputs (TTS) we sample from the full audio logits, because
        the audio stream's EOD markers (``im_media_end`` / ``im_msg_end``) live
        in the text portion of the unified vocabulary.
        """
        # Handle different output formats
        if isinstance(model_outputs, tuple):
            if len(model_outputs) >= 2:
                text_hidden = model_outputs[0]
                audio_logits = model_outputs[1]
            else:
                text_hidden = model_outputs[0]
                audio_logits = None
        elif isinstance(model_outputs, dict):
            text_hidden = model_outputs.get("text_hidden_states")
            audio_logits = model_outputs.get("audio_logits")
        else:
            text_hidden = getattr(model_outputs, "text_hidden_states", None)
            audio_logits = getattr(model_outputs, "audio_logits", None)
            if audio_logits is None and hasattr(model_outputs, "multimodal_outputs"):
                audio_logits = model_outputs.multimodal_outputs.get("audio_logits")

        if audio_logits is None:
            return OmniOutput(
                text_hidden_states=text_hidden,
                multimodal_outputs={"audio_tokens": None},
                next_token_id=None,
            )

        if isinstance(audio_logits, tuple):
            audio_logits = audio_logits[0]

        # During prefill the audio logits have one row per scheduled token.
        # Reduce to one row per request (the last scheduled token) using the
        # same indices the runner used to produce text logits.
        logits_index = kwargs.get("logits_index")
        is_prefill = logits_index is not None and audio_logits.shape[0] != logits_index.shape[0]
        if is_prefill:
            audio_logits = audio_logits[logits_index]

        num_reqs = audio_logits.shape[0]
        device = audio_logits.device
        self._ensure_audio_state(num_reqs, device)

        if not torch.cuda.is_current_stream_capturing():
            logger.debug(
                "[make_omni_output] num_reqs=%d is_prefill=%s logits_index=%s",
                num_reqs,
                is_prefill,
                None if logits_index is None else tuple(logits_index.shape),
            )

        last_audio_tokens: list[torch.Tensor] = []
        slot_to_req_id = getattr(self, "_slot_to_req_id", {})
        for batch_i in range(num_reqs):
            req_id = slot_to_req_id.get(batch_i)
            if req_id is None:
                logger.debug(
                    "[make_omni_output] slot=%d has no req_id mapping; skipping audio emission",
                    batch_i,
                )
                continue
            state = self._audio_state.get(req_id)
            if state is None:
                continue
            output_type = state.get("output_type", "text")
            logger.debug(
                "[make_omni_output] req_id=%s slot=%d output_type=%s gen_step=%d",
                req_id,
                batch_i,
                output_type,
                state.get("generation_step", 0),
            )

            if output_type == "text":
                # No audio emission for text-only outputs (ASR / pure chat).
                state["audio_out_ids"] = None
                continue

            # For audio outputs (TTS) emit an audio token every step, including
            # the prefill step, so the async-chunk transport can stream semantic
            # tokens to Stage 1 immediately.  The first ``_audio_delay`` steps are
            # BLANK placeholders; afterwards sample from the audio semantic
            # vocabulary plus the audio-stream EOD markers
            # (``im_media_end`` / ``im_msg_end``), which live in the text portion
            # of the unified vocabulary.  Normal text tokens are masked so the
            # stream does not collapse to the BLANK placeholder.
            step = state["generation_step"]
            audio_temperature = state.get("audio_temperature", 0.8)
            audio_top_k = state.get("audio_top_k", 10)
            # Diagnostic override: force greedy audio sampling to isolate the
            # Stage-0 forward from stochastic-seed divergence (KIMI_AUDIO_GREEDY=1).
            if os.environ.get("KIMI_AUDIO_GREEDY") == "1":
                audio_temperature = 0.0
                audio_top_k = 0
            if step < self._audio_delay:
                audio_token = torch.full((1, 1), self._blank_token_id, device=device, dtype=torch.long)
            else:
                vocab_size = audio_logits.shape[-1]
                masked_logits = audio_logits[batch_i : batch_i + 1].clone()
                # Mask all text tokens except the audio EOD markers.
                text_token_mask = torch.arange(vocab_size, device=device) < self._token_offset
                for eod_id in self._audio_eod_ids:
                    text_token_mask[eod_id] = False
                masked_logits[:, text_token_mask] = -float("inf")
                audio_token = self._sample_audio_token(
                    masked_logits,
                    audio_temperature,
                    audio_top_k,
                    generator=self._audio_sampling_generator(req_id, step, device),
                ).reshape(1, 1)

            audio_token_id = int(audio_token.item())
            if audio_token_id in self._audio_eod_ids:
                state["audio_finished"] = True
            logger.info(
                "[make_omni_output] req_id=%s slot=%d step=%d audio_token=%d output_type=%s",
                req_id,
                batch_i,
                step,
                audio_token_id,
                output_type,
            )

            if state["audio_out_ids"] is None:
                state["audio_out_ids"] = audio_token.clone()
            else:
                state["audio_out_ids"] = torch.cat([state["audio_out_ids"], audio_token], dim=-1)

            last_audio_tokens.append(state["audio_out_ids"][:, -1:])

        if last_audio_tokens:
            last_audio_tokens_t = torch.cat(last_audio_tokens, dim=0)
        else:
            last_audio_tokens_t = None

        self._last_audio_tokens = last_audio_tokens_t
        return OmniOutput(
            text_hidden_states=text_hidden,
            multimodal_outputs={"audio_tokens": last_audio_tokens_t},
            next_token_id=last_audio_tokens_t,
        )

    def postprocess(
        self,
        hidden_states: torch.Tensor,
        multimodal_outputs: dict,
        **req_infos,
    ) -> dict:
        """Per-request postprocessing to sync state.

        Returns update dict that gets merged into model_intermediate_buffer.
        Returns per-slot audio and text stream state for each request.
        """
        state = getattr(self, "_audio_state", None)
        slot_to_req_id = getattr(self, "_slot_to_req_id", {})
        text_finished = getattr(self, "_text_stream_finished", None)

        if not state and not text_finished:
            return {}

        # Get the number of requests from req_infos
        num_reqs = req_infos.get("num_reqs", 1)

        per_req_state = {}
        for batch_i in range(num_reqs):
            req_id = slot_to_req_id.get(batch_i)
            if req_id is None:
                continue
            req_state = state.get(req_id) if state else None
            if req_state is None:
                continue

            # Audio state
            if req_state.get("audio_out_ids") is not None:
                last_audio_token = req_state["audio_out_ids"][:, -1:].item()
                per_req_state[f"audio_token_{batch_i}"] = last_audio_token
                per_req_state[f"generation_step_{batch_i}"] = req_state["generation_step"]
                per_req_state[f"output_type_{batch_i}"] = req_state.get("output_type", "text")
                per_req_state[f"audio_finished_{batch_i}"] = req_state.get("audio_finished", False)

            # Text stream termination state
            per_req_state[f"text_finished_{batch_i}"] = req_state.get("text_finished", False)

        return per_req_state

    def on_requests_finished(self, finished_req_ids: list[str]) -> None:
        """Reset per-request audio state for finished requests.

        The runner calls this before scheduling the next request, so any
        lingering dual-stream state (e.g. from a TTS request) is cleared
        before the embeddings for the next request are computed.
        """
        self._pending_audio_logits = None
        if not finished_req_ids:
            return
        finished_set = set(finished_req_ids)
        state = getattr(self, "_audio_state", None)
        if state is not None:
            for req_id in list(state.keys()):
                if req_id in finished_set:
                    state.pop(req_id, None)
                    logger.debug(
                        "[on_requests_finished] removed audio state for finished req=%s",
                        req_id,
                    )

        slot_to_req_id = getattr(self, "_slot_to_req_id", None)
        if slot_to_req_id is not None:
            for slot, req_id in list(slot_to_req_id.items()):
                if req_id in finished_set:
                    slot_to_req_id.pop(slot, None)

        # Safety fallback: if any finished request was still carrying audio
        # generation state but wasn't keyed by its real req_id, clear the slot.
        if state is not None:
            for req_id, req_state in list(state.items()):
                if req_state.get("output_type") == "audio" and req_state.get("audio_out_ids") is not None:
                    state.pop(req_id, None)
                    logger.debug(
                        "[on_requests_finished] fallback reset audio state for req=%s",
                        req_id,
                    )

    def load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        """Load weights from checkpoint."""
        # Separate weights by component
        audio_tower_weights_dict: dict[str, torch.Tensor] = {}
        projector_weights = []
        mimo_layers_weights = []
        mimo_output_weights = []
        mimo_norm_weights = []
        model_weights = []

        total_weights = 0
        encoder_weight_count = 0
        for name, tensor in weights:
            total_weights += 1
            # Checkpoint uses different prefixes for different components
            # MIMO layers: "model.mimo_layers.X.*"
            # MIMO norm: "model.mimo_norm.*"
            # MIMO output: "mimo_output.*" (no model. prefix!)
            if name.startswith("model.mimo_layers."):
                # Strip "model.mimo_layers." prefix for our ModuleList structure
                # ModuleList expects keys like "0.self_attn.q_proj.weight"
                new_name = name.replace("model.mimo_layers.", "", 1)
                mimo_layers_weights.append((new_name, tensor))
            elif name.startswith("model.mimo_norm."):
                # Strip "model.mimo_norm." prefix
                new_name = name.replace("model.mimo_norm.", "", 1)
                mimo_norm_weights.append((new_name, tensor))
            elif name.startswith("mimo_output."):
                # Strip "mimo_output." prefix (no model. prefix in checkpoint!)
                new_name = name.replace("mimo_output.", "", 1)
                mimo_output_weights.append((new_name, tensor))
            elif name.startswith("model.encoder."):
                encoder_weight_count += 1
                # Whisper-Large-v3 encoder weights from the secondary source
                new_name = name.replace("model.encoder.", "", 1)
                # vLLM's WhisperEncoderLayer uses "mlp.fc1"/"mlp.fc2" names
                new_name = new_name.replace(".fc1.", ".mlp.fc1.").replace(".fc2.", ".mlp.fc2.")
                audio_tower_weights_dict[new_name] = tensor
            elif name.startswith("audio_tower."):
                # Already-mapped audio tower weights
                new_name = name.replace("audio_tower.", "", 1)
                new_name = new_name.replace(".fc1.", ".mlp.fc1.").replace(".fc2.", ".mlp.fc2.")
                audio_tower_weights_dict[new_name] = tensor
            elif name.startswith("model.vq_adaptor.layers.0."):
                projector_weights.append(
                    (
                        "vq_adaptor_layers_0." + name[len("model.vq_adaptor.layers.0.") :],
                        tensor,
                    )
                )
            elif name.startswith("model.vq_adaptor.layers.3."):
                projector_weights.append(
                    (
                        "vq_adaptor_layers_3." + name[len("model.vq_adaptor.layers.3.") :],
                        tensor,
                    )
                )
            elif name.startswith("model.vq_adaptor.layers.4."):
                projector_weights.append(
                    (
                        "vq_adaptor_layers_4." + name[len("model.vq_adaptor.layers.4.") :],
                        tensor,
                    )
                )
            elif name.startswith("multi_modal_projector."):
                # Already-mapped projector weights
                projector_weights.append((name.replace("multi_modal_projector.", "", 1), tensor))
            elif name.startswith("model.decoder."):
                # Whisper decoder weights are not used for ASR
                continue
            else:
                # Main model weights (Qwen2 backbone) - keep "model." prefix
                # Qwen2ForCausalLM expects parameter names like "model.layers.X.*"
                model_weights.append((name, tensor))

        logger.info(
            "[LOAD-WEIGHTS] total=%d encoder=%d projector=%d mimo=%d mimo_out=%d mimo_norm=%d model=%d",
            total_weights,
            encoder_weight_count,
            len(projector_weights),
            len(mimo_layers_weights),
            len(mimo_output_weights),
            len(mimo_norm_weights),
            len(model_weights),
        )

        # Load audio tower (Whisper encoder from whisper-large-v3 subfolder)
        if audio_tower_weights_dict:
            # The Whisper checkpoint only provides q_proj.bias for attention layers.
            # vLLM's fused qkv_proj.bias expects k/v bias shards as well, so zero-fill
            # any missing k_proj.bias/v_proj.bias entries before fusing.
            for name in list(audio_tower_weights_dict.keys()):
                if name.endswith(".self_attn.q_proj.bias"):
                    layer_prefix = name[: -len(".self_attn.q_proj.bias")]
                    k_name = f"{layer_prefix}.self_attn.k_proj.bias"
                    v_name = f"{layer_prefix}.self_attn.v_proj.bias"
                    if k_name not in audio_tower_weights_dict:
                        audio_tower_weights_dict[k_name] = torch.zeros_like(audio_tower_weights_dict[name])
                    if v_name not in audio_tower_weights_dict:
                        audio_tower_weights_dict[v_name] = torch.zeros_like(audio_tower_weights_dict[name])

            audio_tower_weights = list(audio_tower_weights_dict.items())
            logger.info("Loading %s audio tower weights", len(audio_tower_weights))
            self.audio_tower.load_weights(audio_tower_weights)
        else:
            logger.warning("NO audio tower weights found in checkpoint!")

        # Load projector (VQ-Adaptor from main checkpoint)
        if projector_weights:
            logger.debug("Loading %s projector weights", len(projector_weights))
            projector_state_dict = {k: v for k, v in projector_weights}
            missing, unexpected = self.multi_modal_projector.load_state_dict(projector_state_dict, strict=False)
            if missing:
                logger.warning("Missing projector weights: %s", missing)
            if unexpected:
                logger.warning("Unexpected projector weights: %s", unexpected)
            logger.debug("Projector loaded successfully")

        # Load MIMO layers through each parameter's TP-aware weight_loader.
        # These are Qwen2DecoderLayer modules (QKVParallelLinear /
        # MergedColumnParallelLinear / RowParallelLinear), so checkpoint
        # tensors must be sharded per rank — a plain load_state_dict of the
        # full tensors only works at TP=1.
        if mimo_layers_weights:
            logger.debug("Loading %s MIMO layer weights", len(mimo_layers_weights))
            stacked_params_mapping = [
                # (param_name, shard_name, shard_id)
                ("qkv_proj", "q_proj", "q"),
                ("qkv_proj", "k_proj", "k"),
                ("qkv_proj", "v_proj", "v"),
                ("gate_up_proj", "gate_proj", 0),
                ("gate_up_proj", "up_proj", 1),
            ]
            params_dict = dict(self.mimo_layers.named_parameters())
            mimo_loaded = 0
            for name, loaded_weight in mimo_layers_weights:
                for param_name, shard_name, shard_id in stacked_params_mapping:
                    if shard_name not in name:
                        continue
                    new_name = name.replace(shard_name, param_name)
                    param = params_dict.get(new_name)
                    if param is None:
                        logger.warning("MIMO weight %s has no parameter %s", name, new_name)
                        break
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight, shard_id)
                    mimo_loaded += 1
                    break
                else:
                    param = params_dict.get(name)
                    if param is None:
                        logger.warning("Unexpected MIMO weight: %s", name)
                        continue
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    mimo_loaded += 1
            logger.debug("MIMO layers loaded: %s weights", mimo_loaded)

        # Load audio output head through its TP-aware weight_loader
        # (ColumnParallelLinear shards the vocab dimension across ranks).
        if mimo_output_weights:
            logger.debug("Loading %s mimo_output weights", len(mimo_output_weights))
            mimo_output_params = dict(self.mimo_output.named_parameters())
            for name, tensor in mimo_output_weights:
                param = mimo_output_params.get(name)
                if param is None:
                    logger.warning("Unexpected mimo_output weight: %s", name)
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, tensor)
            logger.debug("mimo_output loaded successfully")
            # Verify weight shape
            logger.debug(
                "mimo_output.weight shape: %s, mean: %.6f, std: %.6f",
                self.mimo_output.weight.shape,
                self.mimo_output.weight.mean(),
                self.mimo_output.weight.std(),
            )

        if mimo_norm_weights:
            logger.debug("Loading %s mimo_norm weights", len(mimo_norm_weights))
            missing, unexpected = self.mimo_norm.load_state_dict({k: v for k, v in mimo_norm_weights}, strict=False)
            if missing:
                logger.warning("Missing mimo_norm weights: %s", missing)
            if unexpected:
                logger.warning("Unexpected mimo_norm weights: %s", unexpected)
            logger.debug("mimo_norm loaded successfully")

        # Load main model (Qwen2 backbone)
        # Pass weights as-is - parameters are named "model.layers.*"
        if model_weights:
            logger.debug("Loading %s main model weights", len(model_weights))
            # Show first few weight names for debugging
            for name, tensor in model_weights[:5]:
                logger.debug("Main model weight: %s shape=%s", name, tensor.shape)
            self.model.load_weights(model_weights)
            logger.debug("Main model weights loaded successfully")

            # Debug: Check if embeddings are loaded
            embed_weight = self.model.model.embed_tokens.weight
            lm_head_weight = self.model.lm_head.weight
            logger.debug(
                "Embeddings loaded: shape=%s, mean=%.6f, std=%.6f",
                embed_weight.shape,
                embed_weight.mean(),
                embed_weight.std(),
            )
            logger.debug(
                "LM head loaded: shape=%s, mean=%.6f, std=%.6f",
                lm_head_weight.shape,
                lm_head_weight.mean(),
                lm_head_weight.std(),
            )

            # Check a few specific weights to verify they're not random
            logger.debug("Embed weight sample [0,:5]: %s", embed_weight[0, :5].tolist())
            logger.debug("LM head weight sample [0,:5]: %s", lm_head_weight[0, :5].tolist())

            # Check layer 0 weights
            layer0_attn_q = self.model.model.layers[0].self_attn.qkv_proj.weight
            logger.debug(
                "Layer 0 attention qkv weight: shape=%s, mean=%.6f, std=%.6f",
                layer0_attn_q.shape,
                layer0_attn_q.mean(),
                layer0_attn_q.std(),
            )
