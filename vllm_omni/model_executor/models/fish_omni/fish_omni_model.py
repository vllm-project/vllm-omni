# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fish Audio Drama 3, stage 0: text (+ voice references) -> codec frames.

Runs on Model Runner V2. One forward samples one whole audio frame per request:

1. Every row's input frame ``[main, c0..c9]`` enters the backbone (vLLM's
   Qwen3.5 MoE: gated delta net + full attention, block-FP8 routed experts) as
   ``(E_lm[main] + E_cb[c1]) / sqrt(2)`` (semantic rows) or ``E_lm[main]``.
2. The backbone's final hidden state of each request's last row gives the
   logits of end-of-speech and semantic tokens 1..4095 only (the lm_head is
   sliced to those rows; see ``main_candidate_ids``), and a repetition-aware
   draw picks the main token (= code 0).
3. The 13 codebook-tower layers run over every row of the batch. Prompt rows
   are teacher-forced with the next prompt frame; a request's sampling row
   injects the codes drawn in this step, code k from the head of tower
   layer 4 + k.

All per-step inputs live in model-owned buffers at fixed addresses, which the
model state (``model_state.py``) fills before each step, so a decode step
replays as one FULL CUDA graph with no host work inside; steps with prompt
rows replay the backbone, tower and sampling as PIECEWISE graphs.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase, Qwen3_5Model
from vllm.model_executor.models.utils import maybe_prefix
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.fish_speech.dac_encoder import (
    _load_dac_codec,
    encode_reference_audio_codes,
    reference_audio_samples,
)
from vllm_omni.utils.speaker_cache import get_speaker_cache

from .codebook_tower import TowerLayer, TowerRotary
from .configuration_fish_omni import FishOmniConfig
from .prompt import Reference, build_prompt_frames, prompt_length
from .sampling import EMPTY_RAS_SLOT, RAS_TOP_K, RAS_WINDOW_FRAMES, FusedFrameSampler

logger = init_logger(__name__)

BACKBONE_PREFIX = "model.language_model."
# DAC: 44.1 kHz audio, 2048 samples per frame.
_CODEC_SAMPLE_RATE = 44100
_CODEC_HOP = 2048
_LAYER_RENAMES = (
    ("semantic_embedding.", "token_embedding."),
    ("codebook_embedding.", "token_embedding."),
    ("transformer_layer.attention.wqkv.", "qkv_proj."),
    ("transformer_layer.attention.wo.", "o_proj."),
    ("transformer_layer.attention.q_norm.", "q_norm."),
    ("transformer_layer.attention.k_norm.", "k_norm."),
    ("transformer_layer.attention_norm.", "input_layernorm."),
    ("transformer_layer.ffn_norm.", "post_attention_layernorm."),
    ("transformer_layer.feed_forward.w2.", "down_proj."),
)


def _first(value: Any) -> Any:
    return value[0] if isinstance(value, list) and value else value


def max_top_k(config: Any, codebook_size: int) -> int:
    """Candidates the fused draws keep: 64, or ``hf_overrides`` fish_omni_max_top_k for
    requests that ask for more. The draw kernel spans them with one ``tl.arange``, and
    every draw keeps at least the RAS re-draw's fixed top-k."""
    value = int(getattr(config, "fish_omni_max_top_k", 64))
    least = 1 << (RAS_TOP_K - 1).bit_length()
    if value < least or value & (value - 1) or value > codebook_size:
        raise ValueError(f"fish_omni_max_top_k={value}: needs a power of two in [{least}, {codebook_size}]")
    return value


def main_candidate_ids(eos_token_id: int, semantic_begin_id: int, semantic_end_id: int) -> list[int]:
    """Token ids the main token is drawn from: end of speech and semantic codes 1..end.

    Semantic code 0 is left out, which is the same as a -inf logit bias on it:
    it is not a speech code, but the checkpoint still samples it occasionally.
    """
    return sorted({eos_token_id, *range(semantic_begin_id + 1, semantic_end_id + 1)})


def _full_cudagraph() -> bool:
    if not is_forward_context_available():
        return False
    return get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL


class FishOmniBuffers:
    """Per-step inputs and outputs at fixed device addresses.

    Token rows (``max_num_tokens``): input frame, next-frame label, whether the
    row's next frame is sampled and by which sample row. Sample rows
    (``max_num_seqs``, one per request; a step always runs
    ``min(num_tokens, max_num_seqs)`` of them so that captured graphs and the
    model state agree): which token row they read, the request's state slot,
    sampling parameters and RAS window. Slot pools (``max_num_seqs + 1``; the
    last slot absorbs padding rows) keep each request's newest frame and the
    codes of every frame it generated (to resume it after a preemption).
    """

    def __init__(
        self,
        *,
        max_num_tokens: int,
        max_num_seqs: int,
        max_model_len: int,
        num_codebooks: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        long = dict(dtype=torch.int64, device=device)
        frame_width = 1 + num_codebooks
        # Backbone input embeddings. PIECEWISE graphs replay the backbone on
        # the addresses seen at capture, so the embeddings must live here.
        self.inputs_embeds = torch.zeros((max_num_tokens, hidden_size), dtype=dtype, device=device)
        self.max_num_tokens = max_num_tokens
        self.max_num_seqs = max_num_seqs
        self.dummy_slot = max_num_seqs
        # Token rows. A row reads its input frame from its request's slot
        # (decode) or from ``frames`` (prompt rows staged by the model state).
        self.frames = torch.zeros((max_num_tokens, frame_width), **long)
        self.labels = torch.zeros((max_num_tokens, frame_width), **long)
        self.token_slots = torch.full((max_num_tokens,), self.dummy_slot, **long)
        self.token_from_slot = torch.ones(max_num_tokens, dtype=torch.bool, device=device)
        self.sample_token_rows = torch.ones(max_num_tokens, dtype=torch.bool, device=device)
        self.token_sample_index = torch.zeros(max_num_tokens, **long)
        # Sample rows.
        self.sample_rows = torch.zeros(max_num_seqs, **long)
        self.sample_valid = torch.zeros(max_num_seqs, dtype=torch.bool, device=device)
        self.sample_slots = torch.full((max_num_seqs,), self.dummy_slot, **long)
        self.sampled_frames = torch.zeros((max_num_seqs, frame_width), **long)
        # Slot pools.
        self.slot_frames = torch.zeros((max_num_seqs + 1, frame_width), **long)
        self.slot_ras = torch.full((max_num_seqs + 1, RAS_WINDOW_FRAMES), EMPTY_RAS_SLOT, **long)
        self.slot_temperature = torch.ones(max_num_seqs + 1, dtype=torch.float32, device=device)
        self.slot_top_k = torch.ones(max_num_seqs + 1, **long)
        self.slot_top_p = torch.ones(max_num_seqs + 1, dtype=torch.float32, device=device)
        # Codes of each generated frame (int16: codes < 4096) and their count;
        # read back only when a request is preempted.
        self.slot_history = torch.zeros(
            (max_num_seqs + 1, max_model_len, num_codebooks), dtype=torch.int16, device=device
        )
        self.slot_generated = torch.zeros(max_num_seqs + 1, **long)
        # Sampling noise (set with the sampler), redrawn by the model state before
        # every step so no graph (compiled or CUDA) holds an RNG op.
        self.noise: torch.Tensor


def _reject_unsupported_fish_parallel(parallel_config: Any) -> None:
    """Drama 3 can shard the Qwen3.5 backbone (tensor parallel) and its routed
    experts (expert parallel) on NPU. CUDA tensor parallel, pipeline parallel,
    and prefill-context parallelism are not wired through the Omni V2 runner.
    """
    if parallel_config.pipeline_parallel_size != 1:
        raise NotImplementedError("fish_omni runs with one pipeline rank")
    if getattr(parallel_config, "prefill_context_parallel_size", 1) != 1:
        raise NotImplementedError("fish_omni does not support prefill context parallelism")
    # Packed GDN decode, the replicated tower, and the per-step noise broadcast
    # have been run at TP=2 only on NPU. CUDA stays single-rank until then.
    if parallel_config.tensor_parallel_size != 1 and current_platform.device_type != "npu":
        raise NotImplementedError("fish_omni tensor parallel is supported on NPU only")


# The tower and frame sampling compile with the backbone (its own decorator
# inlines), so mixed steps replay them in PIECEWISE graphs.
@support_torch_compile(
    dynamic_arg_dims={"input_ids": 0, "positions": -1, "intermediate_tensors": 0, "inputs_embeds": 0},
)
class FishOmniForConditionalGeneration(nn.Module, HasInnerState, IsHybrid):
    """Qwen3.5 MoE backbone + Drama 3 codebook tower on Model Runner V2."""

    have_multimodal_outputs = True
    # Every sampled main token is decided inside forward.
    mrv2_direct_tokens = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: FishOmniConfig = vllm_config.model_config.hf_config  # type: ignore[assignment]
        _reject_unsupported_fish_parallel(vllm_config.parallel_config)
        self.vllm_config = vllm_config
        self.config = config
        self.model_path = vllm_config.model_config.model
        text_config = config.text_config
        decoder_config = config.audio_decoder_config
        self.text_config = text_config
        self.decoder_config = decoder_config
        self.num_codebooks = int(decoder_config.num_codebooks)
        self.codebook_size = int(decoder_config.vocab_size)
        self.frame_width = 1 + self.num_codebooks
        self.semantic_begin_id = int(config.semantic_start_token_id)
        self.semantic_end_id = int(config.semantic_end_token_id)
        self.eos_token_id = int(config.eos_token_id)
        self.backbone_codebooks = int(decoder_config.vq_backbone_codebooks)
        self.embedding_scale = 1.0 / math.sqrt(1.0 + self.backbone_codebooks)
        hidden_size = text_config.hidden_size

        if current_platform.device_type == "npu":
            # Before the backbone is built, so construction and the memory
            # profile both miss the fused Triton kernel.
            from vllm_omni.platforms.npu.models.fish_omni import _install_npu_eager_qk_rope

            _install_npu_eager_qk_rope()
        self.model = Qwen3_5Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        if current_platform.device_type == "npu":
            from vllm_omni.platforms.npu.models.fish_omni import _require_npu_eager_qk_rope

            _require_npu_eager_qk_rope(self.model)
        # Triton tower / GDN kernels build on CUDA and ROCm; Ascend patches the Qwen class itself.
        if current_platform.is_cuda_alike():
            if vllm_config.speculative_config is None:
                # vLLM routes GDN decode through the fused-norm packed wrapper by
                # default (VLLM_GDN_DECODE_KERNEL=cuda). Without speculative decoding
                # it never reaches its MTP kernel and only adds copies and an
                # unfusable gated norm around the standard path (vllm-project/vllm#59735):
                # RTF 0.520 -> 0.512 at 128 streams with the standard path.
                for module in self.model.modules():
                    if getattr(module, "enable_fused_gdn_decode", False):
                        module.enable_fused_gdn_decode = False
            # Steps with prefills: packed decode kernel, FlashInfer on the state
            # pool in place, outputs written straight into the buffer.
            from .gdn import install_drama3_gdn

            install_drama3_gdn(self.model)

        # Only end-of-speech and the semantic band are ever sampled: keep those
        # lm_head rows, which cuts the per-frame head read from 1 GB to 16 MB.
        candidate_ids = main_candidate_ids(self.eos_token_id, self.semantic_begin_id, self.semantic_end_id)
        self.register_buffer("candidate_ids", torch.tensor(candidate_ids, dtype=torch.int64), persistent=False)
        self.lm_head = nn.Parameter(torch.empty(len(candidate_ids), hidden_size), requires_grad=False)

        self.codebook_embeddings = nn.Embedding(self.codebook_size * self.num_codebooks, hidden_size)
        self.register_buffer(
            "codebook_offsets",
            torch.arange(self.num_codebooks, dtype=torch.int64) * self.codebook_size,
            persistent=False,
        )
        max_position = vllm_config.model_config.max_model_len + 1
        self.tower_rotary = TowerRotary(decoder_config, max_position)
        self.num_shared_layers = int(decoder_config.shared_enrichment_layers)
        num_tower_layers = self.num_shared_layers + self.num_codebooks - 1
        # Every layer but the first (which also adds the input frame's other
        # codes) folds its embedding projection into a per-code table.
        self.tower_layers = nn.ModuleList(
            TowerLayer(
                decoder_config,
                self.tower_rotary,
                vllm_config.cache_config,
                prefix=maybe_prefix(prefix, f"audio_decoder.layers.{index}"),
                code_table=index > 0,
            )
            for index in range(num_tower_layers)
        )
        self.codebook_heads = nn.ModuleList(
            nn.Linear(hidden_size, self.codebook_size, bias=False) for _ in range(self.num_codebooks - 1)
        )

        scheduler_config = vllm_config.scheduler_config
        device = torch.device(torch.get_default_device())
        self.buffers = FishOmniBuffers(
            max_num_tokens=scheduler_config.max_num_batched_tokens,
            max_num_seqs=scheduler_config.max_num_seqs,
            max_model_len=vllm_config.model_config.max_model_len,
            num_codebooks=self.num_codebooks,
            hidden_size=hidden_size,
            dtype=vllm_config.model_config.dtype,
            device=device,
        )
        self.max_top_k = max_top_k(config, self.codebook_size)
        self.sampler = FusedFrameSampler(
            max_top_k=self.max_top_k,
            num_codebooks=self.num_codebooks,
            max_rows=scheduler_config.max_num_seqs,
            device=device,
        )
        self.buffers.noise = self.sampler.noise(scheduler_config.max_num_seqs, device)
        # Requests of the step, set by the model state (None: dummy runs).
        self.num_sample_rows: int | None = None
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
        self._speaker_cache = get_speaker_cache()
        self._tokenizer = None

    # ------------------------------------------------------------------
    # vLLM hybrid-model contract (delegated to the Qwen3.5 backbone).
    # ------------------------------------------------------------------

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        return Qwen3_5ForCausalLMBase.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        return Qwen3_5ForCausalLMBase.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        return Qwen3_5ForCausalLMBase.get_mamba_state_copy_func()

    def create_mrv2_model_state(self, vllm_config, encoder_cache, device):
        from .model_state import create_fish_omni_model_state

        return create_fish_omni_model_state(vllm_config, self, encoder_cache, device)

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        return self._tokenizer

    def _reference_codes(self, info: dict[str, Any], device: torch.device) -> torch.Tensor | None:
        """``[frames, num_codebooks]`` codes of the request's reference, encoding its audio if needed."""
        codes = _first(info.get("ref_codes"))
        if isinstance(codes, torch.Tensor):
            return codes.to(dtype=torch.long)
        wav = _first(info.get("ref_audio_wav"))
        if wav is None:
            return None
        voice_name = _first(info.get("voice_name"))
        cache_key = None
        if isinstance(voice_name, str) and voice_name:
            cache_key = self._speaker_cache.make_cache_key(
                voice_name, model_type="fish_omni", created_at=int(_first(info.get("voice_created_at")) or 0)
            )
            cached = self._speaker_cache.get(cache_key)
            if cached is not None:
                return cached["ref_codes_fq"]
        sample_rate = int(_first(info.get("ref_audio_sr")))
        wav_np = wav.cpu().numpy() if isinstance(wav, torch.Tensor) else np.asarray(wav, dtype=np.float32)
        codes = encode_reference_audio_codes(self.model_path, wav_np, sample_rate, device=device).cpu()
        if cache_key is not None:
            self._speaker_cache.put(cache_key, {"ref_codes_fq": codes})
        return codes

    def build_frames(self, info: dict[str, Any], device: torch.device) -> torch.Tensor:
        """CPU int64 prompt rows ``[prompt_len, 1 + num_codebooks]`` of a request."""
        text = _first(info.get("text"))
        if not isinstance(text, str) or not text.strip():
            raise ValueError("fish_omni needs a non-empty 'text'")
        references = []
        codes = self._reference_codes(info, device)
        if codes is not None:
            transcript = _first(info.get("ref_text"))
            if not isinstance(transcript, str) or not transcript.strip():
                raise ValueError("fish_omni voice cloning needs 'ref_text' with the reference audio")
            references.append(Reference(transcript, codes))
        return build_prompt_frames(
            self._get_tokenizer(),
            text,
            references,
            semantic_begin_id=self.semantic_begin_id,
            num_codebooks=self.num_codebooks,
        )

    @staticmethod
    def estimate_prompt_len_from_additional_information(
        additional_information: dict[str, Any] | None,
        *,
        tokenizer: Any = None,
        model_path: str | None = None,
        **kwargs: Any,
    ) -> int:
        """Exact prompt rows: text tokens plus one row per reference frame."""
        info = additional_information or {}
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        transcripts, frames = [], []
        codes = _first(info.get("ref_codes"))
        wav = _first(info.get("ref_audio_wav"))
        if isinstance(codes, torch.Tensor):
            frames.append(int(codes.shape[0]))
        elif wav is not None:
            rate = int(_first(info.get("ref_audio_sr")))
            samples = reference_audio_samples(np.shape(wav)) * _CODEC_SAMPLE_RATE / rate
            frames.append(math.ceil(math.ceil(samples) / _CODEC_HOP))
        if frames:
            transcripts.append(str(_first(info.get("ref_text")) or ""))
        return prompt_length(tokenizer, str(_first(info.get("text")) or ""), transcripts, frames)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def is_semantic(self, token_ids: torch.Tensor) -> torch.Tensor:
        return (token_ids >= self.semantic_begin_id) & (token_ids <= self.semantic_end_id)

    def codebook_sum(self, frames: torch.Tensor, first: int, end: int) -> torch.Tensor:
        """Summed embeddings of codes ``first..end - 1`` of every frame row."""
        rows = frames[:, 1 + first : 1 + end].clamp(0, self.codebook_size - 1) + self.codebook_offsets[first:end]
        return self.codebook_embeddings(rows).sum(dim=1)

    def backbone_embeddings(self, frames: torch.Tensor) -> torch.Tensor:
        """Write the backbone inputs of ``frames`` into the persistent embedding buffer."""
        text = self.model.embed_input_ids(frames[:, 0])
        fused = (text + self.codebook_sum(frames, 1, 1 + self.backbone_codebooks).to(text.dtype)) * self.embedding_scale
        semantic = self.is_semantic(frames[:, 0]).unsqueeze(-1)
        if torch.compiler.is_compiling():
            # Inside the compiled forward the embeddings are an intermediate.
            return torch.where(semantic, fused, text)
        out = self.buffers.inputs_embeds[: frames.shape[0]]
        return torch.where(semantic, fused, text, out=out)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        buffers = self.buffers
        num_tokens = positions.shape[0]
        # A FULL graph replays the sample rows of its capture, so they are a
        # function of the padded token count; the tower and sampling of other
        # steps run eagerly and take only the step's requests.
        if torch.compiler.is_compiling():
            # Compiled graphs replay per padded token count.
            num_samples = torch.sym_min(num_tokens, buffers.max_num_seqs)
        else:
            num_samples = min(num_tokens, buffers.max_num_seqs)
            if self.num_sample_rows is not None and not _full_cudagraph():
                num_samples = min(num_samples, self.num_sample_rows)
        frames = torch.where(
            buffers.token_from_slot[:num_tokens].unsqueeze(-1),
            buffers.slot_frames[buffers.token_slots[:num_tokens]],
            buffers.frames[:num_tokens],
        )
        labels = buffers.labels[:num_tokens]

        hidden_states = self.model(None, positions, intermediate_tensors, self.backbone_embeddings(frames))

        sample_rows = buffers.sample_rows[:num_samples]
        sample_valid = buffers.sample_valid[:num_samples]
        slots = buffers.sample_slots[:num_samples]
        temperature = buffers.slot_temperature[slots]
        top_k = buffers.slot_top_k[slots]
        top_p = buffers.slot_top_p[slots]
        ras_window = buffers.slot_ras[slots]

        sampler = self.sampler
        noise = buffers.noise[:num_samples]
        main_tokens, code_zero = sampler.main_tokens(
            hidden_states[sample_rows] @ self.lm_head.t(),
            self.candidate_ids,
            ras_window,
            temperature,
            top_k,
            top_p,
            noise,
            self.semantic_begin_id,
            self.codebook_size,
        )

        # Codes injected per token row: this step's draws on sampling rows
        # (gathered from the row's sample row), else the next prompt frame's.
        sample_token_rows = buffers.sample_token_rows[:num_tokens]
        sample_index = buffers.token_sample_index[:num_tokens]

        def inject(sampled: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
            return torch.where(sample_token_rows, sampled[sample_index], label)

        next_main = inject(main_tokens, labels[:, 0])
        code_is_semantic = self.is_semantic(next_main)
        next_code = inject(code_zero, labels[:, 1])

        shared_input = torch.where(
            self.is_semantic(frames[:, 0]).unsqueeze(-1),
            self.codebook_sum(frames, 1 + self.backbone_codebooks, self.num_codebooks).to(hidden_states.dtype),
            0.0,
        )
        hidden = hidden_states
        for index in range(self.num_shared_layers):
            hidden = self.tower_layers[index](
                hidden, next_code, code_is_semantic, shared_input if index == 0 else None, positions
            )
        codes = [code_zero]
        for k, head in enumerate(self.codebook_heads, start=1):
            layer = self.tower_layers[self.num_shared_layers + k - 1]
            hidden = layer(hidden, next_code, code_is_semantic, None, positions)
            sampled = sampler.code(head(hidden[sample_rows]), temperature, top_k, top_p, noise, k - 1)
            codes.append(sampled)
            if k < self.num_codebooks - 1:
                next_code = inject(sampled, labels[:, 1 + k])

        frames_out = torch.stack([main_tokens, *codes], dim=-1)
        # Non-semantic (end-of-speech) frames carry no codes.
        frames_out[:, 1:] *= self.is_semantic(main_tokens).unsqueeze(-1).to(frames_out.dtype)
        buffers.sampled_frames[:num_samples].copy_(frames_out)

        # Newest frame, RAS window and code history of every request whose
        # frame was sampled.
        keep = sample_valid.unsqueeze(-1)
        buffers.slot_frames[slots] = torch.where(keep, frames_out, buffers.slot_frames[slots])
        shifted = torch.cat((ras_window[:, 1:], main_tokens.unsqueeze(-1)), dim=-1)
        buffers.slot_ras[slots] = torch.where(keep, shifted, ras_window)
        generated = buffers.slot_generated[slots]
        index = generated.clamp(max=buffers.slot_history.shape[1] - 1)
        buffers.slot_history[slots, index] = torch.where(
            keep, frames_out[:, 1:].to(torch.int16), buffers.slot_history[slots, index]
        )
        buffers.slot_generated[slots] = generated + sample_valid.to(generated.dtype)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor, *_: Any, **__: Any) -> torch.Tensor:
        """Full-vocabulary logits; only vLLM's warmup and dummy sampler runs use them."""
        logits = hidden_states.new_full((hidden_states.shape[0], self.text_config.vocab_size), float("-inf"))
        logits[:, self.candidate_ids] = (hidden_states @ self.lm_head.t()).to(logits.dtype)
        return logits

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        intermediate = self.decoder_config.intermediate_size

        def backbone() -> Iterable[tuple[str, torch.Tensor]]:
            for name, tensor in weights:
                if name.startswith(BACKBONE_PREFIX):
                    yield name[len(BACKBONE_PREFIX) :], tensor
                elif name == "lm_head.weight":
                    self.lm_head.data.copy_(tensor.index_select(0, self.candidate_ids.to(tensor.device)))
                    loaded.add("lm_head")
                else:
                    target = self._tower_param_name(name)
                    if target is None:
                        raise ValueError(f"fish_omni: unexpected checkpoint key {name!r}")
                    param_name, shard = target
                    param = params[param_name]
                    if shard == "gate":
                        param.data[:intermediate].copy_(tensor)
                    elif shard == "up":
                        param.data[intermediate:].copy_(tensor)
                    else:
                        getattr(param, "weight_loader", default_weight_loader)(param, tensor)
                    loaded.add(param_name)

        for name in self.model.load_weights(backbone()):
            loaded.add(f"model.{name}")
        # vLLM's loader validates the backbone (fused experts report their own
        # names); the tower and the sliced head are checked here.
        missing = [name for name in params if not name.startswith("model.") and name not in loaded]
        if missing:
            raise ValueError(f"fish_omni: {len(missing)} parameters left unfilled, e.g. {missing[:8]}")
        for layer in self.tower_layers:
            layer.finalize_weights()
        # Load the reference-audio encoder now, so the first voice-clone request
        # does not pay for it.
        _load_dac_codec(self.model_path, device=self.lm_head.device, dtype=torch.float32)
        return loaded

    def _tower_param_name(self, name: str) -> tuple[str, str] | None:
        if name == "audio_decoder.codebook_embeddings.weight":
            return "codebook_embeddings.weight", "whole"
        if name.startswith("audio_decoder_mtp.heads."):
            index, rest = name[len("audio_decoder_mtp.heads.") :].split(".", 1)
            return f"codebook_heads.{index}.{rest}", "whole"
        for prefix, offset in (("audio_decoder_mtp_shared.layers.", 0), ("audio_decoder_mtp.layers.", None)):
            if not name.startswith(prefix):
                continue
            index, rest = name[len(prefix) :].split(".", 1)
            layer = int(index) + (self.num_shared_layers if offset is None else offset)
            if rest.startswith("transformer_layer.feed_forward.w1."):
                return f"tower_layers.{layer}.gate_up_proj.weight", "gate"
            if rest.startswith("transformer_layer.feed_forward.w3."):
                return f"tower_layers.{layer}.gate_up_proj.weight", "up"
            for old, new in _LAYER_RENAMES:
                if rest.startswith(old):
                    rest = new + rest[len(old) :]
                    break
            return f"tower_layers.{layer}.{rest}", "whole"
        return None


__all__ = ["FishOmniForConditionalGeneration"]
