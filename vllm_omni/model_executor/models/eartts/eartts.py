# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inference-only EarTTS model definition for vLLM-Omni.

The model architecture (RMSNorm, MLP, MLPLayer, GatedProjectedSumRMSNorm,
SubwordFlagEmbedding, BOSEOSEmbedding, CharAwareSubwordEncoder,
EarTTSInputEmbedding, MoGHead, MaskGITSampler, EarTTSModel) is preserved
from the original vLLM fork implementation but with all CFG / classifier-
free guidance code paths removed (vLLM-Omni does not support CFG).

The outer :class:`EarTTSForCausalLM` exposes the minimal vLLM-Omni
preprocess/postprocess hooks. Per-request inputs are passed via
``additional_information``:

  * ``reference_audio_tokens`` — Tensor of shape ``(Tref, 31)`` with the
    reference speaker's acoustic tokens. ``Tref`` is also the prefill
    placeholder length (the user passes ``prompt_token_ids = [0] * Tref``).
  * ``text`` — string to synthesize. The model tokenizes it once during
    the first prefill chunk and pads the result to ``round(4.0 * N)``
    text tokens (where ``N = len(tokenize(text))``) — one padded token
    is consumed per decode step.

Per-step flow:
  1. ``preprocess`` builds (or reads from cached buffers) the per-token
     tensors consumed by :class:`EarTTSInputEmbedding` —
     ``acoustic_tokens (BTx31)``, ``text_tokens (BT)``,
     ``text_mask (BT)``, ``bos_mask (BT)`` — and writes them into the
     model-owned static-address buffers at the request's flat-batch
     offset. Returns placeholder ``input_ids`` and a zero
     ``inputs_embeds`` (the actual embedding is computed inside the
     compiled ``forward`` so the text transformer encoder runs inside
     CUDA graphs).
  2. ``forward`` slices the buffers up to ``num_tokens`` and calls the
     compiled :class:`EarTTSModel` (embedding + Gemma3 backbone +
     MaskGIT sampler). The generated codes are copied into a stable
     ``_out_codes`` buffer for :meth:`make_omni_output`.
  3. ``compute_logits`` returns trivial 2-class logits (``[0, -inf]``)
     so vLLM's standard sampler always picks index ``0`` — the actual
     audio output is the codes tensor exposed via :meth:`make_omni_output`.
  4. ``postprocess`` stashes the last frame's codes as
     ``last_acoustic_codes`` for the next step's :meth:`preprocess` to
     consume as the decode input.
"""

import bisect
from collections.abc import Callable, Iterable
from typing import Any, Optional, Union

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoTokenizer
from transformers.generation.logits_process import (
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import (
    ignore_torch_compile,
    support_torch_compile,
)
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.model_executor.models.gemma3 import Gemma3Model
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .optimized_t5gemma import OptimizedT5GemmaEncoderModel

# Number of EarTTS frames per text token (one frame ≈ 80ms of audio; English
# speech at ~150wpm with ~1.3-1.5 BPE subwords/word lands around 3-4 frames
# per text token). 4.0 is a safe default that gives the model breathing room.
EARTTS_FRAMES_PER_TEXT_TOKEN = 4.0

# Hardcoded prefix string placed at the start of the prefill text positions
# (immediately followed by an EOS, then padding, then a final EOS).
# Matches the reference EarTTS prefill data layout.
_PREFILL_PREFIX_TEXT = "fisher"

# Pad token used to fill prefill text and decode text padding.
# Corresponds to ``<SPECIAL_12>`` in the EarTTS tokenizer vocab.
_TEXT_PAD_TOKEN_ID = 12


# ---------------------------------------------------------------------------
# Components ported verbatim from the original EarTTS implementation.
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # TODO: is casting really needed?
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class MLPLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.pre_norm = RMSNorm(hidden_size, eps=eps)
        self.mlp = MLP(hidden_size, intermediate_size)
        self.post_norm = RMSNorm(hidden_size, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pre_norm(x)
        y = self.mlp(y)
        y = self.post_norm(y)
        x = x + y
        return x


class GatedProjectedSumRMSNorm(nn.Module):
    def __init__(
        self,
        audio_dim,
        text_dim,
        hidden_dim,
        final_norm=True,
        num_codebooks=31,
        init_residual_scale=0.5,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks

        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        nn.init.normal_(self.audio_proj.weight, mean=0.0, std=0.015)
        nn.init.zeros_(self.audio_proj.bias)
        nn.init.normal_(self.text_proj.weight, mean=0.0, std=0.015)
        nn.init.zeros_(self.text_proj.bias)

        # FP32 gate params
        self.gate = nn.Parameter(
            torch.zeros(hidden_dim, dtype=torch.float32), requires_grad=False
        )
        self.residual_scale = nn.Parameter(
            torch.tensor(init_residual_scale, dtype=torch.float32),
            requires_grad=False,
        )

        self.final_norm = RMSNorm(hidden_dim) if final_norm else nn.Identity()

    def forward(self, audio_emb, text_emb):
        audio_emb = audio_emb / self.num_codebooks

        # projections run in model dtype (BF16)
        audio_h = self.audio_proj(audio_emb)
        text_h = self.text_proj(text_emb)

        dtype = audio_h.dtype

        gate = torch.sigmoid(self.gate)  # FP32
        res = torch.sigmoid(self.residual_scale)  # FP32

        h = gate.to(dtype) * audio_h + (1 - gate).to(dtype) * text_h
        h = res.to(dtype) * h
        h = self.final_norm(h.float()).to(dtype)

        return h


class SubwordFlagEmbedding(nn.Module):
    """
    Adds a small continuation embedding for subwords (tokens without
    word-boundary marker). Automatically adds a custom padding token at
    index ``vocab_size``. Ignores special tokens (starting with ``<``)
    when computing continuation flags.
    """

    def __init__(self, model_name: str, d_model: int):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = self.tokenizer.vocab_size
        self.d_model = d_model

        # Custom pad token at vocab_size
        self.pad_id = self.vocab_size
        # register pad_id as a tensor buffer to avoid device issues
        self.pad_tensor = nn.Parameter(
            torch.tensor(self.pad_id, dtype=torch.long), requires_grad=False
        )

        # Precompute continuation flags
        tokens = [
            self.tokenizer.convert_ids_to_tokens(i) for i in range(self.vocab_size)
        ]
        cont_flags = [
            1
            if not (tok.startswith("Ġ") or tok.startswith("▁") or tok.startswith("<"))
            else 0
            for tok in tokens
        ]
        cont_flags.append(0)  # for the custom pad token
        self.is_continuation = nn.Parameter(
            torch.tensor(cont_flags, dtype=torch.long), requires_grad=False
        )

        # Continuation embedding
        init_std = self.d_model ** -0.5
        self.cont_emb = nn.Embedding(2, self.d_model)
        nn.init.normal_(self.cont_emb.weight, mean=0.0, std=init_std)
        self.cont_emb.weight.data[0].zero_()

    def forward(self, subword_embeds: torch.Tensor, token_ids: torch.LongTensor):
        # Replace OOV token IDs with pad_id safely
        token_ids_clamped = torch.where(
            token_ids >= self.vocab_size, self.pad_tensor, token_ids
        )
        # Continuation flags
        cont_flags = self.is_continuation[token_ids_clamped]
        # Add continuation embedding
        cont_emb = self.cont_emb(cont_flags)
        return subword_embeds + cont_emb


class BOSEOSEmbedding(nn.Module):
    """
    Adds independent embeddings for BOS and EOS tokens using a single
    embedding table. Index 0 = regular token (ignored), 1 = BOS, 2 = EOS.
    Compatible with Hugging Face tokenizers that may or may not have
    BOS/EOS.
    """

    def __init__(self, model_name: str, d_model: int):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # vocab size that includes special tokens
        vocab_dict = self.tokenizer.get_vocab()
        self.vocab_size = max(vocab_dict.values())
        self.d_model = d_model

        # Custom pad token for OOVs
        self.pad_id = self.vocab_size
        self.pad_tensor = nn.Parameter(
            torch.tensor(self.pad_id, dtype=torch.long), requires_grad=False
        )

        # Identify BOS and EOS tokens (may be None)
        tokens = [
            self.tokenizer.convert_ids_to_tokens(i) for i in range(self.vocab_size)
        ]

        if "Qwen2.5" in model_name:
            # For Qwen, '<|im_start|>' is a common choice for a BOS token.
            # You can check your tokenizer's vocabulary for the best candidate.
            print(
                "Tokenizer does not have a `bos_token`. Setting it to "
                "'<|im_start|>'.",
                flush=True,
            )
            self.tokenizer.bos_token = "<|im_start|>"
            self.tokenizer.eos_token = "<|im_end|>"

        special_flags = []
        for tok in tokens:
            if (
                self.tokenizer.bos_token is not None
                and tok == self.tokenizer.bos_token
            ):
                special_flags.append(1)
            elif (
                self.tokenizer.eos_token is not None
                and tok == self.tokenizer.eos_token
            ):
                special_flags.append(2)
            else:
                special_flags.append(0)
        special_flags.append(0)  # for custom pad token
        self.special_flags = nn.Parameter(
            torch.tensor(special_flags, dtype=torch.long), requires_grad=False
        )
        # Embedding table: 0 = regular, 1 = BOS, 2 = EOS
        init_std = self.d_model ** -0.5
        self.special_emb = nn.Embedding(3, d_model)
        nn.init.normal_(self.special_emb.weight, mean=0.0, std=init_std)
        self.special_emb.weight.data[0].zero_()  # regular tokens ignored

    def forward(self, token_embeds: torch.Tensor, token_ids: torch.LongTensor):
        """
        token_embeds: (B, T, d_model)
        token_ids:    (B, T)
        """
        # Clamp OOVs to custom pad token
        safe_ids = torch.where(
            token_ids >= self.vocab_size, self.pad_tensor, token_ids
        )

        # Lookup flags (0=regular, 1=BOS, 2=EOS)
        flags = self.special_flags[safe_ids]
        return token_embeds + self.special_emb(flags)


class CharAwareSubwordEncoder(nn.Module):
    """
    An encoder that creates subword embeddings from character-level
    embeddings. This module replaces a standard subword embedding
    layer. It breaks down each subword into its constituent characters,
    embeds the characters, and then aggregates these character
    embeddings (e.g., via mean pooling) to form the final subword
    representation. This allows the model to handle rare or
    out-of-vocabulary subwords more gracefully.
    """

    def __init__(
        self,
        out_size: int,
        vocab_size: int,
        char_vocab_size: int,
        max_char_len: int,
        backbone_type: str,
        backbone_config: dict,
    ):
        super().__init__()
        self.max_char_len = max_char_len
        # 1. Initialize the backbone model for encoding characters
        config = AutoConfig.for_model(backbone_type, **backbone_config)
        self.backbone = OptimizedT5GemmaEncoderModel(config)
        self.backbone.eval()
        self.hidden_size = self.backbone.get_input_embeddings().weight.size(-1)
        delattr(self.backbone.encoder, "embed_tokens")
        # 2. Initialize embedding layer to embed characters
        self.embed_tokens = nn.Embedding(
            char_vocab_size + 1,
            self.hidden_size,
            padding_idx=char_vocab_size,
        )
        # 3. Initialize embedding layer to convert subword ids to char ids.
        # Also requires a layer which creates a mask for the backbone transformer
        self.embed_subwords = nn.Embedding(
            vocab_size,
            max_char_len,
        )
        self.embed_subwords_mask = nn.Embedding(
            vocab_size,
            max_char_len,
        )
        self.proj_embedding = nn.Linear(self.hidden_size, out_size, bias=False)

    def forward(self, subword_ids: torch.Tensor) -> torch.Tensor:
        char_ids = torch.round(self.embed_subwords(subword_ids)).to(
            torch.int32
        )  # BT x 128
        char_ids_mask = self.embed_subwords_mask(subword_ids)  # BT x 128
        char_embeds = self.embed_tokens(char_ids)  # bt x 128 x hidden_size

        char_hidden_states = self.backbone(
            inputs_embeds=char_embeds, attention_mask=char_ids_mask
        ).last_hidden_state  # BT x 128 x hidden_size
        # 3. Aggregate character embeddings to form subword embeddings (mean pooling)
        # We mask the padding characters before summing to get a correct mean.
        masked_sum = (char_hidden_states * char_ids_mask.unsqueeze(-1)).sum(
            dim=1
        )  # BT x hidden_size
        # Avoid division by zero for empty sequences
        char_ids_lengths = char_ids_mask.sum(dim=1)  # (bt,)
        mean_emb = masked_sum / (
            char_ids_lengths.unsqueeze(-1).clamp(min=1)
        )  # BT x hidden_size
        # 4. Scatter the aggregated embeddings back to the original subword sequence shape
        out_emb = self.proj_embedding(mean_emb)  # bt x hidden_size
        return out_emb


class EarTTSInputEmbedding(nn.Module):
    """Module that takes text tokens, audio tokens and prepares input
    embedding for EarTTS model.
    """

    def __init__(self, config):
        super().__init__()

        hidden_size = config.hidden_size
        vocab_size = config.emb_vocab_size
        char_vocab_size = config.emb_char_vocab_size
        max_char_len = config.max_char_len
        backbone_type = config.emb_backbone_type
        backbone_config = config.emb_backbone_config

        # allows to embed acoustic tokens into a single embeddings
        self.rvq_embs = nn.ModuleList(
            [
                nn.Embedding(config.codebook_size + 1, config.latent_size)
                for _ in range(config.num_quantizers)
            ]
        )
        self.embed_code = nn.Linear(config.latent_size, hidden_size, bias=False)
        self.embed_subword = CharAwareSubwordEncoder(
            out_size=hidden_size,
            vocab_size=vocab_size,
            char_vocab_size=char_vocab_size,
            max_char_len=max_char_len,
            backbone_type=backbone_type,
            backbone_config=backbone_config,
        )
        self.bos_emb = nn.Parameter(torch.empty(hidden_size))

        self.use_subword_flag_emb = config.use_subword_flag_emb
        pretrained_tokenizer_name = config.pretrained_tokenizer_name
        if self.use_subword_flag_emb:
            self.subword_flag_emb = SubwordFlagEmbedding(
                pretrained_tokenizer_name, hidden_size
            )
        self.use_bos_eos_emb = config.use_bos_eos_emb
        if self.use_bos_eos_emb:
            self.bos_eos_emb = BOSEOSEmbedding(
                pretrained_tokenizer_name, hidden_size
            )
        self.use_gated_fusion_for_text_audio = config.use_gated_fusion_for_text_audio
        if self.use_gated_fusion_for_text_audio:
            self.gated_fusion_audio_text = GatedProjectedSumRMSNorm(
                hidden_size, hidden_size, hidden_size, config.num_quantizers
            )

        self.use_audio_prompt_frozen_projection = (
            config.use_audio_prompt_frozen_projection
        )
        if self.use_audio_prompt_frozen_projection:
            self.audio_prompt_projection_W = nn.Parameter(
                torch.empty(hidden_size, hidden_size),
                requires_grad=False,
            )

    def forward(
        self,
        acoustic_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        bos_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Works for context and generation phases to prepare total input
        embeddings for EarTTS model.

        Inputs:
            acoustic_tokens: (BT x 31) - audio tokens
            text_tokens: (BT) - text token to embed
            text_mask: (BT) - masks text embeddings for prefill
            bos_mask: (BT) - specifies where BOS is applied (first frame of prefill)

        Returns:
            embedding of shape (BT x dim)
        """

        # prepare bos emb that is applied to audio embedding
        bos_emb = bos_mask.unsqueeze(1) * self.bos_emb  # BT x dim

        acoustic_tokens = acoustic_tokens.transpose(0, 1)  # 31 x BT
        audio_emb = sum(
            emb(acoustic_tokens[i]) for i, emb in enumerate(self.rvq_embs)
        )  # BT x latent_size
        audio_emb = self.embed_code(audio_emb)  # BT x hidden_size

        if self.use_audio_prompt_frozen_projection:
            # need to compute audio_prompt_lantent and use it instead of audio_emb
            audio_prompt_lantent = torch.nn.functional.linear(
                audio_emb, self.audio_prompt_projection_W.T
            )
            # WARNING! this is a hack! this only works if bos_mask is
            # [0, 0, ..., 0, 1] for prompt requests and [0] for decoding ones.
            # NeMo does pre_bos_mask = (bos_mask.cumsum(dim=1) == 0), but since
            # we have multiple bos_mask concatenated, we can't do that.
            # we just invert the bos_mask
            pre_bos_mask = (bos_mask == 0).unsqueeze(-1)  # BT x 1
            audio_emb = torch.where(pre_bos_mask, audio_prompt_lantent, audio_emb)

        audio_emb = audio_emb + bos_emb

        # embed text tokens by expanding them to chars and passing through transformer
        # apply the mask that turns this embedding to zeros for prefill tokens
        text_emb = self.embed_subword(text_tokens) * text_mask.unsqueeze(1)  # BT x dim
        # update text embeddings with flags
        if self.use_subword_flag_emb:
            text_emb = self.subword_flag_emb(text_emb, text_tokens)
        if self.use_bos_eos_emb:
            text_emb = self.bos_eos_emb(text_emb, text_tokens)

        # prepare total embedding by adding all components
        if self.use_gated_fusion_for_text_audio:
            total_emb = self.gated_fusion_audio_text(audio_emb, text_emb)
        else:
            total_emb = audio_emb + text_emb  # BT x dim
        return total_emb


def gumbel_like(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Generates a tensor of Gumbel noise with the same shape as the input
    tensor. Used for the Gumbel-Max trick.
    """
    u = torch.rand_like(tensor)
    return -torch.log(-torch.log(u + eps) + eps)


def batch_matmul(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Performs a batched matrix multiplication using PyTorch's native functions.
    In NeMo this is implemented as a custom kernel using triton.

    Args:
        x: ``[batch_size, d_in]``
        w: ``[num_weights, d_out, d_in]``
        y: ``[batch_size]``

    Returns:
        Tensor of shape ``[batch_size, d_out]``.
    """
    return torch.bmm(w[y], x.unsqueeze(2)).squeeze(2)


class MoGHead(nn.Module):
    """A Mixture of Gaussians (MoG) prediction head.

    This module takes a hidden state and predicts the parameters for a
    mixture of Gaussian distributions. It's suitable for modeling
    continuous, multi-modal data.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        out_size: int,
        num_layers: int,
        num_predictions: int,
        low_rank: Optional[int] = 64,
        top_p_or_k: Optional[Union[float, int]] = 1.0,
        min_log_std: float = -4.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.out_size = out_size
        self.low_rank = low_rank
        self.num_predictions = num_predictions
        self.min_log_std = min_log_std
        self.top_p_or_k = top_p_or_k

        self.logits_processor = (
            TopPLogitsWarper(self.top_p_or_k)
            if isinstance(self.top_p_or_k, float)
            else (
                TopKLogitsWarper(self.top_p_or_k)
                if isinstance(self.top_p_or_k, int)
                else None
            )
        )

        self.mlp_stack = nn.Sequential(
            *[
                MLPLayer(hidden_size, intermediate_size, eps=eps)
                for _ in range(num_layers)
            ],
            RMSNorm(hidden_size, eps=eps),
        )

        if low_rank is None:
            self.proj_logits = nn.Linear(hidden_size, num_predictions, bias=False)
            self.proj_mus = nn.Linear(
                hidden_size, num_predictions * out_size, bias=False
            )
            self.proj_logs = nn.Linear(hidden_size, 1, bias=False)
        else:
            assert low_rank < out_size
            self.proj_logits = nn.Linear(hidden_size, num_predictions, bias=False)
            self.proj_mus = nn.Linear(
                hidden_size, num_predictions * low_rank, bias=False
            )
            self.proj_logs = nn.Linear(hidden_size, 1, bias=False)
            self.proj_else = nn.Linear(hidden_size, out_size, bias=False)
            self.low_mat = nn.Parameter(
                torch.empty(num_predictions, out_size, low_rank)
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bt = x.size(0)
        n, d = self.num_predictions, self.low_rank or self.out_size

        x = self.mlp_stack(x)

        logits = self.proj_logits(x)

        # Apply top-p or top-k filtering to the mixture logits
        if self.logits_processor is not None:
            logits = self.logits_processor(None, logits.view(-1, n)).view_as(logits)

        # Sample a mixture component using the Gumbel-Max trick
        mixture_indices = (
            nn.functional.log_softmax(logits, dim=-1) + gumbel_like(logits)
        ).argmax(-1)

        # Select the mean corresponding to the sampled component
        mu = batch_matmul(
            x.view(bt, -1),
            self.proj_mus.weight.detach().view(n, d, -1),
            mixture_indices.view(bt),
        ).view(bt, d)
        if self.proj_mus.bias is not None:
            mu += self.proj_mus.bias.detach().view(n, d)[mixture_indices]

        if self.low_rank:
            mu = batch_matmul(
                mu.view(bt, -1),
                self.low_mat.detach().view(n, self.out_size, -1),
                mixture_indices.view(bt),
            ).view(bt, self.out_size)
            mu_res = self.proj_else(x)
        else:
            mu_res = torch.zeros((bt, d), device=x.device)

        logs = self.proj_logs(x).clamp_min(self.min_log_std)
        return mu * torch.exp(logs) + mu_res, logs


class MaskGITSampler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # easy access of cruicial config params
        self.num_quantizers = self.config.num_quantizers
        self.codebook_size = self.config.codebook_size
        self.noise_scale = self.config.noise_scale

        # pre-compute how many tokens are unmasked at each iteration
        rates = np.linspace(0.0, 1.0, self.config.num_iter + 1)[:-1].reshape(-1, 1)
        masking_rates = np.power(
            1 - np.power(rates, self.config.exponent), 1 / self.config.exponent
        )
        num_maskings = np.ceil(masking_rates * self.num_quantizers).astype(int)
        num_maskings_shifted = np.pad(
            num_maskings[1:], ((0, 1), (0, 0)), constant_values=0
        )
        sampling_per_step = num_maskings - num_maskings_shifted
        sampling_per_step_flat = sampling_per_step.flatten()
        # Drop any values at the beginning that are 0
        first_nonzero = np.argmax(sampling_per_step_flat != 0)
        self.num_to_sample = sampling_per_step_flat[first_nonzero:].tolist()

        # create layers used for acoustic tokens embedding
        self.rvq_embs = nn.Parameter(
            torch.empty(
                self.config.num_quantizers,
                self.config.codebook_size,
                self.config.latent_size,
            )
        )
        self.embed_code = nn.Linear(
            self.config.latent_size, self.config.hidden_size, bias=False
        )
        # MoG head for generation (uncompiled part)
        self.mog_head = MoGHead(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            out_size=self.config.latent_size,
            num_layers=self.config.mog_num_layers,
            num_predictions=self.config.mog_num_predictions,
            low_rank=self.config.mog_low_rank,
            top_p_or_k=self.config.top_p_or_k,
            min_log_std=self.config.mog_min_log_std,
            eps=self.config.mog_eps,
        )

    def _depthsum_embedding(self, code: torch.Tensor) -> torch.Tensor:
        """Embeds all codes into a single embedding."""
        embs = nn.functional.pad(
            self.rvq_embs, [0, 0, 0, 1]
        )  # num_quantizers x (codebook_size + 1) x latent_size
        res = nn.functional.embedding(code[0], embs[0])
        for i in range(1, len(embs)):
            res = res + nn.functional.embedding(code[i], embs[i])
        return res

    def _depthsum_encoding_step_reshaped(
        self,
        r: torch.Tensor,  # [B*T, hidden_size]
        code: torch.Tensor,  # [num_quantizers, B*T]
        depth_str: int,
        k: int,
    ) -> torch.Tensor:
        """RVQ encoding with reshaped code tensor."""
        for i in range(depth_str, depth_str + k):
            # Compute distances: ||emb||² - 2⟨r, emb⟩
            idx_sel = (
                self.rvq_embs[i].pow(2).sum(-1)  # [vocab_size]
                - 2 * (r @ self.rvq_embs[i].T)  # [B*T, vocab_size]
            ).argmin(-1)  # [B*T]

            # Update residual
            emb_i = nn.functional.embedding(
                idx_sel,
                self.rvq_embs[i],
            )  # [B*T, latent_size]
            r = r - emb_i

            # Store selected indices
            code[i] = idx_sel

        return code

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Performs the iterative unmasking process for a single
        generation step.
        """

        device = hidden_states.device
        # Initialize the full code tensor
        code = (
            torch.zeros(
                (self.num_quantizers, hidden_states.shape[0]),
                dtype=torch.long,
                device=device,
            )
            + self.codebook_size
        )
        # Iteratively unmask the continuous part of the code
        cnt = 0
        for k in self.num_to_sample:
            # Prepare input for the MoG head
            mog_input_embeds = self.embed_code(
                self._depthsum_embedding(code)
            )  # (BT x hidden_size)
            mog_input_embeds += hidden_states

            mog_mu, mog_logs = self.mog_head(
                mog_input_embeds,
            )
            z = (
                mog_mu
                + torch.exp(mog_logs) * torch.randn_like(mog_mu) * self.noise_scale
            )
            code = self._depthsum_encoding_step_reshaped(z, code, cnt, k)

            cnt += k
        return code.transpose(0, 1)  # BT x num_quantizers


@support_torch_compile
class EarTTSModel(nn.Module):
    """Embedding preparation + Gemma3 backbone (compiled together).

    The MaskGIT sampler used to live inside this module's compiled
    forward, but it is now hosted in :class:`EarTTSSamplerModel` so that
    the (expensive) iterative MaskGIT sampling can be skipped on prefill
    positions while still being CUDA-graph captured for decode-only
    batches. See :meth:`EarTTSForCausalLM.forward` for the orchestration.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.total_emb = EarTTSInputEmbedding(vllm_config.model_config.hf_config)
        self.backbone = Gemma3Model(vllm_config=vllm_config, prefix=prefix)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        acoustic_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        bos_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through embeddings and backbone transformer.
        Returns the backbone's ``hidden_states``.
        """
        total_emb = self.total_emb(
            acoustic_tokens=acoustic_tokens,
            text_tokens=text_tokens,
            text_mask=text_mask,
            bos_mask=bos_mask,
        )
        hidden_states = self.backbone(
            input_ids, positions, intermediate_tensors, inputs_embeds=total_emb
        )
        return hidden_states


@support_torch_compile
class EarTTSSamplerModel(nn.Module):
    """MaskGIT sampler in its own compile group.

    Hosting the sampler in a separate ``@support_torch_compile`` module
    is what makes it possible for :meth:`EarTTSForCausalLM.forward` to:

    * Capture and replay a CUDA-graph for decode-only batches (where
      every position needs sampling).
    * Skip the sampler entirely on prefill positions, where the audio
      output isn't actually needed.
    * Run the sampler on a sliced subset of positions in mixed
      prefill+decode batches, with a ``BatchDescriptor`` override so the
      sampler's CUDA-graph cache is hit at the padded decode-batch size.

    The :meth:`forward` operates on a stable-address scratch buffer
    (:attr:`_sampler_input`) so callers can pass a transient slice
    (e.g. ``hidden_states[decode_idx]``) without breaking CUDA-graph
    replay. The non-compiled :meth:`sample` wrapper does that copy and
    then invokes the compiled :meth:`forward`.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.sampler = MaskGITSampler(config)

        # Stable-address scratch buffer for the sampler's input. Every
        # CUDA-graph replay must read from the same ``data_ptr()``; the
        # caller may pass either the full backbone output or a fresh
        # ``hidden_states[decode_idx]`` slice, so we copy into this
        # buffer (in :meth:`sample`) before invoking :meth:`forward`.
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        hidden_size = config.hidden_size
        dtype = vllm_config.model_config.dtype
        self._sampler_input = torch.zeros(
            max_num_tokens, hidden_size, dtype=dtype
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compiled — runs MaskGIT on a (stable-address) hidden buffer."""
        return self.sampler(hidden_states)

    def sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Non-compiled wrapper — copies into the stable buffer first.

        Mirrors the qwen3-tts code-predictor pattern: transient inputs
        are first written into a model-owned static-address buffer so
        the captured CUDA-graph for the compiled :meth:`forward` always
        reads from the recorded ``data_ptr()``.
        """
        seq_len = int(hidden_states.shape[0])
        buf = self._sampler_input[:seq_len]
        buf.copy_(hidden_states)
        return self(buf)


# ---------------------------------------------------------------------------
# Outer model — refactored for vLLM-Omni preprocess/postprocess hooks.
# ---------------------------------------------------------------------------


# Placeholder token id used to fill the per-step ``input_ids`` returned
# by :meth:`preprocess`. Must be a valid id in ``[0, config.vocab_size)``
# but is otherwise unused — the actual decode-vs-prefill behaviour is
# driven by the per-token buffers populated in :meth:`preprocess`.
#
# The width of the dummy logits tensor returned by
# :meth:`compute_logits` is taken from ``config.vocab_size`` (see
# :class:`EarTTSConfig`) so vLLM's sampler / ``LogitsProcessor`` and the
# model agree on the logits shape. ``compute_logits`` returns
# ``[0, -inf, ..., -inf]`` so the sampler's argmax always picks index 0
# regardless of how wide ``vocab_size`` is — the real audio output is
# the codes tensor exposed via :meth:`make_omni_output`.
_DUMMY_TOKEN_ID = 0


@ignore_torch_compile
@support_torch_compile
class EarTTSForCausalLM(nn.Module):
    """EarTTS for vLLM-Omni.

    Inputs (passed via ``additional_information``):

      * ``reference_audio_tokens`` — Tensor of shape ``(Tref, 31)`` with
        the reference speaker's acoustic tokens. The user must also pass
        ``prompt_token_ids = [0] * Tref`` so the prefill placeholder
        length matches.
      * ``text`` — string to synthesize. The model tokenizes it once
        during the first prefill chunk and pads the result to
        ``round(EARTTS_FRAMES_PER_TEXT_TOKEN * N)`` tokens.

    Per-step flow (see module docstring for details):

    ``preprocess`` populates four model-owned buffers
    (:attr:`_acoustic_tokens`, :attr:`_text_tokens`, :attr:`_text_mask`,
    :attr:`_bos_mask`) at each request's flat-batch offset. ``forward``
    slices them up to ``num_tokens`` and runs the compiled
    :class:`EarTTSModel` (text transformer + Gemma3 backbone) for
    every position, then conditionally invokes the compiled
    :class:`EarTTSSamplerModel` (MaskGIT) to produce codes. The sampler
    is skipped on prefill positions — see :meth:`forward` for details.
    The generated codes (BTx31) are written to :attr:`_out_codes` and
    exposed under the conventional ``"audio_codes"`` multimodal key by
    :meth:`make_omni_output`. ``postprocess`` stashes the final-frame
    codes under ``last_acoustic_codes`` for the next decode step's
    :meth:`preprocess`.

    Sampler skipping mirrors the qwen3-tts code-predictor pattern:

    * **Profile / dummy run** (``attn_metadata is None``) and
      **decode-only batches** (``max_query_len == 1``) run the sampler
      on every token so the captured CUDA graph covers all of
      ``cudagraph_capture_sizes``.
    * **Mixed prefill+decode batches**: only decode-token positions go
      through the sampler. The sampler's ``BatchDescriptor`` is
      overridden to the padded decode-batch size so the right captured
      graph is replayed.
    * **Prefill-only batches**: the sampler is skipped entirely.
    * For prefill positions, :attr:`_out_codes` is initialized from
      :attr:`_acoustic_tokens` (i.e. the input reference audio frames),
      so :meth:`postprocess` on a request whose last position falls in
      prefill returns the last reference audio frame as
      ``last_acoustic_codes`` — the natural acoustic input for the
      first decode step that follows.
    """

    # Map raw HuggingFace checkpoint names to the vLLM module layout.
    # Only ``model.sampler.*`` needed remapping when the MaskGIT sampler
    # was lifted out of :class:`EarTTSModel` into its own compile group
    # (:attr:`sampler_module`). All other prefixes (``model.total_emb.``,
    # ``model.backbone.``) match the new layout 1:1.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.sampler.": "sampler_module.sampler.",
        }
    )

    # Omni preprocess/postprocess hooks (consumed by the gpu model runner).
    has_preprocess = True
    has_postprocess = True
    have_multimodal_outputs = True

    # Hardcoded prefix string placed at the start of the prefill text
    # positions (immediately followed by an EOS, padding, then a final
    # EOS). Matches the reference EarTTS prefill data layout.
    PREFILL_PREFIX_TEXT: str = _PREFILL_PREFIX_TEXT

    # Pad token id used to fill prefill text positions and to pad the
    # tokenized synthesis text up to the per-step decode length.
    TEXT_PAD_TOKEN_ID: int = _TEXT_PAD_TOKEN_ID

    # Keys whose tensors should stay on GPU in ``model_intermediate_buffer``
    # to avoid a D2H/H2D round-trip on every step. These are consumed only
    # in eager ``preprocess``/``postprocess`` Python (where they're copied
    # into the static-address ``_acoustic_tokens`` / ``_text_tokens`` /
    # ``_text_mask`` / ``_bos_mask`` buffers used by the compiled forward),
    # so address stability across steps is not required.
    gpu_resident_buffer_keys: set[str] = {
        "last_acoustic_codes",
        "ear_prefill_text_tokens",
        "ear_prefill_text_mask",
        "ear_prefill_acoustic_tokens",
        "ear_prefill_bos_mask",
        "ear_decode_text_tokens",
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        # Embedding + Gemma3 backbone — runs on every position. Built
        # under the default ``"backbone"`` model tag (vLLM's compile
        # cache key for the main model). We don't wrap this in a
        # ``set_model_tag`` block because :func:`set_model_tag` asserts
        # the new tag differs from the current one and the default is
        # already ``"backbone"``.
        self.model = EarTTSModel(
            vllm_config=vllm_config,
            prefix=prefix,
        )

        # MaskGIT sampler in its own compile group, so it can be invoked
        # conditionally (decode positions only, or skipped entirely on
        # prefill-only batches) while still being CUDA-graph captured
        # for decode-only batches over ``cudagraph_capture_sizes``. The
        # ``"sampler"`` tag keys the sampler's compile cache separately
        # from the backbone's.
        with set_model_tag("sampler"):
            self.sampler_module = EarTTSSamplerModel(
                vllm_config=vllm_config,
                prefix=prefix,
            )

        # Pad ids used by buffers / preprocess. Match the conventions of
        # the original EarTTSInputEmbedding: an acoustic token id of
        # ``codebook_size`` is the trailing "no audio" pad row in
        # ``rvq_embs`` (which has ``codebook_size + 1`` entries).
        self._num_quantizers: int = int(self.config.num_quantizers)
        self._acoustic_pad_id: int = int(self.config.codebook_size)
        self._text_pad_id: int = int(self.TEXT_PAD_TOKEN_ID)

        # HF tokenizer loaded from the model directory. The ids it
        # produces must match the vocab expected by
        # :class:`SubwordFlagEmbedding` / :class:`BOSEOSEmbedding`.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._eos_token_id: int = int(
            self.tokenizer.eos_token_id
            if self.tokenizer.eos_token_id is not None
            else 2
        )

        # ── Persistent stable-address buffers ────────────────────────
        # Plain tensor attributes (not nn.Parameter / not register_buffer):
        #   * AutoWeightsLoader only walks named_parameters() and persistent
        #     registered buffers, so plain attributes are invisible to it
        #     (no spurious "missing weight" errors during load_weights).
        #   * vLLM constructs models inside
        #     ``with torch.device(device_config.device):`` so a bare
        #     ``torch.zeros(...)`` here is allocated directly on the GPU.
        #   * Addresses stay stable across CUDA graph replays as long as
        #     we never re-assign these names (only do in-place writes via
        #     copy_/fill_/indexed assignment), which is what the rest of
        #     this class does. The piecewise CUDAGraphWrapper records
        #     data_ptr() at capture time and expects the same pointer at
        #     replay time — that holds with plain tensors.
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens

        self._acoustic_tokens = torch.full(
            (max_num_tokens, self._num_quantizers),
            self._acoustic_pad_id,
            dtype=torch.long,
        )
        self._text_tokens = torch.zeros(max_num_tokens, dtype=torch.long)
        self._text_mask = torch.zeros(max_num_tokens, dtype=torch.long)
        self._bos_mask = torch.zeros(max_num_tokens, dtype=torch.long)
        self._out_codes = torch.zeros(
            max_num_tokens, self._num_quantizers, dtype=torch.long
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compatibility shim — not actually consumed at runtime since
        every forward goes through ``inputs_embeds`` assembled inside
        :meth:`forward`.
        """
        return self.model.backbone.embed_input_ids(input_ids)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.get_input_embeddings(input_ids)

    @staticmethod
    def _first_str(value: Any) -> str:
        """Return the first element of a list-wrapped scalar, or the
        scalar itself.
        """
        if isinstance(value, list):
            return str(value[0]) if value else ""
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def _unwrap_singleton(value: Any) -> Any:
        """Unwrap a possibly list-wrapped scalar (e.g. ``[tensor]``)."""
        if isinstance(value, list):
            return value[0] if value else None
        return value

    @staticmethod
    def _coerce_ref_audio_tokens(value: Any) -> Optional[torch.Tensor]:
        """Normalize ``reference_audio_tokens`` to a 2D LongTensor on CPU.

        Accepts ``torch.Tensor``, ``np.ndarray``, ``list``-wrapped
        variants of either. Returns ``None`` when the input cannot be
        interpreted as acoustic tokens.
        """
        x = EarTTSForCausalLM._unwrap_singleton(value)
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            t = x
        elif isinstance(x, np.ndarray):
            t = torch.from_numpy(x)
        elif isinstance(x, list):
            try:
                t = torch.as_tensor(x)
            except Exception:
                return None
        else:
            return None
        if t.numel() == 0:
            return None
        if t.ndim == 3:
            t = t[0]
        return t.to(dtype=torch.long).contiguous()

    def _zero_buffers_at(self, start: int, length: int) -> None:
        """Reset the per-token buffers in [start, start+length)."""
        s, e = int(start), int(start + length)
        self._acoustic_tokens[s:e].fill_(self._acoustic_pad_id)
        self._text_tokens[s:e].fill_(self._text_pad_id)
        self._text_mask[s:e].zero_()
        self._bos_mask[s:e].zero_()

    def _build_prefill_tensors(
        self,
        text: str,
        ref_audio_tokens: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct the full prefill per-token tensors and the cached
        padded text tokens for decode.

        Layout (length ``num = ref_audio_tokens.shape[0]``):

          * ``acoustic`` = ``ref_audio_tokens`` (Tref, 31), copied as-is
          * ``text_tokens``: ``tokenize(PREFILL_PREFIX_TEXT) + [EOS] +
            [PAD] * missing + [EOS]`` (length ``num``). Truncated from
            the front if the prefix doesn't fit; this should never happen
            in practice since ``num`` is the reference audio length and
            comfortably exceeds the few-token prefix.
          * ``text_mask`` = ``[0] * num``; ``mask[-2:] = 1`` so only the
            last two positions contribute text embeddings.
          * ``bos_mask`` = ``[0] * num``; ``bos_mask[-1] = 1`` (BOS
            transition at the final prefill frame).

        Decode-side cache (returned separately):

          * ``decode_text_tokens`` = ``tokenize(text) + [PAD] * pad_n``
            with total length ``round(EARTTS_FRAMES_PER_TEXT_TOKEN * N)``
            where ``N = len(tokenize(text))``. One entry is consumed per
            decode step.
        """
        num = int(ref_audio_tokens.shape[0])
        if num <= 0:
            raise ValueError(
                "reference_audio_tokens must have at least one frame "
                f"(got shape={tuple(ref_audio_tokens.shape)})."
            )

        prefix_ids: list[int] = list(
            self.tokenizer.encode(
                self.PREFILL_PREFIX_TEXT, add_special_tokens=False
            )
        )
        prefix_ids.append(self._eos_token_id)

        # Reserve one slot for the trailing EOS; pad the rest. Truncate
        # the prefix from the front if num is unrealistically small.
        usable = max(0, num - 1)
        if len(prefix_ids) > usable:
            prefix_ids = prefix_ids[-usable:] if usable > 0 else []
        missing = max(0, usable - len(prefix_ids))
        prefill_text_ids = (
            prefix_ids
            + [self._text_pad_id] * missing
            + [self._eos_token_id]
        )
        full_text_tokens = torch.tensor(
            prefill_text_ids[:num], dtype=torch.long, device=device
        )

        full_text_mask = torch.zeros(num, dtype=torch.long, device=device)
        # Only the last two positions carry text contribution.
        full_text_mask[-min(2, num):] = 1

        full_bos_mask = torch.zeros(num, dtype=torch.long, device=device)
        full_bos_mask[-1] = 1

        full_acoustic = ref_audio_tokens.to(
            device=device, dtype=torch.long, non_blocking=True
        )
        if full_acoustic.shape != (num, self._num_quantizers):
            raise ValueError(
                "reference_audio_tokens must have shape "
                f"(Tref, {self._num_quantizers}); got "
                f"{tuple(full_acoustic.shape)}."
            )

        # Cached padded text tokens consumed one-per-step in decode.
        text_ids = list(
            self.tokenizer.encode(text, add_special_tokens=True)
        )
        n = len(text_ids)
        if n <= 0:
            raise ValueError("text tokenized to an empty sequence.")
        total_frames = max(
            n, int(round(n * EARTTS_FRAMES_PER_TEXT_TOKEN))
        )
        pad_n = max(0, total_frames - n)
        decode_text_tokens = torch.tensor(
            text_ids + [self._text_pad_id] * pad_n,
            dtype=torch.long,
            device=device,
        )

        return (
            full_text_tokens,
            full_text_mask,
            full_acoustic,
            full_bos_mask,
            decode_text_tokens,
        )

    # ------------------------------------------------------------------
    # preprocess
    # ------------------------------------------------------------------

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: Optional[torch.Tensor],
        *,
        start: int = 0,
        end: int = 0,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Build per-request ``(input_ids, inputs_embeds)`` for this step.

        Prefill (``span_len > 1``):
            On the first prefill chunk, tokenizes the synthesis text
            (cached for decode) and constructs the per-position prefill
            tensors of length ``num = reference_audio_tokens.shape[0]``:

            * ``acoustic_tokens`` = ``reference_audio_tokens`` (copy)
            * ``text_tokens``     = ``tokenize(PREFILL_PREFIX_TEXT) +
              [EOS] + [PAD] * missing + [EOS]`` (length ``num``)
            * ``text_mask``       = ``[0] * num``; ``mask[-2:] = 1``
            * ``bos_mask``        = ``[0] * num``; ``mask[-1] = 1``

            The full per-position tensors are cached on GPU under
            ``ear_prefill_*`` keys (kept device-resident via
            :attr:`gpu_resident_buffer_keys`) and a running
            ``ear_prefill_offset`` tracks how much of the prefill has
            already been copied into the static-address buffers across
            multi-chunk prefill. The tokenized + padded synthesis text
            is cached under ``ear_decode_text_tokens`` for the decode
            phase.

        Decode (``span_len == 1``):
            Reads ``last_acoustic_codes`` (stashed by
            :meth:`postprocess` after the previous step) into
            :attr:`_acoustic_tokens` at the request's offset, picks the
            next padded text token from ``ear_decode_text_tokens`` at
            ``ear_decode_offset``, and sets ``text_mask = 1`` /
            ``bos_mask = 0``.
        """
        # Normalize: some runner paths still pass per-request state
        # nested under ``additional_information`` instead of flattened.
        nested = info_dict.get("additional_information")
        if isinstance(nested, dict):
            merged = {
                k: v for k, v in info_dict.items() if k != "additional_information"
            }
            for k, v in nested.items():
                merged.setdefault(k, v)
            info_dict = merged

        device = input_ids.device
        span_len = int(input_ids.shape[0])
        if span_len <= 0:
            base = (
                input_embeds
                if input_embeds is not None
                else self.embed_input_ids(input_ids)
            )
            return input_ids, base, {}

        # ----- Prefill ----------------------------------------------
        if span_len > 1:
            cached_text_tokens = info_dict.get("ear_prefill_text_tokens")
            cached_text_mask = info_dict.get("ear_prefill_text_mask")
            cached_acoustic = info_dict.get("ear_prefill_acoustic_tokens")
            cached_bos_mask = info_dict.get("ear_prefill_bos_mask")
            cached_decode_text = info_dict.get("ear_decode_text_tokens")
            is_first = not (
                isinstance(cached_text_tokens, torch.Tensor)
                and isinstance(cached_text_mask, torch.Tensor)
                and isinstance(cached_acoustic, torch.Tensor)
                and isinstance(cached_bos_mask, torch.Tensor)
                and isinstance(cached_decode_text, torch.Tensor)
            )

            info_update: dict[str, Any] = {}
            if is_first:
                # First prefill chunk: build the full per-position
                # tensors + decode text cache once. Allocate them on
                # GPU so the runner's ``model_intermediate_buffer``
                # keeps them device-resident across multi-chunk prefill
                # and decode (avoids per-chunk / per-step H2D uploads).
                text = self._first_str(info_dict.get("text"))
                if not text:
                    raise ValueError(
                        "EarTTS preprocess requires non-empty "
                        "additional_information['text'] for prefill."
                    )
                ref_audio_tokens = self._coerce_ref_audio_tokens(
                    info_dict.get("reference_audio_tokens")
                )
                if ref_audio_tokens is None:
                    raise ValueError(
                        "EarTTS preprocess requires "
                        "additional_information['reference_audio_tokens'] "
                        "as a (Tref, num_quantizers) tensor for prefill."
                    )

                (
                    cached_text_tokens,
                    cached_text_mask,
                    cached_acoustic,
                    cached_bos_mask,
                    cached_decode_text,
                ) = self._build_prefill_tensors(
                    text=text,
                    ref_audio_tokens=ref_audio_tokens,
                    device=device,
                )

                info_update["ear_prefill_text_tokens"] = cached_text_tokens
                info_update["ear_prefill_text_mask"] = cached_text_mask
                info_update["ear_prefill_acoustic_tokens"] = cached_acoustic
                info_update["ear_prefill_bos_mask"] = cached_bos_mask
                info_update["ear_decode_text_tokens"] = cached_decode_text
                info_update["ear_prefill_offset"] = 0
                info_update["ear_decode_offset"] = 0

            offset = int(info_dict.get("ear_prefill_offset", 0) or 0)
            full_len = int(cached_text_tokens.shape[0])
            s = max(0, min(offset, full_len))
            e = max(0, min(offset + span_len, full_len))
            take_len = e - s

            # Slice the chunk out of the GPU-resident cached prefill
            # tensors and D2D-copy into the static-address buffers at
            # this request's offset. Pad with the trailing layout
            # (text_mask=0, bos_mask=0, acoustic=pad) when the scheduled
            # chunk overshoots — this shouldn't happen if the scheduler
            # placeholder length matches the true prefill length.
            buf_s = int(start)
            buf_e = buf_s + span_len

            self._zero_buffers_at(buf_s, span_len)

            if take_len > 0:
                self._text_tokens[buf_s:buf_e].copy_(cached_text_tokens[s:e])
                self._text_mask[buf_s:buf_e].copy_(cached_text_mask[s:e])
                self._acoustic_tokens[buf_s:buf_e].copy_(cached_acoustic[s:e])
                self._bos_mask[buf_s:buf_e].copy_(cached_bos_mask[s:e])

            info_update["ear_prefill_offset"] = offset + span_len

            # Token ids for the prefill span — the runner uses them only
            # for vLLM bookkeeping; the actual decode-vs-prefill
            # behaviour is driven by the buffers above. Using a constant
            # in-vocab placeholder keeps everything simple.
            input_ids_out = torch.full_like(input_ids, _DUMMY_TOKEN_ID)
            inputs_embeds_out = torch.zeros(
                (span_len, self.config.hidden_size),
                device=device,
                dtype=self.vllm_config.model_config.dtype,
            )
            return input_ids_out, inputs_embeds_out, info_update

        # ----- Decode (span_len == 1) -------------------------------
        # Acoustic input = previous-step codes (stashed by postprocess).
        # Text input = next padded text token from the decode cache.
        # ``text_mask = 1`` and ``bos_mask = 0`` for every decode step.
        last_codes = info_dict.get("last_acoustic_codes")
        cached_decode_text = info_dict.get("ear_decode_text_tokens")
        decode_offset = int(info_dict.get("ear_decode_offset", 0) or 0)

        buf_s = int(start)
        self._zero_buffers_at(buf_s, 1)

        if isinstance(last_codes, torch.Tensor) and last_codes.numel() > 0:
            ac = (
                last_codes.to(device=device, dtype=torch.long)
                .reshape(-1)[: self._num_quantizers]
            )
            self._acoustic_tokens[buf_s, : ac.shape[0]].copy_(ac)

        info_update = {}
        if isinstance(cached_decode_text, torch.Tensor) and cached_decode_text.numel() > 0:
            total = int(cached_decode_text.shape[0])
            idx = decode_offset if decode_offset < total else total - 1
            self._text_tokens[buf_s].copy_(
                cached_decode_text[idx].to(device=device, dtype=torch.long)
            )
            self._text_mask[buf_s] = 1
            info_update["ear_decode_offset"] = decode_offset + 1

        inputs_embeds_out = torch.zeros(
            (1, self.config.hidden_size),
            device=device,
            dtype=self.vllm_config.model_config.dtype,
        )
        return input_ids, inputs_embeds_out, info_update

    # ------------------------------------------------------------------
    # forward — runs the compiled embedding + backbone, then the sampler
    # only on decode positions (skipping the expensive MaskGIT loop on
    # prefill positions).
    # ------------------------------------------------------------------

    def _get_decode_idxs(self):
        """Return ``(decode_token_indices, num_requests)`` for sampler dispatch.

        Mirrors the qwen3-tts code-predictor pattern:

        * ``(None, 0)`` → run sampler on every token. Used during
          profile / dummy runs (no ``attn_metadata``) and decode-only
          batches (``max_query_len == 1``), so the captured CUDA graph
          covers all of ``cudagraph_capture_sizes``.
        * ``(decode_token_indices, num_requests)`` → run sampler only on
          the listed positions. ``decode_token_indices`` is padded up to
          the next captured CUDA-graph size (so the sampler's graph
          cache is hit) and ``num_requests`` is the unpadded count of
          real decode tokens (used to scatter codes back into the right
          rows of :attr:`_out_codes`).
        """
        ctx = get_forward_context()
        attn_metadata = ctx.attn_metadata
        if attn_metadata is None:
            # Profile / dummy run. Apply sampler everywhere so capture
            # covers every cudagraph_capture_sizes value.
            return None, 0

        if isinstance(attn_metadata, dict):
            any_layer_meta = next(iter(attn_metadata.values()))
        else:
            any_layer_meta = attn_metadata

        if any_layer_meta.max_query_len == 1:
            # Decode-only batch: every position is a decode position,
            # so just run the sampler over the whole flat batch.
            return None, 0

        start_loc = any_layer_meta.query_start_loc
        tokens_per_req = start_loc[1:] - start_loc[:-1]
        is_decode = (tokens_per_req == 1)
        decode_token_indices = start_loc[:-1][is_decode]

        num_requests = decode_token_indices.shape[0]
        padded_num_requests = num_requests
        if (
            self.vllm_config.compilation_config.cudagraph_mode
            != CUDAGraphMode.NONE
        ):
            sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
            idx = bisect.bisect_left(sizes, num_requests)
            if idx < len(sizes):
                padded_num_requests = sizes[idx]
        if padded_num_requests != num_requests:
            decode_token_indices = torch.nn.functional.pad(
                decode_token_indices,
                (0, padded_num_requests - num_requests),
            )
        return decode_token_indices, num_requests

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> torch.Tensor:
        """Forward pass.

        1. Slice the per-token input buffers populated by
           :meth:`preprocess` up to ``num_tokens`` and call the compiled
           :class:`EarTTSModel` (embedding + backbone) on every position.
        2. Initialize :attr:`_out_codes` from :attr:`_acoustic_tokens`
           so prefill positions carry the input reference-audio frames
           (decode positions are overwritten in step 3 below). This way
           :meth:`postprocess` on a request whose last position falls in
           prefill returns the last reference audio frame as
           ``last_acoustic_codes`` — the natural acoustic input for the
           first decode step that follows.
        3. Conditionally invoke the compiled :class:`EarTTSSamplerModel`:

           * **No ``attn_metadata`` (dummy / profile run)** or
             **decode-only batch**: run sampler on every position; the
             compiled forward replays the captured CUDA graph for the
             matching ``cudagraph_capture_sizes`` entry.
           * **Mixed prefill+decode batch**: gather decode positions
             into a contiguous tensor, override the sampler's
             ``BatchDescriptor`` to the padded decode-batch size so the
             right captured graph is replayed, and scatter the produced
             codes back into the corresponding rows of
             :attr:`_out_codes`.
           * **Prefill-only batch**: skip the sampler entirely.

        ``inputs_embeds`` is ignored: ``preprocess`` returns zeros for
        it and the actual embedding is built inside the compiled
        :class:`EarTTSInputEmbedding` (this keeps the text transformer
        encoder inside the CUDA graph).
        """
        num_tokens = int(input_ids.shape[0])

        acoustic_tokens = self._acoustic_tokens[:num_tokens]
        text_tokens = self._text_tokens[:num_tokens]
        text_mask = self._text_mask[:num_tokens]
        bos_mask = self._bos_mask[:num_tokens]

        # Step 1: embedding + backbone on every position (compiled).
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            acoustic_tokens=acoustic_tokens,
            text_tokens=text_tokens,
            text_mask=text_mask,
            bos_mask=bos_mask,
        )

        # Step 2: prefill fallback. Initialize _out_codes from the input
        # acoustic tokens so prefill rows always contain something
        # sensible (the reference-audio frame at that position). Decode
        # rows are overwritten by the sampler in step 3 below.
        self._out_codes[:num_tokens].copy_(acoustic_tokens)

        # Step 3: conditionally run the (separately compiled) sampler.
        decode_idx, num_req = self._get_decode_idxs()
        if decode_idx is None:
            # Dummy / profile run, or decode-only batch: sample every
            # position. The captured CUDA-graph for the sampler is
            # selected by num_tokens, which is in cudagraph_capture_sizes.
            codes = self.sampler_module.sample(hidden_states)
            self._out_codes[:num_tokens].copy_(codes.to(dtype=torch.long))
        elif num_req > 0:
            # Mixed batch: gather decode positions and run sampler only
            # on those. Override the BatchDescriptor so the sampler's
            # CUDA-graph cache is hit at the padded decode-batch size
            # (set by _get_decode_idxs to the next cudagraph_capture
            # bucket above num_req).
            ctx = get_forward_context()
            orig_batch_descriptor = ctx.batch_descriptor
            ctx.batch_descriptor = BatchDescriptor(
                num_tokens=decode_idx.shape[0],
            )
            decode_hidden = hidden_states[decode_idx]
            codes = self.sampler_module.sample(decode_hidden)
            ctx.batch_descriptor = orig_batch_descriptor

            valid_dec_idx = decode_idx[:num_req]
            self._out_codes[valid_dec_idx] = codes[:num_req].to(
                dtype=torch.long
            )
        # else: prefill-only batch — sampler is skipped entirely.

        return hidden_states

    # ------------------------------------------------------------------
    # compute_logits — sampler bypass (the real output is ``codes``)
    # ------------------------------------------------------------------

    def compute_logits(
        self,
        hidden_states: Union[torch.Tensor, OmniOutput],
        sampling_metadata: Any = None,
    ) -> Optional[torch.Tensor]:
        """Return zero logits of width ``config.vocab_size``.

        ``config.vocab_size`` is what vLLM's sampler / ``LogitsProcessor``
        use to size their working buffers, so deriving the width from
        the same field guarantees the two agree. The sampled token id
        is irrelevant: ``input_ids`` are never consumed by the model
        (the per-step decode behaviour is driven by the buffers
        populated in :meth:`preprocess`), and the real audio output is
        the codes tensor exposed via :meth:`make_omni_output`.
        """
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        batch_size = hidden_states.shape[0]
        return hidden_states.new_zeros(batch_size, int(self.config.vocab_size))

    # ------------------------------------------------------------------
    # multimodal output plumbing
    # ------------------------------------------------------------------

    def make_omni_output(
        self,
        model_outputs: Union[torch.Tensor, OmniOutput],
        **_: Any,
    ) -> OmniOutput:
        """Wrap backbone hidden states with the codes generated by the
        sampler (BTx31) under the conventional ``"audio_codes"`` key.
        """
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        hidden = model_outputs
        num_tokens = int(hidden.shape[0])
        audio_codes = self._out_codes[:num_tokens].clone()
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={"audio_codes": audio_codes},
        )

    # ------------------------------------------------------------------
    # postprocess — stash last-frame codes for the next decode step
    # ------------------------------------------------------------------

    def postprocess(
        self,
        hidden_states: torch.Tensor,
        multimodal_outputs: Optional[dict[str, Any]] = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Pull the last-frame codes out of the multimodal output (or,
        as a fallback, out of :attr:`_out_codes` using the slice's
        storage offset) and stash them under ``last_acoustic_codes`` so
        the next step's :meth:`preprocess` can use them as the decode
        input.
        """
        if hidden_states.numel() == 0:
            return {}

        audio_codes = (multimodal_outputs or {}).get("audio_codes")
        if isinstance(audio_codes, torch.Tensor) and audio_codes.numel() > 0:
            # ``hidden_states`` is a slice of the flat batch. Recover
            # the request's last position via storage_offset and pick
            # the corresponding row from ``audio_codes``.
            stride0 = hidden_states.stride(0) or 1
            req_start = hidden_states.storage_offset() // stride0
            last = req_start + hidden_states.shape[0] - 1
            last_codes = audio_codes[last : last + 1].detach()
            return {"last_acoustic_codes": last_codes}

        return {}

    # ------------------------------------------------------------------
    # weight loading
    # ------------------------------------------------------------------

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        # ``null_emb`` is part of the upstream EarTTS checkpoint but is
        # only used for CFG (classifier-free guidance), which vLLM-Omni
        # does not support. Skip it explicitly so weight loading is not
        # tripped up by the now-removed parameter.
        skip_prefixes: list[str] = ["model.total_emb.null_emb"]
        if self.config.tie_word_embeddings:
            skip_prefixes.append("lm_head.")

        # The Gemma3 backbone keeps a vestigial ``embed_tokens`` layer,
        # but this model never consumes ``input_ids`` (every forward
        # goes through ``inputs_embeds`` assembled from the audio / text
        # buffers in :meth:`preprocess`). We expose a 2-class placeholder
        # ``vocab_size`` purely so the vLLM sampler's working buffers
        # match the dummy logits returned by :meth:`compute_logits`.
        # The checkpoint, however, ships ``embed_tokens.weight`` at the
        # original tokenizer vocab size — which trips
        # ``VocabParallelEmbedding``'s
        # ``loaded_weight.shape[output_dim] == self.org_vocab_size``
        # assertion. Truncate (or pad) the loaded weight to
        # ``(config.vocab_size, hidden_size)`` so the assertion passes;
        # the surviving rows are never consumed at runtime.
        target_vocab = int(self.config.vocab_size)
        embed_weight_name = "model.backbone.embed_tokens.weight"

        def _adjusted_weights() -> Iterable[tuple[str, torch.Tensor]]:
            for name, w in weights:
                if name == embed_weight_name and w.dim() >= 1 and w.shape[0] != target_vocab:
                    if w.shape[0] >= target_vocab:
                        yield name, w[:target_vocab].contiguous()
                    else:
                        pad = torch.zeros(
                            target_vocab - w.shape[0],
                            *w.shape[1:],
                            dtype=w.dtype,
                            device=w.device,
                        )
                        yield name, torch.cat([w, pad], dim=0).contiguous()
                else:
                    yield name, w

        # ``hf_to_vllm_mapper`` rewrites ``model.sampler.*`` to
        # ``sampler_module.sampler.*`` so the upstream EarTTS checkpoint
        # (which still places the MaskGIT sampler under ``model.``) lands
        # on the dedicated :attr:`sampler_module` compile group.
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(
            _adjusted_weights(), mapper=self.hf_to_vllm_mapper
        )

    # ------------------------------------------------------------------
    # Generation length estimation
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_generation_len(
        additional_information: Optional[dict[str, Any]] = None,
        *,
        tokenize_prompt: Callable[[str], list[int]],
        frames_per_text_token: float = EARTTS_FRAMES_PER_TEXT_TOKEN,
    ) -> int:
        """Compute the number of decode steps required to synthesize
        ``additional_information['text']``.

        The synthesis text is tokenized (with special tokens, matching
        :meth:`preprocess`) and padded to ``round(frames_per_text_token *
        N)`` tokens, where ``N`` is the tokenized length. Each decode
        step consumes one padded text token, so the returned value is
        also the total number of decode steps (``max_tokens``).

        Mirrors the qwen3-tts pattern of accepting a ``tokenize_prompt``
        callback so callers (e.g. ``serving_speech.py``) can plug in
        their own pre-loaded tokenizer without paying for a second copy.
        """
        info: dict[str, Any] = additional_information or {}
        text_value = info.get("text")
        if isinstance(text_value, list):
            text = text_value[0] if text_value else ""
        else:
            text = text_value or ""
        if not isinstance(text, str) or not text:
            return 1

        text_ids = tokenize_prompt(text)
        n = len(list(text_ids))
        if n <= 0:
            return 1
        return max(n, int(round(n * float(frames_per_text_token))))
