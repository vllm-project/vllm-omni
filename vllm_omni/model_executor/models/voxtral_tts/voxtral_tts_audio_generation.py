# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from functools import cached_property
from math import ceil
from typing import Any, Union, get_args, get_origin  # Literal, cast,

import numpy as np
import regex as re
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from apex.normalization import FusedRMSNorm

    rms_norm = FusedRMSNorm
except ImportError:
    from torch.nn import RMSNorm as RMSNorm

    rms_norm = RMSNorm

from mistral_common.protocol.instruct.chunk import AudioChunk, RawAudio
from mistral_common.tokens.tokenizers.audio import Audio, AudioEncoder
from transformers import BatchFeature
from transformers.tokenization_utils_base import TextInput
from vllm.config import VllmConfig
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.utils import (
    flatten_bn,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    NestedTensors,
)
from vllm.multimodal.parse import AudioProcessorItems, MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import BaseDummyInputsBuilder, BaseMultiModalProcessor
from vllm.multimodal.processing.processor import (
    BaseProcessingInfo,
    MultiModalProcessingInfo,
    ProcessorInputs,
    PromptReplacement,
    PromptUpdate,
    TimingContext,
)
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.tokenizers.mistral import MistralTokenizer

weight_norm = torch.nn.utils.parametrizations.weight_norm


logger = init_logger(__name__)

SUPPORTED_LANGS = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "ar": "Arabic",
    "hi": "Hindi",
    "it": "Italian",
    "pt": "Portuguese",
}


# Audio
class AudioSpecialTokens(str, Enum):
    """Special tokens predicted by audio codebook heads.

    These tokens are inserted by `audio_tokens_with_pattern`. They are not part of the text vocabulary.
    We offset the output audio tokens from quantizer by `len(all_special_tokens)` to avoid conflicts with text tokens.
    """

    empty_audio = "[EMPTY_AUDIO]"
    end_audio = "[END_AUDIO]"

    @staticmethod
    def all_special_tokens() -> list["AudioSpecialTokens"]:
        return [token for token in AudioSpecialTokens]

    @staticmethod
    def id(token: "AudioSpecialTokens") -> int:
        return AudioSpecialTokens.all_special_tokens().index(token)


@dataclass
class AcousticTransformerArgs:
    input_dim: int
    # Define some defaults
    dim: int = 768
    n_layers: int = 3
    head_dim: int = 128
    hidden_dim: int = 2048
    n_heads: int = 6
    n_kv_heads: int = 2
    use_biases: bool = False
    norm_eps: float = 1e-5
    sigma: float = 1e-5  # was 0.01 in beta version
    n_decoding_steps: int | None = None  # Number of Euler ODE steps for flow matching


@dataclass
class MultimodalAudioModelArgs:
    # comma-separated list of codebook sizes
    # The first token in a codebook should always be reserved to indicate
    # absence. The codebook size should be inclusive of this.
    semantic_codebook_size: int
    acoustic_codebook_size: int
    n_acoustic_codebook: int
    acoustic_transformer_args: AcousticTransformerArgs

    @property
    def codebook_sizes(self) -> list[int]:
        return [
            self.semantic_codebook_size,
            *[self.acoustic_codebook_size for _ in range(self.n_acoustic_codebook)],
        ]

    def get_codebook_sizes(
        self, pad_to_multiple: int | None = 128, include_special_tokens: bool = True
    ) -> list[int]:  # O
        # TODO(@alexhliu): add one more argument to specify if we are computing the size of
        # a single codebook merging all codebooks
        # These codebook sizes should be exactly same as output of the audio quantizer.
        # We will then add buffer for special tokens and pad to nearest multiple of 8 (for GPU efficiency)

        def _round_up_to_multiple_of_number(n: int, multiple: int) -> int:
            return multiple * ((n + multiple - 1) // multiple)

        result_codebook_sizes = []
        for i, cb_size in enumerate(self.codebook_sizes):
            # TODO(@alexhliu): special tokens resides only in first codebook
            # so ideally we shouldn't need to add them to latter codebooks
            if include_special_tokens:
                cb_size += len(AudioSpecialTokens.all_special_tokens())
            if pad_to_multiple is not None:
                cb_size = _round_up_to_multiple_of_number(cb_size, pad_to_multiple)
            result_codebook_sizes.append(cb_size)
        return result_codebook_sizes


def _repeat_interleave(t: torch.Tensor, repeats: int) -> torch.Tensor:
    """
    Faster than
    keys = torch.repeat_interleave(
            keys, repeats, dim=2, output_size=xq.size(2)
        )
    on gpu ?
    """
    return t.unsqueeze(3).expand([-1, -1, -1, repeats, -1]).flatten(2, 3)


def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int) -> tuple[torch.Tensor, torch.Tensor]:
    if repeats > 1:
        keys = _repeat_interleave(keys, repeats=repeats)
        values = _repeat_interleave(values, repeats=repeats)
    return keys, values


def from_nested_dict(cls, d):
    """Recursively instantiate dataclasses from nested dicts."""
    if not is_dataclass(cls):
        return d

    kwargs = {}
    for f in fields(cls):
        value = d.get(f.name, getattr(cls, f.name, None))
        field_type = f.type

        # Unwrap Optional / Union types
        origin = get_origin(field_type)
        if origin is Union:
            args = get_args(field_type)
            # take first non-None type
            non_none_types = [a for a in args if a is not type(None)]
            if len(non_none_types) == 1:
                field_type = non_none_types[0]

        # Recurse if nested dataclass
        if is_dataclass(field_type) and isinstance(value, dict):
            value = from_nested_dict(field_type, value)

        kwargs[f.name] = value

    return cls(**kwargs)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        use_biases: bool,
    ) -> None:
        super().__init__()

        self.w1 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.w2 = nn.Linear(
            hidden_dim,
            dim,
            bias=use_biases,
        )
        self.w3 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class BidirectionalAttention(nn.Module):
    # TODO(@alexhliu): see if we can merge this with codec attention
    """Attention layer (without any RoPE embeddings)."""

    def __init__(
        self,
        args: AcousticTransformerArgs,
        layer_id: int,
    ) -> None:
        super().__init__()
        self.args = args

        self.n_local_heads: int = args.n_heads
        self.n_local_kv_heads: int = args.n_kv_heads
        self.repeats = self.n_local_heads
        self.layer_id = layer_id

        self.head_dim = args.head_dim

        # Note difference in how we set biases:
        # - Mistral LM: only accepts bias=args.use_biases for wo,
        #               everything else is False
        # - Whisper: wk always has bias=False, and bias=True for wq, wv and wo
        self.wq = nn.Linear(
            args.dim,
            args.n_heads * args.head_dim,
            bias=args.use_biases,
        )
        self.wk = nn.Linear(
            args.dim,
            args.n_kv_heads * args.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            args.dim,
            args.n_kv_heads * args.head_dim,
            bias=args.use_biases,
        )
        self.wo = nn.Linear(
            (args.n_heads * args.head_dim),
            args.dim,
            bias=args.use_biases,
        )

        self.softmax_scale: float = self.args.head_dim**-0.5
        self.repeats = self.n_local_heads // self.n_local_kv_heads

    def _native_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.tensor:
        scale = 1.0 / query.shape[-1] ** 0.5
        query = query * scale
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attn = query @ key.transpose(-2, -1)
        attn = attn.softmax(-1)
        attn = attn @ value
        return attn.transpose(1, 2).contiguous()

    def _forward_attention(  # type: ignore
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        key, value = repeat_kv(key, value, repeats=self.repeats)
        bsz, seqlen, _, _ = query.shape

        output = self._native_attention(query, key, value)

        return output.view(bsz, seqlen, -1)

    def forward(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if x.dim() == 2:
            bsz, (seqlen, _) = 1, x.shape
        else:
            bsz, seqlen, _ = x.shape

        # compute xq
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        output = self._forward_attention(query=xq, key=xk, value=xv, **kwargs)
        output = output.view(bsz, seqlen, self.n_local_heads * self.head_dim)
        return self.wo(output).squeeze(0)


class AcousticTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        args: AcousticTransformerArgs,
    ) -> None:
        super().__init__()
        self._layer_id = layer_id
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = BidirectionalAttention(
            args,
            layer_id=layer_id,
        )

        self.feed_forward = FeedForward(
            args.dim,
            args.hidden_dim,
            args.use_biases,
        )

        self.attention_norm = rms_norm(
            args.dim,
            eps=args.norm_eps,
        )
        self.ffn_norm = rms_norm(
            args.dim,
            eps=args.norm_eps,
        )
        self.args = args

    @property
    def layer_id(self) -> int:
        return self._layer_id

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x))
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out


### Flow Matching Acoustic Transformer ###


class TimeEmbedding(nn.Module):
    """Sinusoidal Embedding for encoding time"""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = torch.exp(-math.log(theta) * torch.arange(dim // 2).float() / (dim // 2))
        # NOTE(@alexhliu): mistral codebase requires persistent = True for save/load
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = torch.einsum(
            "bi, j -> bj",
            t,
            self.inv_freq,
        )
        return torch.cat((emb.cos(), emb.sin()), dim=-1)


class FlowMatchingAudioTransformer(nn.Module):
    def __init__(
        self,
        audio_model_args: dict,
    ) -> None:
        super().__init__()
        if "codebook_sizes" in audio_model_args:
            codebook_sizes = [int(c) for c in audio_model_args.pop("codebook_sizes").split(",")]
            audio_model_args.update(
                {
                    "semantic_codebook_size": codebook_sizes[0],
                    "acoustic_codebook_size": codebook_sizes[1],
                    "n_acoustic_codebook": len(codebook_sizes) - 1,
                }
            )
        self.model_args: MultimodalAudioModelArgs = from_nested_dict(MultimodalAudioModelArgs, audio_model_args)
        # MultimodalAudioModelArgs
        assert isinstance(self.model_args, MultimodalAudioModelArgs)
        args = self.model_args.acoustic_transformer_args
        # AcousticTransformerArgs
        self.acoustic_transformer_args = args
        assert isinstance(self.acoustic_transformer_args, AcousticTransformerArgs)

        # currently assuming always 1 semantic codebook + N acoustic codebooks
        self.num_non_acoustic_embeddings = 1
        self.num_acoustic_codebooks = len(self.model_args.get_codebook_sizes()) - self.num_non_acoustic_embeddings

        # flow matching utils
        self.sigma = args.sigma

        # codebook sizes
        acoustic_codebook_sizes = self.model_args.get_codebook_sizes(
            pad_to_multiple=None, include_special_tokens=False
        )[1:]
        assert len(set(acoustic_codebook_sizes)) == 1, "only 1 size for acoustic codebooks supported"
        self.acoustic_embeddings_levels = acoustic_codebook_sizes[0]
        self.acoustic_embeddings_dim = len(acoustic_codebook_sizes)

        self._init_audio_embeddings_layer()
        self._init_output_layer()
        self._init_layers()

        self._end_audio_token_id = AudioSpecialTokens.id(AudioSpecialTokens.end_audio)
        self._empty_audio_token_id = AudioSpecialTokens.id(AudioSpecialTokens.empty_audio)

        # Flow matching constants
        self._n_steps = args.n_decoding_steps
        # TODO(chenyo): hardcoded, need to fix
        self._cfg_alpha = 1.2
        self._noise_scale = 1.0
        self.register_buffer(
            "_timesteps",
            torch.linspace(0, 1, self._n_steps + 1),
            persistent=False,
        )

    def load_weight(self, weight: tuple[str, torch.Tensor]) -> str:
        params_dict = dict(self.named_parameters())
        name, loaded_weight = weight
        if name not in params_dict:
            logger.warning(f"{name} not found in FlowMatchingAudioTransformer (UNUSED)")
            return name
        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)
        return name

    def _init_audio_embeddings_layer(self) -> None:
        # time embedding for flow step
        self.time_embedding = TimeEmbedding(self.acoustic_transformer_args.dim)
        # override the input projection embedding
        input_dim = self.acoustic_embeddings_dim

        self.input_projection = nn.Linear(input_dim, self.acoustic_transformer_args.dim, bias=False)
        self.time_projection = nn.Linear(
            self.acoustic_transformer_args.dim,
            self.acoustic_transformer_args.dim,
            bias=False,
        )
        self.llm_projection = nn.Linear(
            self.acoustic_transformer_args.input_dim,
            self.acoustic_transformer_args.dim,
            bias=False,
        )

    def _init_output_layer(self) -> None:
        padded_codebook_sizes = self.model_args.get_codebook_sizes(pad_to_multiple=128)
        self.semantic_codebook_output = nn.Linear(
            self.acoustic_transformer_args.dim,
            padded_codebook_sizes[0],
            self.acoustic_transformer_args.use_biases,
        )
        self.acoustic_codebook_output = nn.Linear(
            in_features=self.acoustic_transformer_args.dim,
            # we predict a float for every codebook
            out_features=self.model_args.n_acoustic_codebook,
            bias=False,
        )

    def _init_layers(self) -> None:
        self.layers_ids: list[int] = list(range(self.acoustic_transformer_args.n_layers))
        self.layers = nn.ModuleDict()
        for layer_id in self.layers_ids:
            block = AcousticTransformerBlock(
                layer_id=layer_id,
                args=self.acoustic_transformer_args,
            )
            self.layers[str(layer_id)] = block

        self.norm = rms_norm(self.acoustic_transformer_args.dim, self.acoustic_transformer_args.norm_eps)

    def forward_attention_layers(self, h: torch.Tensor) -> torch.Tensor:
        for layer_id in self.layers_ids:
            layer = self.layers[str(layer_id)]
            h = layer(h)
        return h

    def decode_one_frame(
        self,
        semantic_code: torch.Tensor,
        llm_hidden: torch.Tensor,
    ) -> torch.Tensor:
        B = semantic_code.shape[0]

        # Skip decoding if codebook 0 is [END_AUDIO] token.
        should_decode = semantic_code != self._end_audio_token_id

        # acoustic_codes starts from x_0
        x_0 = torch.randn(B, self.model_args.n_acoustic_codebook).to(dtype=llm_hidden.dtype, device=llm_hidden.device)
        x_0 = self._noise_scale * x_0

        timesteps = self._timesteps.to(dtype=llm_hidden.dtype)
        llm_hidden_zero = torch.zeros_like(llm_hidden)

        # Euler integration with batched conditional + unconditional velocity
        sampled = x_0
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            dt = timesteps[i + 1] - timesteps[i]

            t_emb = self.time_embedding(t.view(-1, 1).repeat(B, 1)).to(llm_hidden.dtype)

            # Batch cond + uncond into a single forward pass (2B batch)
            x_batched = torch.cat([sampled, sampled], dim=0)
            llm_batched = torch.cat([llm_hidden, llm_hidden_zero], dim=0)
            t_emb_batched = torch.cat([t_emb, t_emb], dim=0)

            v_all = self._predict_velocity(
                x_t=x_batched,
                llm_output=llm_batched,
                t_emb=t_emb_batched,
            )
            v_t, uncond_v_t = v_all[:B], v_all[B:]
            v_t = self._cfg_alpha * v_t + (1 - self._cfg_alpha) * uncond_v_t

            sampled = sampled + v_t * dt

        # quantize & mask end_of_audio
        sampled = torch.clamp(sampled, -1, 1)  # manually clip to [-1, 1]
        scaled_x = ((sampled + 1) / 2) * (self.acoustic_embeddings_levels - 1)  # scale to 0 ~ max level
        output_codes = scaled_x.round().long()
        output_codes[~should_decode] = self._empty_audio_token_id
        return output_codes + len(AudioSpecialTokens)

    def _predict_velocity(
        self,
        x_t: torch.Tensor,  # BxC
        llm_output: torch.Tensor,  # BxD
        t_emb: torch.Tensor,  # BxD
    ) -> torch.Tensor:
        x_t = x_t.to(llm_output.dtype)

        t_emb = self.time_projection(t_emb)
        llm_output = self.llm_projection(llm_output)

        acoustic_and_semantic_embeddings = [
            self.input_projection(x_t.unsqueeze(1)),  # Bx1xD
            t_emb.unsqueeze(1),
            llm_output.unsqueeze(1),
        ]
        acoustic_transformer_inputs = torch.concatenate(acoustic_and_semantic_embeddings, dim=1)

        # forward transformer
        attn_output = self.forward_attention_layers(acoustic_transformer_inputs)
        final_hidden = self.norm(attn_output)
        final_hidden = final_hidden.view(-1, acoustic_transformer_inputs.shape[1], final_hidden.shape[-1])
        # predict v_t
        v_t = self.acoustic_codebook_output(final_hidden[:, 0, :])

        return v_t

    def forward(
        self,
        llm_hidden: torch.Tensor,
    ) -> torch.Tensor:
        # llm_hidden: BxD
        semantic_logit = self.semantic_codebook_output(llm_hidden).float()
        semantic_logit[:, self._empty_audio_token_id] = -float("inf")  # 1 = eoa is allowed
        semantic_logit[:, (len(AudioSpecialTokens) + self.model_args.semantic_codebook_size) :] = -float("inf")

        # semantic_logit: Bx1
        semantic_code = semantic_logit.argmax(dim=-1, keepdim=True)

        # acoustic codes, TODO(@chenyo): config sampling
        acoustic_codes = self.decode_one_frame(
            semantic_code.squeeze(1),
            llm_hidden,
        )

        audio_codes = torch.concatenate(
            [semantic_code, acoustic_codes],
            dim=1,
        )
        return audio_codes


class VoxtralTTSProcessorAdapter:
    """
    Provide a HF-compatible interface
    """

    def __init__(self, tokenizer: MistralTokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer

    @cached_property
    def _audio_processor(self) -> AudioEncoder:
        audio_encoder = self.tokenizer.instruct.audio_encoder
        assert isinstance(audio_encoder, AudioEncoder)
        return audio_encoder

    @cached_property
    def audio_token_id(self) -> int:
        return self._audio_processor.special_ids.audio

    @cached_property
    def begin_audio_token_id(self) -> int:
        return self._audio_processor.special_ids.begin_audio

    @cached_property
    def sampling_rate(self) -> int:
        return self._audio_processor.audio_config.sampling_rate

    @cached_property
    def frame_rate(self) -> float:
        return self._audio_processor.audio_config.frame_rate

    def get_num_audio_tokens(
        self,
        audio_length: int,
    ) -> int:
        # TODO(@chenyo): only +1 for TTS
        return 1 + ceil(audio_length / (self.sampling_rate // self.frame_rate))

    def __call__(
        self,
        text: TextInput | list[TextInput] | None = None,
        audios: np.ndarray | list[np.ndarray] | None = None,
        audio_tokens: np.ndarray | list[np.ndarray] | None = None,
        **kwargs,
    ) -> Mapping[str, NestedTensors]:
        if text is None:
            text = []
        if not isinstance(text, list):
            text = [text]
        if audios is None:
            audios = []
        if not isinstance(audios, list):
            audios = [audios]
        if audio_tokens is None:
            audio_tokens = []
        if not isinstance(audio_tokens, list):
            # should be single audio tokens input?
            audio_tokens = [audio_tokens]

        if audios:
            # Allow dummy text, which is used for profiling as well as token inputs
            if any(len(t) > 0 for t in text):
                raise ValueError(
                    "You've passed text inputs instead of token inputs. "
                    "Make sure to process your input via `mistral_common`'s "
                    "tokenizer or pass a chat completion request. "
                    "For more info, see: "
                    "https://github.com/vllm-project/vllm/issues/8411."
                )

            audios_tokens = list[torch.Tensor]()
            audios_processed = list[torch.Tensor]()
            for audio in audios:
                assert isinstance(audio, np.ndarray)
                assert audio.ndim == 1
                num_audio_token = self.get_num_audio_tokens(len(audio))
                audio_tokens = [self.begin_audio_token_id] + [self.audio_token_id] * num_audio_token
                audios_tokens.append(torch.tensor(audio_tokens))
                audios_processed.append(torch.tensor(audio))
            input_ids = torch.cat(audios_tokens)[None].expand(len(text), -1)
            return BatchFeature(
                {
                    "input_ids": input_ids,
                    "audio_arrays": audios_processed,
                }
            )
        elif audio_tokens:
            # Allow dummy text, which is used for profiling as well as token inputs
            if any(len(t) > 0 for t in text):
                raise ValueError(
                    "You've passed text inputs instead of token inputs. "
                    "Make sure to process your input via `mistral_common`'s "
                    "tokenizer or pass a chat completion request. "
                    "For more info, see: "
                    "https://github.com/vllm-project/vllm/issues/8411."
                )

            text_tokens_for_audio = list[torch.Tensor]()
            assert audios is not None
            audio_tokens_pt = list[torch.Tensor]()
            for audio_token_array in audio_tokens:
                assert isinstance(audio_token_array, np.ndarray)
                assert audio_token_array.ndim == 2  # (T, C)
                num_audio_token = audio_token_array.shape[0]
                text_token_ids = [self.begin_audio_token_id] + [self.audio_token_id] * num_audio_token
                text_tokens_for_audio.append(torch.tensor(text_token_ids))
                audio_tokens_pt.append(torch.tensor(audio_token_array))
            input_ids = torch.cat(text_tokens_for_audio)[None].expand(len(text), -1)

            return BatchFeature(
                {
                    "input_ids": input_ids,
                    "audio_tokens": audio_tokens_pt,
                }
            )
        else:
            assert text is not None
            return BatchFeature({"input_ids": torch.tensor(self.tokenizer(text).input_ids)})


class VoxtralTTSProcessingInfo(BaseProcessingInfo):
    def get_tokenizer(self) -> MistralTokenizer:
        tokenizer = cached_tokenizer_from_config(self.ctx.model_config)
        if not isinstance(tokenizer, MistralTokenizer):
            raise ValueError("This model requires `--tokenizer-mode mistral`")

        return tokenizer

    def get_data_parser(self):
        return MultiModalDataParser(
            target_sr=self.get_hf_processor().sampling_rate,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_hf_processor(self, **kwargs: object) -> VoxtralTTSProcessorAdapter:
        return VoxtralTTSProcessorAdapter(self.get_tokenizer())

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {"audio": self.get_max_audio_tokens()}

    def get_max_audio_tokens(self) -> int:
        return self.ctx.model_config.max_model_len

    def get_max_audio_array_len(self) -> int:
        processor = self.get_hf_processor()
        return self.get_max_audio_tokens() * int(processor.sampling_rate // processor.frame_rate)


class VoxtralTTSDummyInputsBuilder(BaseDummyInputsBuilder[VoxtralTTSProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        # mm_options: Mapping[str, BaseDummyOptions] | None = None,
        mm_options: Mapping[str, Any] | None = None,
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        length = self.info.get_max_audio_array_len()
        audio_overrides = mm_options.get("audio") if mm_options else None
        return {"audio": self._get_dummy_audios(length=length, num_audios=num_audios, overrides=audio_overrides)}

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any] | None = None,
    ) -> ProcessorInputs:
        dummy_mm_items = self.get_dummy_mm_data(seq_len, mm_counts, mm_options)
        dummy_audios = dummy_mm_items.get("audio", [])

        audio_chunks: list[AudioChunk] = []
        for audio in dummy_audios:
            audio_item = Audio(
                audio_array=audio,
                sampling_rate=self.info.get_hf_processor().sampling_rate,
                format="wav",
            )
            chunk = AudioChunk(
                input_audio=RawAudio(
                    format="wav",
                    data=audio_item.to_base64(format="wav"),
                ),
            )
            audio_chunks.append(chunk)

        # hacked dummy tokens to make tts work
        assert len(dummy_audios) == 1
        aud = [24] * (1 + int(math.ceil(len(dummy_audios[0]) / (24_000 // 12.5))))
        audio_tokens_all = [25, *aud]
        dummy_tokens = [
            1,
            *audio_tokens_all,
            35,
            4380,
            1395,
            1261,
            2688,
            5117,
            1046,
            36,
            25,
        ]

        parsed_mm_items = self.info.parse_mm_data(dummy_mm_items, validate=False)

        return ProcessorInputs(prompt=dummy_tokens, mm_data_items=parsed_mm_items)


class VoxtralTTSMultiModalProcessor(BaseMultiModalProcessor[VoxtralTTSProcessingInfo]):
    def _get_mm_fields_config(
        self,
        hf_inputs: Mapping[str, NestedTensors],
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            audio_arrays=MultiModalFieldConfig.batched("audio"),
            audio_tokens=MultiModalFieldConfig.batched("audio"),
        )
        # TODO(@alexhliu):
        # set self.target_sr for resampling[]

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        audio_id = processor.audio_token_id

        def get_replacement(item_idx: int):
            audios = mm_items.get_items("audio", AudioProcessorItems)
            if isinstance(audios, AudioProcessorItems):
                audio_len = audios.get_audio_length(item_idx)
                audio_token_len = processor.get_num_audio_tokens(audio_len)
            else:
                raise ValueError(f"Unknown audio item type {type(audios)}")
            return [audio_id] * audio_token_len

        return [
            PromptReplacement(
                modality="audio",
                target="",  # Never match the prompt (see below note)
                replacement=get_replacement,
            ),
        ]

    def _cached_apply_hf_processor(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ) -> tuple[list[int], MultiModalProcessingInfo, bool]:
        prompt_ids, mm_info, _ = super()._cached_apply_hf_processor(inputs, timing_ctx)

        # NOTE: The tokens are already inserted by the chat template
        return prompt_ids, mm_info, True


@MULTIMODAL_REGISTRY.register_processor(
    VoxtralTTSMultiModalProcessor,
    info=VoxtralTTSProcessingInfo,
    dummy_inputs=VoxtralTTSDummyInputsBuilder,
)
class VoxtralTTSAudioGenerationForConditionalGeneration(nn.Module, SupportsMultiModal):
    supported_languages = SUPPORTED_LANGS

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.tokenizer = cached_tokenizer_from_config(vllm_config.model_config)

        config = vllm_config.model_config.hf_config
        self.config = config
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),  # Is this the backbone LLM?
        )
        self.audio_tokenizer = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config,
            prefix=maybe_prefix(prefix, "audio_tokenizer"),
            architectures=["VoxtralTTSAudioTokenizer"],
        )
        self.downsample_factor = self.audio_tokenizer.downsample_factor
        self.acoustic_transformer = FlowMatchingAudioTransformer(
            self.config.audio_config["audio_model_args"],
        )

        audio_encoder = self.tokenizer.instruct.audio_encoder
        self.audio_tok_id = audio_encoder.audio_token
        self.eos_tok_id = self.tokenizer.instruct.tokenizer.eos_id
        self.vocab_size = config.text_config.vocab_size

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        hidden_states = self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def embed_multimodal(self, **kwargs) -> list[torch.Tensor] | torch.Tensor | tuple[torch.Tensor, ...] | None:
        audio_waveforms, audio_tokens = self._parse_and_validate_audio_arrays(**kwargs)
        if (audio_waveforms is None) and (audio_tokens is None):
            return None

        if audio_waveforms is not None and audio_tokens is None:
            assert audio_tokens is None, f"{audio_tokens=} {audio_waveforms=}"
            res = self.audio_tokenizer.encode_waveforms(audio_waveforms)
        else:
            res = self.audio_tokenizer.encode_tokens(audio_tokens)
        return res

    def _parse_and_validate_audio_arrays(
        self, **kwargs: object
    ) -> tuple[list[torch.Tensor] | None, list[torch.Tensor] | None]:
        audio_arrays = kwargs.pop("audio_arrays", None)
        audio_tokens = kwargs.pop("audio_tokens", None)
        if (audio_arrays is None) and (audio_tokens is None):
            return None, None

        if audio_arrays is not None:
            if not isinstance(audio_arrays, (torch.Tensor, list)):
                raise ValueError(f"Incorrect type of images. Got type: {type(audio_arrays)}")
            if isinstance(audio_arrays, torch.Tensor) and audio_arrays.dim() == 3:
                audio_arrays = flatten_bn(audio_arrays)
            if isinstance(audio_arrays, torch.Tensor):
                audio_arrays = list(audio_arrays.unbind(0))

        if audio_tokens is not None:
            # audio tokens MIGHT come from previous decoding step: shape will be BxLxCB

            if isinstance(audio_tokens, list):
                # concatenate along batch dim? is that ok/expected?
                # audio_tokens = torch.concatenate(
                #   [a.transpose(1, 2) for a in audio_tokens], dim=0)
                # audio_tokens = torch.concatenate(audio_tokens, dim=1)

                # these are of shape 1xLxCB
                return None, [a.transpose(1, 2) for a in audio_tokens]

            if isinstance(audio_tokens, torch.Tensor):
                if audio_tokens.dim() == 4:
                    # TODO(chenyo): Fix after rebase with main
                    audio_tokens = audio_tokens.squeeze(0).view(1, audio_tokens.size(-2), audio_tokens.size(-1))
                assert audio_tokens.dim() == 3, f"{audio_tokens.ndim=} {audio_tokens.shape=}"
                audio_tokens = audio_tokens.transpose(1, 2)
                audio_tokens = list(audio_tokens.unsqueeze(1).unbind(0))
            else:
                raise NotImplementedError(f"Unsupported type for audio tokens: {type(audio_tokens)=} {audio_tokens=}")

        return audio_arrays, audio_tokens

    def fake_logits_for_audio_tokens(
        self,
        fake_eos: torch.Tensor,
    ) -> torch.Tensor:
        """
        creates fake logits with all-but-one -inf
        to force decoding of audio tokens
        """
        shape = (fake_eos.shape[0], self.vocab_size)
        fake_logits = torch.full(shape, float("-inf"), device=fake_eos.device)
        is_eos = fake_eos[:, 0].bool()
        fake_logits[is_eos, self.eos_tok_id] = 1.0
        fake_logits[~is_eos, self.audio_tok_id] = 1.0
        return fake_logits

    # TODO(chenyo): Remove this
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        text_logits = self.language_model.compute_logits(
            hidden_states,
        )
        assert text_logits is not None
        return text_logits

    def compute_mm_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        audio_codes = None
        mm_tokens = None
        audio_codes = self.acoustic_transformer(
            llm_hidden=hidden_states,
        )
        fake_eos = torch.where(
            audio_codes[:, 0] == AudioSpecialTokens.id(AudioSpecialTokens.end_audio),
            torch.tensor(1.0, dtype=torch.bfloat16),
            torch.tensor(0.0, dtype=torch.bfloat16),
        )
        # BxC -> Bx1xC since this is per-step
        # Make it a list for vllm-omni processing
        audio_list = list(torch.split(audio_codes.unsqueeze(1), 1, dim=0))
        mm_tokens = {"audio": audio_list}

        return fake_eos, mm_tokens

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # fmt: on
        remapping_rules = [
            (r"^acoustic_transformer\.(.*)$", r"\1"),  # noqa: E501
            (r"^audio_tokenizer\.(.*)$", r"\1"),  # noqa: E501
            (
                r"^mm_audio_embeddings\.audio_codebook_embeddings\.embeddings\.(weight|bias)",
                r"audio_token_embedding.embeddings.\1",
            ),  # noqa: E501
            (r"^mm_audio_embeddings\.tok_embeddings\.weight", r"tok_embeddings.weight"),  # noqa: E501
        ]

        loaded_weights = set()

        def llm_weights_generator():
            nonlocal loaded_weights
            for name, w in weights:
                is_audio_tokenizer = name.startswith(
                    "mm_audio_embeddings.audio_codebook_embeddings"
                ) or name.startswith("audio_tokenizer.")
                is_acoustic_transformer = name.startswith("acoustic_transformer.")

                for pattern, repl in remapping_rules:
                    if re.fullmatch(pattern, name):
                        name = re.sub(pattern, repl, name)

                if is_audio_tokenizer:
                    name = self.audio_tokenizer.load_weight((name, w))
                    loaded_weights.add(f"audio_tokenizer.{name}")
                    continue

                if is_acoustic_transformer:
                    if self.acoustic_transformer is not None:
                        name = self.acoustic_transformer.load_weight((name, w))
                        loaded_weights.add(f"acoustic_transformer.{name}")
                    continue

                yield (name, w)

        for name in self.language_model.load_weights(llm_weights_generator()):
            loaded_weights.add(f"language_model.{name}")

        # If encoder weights were not in the checkpoint, mark them as
        # "loaded" so the weight-validation does not fail.
        # encode_waveforms() will raise at runtime if called without encoder weights.
        if not self.audio_tokenizer._encoder_loaded:
            for name, _ in self.audio_tokenizer.named_parameters():
                if name.startswith(self.audio_tokenizer._encoder_weight_prefixes):
                    loaded_weights.add(f"audio_tokenizer.{name}")

        return loaded_weights
