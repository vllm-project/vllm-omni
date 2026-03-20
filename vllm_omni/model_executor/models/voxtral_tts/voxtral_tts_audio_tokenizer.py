# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig

from vllm_omni.model_executor.models.voxtral_tts.voxtral_tts_audio_generation import (
    AudioSpecialTokens,
    FeedForward,
    MultimodalAudioModelArgs,
    from_nested_dict,
)

try:
    from flash_attn import flash_attn_func

    HAS_FLASH_ATTN = True
except ImportError:
    flash_attn_func = None
    HAS_FLASH_ATTN = False

try:
    from apex.normalization import FusedRMSNorm

    rms_norm = FusedRMSNorm
except ImportError:
    from torch.nn import RMSNorm as RMSNorm

    rms_norm = RMSNorm
from einops import rearrange
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

logger = init_logger(__name__)

if not HAS_FLASH_ATTN:
    logger.warning(
        "flash_attn is not installed. Falling back to PyTorch SDPA for "
        "audio tokenizer attention. Install flash-attn for better performance."
    )

weight_norm = torch.nn.utils.parametrizations.weight_norm


CODEC_NORM_EPS = 1e-2


@dataclass
class AudioTokenizerArgs:
    # audio setting
    channels: int = 1
    sampling_rate: int = 24000
    pretransform_patch_size: int = 240
    patch_proj_kernel_size: int = 7

    # quantizer setting
    semantic_codebook_size: int = 8192
    semantic_dim: int = 256
    acoustic_codebook_size: int = 21
    acoustic_dim: int = 36

    # architecture (general)
    conv_weight_norm: bool = True
    causal: bool = True
    attn_sliding_window_size: int = 16
    half_attn_window_upon_downsampling: bool = True
    dim: int = 1024
    hidden_dim: int = 4096
    head_dim: int = 128
    n_heads: int = 8
    n_kv_heads: int = 8
    qk_norm_eps: float = 1e-6
    qk_norm: bool = True
    use_biases: bool = False
    norm_eps: float = 1e-2
    layer_scale: bool = True
    layer_scale_init: float | None = None

    # architecture (encoder)
    encoder_transformer_lengths_str: str = "2,2,2,2"
    encoder_convs_kernels_str: str = "4,4,4,3"
    encoder_convs_strides_str: str = "2,2,2,1"

    # architecture (decoder)
    decoder_transformer_lengths_str: str = "2,2,2,2"
    decoder_convs_kernels_str: str = "3,4,4,4"
    decoder_convs_strides_str: str = "1,2,2,2"

    def __post_init__(self) -> None:
        assert (
            len(self.encoder_transformer_lengths) == len(self.encoder_convs_kernels) == len(self.encoder_convs_strides)
        )
        assert (
            len(self.decoder_transformer_lengths) == len(self.decoder_convs_kernels) == len(self.decoder_convs_strides)
        )

    def __str2list__(self, input_str: str) -> tuple[int, ...]:
        return tuple(int(i) for i in input_str.split(","))

    @property
    def encoder_transformer_lengths(self) -> tuple[int, ...]:
        return self.__str2list__(self.encoder_transformer_lengths_str)

    @property
    def encoder_convs_kernels(self) -> tuple[int, ...]:
        return self.__str2list__(self.encoder_convs_kernels_str)

    @property
    def encoder_convs_strides(self) -> tuple[int, ...]:
        return self.__str2list__(self.encoder_convs_strides_str)

    @property
    def decoder_transformer_lengths(self) -> tuple[int, ...]:
        return self.__str2list__(self.decoder_transformer_lengths_str)

    @property
    def decoder_convs_kernels(self) -> tuple[int, ...]:
        return self.__str2list__(self.decoder_convs_kernels_str)

    @property
    def decoder_convs_strides(self) -> tuple[int, ...]:
        return self.__str2list__(self.decoder_convs_strides_str)

    @property
    def frame_rate(self) -> float:
        return self.sampling_rate / (self.pretransform_patch_size * math.prod(self.encoder_convs_strides))


class SemanticCodebook(nn.Module):
    """Euclidean distance-based codebook for semantic quantization."""

    def __init__(self, codebook_size: int, codebook_dim: int) -> None:
        super().__init__()
        self.epsilon: float = 1e-5
        self.cluster_usage: torch.Tensor
        self.register_buffer("cluster_usage", torch.ones(codebook_size))
        embedding = torch.zeros(codebook_size, codebook_dim)
        self.embedding_sum: torch.Tensor
        self.register_buffer("embedding_sum", embedding)
        self.register_buffer("_embedding", None, persistent=False)

    @property
    def embedding(self) -> torch.Tensor:
        if self._embedding is None:
            embedding = self.embedding_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]
            self._embedding: torch.Tensor
            self.register_buffer("_embedding", embedding, persistent=False)
            return embedding
        return self._embedding

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dtype.is_floating_point, f"Input should be floats, got {x.dtype}"
        B, D, T = x.shape
        x = rearrange(x, "b d t -> b t d").view(B * T, D)
        embedding = self.embedding.to(x.device)
        distances = torch.cdist(x, embedding, p=2)  # (B*T, V)
        codes = distances.argmin(dim=-1).view(B, 1, T)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        assert not codes.dtype.is_floating_point, f"Codes should be integers, got {codes.dtype}"
        assert codes.shape[1] == self.num_codebooks == 1  # only 1 semantic codebook for now
        codes = codes.squeeze(1)  # BxT
        embedding = self.embedding.to(codes.device)
        quantized = F.embedding(codes, embedding)
        quantized = rearrange(quantized, "b t d -> b d t")
        return quantized

    @property
    def num_codebooks(self) -> int:
        return 1

    @property
    def codebook_sizes(self) -> list[int]:
        return [self.codebook_size]


class AcousticCodebook(nn.Module):
    """Finite Scalar Quantization for acoustic codebooks."""

    def __init__(self, codebook_size: int, codebook_dim: int) -> None:
        super().__init__()
        self.dim = codebook_dim
        self.n_levels = codebook_size
        self.num_codebooks = codebook_dim

    def _quantize(self, x: torch.Tensor, levels: torch.Tensor, ste: bool = True) -> torch.Tensor:
        scaled_x = ((x + 1) / 2) * (levels - 1)
        if ste:
            quant_x = scaled_x + (scaled_x.round() - scaled_x).detach()
        else:
            quant_x = scaled_x.round()
        return quant_x

    def _rescale(self, x: torch.Tensor, levels: int | torch.Tensor) -> torch.Tensor:
        return (x * 2 / (levels - 1)) - 1

    def _codes_from_quantized(self, quantized: torch.Tensor) -> torch.Tensor:
        return quantized.long()

    def _quantized_from_codes(self, codes: torch.Tensor, levels: int) -> torch.Tensor:
        return self._rescale(codes, levels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dtype.is_floating_point, f"Input should be floats, got {x.dtype}"
        x = torch.tanh(x)
        levels = torch.ones_like(x) * self.n_levels
        codes = self._codes_from_quantized(self._quantize(x, levels, ste=False))
        return codes

    def decode(self, codes: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        assert not codes.dtype.is_floating_point, f"Codes should be integers, got {codes.dtype}"
        quantized = self._quantized_from_codes(codes, self.n_levels).to(dtype)
        return quantized


class MistralAudioCodebook(nn.Module):
    """Simplified audio codebook that combines semantic and acoustic quantization.

    This class replaces the complex SplitCodebook implementation with a simpler,
    more direct approach that hard-codes the semantic and acoustic codebook types.
    """

    def __init__(
        self,
        audio_tokenizer_args: AudioTokenizerArgs,
    ) -> None:
        super().__init__()

        # Semantic codebook
        self.semantic_codebook = SemanticCodebook(
            codebook_size=audio_tokenizer_args.semantic_codebook_size,
            codebook_dim=audio_tokenizer_args.semantic_dim,
        )

        # Acoustic codebook
        self.acoustic_codebook = AcousticCodebook(
            codebook_size=audio_tokenizer_args.acoustic_codebook_size,
            codebook_dim=audio_tokenizer_args.acoustic_dim,
        )

        # Store dimensions
        self.semantic_dim = audio_tokenizer_args.semantic_dim
        self.acoustic_dim = audio_tokenizer_args.acoustic_dim
        self.total_dim = self.semantic_dim + self.acoustic_dim

    @property
    def num_codebooks(self) -> int:
        """Total number of codebooks (semantic + acoustic)."""
        return self.semantic_codebook.num_codebooks + self.acoustic_codebook.num_codebooks

    @property
    def codebook_sizes(self) -> list[int]:
        """List of sizes for all codebooks."""
        return self.semantic_codebook.codebook_sizes + self.acoustic_codebook.codebook_sizes

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input tensor into discrete codes using both semantic and acoustic codebooks.

        Args:
            x: Input tensor of shape [B, D, T] where D = semantic_dim + acoustic_dim

        Returns:
            codes: Tensor of shape [B, K, T] where K = num_codebooks
        """
        assert x.dtype.is_floating_point, f"Input should be floats, got {x.dtype}"

        # Split input into semantic and acoustic parts
        semantic_part = x[:, : self.semantic_dim, :]
        acoustic_part = x[:, self.semantic_dim :, :]

        # Quantize each part
        semantic_codes = self.semantic_codebook.encode(semantic_part)
        acoustic_codes = self.acoustic_codebook.encode(acoustic_part)

        # Combine codes along the codebook dimension
        codes = torch.cat([semantic_codes, acoustic_codes], dim=1)
        return codes

    def decode(self, codes: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Decode discrete codes back into continuous representations.

        Args:
            codes: Input codes of shape [B, K, T] where K = num_codebooks
            dtype: Target dtype for the output tensor

        Returns:
            emb: Reconstructed tensor of shape [B, D, T] where D = semantic_dim + acoustic_dim
        """
        assert not codes.dtype.is_floating_point, f"Codes should be integers, got {codes.dtype}"

        # Split codes into semantic and acoustic parts
        semantic_codes = codes[:, : self.semantic_codebook.num_codebooks, :]
        acoustic_codes = codes[:, self.semantic_codebook.num_codebooks :, :]

        # Decode each part
        semantic_emb = self.semantic_codebook.decode(semantic_codes).to(dtype)  # [B, semantic_dim, T]
        acoustic_emb = self.acoustic_codebook.decode(acoustic_codes).to(dtype)  # [B, acoustic_dim, T]

        # Combine embeddings along the dimension axis
        emb = torch.cat([semantic_emb, acoustic_emb], dim=1)
        return emb


### Encoder-decoder layers


def prepare_for_attention(
    x: torch.Tensor,
    time_last: bool = True,
) -> torch.Tensor:
    if time_last:
        x = rearrange(x, "b d t -> (b t) d")
    else:
        x = rearrange(x, "b t d -> (b t) d")
    return x


def pad1d(
    x: torch.Tensor,
    paddings: tuple[int, int],
    mode: str = "constant",
    value: float = 0.0,
) -> torch.Tensor:
    """
    Tiny wrapper around F.pad, just to allow for reflect padding
    on small input. If this is the case, we insert extra 0 padding
    to the right before the reflection happen.
    """
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (
        padding_left,
        padding_right,
    )
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "reflect",
        use_weight_norm: bool = True,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            bias=use_bias,
        )
        self.conv = weight_norm(conv) if use_weight_norm else conv
        self.pad_mode = pad_mode
        self._stride = self.conv.stride[0]
        self._effective_kernel_size = (kernel_size - 1) * self.conv.dilation[0] + 1
        self._padding_total = self._effective_kernel_size - self._stride
        self.stride = self.conv.stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_frames = (x.shape[-1] - self._effective_kernel_size + self._padding_total) / self._stride + 1
        target_length = (math.ceil(n_frames) - 1) * self._stride + (self._effective_kernel_size - self._padding_total)
        extra_padding = target_length - x.shape[-1]
        x = pad1d(x, (self._padding_total, extra_padding), mode=self.pad_mode)
        return self.conv(x)


class CausalConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        trim_ratio: float = 1.0,
        use_weight_norm: bool = True,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            groups=groups,
            bias=use_bias,
        )
        self.conv = weight_norm(conv) if use_weight_norm else conv
        self.trim_ratio = trim_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        total_padding = kernel_size - stride
        out = self.conv(x)
        right_padding = math.ceil(total_padding * self.trim_ratio)
        left_padding = total_padding - right_padding
        return out[..., left_padding : out.shape[-1] - right_padding]


class MultiVocabEmbeddings(nn.Module):
    def __init__(
        self,
        audio_model_args: dict,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.model_args = from_nested_dict(MultimodalAudioModelArgs, audio_model_args)
        # add one more for mask token (-1)
        # also we only handle acoustic tokens here,
        # so we skip the first codebook
        self.codebook_sizes = [c for c in self.model_args.get_codebook_sizes(pad_to_multiple=None)]
        self.offsets = torch.from_numpy(np.cumsum([0] + self.codebook_sizes[:-1]))
        self.total_vocab_size = sum(self.codebook_sizes)
        padded_size = 128 * ((self.total_vocab_size + 127) // 128)
        self.embeddings = nn.Embedding(
            padded_size,
            embedding_dim,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: BxCxL
        self.offsets = self.offsets.to(input_ids.device)
        input_ids = input_ids + self.offsets[torch.newaxis, :, torch.newaxis]
        return self.embeddings(input_ids)


class Attention(nn.Module):
    def __init__(
        self,
        args: AudioTokenizerArgs,
        layer_id: int,
    ) -> None:
        super().__init__()
        self.args = args

        self.n_local_heads: int = args.n_heads
        self.n_local_kv_heads: int = args.n_kv_heads
        self.repeats = self.n_local_heads // self.n_local_kv_heads
        self.layer_id = layer_id
        self.sliding_window = args.attn_sliding_window_size

        def get_alibi_slopes(n_heads: int) -> torch.Tensor:
            def slopes_power_of_2(n: int) -> torch.Tensor:
                # geometric sequence: 1, r, r^2, ..., r^(n-1) with r = 2^(-8/n)
                r = 2.0 ** (-8.0 / n)
                return torch.tensor([r**i for i in range(n)], dtype=torch.float32)

            if math.log2(n_heads).is_integer():
                slopes = slopes_power_of_2(n_heads)
            else:
                m = 2 ** math.floor(math.log2(n_heads))  # largest power of 2 < n_heads
                slopes = torch.cat(
                    [
                        slopes_power_of_2(m),
                        slopes_power_of_2(2 * m)[::2][: n_heads - m],
                    ]
                )
            return slopes  # shape [n_heads], values in (2^-8, 1]

        self.register_buffer(
            "alibi_slopes",
            get_alibi_slopes(self.n_local_heads),
            persistent=False,
        )

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * args.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            args.dim,
            args.n_kv_heads * args.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            args.dim,
            args.n_kv_heads * args.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            args.n_heads * args.head_dim,
            args.dim,
            bias=args.use_biases,
        )

        if args.qk_norm:
            self.q_norm = rms_norm(
                args.n_heads * args.head_dim,
                eps=args.qk_norm_eps,
            )
            self.k_norm = rms_norm(
                args.n_kv_heads * args.head_dim,
                eps=args.qk_norm_eps,
            )

    def _native_attention(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
    ) -> torch.Tensor:
        """SDPA fallback when flash_attn is not installed.

        Manually constructs the combined alibi + causal + sliding-window
        bias and delegates to ``F.scaled_dot_product_attention``.
        """
        B, S, H, D = xq.shape
        Hkv = xk.shape[2]

        # (B, S, H, D) -> (B, H, S, D)
        q = xq.transpose(1, 2)
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)

        # Expand KV heads for GQA
        if H != Hkv:
            repeats = H // Hkv
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        # Build attention bias: alibi + causal + sliding window
        positions = torch.arange(S, device=xq.device)
        # rel_pos[i, j] = j - i
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)  # (S, S)

        # Alibi bias: slope * (j - i), shape (H, S, S)
        alibi_slopes = self.alibi_slopes.to(dtype=xq.dtype, device=xq.device)
        attn_bias = alibi_slopes.view(H, 1, 1) * rel_pos.unsqueeze(0).to(xq.dtype)

        # Causal mask: mask out j > i (future positions)
        if self.args.causal:
            attn_bias = attn_bias.masked_fill(rel_pos.unsqueeze(0) > 0, float("-inf"))

        # Sliding window mask
        window_left = self.sliding_window
        window_right = 0 if self.args.causal else self.sliding_window
        outside_window = (rel_pos < -window_left) | (rel_pos > window_right)
        attn_bias = attn_bias.masked_fill(outside_window.unsqueeze(0), float("-inf"))

        output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias.unsqueeze(0))
        # (B, H, S, D) -> (B, S, H, D)
        return output.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            bsz, (seqlen, _) = 1, x.shape
        else:
            bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        if self.args.qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.args.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.args.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.args.head_dim)

        if HAS_FLASH_ATTN:
            alibi_slopes = self.alibi_slopes.to(torch.float32)
            output = flash_attn_func(
                xq,
                xk,
                xv,
                causal=self.args.causal,
                window_size=(
                    self.sliding_window,
                    0 if self.args.causal else self.sliding_window,
                ),
                alibi_slopes=alibi_slopes,
            )
        else:
            output = self._native_attention(xq, xk, xv)

        output = output.view(bsz, seqlen, self.n_local_heads * self.args.head_dim)
        return self.wo(output).squeeze(0)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        args: AudioTokenizerArgs,
    ) -> None:
        super().__init__()
        self._layer_id = layer_id
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args, layer_id=layer_id)

        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            use_biases=args.use_biases,
        )
        self.attention_norm = rms_norm(
            args.dim,
            eps=args.norm_eps,
        )
        self.ffn_norm = rms_norm(
            args.dim,
            eps=args.norm_eps,
        )
        self.post_attention_norm: nn.Module | None
        self.post_attention_norm = None
        self.post_ffn_norm: nn.Module | None
        self.post_ffn_norm = None
        self.args = args

        self.layer_scale = args.layer_scale
        if self.layer_scale:
            if args.layer_scale_init is None:
                if layer_id < 18:
                    init_scale = 0.1
                elif layer_id <= 24:
                    init_scale = 1e-5
                else:
                    init_scale = 1e-6
            else:
                init_scale = args.layer_scale_init
            self.attention_scale = nn.Parameter(torch.full((args.dim,), init_scale, requires_grad=True))
            self.ffn_scale = nn.Parameter(torch.full((args.dim,), init_scale, requires_grad=True))

    @property
    def layer_id(self) -> int:
        return self._layer_id

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        r = self.attention.forward(self.attention_norm(x))
        if self.post_attention_norm is not None:
            r = self.post_attention_norm(r)
        if self.layer_scale:
            r = self.attention_scale * r
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        if self.post_ffn_norm is not None:
            r = self.post_ffn_norm(r)
        if self.layer_scale:
            r = self.ffn_scale * r
        out = h + r
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        args: AudioTokenizerArgs,
        n_layers: int,
    ) -> None:
        super().__init__()

        self.args = args
        self.n_layers = n_layers

        self.layers_ids: list[int] = list(range(n_layers))

        self.layers = torch.nn.ModuleDict()
        for layer_id in self.layers_ids:
            block = TransformerBlock(layer_id=layer_id, args=args)
            self.layers[str(layer_id)] = block

        assert len(self.layers) == len(self.layers_ids), (
            len(self.layers),
            len(self.layers_ids),
        )

    def device(self) -> torch.device:
        return next(self.parameters()).device

    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        assert len(self.layers_ids) == len(self.layers), (
            len(self.layers_ids),
            len(self.layers),
        )
        h = input_ids
        for layer_id in self.layers_ids:
            layer = self.layers[str(layer_id)]
            assert layer.layer_id == layer_id, (layer.layer_id, layer_id)
            h = layer(
                h,
            )

        return h


class VoxtralTTSAudioTokenizer(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        args = from_nested_dict(AudioTokenizerArgs, self.config.audio_config["codec_args"])
        self.args = args

        if not args.causal:
            # causal mask is hard-coded in forward function
            raise NotImplementedError

        self.patch_size = args.pretransform_patch_size
        self.latent_dim = args.semantic_dim + args.acoustic_dim

        self.input_proj = CausalConv1d(
            args.pretransform_patch_size * args.channels,
            args.dim,
            kernel_size=args.patch_proj_kernel_size,
            use_weight_norm=args.conv_weight_norm,
            use_bias=False,
        )

        ### Encoder
        encoder_blocks: list[nn.Module] = []
        encoder_transformer_lengths = args.encoder_transformer_lengths
        encoder_convs_kernels = args.encoder_convs_kernels
        encoder_convs_strides = args.encoder_convs_strides

        assert encoder_transformer_lengths is not None
        assert encoder_convs_kernels is not None
        assert encoder_convs_strides is not None

        # window_size might change at different layer
        cur_window_size = args.attn_sliding_window_size

        for idx, n_layers in enumerate(encoder_transformer_lengths):
            # transformer
            layer_args = deepcopy(args)
            layer_args.attn_sliding_window_size = cur_window_size
            assert layer_args.qk_norm, "qk_norm must be True for decoder"
            encoder_transformer = Transformer(
                args=layer_args,
                n_layers=n_layers,
            )
            encoder_blocks.append(encoder_transformer)
            # projection
            is_last_layer = idx == len(encoder_transformer_lengths) - 1
            proj_output_dim = self.latent_dim if is_last_layer else args.dim
            if (
                (encoder_convs_kernels[idx] != 1)
                or (encoder_convs_strides[idx] != 1)
                or is_last_layer  # (args.dim != proj_output_dim)
            ):
                encoder_blocks.append(
                    CausalConv1d(
                        args.dim,
                        proj_output_dim,
                        kernel_size=encoder_convs_kernels[idx],
                        stride=encoder_convs_strides[idx],
                        pad_mode="replicate",
                        use_bias=False,
                    )
                )
                if args.half_attn_window_upon_downsampling and (encoder_convs_strides[idx] > 1):
                    assert encoder_convs_strides[idx] == 2, "only supporting 2x downsampling"
                    cur_window_size = cur_window_size // 2
                    assert cur_window_size >= 2

        self.encoder_blocks = nn.ModuleList(encoder_blocks)

        ### Audio token lookup table for LLM
        # placed here assuming we will never need to return tokens
        # from encoder to client
        self.audio_token_embedding = MultiVocabEmbeddings(
            audio_model_args=self.config.audio_config["audio_model_args"],
            embedding_dim=self.config.text_config.hidden_size,
        )

        ### Decoder
        decoder_blocks: list[nn.Module] = []
        decoder_convs_kernels = args.decoder_convs_kernels
        decoder_convs_strides = args.decoder_convs_strides
        decoder_transformer_lengths = args.decoder_transformer_lengths
        assert decoder_convs_kernels is not None
        assert decoder_convs_strides is not None
        assert decoder_transformer_lengths is not None
        # first projection layer is necessary
        decoder_blocks.append(
            CausalConv1d(
                self.latent_dim,
                args.dim,
                kernel_size=decoder_convs_kernels[0],
                stride=decoder_convs_strides[0],
                pad_mode="replicate",
                use_bias=False,
            )
        )
        if args.half_attn_window_upon_downsampling and (decoder_convs_strides[0] > 1):
            assert decoder_convs_strides[0] == 2, "only supporting 2x upsampling"
            cur_window_size = cur_window_size * 2

        for idx, n_layers in enumerate(decoder_transformer_lengths):
            # transformer
            layer_args = deepcopy(args)
            layer_args.attn_sliding_window_size = cur_window_size
            decoder_transformer = Transformer(
                args=layer_args,
                n_layers=n_layers,
            )
            decoder_blocks.append(decoder_transformer)
            # projection
            if (idx + 1 != len(decoder_transformer_lengths)) and (
                (decoder_convs_kernels[idx + 1] != 1) or (decoder_convs_strides[idx + 1] != 1)
            ):
                decoder_blocks.append(
                    CausalConvTranspose1d(
                        args.dim,
                        args.dim,
                        kernel_size=decoder_convs_kernels[idx + 1],
                        stride=decoder_convs_strides[idx + 1],
                        use_bias=False,
                    )
                )
                if args.half_attn_window_upon_downsampling and (decoder_convs_strides[idx + 1] > 1):
                    assert decoder_convs_strides[idx + 1] == 2, "only supporting 2x upsampling"
                    cur_window_size = cur_window_size * 2

        self.decoder_blocks = nn.ModuleList(decoder_blocks)

        self.quantizer = MistralAudioCodebook(args)

        self.output_proj = CausalConv1d(
            args.dim,
            args.pretransform_patch_size,
            kernel_size=args.patch_proj_kernel_size,
            use_weight_norm=args.conv_weight_norm,
            use_bias=False,
        )

        scale_factor = math.prod(encoder_convs_strides)
        assert scale_factor == math.prod(decoder_convs_strides)
        self._frame_rate = args.sampling_rate / (self.patch_size * scale_factor)
        self._sampling_rate = args.sampling_rate
        self._channels = args.channels
        if self._channels != 1:
            raise NotImplementedError

        # Encoder weight prefixes — these may be absent in open-source
        # checkpoints that only ship decoder / quantizer / embedding weights.
        self._encoder_weight_prefixes = ("input_proj.", "encoder_blocks.")
        self._encoder_loaded = False

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def frame_rate(self) -> float:
        return self._frame_rate

    @property
    def sampling_rate(self) -> int:
        return self._sampling_rate

    @property
    def downsample_factor(self) -> int:
        assert self._sampling_rate % self._frame_rate == 0
        return int(self._sampling_rate / self._frame_rate)

    @property
    def num_codebooks(self) -> int:
        """Active number of codebooks used by the quantizer."""
        return self.quantizer.num_codebooks

    @property
    def codebook_sizes(self) -> list[int]:
        """List of size of each codebook"""
        return self.quantizer.codebook_sizes

    def load_weight(self, weight: tuple[str, torch.Tensor]) -> str:
        params_dict = dict(self.named_parameters())
        name, loaded_weight = weight
        if name not in params_dict:
            if name == "quantizer.semantic_codebook.cluster_usage":
                setattr(self.quantizer.semantic_codebook, "cluster_usage", loaded_weight)
            elif name == "quantizer.semantic_codebook.embedding_sum":
                setattr(self.quantizer.semantic_codebook, "embedding_sum", loaded_weight)
            else:
                raise KeyError(f"Weight {name} not found in model parameters.")
        else:
            param = params_dict[name]

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            if name.startswith(self._encoder_weight_prefixes):
                self._encoder_loaded = True
        return name

    def _forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3
        emb = rearrange(
            x,
            "b c (t h) -> b (c h) t",
            h=self.patch_size,
        )

        emb = self.input_proj(emb)
        emb = rearrange(emb, "b d t -> b t d").contiguous()

        for _, block in enumerate(self.encoder_blocks):
            if type(block) is CausalConv1d:
                emb = rearrange(emb, "b t d -> b d t")
                emb = block(emb)
                emb = rearrange(emb, "b d t -> b t d")
            else:
                bsz, _, _ = emb.shape
                emb = prepare_for_attention(emb, False)  # (b * t, d)
                emb = block(emb)  # (b * t, d)
                emb = rearrange(emb, "(b t) d -> b t d", b=bsz)  # (b, t, d)

        emb = rearrange(emb, "b t d -> b d t")
        return emb

    def _tokenize_audio(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the given input tensor to quantized representation.

        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T]
            n_q (int): number of codebooks, default each dimension is a codebook

        Returns:
            codes (torch.Tensor): an int tensor of shape [B, K, T]
                with K the number of codebooks used and T the timestep.
        """
        if x.shape[-1] % self.patch_size != 0:
            pad_length = self.patch_size - (x.shape[-1] % self.patch_size)
            x = F.pad(x, (0, pad_length), mode="constant", value=0)
        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            # bf16 to use alibi bias in flash attn
            emb = self._forward_encoder(x)  # (b, d, t)
        codes = self.quantizer.encode(emb)  # (b, k, t)

        return codes

    def encode_tokens(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        audio_embeddings = []

        for audio_code in x:
            # expected input shape: BxCBxL
            # BxCBxLxD (B=1 for now w/o batching)
            multi_codebook_embedding = self.audio_token_embedding(audio_code)
            # BxLxD, TODO(@alexhliu): add assertion to make sure
            # input_embedding_concat_type == sum
            multi_codebook_embedding = multi_codebook_embedding.sum(dim=1)
            # LxD
            multi_codebook_embedding = multi_codebook_embedding.squeeze(0)
            audio_embeddings.append(multi_codebook_embedding)
        return audio_embeddings

    def encode_waveforms(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        if not self._encoder_loaded:
            raise RuntimeError(
                "encode_waveforms requires encoder weights which are not available in the open-source checkpoint."
            )
        audio_codes = []
        for waveform in x:
            # TODO(@alexhliu): we can batch this for better performance
            assert waveform.dim() == 1
            if torch.min(waveform) < -1.0:
                logger.warning("Min value of input waveform signal is %s", torch.min(waveform))
            if torch.max(waveform) > 1.0:
                logger.warning("Max value of input waveform signal is %s", torch.max(waveform))
            audio_code = self._tokenize_audio(waveform.unsqueeze(0).unsqueeze(0))  # BxVxL, B==1
            audio_code = audio_code + len(
                AudioSpecialTokens.all_special_tokens()
            )  # TODO(@alexhliu): do this properly somewhere else
            # there is always EOA after waveform
            B, V, _ = audio_code.shape
            eoa = torch.zeros([B, V, 1], dtype=audio_code.dtype, device=audio_code.device)
            eoa[:, 0, :] = 1
            audio_code = torch.cat([audio_code, eoa], dim=-1)
            audio_codes.append(audio_code)
        return self.encode_tokens(audio_codes)

    def _forward_decoder(self, emb: torch.Tensor) -> torch.Tensor:
        emb = rearrange(emb, "b d t -> b t d").contiguous()

        for idx, block in enumerate(self.decoder_blocks):
            if type(block) in [CausalConvTranspose1d, CausalConv1d]:
                emb = rearrange(emb, "b t d -> b d t")
                emb = block(emb)
                emb = rearrange(emb, "b d t -> b t d")
            else:
                # This part changed for implementing batch inference
                emb = block(emb)  # (b, t, d) - supports batched attention via flash_attn

        emb = rearrange(emb, "b t d -> b d t")
        emb = self.output_proj(emb)

        out = rearrange(
            emb,
            "b (c h) t -> b c (t h)",
            h=self.patch_size,
        )
        return out

    def decode(self, codes: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Decode the given codes to a reconstructed representation.

        Args:
            codes (torch.Tensor): Int tensor of shape [B, K, T]

        Returns:
            out (torch.Tensor): Float tensor of shape [B, C, T],
            the reconstructed audio.
        """
        emb = self.quantizer.decode(codes, dtype)  # (b, k, t)
        return self._forward_decoder(emb)

    def decode_helper_batch_async(self, codes_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """Batch decode a list of code tensors to reconstructed audio.

        Args:
            codes_list: List of Int tensors, each of shape [T_i, K] where T_i
                varies per request and K is codebooks (e.g. 37).

        Returns:
            List of Float 1-D tensors of reconstructed audio waveform samples.
        """
        chunk_size = 375  # TODO(chenyos): Hardcode. Fix it

        # Pre-process: find EOA and unshift tokens
        processed = []
        for codes in codes_list:
            eoa_mask = codes[:, 0] == 1
            eoa_indices = eoa_mask.nonzero(as_tuple=False)
            cutting_point = eoa_indices[0].item() if len(eoa_indices) > 0 else len(codes)
            audio_tokens = codes[:cutting_point] - 2
            processed.append(audio_tokens)

        # Separate empty and non-empty
        results: list[torch.Tensor | None] = [None] * len(processed)
        non_empty: list[tuple[int, torch.Tensor]] = []
        for idx, tokens in enumerate(processed):
            if len(tokens) == 0:
                results[idx] = torch.tensor([], dtype=torch.float32)
            else:
                non_empty.append((idx, tokens))

        if not non_empty:
            return results

        # Split all requests into chunks and collect
        all_chunks: list[torch.Tensor] = []
        chunk_lengths: list[int] = []
        chunk_map: list[tuple[int, list[int]]] = []
        for orig_idx, tokens in non_empty:
            req_chunk_indices = []
            for i in range(0, len(tokens), chunk_size):
                chunk = tokens[i : i + chunk_size]
                req_chunk_indices.append(len(all_chunks))
                chunk_lengths.append(len(chunk))
                all_chunks.append(chunk)
            chunk_map.append((orig_idx, req_chunk_indices))

        # Pad chunks to max length and batch decode
        max_chunk_len = max(chunk_lengths)
        K = all_chunks[0].shape[1]
        padded = torch.zeros(
            len(all_chunks),
            max_chunk_len,
            K,
            dtype=all_chunks[0].dtype,
            device=all_chunks[0].device,
        )
        for i, chunk in enumerate(all_chunks):
            padded[i, : len(chunk)] = chunk

        audio_codes = padded.to(device=torch.device("cuda"))  # [B, T, K]
        audio_values = self.decode(audio_codes.transpose(1, 2), dtype=torch.bfloat16)  # [B, 1, T_out]
        audio_values = audio_values.detach().cpu().float().squeeze(1)  # [B, T_out]
        if torch.min(audio_values) < -1.0:
            logger.warning("Min value of decoded waveform signal is %s", torch.min(audio_values))
        if torch.max(audio_values) > 1.0:
            logger.warning("Max value of decoded waveform signal is %s", torch.max(audio_values))

        # Trim padding and reassemble per request
        for orig_idx, chunk_indices in chunk_map:
            audio_parts = []
            for ci in chunk_indices:
                expected_samples = chunk_lengths[ci] * self.downsample_factor
                audio_parts.append(audio_values[ci, :expected_samples])
            results[orig_idx] = torch.cat(audio_parts, dim=0) if len(audio_parts) > 1 else audio_parts[0]

        return results
