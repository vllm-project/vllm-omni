"""
Step-Audio2 Thinker - Stage 1 LLM for Audio Understanding

This file contains:
- Audio encoder components (AudioEncoder, Adaptor)
- Audio preprocessing utilities (mel-spectrogram extraction)
- Multi-modal processing (StepAudio2MultiModalProcessor)
- Main thinker model (StepAudio2ThinkerForConditionalGeneration)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypedDict

import librosa  # noqa: TID251
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from transformers.processing_utils import ProcessorMixin
from vllm.config import VllmConfig
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import (
    _merge_multimodal_embeddings,
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
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.step_audio2.step_audio2_constants import (
    DEFAULT_MODEL_CONFIG,
    DEFAULT_TOKEN_CONFIG,
    STEP_AUDIO2_AUDIO_PATCH_TOKEN_ID,
)

if TYPE_CHECKING:
    pass


# ============================================================================
# Audio Preprocessing Utilities
# ============================================================================


def _mel_filters(n_mels: int) -> torch.Tensor:
    """Generate mel filter banks"""
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"
    return torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=400, n_mels=n_mels))


def _normalize_audio(audio: Any) -> torch.Tensor:
    """Normalize audio to torch.Tensor"""
    if isinstance(audio, tuple):
        audio = audio[0]
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    elif not torch.is_tensor(audio):
        audio = torch.tensor(audio, dtype=torch.float32)
    return audio.float()


def log_mel_spectrogram(audio: Any, n_mels: int = 128, padding: int = 479) -> torch.Tensor:
    """
    Convert audio to log mel-spectrogram

    Args:
        audio: Audio waveform (any format)
        n_mels: Number of mel frequency bins (default: 128)
        padding: Right padding for STFT (default: 479)

    Returns:
        Log mel-spectrogram tensor of shape (n_mels, T)
    """
    audio = _normalize_audio(audio)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(400, device=audio.device)
    stft = torch.stft(audio, 400, 160, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    filters = _mel_filters(n_mels).to(audio.device)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


def padding_mels(data: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad mel spectrograms to same length

    Args:
        data: List of mel spectrograms, each of shape (n_mels, T)

    Returns:
        padded_feats: Padded tensor of shape (batch, n_mels, max_T)
        feats_lengths: Lengths of each sequence
    """
    feats_lengths = torch.tensor([s.size(1) - 2 for s in data], dtype=torch.int32)
    feats = [s.t() for s in data]
    padded_feats = pad_sequence(feats, batch_first=True, padding_value=0)
    return padded_feats.transpose(1, 2), feats_lengths


# ============================================================================
# Encoder Utilities
# ============================================================================


def make_non_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Create non-padding mask from lengths"""
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return ~mask


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert mask to attention bias"""
    assert mask.dtype == torch.bool
    assert dtype in [torch.float32, torch.bfloat16, torch.float16]
    mask = mask.to(dtype)
    mask = (1.0 - mask) * -1.0e10
    return mask


# ============================================================================
# Audio Encoder Components
# ============================================================================


class MultiHeadAttention(nn.Module):
    """Multi-head attention for audio encoder"""

    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None):
        B, T, D = q.shape
        head_dim = D // self.n_head

        # Reshape: (B, T, D) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # Use F.scaled_dot_product_attention
        # mask is additive bias (0 or -1e10), pass directly as attn_mask
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        # Reshape back: (B, n_head, T, head_dim) -> (B, T, D)
        out = out.transpose(1, 2).reshape(B, T, D)
        return out, None


class ResidualAttentionBlock(nn.Module):
    """Residual attention block for audio encoder"""

    def __init__(self, n_state: int, n_head: int):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(nn.Linear(n_state, n_mlp), nn.GELU(), nn.Linear(n_mlp, n_state))
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    """
    Step-Audio2 Audio Encoder

    Lightweight audio encoder (6 layers, 512 hidden)
    Optimized for 25s audio chunks at 25Hz
    """

    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.positional_embedding = nn.Embedding(n_ctx, n_state)

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.after_norm = nn.LayerNorm(n_state)

    def forward(self, x: torch.Tensor, x_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: mel spectrogram, shape (batch_size, n_mels, T)
            x_len: length of each audio, shape (batch_size,)

        Returns:
            x: encoded features, shape (batch_size, T//4, n_state)
            x_len: updated lengths, shape (batch_size,)
        """
        T = x.size(-1)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)  # (B, T // 2, n_state)

        # Create attention mask
        mask = make_non_pad_mask(x_len, T).unsqueeze(1)  # (B, 1, T)
        mask = mask_to_bias(mask[:, :, (T + 1) % 2 :: 2], x.dtype)  # (B, 1, T // 2)

        # Add positional embedding
        x = (x + self.positional_embedding.weight[: x.shape[1], :]).to(x.dtype).contiguous()

        # Apply attention blocks
        for block in self.blocks:
            x = block(x, mask.unsqueeze(1))

        # Pool and normalize
        x = x.permute(0, 2, 1)
        x = self.avg_pooler(x)
        x = x.permute(0, 2, 1)
        x_len = (x_len + 1) // 2 // 2
        x = self.after_norm(x)

        return x, x_len


class Adaptor(nn.Module):
    """
    Adaptor to project audio features to LLM dimension

    Maps from n_state (512) to n_hidden (LLM dimension, e.g., 4096)
    with optional downsampling via convolution
    """

    def __init__(
        self, n_state: int = 512, n_hidden: int = 4096, kernel_size: int = 3, stride: int = 2, adapter_state: int = 2048
    ):
        super().__init__()
        self.stride = stride

        if self.stride != -1:
            self.conv = nn.Conv1d(n_state, n_state, kernel_size, stride, padding=1)

        self.linear1 = nn.Linear(n_state, adapter_state)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(adapter_state, n_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: audio features, shape (batch_size, T, n_state)

        Returns:
            x: projected features, shape (batch_size, T//stride, n_hidden)
        """
        if self.stride != -1:
            x = x.permute(0, 2, 1)  # (B, n_state, T)
            x = F.gelu(self.conv(x))
            x = x.permute(0, 2, 1)  # (B, T//stride, n_state)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x


# ============================================================================
# Feature Length Calculation Utility
# ============================================================================


def calculate_audio_feature_length(mel_length: int) -> int:
    """
    Calculate output feature length after encoder + adapter processing.

    Processing chain:
    1. Mel spectrogram input: length = mel_length
    2. Encoder (AudioEncoder):
       - conv1: kernel=3, padding=1, stride=1 → no change
       - conv2: kernel=3, padding=1, stride=2 → T // 2
       - avg_pool: kernel=2, stride=2 → T // 4
       - Output length: (mel_length + 1) // 2 // 2 ≈ mel_length // 4
    3. Adapter (Adaptor):
       - conv: kernel=3, padding=1, stride=2 → T // 2
       - Output length: (encoder_output - 1) // 2 + 1

    Args:
        mel_length: Length of mel spectrogram (time steps)

    Returns:
        Final feature length after encoder + adapter

    Example:
        >>> calculate_audio_feature_length(1000)
        125
    """
    encoder_output_len = (mel_length + 1) // 4

    adapter_output_len = (encoder_output_len - 1) // 2 + 1

    return max(1, adapter_output_len)


class StepAudio2Processor(ProcessorMixin):
    """Processor for Step-Audio2 that handles text tokenization and audio mel-spectrograms."""

    attributes = ["tokenizer"]
    attribute_class = {"tokenizer": "PreTrainedTokenizerBase"}

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.audio_token = "<audio_patch>"

    def __call__(self, text=None, audio=None, **kwargs):
        """Process text and audio inputs."""
        if text is None:
            text = ""

        allowed = {
            "padding",
            "truncation",
            "max_length",
            "return_tensors",
            "add_special_tokens",
            "pad_to_multiple_of",
            "stride",
            "return_attention_mask",
            "return_token_type_ids",
            "return_overflowing_tokens",
            "return_offsets_mapping",
            "return_special_tokens_mask",
            "verbose",
            "is_split_into_words",
        }
        tok_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        if "return_tensors" not in tok_kwargs:
            tok_kwargs["return_tensors"] = "pt"

        encoded = self.tokenizer(text, **tok_kwargs)

        if audio is None:
            encoded["audio_mels"] = torch.empty((0, 128, 0))
            encoded["audio_lens"] = torch.tensor([], dtype=torch.int32)
            return encoded

        audio_list = list(audio) if isinstance(audio, (list, tuple)) else [audio]

        mels = [log_mel_spectrogram(a) for a in audio_list]
        audio_mels, audio_lens = padding_mels(mels)

        encoded["audio_mels"] = audio_mels
        encoded["audio_lens"] = audio_lens
        return encoded


class StepAudio2AudioInputs(TypedDict):
    """Audio inputs for Step-Audio2"""

    audio_mels: torch.Tensor
    """Shape: (batch, num_mel_bins, time_steps)"""

    audio_lens: list[int]
    """Shape: (batch,) - length of each audio in time steps"""


class StepAudio2ProcessingInfo(BaseProcessingInfo):
    """Processing info for Step-Audio2"""

    def get_tokenizer(self, **kwargs):
        return AutoTokenizer.from_pretrained(
            self.ctx.model_config.model,
            trust_remote_code=True,
            fix_mistral_regex=True,
            **kwargs,
        )

    def get_hf_processor(self, **kwargs: object):
        return StepAudio2Processor(self.get_tokenizer())

    def build_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(target_sr=16000)

    # Backward/branch compatibility:
    # some code paths still call get_data_parser() on ProcessingInfo.
    def get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(target_sr=16000)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        max_audio_tokens = 250
        return {"audio": max_audio_tokens}


class StepAudio2DummyInputsBuilder(BaseDummyInputsBuilder[StepAudio2ProcessingInfo]):
    """Dummy inputs builder for Step-Audio2"""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        return "<audio_patch>" * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, object] | None = None,
    ) -> MultiModalDataDict:
        audio_len = 16000 * 25
        num_audios = mm_counts.get("audio", 0)
        return {"audio": self._get_dummy_audios(length=audio_len, num_audios=num_audios)}


class StepAudio2MultiModalProcessor(BaseMultiModalProcessor[StepAudio2ProcessingInfo]):
    """Multi-modal processor for Step-Audio2"""

    def _get_mm_fields_config(
        self,
        hf_inputs,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Map processor outputs to vLLM multimodal fields.

        audio_mels is a padded 3D tensor per audio item with matching audio_lens.
        Use flat_from_sizes to keep one mm item per audio, so downstream mm_kwargs
        alignment does not collapse to a single item.
        """
        audio_lens = hf_inputs.get("audio_lens", torch.empty(0))
        if isinstance(audio_lens, torch.Tensor):
            lens_tensor = audio_lens.flatten()
        elif audio_lens is None:
            lens_tensor = torch.empty(0, dtype=torch.int32)
        else:
            lens_tensor = torch.tensor(
                list(audio_lens) if hasattr(audio_lens, "__iter__") else [int(audio_lens)], dtype=torch.int32
            )

        return dict(
            audio_mels=MultiModalFieldConfig.flat_from_sizes("audio", lens_tensor),
            audio_lens=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        audio_token = getattr(processor, "audio_token", "<audio_patch>")

        audio_items = mm_items.get("audio", [])
        num_audio_items = len(audio_items) if audio_items else 0

        out_mm_data = out_mm_kwargs.get_data()
        audio_lens = out_mm_data.get("audio_lens")

        feature_lens = []
        if audio_lens is not None:
            if isinstance(audio_lens, torch.Tensor):
                audio_lens_list = audio_lens.tolist()
            else:
                audio_lens_list = list(audio_lens) if hasattr(audio_lens, "__iter__") else [audio_lens]

            for length in audio_lens_list:
                if length > 0:
                    feature_len = calculate_audio_feature_length(int(length))
                    feature_lens.append(feature_len)

        # CRITICAL: Align feature_lens with mm_items['audio'] count
        if num_audio_items > 0:
            if len(feature_lens) < num_audio_items:
                default_feature_len = 250
                pad_count = num_audio_items - len(feature_lens)
                feature_lens.extend([default_feature_len] * pad_count)
            elif len(feature_lens) > num_audio_items:
                feature_lens = feature_lens[:num_audio_items]
        elif not feature_lens:
            feature_lens = [250]

        def get_replacement_audio(item_idx: int):
            """Generate replacement tokens for audio placeholder.

            Following Qwen2.5-Omni pattern: returns list[int] of token IDs.
            """
            if item_idx >= len(feature_lens):
                num_features = 1
            else:
                num_features = feature_lens[item_idx]

            return [STEP_AUDIO2_AUDIO_PATCH_TOKEN_ID] * num_features

        return [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_replacement_audio,
            )
        ]

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object] | None = None,
        tok_kwargs: Mapping[str, object] | None = None,
        **_: object,
    ):
        """Call HF processor and post-process outputs"""
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", [])

        mm_kwargs = mm_kwargs or {}
        tok_kwargs = tok_kwargs or {}

        # CRITICAL: Ensure audios is ALWAYS a list to prevent string iteration
        if audios:
            if isinstance(audios, str):
                mm_data["audio"] = [audios]
            elif isinstance(audios, (list, tuple)):
                mm_data["audio"] = audios
            else:
                mm_data["audio"] = [audios]

        hf_inputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

        if "audio_mels" not in hf_inputs:
            hf_inputs["audio_mels"] = torch.empty((0, 128, 0))
        if "audio_lens" not in hf_inputs:
            hf_inputs["audio_lens"] = torch.tensor([], dtype=torch.int32)

        return hf_inputs


# ============================================================================
# Main Thinker Model
# ============================================================================


@MULTIMODAL_REGISTRY.register_processor(
    StepAudio2MultiModalProcessor,
    info=StepAudio2ProcessingInfo,
    dummy_inputs=StepAudio2DummyInputsBuilder,
)
class StepAudio2ThinkerForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    """
    Step-Audio2 Thinker - Stage 1 LLM

    Architecture:
        AudioEncoder (6 layers, 512 hidden) →
        Adaptor (512 → LLM dim) →
        Qwen2 LLM (vocab=158720)
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.have_multimodal_outputs = True

        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        audio_enc_cfg = getattr(config, "audio_encoder_config", {})

        def _get_config_value(cfg, attr_name: str, default_value):
            """Get value from config object or dict"""
            if isinstance(cfg, dict):
                return cfg.get(attr_name, default_value)
            else:
                return getattr(cfg, attr_name, default_value)

        n_mels = _get_config_value(audio_enc_cfg, "n_mels", DEFAULT_MODEL_CONFIG.n_mels)
        n_audio_ctx = _get_config_value(audio_enc_cfg, "n_audio_ctx", DEFAULT_MODEL_CONFIG.n_audio_ctx)
        n_audio_state = _get_config_value(audio_enc_cfg, "n_audio_state", DEFAULT_MODEL_CONFIG.n_audio_state)
        n_audio_head = _get_config_value(audio_enc_cfg, "n_audio_head", DEFAULT_MODEL_CONFIG.n_audio_head)
        n_audio_layer = _get_config_value(audio_enc_cfg, "n_audio_layer", DEFAULT_MODEL_CONFIG.n_audio_layer)
        kernel_size = _get_config_value(audio_enc_cfg, "kernel_size", DEFAULT_MODEL_CONFIG.kernel_size)
        adapter_stride = _get_config_value(audio_enc_cfg, "adapter_stride", DEFAULT_MODEL_CONFIG.adapter_stride)

        n_hidden = None

        if audio_enc_cfg:
            n_hidden = _get_config_value(audio_enc_cfg, "llm_dim", None)

        if n_hidden is None:
            text_config = getattr(config, "text_config", None)
            if text_config is not None:
                n_hidden = _get_config_value(text_config, "hidden_size", None)

        if n_hidden is None:
            n_hidden = _get_config_value(config, "hidden_size", None)

        if n_hidden is None:
            n_hidden = DEFAULT_MODEL_CONFIG.hidden_size

        self.encoder = AudioEncoder(
            n_mels=n_mels,
            n_ctx=n_audio_ctx,
            n_state=n_audio_state,
            n_head=n_audio_head,
            n_layer=n_audio_layer,
        )

        self.adapter = Adaptor(
            n_state=n_audio_state,
            n_hidden=n_hidden,
            kernel_size=kernel_size,
            stride=adapter_stride,
        )

        from transformers import PretrainedConfig

        text_config = getattr(config, "text_config", None)
        lm_config = text_config if isinstance(text_config, PretrainedConfig) else config
        architectures = getattr(lm_config, "architectures", None) or getattr(config, "architectures", None)
        if not architectures:
            model_type = getattr(lm_config, "model_type", None) or getattr(config, "model_type", None)
            model_type_to_arch = {
                "qwen2": "Qwen2ForCausalLM",
                "qwen2_5": "Qwen2_5ForCausalLM",
            }
            arch = model_type_to_arch.get(model_type)
            architectures = [arch] if arch else ["Qwen2ForCausalLM"]
        lm_vllm_config = vllm_config.with_hf_config(lm_config, architectures=architectures)
        self.language_model = init_vllm_registered_model(
            vllm_config=lm_vllm_config, hf_config=lm_config, prefix=maybe_prefix(prefix, "language_model")
        )

        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return "<audio_patch>"
        raise ValueError("Only audio modality is supported")

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def _parse_and_validate_audio_input(self, **kwargs: object) -> StepAudio2AudioInputs | None:
        """Parse audio inputs from kwargs"""
        audio_mels = kwargs.get("audio_mels", None)
        audio_lens = kwargs.get("audio_lens", None)

        if audio_mels is None:
            return None

        if isinstance(audio_mels, torch.Tensor):
            if audio_mels.dim() >= 4:
                audio_mels = audio_mels.flatten(0, 1)
            elif audio_mels.dim() == 3:
                # Already (B, n_mels, T)
                pass
            elif audio_mels.dim() == 2:
                audio_mels = audio_mels.unsqueeze(0)
            else:
                return None
        else:
            audio_mels = flatten_bn(audio_mels, concat=True)

        if isinstance(audio_lens, torch.Tensor):
            if audio_lens.dim() >= 2:
                audio_lens = audio_lens.flatten(0, 1)
            elif audio_lens.dim() == 0:
                audio_lens = audio_lens.unsqueeze(0)
            # dim == 1 is already fine
        else:
            audio_lens = flatten_bn(audio_lens, concat=True)

        expected_n_mels = 128
        if audio_mels.ndim == 3 and audio_mels.size(1) != expected_n_mels:
            return None

        return StepAudio2AudioInputs(
            audio_mels=audio_mels.to(self.dtype).to(self.device),
            audio_lens=audio_lens,
        )

    def _process_audio_input(self, audio_input: StepAudio2AudioInputs) -> tuple[torch.Tensor, ...]:
        """Process audio mels through encoder and adapter"""
        audio_mels = audio_input["audio_mels"]
        audio_lens = audio_input["audio_lens"]
        if not isinstance(audio_lens, torch.Tensor):
            audio_lens = torch.tensor(audio_lens, device=self.device)
        else:
            audio_lens = audio_lens.to(self.device)

        audio_features, audio_lens = self.encoder(audio_mels, audio_lens)

        audio_features = self.adapter(audio_features)

        audio_feature_lens = (audio_lens - 1) // 2 + 1

        audio_feature_list = [audio_features[i, : audio_feature_lens[i]] for i in range(audio_features.size(0))]

        return audio_feature_list

    def get_multimodal_embeddings(self, **kwargs) -> NestedTensors | None:
        """Get multimodal embeddings"""
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return None
        else:
            audio_embeddings = self._process_audio_input(audio_input)
            return audio_embeddings

    def embed_multimodal(self, **kwargs: object) -> NestedTensors:
        """vLLM multimodal encoder entrypoint."""
        audio_embeddings = self.get_multimodal_embeddings(**kwargs)
        if audio_embeddings is None:
            return []
        return audio_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
    ) -> torch.Tensor:
        """Get input embeddings with multimodal fusion"""
        # Under newer vLLM versions the registered LM wrapper exposes
        # get_input_embeddings(), while the inner `.model` may be a bare
        # Qwen2Model without that compatibility method.
        inputs_embeds = self.language_model.embed_input_ids(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = _merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings, STEP_AUDIO2_AUDIO_PATCH_TOKEN_ID
            )
        return inputs_embeds

    def get_language_model(self) -> nn.Module:
        return self.language_model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        """Forward pass - returns OmniOutput for multi-stage pipeline"""
        if intermediate_tensors is not None:
            inputs_embeds = None
        elif inputs_embeds is None:
            audio_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids, audio_embeddings)
            input_ids = None

        hidden_states = self.language_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return OmniOutput(
            text_hidden_states=hidden_states,
            multimodal_outputs={},
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with name mapping from HuggingFace to vLLM structure"""
        from vllm.model_executor.models.utils import AutoWeightsLoader

        # Map HuggingFace weight names to our model structure
        # HF: model.*, lm_head.* → Our: language_model.model.*, language_model.lm_head.*
        # HF: encoder.*, adapter.* → Our: encoder.*, adapter.* (no change)
        def weight_mapper(name: str) -> str:
            if name.startswith("lm_head."):
                return name.replace("lm_head.", "language_model.lm_head.")
            elif name.startswith("model."):
                return name.replace("model.", "language_model.model.")
            else:
                # encoder.* and adapter.* remain unchanged
                return name

        # Apply mapping to weights
        mapped_weights = [(weight_mapper(name), tensor) for name, tensor in weights]

        loader = AutoWeightsLoader(self)
        return loader.load_weights(mapped_weights)

    @staticmethod
    def separate_tokens(
        token_ids: list[int],
        text_max: int = DEFAULT_TOKEN_CONFIG.text_max,
        audio_start: int = DEFAULT_TOKEN_CONFIG.audio_start,
    ) -> tuple[list[int], list[int]]:
        """Separate generated tokens into text and audio tokens"""
        text_tokens = [tid for tid in token_ids if tid < text_max]
        audio_tokens = [tid - audio_start for tid in token_ids if tid >= audio_start]
        return text_tokens, audio_tokens

    @staticmethod
    def has_audio_output(token_ids: list[int], audio_start: int = DEFAULT_TOKEN_CONFIG.audio_start) -> bool:
        """Check if generated tokens contain audio tokens"""
        return any(tid >= audio_start for tid in token_ids)
