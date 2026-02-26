from __future__ import annotations

import base64
import dataclasses
import io
import os
from collections.abc import Callable, Iterable, Mapping
from typing import Any
from urllib.parse import urlparse

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
from transformers import AutoTokenizer
from transformers.activations import ACT2FN
from transformers.utils.hub import cached_file
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen3 import Qwen3Model
from vllm.model_executor.models.utils import AutoWeightsLoader, PPMissingLayer, WeightsMapper, maybe_prefix
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .configuration_qwen3_tts import Qwen3TTSConfig, Qwen3TTSSpeakerEncoderConfig, Qwen3TTSTalkerConfig
from .qwen3_tts_code_predictor_vllm import Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM
from .qwen3_tts_tokenizer import Qwen3TTSTokenizer

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Components ported from the HuggingFace Qwen3-TTS reference implementation.
# Only the classes actually needed by the vLLM AR Talker are kept here.
# ---------------------------------------------------------------------------


class Qwen3TTSTalkerResizeMLP(nn.Module):
    """Two-layer MLP that maps between hidden sizes with an activation in between."""

    def __init__(self, input_size: int, intermediate_size: int, output_size: int, act: str, bias=False):
        super().__init__()
        self.linear_fc1 = nn.Linear(input_size, intermediate_size, bias=bias)
        self.linear_fc2 = nn.Linear(intermediate_size, output_size, bias=bias)
        self.act_fn = ACT2FN[act]

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


# ---- Speaker encoder (ECAPA-TDNN) and helpers ----


class TimeDelayNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding="same",
            padding_mode="reflect",
        )
        self.activation = nn.ReLU()

    def forward(self, hidden_states: torch.Tensor):
        return self.activation(self.conv(hidden_states))


class Res2NetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, scale=8, kernel_size=3, dilation=1):
        super().__init__()
        in_channel = in_channels // scale
        hidden_channel = out_channels // scale
        self.blocks = nn.ModuleList(
            [
                TimeDelayNetBlock(in_channel, hidden_channel, kernel_size=kernel_size, dilation=dilation)
                for _ in range(scale - 1)
            ]
        )
        self.scale = scale

    def forward(self, hidden_states):
        outputs = []
        for i, hidden_part in enumerate(torch.chunk(hidden_states, self.scale, dim=1)):
            if i == 0:
                output_part = hidden_part
            elif i == 1:
                output_part = self.blocks[i - 1](hidden_part)
            else:
                output_part = self.blocks[i - 1](hidden_part + output_part)
            outputs.append(output_part)
        return torch.cat(outputs, dim=1)


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels, se_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, se_channels, kernel_size=1, padding="same", padding_mode="reflect")
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(se_channels, out_channels, kernel_size=1, padding="same", padding_mode="reflect")
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_states):
        hidden_states_mean = hidden_states.mean(dim=2, keepdim=True)
        hidden_states_mean = self.relu(self.conv1(hidden_states_mean))
        hidden_states_mean = self.sigmoid(self.conv2(hidden_states_mean))
        return hidden_states * hidden_states_mean


class SqueezeExcitationRes2NetBlock(nn.Module):
    """TDNN-Res2Net-TDNN-SE building block used in ECAPA-TDNN."""

    def __init__(self, in_channels, out_channels, res2net_scale=8, se_channels=128, kernel_size=1, dilation=1):
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TimeDelayNetBlock(in_channels, out_channels, kernel_size=1, dilation=1)
        self.res2net_block = Res2NetBlock(out_channels, out_channels, res2net_scale, kernel_size, dilation)
        self.tdnn2 = TimeDelayNetBlock(out_channels, out_channels, kernel_size=1, dilation=1)
        self.se_block = SqueezeExcitationBlock(out_channels, se_channels, out_channels)

    def forward(self, hidden_state):
        residual = hidden_state
        hidden_state = self.tdnn1(hidden_state)
        hidden_state = self.res2net_block(hidden_state)
        hidden_state = self.tdnn2(hidden_state)
        hidden_state = self.se_block(hidden_state)
        return hidden_state + residual


class AttentiveStatisticsPooling(nn.Module):
    """Attentive statistic pooling layer: returns concatenated mean and std."""

    def __init__(self, channels, attention_channels=128):
        super().__init__()
        self.eps = 1e-12
        self.tdnn = TimeDelayNetBlock(channels * 3, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = nn.Conv1d(attention_channels, channels, kernel_size=1, padding="same", padding_mode="reflect")

    @staticmethod
    def _length_to_mask(length, max_len=None, dtype=None, device=None):
        if max_len is None:
            max_len = length.max().long().item()
        mask = torch.arange(max_len, device=length.device, dtype=length.dtype).expand(
            len(length), max_len
        ) < length.unsqueeze(1)
        return torch.as_tensor(mask, dtype=dtype, device=device)

    @staticmethod
    def _compute_statistics(x, m, dim=2, eps=1e-12):
        mean = (m * x).sum(dim)
        std = torch.sqrt((m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps))
        return mean, std

    def forward(self, hidden_states):
        seq_length = hidden_states.shape[-1]
        lengths = torch.ones(hidden_states.shape[0], device=hidden_states.device)
        mask = self._length_to_mask(
            lengths * seq_length, max_len=seq_length, dtype=hidden_states.dtype, device=hidden_states.device
        )
        mask = mask.unsqueeze(1)
        total = mask.sum(dim=2, keepdim=True)
        mean, std = self._compute_statistics(hidden_states, mask / total)
        mean = mean.unsqueeze(2).repeat(1, 1, seq_length)
        std = std.unsqueeze(2).repeat(1, 1, seq_length)
        attention = torch.cat([hidden_states, mean, std], dim=1)
        attention = self.conv(self.tanh(self.tdnn(attention)))
        attention = attention.masked_fill(mask == 0, float("-inf"))
        attention = F.softmax(attention, dim=2)
        mean, std = self._compute_statistics(hidden_states, attention)
        pooled_stats = torch.cat((mean, std), dim=1)
        return pooled_stats.unsqueeze(2)


class Qwen3TTSSpeakerEncoder(torch.nn.Module):
    """ECAPA-TDNN speaker encoder.

    Reference: "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
    TDNN Based Speaker Verification" (https://huggingface.co/papers/2005.07143).
    """

    def __init__(self, config: Qwen3TTSSpeakerEncoderConfig):
        super().__init__()
        if len(config.enc_channels) != len(config.enc_kernel_sizes) or len(config.enc_channels) != len(
            config.enc_dilations
        ):
            raise ValueError("enc_channels, enc_kernel_sizes and enc_dilations should have same length")
        self.channels = config.enc_channels
        self.blocks = nn.ModuleList()
        self.blocks.append(
            TimeDelayNetBlock(
                config.mel_dim,
                config.enc_channels[0],
                config.enc_kernel_sizes[0],
                config.enc_dilations[0],
            )
        )
        for i in range(1, len(config.enc_channels) - 1):
            self.blocks.append(
                SqueezeExcitationRes2NetBlock(
                    config.enc_channels[i - 1],
                    config.enc_channels[i],
                    res2net_scale=config.enc_res2net_scale,
                    se_channels=config.enc_se_channels,
                    kernel_size=config.enc_kernel_sizes[i],
                    dilation=config.enc_dilations[i],
                )
            )
        self.mfa = TimeDelayNetBlock(
            config.enc_channels[-1], config.enc_channels[-1], config.enc_kernel_sizes[-1], config.enc_dilations[-1]
        )
        self.asp = AttentiveStatisticsPooling(config.enc_channels[-1], attention_channels=config.enc_attention_channels)
        self.fc = nn.Conv1d(
            config.enc_channels[-1] * 2,
            config.enc_dim,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states_list = []
        for layer in self.blocks:
            hidden_states = layer(hidden_states)
            hidden_states_list.append(hidden_states)
        hidden_states = torch.cat(hidden_states_list[1:], dim=1)
        hidden_states = self.mfa(hidden_states)
        hidden_states = self.asp(hidden_states)
        hidden_states = self.fc(hidden_states)
        return hidden_states.squeeze(-1)


# ---- Audio utilities ----


def _dynamic_range_compression(x, c=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * c)


def mel_spectrogram(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax: int | None = None,
    center: bool = False,
) -> torch.Tensor:
    """Calculate mel spectrogram of an input signal using librosa mel filterbank and torch STFT."""
    if torch.min(y) < -1.0:
        logger.warning("Min value of input waveform signal is %s", torch.min(y))
    if torch.max(y) > 1.0:
        logger.warning("Max value of input waveform signal is %s", torch.max(y))
    device = y.device
    mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    mel_basis = torch.from_numpy(mel).float().to(device)
    hann_window = torch.hann_window(win_size).to(device)
    padding = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(y.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    mel_spec = torch.matmul(mel_basis, spec)
    return _dynamic_range_compression(mel_spec)


# ---------------------------------------------------------------------------
# Main AR Talker model
# ---------------------------------------------------------------------------


class Qwen3TTSTalkerForConditionalGeneration(nn.Module):
    """vLLM-AR talker: step-wise layer-0 codec decoding.
    Predicts residual codebooks (1..Q-1) into `audio_codes` and streams text via `tailing_text_hidden`."""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # Talker backbone (Qwen3 decoder-only).
            "talker.model.layers.": "model.layers.",
            "talker.model.norm.": "model.norm.",
            "talker.model.codec_embedding.": "model.embed_tokens.",
            # Heads / side modules.
            "talker.codec_head.": "lm_head.",
            "talker.model.text_embedding.": "text_embedding.",
            "talker.text_projection.": "text_projection.",
            "talker.code_predictor.": "code_predictor.",
            # Speaker encoder (Base only).
            "speaker_encoder.": "speaker_encoder.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model
        self.config: Qwen3TTSConfig = vllm_config.model_config.hf_config  # type: ignore[assignment]
        self.talker_config: Qwen3TTSTalkerConfig = self.config.talker_config

        # Codec ids: only [0, codebook_vocab_size) are real code indices (layer-0 is sampled from talker vocab).
        # codec_eos_token_id is a special stop token and must not be decoded by SpeechTokenizer.
        self._codebook_vocab_size = int(getattr(self.talker_config.code_predictor_config, "vocab_size", 0) or 0)
        if self._codebook_vocab_size <= 0:
            raise ValueError(
                f"Invalid talker_config.code_predictor_config.vocab_size={self._codebook_vocab_size}; "
                "cannot restrict codec logits safely."
            )
        self._codec_eos_token_id = int(getattr(self.talker_config, "codec_eos_token_id", -1))

        self._eos_logit_bias: float = 0.0

        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = True

        # Used by OmniGPUModelRunner for the GPU-side MTP fast-path.
        self.mtp_hidden_size = int(self.talker_config.hidden_size)
        # OmniGPUModelRunner will store talker_mtp output under this key in
        # per-request additional_information.
        self.talker_mtp_output_key = "audio_codes"

        self.model = Qwen3Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.talker_config.vocab_size,
                self.talker_config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(self.talker_config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        # Text embedding is a separate table in the official implementation.
        self.text_embedding = nn.Embedding(self.talker_config.text_vocab_size, self.talker_config.text_hidden_size)
        self.text_projection = Qwen3TTSTalkerResizeMLP(
            self.talker_config.text_hidden_size,
            self.talker_config.text_hidden_size,
            self.talker_config.hidden_size,
            self.talker_config.hidden_act,
            bias=True,
        )

        # Speaker encoder is only needed for Base voice cloning and may be missing in some checkpoints.
        # Keep it optional to avoid strict weight-loading failures.
        self.speaker_encoder: Qwen3TTSSpeakerEncoder | None = None

        # Code predictor uses an isolated vLLM config so its KV cache doesn't
        # pollute the main engine's static_forward_context (shallow-copy shares
        # the dict by reference — must assign a fresh one).
        predictor_compilation = dataclasses.replace(vllm_config.compilation_config)
        predictor_compilation.static_forward_context = {}
        self._code_predictor_vllm_config = dataclasses.replace(vllm_config, compilation_config=predictor_compilation)
        from vllm.config.vllm import set_current_vllm_config as _set_cfg

        with _set_cfg(self._code_predictor_vllm_config):
            self.code_predictor = Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM(
                vllm_config=self._code_predictor_vllm_config,
                config=self.talker_config.code_predictor_config,
                talker_config=self.talker_config,
                prefix="code_predictor",
            )

        # Constant logit mask: allow only codec ids [1, codebook_vocab_size) plus codec EOS.
        vocab = int(self.talker_config.vocab_size)
        codec_mask = torch.zeros((vocab,), dtype=torch.bool)
        lo, hi = 1, min(self._codebook_vocab_size, vocab)
        if hi > lo:
            codec_mask[lo:hi] = True
        if 0 <= self._codec_eos_token_id < vocab:
            codec_mask[self._codec_eos_token_id] = True
        self.register_buffer("_codec_allowed_mask", codec_mask, persistent=False)

        # Tokenizer for prompt building.
        self._tokenizer = None
        self._speech_tokenizer: Qwen3TTSTokenizer | None = None

    # -------------------- vLLM required hooks --------------------

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(
        self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None
    ) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is None:
            return None

        # Mask out invalid codec ids using the pre-built constant buffer.
        logits = logits.masked_fill(~self._codec_allowed_mask, float("-inf"))

        if self._eos_logit_bias != 0.0:
            eos_id = self._codec_eos_token_id
            if 0 <= eos_id < logits.shape[-1]:
                logits[:, eos_id] = logits[:, eos_id] + self._eos_logit_bias

        return logits

    # -------------------- Omni multimodal output plumbing --------------------

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        hidden = model_outputs
        info_dicts = kwargs.get("runtime_additional_information") or []
        audio_codes_list: list[torch.Tensor] = []
        ref_code_len_list: list[torch.Tensor] = []
        codec_streaming_list: list[torch.Tensor] = []
        for info in info_dicts:
            if not isinstance(info, dict):
                continue
            ac = info.get("audio_codes")
            if isinstance(ac, torch.Tensor):
                audio_codes_list.append(ac)
                cs = info.get("codec_streaming")
                if isinstance(cs, bool):
                    codec_streaming_list.append(
                        torch.full((int(ac.shape[0]),), int(cs), dtype=torch.int8, device=ac.device)
                    )
            ref_len = info.get("ref_code_len")
            if ref_len is None:
                continue
            if isinstance(ref_len, torch.Tensor):
                if ref_len.numel() == 0:
                    raise ValueError("ref_code_len is an empty tensor")
                ref_len_val = int(ref_len.reshape(-1)[-1].item())
            elif isinstance(ref_len, list):
                if len(ref_len) != 1:
                    raise ValueError(f"ref_code_len must be scalar or 1-element list, got len={len(ref_len)}")
                ref_len_val = int(ref_len[0])
            else:
                ref_len_val = int(ref_len)
            if isinstance(ac, torch.Tensor):
                # Emit ref_code_len per-token span for runner slicing (consumer takes the last value).
                ref_code_len_list.append(
                    torch.full((int(ac.shape[0]),), ref_len_val, dtype=torch.int32, device=ac.device)
                )

        if not audio_codes_list:
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})

        audio_codes = torch.cat(audio_codes_list, dim=0)
        span_len = int(audio_codes.shape[0])
        hidden = hidden[:span_len]
        mm: dict[str, torch.Tensor] = {"audio_codes": audio_codes}
        if ref_code_len_list:
            mm["ref_code_len"] = torch.cat(ref_code_len_list, dim=0)[:span_len]
        if codec_streaming_list:
            mm["codec_streaming"] = torch.cat(codec_streaming_list, dim=0)[:span_len]
        return OmniOutput(text_hidden_states=hidden, multimodal_outputs=mm)

    # -------------------- preprocess / postprocess --------------------

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        # Metadata may be passed flattened or under `additional_information`; normalize to flattened keys.
        additional_information = info_dict.get("additional_information")
        if isinstance(additional_information, dict):
            merged: dict[str, Any] = {k: v for k, v in info_dict.items() if k != "additional_information"}
            for k, v in additional_information.items():
                merged.setdefault(k, v)
            info_dict = merged

        span_len = int(input_ids.shape[0])
        if span_len <= 0:
            return input_ids, input_embeds if input_embeds is not None else self.embed_input_ids(input_ids), {}

        text_list = info_dict.get("text")
        if not isinstance(text_list, list) or not text_list or not text_list[0]:
            raise ValueError("Missing additional_information.text for Qwen3-TTS AR talker.")

        task_type = (info_dict.get("task_type") or ["CustomVoice"])[0]
        codec_streaming_val = info_dict.get("codec_streaming")
        if isinstance(codec_streaming_val, list):
            codec_streaming_raw = codec_streaming_val[0] if codec_streaming_val else None
        else:
            codec_streaming_raw = codec_streaming_val
        if isinstance(codec_streaming_raw, bool):
            codec_streaming = codec_streaming_raw
        else:
            codec_streaming = task_type == "Base"

        if span_len > 1:
            # Prefill (prompt embeddings)
            prompt_embeds_cpu = info_dict.get("talker_prompt_embeds")
            tts_pad_embed_cpu = info_dict.get("tts_pad_embed")
            tts_pad_embed = None
            if isinstance(tts_pad_embed_cpu, torch.Tensor) and tts_pad_embed_cpu.numel() > 0:
                tts_pad_embed = tts_pad_embed_cpu.to(device=input_ids.device, dtype=torch.bfloat16).reshape(1, -1)

            # First prefill round: prompt_embeds_cpu is not yet populated.
            # Subsequent prefill rounds (multi-chunk): prompt_embeds_cpu is a Tensor stored by the first round.
            is_first_prefill = not isinstance(prompt_embeds_cpu, torch.Tensor) or prompt_embeds_cpu.ndim != 2
            if is_first_prefill:
                full_prompt_embeds, tailing_text_hidden, tts_pad_embed, ref_code_len = self._build_prompt_embeds(
                    task_type=task_type, info_dict=info_dict
                )
                # Store full prompt embeddings + trailing queue on CPU for later chunks/steps.
                prompt_embeds_cpu = full_prompt_embeds.detach().to("cpu").contiguous()
                info_update: dict[str, Any] = {
                    "talker_prompt_embeds": prompt_embeds_cpu,
                    "tailing_text_hidden": tailing_text_hidden.detach().to("cpu").contiguous(),
                    "tts_pad_embed": tts_pad_embed.detach().to("cpu").contiguous(),
                    "talker_prefill_offset": 0,
                    "codec_streaming": codec_streaming,
                }
                if ref_code_len is not None:
                    info_update["ref_code_len"] = int(ref_code_len)
                # Always return a span_len slice; if the scheduled placeholder is longer, pad with tts_pad_embed.
                # This preserves placeholder/embedding alignment.
                offset = 0
                s = 0
                e = span_len
                take = prompt_embeds_cpu[s:e]
                if int(take.shape[0]) < span_len:
                    pad_n = int(span_len - int(take.shape[0]))
                    pad_rows = tts_pad_embed.detach().to("cpu").contiguous().reshape(1, -1).expand(pad_n, -1)
                    take = torch.cat([take, pad_rows], dim=0)
                prompt_embeds = take.to(device=input_ids.device, dtype=torch.bfloat16)
                info_update["talker_prefill_offset"] = int(offset + span_len)
            else:
                # Subsequent prefill chunk: slice from stored embeddings at running offset.
                if tts_pad_embed is None:
                    raise RuntimeError("Missing `tts_pad_embed` in additional_information; prefill must initialize it.")
                offset = int(info_dict.get("talker_prefill_offset", 0) or 0)
                if offset < 0:
                    offset = 0
                s = max(0, min(offset, int(prompt_embeds_cpu.shape[0])))
                e = max(0, min(offset + span_len, int(prompt_embeds_cpu.shape[0])))
                take = prompt_embeds_cpu[s:e]
                if int(take.shape[0]) < span_len:
                    pad_n = int(span_len - int(take.shape[0]))
                    pad_rows = tts_pad_embed.detach().to("cpu").contiguous().reshape(1, -1).expand(pad_n, -1)
                    take = torch.cat([take, pad_rows], dim=0)
                prompt_embeds = take.to(device=input_ids.device, dtype=torch.bfloat16)
                info_update = {"talker_prefill_offset": int(offset + span_len)}
                info_update["codec_streaming"] = codec_streaming

            # When inputs_embeds is set, token ids are ignored by the model but must stay in-vocab for vLLM bookkeeping.
            input_ids_out = input_ids.clone()
            input_ids_out[:] = int(self.talker_config.codec_pad_id)

            zeros = torch.zeros(
                (prompt_embeds.shape[0], int(self.talker_config.num_code_groups)),
                device=input_ids.device,
                dtype=torch.long,
            )
            info_update["audio_codes"] = zeros
            return input_ids_out, prompt_embeds, info_update

        # Decode: span_len == 1
        # Pop one text-step vector from tailing_text_hidden queue.
        tts_pad_embed_cpu = info_dict.get("tts_pad_embed")
        if not isinstance(tts_pad_embed_cpu, torch.Tensor):
            raise RuntimeError("Missing `tts_pad_embed` in additional_information; prefill must run first.")
        tts_pad_embed = tts_pad_embed_cpu.to(device=input_ids.device, dtype=torch.bfloat16).reshape(1, -1)

        tail_cpu = info_dict.get("tailing_text_hidden")
        if isinstance(tail_cpu, torch.Tensor) and tail_cpu.ndim == 2 and tail_cpu.shape[0] > 0:
            text_step = tail_cpu[:1].to(device=input_ids.device, dtype=torch.bfloat16).reshape(1, -1)
            new_tail = tail_cpu[1:].detach().to("cpu").contiguous() if tail_cpu.shape[0] > 1 else tail_cpu[:0]
        else:
            text_step = tts_pad_embed
            new_tail = tail_cpu if isinstance(tail_cpu, torch.Tensor) else torch.empty((0, tts_pad_embed.shape[-1]))

        last_hidden_cpu = info_dict.get("last_talker_hidden")
        if not isinstance(last_hidden_cpu, torch.Tensor):
            raise RuntimeError("Missing `last_talker_hidden` in additional_information; postprocess must run.")
        past_hidden = last_hidden_cpu.to(device=input_ids.device, dtype=torch.bfloat16).reshape(1, -1)

        # Use OmniGPUModelRunner talker_mtp fast-path for residual codebooks and per-step inputs_embeds update.
        last_id_hidden = self.embed_input_ids(input_ids.reshape(1, 1).to(torch.long)).to(
            device=input_ids.device, dtype=torch.bfloat16
        )
        inputs_embeds_out = last_id_hidden.reshape(1, -1)

        info_update = {
            "tailing_text_hidden": new_tail,
            "mtp_inputs": (past_hidden, text_step),
            "codec_streaming": codec_streaming,
        }
        return input_ids, inputs_embeds_out, info_update

    def postprocess(self, hidden_states: torch.Tensor, **_: Any) -> dict[str, Any]:
        # Keep the last token hidden for the next decode step's code predictor.
        if hidden_states.numel() == 0:
            return {}
        last = hidden_states[-1, :].detach().to("cpu").contiguous()
        return {"last_talker_hidden": last}

    # -------------------- prompt construction helpers --------------------

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                fix_mistral_regex=True,
                use_fast=True,
            )
            self._tokenizer.padding_side = "left"
        return self._tokenizer

    @staticmethod
    def _build_assistant_text(text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    @staticmethod
    def _build_ref_text(text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n"

    @staticmethod
    def _build_instruct_text(instruct: str) -> str:
        return f"<|im_start|>user\n{instruct}<|im_end|>\n"

    @staticmethod
    def estimate_prompt_len_from_additional_information(
        additional_information: dict[str, Any] | None,
        *,
        task_type: str,
        tokenize_prompt: Callable[[str], list[int]],
        codec_language_id: Mapping[str, int] | None,
        spk_is_dialect: Mapping[str, object] | None,
        estimate_ref_code_len: Callable[[object], int | None] | None = None,
    ) -> int:
        """Compute Stage-0 placeholder prompt length (length-only mirror of `_build_prompt_embeds()`).
        It must match the model-side `inputs_embeds` length to avoid extra padding and quality drop."""

        def _first(x: object, default: object) -> object:
            if isinstance(x, list):
                return x[0] if x else default
            return x if x is not None else default

        info: dict[str, Any] = additional_information or {}
        text = _first(info.get("text"), "")
        language = _first(info.get("language"), "Auto")
        speaker = _first(info.get("speaker"), "")
        instruct = _first(info.get("instruct"), "")
        non_streaming_mode_raw = _first(info.get("non_streaming_mode"), None)

        if isinstance(non_streaming_mode_raw, bool):
            non_streaming_mode = non_streaming_mode_raw
        else:
            # Official defaults: CustomVoice/VoiceDesign -> non_streaming_mode=True; Base -> False.
            non_streaming_mode = task_type in ("CustomVoice", "VoiceDesign")

        if not isinstance(text, str):
            text = ""
        if not isinstance(instruct, str):
            instruct = ""
        if not isinstance(language, str):
            language = "Auto"

        instruct_len = 0
        if instruct.strip():
            instruct_text = Qwen3TTSTalkerForConditionalGeneration._build_instruct_text(instruct)
            instruct_len = len(tokenize_prompt(instruct_text))

        # ---- codec prefix portion (matches _build_prompt_embeds) ----
        language_id = None
        if language.lower() != "auto" and codec_language_id:
            language_id = codec_language_id.get(language.lower())
        if (
            language_id is None
            and codec_language_id
            and spk_is_dialect
            and isinstance(language, str)
            and language.lower() in ("chinese", "auto")
            and isinstance(speaker, str)
            and speaker.strip()
        ):
            dialect = spk_is_dialect.get(speaker.lower())
            if isinstance(dialect, str) and dialect:
                language_id = codec_language_id.get(dialect)
        prefill_len = 3 if language_id is None else 4

        speaker_len = 1 if task_type in ("CustomVoice", "Base") else 0
        codec_input_len = prefill_len + speaker_len + 2  # + [codec_pad, codec_bos]
        codec_prefix_len = codec_input_len - 1  # codec_input[:-1] + tts_bos

        # Role header: input_ids[:, :3] in model.
        role_len = 3
        prompt_len = instruct_len + role_len + codec_prefix_len

        # ---- text conditioning portion (matches _build_prompt_embeds) ----
        assistant_text = Qwen3TTSTalkerForConditionalGeneration._build_assistant_text(text)
        assistant_len = len(tokenize_prompt(assistant_text))
        if assistant_len < 8:
            raise ValueError(f"Unexpected assistant prompt length: {assistant_len}")

        if task_type in ("CustomVoice", "VoiceDesign"):
            if non_streaming_mode:
                # model: full text ids (input_ids[:, 3:-5]) + eos + codec_bos step
                prompt_len += assistant_len - 6
            else:
                # model: only first text token in prefill
                prompt_len += 1

        if task_type == "Base":
            xvec_only = bool(_first(info.get("x_vector_only_mode"), False))
            in_context_mode = not xvec_only

            voice_clone_prompt = _first(info.get("voice_clone_prompt"), None)
            if isinstance(voice_clone_prompt, dict):
                icl_flag = _first(voice_clone_prompt.get("icl_mode"), None)
                if isinstance(icl_flag, bool):
                    in_context_mode = icl_flag

            if in_context_mode:
                ref_code = None
                if isinstance(voice_clone_prompt, dict):
                    ref_code = _first(voice_clone_prompt.get("ref_code"), None)

                ref_code_len: int | None = None
                if isinstance(ref_code, list):
                    if ref_code and isinstance(ref_code[0], list):
                        ref_code_len = len(ref_code)
                    elif ref_code:
                        ref_code_len = len(ref_code)
                elif hasattr(ref_code, "shape"):
                    try:
                        shape = getattr(ref_code, "shape")
                        if shape and len(shape) >= 1:
                            ref_code_len = int(shape[0])
                    except Exception:
                        ref_code_len = None

                if ref_code_len is None and estimate_ref_code_len is not None:
                    ref_code_len = estimate_ref_code_len(info.get("ref_audio"))

                if ref_code_len is None:
                    raise ValueError(
                        "Base in-context voice cloning requires either `voice_clone_prompt.ref_code` "
                        "or a readable `ref_audio` that can be mapped to a codec frame length."
                    )

                codec_lens = 1 + int(ref_code_len)  # codec_bos + ref_code
                if non_streaming_mode:
                    # _generate_icl_prompt(non_streaming_mode=True):
                    # text_embed = ref_ids + text_ids + eos.
                    ref_ids = _first(info.get("ref_ids"), None)
                    if isinstance(voice_clone_prompt, dict) and ref_ids is None:
                        ref_ids = _first(voice_clone_prompt.get("ref_ids") or voice_clone_prompt.get("ref_id"), None)

                    if ref_ids is None:
                        ref_text = _first(info.get("ref_text"), "")
                        if not isinstance(ref_text, str) or not ref_text.strip():
                            raise ValueError(
                                "Base in-context non-streaming requires `ref_text` or tokenized `ref_ids`."
                            )
                        ref_text_ids = tokenize_prompt(Qwen3TTSTalkerForConditionalGeneration._build_ref_text(ref_text))
                        ref_ids_len = len(ref_text_ids)
                    elif hasattr(ref_ids, "shape"):
                        shape = getattr(ref_ids, "shape", None)
                        ref_ids_len = int(shape[-1]) if shape else 0
                    elif isinstance(ref_ids, list):
                        ref_ids_len = len(ref_ids)
                    else:
                        ref_ids_len = 0

                    # model uses ref_ids[:, 3:-2] (strip 5 tokens) and text_id=input_ids[:, 3:-5] (strip 8).
                    ref_id_len = max(0, int(ref_ids_len) - 5)
                    text_id_len = max(0, int(assistant_len) - 8)
                    text_embed_len = ref_id_len + text_id_len + 1  # + eos
                    prompt_len += text_embed_len + codec_lens
                else:
                    # _generate_icl_prompt(non_streaming_mode=False): aligned to codec_lens.
                    prompt_len += codec_lens
            else:
                # Base without ICL behaves like CustomVoice.
                if non_streaming_mode:
                    prompt_len += assistant_len - 6
                else:
                    prompt_len += 1

        return max(2, int(prompt_len))

    def _is_probably_base64(self, s: str) -> bool:
        if s.startswith("data:audio"):
            return True
        if ("/" not in s and "\\" not in s) and len(s) > 256:
            return True
        return False

    def _is_url(self, s: str) -> bool:
        try:
            u = urlparse(s)
            return u.scheme in ("http", "https") and bool(u.netloc)
        except Exception:
            return False

    def _decode_base64_to_wav_bytes(self, b64: str) -> bytes:
        if "," in b64 and b64.strip().startswith("data:"):
            b64 = b64.split(",", 1)[1]
        return base64.b64decode(b64)

    def _load_audio_to_np(self, x: str) -> tuple[np.ndarray, int]:
        """Load audio from local path or base64 data URI (no network I/O)."""
        import librosa

        if self._is_url(x):
            raise ValueError("ref_audio URLs must be resolved by the serving layer before reaching the model worker.")
        if self._is_probably_base64(x):
            wav_bytes = self._decode_base64_to_wav_bytes(x)
            with io.BytesIO(wav_bytes) as f:
                audio, sr = sf.read(f, dtype="float32", always_2d=False)
        else:
            audio, sr = librosa.load(x, sr=None, mono=True)

        if isinstance(audio, np.ndarray) and audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        return np.asarray(audio, dtype=np.float32), int(sr)

    def _normalize_ref_audio(self, ref_audio: object) -> tuple[np.ndarray, int]:
        # NOTE: additional_information may serialize (wav, sr) into (nested) lists across processes; be tolerant.
        if isinstance(ref_audio, str):
            return self._load_audio_to_np(ref_audio)

        def _is_sr(x: object) -> bool:
            try:
                v = int(x)  # type: ignore[arg-type]
            except Exception:
                return False
            return 1_000 <= v <= 200_000

        def _is_number_sequence(xs: list[object]) -> bool:
            if not xs:
                return False
            for v in xs[:8]:
                if not isinstance(v, (int, float, np.number)):
                    return False
            return True

        wav_candidates: list[object] = []
        sr_candidates: list[int] = []

        def _summarize(obj: object, depth: int = 0) -> str:
            if depth > 2:
                if isinstance(obj, (int, np.integer)):
                    return f"int({int(obj)})"
                return type(obj).__name__
            if obj is None:
                return "None"
            if isinstance(obj, str):
                if len(obj) <= 16:
                    return f"str({obj!r})"
                return f"str(len={len(obj)})"
            if isinstance(obj, (int, float, np.number)):
                return f"{type(obj).__name__}({obj})"
            if isinstance(obj, np.ndarray):
                return f"ndarray(shape={obj.shape}, dtype={obj.dtype})"
            if isinstance(obj, torch.Tensor):
                return f"Tensor(shape={tuple(obj.shape)}, dtype={obj.dtype}, device={obj.device})"
            if isinstance(obj, dict):
                keys = list(obj.keys())
                return f"dict(keys={keys[:8]})"
            if isinstance(obj, (tuple, list)):
                items = list(obj)
                head = ", ".join(_summarize(x, depth + 1) for x in items[:3])
                return f"{type(obj).__name__}(len={len(items)}; head=[{head}])"
            return f"{type(obj).__name__}"

        def _scan(obj: object, depth: int = 0) -> None:
            if depth > 4:
                return
            if obj is None:
                return
            if _is_sr(obj):
                sr_candidates.append(int(obj))  # type: ignore[arg-type]
                return
            if isinstance(obj, np.ndarray) and obj.size > 0:
                wav_candidates.append(obj)
                return
            if isinstance(obj, torch.Tensor) and obj.numel() > 0:
                wav_candidates.append(obj)
                return
            if isinstance(obj, dict):
                # Inlined ndarray/tensor payloads from OmniInputProcessor.
                if obj.get("__ndarray__") and "data" in obj and "dtype" in obj and "shape" in obj:
                    try:
                        data = obj["data"]
                        dtype = obj["dtype"]
                        shape = obj["shape"]
                        if isinstance(data, (bytes, bytearray, memoryview)):
                            arr = np.frombuffer(data, dtype=dtype).reshape(shape)
                            if arr.size > 0:
                                wav_candidates.append(arr)
                                return
                    except Exception:
                        pass
                if obj.get("__tensor__") and "data" in obj and "dtype" in obj and "shape" in obj:
                    try:
                        data = obj["data"]
                        dtype = obj["dtype"]
                        shape = obj["shape"]
                        if isinstance(data, (bytes, bytearray, memoryview)):
                            # Stored as raw CPU bytes; interpret as numpy for audio.
                            np_dtype = np.dtype(dtype)
                            arr = np.frombuffer(data, dtype=np_dtype).reshape(shape)
                            if arr.size > 0:
                                wav_candidates.append(arr)
                                return
                    except Exception:
                        pass
                wav_obj = obj.get("array") or obj.get("wav") or obj.get("audio")
                sr_obj = obj.get("sampling_rate") or obj.get("sr") or obj.get("sample_rate")
                if wav_obj is not None:
                    _scan(wav_obj, depth + 1)
                if sr_obj is not None:
                    _scan(sr_obj, depth + 1)
                return
            if isinstance(obj, (tuple, list)):
                obj_list = list(obj)
                # Unwrap singleton nesting ([[wav, sr]]).
                while isinstance(obj_list, list) and len(obj_list) == 1:
                    inner = obj_list[0]
                    if isinstance(inner, np.ndarray) and inner.size > 0:
                        wav_candidates.append(inner)
                        return
                    if isinstance(inner, torch.Tensor) and inner.numel() > 0:
                        wav_candidates.append(inner)
                        return
                    if isinstance(inner, dict):
                        _scan(inner, depth + 1)
                        return
                    if isinstance(inner, (tuple, list)):
                        obj_list = list(inner)  # type: ignore[list-item]
                        continue
                    break

                # If the *unwrapped* list is a long list of numbers, treat it as waveform.
                if len(obj_list) >= 512 and _is_number_sequence(obj_list):
                    wav_candidates.append(obj_list)
                    return

                # Otherwise, recurse into elements (but avoid descending into huge numeric lists).
                for item in obj_list:
                    if isinstance(item, list) and len(item) >= 512 and _is_number_sequence(item):  # type: ignore[arg-type]
                        wav_candidates.append(item)
                        continue
                    _scan(item, depth + 1)
                return

        _scan(ref_audio)
        if not sr_candidates:
            raise TypeError(f"ref_audio missing sample_rate: {_summarize(ref_audio)}")
        sr = int(sr_candidates[0])

        def _wav_len(x: object) -> int:
            try:
                if isinstance(x, np.ndarray):
                    return int(x.size)
                if isinstance(x, torch.Tensor):
                    return int(x.numel())
                if isinstance(x, list):
                    return int(len(x))
            except Exception:
                pass
            return 0

        if not wav_candidates:
            raise TypeError(f"ref_audio missing waveform: {_summarize(ref_audio)}")
        wav_obj = max(wav_candidates, key=_wav_len)

        def _to_np(x: object) -> np.ndarray:
            if isinstance(x, np.ndarray):
                return x.astype(np.float32).reshape(-1)
            if isinstance(x, torch.Tensor):
                return x.detach().to("cpu").float().contiguous().numpy().reshape(-1)
            if isinstance(x, dict) and x.get("__ndarray__") and "data" in x and "dtype" in x and "shape" in x:
                data = x["data"]
                dtype = x["dtype"]
                shape = x["shape"]
                if isinstance(data, (bytes, bytearray, memoryview)):
                    return np.frombuffer(data, dtype=dtype).reshape(shape).astype(np.float32).reshape(-1)
            if isinstance(x, list):
                # list of numbers
                if len(x) >= 2 and _is_number_sequence(x):  # type: ignore[arg-type]
                    return np.asarray(x, dtype=np.float32).reshape(-1)
                # list of chunks
                parts: list[np.ndarray] = []
                for part in x:
                    if isinstance(part, (np.ndarray, torch.Tensor, list)):
                        parts.append(_to_np(part))
                if parts:
                    return np.concatenate(parts, axis=0)
            raise TypeError(f"Unsupported waveform type: {type(x)}")

        wav_np = _to_np(wav_obj)
        if wav_np.size < 1024:
            raise ValueError(f"ref_audio waveform too short: {wav_np.size} samples")
        return wav_np, sr

    def _extract_speaker_embedding(self, wav: np.ndarray, sr: int) -> torch.Tensor:
        if self.speaker_encoder is None:
            raise ValueError(
                "This checkpoint does not provide `speaker_encoder` weights; "
                "cannot compute ref_spk_embedding from ref_audio."
            )
        # vLLM workers do not automatically move arbitrary torch.nn.Modules to
        # CUDA. Ensure the speaker encoder is on the same device/dtype as the
        # main model before running it.
        dev = next(self.parameters()).device
        try:
            spk_param = next(self.speaker_encoder.parameters())
            if spk_param.device != dev or spk_param.dtype != torch.bfloat16:
                self.speaker_encoder.to(device=dev, dtype=torch.bfloat16)
        except StopIteration:
            pass
        # Resample to 24kHz for speaker encoder.
        target_sr = int(getattr(self.config.speaker_encoder_config, "sample_rate", 24000))
        if sr != target_sr:
            import librosa

            wav = librosa.resample(y=wav.astype(np.float32), orig_sr=int(sr), target_sr=target_sr)
            sr = target_sr

        # Follow official implementation: mel_spectrogram expects 24kHz.
        mels = mel_spectrogram(
            torch.from_numpy(wav).unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)
        spk = self.speaker_encoder(mels.to(dev, dtype=torch.bfloat16))[0]
        return spk.to(dtype=torch.bfloat16)

    def _ensure_speech_tokenizer_loaded(self) -> Qwen3TTSTokenizer:
        if self._speech_tokenizer is not None:
            return self._speech_tokenizer
        speech_tokenizer_path = cached_file(self.model_path, "speech_tokenizer/config.json")
        if speech_tokenizer_path is None:
            raise ValueError(f"{self.model_path}/speech_tokenizer/config.json not found")
        # Ensure the HF feature extractor config is present. Transformers'
        # AutoFeatureExtractor does not proactively fetch this file.
        preprocessor_config_path = cached_file(self.model_path, "speech_tokenizer/preprocessor_config.json")
        if preprocessor_config_path is None:
            raise ValueError(f"{self.model_path}/speech_tokenizer/preprocessor_config.json not found")
        speech_tokenizer_dir = os.path.dirname(speech_tokenizer_path)
        tok = Qwen3TTSTokenizer.from_pretrained(
            speech_tokenizer_dir,
            torch_dtype=torch.bfloat16,
        )
        # Prefer GPU for encoder if available; otherwise keep CPU.
        dev = next(self.parameters()).device
        if getattr(dev, "type", None) == "cuda":
            try:
                tok.model.to(dev)
                tok.device = dev
            except Exception as e:
                raise RuntimeError(f"Failed to move speech tokenizer to {dev}: {e}") from e
        else:
            tok.device = dev
        self._speech_tokenizer = tok
        return tok

    def _encode_ref_audio_to_code(self, wav: np.ndarray, sr: int) -> torch.Tensor:
        tok = self._ensure_speech_tokenizer_loaded()
        enc = tok.encode(wav, sr=int(sr), return_dict=True)
        ref_code = getattr(enc, "audio_codes", None)
        if isinstance(ref_code, list):
            ref_code = ref_code[0] if ref_code else None
        if isinstance(ref_code, torch.Tensor):
            # 12Hz: likely [T, Q] or [B, T, Q]
            if ref_code.ndim == 3:
                ref_code = ref_code[0]
            return ref_code.to(device=next(self.parameters()).device, dtype=torch.long)
        raise ValueError("SpeechTokenizer.encode did not return audio_codes tensor")

    def _generate_icl_prompt(
        self,
        *,
        text_id: torch.Tensor,
        ref_id: torch.Tensor,
        ref_code: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        tts_eos_embed: torch.Tensor,
        non_streaming_mode: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Ported from official Qwen3TTSForConditionalGeneration.generate_icl_prompt
        text_embed = self.text_projection(self.text_embedding(torch.cat([ref_id, text_id], dim=-1)))
        text_embed = torch.cat([text_embed, tts_eos_embed], dim=1)

        # codec embed (codec bos + codec) 1 T2 D
        codec_embed: list[torch.Tensor] = []
        for i in range(int(self.talker_config.num_code_groups)):
            if i == 0:
                codec_embed.append(self.embed_input_ids(ref_code[:, :1]))
            else:
                codec_embed.append(self.code_predictor.get_input_embeddings()[i - 1](ref_code[:, i : i + 1]))
        codec_embed_sum = torch.cat(codec_embed, dim=1).sum(1).unsqueeze(0)  # [1,T,H]
        codec_embed_sum = torch.cat(
            [
                self.embed_input_ids(
                    torch.tensor([[self.talker_config.codec_bos_id]], device=codec_embed_sum.device, dtype=torch.long)
                ),
                codec_embed_sum,
            ],
            dim=1,
        )

        text_lens = int(text_embed.shape[1])
        codec_lens = int(codec_embed_sum.shape[1])
        if non_streaming_mode:
            # Official non-streaming mode: append the full text conditioning in
            # prefill, and use PAD in decode steps.
            icl_input_embed = text_embed + self.embed_input_ids(
                torch.tensor(
                    [[self.talker_config.codec_pad_id] * text_lens],
                    device=codec_embed_sum.device,
                    dtype=torch.long,
                )
            )
            icl_input_embed = torch.cat([icl_input_embed, codec_embed_sum + tts_pad_embed], dim=1)
            return icl_input_embed, tts_pad_embed
        if text_lens > codec_lens:
            return text_embed[:, :codec_lens] + codec_embed_sum, text_embed[:, codec_lens:]
        text_embed = torch.cat([text_embed] + [tts_pad_embed] * (codec_lens - text_lens), dim=1)
        return text_embed + codec_embed_sum, tts_pad_embed

    def _build_prompt_embeds(
        self,
        *,
        task_type: str,
        info_dict: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int | None]:
        text = (info_dict.get("text") or [""])[0]
        language = (info_dict.get("language") or ["Auto"])[0]
        non_streaming_mode_val = info_dict.get("non_streaming_mode")
        if isinstance(non_streaming_mode_val, list):
            non_streaming_mode_raw = non_streaming_mode_val[0] if non_streaming_mode_val else None
        else:
            non_streaming_mode_raw = non_streaming_mode_val
        if isinstance(non_streaming_mode_raw, bool):
            non_streaming_mode = non_streaming_mode_raw
        else:
            # Match official inference defaults:
            # - CustomVoice/VoiceDesign: non_streaming_mode=True
            # - Base: non_streaming_mode=False
            non_streaming_mode = task_type in ("CustomVoice", "VoiceDesign")

        # Text ids for assistant template (always).
        tok = self._get_tokenizer()
        input_ids = tok(self._build_assistant_text(text), return_tensors="pt", padding=False)["input_ids"].to(
            device=next(self.parameters()).device
        )

        # Optional instruct prefix.
        instruct = (info_dict.get("instruct") or [""])[0]
        instruct_embed = None
        if isinstance(instruct, str) and instruct.strip():
            instruct_ids = tok(self._build_instruct_text(instruct), return_tensors="pt", padding=False)["input_ids"].to(
                device=input_ids.device
            )
            instruct_embed = self.text_projection(self.text_embedding(instruct_ids))

        # tts special token embeds (projected into talker hidden).
        tts_tokens = torch.tensor(
            [[self.config.tts_bos_token_id, self.config.tts_eos_token_id, self.config.tts_pad_token_id]],
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        tts_bos_embed, tts_eos_embed, tts_pad_embed = self.text_projection(self.text_embedding(tts_tokens)).chunk(
            3, dim=1
        )

        # Codec prefill tags.
        language_id = None
        if isinstance(language, str) and language.lower() != "auto":
            language_id = self.talker_config.codec_language_id.get(language.lower())
        # Match official dialect override:
        # If language is Chinese/Auto and the selected speaker is a dialect voice,
        # set language_id to that dialect to improve code generation stability.
        if language_id is None and isinstance(language, str) and language.lower() in ("chinese", "auto"):
            speaker_for_dialect = None
            if task_type == "CustomVoice":
                speaker_for_dialect = (info_dict.get("speaker") or [""])[0]
            if isinstance(speaker_for_dialect, str) and speaker_for_dialect.strip():
                spk_is_dialect = getattr(self.talker_config, "spk_is_dialect", None) or {}
                dialect = spk_is_dialect.get(speaker_for_dialect.lower())
                if isinstance(dialect, str) and dialect:
                    language_id = self.talker_config.codec_language_id.get(dialect)
        if language_id is None:
            codec_prefill_list = [
                [
                    self.talker_config.codec_nothink_id,
                    self.talker_config.codec_think_bos_id,
                    self.talker_config.codec_think_eos_id,
                ]
            ]
        else:
            codec_prefill_list = [
                [
                    self.talker_config.codec_think_id,
                    self.talker_config.codec_think_bos_id,
                    int(language_id),
                    self.talker_config.codec_think_eos_id,
                ]
            ]

        codec_input_0 = self.embed_input_ids(
            torch.tensor(codec_prefill_list, device=input_ids.device, dtype=torch.long)
        )
        codec_input_1 = self.embed_input_ids(
            torch.tensor([[self.talker_config.codec_pad_id, self.talker_config.codec_bos_id]], device=input_ids.device)
        )

        # Speaker embedding/token (task-dependent)
        speaker_embed = None
        ref_code_len: int | None = None

        def _as_singleton(x: object) -> object:
            if isinstance(x, list):
                return x[0] if x else None
            return x

        def _to_long_tensor(x: object, *, device: torch.device) -> torch.Tensor | None:
            x = _as_singleton(x)
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                t = x
            elif isinstance(x, np.ndarray):
                t = torch.from_numpy(x)
            elif isinstance(x, list) and x and all(isinstance(v, (int, np.integer)) for v in x):
                t = torch.tensor(x, dtype=torch.long)
            else:
                return None
            if t.ndim == 1:
                t = t.unsqueeze(0)
            return t.to(device=device, dtype=torch.long)

        def _normalize_voice_clone_prompt(raw: object) -> dict[str, object] | None:
            raw = _as_singleton(raw)
            if raw is None:
                return None
            if isinstance(raw, dict):
                return raw
            # Some callers may pass list[dict] directly.
            if isinstance(raw, list) and raw and isinstance(raw[0], dict):
                return raw[0]
            return None

        if task_type == "Base":
            # Base supports voice clone prompt with in-context mode.
            xvec_only = bool((info_dict.get("x_vector_only_mode") or [False])[0])
            in_context_mode = not xvec_only
            voice_clone_prompt = _normalize_voice_clone_prompt(info_dict.get("voice_clone_prompt"))
            # Official implementation may pass `voice_clone_prompt.icl_mode`.
            if voice_clone_prompt is not None and "icl_mode" in voice_clone_prompt:
                icl_flag = _as_singleton(voice_clone_prompt.get("icl_mode"))
                if isinstance(icl_flag, bool):
                    in_context_mode = icl_flag
                    xvec_only = not in_context_mode
            ref_code = None
            if voice_clone_prompt is not None:
                ref_code = _as_singleton(voice_clone_prompt.get("ref_code"))
            ref_code_t = None
            if isinstance(ref_code, torch.Tensor):
                ref_code_t = ref_code
            elif isinstance(ref_code, np.ndarray):
                ref_code_t = torch.from_numpy(ref_code)
            if isinstance(ref_code_t, torch.Tensor):
                if ref_code_t.ndim == 3:
                    ref_code_t = ref_code_t[0]
                ref_code_t = ref_code_t.to(device=input_ids.device, dtype=torch.long)
                ref_code_len = int(ref_code_t.shape[0])
            elif in_context_mode:
                # Compute ref_code from ref_audio if not provided.
                ref_audio_list = info_dict.get("ref_audio")
                if not isinstance(ref_audio_list, list) or not ref_audio_list:
                    raise ValueError("Base requires `ref_audio`.")
                wav_np, sr = self._normalize_ref_audio(ref_audio_list[0])
                ref_code_t = self._encode_ref_audio_to_code(wav_np, sr).to(device=input_ids.device)
                ref_code_len = int(ref_code_t.shape[0])

            # Speaker embedding: use prompt embed if provided; otherwise extract from audio.
            spk = None
            if voice_clone_prompt is not None:
                spk = _as_singleton(voice_clone_prompt.get("ref_spk_embedding"))
            if isinstance(spk, torch.Tensor):
                speaker_embed = spk.to(device=input_ids.device, dtype=torch.bfloat16).view(1, 1, -1)
            else:
                ref_audio_list = info_dict.get("ref_audio")
                if not isinstance(ref_audio_list, list) or not ref_audio_list:
                    raise ValueError("Base requires `ref_audio`.")
                wav_np, sr = self._normalize_ref_audio(ref_audio_list[0])
                speaker_embed = self._extract_speaker_embedding(wav_np, sr).view(1, 1, -1)

            codec_input = torch.cat([codec_input_0, speaker_embed, codec_input_1], dim=1)

            # Role header (<|im_start|>assistant\n) -> projected text embeds.
            role_embed = self.text_projection(self.text_embedding(input_ids[:, :3]))

            codec_prefix = torch.cat((tts_pad_embed.expand(-1, codec_input.shape[1] - 2, -1), tts_bos_embed), dim=1)
            codec_prefix = codec_prefix + codec_input[:, :-1]
            talker_prompt = torch.cat((role_embed, codec_prefix), dim=1)

            if in_context_mode:
                # Prefer explicit tokenized `ref_ids` if provided (matches official signature).
                ref_ids = _to_long_tensor(info_dict.get("ref_ids"), device=input_ids.device)
                if ref_ids is None and voice_clone_prompt is not None:
                    ref_ids = _to_long_tensor(
                        voice_clone_prompt.get("ref_ids") or voice_clone_prompt.get("ref_id"), device=input_ids.device
                    )
                if ref_ids is None:
                    ref_text = _as_singleton(info_dict.get("ref_text"))
                    if not isinstance(ref_text, str) or not ref_text.strip():
                        raise ValueError("Base in-context voice cloning requires `ref_text` or tokenized `ref_ids`.")
                    ref_ids = tok(self._build_ref_text(ref_text), return_tensors="pt", padding=False)["input_ids"].to(
                        device=input_ids.device
                    )
                icl_input_embed, trailing_text_hidden = self._generate_icl_prompt(
                    text_id=input_ids[:, 3:-5],
                    ref_id=ref_ids[:, 3:-2],
                    ref_code=ref_code_t,  # type: ignore[arg-type]
                    tts_pad_embed=tts_pad_embed,
                    tts_eos_embed=tts_eos_embed,
                    non_streaming_mode=non_streaming_mode,
                )
                talker_prompt = torch.cat([talker_prompt, icl_input_embed], dim=1)
            else:
                # First text token (+ codec_bos).
                if non_streaming_mode:
                    # Official non-streaming mode: put the full text into the
                    # prefill prompt and use PAD for decode steps.
                    text_all = self.text_projection(self.text_embedding(input_ids[:, 3:-5]))
                    text_all = torch.cat([text_all, tts_eos_embed], dim=1)
                    pad_ids = torch.full(
                        (1, int(text_all.shape[1])),
                        int(self.talker_config.codec_pad_id),
                        device=input_ids.device,
                        dtype=torch.long,
                    )
                    talker_prompt = torch.cat(
                        [
                            talker_prompt,
                            text_all + self.embed_input_ids(pad_ids),
                            tts_pad_embed
                            + self.embed_input_ids(
                                torch.tensor([[self.talker_config.codec_bos_id]], device=input_ids.device)
                            ),
                        ],
                        dim=1,
                    )
                    trailing_text_hidden = tts_pad_embed
                else:
                    first_text = self.text_projection(self.text_embedding(input_ids[:, 3:4])) + codec_input[:, -1:]
                    talker_prompt = torch.cat([talker_prompt, first_text], dim=1)
                    trailing_text_hidden = torch.cat(
                        (
                            self.text_projection(self.text_embedding(input_ids[:, 4:-5])),
                            tts_eos_embed,
                        ),
                        dim=1,
                    )

        elif task_type == "CustomVoice":
            speaker = (info_dict.get("speaker") or [""])[0]
            if not isinstance(speaker, str) or not speaker.strip():
                raise ValueError("CustomVoice requires additional_information.speaker.")
            spk_id_map = getattr(self.talker_config, "spk_id", None) or {}
            if speaker.lower() not in spk_id_map:
                raise ValueError(f"Unsupported speaker: {speaker}")
            spk_id = spk_id_map[speaker.lower()]
            # Keep it at least 1D; embedding on a 0-d tensor can return 1D.
            spk_tensor = torch.tensor([spk_id], device=input_ids.device, dtype=torch.long)
            spk_embed = self.embed_input_ids(spk_tensor)
            if spk_embed.ndim == 1:
                spk_embed = spk_embed.view(1, 1, -1)
            elif spk_embed.ndim == 2:
                spk_embed = spk_embed.view(1, 1, -1)
            speaker_embed = spk_embed
            codec_input = torch.cat([codec_input_0, speaker_embed, codec_input_1], dim=1)

            role_embed = self.text_projection(self.text_embedding(input_ids[:, :3]))
            codec_prefix = torch.cat((tts_pad_embed.expand(-1, codec_input.shape[1] - 2, -1), tts_bos_embed), dim=1)
            codec_prefix = codec_prefix + codec_input[:, :-1]
            talker_prompt = torch.cat((role_embed, codec_prefix), dim=1)

            if non_streaming_mode:
                text_all = self.text_projection(self.text_embedding(input_ids[:, 3:-5]))
                text_all = torch.cat([text_all, tts_eos_embed], dim=1)
                pad_ids = torch.full(
                    (1, int(text_all.shape[1])),
                    int(self.talker_config.codec_pad_id),
                    device=input_ids.device,
                    dtype=torch.long,
                )
                talker_prompt = torch.cat(
                    [
                        talker_prompt,
                        text_all + self.embed_input_ids(pad_ids),
                        tts_pad_embed
                        + self.embed_input_ids(
                            torch.tensor([[self.talker_config.codec_bos_id]], device=input_ids.device)
                        ),
                    ],
                    dim=1,
                )
                trailing_text_hidden = tts_pad_embed
            else:
                first_text = self.text_projection(self.text_embedding(input_ids[:, 3:4])) + codec_input[:, -1:]
                talker_prompt = torch.cat([talker_prompt, first_text], dim=1)
                trailing_text_hidden = torch.cat(
                    (
                        self.text_projection(self.text_embedding(input_ids[:, 4:-5])),
                        tts_eos_embed,
                    ),
                    dim=1,
                )

        elif task_type == "VoiceDesign":
            # No known speaker identity; only codec tags + text.
            codec_input = torch.cat([codec_input_0, codec_input_1], dim=1)

            role_embed = self.text_projection(self.text_embedding(input_ids[:, :3]))
            codec_prefix = torch.cat((tts_pad_embed.expand(-1, codec_input.shape[1] - 2, -1), tts_bos_embed), dim=1)
            codec_prefix = codec_prefix + codec_input[:, :-1]
            talker_prompt = torch.cat((role_embed, codec_prefix), dim=1)

            if non_streaming_mode:
                text_all = self.text_projection(self.text_embedding(input_ids[:, 3:-5]))
                text_all = torch.cat([text_all, tts_eos_embed], dim=1)
                pad_ids = torch.full(
                    (1, int(text_all.shape[1])),
                    int(self.talker_config.codec_pad_id),
                    device=input_ids.device,
                    dtype=torch.long,
                )
                talker_prompt = torch.cat(
                    [
                        talker_prompt,
                        text_all + self.embed_input_ids(pad_ids),
                        tts_pad_embed
                        + self.embed_input_ids(
                            torch.tensor([[self.talker_config.codec_bos_id]], device=input_ids.device)
                        ),
                    ],
                    dim=1,
                )
                trailing_text_hidden = tts_pad_embed
            else:
                first_text = self.text_projection(self.text_embedding(input_ids[:, 3:4])) + codec_input[:, -1:]
                talker_prompt = torch.cat([talker_prompt, first_text], dim=1)
                trailing_text_hidden = torch.cat(
                    (
                        self.text_projection(self.text_embedding(input_ids[:, 4:-5])),
                        tts_eos_embed,
                    ),
                    dim=1,
                )
        else:
            raise ValueError(f"Unsupported task_type={task_type}")

        if instruct_embed is not None:
            talker_prompt = torch.cat([instruct_embed, talker_prompt], dim=1)

        return (
            talker_prompt.squeeze(0),  # [prompt_len, H]
            trailing_text_hidden.squeeze(0),  # [T, H]
            tts_pad_embed.squeeze(0),  # [1, H]
            ref_code_len,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Consume talker weights, and conditionally consume speaker encoder
        # weights only if they are present in the checkpoint.
        speaker_weights: list[tuple[str, torch.Tensor]] = []

        def _talker_and_collect_speaker(ws: Iterable[tuple[str, torch.Tensor]]):
            for k, v in ws:
                if k.startswith("speaker_encoder."):
                    speaker_weights.append((k, v))
                    continue
                if k.startswith("talker."):
                    yield k, v

        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(_talker_and_collect_speaker(weights), mapper=self.hf_to_vllm_mapper)

        if speaker_weights:
            if self.speaker_encoder is None:
                self.speaker_encoder = Qwen3TTSSpeakerEncoder(self.config.speaker_encoder_config)
            loaded |= loader.load_weights(speaker_weights, mapper=self.hf_to_vllm_mapper)
        logger.info("Loaded %d weights for Qwen3TTSTalkerForConditionalGeneration", len(loaded))
        return loaded

    # -------------------- GPU-side MTP fast-path --------------------

    @torch.inference_mode()
    def talker_mtp(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        text_step: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """GPU fast-path used by OmniGPUModelRunner to predict residual codebooks (1..Q-1).
        Returns (inputs_embeds, audio_codes) for the current step."""
        bsz = int(input_ids.shape[0])
        q = int(self.talker_config.num_code_groups)
        dev = input_embeds.device

        input_ids = input_ids.reshape(bsz, 1).to(dtype=torch.long, device=dev)
        last_id_hidden = input_embeds.reshape(bsz, 1, -1).to(dtype=torch.bfloat16, device=dev)
        past_hidden = last_talker_hidden.reshape(bsz, 1, -1).to(dtype=torch.bfloat16, device=dev)
        text_step = text_step.reshape(bsz, 1, -1).to(dtype=torch.bfloat16, device=dev)

        # Residual predictor runs fixed-length (Q-1) steps via the vLLM-native code_predictor.
        max_steps = q - 1
        if max_steps <= 0:
            audio_codes = input_ids.reshape(bsz, 1)
            return (last_id_hidden + text_step).reshape(bsz, -1), audio_codes

        # Predict residual codes (1..Q-1) with HF reference sampling params.
        audio_codes = self.code_predictor(
            layer0_code=input_ids.reshape(bsz, 1),
            layer0_embed=last_id_hidden,
            last_talker_hidden=past_hidden,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            top_p=1.0,
        )  # [B, Q]

        # Map invalid layer-0 ids (e.g. EOS) to PAD=0 so SpeechTokenizer sees only real codes.
        layer0 = audio_codes[:, :1]
        invalid0 = (layer0 < 0) | (layer0 >= int(self._codebook_vocab_size))
        audio_codes = torch.where(invalid0.expand_as(audio_codes), torch.zeros_like(audio_codes), audio_codes)

        # Sum embeddings of all code groups, then add the current text step.
        residual_ids_t = audio_codes[:, 1:]
        embeds: list[torch.Tensor] = [last_id_hidden]
        for i in range(max_steps):
            embeds.append(self.code_predictor.get_input_embeddings()[i](residual_ids_t[:, i : i + 1]))
        summed = torch.cat(embeds, dim=1).sum(1, keepdim=True)  # [B,1,H]
        inputs_embeds_out = (summed + text_step).reshape(bsz, -1)
        return inputs_embeds_out, audio_codes.to(dtype=torch.long)
