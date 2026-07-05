from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoFeatureExtractor
from transformers.activations import ACT2FN
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.models.qwen3 import Qwen3Model
from vllm.model_executor.models.utils import AutoWeightsLoader, PPMissingLayer, WeightsMapper, maybe_prefix
from vllm.multimodal.audio import AudioResampler
from vllm.sequence import IntermediateTensors

from vllm_omni.data_entry_keys import OmniPayload
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.utils.speaker_cache import (
    get_speaker_cache,
    iter_custom_voice_profiles,
    load_validated_profile_tensors,
)

from .configuration_qwen3_tts import Qwen3TTSConfig, Qwen3TTSSpeakerEncoderConfig, Qwen3TTSTalkerConfig
from .prompt_embeds_builder import PRECOMPUTED_TEXT_IDS_KEY, Qwen3TTSPromptEmbedsBuilder
from .qwen3_tts_code_predictor_vllm import Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM
from .tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config
from .tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Encoder

logger = init_logger(__name__)

_TRAILING_TEXT_COMPACT_MIN_FRAMES = 64


def _has_tts_text_conditioning(info_dict: dict[str, Any], hidden_states: Any | None = None) -> bool:
    text_list = info_dict.get("text")
    if isinstance(text_list, list) and bool(text_list) and bool(text_list[0]):
        return True
    if PRECOMPUTED_TEXT_IDS_KEY in info_dict:
        return True
    if isinstance(hidden_states, dict):
        tail = hidden_states.get("trailing_text")
        if isinstance(tail, torch.Tensor):
            return True
    return False


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

        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = True
        # Qwen3-TTS postprocess() only reads hidden_states[-1, :]. On a prefix-
        # cache hit, the last hidden state is in the newly computed tail, so
        # reconstructing the full cached_prefix + new_tail span is wasted work.
        # Opt out of the per-step GPU->CPU hidden-state cache write and merged-
        # tensor read; postprocess receives the tail-only slice instead, which
        # avoids ~18 ms merge + ~6 ms write per step (Sy0307 profile, #3665).
        self.requires_full_prefix_cached_hidden_states = False
        # ``codes.audio`` is only needed for future prefix-hit reconstruction
        # after a request has produced codec rows. Keep per-step rows on GPU and
        # materialize the CPU OmniTensorPrefixCache entry once at completion.
        self.deferred_prefix_cache_mm_keys = {"codes.audio"}
        # Used by OmniGPUModelRunner for the GPU-side MTP fast-path.
        self.mtp_hidden_size = int(self.talker_config.hidden_size)
        # OmniGPUModelRunner will store talker_mtp output under this key in
        # per-request additional_information.
        self.talker_mtp_output_key = ("codes", "audio")
        # talker_mtp samples with per-row generators, so explicitly-seeded
        # requests stay batched instead of one scalar forward per row (#4883).
        # Only valid while talker_mtp receives the unpadded active batch (this
        # talker is not graph-wrapped); a padded batch would need the runner to
        # pad the generators list as well.
        self.talker_mtp_accepts_per_row_generators = True

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

        # Initialize speaker_encoder from config (random weights).
        # For load_format: dummy this is the final state; for normal loading,
        # load_weights() overwrites with real weights when the checkpoint
        # provides speaker_encoder.* tensors. Constructing eagerly here
        # (rather than lazily inside load_weights) ensures voice-cloning code
        # paths work under load_format: dummy, which bypasses load_weights
        # entirely (DummyModelLoader fills existing params in-place and never
        # iterates a checkpoint).
        self.speaker_encoder = Qwen3TTSSpeakerEncoder(self.config.speaker_encoder_config)

        # Code predictor uses an isolated vLLM config so its KV cache doesn't
        # pollute the main engine's static_forward_context (shallow-copy shares
        # the dict by reference — must assign a fresh one).
        # Use copy.copy rather than dataclasses.replace: CompilationConfig /
        # VllmConfig are pydantic dataclasses, so `replace` re-runs
        # __init__→pydantic validators + __post_init__. If a backend has
        # already rebound compilation_config.backend to a non-stock value, the
        # piecewise-backend validator in vllm/config/compilation.py rejects it
        # and the clone raises. copy.copy goes through __reduce_ex__, skips
        # validation, and leaves the parent's already-initialized state intact.
        predictor_compilation = copy.copy(vllm_config.compilation_config)
        predictor_compilation.static_forward_context = {}
        self._code_predictor_vllm_config = copy.copy(vllm_config)
        self._code_predictor_vllm_config.compilation_config = predictor_compilation
        from vllm.config.vllm import set_current_vllm_config as _set_cfg

        with _set_cfg(self._code_predictor_vllm_config):
            self.code_predictor = Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM(
                vllm_config=self._code_predictor_vllm_config,
                config=self.talker_config.code_predictor_config,
                talker_config=self.talker_config,
                prefix="code_predictor",
            )

        # Constant logit mask: allow only codec ids [1, codebook_vocab_size) plus codec EOS.
        # Register the *disallowed* form so compute_logits doesn't recompute ~mask per step.
        vocab = int(self.talker_config.vocab_size)
        codec_mask = torch.zeros((vocab,), dtype=torch.bool)
        lo, hi = 1, min(self._codebook_vocab_size, vocab)
        if hi > lo:
            codec_mask[lo:hi] = True
        if 0 <= self._codec_eos_token_id < vocab:
            codec_mask[self._codec_eos_token_id] = True
        self.register_buffer("_codec_disallowed_mask", ~codec_mask, persistent=False)

        # Keys that should stay on GPU in model_intermediate_buffer to avoid
        # CPU-to-GPU round-trips on every decode step.
        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {
            ("codes", "audio"),
            ("hidden_states", "last"),
            ("hidden_states", "trailing_text"),
        }

        # ``text_proj(text_emb(tts_pad_token_id))`` is request-independent —
        # it depends only on frozen ``text_embedding`` / ``text_projection``
        # weights, so we precompute it once in :meth:`_init_runtime_buffers`
        # (called from :meth:`load_weights`) and reuse the same buffer at
        # every prefill and decode step instead of round-tripping it through
        # the per-request ``info_dict``. Declared here as zeros so the
        # attribute exists under ``load_format: dummy`` (which bypasses
        # ``load_weights`` entirely and leaves the value uninitialized).
        model_dtype = getattr(vllm_config.model_config, "dtype", torch.bfloat16)
        self.register_buffer(
            "_tts_pad_embed",
            torch.zeros(1, int(self.talker_config.hidden_size), dtype=model_dtype),
            persistent=False,
        )
        self._embedding_dtype = torch.bfloat16

        tokenizer_config = Qwen3TTSTokenizerV2Config.from_pretrained(
            self.model_path,
            subfolder="speech_tokenizer",
        )
        self.encoder = Qwen3TTSTokenizerV2Encoder._from_config(
            tokenizer_config.encoder_config,
        )
        self.encoder.eval()
        self.encoder.to(dtype=torch.bfloat16)
        self._encoder_valid_num_quantizers = int(tokenizer_config.encoder_valid_num_quantizers)
        self._encoder_downsample_rate = int(tokenizer_config.encode_downsample_rate)

        self._encoder_feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.model_path,
            subfolder="speech_tokenizer",
        )

        self._speaker_cache = get_speaker_cache()
        raw_subtalker_sampling = getattr(vllm_config.model_config, "subtalker_sampling_params", None)
        self._subtalker_sampling_params: dict[str, Any] = (
            dict(raw_subtalker_sampling) if isinstance(raw_subtalker_sampling, Mapping) else {}
        )

        self._stacked_codec_embed: torch.Tensor | None = None

        # Stand-alone wrapper around the embedding layers, encoders and
        # tokenizers required to assemble a talker prefill prompt. Owns the
        # text + speech tokenizers (loaded lazily on first use) and the
        # per-ref-audio / resampler caches, so they don't leak onto the
        # talker class. Other talker variants can construct their own builder
        # with the same set of dependencies and reuse ``build_prompt_embeds``
        # verbatim.
        self._prompt_builder = Qwen3TTSPromptEmbedsBuilder(
            config=self.config,
            talker_config=self.talker_config,
            model_path=self.model_path,
            text_embedding=self.text_embedding,
            text_projection=self.text_projection,
            codec_embed=self.embed_input_ids,
            residual_code_embeddings=lambda: self.code_predictor.get_input_embeddings(),
            speaker_encoder=self.speaker_encoder,
            tts_pad_embed=self._tts_pad_embed,
            encode_ref_audio_batch=self._encode_ref_audio_batch,
            speaker_cache=self._speaker_cache,
        )
        self._load_custom_voice_profiles()

    # -------------------- custom voice profiles --------------------

    def _load_custom_voice_profiles(self) -> None:
        """Preload offline Qwen3-TTS custom voice profiles into speaker cache."""
        custom_voice_dir = getattr(self.config, "custom_voice_dir", None)
        if not custom_voice_dir:
            return

        expected_dim = int(getattr(self.config.speaker_encoder_config, "enc_dim", 0) or 0)
        loaded = 0
        for profile in iter_custom_voice_profiles(custom_voice_dir, expected_model_type="qwen3_tts"):
            tensors = load_validated_profile_tensors(
                profile,
                expected_model_type="qwen3_tts",
                qwen3_embedding_dim=expected_dim,
            )
            if tensors is None:
                continue

            speaker_embedding = tensors["speaker_embedding"].reshape(-1).contiguous().cpu()
            mode = str(profile.get("mode") or "xvec").lower()
            ref_code = tensors.get("ref_code")
            artifacts: dict[str, Any] = {
                "ref_spk_embedding": speaker_embedding,
                "ref_code": ref_code.contiguous().cpu() if isinstance(ref_code, torch.Tensor) else None,
                "icl_mode": mode == "icl",
                "ref_text": profile.get("ref_text"),
            }
            key = self._speaker_cache.make_cache_key(
                profile["voice_name_lower"],
                model_type=f"qwen3_tts_{mode}",
                created_at=0,
            )
            self._speaker_cache.put(key, artifacts)
            loaded += 1

        if loaded:
            logger.info("Loaded %d precomputed Qwen3-TTS custom voice profile(s) from %s", loaded, custom_voice_dir)

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
        logits = logits.masked_fill(self._codec_disallowed_mask, float("-inf"))

        return logits

    # -------------------- Omni multimodal output plumbing --------------------

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        hidden = model_outputs
        info_dicts = kwargs.get("model_intermediate_buffer")
        if info_dicts is None:
            info_dicts = kwargs.get("runtime_additional_information") or []
        if "runtime_additional_information" in kwargs and "model_intermediate_buffer" not in kwargs:
            logger.warning_once("runtime_additional_information is deprecated, use model_intermediate_buffer")
        audio_codes_list: list[torch.Tensor] = []
        ref_code_len_list: list[torch.Tensor] = []
        ref_code_list: list[torch.Tensor] = []
        has_ref_code = False
        codec_streaming_list: list[torch.Tensor] = []
        for info in info_dicts:
            if not isinstance(info, dict):
                ref_code_list.append(torch.empty(0, dtype=torch.long))
                continue
            codes = info.get("codes", {})
            meta = info.get("meta", {})
            ac = codes.get("audio")
            if isinstance(ac, torch.Tensor):
                audio_codes_list.append(ac)
                cs = meta.get("codec_streaming")
                if isinstance(cs, bool):
                    codec_streaming_list.append(
                        torch.full((int(ac.shape[0]),), int(cs), dtype=torch.int8, device=ac.device)
                    )
            ref_code = codes.get("ref")
            if isinstance(ref_code, torch.Tensor) and ref_code.numel() > 0:
                ref_code_list.append(ref_code)
                has_ref_code = True
            else:
                ref_code_list.append(torch.empty(0, dtype=torch.long))
            ref_len = meta.get("ref_code_len")
            if ref_len is None:
                continue
            if not isinstance(ac, torch.Tensor):
                continue
            span_len = int(ac.shape[0])
            if isinstance(ref_len, torch.Tensor):
                if ref_len.numel() == 0:
                    raise ValueError("ref_code_len is an empty tensor")
                ref_len_tail = ref_len.reshape(-1)[-1:].to(dtype=torch.int32, device=ac.device)
                ref_code_len_list.append(ref_len_tail.expand(span_len).contiguous())
                continue
            if isinstance(ref_len, list):
                if len(ref_len) != 1:
                    raise ValueError(f"ref_code_len must be scalar or 1-element list, got len={len(ref_len)}")
                ref_len_val = int(ref_len[0])
            else:
                ref_len_val = int(ref_len)
            ref_code_len_list.append(torch.full((span_len,), ref_len_val, dtype=torch.int32, device=ac.device))

        if not audio_codes_list:
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})

        audio_codes = torch.cat(audio_codes_list, dim=0)
        span_len = int(audio_codes.shape[0])
        mm: OmniPayload = {"codes": {"audio": audio_codes}}
        if ref_code_len_list:
            mm.setdefault("meta", {})["ref_code_len"] = torch.cat(ref_code_len_list, dim=0)[:span_len]
        if has_ref_code:
            # Batch-aligned, one entry per request: ``to_payload_element``
            # indexes ``element[idx]`` per request, and a shorter list would
            # silently broadcast one request's reference codes to the others.
            mm.setdefault("codes", {})["ref"] = ref_code_list
        if codec_streaming_list:
            mm.setdefault("meta", {})["codec_streaming"] = torch.cat(codec_streaming_list, dim=0)[:span_len]
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

        payload: OmniPayload = info_dict
        embed = payload.get("embed", {})
        hs = payload.get("hidden_states", {})
        meta = payload.get("meta", {})

        span_len = int(input_ids.shape[0])
        if span_len <= 0:
            return input_ids, input_embeds if input_embeds is not None else self.embed_input_ids(input_ids), {}
        is_prefill_raw = info_dict.get("_omni_is_prefill")
        if isinstance(is_prefill_raw, bool):
            is_prefill = is_prefill_raw
        else:
            try:
                is_prefill = int(info_dict["_omni_num_computed_tokens"]) < int(info_dict["_omni_prompt_len"])
            except Exception:
                is_prefill = span_len > 1

        if not _has_tts_text_conditioning(info_dict, hs):
            raise ValueError("Missing Qwen3-TTS text conditioning: provide `text` or precomputed text token ids.")

        task_type = (info_dict.get("task_type") or ["CustomVoice"])[0]
        codec_streaming_val = meta.get("codec_streaming")
        if isinstance(codec_streaming_val, list):
            codec_streaming_raw = codec_streaming_val[0] if codec_streaming_val else None
        else:
            codec_streaming_raw = codec_streaming_val
        if isinstance(codec_streaming_raw, bool):
            codec_streaming = codec_streaming_raw
        else:
            codec_streaming = task_type == "Base"

        # ``tts_pad_embed`` is a request-independent constant — see
        # :meth:`_init_runtime_buffers`. Materialize once on the right
        # device/dtype and reuse for both the prefill placeholder padding
        # and the decode text-step fallback below.
        dtype = self._embedding_dtype
        tts_pad_embed = self._tts_pad_embed.to(device=input_ids.device, dtype=dtype).reshape(1, -1)

        if is_prefill:
            # Prefill (prompt embeddings)
            prompt_embeds_cpu = embed.get("prefill")
            # First prefill round: prompt_embeds_cpu is not yet populated.
            # Subsequent prefill rounds (multi-chunk): prompt_embeds_cpu is a Tensor stored by the first round.
            is_first_prefill = not isinstance(prompt_embeds_cpu, torch.Tensor) or prompt_embeds_cpu.ndim != 2
            if is_first_prefill:
                full_prompt_embeds, tailing_text_hidden, ref_code_len, ref_code = (
                    self._prompt_builder.build_prompt_embeds(task_type=task_type, info_dict=info_dict)
                )
                # Store full prompt embeddings on CPU (large, prefill-only).
                # tailing_text_hidden stays on GPU (gpu_resident_buffer_keys).
                prompt_embeds_cpu = full_prompt_embeds.detach().to("cpu").contiguous()
                info_update: OmniPayload = {
                    "embed": {"prefill": prompt_embeds_cpu},
                    "hidden_states": {"trailing_text": tailing_text_hidden.detach()},
                    "meta": {
                        "talker_text_offset": 0,
                        "codec_streaming": codec_streaming,
                    },
                }
                if isinstance(ref_code, torch.Tensor) and ref_code.numel() > 0:
                    info_update.setdefault("codes", {})["ref"] = ref_code.detach().to("cpu").contiguous()
                if ref_code_len is not None:
                    info_update["meta"]["ref_code_len"] = int(ref_code_len)
                # First prefill: source the slice offset from `_omni_num_computed_tokens`
                # so cache-recovery (prefill replay at a later offset) lands on the right
                # slice of the stored embeddings. Subsequent chunks below advance from
                # `talker_prefill_offset` written here.
                offset = max(0, int(info_dict.get("_omni_num_computed_tokens", 0) or 0))
            else:
                # Subsequent prefill chunk: slice from stored embeddings at running offset.
                offset = max(0, int(meta.get("talker_prefill_offset", 0) or 0))
                info_update = {"meta": {"codec_streaming": codec_streaming}}

            # Always return a span_len slice; if the scheduled placeholder is longer than what
            # the prompt actually fills, pad with tts_pad_embed (preserves placeholder/embedding alignment).
            s = max(0, min(offset, int(prompt_embeds_cpu.shape[0])))
            e = max(0, min(offset + span_len, int(prompt_embeds_cpu.shape[0])))
            take = prompt_embeds_cpu[s:e]
            if int(take.shape[0]) < span_len:
                pad_n = int(span_len - int(take.shape[0]))
                pad_rows = tts_pad_embed.reshape(1, -1).to("cpu").expand(pad_n, -1)
                take = torch.cat([take, pad_rows], dim=0)
            prompt_embeds = take.to(device=input_ids.device, dtype=dtype)
            info_update["meta"]["talker_prefill_offset"] = int(offset + span_len)

            # When inputs_embeds is set, token ids are ignored by the model but must stay in-vocab for vLLM bookkeeping.
            input_ids_out = input_ids.clone()
            input_ids_out[:] = int(self.talker_config.codec_pad_id)

            zeros = torch.zeros(
                (prompt_embeds.shape[0], int(self.talker_config.num_code_groups)),
                device=input_ids.device,
                dtype=torch.long,
            )
            info_update.setdefault("codes", {})["audio"] = zeros
            return input_ids_out, prompt_embeds, info_update

        if span_len > 1:
            inputs_embeds_out = (
                self.embed_input_ids(input_ids.reshape(-1, 1).to(torch.long))
                .to(device=input_ids.device, dtype=dtype)
                .reshape(span_len, -1)
            )
            return input_ids, inputs_embeds_out, {"meta": {"codec_streaming": codec_streaming}}

        # Decode: span_len == 1
        # Pop one text-step vector from tailing_text_hidden queue.
        # ``tts_pad_embed`` was materialized above from :attr:`_tts_pad_embed`
        # (request-independent buffer) — no per-request fetch needed.

        tail = hs.get("trailing_text")
        text_offset = max(0, int(meta.get("talker_text_offset", 0) or 0))
        trailing_text_update = None
        if isinstance(tail, torch.Tensor) and tail.ndim == 2:
            tail_len = int(tail.shape[0])
            if text_offset < tail_len:
                text_step = (
                    tail[text_offset : text_offset + 1]
                    .to(
                        device=input_ids.device,
                        dtype=dtype,
                    )
                    .reshape(1, -1)
                )
                next_text_offset = text_offset + 1
                should_compact_tail = next_text_offset >= tail_len or (
                    next_text_offset >= _TRAILING_TEXT_COMPACT_MIN_FRAMES and next_text_offset * 2 >= tail_len
                )
                if should_compact_tail:
                    if next_text_offset >= tail_len:
                        trailing_text_update = torch.empty((0, tail.shape[1]), device=tail.device, dtype=tail.dtype)
                    else:
                        trailing_text_update = tail[next_text_offset:].contiguous()
                    next_text_offset = 0
            else:
                text_step = tts_pad_embed
                next_text_offset = 0
                if tail.numel() > 0:
                    trailing_text_update = torch.empty((0, tail.shape[1]), device=tail.device, dtype=tail.dtype)
        else:
            text_step = tts_pad_embed
            next_text_offset = text_offset

        last_hidden = hs.get("last")
        if isinstance(last_hidden, torch.Tensor):
            past_hidden = last_hidden.to(device=input_ids.device, dtype=dtype).reshape(1, -1)
        else:
            # Defensive: EOS step row is zeroed by the invalid-layer-0 mask and filtered downstream.
            past_hidden = torch.zeros_like(text_step)

        # Use OmniGPUModelRunner talker_mtp fast-path for residual codebooks and per-step inputs_embeds update.
        last_id_hidden = self.embed_input_ids(input_ids.reshape(1, 1).to(torch.long)).to(
            device=input_ids.device, dtype=dtype
        )
        inputs_embeds_out = last_id_hidden.reshape(1, -1)

        info_update = {
            "mtp_inputs": (past_hidden, text_step),
            "meta": {
                "talker_text_offset": int(next_text_offset),
                "codec_streaming": codec_streaming,
            },
        }
        if trailing_text_update is not None:
            info_update["hidden_states"] = {"trailing_text": trailing_text_update.detach()}
        return input_ids, inputs_embeds_out, info_update

    def preprocess_decode_batch(
        self,
        *,
        input_ids: torch.Tensor,
        req_infos: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        """Batch the decode-only preprocess path for Qwen3-TTS.

        This mirrors the scalar decode branch in ``preprocess()``, but performs
        the token embedding lookup once for the whole decode batch.
        """
        input_ids_flat = input_ids.reshape(-1)
        if int(input_ids_flat.numel()) != len(req_infos):
            raise ValueError(
                f"preprocess_decode_batch expected {len(req_infos)} input ids, got {int(input_ids_flat.numel())}"
            )

        device = input_ids_flat.device
        dtype = self._embedding_dtype
        # Request-independent constant (see :meth:`_init_runtime_buffers`) —
        # compute once for the batch instead of fetching it per request from
        # ``info_dict["embed"]["tts_pad"]``.
        tts_pad_embed = self._tts_pad_embed.to(device=device, dtype=dtype).reshape(1, -1)
        past_hidden_list: list[torch.Tensor] = []
        text_step_list: list[torch.Tensor] = []
        updates: list[dict[str, Any]] = []

        for info_dict in req_infos:
            additional_information = info_dict.get("additional_information")
            if isinstance(additional_information, dict):
                merged: dict[str, Any] = {k: v for k, v in info_dict.items() if k != "additional_information"}
                for k, v in additional_information.items():
                    merged.setdefault(k, v)
                info_dict = merged

            payload: OmniPayload = info_dict
            hs = payload.get("hidden_states", {})
            meta = payload.get("meta", {})

            if not _has_tts_text_conditioning(info_dict, hs):
                raise ValueError("Missing Qwen3-TTS text conditioning: provide `text` or precomputed text token ids.")

            task_type = (info_dict.get("task_type") or ["CustomVoice"])[0]
            codec_streaming_val = meta.get("codec_streaming")
            if isinstance(codec_streaming_val, list):
                codec_streaming_raw = codec_streaming_val[0] if codec_streaming_val else None
            else:
                codec_streaming_raw = codec_streaming_val
            if isinstance(codec_streaming_raw, bool):
                codec_streaming = codec_streaming_raw
            else:
                codec_streaming = task_type == "Base"

            tail = hs.get("trailing_text")
            text_offset = max(0, int(meta.get("talker_text_offset", 0) or 0))
            trailing_text_update = None
            if isinstance(tail, torch.Tensor) and tail.ndim == 2:
                tail_len = int(tail.shape[0])
                if text_offset < tail_len:
                    text_step = tail[text_offset : text_offset + 1].to(device=device, dtype=dtype).reshape(1, -1)
                    next_text_offset = text_offset + 1
                    should_compact_tail = next_text_offset >= tail_len or (
                        next_text_offset >= _TRAILING_TEXT_COMPACT_MIN_FRAMES and next_text_offset * 2 >= tail_len
                    )
                    if should_compact_tail:
                        if next_text_offset >= tail_len:
                            trailing_text_update = torch.empty((0, tail.shape[1]), device=tail.device, dtype=tail.dtype)
                        else:
                            trailing_text_update = tail[next_text_offset:].contiguous()
                        next_text_offset = 0
                else:
                    text_step = tts_pad_embed
                    next_text_offset = 0
                    if tail.numel() > 0:
                        trailing_text_update = torch.empty((0, tail.shape[1]), device=tail.device, dtype=tail.dtype)
            else:
                text_step = tts_pad_embed
                next_text_offset = text_offset

            last_hidden = hs.get("last")
            if not isinstance(last_hidden, torch.Tensor):
                raise RuntimeError("Missing hidden_states['last'] in additional_information; postprocess must run.")
            past_hidden_list.append(last_hidden.to(device=device, dtype=dtype).reshape(1, -1))
            text_step_list.append(text_step)

            info_update: dict[str, Any] = {
                "meta": {
                    "talker_text_offset": int(next_text_offset),
                    "codec_streaming": codec_streaming,
                },
            }
            if trailing_text_update is not None:
                info_update["hidden_states"] = {"trailing_text": trailing_text_update.detach()}
            updates.append(info_update)

        inputs_embeds_out = self.embed_input_ids(input_ids_flat.reshape(-1, 1).to(torch.long)).to(
            device=device,
            dtype=dtype,
        )
        inputs_embeds_out = inputs_embeds_out.reshape(len(req_infos), -1)
        return (
            input_ids_flat,
            inputs_embeds_out,
            torch.cat(past_hidden_list, dim=0),
            torch.cat(text_step_list, dim=0),
            updates,
        )

    def postprocess(self, hidden_states: torch.Tensor, **_: Any) -> dict[str, Any]:
        # Keep the last token hidden for the next decode step's code predictor.
        # Stays on GPU - gpu_resident_buffer_keys avoids the CPU round-trip.
        if hidden_states.numel() == 0:
            return {}
        last = hidden_states[-1, :].detach()
        return {"hidden_states": {"last": last}}

    @torch.inference_mode()
    def preprocess_batch(
        self,
        *,
        req_ids: list[str],
        model_intermediate_buffer: dict[str, dict[str, Any]],
        device: torch.device,
    ) -> None:
        """Delegate batched preprocess to :class:`Qwen3TTSPromptEmbedsBuilder`."""
        self._prompt_builder.preprocess_batch(
            req_ids=req_ids,
            model_intermediate_buffer=model_intermediate_buffer,
            device=device,
        )

    def _encode_ref_audio_batch(self, wavs: list[np.ndarray], sr: int, *, device: torch.device) -> list[torch.Tensor]:
        fe = self._encoder_feature_extractor
        target_sr = int(fe.sampling_rate)
        if int(sr) != target_sr:
            resampler = AudioResampler(target_sr=target_sr)
            wavs = [resampler.resample(w.astype(np.float32), orig_sr=int(sr)) for w in wavs]

        inputs = fe(
            raw_audio=wavs,
            sampling_rate=target_sr,
            return_tensors="pt",
        )
        inputs = inputs.to(device).to(torch.bfloat16)

        input_values = inputs["input_values"].squeeze(1)
        padding_mask = inputs["padding_mask"].squeeze(1)

        with torch.inference_mode():
            encoded = self.encoder.encode(
                input_values=input_values.unsqueeze(1),
                return_dict=True,
            )

        audio_codes = encoded.audio_codes[:, : self._encoder_valid_num_quantizers]
        downsample = self._encoder_downsample_rate
        return [
            code[..., : -(-mask.sum() // downsample)].transpose(0, 1).to(device=device, dtype=torch.long)
            for code, mask in zip(audio_codes, padding_mask)
        ]

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
            # speaker_encoder module is already constructed in __init__; here we
            # only copy checkpoint tensors into its existing parameters.
            loaded |= loader.load_weights(speaker_weights, mapper=self.hf_to_vllm_mapper)
        else:
            # Some checkpoints do not include speaker_encoder weights; keep the
            # eagerly initialized module and satisfy the strict loader check.
            loaded |= {name for name, _ in self.named_parameters() if name.startswith("speaker_encoder.")}
        # Load speech tokenizer encoder weights from speech_tokenizer/
        # subfolder.  Skip decoder weights — the Talker only uses the
        # encoder for ref_audio encoding.
        model_loader = DefaultModelLoader(self.vllm_config.load_config)
        source = DefaultModelLoader.Source(
            model_or_path=self.model_path,
            revision=self.vllm_config.model_config.revision,
            subfolder="speech_tokenizer",
        )
        subfolder_weights = model_loader._get_weights_iterator(source)
        enc_loaded = AutoWeightsLoader(
            self,
            skip_prefixes=["decoder."],
        ).load_weights(subfolder_weights)
        loaded |= enc_loaded

        # AutoWeightsLoader only loads parameters; the encoder's VQ
        # codebook state (embed, embed_sum, cluster_usage, initialized)
        # are registered as buffers. Load them from the checkpoint
        # directly so the quantizer produces correct codes.
        encoder_buffers = dict(self.encoder.named_buffers())
        source2 = DefaultModelLoader.Source(
            model_or_path=self.model_path,
            revision=self.vllm_config.model_config.revision,
            subfolder="speech_tokenizer",
        )
        for name, tensor in model_loader._get_weights_iterator(source2):
            if not name.startswith("encoder."):
                continue
            buf_name = name[len("encoder.") :]
            if buf_name in encoder_buffers:
                encoder_buffers[buf_name].copy_(tensor)
                loaded.add(name)

        device = self.vllm_config.device_config.device
        self.encoder.to(device=device, dtype=torch.bfloat16)

        self._init_runtime_buffers()

        logger.info("Loaded %d weights for Qwen3TTSTalkerForConditionalGeneration", len(loaded))
        self._build_stacked_codec_embed()
        return loaded

    def _build_stacked_codec_embed(self) -> None:
        embeds = self.code_predictor.get_input_embeddings()
        if not embeds:
            return
        w = embeds[0].weight
        self._stacked_codec_embed = torch.stack([e.weight.detach() for e in embeds], dim=0).to(
            device=w.device, dtype=w.dtype
        )

    @torch.no_grad()
    def _init_runtime_buffers(self) -> None:
        """Populate request-independent runtime buffers from frozen weights.

        Currently this only computes :attr:`_tts_pad_embed`
        (``text_proj(text_emb(tts_pad_token_id))``), which the prefill
        prompt builder and every decode step mix into the input embedding
        in place of an actual text token. The value depends only on the
        ``text_embedding`` / ``text_projection`` weights and is therefore
        the same for every request — computing it once here avoids
        recomputing (and round-tripping through ``info_dict``) on each
        forward call.
        """
        device = next(self.parameters()).device
        pad_ids = torch.tensor([[int(self.config.tts_pad_token_id)]], device=device, dtype=torch.long)
        pad_proj = self.text_projection(self.text_embedding(pad_ids)).reshape(1, -1)
        self._tts_pad_embed.copy_(pad_proj.to(device=self._tts_pad_embed.device, dtype=self._tts_pad_embed.dtype))

    # -------------------- GPU-side MTP fast-path --------------------

    @torch.inference_mode()
    def talker_mtp(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        text_step: torch.Tensor,
        do_sample: bool | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        generator: torch.Generator | None = None,
        generators: Sequence[torch.Generator | None] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """GPU fast-path used by OmniGPUModelRunner to predict residual codebooks (1..Q-1).
        Returns (inputs_embeds, audio_codes) for the current step."""
        bsz = int(input_ids.shape[0])
        q = int(self.talker_config.num_code_groups)
        dev = input_embeds.device
        dtype = self._embedding_dtype

        input_ids = input_ids.reshape(bsz, 1).to(dtype=torch.long, device=dev)
        last_id_hidden = input_embeds.reshape(bsz, 1, -1).to(dtype=dtype, device=dev)
        past_hidden = last_talker_hidden.reshape(bsz, 1, -1).to(dtype=dtype, device=dev)
        text_step = text_step.reshape(bsz, 1, -1).to(dtype=dtype, device=dev)

        # Residual predictor runs fixed-length (Q-1) steps via the vLLM-native code_predictor.
        max_steps = q - 1
        if max_steps <= 0:
            audio_codes = input_ids.reshape(bsz, 1)
            return (last_id_hidden + text_step).reshape(bsz, -1), audio_codes

        subtalker_params = self._subtalker_sampling_params
        if do_sample is None:
            do_sample = bool(subtalker_params.get("do_sample", True))
        if temperature is None:
            temperature = float(subtalker_params.get("temperature", 0.9))
        if top_k is None:
            top_k = int(subtalker_params.get("top_k", 50))
        if top_p is None:
            top_p = float(subtalker_params.get("top_p", 1.0))

        audio_codes = self.code_predictor(
            layer0_code=input_ids.reshape(bsz, 1),
            layer0_embed=last_id_hidden,
            last_talker_hidden=past_hidden,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
            generators=generators,
        )  # [B, Q]

        # Map invalid layer-0 ids (e.g. EOS) to PAD=0 so SpeechTokenizer sees only real codes.
        layer0 = audio_codes[:, :1]
        invalid0 = (layer0 < 0) | (layer0 >= int(self._codebook_vocab_size))
        audio_codes = torch.where(invalid0.expand_as(audio_codes), torch.zeros_like(audio_codes), audio_codes)

        # Single gather over stacked [Q-1, V, H] replaces Q-1 serial embedding kernels.
        residual_ids_t = audio_codes[:, 1:]
        if self._stacked_codec_embed is None:
            self._build_stacked_codec_embed()
        embed_weight = self._stacked_codec_embed.to(device=dev)
        row_idx = torch.arange(max_steps, device=dev).unsqueeze(0).expand(bsz, -1)
        gathered = embed_weight[row_idx, residual_ids_t]
        summed = (last_id_hidden.squeeze(1) + gathered.sum(dim=1)).unsqueeze(1)
        inputs_embeds_out = (summed + text_step).reshape(bsz, -1)
        return inputs_embeds_out, audio_codes.to(dtype=torch.long)
