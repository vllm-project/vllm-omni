# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""DuplexJev thinker (stage 0 of Sophon-Realtime): speech in, reply text out.

DuplexJev is its own model class built from shared vLLM pieces:

* audio tower: vLLM's ``Qwen3OmniMoeAudioEncoder`` (the Qwen3-ASR encoder),
  read after ``ln_post`` (its output projection is not part of DuplexJev);
* projector: stack 2 encoder frames -> RMSNorm -> linear -> SwiGLU -> RMSNorm ->
  linear (the Ultravox v0.5 layout, so checkpoints keep Ultravox weight names);
* language model: any vLLM causal LM (``DuplexJevForConditionalGeneration``,
  e.g. Qwen3) or Qwen3-VL (``DuplexJevVisionForConditionalGeneration`` in
  ``thinker_vision.py``, speech and images in one request).

Nothing here depends on vLLM's Ultravox model code; only its config class reads
the checkpoint's ``config.json`` (``model_type: ultravox``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer, BatchFeature, WhisperFeatureExtractor
from vllm.model_executor.layers.activation import MulAndSilu, get_act_fn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.qwen3_omni_moe_thinker import Qwen3OmniMoeAudioEncoder
from vllm.model_executor.models.utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.parse import MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdateDetails,
)
from vllm.utils.gpu_sync_debug import gpu_sync_allowed

from vllm_omni.model_executor.models.sophon_realtime.thinker_processing import (
    DuplexJevProcessor,
    qwen3_asr_output_lengths,
)
from vllm_omni.transformers_utils.configs.sophon_realtime import Qwen3ASREncoderConfig

AUDIO_TOKEN = "<|audio|>"
AUDIO_KEYS = ("audio_features", "audio_embeds", "audio_lens", "audio_token_len", "audio_num_chunks")
SAMPLE_RATE = 16000
CHUNK_S = 30  # encoder context per chunk (3000 mel frames)
MEL_CONTEXT = CHUNK_S * 100
MAX_AUDIO_CHUNKS = 4  # longest clip budgeted for profiling: 120 s
ENCODER_BATCH = 16  # chunks per encoder call


# ---------------------------------------------------------------- audio path (shared by both thinkers)
class DuplexJevAudioTower(Qwen3OmniMoeAudioEncoder):
    """Qwen3-ASR encoder called as ``(padded chunks, valid lengths) -> [B, N, C]``.
    DuplexJev's projector reads the encoder output after ``ln_post``, so the
    Qwen3-ASR output projection is not part of the model."""

    def __init__(self, config: Qwen3ASREncoderConfig, prefix: str = ""):
        super().__init__(config, prefix=prefix)
        self.proj1 = nn.Identity()
        self.act = nn.Identity()
        self.proj2 = nn.Identity()

    @property
    def max_context_length(self) -> int:
        return MEL_CONTEXT

    def forward(  # type: ignore[override]
        self, input_features: torch.Tensor, audio_lens: torch.Tensor
    ) -> torch.Tensor:
        """``input_features``: ``[B, mel_bins, T]`` (padded chunks);
        ``audio_lens``: valid mel frames per chunk. Returns ``[B, N, d_model]``
        with ``N`` the encoder length of ``T`` frames; frames past a chunk's
        valid length are zero."""
        batch = input_features.shape[0]
        with gpu_sync_allowed():
            lens = [int(n) for n in audio_lens.tolist()]
        packed = torch.cat([input_features[i, :, : lens[i]] for i in range(batch)], dim=-1)
        lens_t = torch.tensor(lens, dtype=torch.long, device=input_features.device)
        after_cnn = qwen3_asr_output_lengths(lens_t, self.n_window)
        hidden = super().forward(packed.to(self.dtype), lens_t, after_cnn)

        n = int(qwen3_asr_output_lengths(torch.tensor(input_features.shape[-1]), self.n_window))
        out = hidden.new_zeros(batch, n, hidden.shape[-1])
        with gpu_sync_allowed():
            sizes = after_cnn.tolist()
        for i, item in enumerate(hidden.split(sizes, dim=0)):
            m = min(item.shape[0], n)
            out[i, :m] = item[:m]
        return out


class DuplexJevProjector(nn.Module):
    """Encoder frames -> language-model embeddings. ``config`` is the
    checkpoint's audio config (``stack_factor``, ``projector_act``,
    ``projector_ln_mid``, ``hidden_size``, ``audio_config``, ``text_config``)."""

    def __init__(self, config, *, quant_config=None, prefix: str = ""):
        super().__init__()
        self.stack_factor = int(config.stack_factor)
        dim_in = config.audio_config.hidden_size * self.stack_factor
        dim_mid = config.hidden_size
        dim_out = config.text_config.hidden_size
        self.ln_pre = RMSNorm(dim_in)
        self.linear_1 = ReplicatedLinear(
            dim_in, dim_mid, bias=False, quant_config=quant_config, prefix=f"{prefix}.linear_1"
        )
        if config.projector_act == "swiglu":
            self.act: nn.Module = MulAndSilu()
            dim_mid //= 2
        else:
            self.act = get_act_fn(config.projector_act)
        self.linear_2 = ReplicatedLinear(
            dim_mid, dim_out, bias=False, quant_config=quant_config, prefix=f"{prefix}.linear_2"
        )
        ln_mid = bool(getattr(config, "projector_ln_mid", True))
        self.ln_mid: nn.Module = RMSNorm(dim_mid) if ln_mid else nn.Identity()
        self.ln_post: nn.Module = nn.Identity() if ln_mid else RMSNorm(dim_out)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        b, t, c = frames.shape
        pad = -t % self.stack_factor
        if pad:
            frames = F.pad(frames, (0, 0, 0, pad))
        x = frames.reshape(b, (t + pad) // self.stack_factor, c * self.stack_factor)
        x, _ = self.linear_1(self.ln_pre(x))
        x = self.ln_mid(self.act(x))
        x, _ = self.linear_2(x)
        return self.ln_post(x)


def _pad_and_concat(features) -> torch.Tensor:
    """``[B, C, T]`` chunks (a tensor, or nested lists of them) -> one
    ``[sum B, C, max T]`` tensor."""
    if isinstance(features, torch.Tensor):
        return features.reshape(-1, *features.shape[-2:])
    tensors = [_pad_and_concat(f) for f in features]
    longest = max(t.shape[-1] for t in tensors)
    return torch.cat([F.pad(t, (0, longest - t.shape[-1])) for t in tensors])


def _flat(x) -> torch.Tensor:
    """A per-chunk / per-item field (tensor or list of tensors) as one 1-D tensor."""
    if isinstance(x, torch.Tensor):
        return x.reshape(-1)
    return torch.cat([torch.as_tensor(t).reshape(-1) for t in x])


class DuplexJevAudioMixin:
    """The audio tower + projector and the audio part of ``embed_multimodal``."""

    audio_tower: DuplexJevAudioTower
    multi_modal_projector: DuplexJevProjector

    def _init_audio(self, audio_cfg, *, prefix: str, quant_config=None) -> None:
        self.audio_tower = DuplexJevAudioTower(audio_cfg.audio_config, prefix=maybe_prefix(prefix, "audio_tower"))
        self.multi_modal_projector = DuplexJevProjector(
            audio_cfg, quant_config=quant_config, prefix=maybe_prefix(prefix, "multi_modal_projector")
        )

    def _embed_audio(self, **kwargs: object) -> list[torch.Tensor]:
        """One ``[tokens, hidden]`` tensor per audio item."""
        if kwargs.get("audio_embeds") is not None:
            embeds = kwargs["audio_embeds"]
            return list(embeds) if isinstance(embeds, (list, tuple)) else list(embeds.unbind(0))
        if kwargs.get("audio_features") is None:
            return []
        features = _pad_and_concat(kwargs["audio_features"]).to(self.audio_tower.dtype)
        lens = _flat(kwargs["audio_lens"])
        token_len = _flat(kwargs["audio_token_len"]).to(features.device)
        num_chunks = _flat(kwargs["audio_num_chunks"])

        parts = []
        for start in range(0, features.shape[0], ENCODER_BATCH):
            end = start + ENCODER_BATCH
            frames = self.audio_tower(features[start:end], lens[start:end])
            parts.append(self.multi_modal_projector(frames.to(self.audio_tower.dtype)))
        embeds = torch.cat(parts)
        keep = torch.arange(embeds.shape[1], device=embeds.device)[None, :] < token_len[:, None]
        with gpu_sync_allowed():
            per_audio = [int(n.sum()) for n in token_len.split(num_chunks.tolist())]
            return list(embeds[keep].split(per_audio))

    def _load_audio_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """``audio_tower.*`` / ``multi_modal_projector.*`` checkpoint weights."""
        weights = list(weights)
        tower = [(k.removeprefix("audio_tower."), v) for k, v in weights if k.startswith("audio_tower.")]
        loaded = {f"audio_tower.{k}" for k in self.audio_tower.load_weights(tower)}
        params = dict(self.multi_modal_projector.named_parameters())
        for name, tensor in weights:
            key = name.removeprefix("multi_modal_projector.")
            if name.startswith("multi_modal_projector.") and key in params:
                param = params[key]
                getattr(param, "weight_loader", default_weight_loader)(param, tensor)
                loaded.add(name)
        prefixes = ("audio_tower.", "multi_modal_projector.")
        missing = [k for k, _ in self.named_parameters() if k.startswith(prefixes) and k not in loaded]
        if missing:
            raise ValueError(f"audio weights missing from the checkpoint: {missing[:5]}")
        return loaded


# ---------------------------------------------------------------- multimodal processing (audio part, shared)
@lru_cache(maxsize=8)
def load_audio_processor(
    feature_extractor_path: str, tokenizer_path: str, revision: str | None, stack_factor: int, full_length: bool
) -> DuplexJevProcessor:
    return DuplexJevProcessor(
        WhisperFeatureExtractor.from_pretrained(feature_extractor_path, revision=revision),
        AutoTokenizer.from_pretrained(tokenizer_path, revision=revision),
        stack_factor=stack_factor,
        encoder_ds_factor=8,
        full_length=full_length,
    )


def audio_tokens_per_chunk(stack_factor: int) -> int:
    return -(-CHUNK_S * 100 // (8 * stack_factor))


def process_audio(processor: DuplexJevProcessor, audios: Sequence[Any]) -> dict[str, torch.Tensor]:
    feats = processor(audio=list(audios), sampling_rate=SAMPLE_RATE, include_audio_num_chunks=True, return_tensors="pt")
    out = dict(feats)
    out["audio_features"] = out.pop("audio_values")
    for key in ("audio_batch_size", "audio_token_start_idx", "input_ids", "attention_mask"):
        out.pop(key, None)
    return out


def audio_fields_config(hf_inputs: Mapping[str, Any]) -> dict[str, MultiModalFieldConfig]:
    num_chunks = hf_inputs.get("audio_num_chunks", torch.zeros(0))
    return dict(
        # audio longer than one encoder context is split into chunks, so the
        # chunk dimension can be larger than the number of audio items
        audio_features=MultiModalFieldConfig.flat_from_sizes("audio", num_chunks),
        audio_token_len=MultiModalFieldConfig.flat_from_sizes("audio", num_chunks),
        audio_lens=MultiModalFieldConfig.flat_from_sizes("audio", num_chunks, keep_on_cpu=True),
        audio_num_chunks=MultiModalFieldConfig.batched("audio", keep_on_cpu=True),
        audio_embeds=MultiModalFieldConfig.batched("audio"),
    )


def audio_prompt_update(audio_token_id: int, out_mm_kwargs) -> PromptReplacement:
    def replacement(i: int):
        n = int(out_mm_kwargs["audio"][i]["audio_token_len"].data.sum())
        return PromptUpdateDetails.select_token_id([audio_token_id] * n, audio_token_id)

    return PromptReplacement(modality="audio", target=[audio_token_id], replacement=replacement)


# ---------------------------------------------------------------- text-only thinker
class DuplexJevProcessingInfo(BaseProcessingInfo):
    def get_audio_config(self):
        return self.get_hf_config()

    def get_audio_processor(self) -> DuplexJevProcessor:
        mc = self.ctx.model_config
        cfg = self.get_audio_config()
        return load_audio_processor(
            mc.model, mc.tokenizer, mc.revision, int(cfg.stack_factor), bool(getattr(cfg, "audio_full_length", False))
        )

    def get_audio_token_id(self) -> int:
        return int(self.get_audio_config().audio_token_index)

    def get_data_parser(self):
        return MultiModalDataParser(
            target_sr=SAMPLE_RATE,
            target_channels=1,
            expected_hidden_size=self._get_expected_hidden_size(),
            allow_missing_mm_embeddings=self.allow_missing_mm_embeddings,
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}

    def get_mm_max_tokens_per_item(self, seq_len: int, mm_counts: Mapping[str, int]) -> Mapping[str, int]:
        return {"audio": MAX_AUDIO_CHUNKS * audio_tokens_per_chunk(int(self.get_audio_config().stack_factor))}


class DuplexJevDummyInputsBuilder(BaseDummyInputsBuilder[DuplexJevProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return AUDIO_TOKEN * mm_counts.get("audio", 0)

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options):
        return {
            "audio": self._get_dummy_audios(
                length=SAMPLE_RATE * CHUNK_S * MAX_AUDIO_CHUNKS,
                num_audios=mm_counts.get("audio", 0),
                overrides=mm_options.get("audio") if mm_options else None,
            )
        }


class DuplexJevMultiModalProcessor(BaseMultiModalProcessor[DuplexJevProcessingInfo]):
    def _apply_hf_processor_main(self, mm_items, hf_kwargs) -> BatchFeature:
        out = BatchFeature({})
        if mm_items.get_all_counts().get("audio", 0) > 0:
            hf_data, _, passthrough = self._get_hf_mm_inputs(mm_items.select({"audio"}), hf_kwargs)
            audios = hf_data.get("audio") or hf_data.get("audios") or []
            if not isinstance(audios, (list, tuple)):
                audios = [audios]
            if audios:
                out.update(process_audio(self.info.get_audio_processor(), audios))
            out.update(passthrough)
        return out

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs) -> Mapping[str, MultiModalFieldConfig]:
        return audio_fields_config(hf_inputs)

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs) -> Sequence[Any]:
        return [audio_prompt_update(self.info.get_audio_token_id(), out_mm_kwargs)]


@MULTIMODAL_REGISTRY.register_processor(
    DuplexJevMultiModalProcessor,
    info=DuplexJevProcessingInfo,
    dummy_inputs=DuplexJevDummyInputsBuilder,
)
class DuplexJevForConditionalGeneration(nn.Module, DuplexJevAudioMixin, SupportsMultiModal, SupportsPP):
    """DuplexJev on a causal LM (e.g. Qwen3): speech in, text out."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return AUDIO_TOKEN
        raise ValueError("DuplexJev (text-only) accepts audio only")

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.configure_mm_token_handling(config.vocab_size, [config.audio_token_index])
        with self._mark_tower_model(vllm_config, "audio"):
            self._init_audio(config, prefix=prefix, quant_config=vllm_config.quant_config)
        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.wrapped_model_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model.", connector="multi_modal_projector.", tower_model="audio_tower."
        )

    def embed_multimodal(self, **kwargs: object):
        return self._embed_audio(**{k: v for k, v in kwargs.items() if k in AUDIO_KEYS})

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs: object):
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model.model(input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        audio: list[tuple[str, torch.Tensor]] = []

        def split(items):
            for name, tensor in items:
                if name.startswith(("audio_tower.", "multi_modal_projector.")):
                    audio.append((name, tensor))
                else:
                    yield name, tensor

        loaded = AutoWeightsLoader(self).load_weights(split(weights))
        return loaded | self._load_audio_weights(audio)


def pad_audio_embeddings_for_deepstack(embeddings, extra: int):
    """Qwen3-VL concatenates deep-stack features after the hidden size for
    vision tokens; audio tokens have none, so they are zero-padded."""
    return [F.pad(e, (0, extra)) for e in embeddings]
