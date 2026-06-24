# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage 1 (codec decoder) for higgs-audio v3.

Reuses higgs-audio-v2's RVQ + DAC codec decoder but loads weights from the
v3 checkpoint's bundled codec (``tied.embedding.modality_embeddings.0.model.*``
prefix) rather than from a standalone tokenizer repo.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_decoder import (
    HiggsAudioRVQ,
    build_boson_dac_decoder,
    build_higgs_audio_acoustic_decoder,
    load_higgs_audio_codec,
)
from vllm_omni.model_executor.models.higgs_audio_v3.configuration_higgs_audio_v3 import (
    HiggsAudioV3Config,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.platforms import current_omni_platform

__all__ = [
    "HiggsAudioV3Code2Wav",
    "HiggsAudioV3Code2WavForConditionalGeneration",
]

logger = init_logger(__name__)


# Prefix under which codec weights are stored in the v3 checkpoint.
_CODEC_PREFIX = "tied.embedding.modality_embeddings.0.model."


class HiggsAudioV3Code2Wav(nn.Module):
    """Stage-1 codec decoder for higgs-audio v3.

    The codec architecture is identical to v2 (HiggsAudioRVQ + fc2 +
    BosonDacDecoder); only the weight loading path differs. V3 bundles
    codec weights inside the main checkpoint under a known prefix, while
    v2 uses a standalone tokenizer repo.

    For backward compatibility and simplicity, we delegate to the v2
    ``load_higgs_audio_codec`` function when the audio tokenizer repo
    is available. When not (or as primary path), we extract codec weights
    from the v3 safetensors via the engine's weight iterator.
    """

    input_modalities = "audio"

    def __init__(
        self,
        config: HiggsAudioV3Config | None = None,
        *,
        vllm_config: VllmConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        if vllm_config is not None:
            hf_config = vllm_config.model_config.hf_config
            if isinstance(hf_config, HiggsAudioV3Config):
                self.config = hf_config
            else:
                self.config = HiggsAudioV3Config(**hf_config.to_dict())
            self._model_path: str | None = vllm_config.model_config.model
            self.vllm_config: VllmConfig | None = vllm_config
        else:
            if config is None:
                raise TypeError("HiggsAudioV3Code2Wav: provide either `config` or `vllm_config`.")
            self.config = config
            self._model_path = None
            self.vllm_config = None

        self.sample_rate: int = int(self.config.sample_rate)
        self.num_codebooks: int = int(self.config.num_codebooks)
        self.num_real_codes: int = int(self.config.num_real_codes)
        self.hop_length: int = 960

        # Engine-runner hooks
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True

        # Codec modules (populated by load_weights)
        self.quantizer: HiggsAudioRVQ | None = None
        self.fc2: nn.Linear | None = None
        self.acoustic_decoder: nn.Module | None = None
        self._loaded: bool = False
        # Do NOT eagerly load from standalone tokenizer here.
        # load_weights() will try bundled V3 codec first, then fall back.

    def _load_from_tokenizer_repo(self) -> None:
        """Load codec from the standalone higgs-audio-v2-tokenizer repo."""
        tokenizer_id = "bosonai/higgs-audio-v2-tokenizer"
        # Try to resolve from HF cache
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(repo_id=tokenizer_id, filename="config.json")
        if isinstance(cached, str) and os.path.isfile(cached):
            tokenizer_dir = os.path.dirname(cached)
        else:
            from huggingface_hub.constants import HF_HUB_CACHE

            safe = tokenizer_id.replace("/", "--")
            snapshots_dir = os.path.join(HF_HUB_CACHE, f"models--{safe}", "snapshots")
            if os.path.isdir(snapshots_dir):
                tokenizer_dir = None
                for rev in os.listdir(snapshots_dir):
                    candidate = os.path.join(snapshots_dir, rev)
                    if os.path.isfile(os.path.join(candidate, "config.json")):
                        tokenizer_dir = candidate
                        break
                if tokenizer_dir is None:
                    raise FileNotFoundError(f"No cached snapshot for {tokenizer_id}")
            else:
                raise FileNotFoundError(f"No cached snapshot for {tokenizer_id}")

        device = current_omni_platform.get_torch_device()
        quantizer, fc2, acoustic_decoder, _cfg = load_higgs_audio_codec(tokenizer_dir, device)
        self.quantizer = quantizer
        self.fc2 = fc2
        self.acoustic_decoder = acoustic_decoder
        self._loaded = True
        logger.info("Loaded HiggsAudioV3Code2Wav from standalone tokenizer repo.")

    def _resolve_model_dir(self) -> str | None:
        """Resolve ``self._model_path`` to a local checkpoint directory.

        Accepts an already-local path or an HF repo id (resolved via the
        local cache). Returns ``None`` when nothing is reachable on disk.
        """
        path = self._model_path
        if not path:
            return None
        if os.path.isdir(path):
            return path
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(repo_id=path, filename="config.json")
        if isinstance(cached, str) and os.path.isfile(cached):
            return os.path.dirname(cached)
        return None

    def _read_bundled_codec_state(self) -> dict[str, torch.Tensor]:
        """Read bundled codec tensors directly from the on-disk checkpoint.

        Returns the ``_CODEC_PREFIX``-stripped tensors, or an empty dict when
        the checkpoint cannot be located (so callers can fall back).
        """
        model_dir = self._resolve_model_dir()
        if model_dir is None:
            return {}
        import glob

        from safetensors import safe_open

        codec_state: dict[str, torch.Tensor] = {}
        for st in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
            with safe_open(st, framework="pt") as f:
                for key in f.keys():
                    if key.startswith(_CODEC_PREFIX):
                        codec_state[key[len(_CODEC_PREFIX) :]] = f.get_tensor(key)
        return codec_state

    def _ensure_codec_loaded(self) -> None:
        """Build the codec when vLLM's weight loader skipped ``load_weights``.

        ``load_format=dummy`` (and any loader that bypasses
        ``model.load_weights``) never hands us the checkpoint weight iterator,
        so the lazily-built codec submodules stay ``None`` and the warmup
        ``_dummy_run`` crashes in :meth:`decode_codes`. Mirror fish_speech's
        ``_ensure_codec_loaded`` and load the bundled V3 codec directly from
        the on-disk checkpoint, falling back to the standalone tokenizer repo.
        A no-op once ``load_weights`` has already populated the codec.
        """
        if self._loaded:
            return

        device: torch.device | None = None
        if self.vllm_config is not None:
            try:
                device = self.vllm_config.device_config.device
            except AttributeError:
                device = None

        codec_state = self._read_bundled_codec_state()
        if codec_state:
            self._load_from_bundled_state(codec_state, device)
            logger.info(
                "Lazily loaded HiggsAudioV3Code2Wav from bundled V3 checkpoint "
                "(%d codec keys) because the weight loader skipped load_weights "
                "(e.g. load_format=dummy).",
                len(codec_state),
            )
            return

        # No bundled checkpoint reachable on disk — fall back to the standalone
        # tokenizer repo (raises a clear error if that is missing too).
        self._load_from_tokenizer_repo()

    # ------------------------------------------------------------------ engine hooks
    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: Any, sampling_metadata: Any = None) -> None:
        return None

    # ------------------------------------------------------------------ load
    def load_weights(self, weights_or_model_dir, device: torch.device | None = None):
        if self._loaded:
            if not isinstance(weights_or_model_dir, (str, bytes, os.PathLike)):
                for _ in weights_or_model_dir:
                    pass
            return {name for name, _ in self.named_parameters()}

        if not isinstance(weights_or_model_dir, (str, bytes, os.PathLike)):
            # Engine path: try to extract V3 bundled codec weights first
            codec_state: dict[str, torch.Tensor] = {}
            for name, tensor in weights_or_model_dir:
                if name.startswith(_CODEC_PREFIX):
                    stripped = name[len(_CODEC_PREFIX) :]
                    codec_state[stripped] = tensor

            if codec_state:
                # Bundled keys found — load strictly, no fallback on error.
                self._load_from_bundled_state(codec_state, device)
                logger.info(
                    "Loaded HiggsAudioV3Code2Wav from bundled V3 checkpoint (%d codec keys consumed).",
                    len(codec_state),
                )
                return {name for name, _ in self.named_parameters()}

            # No bundled keys — fallback to standalone tokenizer repo
            if not self._loaded:
                try:
                    self._load_from_tokenizer_repo()
                except (FileNotFoundError, OSError) as exc2:
                    raise RuntimeError(
                        "HiggsAudioV3Code2Wav: could not load codec weights. "
                        "Neither V3 bundled codec keys (tied.embedding."
                        "modality_embeddings.0.model.*) were found in the "
                        "checkpoint, nor is bosonai/higgs-audio-v2-tokenizer "
                        f"cached in HF_HOME. Last error: {exc2}"
                    ) from exc2
            return {name for name, _ in self.named_parameters()}

        # File-based path
        if not self._loaded:
            self._load_from_tokenizer_repo()
        return None

    def _load_from_bundled_state(
        self,
        codec_state: dict[str, torch.Tensor],
        device: torch.device | None = None,
    ) -> None:
        """Load codec from V3 checkpoint's bundled weights.

        Keys arrive with the ``tied.embedding.modality_embeddings.0.model.``
        prefix already stripped. We remap to the same layout that
        ``load_higgs_audio_codec`` expects and build the modules.
        """
        from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_decoder import (
            _remap_boson_model_pth_state_dict,
        )

        if device is None:
            device = current_omni_platform.get_torch_device()

        # The bundled codec may use boson-ai key naming. Try remapping.
        needs_remap = any(k.startswith("quantizer.vq.layers.") for k in codec_state)
        if needs_remap:
            codec_state = _remap_boson_model_pth_state_dict(codec_state)

        # Discover parameters from the state dict
        proj_key = "quantizer.quantizers.0.project_out.weight"
        if proj_key not in codec_state:
            raise KeyError(
                f"Bundled codec state dict missing '{proj_key}'. "
                f"Available keys (first 10): {list(codec_state.keys())[:10]}"
            )
        hidden_size = codec_state[proj_key].shape[0]
        codebook_dim = 64
        codebook_size = 1024
        required_quantizers = self.num_codebooks

        # Build RVQ — require exactly num_codebooks quantizers
        quantizer = HiggsAudioRVQ(
            num_quantizers=required_quantizers,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            hidden_size=hidden_size,
        ).to(device)
        for i in range(required_quantizers):
            prefix = f"quantizer.quantizers.{i}"
            # Require codebook weight
            embed_key = None
            for key_suffix in [".codebook.embed", ".codebook.weight"]:
                candidate = f"{prefix}{key_suffix}"
                if candidate in codec_state:
                    embed_key = candidate
                    break
            if embed_key is None:
                raise KeyError(
                    f"Bundled codec missing codebook weight for quantizer {i}. "
                    f"Expected {prefix}.codebook.embed or {prefix}.codebook.weight"
                )
            quantizer.quantizers[i].codebook.weight.data.copy_(codec_state[embed_key].to(device))
            # Require projector weights
            proj_w = f"{prefix}.project_out.weight"
            proj_b = f"{prefix}.project_out.bias"
            if proj_w not in codec_state:
                raise KeyError(f"Bundled codec missing {proj_w}")
            if proj_b not in codec_state:
                raise KeyError(f"Bundled codec missing {proj_b}")
            quantizer.quantizers[i].project_out.weight.data.copy_(codec_state[proj_w].to(device))
            quantizer.quantizers[i].project_out.bias.data.copy_(codec_state[proj_b].to(device))

        # Build fc2
        fc2_w = codec_state.get("fc2.weight")
        fc2_b = codec_state.get("fc2.bias")
        if fc2_w is None or fc2_b is None:
            raise KeyError(
                "Bundled codec state dict missing fc2.weight and/or fc2.bias. "
                f"Available keys (first 10): {list(codec_state.keys())[:10]}"
            )
        fc2 = nn.Linear(fc2_w.shape[1], fc2_w.shape[0]).to(device)
        fc2.weight.data.copy_(fc2_w.to(device))
        fc2.bias.data.copy_(fc2_b.to(device))

        # Build acoustic decoder — detect layout from key names.
        # V3 checkpoint uses OmniVoice layout (block.N/conv1/conv2/snake1),
        # not boson-ai standalone layout (model.0/model.1).
        ad_keys = {
            k[len("acoustic_decoder.") :]: v for k, v in codec_state.items() if k.startswith("acoustic_decoder.")
        }
        is_boson_layout = any(k.startswith("model.") for k in ad_keys)
        if is_boson_layout:
            acoustic_decoder = build_boson_dac_decoder(device)
        else:
            # OmniVoice layout — need the v2 tokenizer config for DacConfig
            tokenizer_config = {
                "acoustic_model_config": {
                    "codebook_dim": 8,
                    "codebook_size": 1024,
                    "decoder_hidden_size": 1024,
                    "downsampling_ratios": [8, 5, 4, 2, 3],
                    "encoder_hidden_size": 64,
                    "hidden_size": 256,
                    "hop_length": 960,
                    "model_type": "dac",
                    "n_codebooks": 9,
                    "sampling_rate": 16000,
                    "upsampling_ratios": [8, 5, 4, 2, 3],
                }
            }
            acoustic_decoder = build_higgs_audio_acoustic_decoder(tokenizer_config, device)
        if not ad_keys:
            raise KeyError(
                "Bundled codec state dict has no acoustic_decoder.* keys. "
                f"Available keys (first 10): {list(codec_state.keys())[:10]}"
            )
        load_report = acoustic_decoder.load_state_dict(ad_keys, strict=False)
        if load_report.missing_keys:
            raise RuntimeError(
                f"Bundled DAC decoder missing {len(load_report.missing_keys)} keys: {load_report.missing_keys[:5]}..."
            )
        if load_report.unexpected_keys:
            raise RuntimeError(
                f"Bundled DAC decoder has {len(load_report.unexpected_keys)} unexpected keys: "
                f"{load_report.unexpected_keys[:5]}..."
            )
        acoustic_decoder.eval()

        # Verify required decoder-side codec keys were consumed. The V3
        # checkpoint also bundles encoder-side keys (acoustic_encoder.*,
        # encoder_semantic.*, decoder_semantic.*, semantic_model.*, fc.*, fc1.*)
        # that the DAC decoder does not need — those are allowed but ignored.
        consumed: set[str] = set()
        for i in range(required_quantizers):
            prefix = f"quantizer.quantizers.{i}"
            for suffix in [".codebook.embed", ".codebook.weight"]:
                if f"{prefix}{suffix}" in codec_state:
                    consumed.add(f"{prefix}{suffix}")
            consumed.add(f"{prefix}.project_out.weight")
            consumed.add(f"{prefix}.project_out.bias")
        consumed.add("fc2.weight")
        consumed.add("fc2.bias")
        for k in codec_state:
            if k.startswith("acoustic_decoder."):
                consumed.add(k)

        # Known unused prefixes: encoder-side modules and quantizer training
        # artifacts (cluster_size, embed_avg, inited, project_in) that are
        # bundled in the checkpoint but not needed for inference decoding.
        _KNOWN_UNUSED_PREFIXES = (
            "acoustic_encoder.",
            "encoder_semantic.",
            "decoder_semantic.",
            "semantic_model.",
            "fc.",
            "fc1.",
        )
        # Quantizer training-only suffixes (per-codebook)
        _QUANTIZER_TRAINING_SUFFIXES = (
            ".codebook.cluster_size",
            ".codebook.embed_avg",
            ".codebook.inited",
            ".project_in.weight",
            ".project_in.bias",
        )
        unconsumed = set()
        for k in codec_state:
            if k in consumed:
                continue
            if any(k.startswith(p) for p in _KNOWN_UNUSED_PREFIXES):
                continue
            if any(k.endswith(s) for s in _QUANTIZER_TRAINING_SUFFIXES):
                continue
            unconsumed.add(k)
        if unconsumed:
            raise RuntimeError(
                f"Bundled codec has {len(unconsumed)} unexpected decoder-side keys: {sorted(unconsumed)[:10]}..."
            )

        self.quantizer = quantizer
        self.fc2 = fc2
        self.acoustic_decoder = acoustic_decoder
        self._loaded = True

    # ------------------------------------------------------------------ decode
    @torch.inference_mode()
    def decode_codes(self, audio_codes: torch.Tensor) -> torch.Tensor:
        """Decode [B, num_codebooks=8, T] codes to PCM [B, 1, T*960]."""
        if not self._loaded:
            # load_weights may have been skipped (e.g. load_format=dummy);
            # build the codec lazily from the on-disk checkpoint.
            self._ensure_codec_loaded()

        codes = self._validate_codes(audio_codes)
        rvq_codes = codes.transpose(0, 1).long()
        quantized = self.quantizer.decode(rvq_codes)
        quantized = quantized.to(dtype=self.fc2.weight.dtype)
        quantized = self.fc2(quantized.transpose(1, 2)).transpose(1, 2)
        first_param = next(self.acoustic_decoder.parameters(), None)
        if first_param is not None and quantized.dtype != first_param.dtype:
            quantized = quantized.to(dtype=first_param.dtype)
        audio = self.acoustic_decoder(quantized)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        return audio.to(codes.device)

    @torch.inference_mode()
    def forward_chunk(
        self,
        audio_codes: torch.Tensor,
        *,
        left_context_size: int = 0,
        right_holdback_size: int = 0,
        hop_length: int | None = None,
    ) -> torch.Tensor:
        # ``left_context_size`` frames trimmed off the front (already-emitted
        # prefix). ``right_holdback_size`` frames trimmed off the end
        # (deferred to the next chunk so the codec has future context for
        # the emit region's trailing samples).
        hop = int(hop_length) if hop_length is not None else self.hop_length
        pcm = self.decode_codes(audio_codes)
        right_trim = right_holdback_size * hop
        left_trim = left_context_size * hop
        if right_trim > 0:
            if pcm.shape[-1] <= right_trim:
                return pcm[..., :0]
            pcm = pcm[..., :-right_trim]
        if left_trim == 0:
            return pcm
        if pcm.shape[-1] <= left_trim:
            return pcm[..., :0]
        return pcm[..., left_trim:]

    # ------------------------------------------------------------------ runtime forward
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | OmniOutput:
        if isinstance(input_ids, torch.Tensor) and input_ids.ndim == 3:
            return self.decode_codes(input_ids)

        sr_val = int(self.sample_rate)
        sr_tensor = torch.tensor(sr_val, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)

        if input_ids is None or input_ids.numel() == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty], "sr": [sr_tensor]},
            )

        ids = input_ids.reshape(-1).to(dtype=torch.long)
        request_ids_list = self._split_request_ids(ids, kwargs.get("seq_token_counts"))

        left_context_size = [0] * len(request_ids_list)
        right_holdback_size = [0] * len(request_ids_list)
        if runtime_additional_information is not None:
            for i, info in enumerate(runtime_additional_information):
                if i >= len(left_context_size):
                    break
                meta = info.get("meta", {}) if isinstance(info, dict) else {}
                if "left_context_size" in meta:
                    left_context_size[i] = int(meta["left_context_size"])
                if "right_holdback_size" in meta:
                    right_holdback_size[i] = int(meta["right_holdback_size"])

        wavs: list[torch.Tensor] = []
        for i, req_ids in enumerate(request_ids_list):
            n = int(req_ids.numel())
            if n == 0:
                wavs.append(empty)
                continue
            if n % self.num_codebooks != 0:
                logger.warning(
                    "HiggsAudioV3Code2Wav: flat code length %d not divisible by %d",
                    n,
                    self.num_codebooks,
                )
                wavs.append(empty)
                continue
            frames = n // self.num_codebooks
            codes_qf = req_ids.reshape(self.num_codebooks, frames)
            codes_bqf = codes_qf.unsqueeze(0)
            try:
                pcm = self.forward_chunk(
                    codes_bqf,
                    left_context_size=left_context_size[i],
                    right_holdback_size=right_holdback_size[i],
                    hop_length=self.hop_length,
                )
            except ValueError as exc:
                logger.warning("HiggsAudioV3Code2Wav: decode skipped (%s)", exc)
                wavs.append(empty)
                continue
            wavs.append(pcm.squeeze(0).squeeze(0).to(torch.float32).cpu())

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": wavs,
                "sr": [sr_tensor] * len(wavs),
            },
        )

    # ------------------------------------------------------------------ helpers
    def _validate_codes(self, audio_codes: torch.Tensor) -> torch.Tensor:
        if audio_codes.ndim != 3:
            raise ValueError(
                f"audio_codes must have shape [B, {self.num_codebooks}, T]; got {tuple(audio_codes.shape)}"
            )
        if int(audio_codes.shape[1]) != self.num_codebooks:
            raise ValueError(f"dim 1 must equal num_codebooks={self.num_codebooks}; got {int(audio_codes.shape[1])}")
        # ``.item()`` triggers a GPU->CPU sync and is illegal inside CUDA-graph
        # capture (the dummy run before warmup hits this path). Skip the
        # range check while a graph is being captured; the per-step validator
        # still fires at real inference time.
        if audio_codes.numel() > 0 and not torch.cuda.is_current_stream_capturing():
            max_val = int(audio_codes.max().item())
            min_val = int(audio_codes.min().item())
            if max_val >= self.num_real_codes or min_val < 0:
                raise ValueError(
                    f"audio_codes out of range: min={min_val}, max={max_val}; expected [0, {self.num_real_codes - 1}]"
                )
        return audio_codes

    @staticmethod
    def _split_request_ids(ids: torch.Tensor, seq_token_counts: list[int] | None = None) -> list[torch.Tensor]:
        if seq_token_counts is not None and len(seq_token_counts) > 1:
            boundaries = [0]
            for count in seq_token_counts:
                boundaries.append(boundaries[-1] + int(count))
            n = int(ids.numel())
            return [ids[boundaries[i] : min(boundaries[i + 1], n)] for i in range(len(seq_token_counts))]
        return [ids]


HiggsAudioV3Code2WavForConditionalGeneration = HiggsAudioV3Code2Wav
