# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""MOSS-TTS Stage-1 codec decoder: RVQ codes → 24 kHz waveform."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.model_executor.models.moss_tts.audio_tokenizer import (
    MossAudioTokenizerConfig,
    MossAudioTokenizerModel,
)
from vllm_omni.model_executor.models.moss_tts.moss_codec_cudagraph import (
    MossTTSCUDAGraphCodecWrapper,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


class MossTTSCodecDecoder(nn.Module):
    """Stage-1 decoder for all MOSS-TTS variants.

    Consumes ``(NQ, T)`` audio VQ codes emitted by Stage 0 and decodes them
    to a 24 kHz mono waveform using the vendored
    ``MossAudioTokenizerModel``.

    All five variants share the same codec checkpoint
    ``OpenMOSS-Team/MOSS-Audio-Tokenizer``.  The number of quantizers
    (``n_vq``) is read from ``hf_config`` at construction time and fixed for
    the lifetime of the instance; the same checkpoint can be configured as
    ``n_vq=32`` (MOSS-TTS) or ``n_vq=16`` (all other variants) without
    swapping weights.

    The codec checkpoint path comes from
    ``vllm_config.model_config.hf_config.codec_model_name_or_path``.
    """

    input_modalities = "audio"

    have_multimodal_outputs: bool = True
    has_preprocess: bool = False
    has_postprocess: bool = False
    enable_update_additional_information: bool = True
    requires_raw_input_tokens: bool = True

    _OUTPUT_SAMPLE_RATE: int = 24_000

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config

        cfg = vllm_config.model_config.hf_config
        self._n_vq: int = int(getattr(cfg, "n_vq", getattr(cfg, "rvq", 16)))
        self._codec_path: str = str(
            getattr(
                cfg,
                "codec_model_name_or_path",
                getattr(cfg, "audio_tokenizer_name_or_path", "OpenMOSS-Team/MOSS-Audio-Tokenizer"),
            )
        )

        # Resolved at load_weights() time, once the codec checkpoint's own
        # config (sampling rate, channel count) is known.
        self._codec: MossAudioTokenizerModel | None = None
        self._cuda_graph_wrapper: MossTTSCUDAGraphCodecWrapper | None = None
        self._n_channels: int = 1
        self._sr_tensor = torch.tensor(self._OUTPUT_SAMPLE_RATE, dtype=torch.int32)

    # ------------------------------------------------------------------
    # vLLM-Omni stubs (codec has no AR loop)
    # ------------------------------------------------------------------

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> None:
        return None

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        """Decode audio VQ codes to waveform.

        Stage 0 emits flat codebook-major ``[NQ * T_chunk]`` audio codes. The
        chunk transfer adapter assigns those to ``request.prompt_token_ids``,
        which arrives here as ``input_ids`` concatenated across all requests.
        Per-request slice boundaries are computed from
        ``kwargs["num_scheduled_tokens"]`` (a list of token counts, one per
        request).  ``runtime_additional_information`` carries per-request
        metadata such as ``left_context_size``.

        Returns
        -------
        OmniOutput with:
          multimodal_outputs["model_outputs"] — list of (T_wav,) float32 tensors
          multimodal_outputs["sr"]            — list of scalar int32 tensors
        """
        sr_tensor = self._sr_tensor
        empty = torch.zeros((0,), dtype=torch.float32)
        info_list: list[dict[str, Any]] = runtime_additional_information or [{}]
        num_req = max(len(info_list), 1)

        if self._codec is None:
            logger.warning("MossTTSCodecDecoder called before load_weights(); returning silence.")
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": [empty] * num_req,
                    "sr": [sr_tensor] * num_req,
                },
            )

        audios: list[torch.Tensor] = [empty] * num_req
        srs: list[torch.Tensor] = [sr_tensor] * num_req
        device = next(self._codec.parameters()).device

        if input_ids is None or input_ids.numel() == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": audios, "sr": srs},
            )

        # ``input_ids`` is concatenated across all requests; split by
        # query_start_loc-style offsets from ``num_scheduled_tokens`` in
        # kwargs if available, else assume one request.
        ids_flat = input_ids.reshape(-1).to(dtype=torch.long)
        num_scheduled_tokens = kwargs.get("num_scheduled_tokens")
        if isinstance(num_scheduled_tokens, (list, tuple)) and len(num_scheduled_tokens) == num_req:
            offsets = [0]
            for n in num_scheduled_tokens:
                offsets.append(offsets[-1] + int(n))
        else:
            offsets = [0, int(ids_flat.shape[0])]

        for i, info in enumerate(info_list):
            if i + 1 >= len(offsets):
                break
            seg = ids_flat[offsets[i] : offsets[i + 1]]
            if seg.numel() == 0:
                continue
            if seg.numel() % self._n_vq != 0:
                logger.warning(
                    "MossTTS codec input length %d not divisible by n_vq %d; skipping.",
                    int(seg.numel()),
                    self._n_vq,
                )
                continue
            t_chunk = int(seg.numel() // self._n_vq)
            codes_nq_t = seg.reshape(self._n_vq, t_chunk).to(device=device)
            # Clamp out-of-range codes: the talker uses ``audio_pad_code``
            # (= ``codebook_size``) for delay-pattern padding.  The stage input
            # processor de-delays and drops pad rows before forwarding here, but
            # clamp as a defensive guard against any edge-case leakage.
            codebook_size = self._codec.config.codebook_size
            codes_nq_t = codes_nq_t.clamp_(0, int(codebook_size) - 1)

            meta = (info.get("meta", {}) if isinstance(info, dict) else {}) or {}
            left_ctx = self._runtime_value(info, meta, "left_context_size", 0)
            if isinstance(left_ctx, (list, tuple)):
                left_ctx = int(left_ctx[0]) if left_ctx else 0
            elif isinstance(left_ctx, torch.Tensor):
                left_ctx = int(left_ctx.reshape(-1)[0].item()) if left_ctx.numel() else 0
            left_ctx = int(left_ctx)

            if self._cuda_graph_wrapper is not None:
                out = self._cuda_graph_wrapper.decode(codes_nq_t)
            else:
                out = self._codec.batch_decode(codes_list=[codes_nq_t], num_quantizers=self._n_vq)

            if out.audio is None:
                continue

            # ``out.audio`` is ``(1, C, T)``; keep the channel axis for
            # stereo codecs (Local-v1.5) and flatten to ``(T,)`` for mono
            # ones (Delay/Realtime) to preserve their existing output shape.
            wav = out.audio[0].to(dtype=torch.float32).cpu()
            if out.audio_lengths is not None:
                wav = wav[..., : int(out.audio_lengths[0].item())]

            # Trim left-context samples (per-channel sample axis, so the
            # trim amount is identical for mono and interleaved-stereo).
            if left_ctx > 0:
                trim = min(left_ctx * self._codec.downsample_rate, wav.shape[-1])
                if trim < left_ctx * self._codec.downsample_rate:
                    logger.warning(
                        "left_ctx trim (%d samples) exceeds wav length (%d); returning empty audio.",
                        left_ctx * self._codec.downsample_rate,
                        wav.shape[-1],
                    )
                wav = wav[..., trim:]

            audios[i] = wav.reshape(-1) if self._n_channels == 1 else wav

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audios, "sr": srs},
        )

    @staticmethod
    def _runtime_value(info: Any, meta: dict[str, Any], name: str, default: Any = None) -> Any:
        if name in meta:
            return meta[name]
        if isinstance(info, dict) and name in info:
            return info[name]
        return default

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Drain the Stage-0 weights iterator, then load the codec from its own checkpoint.

        The codec lives in a separate HuggingFace repo
        (``OpenMOSS-Team/MOSS-Audio-Tokenizer``) and is loaded independently
        of the talker weights.
        """
        # Drain the incoming weights iterator — all Stage-0 weights are
        # irrelevant to this stage.
        for _ in weights:
            pass

        codec_path = self._codec_path
        logger.info("Loading MOSS Audio Tokenizer from %s", codec_path)

        codec_cfg, codec = self._build_codec(codec_path)

        model_loader = DefaultModelLoader(self.vllm_config.load_config)
        source = DefaultModelLoader.Source(
            model_or_path=codec_path,
            revision=None,
            subfolder=None,
        )
        codec_weights = model_loader._get_weights_iterator(source)
        params_dict = dict(codec.named_parameters())

        # Upstream MossAudioTokenizer uses different submodule names than the
        # vendored re-implementation in ``audio_tokenizer.py``. Without this
        # remap only ~half the codec parameters load (codebooks + WN convs)
        # and the rest stay at their random init, which produces noise that
        # sounds correct in duration but is structurally garbage.
        _SUFFIX_REMAP: list[tuple[str, str]] = [
            # v1 (MOSS-Audio-Tokenizer) naming.
            (".self_attn.in_projs.0.", ".attn.in_proj."),
            (".self_attn.out_projs.0.", ".attn.out_proj."),
            (".linear1.", ".ff1."),
            (".linear2.", ".ff2."),
            # v2 (MOSS-Audio-Tokenizer-v2) uses different submodule names for
            # the same attention/FFN sublayers — no trailing "s"/index on
            # in_proj/out_proj, and an `ffn.{0,2}` Sequential instead of
            # separate linear1/linear2 attributes.
            (".self_attn.in_proj.", ".attn.in_proj."),
            (".self_attn.out_proj.", ".attn.out_proj."),
            (".ffn.0.", ".ff1."),
            (".ffn.2.", ".ff2."),
            (".layer_scale_1.", ".ls1."),
            (".layer_scale_2.", ".ls2."),
            (".input_proj.", ".in_proj."),
            (".output_proj.", ".out_proj."),
        ]

        def _remap(name: str) -> str:
            for src, dst in _SUFFIX_REMAP:
                if src in name:
                    return name.replace(src, dst)
            return name

        loaded_total = 0
        skipped: list[str] = []
        for name, tensor in codec_weights:
            # Try direct name first (e.g. ``quantizer.input_proj.*`` exists
            # under the same name in both layouts), then the remap (transformer
            # submodules need ``.linear1.``→``.ff1.`` etc.).
            tgt = name if name in params_dict else _remap(name)
            if tgt in params_dict:
                default_weight_loader(params_dict[tgt], tensor)
                loaded_total += 1
            else:
                skipped.append(name)
        logger.info(
            "MOSS Audio Tokenizer weights: loaded=%d/%d skipped=%d (first skipped: %s)",
            loaded_total,
            len(params_dict),
            len(skipped),
            skipped[:3] if skipped else "none",
        )

        device = self.vllm_config.device_config.device
        codec.to(device=device, dtype=torch.float32)
        codec.eval()
        self._codec = codec
        self._n_channels = int(getattr(codec_cfg, "number_channels", 1) or 1)
        self._sr_tensor = torch.tensor(int(codec_cfg.sampling_rate), dtype=torch.int32)

        logger.info(
            "MOSS Audio Tokenizer loaded: sampling_rate=%d, n_vq=%d, n_channels=%d",
            codec_cfg.sampling_rate,
            codec_cfg.num_quantizers,
            self._n_channels,
        )

        self._maybe_enable_decoder_cudagraph(device)

        # vLLM's track_weights_loading() compares the returned set against
        # ``self.named_parameters()``. After ``self._codec = codec`` above,
        # those parameters are registered with the ``_codec.`` prefix, so
        # mirror that here.
        return {f"_codec.{name}" for name, _ in codec.named_parameters()}

    def _build_codec(self, codec_path: str) -> tuple[Any, nn.Module]:
        try:
            from transformers import AutoConfig, AutoModel

            codec_cfg = AutoConfig.from_pretrained(codec_path, trust_remote_code=True)
            codec = AutoModel.from_config(codec_cfg, trust_remote_code=True)
            logger.info("Using MOSS Audio Tokenizer remote-code classes from %s", codec_path)
            return codec_cfg, codec
        except Exception:
            logger.exception(
                "Failed to instantiate official MOSS Audio Tokenizer via HF remote code; "
                "falling back to vendored codec."
            )

        codec_cfg = MossAudioTokenizerConfig.from_pretrained(codec_path)
        codec = MossAudioTokenizerModel(codec_cfg)
        return codec_cfg, codec

    def _maybe_enable_decoder_cudagraph(self, device: torch.device) -> None:
        """Capture CUDA Graphs for the codec decoder if enforce_eager is False."""
        if getattr(self.vllm_config.model_config, "enforce_eager", True):
            return
        if self._codec is None:
            return

        # Read capture sizes from the connector's extra config (same convention
        # as Qwen3-TTS), falling back to a sensible default covering common
        # codec_chunk_frames values used in moss_tts.yaml.
        capture_sizes: list[int] = [4, 8, 16, 25, 32, 50, 64, 100, 128, 200, 256]
        model_cfg = getattr(self.vllm_config, "model_config", None)
        connector_cfg = getattr(model_cfg, "stage_connector_config", None)
        if isinstance(connector_cfg, dict):
            extra_cfg: dict | None = connector_cfg.get("extra", connector_cfg)
        else:
            extra_cfg = getattr(connector_cfg, "extra", None)
        if isinstance(extra_cfg, dict):
            raw = extra_cfg.get("decode_cudagraph_capture_sizes")
            if raw is not None:
                if isinstance(raw, (list, tuple)):
                    parsed = sorted({int(v) for v in raw if int(v) > 0})
                elif isinstance(raw, str):
                    parsed = sorted({int(v.strip()) for v in raw.split(",") if v.strip()})
                else:
                    parsed = [int(raw)]
                if parsed:
                    capture_sizes = parsed

        self._cuda_graph_wrapper = MossTTSCUDAGraphCodecWrapper(
            model=self._codec,
            capture_sizes=capture_sizes,
            num_quantizers=self._n_vq,
            enabled=True,
        )
        self._cuda_graph_wrapper.warmup(device)


__all__ = ["MossTTSCodecDecoder"]
