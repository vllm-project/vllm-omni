# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Complete-request audio decoding through Kimi's in-tree acoustic modules."""

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from vllm_omni.model_executor.models.output_templates import OmniOutput

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class KimiAudioDecoder(nn.Module):
    """Stage 1: semantic codes -> Flow Matching -> BigVGAN -> waveform.

    Networks are loaded once by load_weights, never by a request. A complete
    request is decoded before the next one starts, so the official mutable
    acoustic context is cleared between requests. Cross-step async_chunk
    decoding needs separate request-owned acoustic states and is not enabled.
    """

    have_multimodal_outputs = True
    has_preprocess = False
    has_postprocess = False
    enable_update_additional_information = True
    requires_raw_input_tokens = True

    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = "") -> None:
        super().__init__()
        if getattr(vllm_config.model_config, "async_chunk", False):
            raise ValueError("Kimi-Audio decoder currently requires complete requests; async_chunk is not implemented")
        if (
            vllm_config.parallel_config.tensor_parallel_size != 1
            or vllm_config.parallel_config.pipeline_parallel_size != 1
        ):
            raise ValueError("Kimi-Audio acoustic runtime requires TP=1 and PP=1")
        if vllm_config.quant_config is not None:
            raise ValueError("Kimi-Audio acoustic runtime does not support vLLM quantization")
        self.vllm_config = vllm_config
        config = vllm_config.model_config.hf_config
        self.audio_vocab_size = config.vocab_size - config.kimia_token_offset
        self.detokenizer = None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # This stage owns four acoustic files, not the root LLM checkpoint.
        # Do not iterate the lazy root-weight generator and load the 7B model.
        del weights
        if self.detokenizer is not None:
            return {name for name, _ in self.named_parameters()}

        from .detokenizer import PrefixStreamingFlowMatchingDetokenizer

        model_config = self.vllm_config.model_config
        model_path = Path(model_config.model)
        if not model_path.is_dir():
            from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

            model_path = Path(
                download_weights_from_hf_specific(
                    model_name_or_path=model_config.model,
                    revision=model_config.revision,
                    cache_dir=self.vllm_config.load_config.download_dir,
                    allow_patterns=[
                        "audio_detokenizer/config.yaml",
                        "audio_detokenizer/model.pt",
                        "vocoder/config.json",
                        "vocoder/model.pt",
                    ],
                    require_all=True,
                )
            )
        # Match official get_audio_detokenizer without its implicit current GPU.
        detokenizer = PrefixStreamingFlowMatchingDetokenizer.from_pretrained(
            vocoder_config=str(model_path / "vocoder/config.json"),
            vocoder_ckpt=str(model_path / "vocoder/model.pt"),
            fm_config=str(model_path / "audio_detokenizer/config.yaml"),
            fm_ckpt=str(model_path / "audio_detokenizer/model.pt"),
            device=self.vllm_config.device_config.device,
            use_cfg=False,
            look_ahead_tokens=12,
        )
        if int(detokenizer.vocoder.h["sampling_rate"]) != 24000:
            raise ValueError("Kimi-Audio acoustic runtime expects a 24000 Hz vocoder")
        # The acoustic wrappers are plain Python objects. Register their two
        # actual networks with nn.Module for parameter discovery and profiling.
        self.speech_model = detokenizer.semantic_fm.speech_model
        self.vocoder = detokenizer.vocoder.vocoder
        self.detokenizer = detokenizer
        return {name for name, _ in self.named_parameters()}

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        # GenerationModelRunner requires an embedding-shaped buffer; acoustic
        # inference consumes raw codec IDs instead of these placeholders.
        return torch.zeros((input_ids.numel(), 1), device=input_ids.device)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        if self.detokenizer is None:
            raise RuntimeError("Kimi-Audio acoustic weights must be loaded before decoding")
        counts = kwargs.get("seq_token_counts")
        if counts is None:
            raise ValueError("Kimi-Audio decoder requires per-request seq_token_counts")
        counts = [int(count) for count in counts]
        if any(count <= 0 for count in counts) or sum(counts) > input_ids.numel():
            raise ValueError("Invalid Kimi-Audio decoder request spans")
        infos = kwargs.get("model_intermediate_buffer") or runtime_additional_information or [{} for _ in counts]
        if len(infos) != len(counts):
            raise ValueError("Kimi-Audio decoder request payloads and spans must align")

        audios, start = [], 0
        for count, info in zip(counts, infos, strict=True):
            if not bool(info.get("meta", {}).get("finished", True)):
                raise ValueError("Kimi-Audio decoder requires a complete semantic sequence")
            codes = info.get("codes", {}).get("audio", input_ids.reshape(-1)[start : start + count])
            start += count
            codes = torch.as_tensor(codes, device=input_ids.device)
            if codes.ndim != 1 or (codes.numel() and codes.dtype not in (torch.int32, torch.int64)):
                raise ValueError("Kimi-Audio decoder expects one-dimensional integer codebook IDs")
            if torch.any(codes < 0) or torch.any(codes >= self.audio_vocab_size):
                raise ValueError("Kimi-Audio decoder expects raw codebook IDs without the LLM offset")
            if codes.numel() == 0:
                audios.append(torch.empty(0, dtype=torch.float32, device=input_ids.device))
                continue

            chunks = []
            self.detokenizer.clear_states()
            try:
                # Preserve official detokenize_audio as one cohesive flow:
                # 30 semantic tokens per call, fourfold upsampling, final flush.
                for offset in range(0, codes.numel(), 30):
                    chunks.append(
                        self.detokenizer.detokenize_streaming(
                            codes[offset : offset + 30].long().unsqueeze(0),
                            upsample_factor=4,
                            is_final=offset + 30 >= codes.numel(),
                        )
                    )
                audios.append(torch.cat(chunks, dim=-1).float().reshape(-1))
            finally:
                self.detokenizer.clear_states()
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": audios,
                "sr": [torch.tensor(24000, dtype=torch.int32) for _ in audios],
            },
        )
