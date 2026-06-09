"""Moshi Mimi Decoder — Stage 1: audio codes → waveform.

Loads the Mimi audio codec from the HF Moshi checkpoint and decodes
multi-codebook audio tokens to a 24kHz waveform.

Analogous to FishSpeechDACDecoder in the Fish Speech pipeline.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch
import torch.nn as nn
from vllm.config import CUDAGraphMode, VllmConfig

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .profiling import nvtx_annotate, nvtx_range

logger = logging.getLogger(__name__)

MIMI_SAMPLE_RATE = 24000
MIMI_FRAME_RATE = 12.5


def _inside_vllm_cuda_graph_forward() -> bool:
    try:
        from vllm.forward_context import get_forward_context, is_forward_context_available
    except Exception:
        return False
    if not is_forward_context_available():
        return False
    mode = getattr(get_forward_context(), "cudagraph_runtime_mode", None)
    return getattr(mode, "name", "NONE") != "NONE"


def _disable_vllm_cuda_graph_capture(vllm_config: VllmConfig) -> None:
    """Disable vLLM's model-level CUDA graph capture for this stage.

    The Mimi decoder manages its own stage-local CUDA graphs keyed by
    (device, num_codebooks, num_frames). vLLM's outer graph capture is
    incompatible with the stateful streaming convolutions.
    """
    compilation_config = getattr(vllm_config, "compilation_config", None)
    if compilation_config is None:
        return
    mode = getattr(compilation_config, "cudagraph_mode", None)
    if mode is None or mode == CUDAGraphMode.NONE:
        return

    compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    compilation_config.max_cudagraph_capture_size = 0
    compilation_config.cudagraph_capture_sizes = []
    logger.info(
        "Mimi decoder uses stage-local CUDA graphs; disabled vLLM model CUDA graph capture (was %s)",
        mode,
    )


@dataclass
class _DecoderCudaGraphState:
    graph: torch.cuda.CUDAGraph
    input_buffer: torch.Tensor
    output_buffer: torch.Tensor


class MoshiMimiDecoder(nn.Module):
    """Stage-1 Mimi decoder for Moshi (GenerationModelRunner).

    Consumes frame-aligned audio codes from input_ids and decodes waveform
    via the HF transformers MimiModel decoder.

    CUDA graphs are captured lazily per unique (device, num_codebooks,
    num_frames) shape on the first call with that shape, then replayed on
    subsequent calls.  This is most effective for streaming TTS where chunk
    sizes are fixed.
    """

    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True

        self._mimi: nn.Module | None = None
        hf_config = vllm_config.model_config.hf_config
        self._num_codebooks = getattr(hf_config, "num_codebooks", 8)
        self._output_sample_rate: int = MIMI_SAMPLE_RATE
        self._frame_rate: float = MIMI_FRAME_RATE

        self._cuda_graphs_requested = not bool(getattr(vllm_config.model_config, "enforce_eager", False))
        if self._cuda_graphs_requested:
            _disable_vllm_cuda_graph_capture(vllm_config)
        else:
            logger.info("Mimi decoder CUDA graphs disabled by enforce_eager=True")

        # Per-shape graph cache: key = (device, num_codebooks, num_frames)
        self._cuda_graphs: dict[tuple[torch.device, int, int], _DecoderCudaGraphState] = {}
        self._cuda_graph_failed_keys: set[tuple[torch.device, int, int]] = set()
        self._cuda_graph_capture_count = 0
        self._cuda_graph_replay_count = 0

    def _ensure_mimi_loaded(self) -> None:
        if self._mimi is not None:
            return

        from transformers import AutoModel, MoshiConfig

        config = MoshiConfig.from_pretrained(self.model_path)
        audio_config = config.audio_encoder_config
        mimi = AutoModel.from_config(audio_config)

        import os

        from safetensors import safe_open

        checkpoint_files = self._find_checkpoint_files()
        audio_state: dict[str, torch.Tensor] = {}

        hf_prefix = "audio_encoder."
        for ckpt_file in checkpoint_files:
            with safe_open(ckpt_file, framework="pt") as f:
                for key in f.keys():
                    if key.startswith(hf_prefix):
                        audio_state[key[len(hf_prefix) :]] = f.get_tensor(key)

        if not audio_state:
            mimi_file = os.path.join(self.model_path, "mimi.safetensors")
            if os.path.exists(mimi_file):
                with safe_open(mimi_file, framework="pt") as f:
                    for key in f.keys():
                        audio_state[key] = f.get_tensor(key)
                logger.info("Loaded %d Mimi weights from mimi.safetensors", len(audio_state))

        model_keys = set(mimi.state_dict().keys())
        if audio_state and not (set(audio_state.keys()) & model_keys):
            from .mimi_remap import remap_kyutai_mimi_keys

            logger.info("Remapping Kyutai Mimi keys to HF format...")
            audio_state = remap_kyutai_mimi_keys(audio_state)
            matched = set(audio_state.keys()) & model_keys
            logger.info("  Remapped: %d/%d keys match", len(matched), len(audio_state))

        missing, unexpected = mimi.load_state_dict(audio_state, strict=False)
        if missing:
            logger.warning("Mimi: missing keys: %s", missing[:10])
        if unexpected:
            logger.warning("Mimi: unexpected keys: %s", unexpected[:10])
        if not missing:
            logger.info("Mimi decoder: all keys loaded successfully")

        device = self.vllm_config.device_config.device
        mimi = mimi.to(device=device, dtype=torch.float32).eval()
        self._mimi = mimi

        if hasattr(audio_config, "sampling_rate"):
            self._output_sample_rate = audio_config.sampling_rate
        if hasattr(audio_config, "frame_rate"):
            self._frame_rate = float(audio_config.frame_rate)

        self._patch_mimi_conv1d_for_cuda_graphs()

        logger.info(
            "Mimi codec loaded from %s (device=%s, sample_rate=%d, frame_rate=%.3f, cuda_graphs=%s)",
            self.model_path,
            device,
            self._output_sample_rate,
            self._frame_rate,
            self._cuda_graphs_requested,
        )

    def _find_checkpoint_files(self) -> list[str]:
        import glob
        import os

        patterns = [
            os.path.join(self.model_path, "*.safetensors"),
            os.path.join(self.model_path, "model*.safetensors"),
        ]
        files = []
        for pat in patterns:
            files.extend(glob.glob(pat))

        if files:
            return sorted(set(files))

        try:
            from huggingface_hub import snapshot_download

            cache_dir = snapshot_download(self.model_path)
            files = glob.glob(os.path.join(cache_dir, "*.safetensors"))
            return sorted(files)
        except Exception:
            pass

        raise FileNotFoundError(f"No safetensors files found for {self.model_path}")

    @staticmethod
    def _pad1d_graph_safe(
        hidden_states: torch.Tensor,
        paddings: tuple[int, int],
        mode: str = "constant",
        value: float = 0.0,
    ) -> torch.Tensor:
        """Like F.pad but avoids reflect-mode errors when input is too short."""
        if mode != "reflect":
            return nn.functional.pad(hidden_states, paddings, mode, value)

        length = hidden_states.shape[-1]
        padding_left, padding_right = paddings
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            hidden_states = nn.functional.pad(hidden_states, (0, extra_pad))
        padded = nn.functional.pad(hidden_states, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]

    @classmethod
    def _mimi_conv1d_graph_safe_forward(
        cls,
        module: nn.Module,
        hidden_states: torch.Tensor,
        padding_cache: Any = None,
    ) -> torch.Tensor:
        """Replacement forward for MimiConv1d that is CUDA graph-safe.

        The original forward calls F.pad with reflect mode on potentially
        short sequences, which can fail inside a graph capture.
        """
        stride = int(module.conv.stride[0])  # type: ignore[attr-defined]
        dilation = int(module.conv.dilation[0])  # type: ignore[attr-defined]
        kernel_size = (int(module.conv.kernel_size[0]) - 1) * dilation + 1  # type: ignore[attr-defined]
        padding_total = kernel_size - stride
        length = int(hidden_states.shape[-1])
        n_frames = math.ceil((length - kernel_size + padding_total) / stride + 1) - 1
        ideal_length = n_frames * stride + kernel_size - padding_total
        extra_padding = int(ideal_length - length)

        if module.causal and padding_cache is not None:  # type: ignore[attr-defined]
            layer_padding_cache = padding_cache.update(hidden_states, module.layer_idx)  # type: ignore[attr-defined]
            hidden_states = torch.cat([layer_padding_cache, hidden_states], dim=2)
        elif module.causal:  # type: ignore[attr-defined]
            hidden_states = cls._pad1d_graph_safe(
                hidden_states,
                (padding_total, extra_padding),
                mode=module.pad_mode,  # type: ignore[attr-defined]
            )
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            hidden_states = cls._pad1d_graph_safe(
                hidden_states,
                (padding_left, padding_right + extra_padding),
                mode=module.pad_mode,  # type: ignore[attr-defined]
            )

        return module.conv(hidden_states)  # type: ignore[attr-defined]

    def _patch_mimi_conv1d_for_cuda_graphs(self) -> None:
        if self._mimi is None:
            return
        for module in self._mimi.modules():
            if module.__class__.__name__ != "MimiConv1d":
                continue
            if getattr(module, "_mimi_cuda_graph_forward_patched", False):
                continue

            def forward(
                patched_module: nn.Module,
                hidden_states: torch.Tensor,
                padding_cache: Any = None,
            ) -> torch.Tensor:
                return self._mimi_conv1d_graph_safe_forward(patched_module, hidden_states, padding_cache)

            module.forward = MethodType(forward, module)  # type: ignore[method-assign]
            module._mimi_cuda_graph_forward_patched = True  # type: ignore[attr-defined]

    @staticmethod
    def _decode_residual_vector_quantizer_graph_safe(rvq: nn.Module, codes: torch.Tensor) -> torch.Tensor:
        quantized_out: torch.Tensor | None = None
        for i, indices in enumerate(codes.transpose(0, 1)):
            quantized = rvq.layers[i].decode(indices)  # type: ignore[attr-defined]
            quantized_out = quantized if quantized_out is None else quantized_out + quantized
        assert quantized_out is not None
        output_proj = getattr(rvq, "output_proj", None)
        if output_proj is not None:
            quantized_out = output_proj(quantized_out)
        return quantized_out

    def _decode_quantizer_graph_safe(self, codes: torch.Tensor) -> torch.Tensor:
        assert self._mimi is not None
        quantizer = self._mimi.quantizer
        semantic_rvq = getattr(quantizer, "semantic_residual_vector_quantizer", None)
        acoustic_rvq = getattr(quantizer, "acoustic_residual_vector_quantizer", None)
        if semantic_rvq is None or acoustic_rvq is None:
            return self._decode_residual_vector_quantizer_graph_safe(quantizer, codes)

        num_semantic = int(getattr(quantizer, "num_semantic_quantizers", 0))
        quantized_out = self._decode_residual_vector_quantizer_graph_safe(semantic_rvq, codes[:, :num_semantic])
        if codes.shape[1] > num_semantic:
            quantized_out = quantized_out + self._decode_residual_vector_quantizer_graph_safe(
                acoustic_rvq, codes[:, num_semantic:]
            )
        return quantized_out

    @staticmethod
    def _as_batched_waveform(waveform: torch.Tensor, batch_size: int) -> torch.Tensor:
        waveform = waveform.to(dtype=torch.float32)
        if waveform.ndim <= 1:
            return waveform.reshape(1, -1) if batch_size <= 1 else waveform.reshape(batch_size, -1)
        return waveform.reshape(int(waveform.shape[0]), -1)

    def _decode_codes_graph_safe(self, decode_codes: torch.Tensor) -> torch.Tensor:
        """Decode codes using explicit op sequence safe for CUDA graph capture."""
        assert self._mimi is not None
        embeddings = self._decode_quantizer_graph_safe(decode_codes)
        embeddings = self._mimi.upsample(embeddings)
        decoder_outputs = self._mimi.decoder_transformer(
            embeddings.transpose(1, 2),
            past_key_values=None,
            return_dict=True,
        )
        embeddings = decoder_outputs[0].transpose(1, 2)
        waveform = self._mimi.decoder(embeddings)
        return self._as_batched_waveform(waveform, int(decode_codes.shape[0]))

    def _decode_codes_eager(self, decode_codes: torch.Tensor) -> torch.Tensor:
        assert self._mimi is not None
        audio_output = self._mimi.decode(decode_codes)
        waveform = audio_output.audio_values if hasattr(audio_output, "audio_values") else audio_output
        return waveform.reshape(-1).to(dtype=torch.float32)

    def _cuda_graphs_enabled_for_device(self, device: torch.device) -> bool:
        if not self._cuda_graphs_requested:
            return False
        if device.type != "cuda" or not torch.cuda.is_available():
            return False
        # Don't capture inside another graph capture
        if _inside_vllm_cuda_graph_forward() or torch.cuda.is_current_stream_capturing():
            return False
        return True

    def _decode_codes_cuda_graph(self, decode_codes: torch.Tensor) -> torch.Tensor | None:
        if decode_codes.ndim != 3:
            return None

        batch_size = int(decode_codes.shape[0])
        num_codebooks = int(decode_codes.shape[1])
        num_frames = int(decode_codes.shape[2])
        key = (decode_codes.device, num_codebooks, num_frames)

        if (
            not self._cuda_graphs_enabled_for_device(decode_codes.device)
            or key in self._cuda_graph_failed_keys
            or batch_size != 1  # TTS is single-request; skip batched shapes
        ):
            return None

        try:
            graph_state = self._cuda_graphs.get(key)
            if graph_state is None:
                with nvtx_range("moshi.model.mimi_decoder.cuda_graph.capture"):
                    input_buffer = decode_codes.detach().clone()
                    # Warmup pass before capture
                    _ = self._decode_codes_graph_safe(input_buffer)
                    torch.cuda.synchronize(decode_codes.device)

                    output_ref = self._decode_codes_graph_safe(input_buffer)
                    output_buffer = torch.empty_like(output_ref)

                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        waveform = self._decode_codes_graph_safe(input_buffer)
                        output_buffer.copy_(waveform)

                    graph_state = _DecoderCudaGraphState(
                        graph=graph,
                        input_buffer=input_buffer,
                        output_buffer=output_buffer,
                    )
                    self._cuda_graphs[key] = graph_state
                    self._cuda_graph_capture_count += 1
                    logger.info(
                        "Mimi decoder captured CUDA graph for codebooks=%d frames=%d output_samples=%d "
                        "(total captures=%d)",
                        num_codebooks,
                        num_frames,
                        int(output_buffer.shape[-1]),
                        self._cuda_graph_capture_count,
                    )
                    graph.replay()
                    return output_buffer

            with nvtx_range("moshi.model.mimi_decoder.cuda_graph.replay"):
                graph_state.input_buffer.copy_(decode_codes)
                graph_state.graph.replay()
                self._cuda_graph_replay_count += 1
                return graph_state.output_buffer

        except Exception as exc:
            self._cuda_graphs.pop(key, None)
            self._cuda_graph_failed_keys.add(key)
            logger.warning(
                "Mimi decoder CUDA graph capture/replay failed (key=%s); falling back to eager: %s",
                key,
                exc,
            )
            return None

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros(
            (input_ids.shape[0], 1),
            device=input_ids.device,
            dtype=torch.float32,
        )

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
        return None

    @torch.no_grad()
    @nvtx_annotate("moshi.model.mimi_decoder.forward")
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        """Decode audio codes into waveform.

        input_ids layout: flat codes [num_codebooks * num_frames],
        codebook-major: [cb0_f0, cb0_f1, ..., cb0_fN, cb1_f0, ...].
        """
        self._ensure_mimi_loaded()
        assert self._mimi is not None

        q = self._num_codebooks
        sr_val = self._output_sample_rate
        sr_tensor = torch.tensor(sr_val, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)

        if input_ids is None or input_ids.numel() == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty], "sr": [sr_tensor]},
            )

        ids = input_ids.reshape(-1).to(dtype=torch.long)
        total = ids.numel()

        if total % q != 0:
            total = (total // q) * q
            ids = ids[:total]

        num_frames = total // q
        if num_frames == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty], "sr": [sr_tensor]},
            )

        # [1, num_codebooks, num_frames]
        codes = ids.reshape(q, num_frames).unsqueeze(0).to(device=ids.device)

        # Try CUDA graph → graph-safe eager → mimi.decode() eager
        waveform = self._decode_codes_cuda_graph(codes)
        if waveform is not None:
            waveform = waveform.reshape(-1).clone()
        else:
            try:
                waveform = self._decode_codes_graph_safe(codes).reshape(-1)
            except Exception as exc:
                logger.warning("Mimi graph-safe decode failed, falling back to mimi.decode(): %s", exc)
                waveform = self._decode_codes_eager(codes)

        waveform = waveform.to(dtype=torch.float32)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "audio": waveform,
                "model_outputs": [waveform],
                "sr": [sr_tensor],
            },
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Weights are loaded lazily in _ensure_mimi_loaded."""
        return set()
