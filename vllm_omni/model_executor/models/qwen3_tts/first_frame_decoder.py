# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""First codec frame -> PCM inside the Talker process.

A stream's first frame has no decoder context, so its waveform does not depend
on the Code2Wav stage's streaming state. Decoding it where the frame is
produced removes the stage hop from time to first audio. Code2Wav advances
its causal state without emitting that delivered frame again. Only streams
without reference-code context use this path.

The decode runs beside the Talker's next step, so its latency is set by how
many dependent kernels it issues. A single frame with no context attends to
itself only: each softmax is over one key and equals 1 exactly, so attention
returns the value projection. The single-frame path therefore
skips q/k, RoPE, masks and softmax (about two thirds of the kernels).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.cuda import CUDAGraph
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import DefaultModelLoader
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper

from .tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config
from .tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Decoder

logger = init_logger(__name__)

_SUBFOLDER = "speech_tokenizer"


class Qwen3TTSFirstFrameDecoder(nn.Module):
    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.model_path = model_path
        config = Qwen3TTSTokenizerV2Config.from_pretrained(model_path, subfolder=_SUBFOLDER)
        self.decoder = Qwen3TTSTokenizerV2Decoder._from_config(config.decoder_config)
        self.decoder.eval()
        self.num_quantizers = int(config.decoder_config.num_quantizers)
        self.sample_rate = int(config.output_sample_rate)
        self._graphs: dict[int, tuple[CUDAGraph, torch.Tensor, torch.Tensor]] = {}

    def load(self, vllm_config: VllmConfig) -> set[str]:
        """Load decoder weights exactly as Code2Wav does; returns the loaded parameter names."""
        loader = DefaultModelLoader(vllm_config.load_config)
        source = DefaultModelLoader.Source(
            model_or_path=self.model_path,
            revision=vllm_config.model_config.revision,
            subfolder=_SUBFOLDER,
        )
        loaded = AutoWeightsLoader(self).load_weights(
            loader._get_weights_iterator(source),
            mapper=WeightsMapper(orig_to_new_prefix={"encoder.": None}),
        )
        self.decoder.to(device=vllm_config.device_config.device, dtype=vllm_config.model_config.dtype)
        if hasattr(self.decoder, "precompute_snake_caches"):
            self.decoder.precompute_snake_caches()
        from .qwen3_tts_code_predictor_vllm import Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM as Predictor

        extra = Predictor._stage_connector_extra_config(vllm_config)
        if Predictor._parse_bool_config(extra.get("decode_time_major_conv")):
            self.decoder.enable_time_major_conv()
        return loaded

    def _decode_single_frame(self, codes: torch.Tensor) -> torch.Tensor:
        """``codes`` [n, num_quantizers, 1] -> waveform; equals ``_decode_xvec_first_chunk``."""
        d = self.decoder
        hidden = d.pre_conv(d.quantizer.decode(codes)).transpose(1, 2)
        t = d.pre_transformer
        h = t.input_proj(hidden)
        for layer in t.layers[: t.config.num_hidden_layers]:
            attention = layer.self_attn
            values = attention.v_proj(layer.input_layernorm(h))
            if attention.num_key_value_groups != 1:
                values = values.reshape(*values.shape[:-1], -1, attention.head_dim)
                values = values.repeat_interleave(attention.num_key_value_groups, dim=-2).flatten(-2)
            h = h + layer.self_attn_layer_scale(attention.o_proj(values))
            h = h + layer.mlp_layer_scale(layer.mlp(layer.post_attention_layernorm(h)))
        return d._conv_decode(t.output_proj(t.norm(h))).clamp(min=-1, max=1)

    @torch.inference_mode()
    def capture(self, batch_sizes: Sequence[int] = (1, 2, 4, 8)) -> None:
        device = next(self.decoder.parameters()).device
        for batch_size in sorted(set(batch_sizes)):
            static_input = torch.zeros(batch_size, self.num_quantizers, 1, dtype=torch.long, device=device)
            self._decode_single_frame(static_input)
            torch.accelerator.synchronize(device)
            graph = CUDAGraph()
            # A separate pool prevents overlap with Talker/MTP graph allocations.
            with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
                static_output = self._decode_single_frame(static_input)
            self._graphs[batch_size] = (graph, static_input, static_output)
        logger.info("Captured Talker first-frame decoder graphs for batch sizes %s", sorted(self._graphs))

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """``codes`` [n, num_quantizers] long on device -> float32 PCM [n, samples] (new tensor)."""
        if not self._graphs:
            return self._decode_single_frame(codes.unsqueeze(-1))[:, 0, :].float()
        outputs = []
        largest = max(self._graphs)
        for start in range(0, int(codes.shape[0]), largest):
            chunk = codes[start : start + largest]
            rows = int(chunk.shape[0])
            batch_size = min(size for size in self._graphs if size >= rows)
            graph, static_input, static_output = self._graphs[batch_size]
            static_input.zero_()
            static_input[:rows, :, 0].copy_(chunk)
            graph.replay()
            outputs.append(static_output[:rows, 0, :].to(dtype=torch.float32, copy=True))
        return torch.cat(outputs, dim=0)
