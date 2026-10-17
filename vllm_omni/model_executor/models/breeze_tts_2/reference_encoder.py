# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The bundled Qwen3 codec encoder used by Breeze voice cloning."""

import torch
from torch import nn
from transformers.models.mimi.modeling_mimi import MimiConv1d
from vllm.config import VllmConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Config,
)
from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Encoder,
)


class BreezeReferenceEncoder(nn.Module):
    def __init__(self, vllm_config: VllmConfig) -> None:
        super().__init__()
        config = Qwen3TTSTokenizerV2Config.from_pretrained(
            vllm_config.model_config.model,
            subfolder="audio_tokenizer",
            revision=vllm_config.model_config.revision,
        )
        if config.input_sample_rate != 24000 or config.encode_downsample_rate != 1920:
            raise ValueError("Breeze requires the bundled 24 kHz, 12.5 Hz Qwen3 codec")
        config.encoder_config._attn_implementation = "sdpa"
        # The released Breeze runtime encodes references in FP32. Quantizer
        # boundaries affect speaker conditioning, so preserve that precision.
        self.encoder = Qwen3TTSTokenizerV2Encoder._from_config(config.encoder_config, dtype=torch.float32).eval()
        self._convolutions: list[BreezeReferenceConv] = []
        hop = 1
        for name, module in list(self.encoder.named_modules()):
            if isinstance(module, MimiConv1d):
                hop *= module.conv.stride[0]
                convolution = BreezeReferenceConv(module, hop)
                self._convolutions.append(convolution)
                parent, _, attribute = name.rpartition(".")
                self.encoder.get_submodule(parent).add_module(attribute, convolution)
        if hop != config.encode_downsample_rate:
            raise ValueError("Breeze reference convolution strides do not match the codec frame rate")
        self.num_quantizers = config.encoder_valid_num_quantizers
        # Reference convolutions have large temporary workspaces. Compile
        # without a graph's permanently reserved workspace for every length.
        self._compiled_encode = torch.compile(
            self._encode_impl,
            fullgraph=True,
            dynamic=True,
            options={"triton.cudagraphs": False, "epilogue_fusion": False},
        )

    def forward(self, waveform: torch.Tensor, frames: int) -> torch.Tensor:
        return self.encode_batch([waveform], [frames])[0]

    def encode_batch(self, waveforms: list[torch.Tensor], frame_counts: list[int]) -> list[torch.Tensor]:
        device = next(self.encoder.parameters()).device
        waveform = nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
        waveform = nn.functional.pad(waveform, (0, -waveform.shape[1] % 1920))
        waveform_frames = waveform.reshape(len(waveforms), -1, 1920).to(device=device, dtype=torch.float32)
        lengths = torch.tensor([waveform.numel() for waveform in waveforms], device=device, dtype=torch.long)
        with torch.backends.cudnn.flags(allow_tf32=False):
            encoded = self._compiled_encode(waveform_frames, lengths)
        return [
            encoded[row, :, :frames].transpose(0, 1).clone(memory_format=torch.contiguous_format)
            for row, frames in enumerate(frame_counts)
        ]

    def warmup(self, max_batch_size: int) -> None:
        device = next(self.encoder.parameters()).device
        with torch.backends.cudnn.flags(allow_tf32=False):
            for batch in range(1, max_batch_size + 1):
                waveform_frames = torch.zeros((batch, 32, 1920), device=device, dtype=torch.float32)
                lengths = torch.full((batch,), 32 * 1920, device=device, dtype=torch.long)
                # Initialize the codec's normalized codebooks before tracing.
                self._encode_impl(waveform_frames, lengths)
                self._compiled_encode(waveform_frames, lengths)
        torch.accelerator.empty_cache()

    def _encode_impl(self, waveform_frames: torch.Tensor, sample_lengths: torch.Tensor) -> torch.Tensor:
        # Expose the complete 1920-sample stride product to the compiler.
        # Nested ceil/mod expressions for arbitrary waveform lengths otherwise
        # cause expensive symbolic simplification during kernel tiling. Every
        # convolution below masks the true per-recording boundary, so aligning
        # the storage to full frames preserves partial final-frame semantics.
        torch._dynamo.mark_static(waveform_frames, 2)
        waveform = waveform_frames.flatten(1)
        # The released Breeze reference encoder uses full causal attention
        # during offline encoding. Recent Transformers Mimi defaults to a
        # sliding mask, which changes references longer than ten seconds.
        # Supply the actual reference mask rather than changing the checkpoint
        # configuration or relying on a Transformers-version default.
        for convolution in self._convolutions:
            convolution.sample_lengths = sample_lengths
        hidden = self.encoder.encoder(waveform.unsqueeze(1)).transpose(1, 2)
        length = hidden.shape[1]
        causal_mask = torch.ones((length, length), device=hidden.device, dtype=torch.bool).tril()[None, None]
        hidden = self.encoder.encoder_transformer(
            hidden, attention_mask=causal_mask, use_cache=False, return_dict=True
        ).last_hidden_state
        hidden = self.encoder.downsample(hidden.transpose(1, 2))
        return self.encoder.quantizer.encode(hidden, self.num_quantizers).transpose(0, 1)

    def load_weights(self, vllm_config: VllmConfig) -> set[str]:
        loader = DefaultModelLoader(vllm_config.load_config)
        source = DefaultModelLoader.Source(
            model_or_path=vllm_config.model_config.model,
            revision=vllm_config.model_config.revision,
            subfolder="audio_tokenizer",
        )
        parameters = dict(self.named_parameters())
        buffers = dict(self.named_buffers())
        loaded: set[str] = set()
        for name, value in loader._get_weights_iterator(source):
            if name.startswith("decoder."):
                continue
            if name in parameters:
                default_weight_loader(parameters[name], value)
            elif name in buffers:
                buffers[name].copy_(value)
            else:
                raise ValueError(f"Unexpected reference codec weight: {name}")
            loaded.add(name)
        required = set(parameters) | {name for name in buffers if ".quantizer." in name}
        missing = required - loaded
        if missing:
            raise ValueError(f"Uninitialized reference codec weights or codebooks: {sorted(missing)}")
        return loaded


class BreezeReferenceConv(nn.Module):
    """Offline Mimi convolution with padding derived from static shape metadata."""

    def __init__(self, source: MimiConv1d, output_hop: int) -> None:
        super().__init__()
        if not source.causal or source.pad_mode not in ("constant", "replicate"):
            raise ValueError("Breeze reference encoding requires causal zero/replicate-padded convolutions")
        self.conv = source.conv
        self.pad_mode = source.pad_mode
        self.stride = self.conv.stride[0]
        self.padding = (self.conv.kernel_size[0] - 1) * self.conv.dilation[0] + 1 - self.stride
        self.output_hop = output_hop
        self.sample_lengths: torch.Tensor | None = None

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # HF stores these integers in GPU tensors, introducing a device sync
        # when F.pad converts them back to integers. Shapes determine padding.
        if self.sample_lengths is None:
            raise RuntimeError("Reference convolution requires the batch's input lengths")
        if self.pad_mode == "replicate":
            # Mimi's final downsampler repeats the actual last encoder state
            # for a partial frame. Each row has its own boundary in a bucket.
            input_hop = self.output_hop // self.stride
            lengths = (self.sample_lengths + input_hop - 1) // input_hop
            indices = torch.arange(hidden.shape[-1], device=hidden.device)[None, :]
            indices = torch.minimum(indices, lengths[:, None] - 1)
            hidden = hidden.gather(2, indices[:, None, :].expand(-1, hidden.shape[1], -1))
        right = -hidden.shape[-1] % self.stride
        hidden = nn.functional.pad(hidden, (self.padding, right), mode=self.pad_mode)
        hidden = self.conv(hidden)
        lengths = (self.sample_lengths + self.output_hop - 1) // self.output_hop
        invalid = torch.arange(hidden.shape[-1], device=hidden.device)[None, :] >= lengths[:, None]
        return hidden.masked_fill(invalid[:, None, :], 0)
