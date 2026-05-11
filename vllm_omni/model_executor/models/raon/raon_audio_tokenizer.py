# SPDX-License-Identifier: Apache-2.0

"""Streaming Mimi audio tokenizer with causal padding-cache support."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
from transformers import MimiConfig, MimiModel
from transformers.cache_utils import Cache
from transformers.models.mimi.modeling_mimi import (
    MimiConv1d,
    MimiConv1dPaddingCache,
    MimiConvTranspose1d,
    MimiResidualVectorQuantizer,
    MimiResnetBlock,
)
from transformers.utils.generic import ModelOutput


class MimiConvTranspose1dPaddingCache:
    """Padding cache for causal MimiConvTranspose1d layers."""

    def __init__(
        self,
        num_layers: int,
        per_layer_padding: list[torch.Tensor],
        per_layer_in_channels: list[int],
    ) -> None:
        from_args_num_layers = {len(per_layer_padding), len(per_layer_in_channels)}

        if len(from_args_num_layers) != 1 or from_args_num_layers.pop() != num_layers:
            raise ValueError(
                f"Expected `num_layers` ({num_layers}) values in `per_layer_padding`, and `per_layer_in_channels`."
            )
        self.per_layer_padding = [
            int(p.long().item()) if isinstance(p, torch.Tensor) else int(p) for p in per_layer_padding
        ]
        self.per_layer_in_channels = per_layer_in_channels
        self.per_layer_is_init = [True] * num_layers
        self.padding_cache: list[torch.Tensor | None] = [None] * num_layers

    def update(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        batch_size, dtype, device = (
            hidden_states.shape[0],
            hidden_states.dtype,
            hidden_states.device,
        )
        padding = self.per_layer_padding[layer_idx]
        in_channels = self.per_layer_in_channels[layer_idx]

        cached = self.padding_cache[layer_idx]
        current_cache = (
            cached if cached is not None else torch.zeros(batch_size, in_channels, padding, device=device, dtype=dtype)
        )

        padding_states = (
            hidden_states[:, :, -padding:]
            if padding > 0
            else torch.empty(batch_size, in_channels, padding, dtype=dtype, device=device)
        )
        self.padding_cache[layer_idx] = padding_states

        return current_cache


@dataclass
class StreamingMimiDecoderOutput(ModelOutput):
    audio_values: torch.FloatTensor | None = None
    decoder_past_key_values: Cache | list[torch.FloatTensor] | None = None
    conv1d_padding_cache: MimiConv1dPaddingCache | None = None
    convtranspose1d_padding_cache: MimiConvTranspose1dPaddingCache | None = None


class StreamingMimiConvTranspose1d(MimiConvTranspose1d):
    def __init__(
        self,
        config: MimiConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__(config, in_channels, out_channels, kernel_size, stride, groups, bias)

        self.in_channels = in_channels
        self.layer_idx = layer_idx
        kernel_size_tensor = torch.tensor(self.conv.kernel_size[0], dtype=torch.int64)
        stride_tensor = torch.tensor(self.conv.stride[0], dtype=torch.int64)
        padding_total = kernel_size_tensor - stride_tensor

        self.register_buffer("stride", stride_tensor, persistent=False)
        self.register_buffer("kernel_size", kernel_size_tensor, persistent=False)
        self.register_buffer("padding_total", padding_total, persistent=False)

        # Plain ints required — tensor slicing is illegal during CUDA graph capture.
        self._stride_int: int = int(self.conv.stride[0])
        self._padding_left_int: int = int(self.padding_left)  # type: ignore[arg-type]
        self._padding_right_int: int = int(self.padding_right)  # type: ignore[arg-type]

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_cache: MimiConvTranspose1dPaddingCache | None = None,
    ) -> torch.Tensor:
        if not self.causal and padding_cache is not None:
            raise ValueError("`padding_cache` is only defined for causal convolutions.")
        if self.causal and padding_cache is not None:
            if self.layer_idx is None:
                raise RuntimeError("StreamingMimiConvTranspose1d.layer_idx must be set for causal padding cache.")
            layer_padding_cache = padding_cache.update(hidden_states, self.layer_idx)
            padding_len = padding_cache.per_layer_padding[self.layer_idx]
            extra_padding = padding_len - layer_padding_cache.shape[-1]
            if extra_padding > 0:
                layer_padding_cache = nn.functional.pad(
                    layer_padding_cache,
                    (int(extra_padding), 0),
                    mode="constant",
                    value=0,
                )
            hidden_states = torch.cat([layer_padding_cache, hidden_states], dim=-1)
            padding_left = layer_padding_cache.shape[-1] * self._stride_int + self._padding_left_int
        else:
            padding_left = self._padding_left_int
        hidden_states = self.conv(hidden_states)

        end = hidden_states.shape[-1] - self._padding_right_int
        hidden_states = hidden_states[..., padding_left:end]

        return hidden_states


class StreamingMimiDecoder(nn.Module):
    """SEANet decoder as used by Mimi."""

    def __init__(self, config: MimiConfig) -> None:
        super().__init__()
        scaling = int(2 ** len(config.upsampling_ratios))
        model: list[nn.Module] = [
            MimiConv1d(
                config,
                config.hidden_size,
                scaling * config.num_filters,
                config.kernel_size,
            )
        ]
        mimiconv1d_layer_names = ["layers.0"]
        mimiconvtranspose1d_layer_names: list[str] = []

        for ratio in config.upsampling_ratios:
            current_scale = scaling * config.num_filters
            model += [nn.ELU()]
            mimiconvtranspose1d_layer_names.append(f"layers.{len(model)}")
            model += [
                StreamingMimiConvTranspose1d(
                    config,
                    current_scale,
                    current_scale // 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                )
            ]
            for j in range(config.num_residual_layers):
                mimiconv1d_layer_names.extend([f"layers.{len(model)}.block.{1}", f"layers.{len(model)}.block.{3}"])
                model += [
                    MimiResnetBlock(config, current_scale // 2, (config.dilation_growth_rate**j, 1))  # type: ignore
                ]
            scaling //= 2

        model += [nn.ELU()]
        mimiconv1d_layer_names.append(f"layers.{len(model)}")
        model += [
            MimiConv1d(
                config,
                config.num_filters,
                config.audio_channels,
                config.last_kernel_size,
            )
        ]
        self.layers = nn.ModuleList(model)

        self._mimiconv1d_layer_names = mimiconv1d_layer_names
        self._mimiconvtranspose1d_layer_names = mimiconvtranspose1d_layer_names

        for layer_idx, layer_name in enumerate(self._mimiconv1d_layer_names):
            conv_layer = self.get_submodule(layer_name)
            conv_layer.layer_idx = layer_idx  # type: ignore
        for layer_idx, layer_name in enumerate(self._mimiconvtranspose1d_layer_names):
            convtranspose_layer = self.get_submodule(layer_name)
            convtranspose_layer.layer_idx = layer_idx  # type: ignore

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv1d_padding_cache: MimiConv1dPaddingCache | None = None,
        convtranspose1d_padding_cache: MimiConvTranspose1dPaddingCache | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            if isinstance(layer, (MimiConv1d, MimiResnetBlock)):
                hidden_states = layer(hidden_states, padding_cache=conv1d_padding_cache)
            elif isinstance(layer, MimiConvTranspose1d):
                hidden_states = layer(hidden_states, padding_cache=convtranspose1d_padding_cache)
            else:
                hidden_states = layer(hidden_states)
        return hidden_states


class StreamingMimiModel(MimiModel):
    def __init__(self, config: MimiConfig) -> None:
        super().__init__(config)
        self.decoder = StreamingMimiDecoder(config)
        self.upsample = StreamingMimiConvTranspose1d(
            config,
            config.hidden_size,
            config.hidden_size,
            kernel_size=2 * int(config.encodec_frame_rate / config.frame_rate),
            stride=2,
            bias=False,
            groups=config.upsample_groups,
            layer_idx=len(self.decoder._mimiconvtranspose1d_layer_names),
        )
        # Instance-level patch: upstream MimiConv1d.forward lacks padding_cache support.
        for module in self.decoder.modules():
            if isinstance(module, MimiConv1d):
                module.forward = partial(self.mimi_conv1d_forward, module)  # type: ignore[method-assign]

        # Eagerly init padding caches — .item() calls are illegal during CUDA graph capture.
        self._default_conv1d_padding_cache: MimiConv1dPaddingCache = self._init_conv1d_padding_cache()
        self._default_convtranspose1d_padding_cache: MimiConvTranspose1dPaddingCache = (
            self._init_convtranspose_padding_cache()
        )

        # Patch RVQ decode: torch.zeros(()) avoids CPU scalar from torch.tensor(0.0)
        # which is illegal during CUDA graph capture.
        def _rvq_decode_cuda_graph_safe(
            self_rvq: MimiResidualVectorQuantizer,
            codes: torch.Tensor,
        ) -> torch.Tensor:
            quantized_out = torch.zeros((), dtype=codes.dtype, device=codes.device)
            codes = codes.transpose(0, 1)
            for i, indices in enumerate(codes):
                layer = self_rvq.layers[i]
                quantized = layer.decode(indices)
                quantized_out = quantized_out + quantized
            if self_rvq.output_proj is not None:
                quantized_out = self_rvq.output_proj(quantized_out)
            return quantized_out

        for module in self.modules():
            if isinstance(module, MimiResidualVectorQuantizer):
                module.decode = partial(_rvq_decode_cuda_graph_safe, module)  # type: ignore[method-assign]

    @staticmethod
    def _compute_convtranspose_padding(
        kernel_size: torch.Tensor,
        stride: torch.Tensor,
    ) -> torch.Tensor:
        kernel = torch.as_tensor(kernel_size)
        stride_value = torch.as_tensor(stride)
        if bool((kernel % stride_value == 0).item()):
            return (kernel / stride_value - 1) * stride_value
        return torch.floor(kernel / stride_value) * stride_value

    def _init_conv1d_padding_cache(self) -> MimiConv1dPaddingCache:
        per_layer_padding: list[int] = []
        per_layer_padding_mode: list[str] = []
        per_layer_in_channels: list[int] = []
        for layer_name in self.decoder._mimiconv1d_layer_names:
            layer = self.decoder.get_submodule(layer_name)
            raw_padding = layer.padding_total  # type: ignore[attr-defined]
            padding_int = int(raw_padding.long().item()) if isinstance(raw_padding, torch.Tensor) else int(raw_padding)
            per_layer_padding.append(padding_int)
            per_layer_padding_mode.append(layer.pad_mode)  # type: ignore[attr-defined]
            per_layer_in_channels.append(layer.in_channels)  # type: ignore[attr-defined]

        return MimiConv1dPaddingCache(
            num_layers=len(self.decoder._mimiconv1d_layer_names),
            per_layer_padding=per_layer_padding,
            per_layer_padding_mode=per_layer_padding_mode,
            per_layer_in_channels=per_layer_in_channels,
        )

    def _init_convtranspose_padding_cache(self) -> MimiConvTranspose1dPaddingCache:
        per_layer_padding: list[torch.Tensor] = []
        per_layer_in_channels: list[int] = []
        for layer_name in self.decoder._mimiconvtranspose1d_layer_names:
            layer = self.decoder.get_submodule(layer_name)
            per_layer_padding.append(
                self._compute_convtranspose_padding(
                    layer.kernel_size,  # type: ignore[attr-defined]
                    layer.stride,  # type: ignore[attr-defined]
                )
            )
            per_layer_in_channels.append(layer.in_channels)  # type: ignore[attr-defined]

        if self.upsample is None:
            raise RuntimeError("StreamingMimiModel.upsample must be initialized before padding cache setup.")
        per_layer_padding.append(
            self._compute_convtranspose_padding(
                self.upsample.kernel_size,
                self.upsample.stride,
            )
        )
        per_layer_in_channels.append(self.upsample.in_channels)  # type: ignore[attr-defined]

        return MimiConvTranspose1dPaddingCache(
            num_layers=len(self.decoder._mimiconvtranspose1d_layer_names) + 1,
            per_layer_padding=per_layer_padding,
            per_layer_in_channels=per_layer_in_channels,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load parameters and persistent buffers (Mimi codebook buffers are
        not covered by vLLM's default AutoWeightsLoader).
        """
        params = dict(self.named_parameters())
        persistent_keys = set(self.state_dict().keys())
        buffers = {name: buf for name, buf in self.named_buffers() if name in persistent_keys}

        loaded: set[str] = set()
        with torch.no_grad():
            for name, tensor in weights:
                target = params.get(name)
                if target is None:
                    target = buffers.get(name)
                if target is not None:
                    target.copy_(tensor.to(device=target.device, dtype=target.dtype))
                    loaded.add(name)
        return loaded

    def mimi_conv1d_forward(
        self,
        module: MimiConv1d,
        hidden_states: torch.Tensor,
        padding_cache: MimiConv1dPaddingCache | None = None,
    ) -> torch.Tensor:
        extra_padding = module._get_extra_padding_for_conv1d(hidden_states)

        if not module.causal and padding_cache is not None:
            raise ValueError("`padding_cache` is not supported for non-causal convolutions.")

        if module.causal and padding_cache is not None:
            if module.layer_idx is None:
                raise RuntimeError("MimiConv1d.layer_idx must be set for causal padding cache.")
            layer_padding_cache = padding_cache.update(hidden_states, module.layer_idx)
            if layer_padding_cache is None:
                raise RuntimeError("MimiConv1dPaddingCache.update returned None for a causal layer.")
            hidden_states = torch.cat([layer_padding_cache, hidden_states], dim=2)
            if isinstance(module.padding_total, nn.Module):
                raise RuntimeError("MimiConv1d.padding_total must be tensor-like, not an nn.Module.")
            padding_total_int: int = padding_cache.per_layer_padding[module.layer_idx]
            hidden_states = module._pad1d(
                hidden_states,
                (
                    max(0, padding_total_int - layer_padding_cache.shape[2]),
                    extra_padding,  # type: ignore
                ),
                mode=module.pad_mode,
            )

        elif module.causal and padding_cache is None:
            hidden_states = module._pad1d(
                hidden_states,
                (module.padding_total, extra_padding),  # type: ignore
                mode=module.pad_mode,
            )

        else:
            hidden_states = module._pad1d(
                hidden_states,
                (module.padding_left, module.padding_right + extra_padding),  # type: ignore
                mode=module.pad_mode,
            )

        hidden_states = module.conv(hidden_states)
        return hidden_states

    def _decode_frame(  # type: ignore[override]
        self,
        codes: torch.Tensor,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        conv1d_padding_cache: MimiConv1dPaddingCache | None = None,
        convtranspose1d_padding_cache: MimiConvTranspose1dPaddingCache | None = None,
        return_dict: bool | None = None,
    ) -> tuple[
        torch.Tensor,
        Cache | list[torch.FloatTensor] | None,
        MimiConv1dPaddingCache | None,
        MimiConvTranspose1dPaddingCache | None,
    ]:
        embeddings = self.quantizer.decode(codes)

        if self.upsample is None:
            raise RuntimeError("StreamingMimiModel.upsample must be initialized before decode.")
        embeddings = self.upsample(embeddings, padding_cache=convtranspose1d_padding_cache)
        decoder_outputs = self.decoder_transformer(
            embeddings.transpose(1, 2),
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=return_dict,
        )
        if return_dict:
            past_key_values = decoder_outputs.get("past_key_values")
        elif len(decoder_outputs) > 1:
            past_key_values = decoder_outputs[1]
        embeddings = decoder_outputs[0].transpose(1, 2)
        outputs = self.decoder(
            embeddings,
            conv1d_padding_cache=conv1d_padding_cache,
            convtranspose1d_padding_cache=convtranspose1d_padding_cache,
        )
        return outputs, past_key_values, conv1d_padding_cache, convtranspose1d_padding_cache

    def decode(  # type: ignore
        self,
        audio_codes: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        decoder_past_key_values: Cache | list[torch.FloatTensor] | None = None,
        conv1d_padding_cache: MimiConv1dPaddingCache | None = None,
        convtranspose1d_padding_cache: MimiConvTranspose1dPaddingCache | None = None,
        use_streaming: bool | None = True,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | StreamingMimiDecoderOutput:
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        use_streaming = use_streaming if use_streaming is not None else self.config.use_streaming

        if use_streaming and conv1d_padding_cache is None:
            conv1d_padding_cache = self._default_conv1d_padding_cache

        if use_streaming and convtranspose1d_padding_cache is None:
            convtranspose1d_padding_cache = self._default_convtranspose1d_padding_cache

        (
            audio_values,
            decoder_past_key_values,
            conv1d_padding_cache,
            convtranspose1d_padding_cache,
        ) = self._decode_frame(
            audio_codes,
            past_key_values=decoder_past_key_values,
            conv1d_padding_cache=conv1d_padding_cache,
            convtranspose1d_padding_cache=convtranspose1d_padding_cache,
            return_dict=return_dict,
        )

        if padding_mask is not None and padding_mask.shape[-1] < audio_values.shape[-1]:
            audio_values = audio_values[..., : padding_mask.shape[-1]]

        if not return_dict:
            return (  # type: ignore
                audio_values,
                decoder_past_key_values,
                conv1d_padding_cache,
                convtranspose1d_padding_cache,
            )
        return StreamingMimiDecoderOutput(
            audio_values=audio_values,  # type: ignore
            decoder_past_key_values=decoder_past_key_values,
            conv1d_padding_cache=conv1d_padding_cache,
            convtranspose1d_padding_cache=convtranspose1d_padding_cache,
        )
