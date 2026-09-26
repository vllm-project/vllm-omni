# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from threading import Lock

import torch
from torch import nn
from transformers import Qwen3ForCausalLM
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization import ComponentQuantizationConfig


def prepare_flux2_klein_text_encoder_fp8(
    encoder: Qwen3ForCausalLM, quant_config: QuantizationConfig | None, device: torch.device
) -> int:
    """Replace Qwen3 decoder projections with online FP8 linears."""
    if not isinstance(quant_config, ComponentQuantizationConfig):
        return 0
    config = quant_config.component_configs.get("text_encoder")
    if config is None:
        return 0
    if (
        not isinstance(config, Fp8Config)
        or config.is_checkpoint_fp8_serialized
        or config.activation_scheme != "dynamic"
    ):
        raise ValueError("FLUX.2-klein text_encoder supports dynamic online FP8 from an unquantized checkpoint only.")

    layers = encoder.model.layers
    replaced = 0
    linear_names = [name for name, layer in layers.named_modules() if isinstance(layer, nn.Linear)]
    for name in linear_names:
        layer = layers.get_submodule(name)
        dtype = layer.weight.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("FLUX.2-klein text_encoder FP8 requires BF16 or FP16 weights.")
        with torch.device(device), set_default_torch_dtype(dtype):
            replacement = ReplicatedLinear(
                layer.in_features,
                layer.out_features,
                bias=layer.bias is not None,
                params_dtype=dtype,
                quant_config=config,
                prefix=f"text_encoder.model.layers.{name}",
                return_bias=False,
                disable_tp=True,
            )
        if isinstance(replacement.quant_method, UnquantizedLinearMethod):
            continue
        with torch.no_grad():
            replacement.weight.weight_loader(replacement.weight, layer.weight)
            if layer.bias is not None:
                replacement.bias.weight_loader(replacement.bias, layer.bias)
        replacement.to(layer.weight.device)
        replacement.train(layer.training)
        parent, _, child = name.rpartition(".")
        setattr(layers.get_submodule(parent), child, replacement)
        replaced += 1
    return replaced


class Flux2KleinTextEncoderGraph:
    """Replay the fixed-shape FP8 encoder without per-layer Python launches."""

    def __init__(self, encoder: Qwen3ForCausalLM, hidden_states_layers: tuple[int, ...]):
        self.encoder = encoder
        self.hidden_states_layers = hidden_states_layers
        self.graph: torch.cuda.CUDAGraph | None = None
        self._lock = Lock()

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if input_ids.shape != (1, 512) or not current_omni_platform.is_cuda():
            output = self.encoder.model(
                input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False
            )
            return torch.stack([output.hidden_states[k] for k in self.hidden_states_layers], dim=1)

        with self._lock:
            return self._replay(input_ids, attention_mask)

    def _replay(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.graph is None:
            self.input_ids = input_ids.clone()
            self.attention_mask = attention_mask.clone()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side), torch.inference_mode():
                for _ in range(3):
                    self.encoder.model(
                        input_ids=self.input_ids,
                        attention_mask=self.attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                    )
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                output = self.encoder.model(
                    input_ids=self.input_ids,
                    attention_mask=self.attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
            self.outputs = tuple(output.hidden_states[k] for k in self.hidden_states_layers)
            self.graph = graph

        self.input_ids.copy_(input_ids)
        self.attention_mask.copy_(attention_mask)
        self.graph.replay()
        return torch.stack(self.outputs, dim=1)
