# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flux3 text encoder.

Thin wrapper around the Qwen3-VL text encoder that produces the ``(ctx, vector)``
conditioning the DiT expects:

- ``ctx``:    ``(batch, n_tokens, context_in_dim)`` per-token text hidden states
              (consumed by the DiT ``txt_in`` projection). Built by stacking hidden
              states from a fixed set of Qwen layers (``output_layer``) along the
              channel axis.
- ``vector``: ``(batch, vec_in_dim)`` zeros when a pooled-vector projection exists,
              or ``(batch, 0)`` when ``vec_in_dim=None``. F3 image checkpoints do
              not condition the DiT on a pooled text vector.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from transformers import AutoProcessor, PretrainedConfig, Qwen3VLForConditionalGeneration
from vllm.logger import init_logger
from vllm.model_executor.models.transformers.utils import init_on_device_without_buffers

from vllm_omni.diffusion.models.utils import init_parameters

logger = init_logger(__name__)

DEFAULT_VEC_IN_DIM = 768
DEFAULT_TEXT_TOKENS = 8192
DEFAULT_PAD_MULTIPLE = 80


_DEFAULT_OUTPUT_LAYER = [4, 8, 12, 16, 20, 24, 28, 32]


def _text_position_ids_from_attention_mask(attention_mask: Tensor) -> Tensor:
    """Assign sequential text positions and position 1 to padding tokens."""
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    return position_ids.masked_fill(attention_mask == 0, 1)


class Flux3TextEncoder(nn.Module):
    """Wraps a Qwen3-VL model + tokenizer and emits ``(ctx, vector)``.

    Args:
        text_encoder: a loaded ``Qwen3VLForConditionalGeneration``.
        tokenizer: the matching tokenizer.
        context_in_dim: width of ``ctx`` expected by the DiT ``txt_in`` layer.
        vec_in_dim: width of the zero ``vector`` tensor, or ``None`` when the DiT
            has no pooled-vector input projection.
        max_sequence_length: token count ``n_tokens`` for ``ctx``.
        output_layer: indices into ``hidden_states`` to stack for ``ctx``.
            Defaults to ``[4, 8, 12, 16, 20, 24, 28, 32]``.
    """

    def __init__(
        self,
        text_encoder: Qwen3VLForConditionalGeneration,
        processor: AutoProcessor,
        *,
        context_in_dim: int,
        vec_in_dim: int | None = DEFAULT_VEC_IN_DIM,
        max_sequence_length: int = DEFAULT_TEXT_TOKENS,
        pad_multiple: int = DEFAULT_PAD_MULTIPLE,
        output_layer: list[int] | None = None,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.processor = processor
        self.context_in_dim = context_in_dim
        self.vec_in_dim = vec_in_dim
        self.max_sequence_length = max_sequence_length
        self.pad_multiple = pad_multiple
        self.output_layer = list(output_layer) if output_layer is not None else list(_DEFAULT_OUTPUT_LAYER)
        self._compiled_language_model_forward = None

    def finalize_after_loading(self) -> None:
        """Drop decoder layers that are not needed after loading their weights."""
        language_model = self.text_encoder.model.language_model
        last_output_layer = max(self.output_layer)
        if last_output_layer > len(language_model.layers):
            raise ValueError(
                f"output_layer index {last_output_layer} out of range; text encoder "
                f"has {len(language_model.layers)} decoder layers"
            )
        if last_output_layer < len(language_model.layers):
            language_model.layers = nn.ModuleList(list(language_model.layers[:last_output_layer]))
            # Preserve the intermediate layer output that Flux3 consumed before
            # truncation instead of treating it as a final normalized state.
            language_model.norm = nn.Identity()

    def setup_compile(self, *, dynamic: bool = False) -> None:
        """Compile the tensor-only language-model forward after weight loading."""
        self._compiled_language_model_forward = torch.compile(
            self._language_model_forward,
            dynamic=dynamic,
        )

    def _language_model_forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
    ) -> tuple[Tensor, ...]:
        output = self.text_encoder.model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
        )
        return output.hidden_states

    @classmethod
    def from_config(
        cls,
        config: PretrainedConfig,
        *,
        processor: AutoProcessor,
        context_in_dim: int,
        device: str | torch.device,
        output_layer: list[int] | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> Flux3TextEncoder:
        """Construct an empty Qwen3-VL backbone for external weight loading."""
        with init_on_device_without_buffers("meta"):
            text_encoder = Qwen3VLForConditionalGeneration(config)
        init_parameters(text_encoder, dtype=torch_dtype, device=torch.device(device))
        if config.tie_word_embeddings:
            text_encoder.tie_weights()
        return cls(
            text_encoder,
            processor,
            context_in_dim=context_in_dim,
            output_layer=output_layer,
            **kwargs,
        )

    @property
    def device(self) -> torch.device:
        return self.text_encoder.device

    @property
    def dtype(self) -> torch.dtype:
        return self.text_encoder.dtype

    def _encode_qwen3(self, prompt: list[str], max_sequence_length: int):
        """Run Qwen3-VL's truncated text decoder and return its hidden states."""
        texts = []
        real_lengths = []
        for single_prompt in prompt:
            messages = [{"role": "user", "content": single_prompt}]
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            texts.append(text)
            real = self.processor.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_sequence_length,
            )["input_ids"].shape[1]
            real_lengths.append(real)

        target_length = max(real_lengths) if real_lengths else 1
        if self.pad_multiple > 1:
            target_length = math.ceil(target_length / self.pad_multiple) * self.pad_multiple
        target_length = min(target_length, max_sequence_length)

        all_input_ids = []
        all_attention_masks = []
        for text in texts:
            inputs = self.processor.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=target_length,
            )
            all_input_ids.append(inputs["input_ids"])
            all_attention_masks.append(inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(self.device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(self.device)

        # FLUX.3 consumes all packed hidden states, including padding.
        # Pass explicit position IDs so newer Transformers
        # releases cannot change the padded context by inferring them differently.
        position_ids = _text_position_ids_from_attention_mask(attention_mask)
        language_model_forward = self._compiled_language_model_forward or self._language_model_forward
        hidden_states = language_model_forward(input_ids, attention_mask, position_ids)
        return hidden_states, attention_mask

    def _build_outputs(self, hidden_states: tuple[Tensor, ...]) -> tuple[Tensor, Tensor]:
        """Build ``(ctx, vector)`` from Qwen hidden states.

        - ``ctx``: stacks hidden states at the fixed ``output_layer`` indices along the
          channel axis → ``(B, seq, len(output_layer) * qwen_hidden)``. Must equal
          ``context_in_dim``.
        - ``vector``: zero tensor ``(B, vec_in_dim)`` or ``(B, 0)`` when the image
          DiT has no pooled-vector input projection.
        """
        sample = hidden_states[self.output_layer[0]]
        hidden = sample.shape[-1]
        expected = len(self.output_layer) * hidden
        if expected != self.context_in_dim:
            raise ValueError(
                f"len(output_layer)={len(self.output_layer)} × qwen_hidden={hidden} "
                f"= {expected}, but context_in_dim={self.context_in_dim}."
            )
        max_idx = max(self.output_layer)
        if max_idx >= len(hidden_states):
            raise ValueError(
                f"output_layer index {max_idx} out of range; model exposes "
                f"{len(hidden_states)} hidden-state tensors (indices 0–{len(hidden_states) - 1})."
            )
        ctx = torch.cat([hidden_states[k] for k in self.output_layer], dim=-1).to(torch.bfloat16)
        vector_dim = 0 if self.vec_in_dim is None else self.vec_in_dim
        vector = torch.zeros(ctx.shape[0], vector_dim, device=ctx.device, dtype=torch.bfloat16)
        return ctx, vector

    @torch.inference_mode()
    def encode(self, prompt: str | list[str]) -> tuple[Tensor, Tensor]:
        """Encode text to ``(ctx, vector)``."""
        prompt = [prompt] if isinstance(prompt, str) else list(prompt)
        hidden_states, _attention_mask = self._encode_qwen3(prompt, self.max_sequence_length)
        return self._build_outputs(hidden_states)

    # Alias so the module can be called directly as an nn.Module.
    forward = encode
