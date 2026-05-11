# SPDX-License-Identifier: Apache-2.0

"""Raon adaptor modules: EmbeddingAdaptor and ThinkerToTalkerProjection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class EmbeddingAdaptorOutput:
    outputs_embeds: torch.Tensor
    mask: torch.Tensor | None = None


class EmbeddingAdaptor(nn.Module):
    @staticmethod
    def _resolve_time_scale(output_time_scale: float) -> tuple[int, bool]:
        if output_time_scale <= 0:
            raise ValueError(f"`output_time_scale` must be positive, got `{output_time_scale}`.")

        if output_time_scale >= 1:
            scale = int(output_time_scale)
            if float(scale) != float(output_time_scale):
                raise ValueError(f"`output_time_scale` must be an integer when >= 1, got `{output_time_scale}`.")
            return scale, True

        inverse_scale = 1 / output_time_scale
        scale = int(inverse_scale)
        if float(scale) != float(inverse_scale):
            raise ValueError(f"`1/output_time_scale` must be an integer when < 1, got `{output_time_scale}`.")
        return scale, False

    def __init__(
        self,
        input_size: int,
        output_size: int,
        output_time_scale: float = 1.0,
        num_layers: int = 1,
        hidden_size: int | None = None,
        decoder_config: object | None = None,
        use_post_norm: bool = False,
        norm_eps: float = 1e-6,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.output_time_scale = output_time_scale
        self.decoder_config = decoder_config
        self._time_scale, self._expand_time_axis = self._resolve_time_scale(output_time_scale)
        self.post_norm: nn.Module | None = None

        if self._expand_time_axis:
            proj_input_size = input_size
            final_output_size = output_size * self._time_scale
        else:
            proj_input_size = input_size * self._time_scale
            final_output_size = output_size

        if num_layers == 1:
            self.proj = nn.Linear(proj_input_size, final_output_size, bias=False, dtype=dtype)
        elif num_layers == 2:
            hidden = hidden_size or final_output_size
            self.proj = nn.Sequential(
                nn.Linear(proj_input_size, hidden, bias=False, dtype=dtype),
                nn.GELU(),
                nn.Linear(hidden, final_output_size, bias=False, dtype=dtype),
            )
        else:
            raise ValueError(f"num_layers must be 1 or 2, got {num_layers}")

        if use_post_norm:
            self.post_norm = nn.RMSNorm(output_size, eps=norm_eps)

    @property
    def proj_dtype(self) -> torch.dtype:
        parameter = next(self.proj.parameters(), None)
        if parameter is None:
            raise RuntimeError("EmbeddingAdaptor has no parameters.")
        return parameter.dtype

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> EmbeddingAdaptorOutput:
        batch_size, seq_length, _ = inputs.shape

        if self._expand_time_axis:
            scale = self._time_scale
            outputs_embeds = self.proj(inputs)

            outputs_embeds = outputs_embeds.view(batch_size, seq_length * scale, self.output_size)
            output_mask = mask.repeat_interleave(scale, dim=1) if mask is not None else None
        else:
            scale = self._time_scale
            remainder = seq_length % scale
            if remainder != 0:
                pad_len = scale - remainder
                last_embed = inputs[:, -1:].expand(-1, pad_len, -1)
                inputs = torch.cat([inputs, last_embed], dim=1)
                if mask is not None:
                    mask = F.pad(mask, (0, pad_len), value=False)
            new_seq_len = inputs.shape[1] // scale
            inputs = inputs.view(batch_size, new_seq_len, scale * self.input_size)
            outputs_embeds = self.proj(inputs)

            output_mask = mask.view(batch_size, new_seq_len, scale).any(dim=-1) if mask is not None else None

        if self.post_norm is not None:
            outputs_embeds = self.post_norm(outputs_embeds)

        return EmbeddingAdaptorOutput(outputs_embeds=outputs_embeds, mask=output_mask)


class ThinkerToTalkerProjection(nn.Module):
    """Project thinker hidden states into the separate talker hidden space."""

    def __init__(
        self,
        thinker_hidden_size: int,
        talker_hidden_size: int,
        intermediate_size: int | None = None,
        mode: str = "mlp",
        use_norm: bool = True,
        rms_norm_eps: float = 1e-6,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if mode != "mlp":
            raise ValueError(f"Only mlp mode is supported, got: {mode}")
        self.mode = "mlp"
        self.norm: nn.Module | None = (
            nn.RMSNorm(thinker_hidden_size, eps=rms_norm_eps, dtype=dtype) if use_norm else None
        )
        if intermediate_size is None:
            raise ValueError("intermediate_size is required for mlp thinker_to_talker projection.")
        self.linear_fc1 = nn.Linear(
            thinker_hidden_size,
            intermediate_size,
            bias=True,
            dtype=dtype,
        )
        self.linear_fc2 = nn.Linear(
            intermediate_size,
            talker_hidden_size,
            bias=True,
            dtype=dtype,
        )
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_states)))
