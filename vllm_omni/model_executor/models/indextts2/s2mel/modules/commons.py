# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels if isinstance(n_channels, int) else n_channels[0]
    in_act = input_a + input_b
    t_act_part, s_act_part = torch.split(in_act, n_channels_int, dim=1)
    t_act = torch.tanh(t_act_part)
    s_act = torch.sigmoid(s_act_part)
    acts = t_act * s_act
    return acts


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


class MyModel(nn.Module):
    def __init__(self, args, use_emovec=False, use_gpt_latent=False):
        super().__init__()
        from vllm_omni.model_executor.models.indextts2.s2mel.modules.flow_matching import CFM
        from vllm_omni.model_executor.models.indextts2.s2mel.modules.length_regulator import (
            InterpolateRegulator,
        )

        length_regulator = InterpolateRegulator(
            channels=args.length_regulator.channels,
            sampling_ratios=args.length_regulator.sampling_ratios,
            is_discrete=args.length_regulator.is_discrete,
            in_channels=getattr(args.length_regulator, "in_channels", None),
            vector_quantize=getattr(args.length_regulator, "vector_quantize", False),
            codebook_size=args.length_regulator.content_codebook_size,
            n_codebooks=getattr(args.length_regulator, "n_codebooks", 1),
            quantizer_dropout=getattr(args.length_regulator, "quantizer_dropout", 0.0),
            f0_condition=getattr(args.length_regulator, "f0_condition", False),
            n_f0_bins=getattr(args.length_regulator, "n_f0_bins", 512),
        )

        if use_gpt_latent:
            self.models = nn.ModuleDict(
                {
                    "cfm": CFM(args),
                    "length_regulator": length_regulator,
                    "gpt_layer": torch.nn.Sequential(
                        torch.nn.Linear(1280, 256),
                        torch.nn.Linear(256, 128),
                        torch.nn.Linear(128, 1024),
                    ),
                }
            )
        else:
            self.models = nn.ModuleDict({"cfm": CFM(args), "length_regulator": length_regulator})

    def forward_gpt(self, x):
        x = self.models["gpt_layer"](x)
        return x

    def enable_torch_compile(self):
        if "cfm" in self.models:
            self.models["cfm"].enable_torch_compile()
