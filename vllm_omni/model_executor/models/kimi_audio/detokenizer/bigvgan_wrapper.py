# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from MoonshotAI/Kimi-Audio (MIT), revision
# 349251e1d8f4f98d58fda59246381faecd7392e0, kimia_infer/models/detokenizer/.
# See vllm_omni/model_executor/models/kimi_audio/NOTICE for the upstream license.

"""Kimi acoustic loading and mel layout around Omni's existing BigVGAN."""

import json

import torch

from vllm_omni.model_executor.models.common.alias_free_activation import AliasFreeActivation1d
from vllm_omni.model_executor.models.indextts2.s2mel.modules.bigvgan import BigVGAN
from vllm_omni.model_executor.models.indextts2.s2mel.modules.commons import AttrDict


class BigVGANWrapper:
    def __init__(self, vocoder: BigVGAN, device: torch.device, h: AttrDict, dtype=None):
        self.vocoder = vocoder.to(device)
        if dtype is not None:
            self.vocoder = self.vocoder.to(dtype)
        self.vocoder = self.vocoder.eval()
        self.device = device
        self.h = h

    def to_dtype(self, dtype):
        self.vocoder = self.vocoder.to(dtype)

    def decode_mel(self, mel):
        """[T, num_mels] -> [1, samples], preserving the official layout."""
        mel = mel.transpose(0, 1).unsqueeze(0).to(self.device)
        return self.vocoder(mel).squeeze(0)

    @classmethod
    def from_pretrained(cls, model_config, ckpt_path, device):
        with open(model_config, encoding="utf-8") as f:
            h = AttrDict(json.load(f))
        vocoder = BigVGAN(h)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)["generator"]

        # The original saves these fixed filters as persistent buffers. Omni
        # generates non-persistent buffers and flattens downsample.lowpass.
        # Restore only those exact buffers; all learned keys remain strict.
        for name, module in vocoder.named_modules():
            if isinstance(module, AliasFreeActivation1d):
                for suffix, buffer in (
                    ("upsample.filter", module.upsample.filter),
                    ("downsample.lowpass.filter", module.downsample.filter),
                ):
                    key = f"{name}.{suffix}"
                    if key in state:
                        value = state.pop(key)
                        if value.shape != buffer.shape:
                            raise ValueError(f"Unexpected Kimi BigVGAN filter shape: {key}")
                        buffer.copy_(value)
        vocoder.load_state_dict(state, strict=True)
        return cls(vocoder, device, h)
