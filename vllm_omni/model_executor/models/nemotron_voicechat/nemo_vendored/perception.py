# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Vendored from NVIDIA-NeMo/Speech (branch nemotron-labs-voicechat) nemo/collections/speechlm2/modules/perception.py; training-only code removed.

import torch
from omegaconf import DictConfig
from torch import nn

from .compat import (
    Exportable,
    NeuralModule,
    typecheck,
)


class AudioPerceptionModule(NeuralModule, Exportable):
    """Audio perception module that consists of audio encoder(s) and modality adapter."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        # Initialize components
        self.cfg = cfg
        self.preprocessor = self.from_config_dict(cfg.preprocessor)
        self.encoder = self.from_config_dict(cfg.encoder)

        if "spec_augment" in cfg and cfg.spec_augment is not None:
            self.spec_augmentation = self.from_config_dict(cfg.spec_augment)
        else:
            self.spec_augmentation = None
        self.modality_adapter = self.from_config_dict(cfg.modality_adapter)
        if "output_dim" not in cfg.modality_adapter and "d_model" in cfg.modality_adapter:  # e.g., conformer encoder
            self.proj = nn.Linear(cfg.modality_adapter.d_model, cfg.output_dim)
        else:
            self.proj = nn.Identity()

        self.modality_adapter_quantizer_levels = cfg.get("modality_adapter_quantizer_levels", None)
        if self.modality_adapter_quantizer_levels:
            # FiniteScalarQuantizer (nemo.collections.tts.modules.audio_codec_modules) was not
            # vendored: `modality_adapter_quantizer_levels` is unset in the
            # NVIDIA-NemotronLabs-VoiceChat-11B checkpoint config.json.
            raise RuntimeError(
                "modality_adapter_quantizer_levels is set in the perception config, but "
                "FiniteScalarQuantizer was not vendored (it is unused by the "
                "NVIDIA-NemotronLabs-VoiceChat-11B checkpoint). Vendor "
                "nemo.collections.tts.modules.audio_codec_modules.FiniteScalarQuantizer to enable it."
            )

    def maybe_preprocess_audio(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
    ):
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) is False:
            raise ValueError(
                f"{self.__class__} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )
        return processed_signal, processed_signal_length

    # disable type checks to avoid type-check errors when using Conformer as modality adapter
    @typecheck.disable_checks()
    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        return_encoder_emb=False,
        sample_id=None,
    ):
        processed_signal, processed_signal_length = self.maybe_preprocess_audio(
            input_signal, input_signal_length, processed_signal, processed_signal_length
        )

        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoder_emb, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)

        if self.modality_adapter_quantizer_levels is not None:
            encoded = self.modality_adapter_quantizer_bottleneck(encoder_emb.transpose(1, 2))
            encoded, _ = self.modality_adapter_vector_quantizer(inputs=encoded.transpose(1, 2), input_len=None)
            encoded = self.modality_adapter_quantizer_projection(encoded.transpose(1, 2)).transpose(1, 2)
        else:
            encoded = encoder_emb

        encoded, encoded_len = self.modality_adapter(audio_signal=encoded, length=encoded_len)

        # encoded_orig = encoded.clone()

        # b, c, t -> b, t, c
        encoded = self.proj(encoded.transpose(1, 2))

        if return_encoder_emb:
            return encoded, encoded_len, encoder_emb.transpose(1, 2)
        else:
            return encoded, encoded_len


class IdentityConnector(NeuralModule, Exportable):
    """User to pass encoder's representations as-is to the LLM."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()

    def forward(self, audio_signal, length=None, *args, **kwargs):
        return audio_signal, length
