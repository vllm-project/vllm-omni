# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transformers HF config wrapper for Voxtream2.

This config exists so AutoConfig can resolve model_type='voxtream2'
for repos whose config.json is empty or missing required fields.
"""

from transformers.configuration_utils import PretrainedConfig


class Voxtream2HFConfig(PretrainedConfig):
    model_type = "voxtream2"

    def __init__(
        self,
        sample_rate: int = 24000,
        num_codebooks: int = 16,
        generator_config_path: str = "configs/generator.json",
        speaking_rate_config_path: str = "configs/speaking_rate.json",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.num_codebooks = num_codebooks
        self.generator_config_path = generator_config_path
        self.speaking_rate_config_path = speaking_rate_config_path
