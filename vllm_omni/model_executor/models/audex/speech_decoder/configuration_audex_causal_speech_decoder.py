# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
from __future__ import annotations

from transformers import PretrainedConfig


class AudexCausalSpeechDecoderConfig(PretrainedConfig):
    model_type = "audex_causal_speech_decoder"

    def __init__(
        self,
        hidden_dim: int = 2048,
        depth: int = 12,
        heads: int = 32,
        pos_meb_dim: int = 64,
        hop_length: int = 320,
        vq_dim: int = 2048,
        lookahead_steps: int = 4,
        sample_rate: int = 16000,
        codebook_levels: list[int] | None = None,
        codebook_size: int = 65536,
        token_embed_dim: int = 8,
        embed_tokens_from_codes: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.heads = heads
        self.pos_meb_dim = pos_meb_dim
        self.hop_length = hop_length
        self.vq_dim = vq_dim
        self.lookahead_steps = lookahead_steps
        self.sample_rate = sample_rate
        self.codebook_levels = codebook_levels or [4, 4, 4, 4, 4, 4, 4, 4]
        self.codebook_size = codebook_size
        self.token_embed_dim = token_embed_dim
        self.embed_tokens_from_codes = embed_tokens_from_codes
