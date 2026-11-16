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

# Vendored from NVIDIA-NeMo/Speech (branch nemotron-labs-voicechat); dependency-stripped
# copies of the NeMo inference modules needed to run NVIDIA-NemotronLabs-VoiceChat-11B.

from .duplex_ear_tts import DuplexEARTTS
from .ear_tts_commons import PreTrainedModel
from .ear_tts_vae_codec import RVQVAEModel
from .fusion import create_fusion_module
from .perception import AudioPerceptionModule, IdentityConnector
from .tokenizer import AutoTokenizer

__all__ = [
    "AudioPerceptionModule",
    "AutoTokenizer",
    "DuplexEARTTS",
    "IdentityConnector",
    "PreTrainedModel",
    "RVQVAEModel",
    "create_fusion_module",
]
