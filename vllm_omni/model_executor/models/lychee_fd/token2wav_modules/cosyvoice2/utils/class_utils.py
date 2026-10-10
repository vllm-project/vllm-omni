# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright [2023-11-28] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>
#            2024 Alibaba Inc (authors: Xiang Lyu)
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
import torch

from vllm_omni.model_executor.models.lychee_fd.token2wav_modules.cosyvoice2.transformer.attention import (
    RelPositionMultiHeadedAttention,
)
from vllm_omni.model_executor.models.lychee_fd.token2wav_modules.cosyvoice2.transformer.embedding import (
    EspnetRelPositionalEncoding,
)
from vllm_omni.model_executor.models.lychee_fd.token2wav_modules.cosyvoice2.transformer.subsampling import (
    LinearNoSubsampling,
)

COSYVOICE_ACTIVATION_CLASSES = {
    "hardtanh": torch.nn.Hardtanh,
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
    "selu": torch.nn.SELU,
    "swish": torch.nn.SiLU,
    "gelu": torch.nn.GELU,
}

COSYVOICE_SUBSAMPLE_CLASSES = {
    "linear": LinearNoSubsampling,
}

COSYVOICE_EMB_CLASSES = {
    "rel_pos_espnet": EspnetRelPositionalEncoding,
}

COSYVOICE_ATTENTION_CLASSES = {
    "rel_selfattn": RelPositionMultiHeadedAttention,
}
