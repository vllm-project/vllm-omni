# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
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
import torch.nn as nn
from torch.nn import functional as F

from vllm_omni.model_executor.models.lychee_fd.token2wav_modules.cosyvoice2.flow.flow_matching import (
    CausalConditionalCFM,
)
from vllm_omni.model_executor.models.lychee_fd.token2wav_modules.cosyvoice2.transformer.upsample_encoder_v2 import (
    UpsampleConformerEncoderV2,
)
from vllm_omni.model_executor.models.lychee_fd.token2wav_modules.cosyvoice2.utils.mask import make_pad_mask


class CausalMaskedDiffWithXvec(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        output_type: str = "mel",
        vocab_size: int = 5121,
        encoder: UpsampleConformerEncoderV2 = None,
        decoder: CausalConditionalCFM = None,
        input_embedding: torch.nn.Module = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.pre_lookahead_len = int(encoder.pre_lookahead_layer.pre_lookahead_len)
        self.up_rate = int(encoder.up_layer.stride)
        if input_embedding is None:
            self.input_embedding = nn.Embedding(vocab_size, input_size)
        else:
            self.input_embedding = input_embedding
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder

        # xvec projection with CUDA Graph optimization
        # 初始化 CUDA Graph 相关变量
        self.enable_cuda_graph = False
        self.static_embedding = None
        self.static_output = None
        self.graph = None
        self.embedding_shape = None

    def scatter_cuda_graph(self, enable_cuda_graph: bool):
        self.enable_cuda_graph = enable_cuda_graph
        if self.enable_cuda_graph:
            # self.encoder.scatter_cuda_graph(enable_cuda_graph)
            self.decoder.scatter_cuda_graph(enable_cuda_graph)

    @torch.inference_mode()
    def inference(
        self,
        token,
        token_len,
        prompt_token,
        prompt_token_len,
        prompt_feat,
        prompt_feat_len,
        embedding,
        n_timesteps: int = 10,
    ):
        assert token.shape[0] == 1

        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        token_len = prompt_token_len + token_len
        token = torch.concat([prompt_token, token], dim=1)

        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # token encode
        h, _ = self.encoder.forward(token, token_len)
        h = self.encoder_proj(h)

        # condition
        mel_len1 = prompt_feat.shape[1]
        mel_len2 = h.shape[1] - prompt_feat.shape[1]

        conds = torch.zeros_like(h)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2).contiguous()

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)

        feat = self.decoder.forward(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=n_timesteps,
        )

        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat

    @torch.inference_mode()
    def setup_cache(
        self,
        token: torch.Tensor,
        mel: torch.Tensor,
        spk: torch.Tensor,
        n_timesteps: int = 10,
    ):
        """
        Args:
            token: shape (b, t), with look ahead tokens
            mel: shape (b, t, c), groundtruth mel
            spk: shape (b, 192), speaker embedding
        Returns:
            cache: dict {
                'conformer': {'cnn_cache': xxx, 'att_cache': xxx},
                'estimator': {'cnn_cache': xxx, 'att_cache': xxx}
            }
        """
        # check if look ahead token included
        assert (token.shape[1] - self.pre_lookahead_len) * self.up_rate == mel.shape[1], (token.shape, mel.shape)

        # xvec projection
        spk = F.normalize(spk, dim=1)
        spk = self.spk_embed_affine_layer(spk)

        token = self.input_embedding(token)
        # NOTE encoder.forward_chunk will strip the look ahead part
        h, conformer_cnn_cache, conformer_att_cache = self.encoder.forward_chunk(
            xs=token,
            last_chunk=False,
            cnn_cache=None,
            att_cache=None,
        )
        h = self.encoder_proj(h)

        feat, estimator_cnn_cache, estimator_att_cache = self.decoder.forward_chunk(
            mu=h.transpose(1, 2).contiguous(),
            spks=spk,
            cond=mel.transpose(1, 2).contiguous(),
            n_timesteps=n_timesteps,
            temperature=1.0,
            cnn_cache=None,
            att_cache=None,
        )

        cache = {
            "conformer_cnn_cache": conformer_cnn_cache,
            "conformer_att_cache": conformer_att_cache,
            "estimator_cnn_cache": estimator_cnn_cache,
            "estimator_att_cache": estimator_att_cache,
        }
        return cache

    @torch.inference_mode()
    def inference_chunk(
        self,
        token: torch.Tensor,
        spk: torch.Tensor,
        cache: dict,
        last_chunk: bool = False,
        n_timesteps: int = 10,
    ):
        """
        Args:
            token: shape (b, t), with look ahead tokens
            spk: shape (b, 192), speaker embedding
            cache: dict {
                'conformer_cnn_cache': xxx,
                ...
            }
        """
        # unpack cache
        conformer_cnn_cache = cache["conformer_cnn_cache"]
        conformer_att_cache = cache["conformer_att_cache"]
        estimator_cnn_cache = cache["estimator_cnn_cache"]
        estimator_att_cache = cache["estimator_att_cache"]

        # xvec projection
        spk = F.normalize(spk, dim=1)
        spk = self.spk_embed_affine_layer(spk)

        token = self.input_embedding(token)
        # if not the last chunk, h is shorter than xs for a length of lookahead_length * stride (6)
        h, conformer_cnn_cache, conformer_att_cache = self.encoder.forward_chunk(
            xs=token,
            last_chunk=last_chunk,
            cnn_cache=conformer_cnn_cache,
            att_cache=conformer_att_cache,
        )
        h = self.encoder_proj(h)

        cond = torch.zeros_like(h)
        # forward estimator
        feat, estimator_cnn_cache, estimator_att_cache = self.decoder.forward_chunk(
            mu=h.transpose(1, 2).contiguous(),
            spks=spk,
            cond=cond.transpose(1, 2).contiguous(),
            n_timesteps=n_timesteps,
            temperature=1.0,
            cnn_cache=estimator_cnn_cache,
            att_cache=estimator_att_cache,
        )

        new_cache = {
            "conformer_cnn_cache": conformer_cnn_cache,
            "conformer_att_cache": conformer_att_cache,
            "estimator_cnn_cache": estimator_cnn_cache,
            "estimator_att_cache": estimator_att_cache,
        }

        return feat, new_cache
