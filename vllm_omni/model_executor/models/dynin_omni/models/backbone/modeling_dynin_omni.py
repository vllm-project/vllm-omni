# coding=utf-8
# Copyright 2026 Dynin-Omni Team, AIDAS Lab, Seoul National University
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

import logging
import math
import sys
import warnings
from abc import abstractmethod
from collections import defaultdict
from functools import partial
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)
from dataclasses import fields
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto import AutoModel, AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import Cache
from PIL import Image
from .configuration_llada import (
    LLaDAConfig,
    StrEnum,
    InitFnType,
    ActivationType,
    BlockType,
    LayerNormType,
    ModelConfig,
    ActivationCheckpointingStrategy,
)

from .modeling_llada import LLaDAModelLM
from .sampling import cosine_schedule, mask_by_random_topk
from transformers import PretrainedConfig

try:
    from fastdllm_v1 import mmu_generate_fastdllm_v1 as _mmu_generate_fastdllm_v1
except ModuleNotFoundError:
    _mmu_generate_fastdllm_v1 = None

logger = logging.getLogger(__name__)

def calculate_mmu_style_loss(logits_batch, labels_batch, masked_indices_batch, p_mask, answer_lengths, output_size, device):
    if logits_batch.shape[0] == 0:
        return logits_batch.new_zeros(())

    p_mask_flat = p_mask.to(device)[masked_indices_batch]
    p_mask_flat = torch.clamp(p_mask_flat, min=1e-4)
    answer_lengths_flat = answer_lengths.to(device)[masked_indices_batch]
    answer_lengths_flat = torch.clamp(answer_lengths_flat, min=1)

    loss = F.cross_entropy(
        logits_batch[masked_indices_batch].contiguous().view(-1, output_size),
        labels_batch[masked_indices_batch].contiguous().view(-1), ignore_index=-100, reduction='none'
    ) / p_mask_flat

    loss = torch.sum(loss / answer_lengths_flat) / logits_batch.shape[0]
    return loss


def calculate_t2s_loss(
    logits_batch,
    labels_batch,
    masked_indices_batch,
    p_mask,
    answer_lengths,
    vocab_start,
    codebook_size,
    eoa_token_id,
    eos_token_id,
    device,
    ignore_index=-100,
):
    if logits_batch.shape[0] == 0:
        return logits_batch.new_zeros(())

    selected_logits = logits_batch[masked_indices_batch]
    selected_labels = labels_batch[masked_indices_batch].to(torch.long)

    if selected_logits.shape[0] == 0:
        return logits_batch.new_zeros(())

    work_dtype = torch.float32
    selected_logits_fp32 = selected_logits.to(dtype=work_dtype)

    speech_logits = selected_logits_fp32[:, vocab_start : vocab_start + codebook_size]
    eoa_logits = selected_logits_fp32[:, eoa_token_id : eoa_token_id + 1]
    eos_logits = selected_logits_fp32[:, eos_token_id : eos_token_id + 1]
    combined_logits = torch.cat([speech_logits, eoa_logits, eos_logits], dim=-1)

    p_mask_flat = p_mask.to(device=device, dtype=work_dtype)[masked_indices_batch]
    p_mask_flat = torch.clamp(p_mask_flat, min=1e-4)
    answer_lengths_flat = answer_lengths.to(device=device, dtype=work_dtype)[masked_indices_batch]
    answer_lengths_flat = torch.clamp(answer_lengths_flat, min=1.0)

    relative_labels = torch.full_like(selected_labels, ignore_index)
    audio_mask = (selected_labels >= vocab_start) & (selected_labels < vocab_start + codebook_size)
    relative_labels[audio_mask] = selected_labels[audio_mask] - vocab_start
    relative_labels[selected_labels == eoa_token_id] = codebook_size
    relative_labels[selected_labels == eos_token_id] = codebook_size + 1

    loss_vec = F.cross_entropy(
        combined_logits,
        relative_labels,
        ignore_index=ignore_index,
        reduction='none'
    )

    loss_vec = loss_vec / p_mask_flat
    loss_vec = loss_vec / answer_lengths_flat

    loss = torch.sum(loss_vec) / logits_batch.shape[0]
    return loss.to(dtype=logits_batch.dtype)

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

class DyninOmniConfig(PretrainedConfig):
    model_type = "dynin_omni"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        allowed_keys = [
            "vocab_size",
            "llm_vocab_size",
            "llm_model_path",
            "codebook_size",
            "num_vq_tokens",
            "num_new_special_tokens",
            "gradient_checkpointing",
            "new_vocab_size",
            "vid_merge",
            "vid_merge_stride",
        ]

        for key in allowed_keys:
            if key in kwargs:
                setattr(self, key, kwargs[key])
        if not hasattr(self, "vid_merge"):
            self.vid_merge = False
        if not hasattr(self, "vid_merge_stride"):
            self.vid_merge_stride = 2


class VideoTokenMerger(nn.Module):
    def __init__(self, embed_dim: int, stride: int = 2):
        super().__init__()
        self.stride = stride
        hidden_size = embed_dim * stride * stride
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class DyninOmniModelLM(LLaDAModelLM):
    config_class = DyninOmniConfig
    base_model_prefix = "model"
    def __init__(self, config: DyninOmniConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.video_token_merger = None

    def enable_video_merge(self, stride: int = 2):
        embed_weight = self.get_input_embeddings().weight
        merger = VideoTokenMerger(embed_weight.shape[1], stride=stride).to(
            device=embed_weight.device,
            dtype=embed_weight.dtype,
        )
        self.video_token_merger = merger
        self.config.vid_merge = True
        self.config.vid_merge_stride = stride
        return merger
        
    @torch.no_grad()
    def t2i_generate(
            self,
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            uncond_attention_mask=None,
            temperature=1.0,
            timesteps=18,  # ideal number of steps is 18 in maskgit paper
            guidance_scale=0,
            noise_schedule=cosine_schedule,
            generator: torch.Generator = None,
            config=None,
            seq_len=1024,
            mask_token_id = 126336,
            resolution = 512,
            codebook_size = 8192,
            **kwargs,
    ):
        """
        Generate 1:1 similar to the original MaskGit repo
        https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
        """

        mask_count = (input_ids == mask_token_id).sum().item()
        num_vq_tokens = seq_len
        num_new_special_tokens = 0
        uni_prompting = kwargs.get("uni_prompting", None)
        input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()
        input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id, mask_token_id, input_ids_minus_lm_vocab_size - len(uni_prompting.text_tokenizer) - num_new_special_tokens)

        # for classifier-free guidance
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, :resolution + 1]

        for step in range(timesteps):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, resolution + 1:]], dim=1)
                model_input = torch.cat([input_ids, uncond_input_ids])
                all_attention_mask = torch.cat([attention_mask, uncond_attention_mask], dim=0)
                attention_bias = (all_attention_mask[:, :, None] & all_attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(model_input, attention_bias=attention_bias).logits
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has a different cfg setting
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]
            else:
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(input_ids, attention_bias=attention_bias).logits
                logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]

            # logits: 1, 1024, 8192
            probs = logits.softmax(dim=-1)
            sampled = probs.reshape(-1, logits.size(-1))
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1]) # 1, 1024

            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
            # Defines the mask ratio for the next round. The number to mask out is
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))
            # Computes the probabilities of each selected tokens.
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
            selected_probs = selected_probs.squeeze(-1)

            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
            # Gets mask lens for each sample in the batch according to the mask ratio.
            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            mask_len = torch.max(
                torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )
            # Adds noise for randomness
            temperature = temperature * (1.0 - ratio)
            masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
            # Masks tokens with lower confidence.
            input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id,
                                                          sampled_ids + len(uni_prompting.text_tokenizer)
                                                          + num_new_special_tokens)
            input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)

        return sampled_ids

    @torch.no_grad()
    def ti2ti_generate(
            self,
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            uncond_attention_mask=None,
            temperature=1.0,
            timesteps=18,
            timesteps_text: int | None = None,
            timesteps_image: int | None = None,
            guidance_scale=0,
            noise_schedule=cosine_schedule,
            generator: torch.Generator = None,
            config=None,
            seq_len=1024,
            mask_token_id=126336,
            resolution=512,
            codebook_size=8192,
            uni_prompting=None,
            **kwargs,
    ):
        """
        TI2TI generation that fills masked text and image tokens; allows separate timesteps.
        Returns (filled_tokens, decoded_texts).
        """
        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask are required for ti2ti_generate.")
        if uni_prompting is None:
            raise ValueError("uni_prompting is required for ti2ti_generate.")

        device = input_ids.device
        text_vocab_size = len(uni_prompting.text_tokenizer)
        image_vocab_start = text_vocab_size
        image_vocab_end = image_vocab_start + codebook_size
        timesteps_text = timesteps if timesteps_text is None else timesteps_text
        timesteps_image = timesteps if timesteps_image is None else timesteps_image

        seq = input_ids.clone()
        if attention_mask is None:
            attn = torch.ones_like(seq, dtype=torch.long)
        else:
            attn = attention_mask
        use_guidance = uncond_input_ids is not None and guidance_scale > 0
        if use_guidance:
            seq_uncond = uncond_input_ids.clone()
            if uncond_attention_mask is None:
                attn_uncond = torch.ones_like(seq_uncond, dtype=torch.long)
            else:
                attn_uncond = uncond_attention_mask
        else:
            seq_uncond = None
            attn_uncond = None
        total_len = seq.shape[1]

        def _uniform_transfer_plan(mask_bool: torch.Tensor, steps_count: int) -> Optional[torch.Tensor]:
            """Evenly divide masked token updates across steps."""
            if steps_count is None or steps_count <= 0:
                return None
            mask_num = mask_bool.sum(dim=1, keepdim=True)
            if mask_num.numel() == 0:
                return None
            base = mask_num // steps_count
            remainder = mask_num % steps_count
            plan = torch.zeros(mask_num.size(0), steps_count, device=mask_bool.device, dtype=torch.int64) + base
            for idx in range(mask_num.size(0)):
                rem_val = remainder[idx].item()
                if rem_val > 0:
                    plan[idx, :rem_val] += 1
            return plan

        prompt_block_len = uni_prompting.max_text_len
        soi_id = int(uni_prompting.sptids_dict.get("<|soi|>", torch.tensor([-1]))[0].item())
        eoi_id = int(uni_prompting.sptids_dict.get("<|eoi|>", torch.tensor([-1]))[0].item())
        pad_id = int(getattr(uni_prompting, "pad_id", 0))

        def _locate_blocks(sample_seq: torch.Tensor, sample_attn: Optional[torch.Tensor]):
            # Find second (target) soi/eoi pair; fallback to template formula.
            soi_positions = (sample_seq == soi_id).nonzero(as_tuple=True)[0]
            eoi_positions = (sample_seq == eoi_id).nonzero(as_tuple=True)[0]
            tgt_soi = None
            tgt_eoi = None
            if soi_positions.numel() >= 2:
                tgt_soi = int(soi_positions[1].item())
                tgt_eoi_candidates = [int(e.item()) for e in eoi_positions if int(e.item()) > tgt_soi]
                if tgt_eoi_candidates:
                    tgt_eoi = tgt_eoi_candidates[0]

            if tgt_soi is None or tgt_eoi is None:
                # fallback: compute with pad offset the old way
                non_pad = (sample_seq != pad_id).nonzero(as_tuple=True)
                pad_offset = int(non_pad[0][0].item()) if len(non_pad) > 0 and non_pad[0].numel() > 0 else 0
                tgt_soi = pad_offset + 1 + 1 + seq_len + 1 + prompt_block_len + 1  # soi before target img
                tgt_eoi = tgt_soi + seq_len + 1  # eoi after target img

            img_start_local = tgt_soi + 1
            img_end_local = min(tgt_eoi, sample_seq.size(0))

            if sample_attn is not None:
                text_attn = sample_attn[tgt_eoi + 1 :]
                nonzero = (text_attn != 0).nonzero(as_tuple=True)
                if len(nonzero) > 0 and nonzero[0].numel() > 0:
                    last_idx = int(nonzero[0][-1].item())
                    text_end_local = tgt_eoi + 1 + last_idx + 1
                else:
                    text_end_local = tgt_eoi + 1 + prompt_block_len
            else:
                text_end_local = tgt_eoi + 1 + prompt_block_len
            text_start_local = tgt_eoi + 1
            text_end_local = min(text_end_local, sample_seq.size(0))
            return img_start_local, img_end_local, text_start_local, text_end_local

        img_start, img_end, text_start, text_end = _locate_blocks(seq[0], attn[0] if attn is not None else None)
        text_indices = torch.arange(total_len, device=device)
        initial_text_mask = (seq == mask_token_id) & (text_indices >= text_start) & (text_indices < text_end)
        text_transfer_plan = _uniform_transfer_plan(initial_text_mask, timesteps_text)
        text_step_idx = 0

        # Simultaneous fill: at each step, update image/text masks that still remain
        max_steps = max(timesteps_image, timesteps_text)
        for step in range(max_steps):
            mask_map = seq == mask_token_id
            img_mask = mask_map & (text_indices >= img_start) & (text_indices < img_end) if step < timesteps_image else None
            text_mask = mask_map & (text_indices >= text_start) & (text_indices < text_end) if step < timesteps_text else None
            if not ((img_mask is not None and img_mask.any()) or (text_mask is not None and text_mask.any())):
                break

            attn_bias = (attn[:, :, None] & attn[:, None, :]).bool().unsqueeze(1)
            logits_cond = self(seq, attention_bias=attn_bias).logits
            if use_guidance:
                attn_bias_uncond = (attn_uncond[:, :, None] & attn_uncond[:, None, :]).bool().unsqueeze(1)
                logits_uncond = self(seq_uncond, attention_bias=attn_bias_uncond).logits
                logits = logits_uncond + (guidance_scale + 1.0) * (logits_cond - logits_uncond)
            else:
                logits = logits_cond

            if text_mask is not None and text_mask.any():
                logits_text = logits[..., :text_vocab_size]
                probs_text = logits_text.softmax(dim=-1)
                sampled_text = torch.multinomial(
                    probs_text.view(-1, text_vocab_size),
                    1,
                    replacement=False
                ).view(*logits_text.shape[:2])
                sampled_probs = torch.gather(
                    probs_text, dim=-1, index=sampled_text.unsqueeze(-1)
                ).squeeze(-1)
                candidate_seq = torch.where(text_mask, sampled_text, seq)
                confidence = torch.full_like(sampled_probs, float("-inf"))
                confidence = torch.where(text_mask, sampled_probs, confidence)
                if text_transfer_plan is not None and text_step_idx < text_transfer_plan.shape[1]:
                    transfer_counts = text_transfer_plan[:, text_step_idx]
                else:
                    transfer_counts = text_mask.sum(dim=1)
                transfer_mask = torch.zeros_like(text_mask, dtype=torch.bool)
                for b_idx in range(seq.shape[0]):
                    mask_count = int(text_mask[b_idx].sum().item())
                    if mask_count == 0:
                        continue
                    k = int(min(max(transfer_counts[b_idx].item(), 0), mask_count))
                    if k <= 0:
                        continue
                    _, top_idx = torch.topk(confidence[b_idx], k=k)
                    transfer_mask[b_idx, top_idx] = True
                if transfer_mask.any():
                    seq = torch.where(transfer_mask, candidate_seq, seq)
                text_step_idx += 1

            if img_mask is not None and img_mask.any():
                logits_img = logits[..., image_vocab_start:image_vocab_end]
                probs_img = logits_img.softmax(dim=-1)
                sampled_img = torch.multinomial(
                    probs_img.view(-1, codebook_size),
                    1,
                    replacement=False
                ).view(*logits_img.shape[:2]) + image_vocab_start
                seq = torch.where(img_mask, sampled_img, seq)

            if use_guidance:
                updated_mask = torch.zeros_like(seq, dtype=torch.bool)
                if img_mask is not None:
                    updated_mask |= img_mask
                if text_mask is not None:
                    updated_mask |= text_mask
                seq_uncond = torch.where(updated_mask, seq, seq_uncond)

        # Decode text tokens from filled sequence
        pred_texts = []
        for row in seq:
            text_tokens = [int(t) for t in row.tolist() if 0 <= t < text_vocab_size]
            pred_texts.append(uni_prompting.text_tokenizer.decode(text_tokens, skip_special_tokens=True))

        return seq, pred_texts

    @torch.no_grad()
    def t2s_generate(
            self,
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            uncond_attention_mask=None,
            temperature=1.0,
            timesteps=18,
            guidance_scale=0,
            noise_schedule=None,
            generator: torch.Generator = None,
            config=None,
            seq_len=256,
            mask_token_id=126336,
            **kwargs,
    ):
        uni_prompting = kwargs.get("uni_prompting", None)
        if uni_prompting is None:
            raise ValueError("uni_prompting object must be provided in kwargs.")
        
        eoa_token_id = uni_prompting.sptids_dict['<|eoa|>'][0].item()
        eos_token_id = uni_prompting.text_tokenizer.eos_token_id

        num_vq_tokens = (input_ids == mask_token_id).sum(dim=-1).max().item()
        if num_vq_tokens == 0:
            raise ValueError("No mask tokens found in input_ids.")

        speech_vocab_start_idx = len(uni_prompting.text_tokenizer) + 8192
        speech_vocab_end_idx = speech_vocab_start_idx + 4096
        
        # VQ Codes: 0 ~ 4095
        # EOA: 4096
        # EOS: 4097
        vq_code_relative_eoa_id = 4096 
        vq_code_relative_eos_id = 4097

        input_ids_relative = input_ids[:, -(num_vq_tokens):].clone()
        input_ids_relative = torch.where(
            input_ids_relative == mask_token_id,
            mask_token_id,
            input_ids_relative - speech_vocab_start_idx
        )

        if uncond_input_ids is not None:
            start_gen_idx = (uncond_input_ids[0] == uni_prompting.sptids_dict['<|soa|>'][0].item()).nonzero(as_tuple=True)[0][0].item() + 1
            uncond_prefix = uncond_input_ids[:, :start_gen_idx]

        for step in range(timesteps):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat([uncond_prefix, input_ids[:, start_gen_idx:]], dim=1)
                model_input = torch.cat([input_ids, uncond_input_ids])
                all_attention_mask = torch.cat([attention_mask, uncond_attention_mask], dim=0)
                
                attention_bias = (all_attention_mask[:, :, None] & all_attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(model_input, attention_bias=attention_bias).logits
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                
            else:
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(input_ids, attention_bias=attention_bias).logits

            logits_vq = logits[:, -(num_vq_tokens):, speech_vocab_start_idx:speech_vocab_end_idx]
            logits_eoa = logits[:, -(num_vq_tokens):, eoa_token_id:eoa_token_id+1]
            logits_eos = logits[:, -(num_vq_tokens):, eos_token_id:eos_token_id+1]
            
            combined_logits = torch.cat([logits_vq, logits_eoa, logits_eos], dim=-1)
            
            probs = combined_logits.softmax(dim=-1)
            sampled = probs.reshape(-1, combined_logits.size(-1))
            
            sampled_ids_relative = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*combined_logits.shape[:-1])
            
            unknown_map = input_ids_relative == mask_token_id
            
            sampled_ids_relative = torch.where(unknown_map, sampled_ids_relative, input_ids_relative)

            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio, device=logits.device))
            
            selected_probs = torch.gather(probs, -1, sampled_ids_relative.long()[..., None]).squeeze(-1)
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)

            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
            mask_len = torch.max(
                torch.tensor([1], device=logits.device), 
                torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )
            
            temperature = temperature * (1.0 - ratio)
            masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)

            input_ids[:, -(num_vq_tokens):] = torch.where(
                masking, 
                mask_token_id,
                torch.where(
                    sampled_ids_relative == vq_code_relative_eos_id, 
                    eos_token_id,                                    
                    torch.where(
                        sampled_ids_relative == vq_code_relative_eoa_id, 
                        eoa_token_id,                                    
                        sampled_ids_relative + speech_vocab_start_idx    
                    )
                )
            )
            
            input_ids_relative = torch.where(masking, mask_token_id, sampled_ids_relative)

        final_output_ids = []
        for i in range(input_ids_relative.shape[0]):
            seq = input_ids_relative[i]
            
            eoa_indices = (seq >= vq_code_relative_eoa_id).nonzero(as_tuple=True)[0]
            
            if eoa_indices.numel() > 0:
                first_eoa_idx = eoa_indices[0]
                seq = seq[:first_eoa_idx]
            
            valid_tokens = seq[seq != mask_token_id]
            
            final_output_ids.append(valid_tokens)
        
        return final_output_ids

    @torch.no_grad()
    def t2s_generate_mmu_like(
            self,
            input_ids: torch.LongTensor,
            max_new_tokens: Optional[int] = None,
            steps: int = 256,
            block_length: int = 128,
            temperature: float = 0.0,
            cfg_scale: float = 0.0,
            mask_token_id: int = 126336,
            attention_mask: Optional[torch.LongTensor] = None,
            uni_prompting=None,
            codebook_size: Optional[int] = None,
            audio_codebook_size: int = 4096,
    ):
        """
        Generate speech tokens with MMU-style block-wise refinement.
        Assumes the speech region within ``input_ids`` is contiguous and filled with ``mask_token_id``
        prior to generation.
        """

        if uni_prompting is None:
            raise ValueError("uni_prompting must be provided")
        if block_length <= 0:
            raise ValueError("block_length must be positive")

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        mask_positions_full = (input_ids == mask_token_id)
        if not mask_positions_full.any():
            raise ValueError("No mask tokens detected for T2S generation")

        mask_cols = torch.where(mask_positions_full[0])[0]
        speech_region_start = mask_cols[0].item()
        speech_region_len = mask_cols.numel()

        mask_counts = mask_positions_full.sum(dim=1)
        if not torch.all(mask_counts == mask_counts[0]):
            raise ValueError("All batch items must contain the same number of masked speech tokens for MMU-like generation")

        if max_new_tokens is None:
            max_new_tokens = speech_region_len
        else:
            max_new_tokens = min(max_new_tokens, speech_region_len)

        block_length = max(1, min(block_length, max_new_tokens))
        num_blocks = math.ceil(max_new_tokens / block_length)
        inner_steps = max(1, steps // num_blocks)

        codebook_base = codebook_size if codebook_size is not None else getattr(self.config, "codebook_size", 8192)
        speech_vocab_start = len(uni_prompting.text_tokenizer) + codebook_base
        speech_vocab_end = speech_vocab_start + audio_codebook_size

        eoa_token_id = uni_prompting.sptids_dict['<|eoa|>'][0].item()
        eos_token_id = uni_prompting.text_tokenizer.eos_token_id
        vq_code_relative_eoa_id = audio_codebook_size
        vq_code_relative_eos_id = audio_codebook_size + 1

        work = input_ids.clone()

        attention_bias = None
        if attention_mask is not None:
            attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)

        speech_indices = mask_cols[:max_new_tokens]

        for block_idx in range(num_blocks):
            block_start = block_idx * block_length
            block_end = min(block_start + block_length, max_new_tokens)
            curr_indices = speech_indices[block_start:block_end]
            if curr_indices.numel() == 0:
                continue

            block_mask = mask_positions_full[:, curr_indices]
            num_transfer_tokens = get_num_transfer_tokens(block_mask, inner_steps)

            for inner_step in range(inner_steps):
                if cfg_scale > 0.0:
                    un_cond = work.clone()
                    un_cond[:, speech_indices] = mask_token_id
                    stacked = torch.cat([work, un_cond], dim=0)
                    if attention_bias is not None:
                        att_bias = torch.cat([attention_bias, attention_bias], dim=0)
                    else:
                        att_bias = None
                    logits = self(stacked, attention_bias=att_bias).logits
                    cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                    logits = uncond_logits + (cfg_scale + 1.0) * (cond_logits - uncond_logits)
                else:
                    logits = self(work, attention_bias=attention_bias).logits

                logits_block = logits.index_select(1, curr_indices.to(device))
                logits_vq = logits_block[:, :, speech_vocab_start:speech_vocab_end]
                logits_eoa = logits_block[:, :, eoa_token_id:eoa_token_id + 1]
                logits_eos = logits_block[:, :, eos_token_id:eos_token_id + 1]

                combined_logits = torch.cat([logits_vq, logits_eoa, logits_eos], dim=-1)
                if temperature > 0.0:
                    combined_logits = combined_logits / max(temperature, 1e-5)
                probs = F.softmax(combined_logits, dim=-1)

                sampled = torch.multinomial(
                    probs.view(-1, probs.size(-1)), 1
                ).view(batch_size, curr_indices.numel())

                selected_probs = torch.gather(probs, -1, sampled.unsqueeze(-1)).squeeze(-1)

                eos_tensor = sampled.new_full(sampled.shape, eos_token_id)
                eoa_tensor = sampled.new_full(sampled.shape, eoa_token_id)
                sampled_absolute = torch.where(
                    sampled == vq_code_relative_eos_id,
                    eos_tensor,
                    torch.where(
                        sampled == vq_code_relative_eoa_id,
                        eoa_tensor,
                        sampled + speech_vocab_start
                    )
                )

                current_block_vals = work.index_select(1, curr_indices)
                mask_current = current_block_vals == mask_token_id

                confidence = torch.where(
                    mask_current,
                    selected_probs,
                    torch.full_like(selected_probs, float('-inf'))
                )

                finalize = torch.zeros_like(mask_current, dtype=torch.bool)
                for b in range(batch_size):
                    available = mask_current[b].sum().item()
                    if available == 0:
                        continue
                    transfer = min(int(num_transfer_tokens[b, inner_step].item()), available)
                    if transfer <= 0:
                        continue
                    _, idxs = torch.topk(confidence[b], k=transfer, largest=True)
                    finalize[b, idxs] = True

                mask_fill = sampled_absolute.new_full(sampled_absolute.shape, mask_token_id)
                updates = torch.where(finalize, sampled_absolute, mask_fill)
                new_block = torch.where(mask_current, updates, current_block_vals)

                work[:, curr_indices] = new_block
                mask_positions_full[:, curr_indices] = new_block == mask_token_id

                if not mask_positions_full[:, curr_indices].any():
                    break

        final_outputs = []
        audio_slice = slice(speech_region_start, speech_region_start + speech_region_len)
        audio_region = work[:, audio_slice]

        for seq in audio_region:
            mask_tensor = seq.new_full(seq.shape, mask_token_id)
            rel_eoa = seq.new_full(seq.shape, vq_code_relative_eoa_id)
            rel_eos = seq.new_full(seq.shape, vq_code_relative_eos_id)
            relative = torch.where(
                seq == mask_token_id,
                mask_tensor,
                torch.where(
                    seq == eoa_token_id,
                    rel_eoa,
                    torch.where(
                        seq == eos_token_id,
                        rel_eos,
                        seq - speech_vocab_start
                    )
                )
            )

            eoa_positions = (relative >= vq_code_relative_eoa_id).nonzero(as_tuple=True)[0]
            if eoa_positions.numel() > 0:
                relative = relative[:eoa_positions[0]]

            final_outputs.append(relative[relative != mask_token_id])

        return final_outputs

    @torch.no_grad()
    def t2s_fixed_generate(
            self,
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            uncond_attention_mask=None,
            temperature=1.0,
            timesteps=18,
            guidance_scale=0,
            noise_schedule=None,
            generator: torch.Generator = None,
            config=None,
            seq_len=256,
            mask_token_id=126336,
            **kwargs,
    ):
        """
        Generate 1:1 similar to the original MaskGit repo
        https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
        """

        # begin with all image token ids masked
        # 计算有多少个mask token
        mask_count = (input_ids == mask_token_id).sum().item()
        num_vq_tokens = seq_len
        num_new_special_tokens = 0
        uni_prompting = kwargs.get("uni_prompting", None)
        input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()
        input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id, mask_token_id, input_ids_minus_lm_vocab_size - len(uni_prompting.text_tokenizer) - num_new_special_tokens - 8192)

        # for classifier-free guidance
        if uncond_input_ids is not None:
            start_gen_idx = (uncond_input_ids[0] == uni_prompting.sptids_dict['<|soa|>'][0].item()).nonzero(as_tuple=True)[0][0].item() + 1
            uncond_prefix = uncond_input_ids[:, :start_gen_idx]

        for step in range(timesteps):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, start_gen_idx:]], dim=1)
                model_input = torch.cat([input_ids, uncond_input_ids])
                all_attention_mask = torch.cat([attention_mask, uncond_attention_mask], dim=0)
                attention_bias = (all_attention_mask[:, :, None] & all_attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(model_input, attention_bias=attention_bias).logits 
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has a different cfg setting
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens + 8192 : len(uni_prompting.text_tokenizer) + num_new_special_tokens + 8192 + 4096]
            else:
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(input_ids, attention_bias=attention_bias).logits
                logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens + 8192 : len(uni_prompting.text_tokenizer) + num_new_special_tokens + 8192 + 4096]

            # logits: 1, 1024, 8192
            probs = logits.softmax(dim=-1)
            sampled = probs.reshape(-1, logits.size(-1))
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1]) # 1, 1024

            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
            # Defines the mask ratio for the next round. The number to mask out is
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))
            # Computes the probabilities of each selected tokens.
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
            selected_probs = selected_probs.squeeze(-1)

            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
            # Gets mask lens for each sample in the batch according to the mask ratio.
            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            mask_len = torch.max(
                torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )
            # Adds noise for randomness
            temperature = temperature * (1.0 - ratio)
            masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
            # Masks tokens with lower confidence.
            input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id,
                                                          sampled_ids + len(uni_prompting.text_tokenizer)
                                                          + num_new_special_tokens + 8192)
            input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)

        return sampled_ids

    @torch.no_grad()
    def i2i_generate( 
            self,
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            uncond_attention_mask=None,
            temperature=1.0,
            timesteps=64,
            guidance_scale=0,
            noise_schedule=cosine_schedule,
            generator: torch.Generator = None,
            config=None,
            seq_len=1024,
            mask_token_id = 126336,
            resolution = 336,
            codebook_size = 8192,
            **kwargs,
    ):
        """
        Generate 1:1 similar to the original MaskGit repo
        https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
        """

        # begin with all image token ids masked
        mask_count = (input_ids == mask_token_id).sum().item()
        num_vq_tokens = seq_len
        num_new_special_tokens = 0
        uni_prompting = kwargs.get("uni_prompting", None)
        input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()
        input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id, mask_token_id, input_ids_minus_lm_vocab_size - len(uni_prompting.text_tokenizer) - num_new_special_tokens)

        # for classifier-free guidance
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, :resolution + 1]

        for step in range(timesteps):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, resolution + 1:]], dim=1)
                model_input = torch.cat([input_ids, uncond_input_ids])
                all_attention_mask = torch.cat([attention_mask, uncond_attention_mask], dim=0)
                attention_bias = (all_attention_mask[:, :, None] & all_attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(model_input, attention_bias=attention_bias).logits 
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has a different cfg setting
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]
            else:
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(input_ids, attention_bias=attention_bias).logits
                logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]

            # logits: 1, 1024, 8192
            probs = logits.softmax(dim=-1)
            sampled = probs.reshape(-1, logits.size(-1))
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1]) # 1, 1024

            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
            # Defines the mask ratio for the next round. The number to mask out is
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))
            # Computes the probabilities of each selected tokens.
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
            selected_probs = selected_probs.squeeze(-1)

            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
            # Gets mask lens for each sample in the batch according to the mask ratio.
            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            mask_len = torch.max(
                torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )
            # Adds noise for randomness
            temperature = temperature * (1.0 - ratio)
            masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
            # Masks tokens with lower confidence.
            input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id,
                                                          sampled_ids + len(uni_prompting.text_tokenizer)
                                                          + num_new_special_tokens)
            input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)

        return sampled_ids


    def forward_process(
            self,
            input_ids,
            inputs_embeds=None,
            labels=None,
            batch_size_t2i=0,
            batch_size_i2i=0,
            batch_size_ti2ti=0,
            batch_size_lm=0,
            batch_size_mmu=0,
            batch_size_v2t=0,
            batch_size_v2s=0,
            batch_size_s2t=0,
            batch_size_s2s=0,
            batch_size_t2s=0,
            max_seq_length=128,
            attention_mask=None,
            p_mask_lm=None,
            p_mask_mmu=None,
            p_mask_vid=None,
            p_mask_v2s=None,
            p_mask_s2t=None,
            p_mask_s2s=None,
            p_mask_t2s=None,
            answer_lengths_lm=None,
            answer_lengths_mmu=None,
            answer_lengths_vid=None,
            answer_lengths_v2s=None,
            answer_lengths_s2t=None,
            answer_lengths_s2s=None,
            answer_lengths_t2s=None,
            t2i_masks=None,
            attention_masks_i2i=None,
            attention_masks_ti2ti=None,
            t2s_vocab_start=None,
            t2s_codebook_size=None,
            t2s_special_token_ids=None,
            text_vocab_size_override=None
            ):
        # 1. Attention Bias Setup (no changes)
        if attention_mask is not None:
            global_attn_mask = attention_mask.to(input_ids.device)
            if global_attn_mask.dtype != torch.bool:
                global_attn_mask = global_attn_mask.bool()
            attention_bias = (global_attn_mask[:, :, None] & global_attn_mask[:, None, :]).unsqueeze(1)
        else:
            global_attn_mask = None
            attention_bias = torch.ones(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1], device=input_ids.device)
        if batch_size_t2i > 0 and t2i_masks is not None:
            t2i_mask = t2i_masks
            if global_attn_mask is not None:
                t2i_mask = t2i_mask.bool() & global_attn_mask[:batch_size_t2i]
            attention_bias_t2i = (t2i_mask[:, :, None] & t2i_mask[:, None, :]).bool().unsqueeze(1)
            attention_bias[:batch_size_t2i] = attention_bias_t2i

        if batch_size_i2i > 0 and attention_masks_i2i is not None:
            start_i2i = batch_size_t2i
            end_i2i = start_i2i + batch_size_i2i
            attn_mask_i2i = attention_masks_i2i.to(input_ids.device)
            if attn_mask_i2i.dtype != torch.bool:
                attn_mask_i2i = attn_mask_i2i.bool()
            if global_attn_mask is not None:
                attn_mask_i2i = attn_mask_i2i & global_attn_mask[start_i2i:end_i2i]
            attention_bias_i2i = (attn_mask_i2i[:, :, None] & attn_mask_i2i[:, None, :]).unsqueeze(1)
            attention_bias[start_i2i:end_i2i] = attention_bias_i2i
        if batch_size_ti2ti > 0 and attention_masks_ti2ti is not None:
            start_ti2ti = batch_size_t2i + batch_size_i2i
            end_ti2ti = start_ti2ti + batch_size_ti2ti
            attn_mask_ti2ti = attention_masks_ti2ti.to(input_ids.device)
            if attn_mask_ti2ti.dtype != torch.bool:
                attn_mask_ti2ti = attn_mask_ti2ti.bool()
            if global_attn_mask is not None:
                attn_mask_ti2ti = attn_mask_ti2ti & global_attn_mask[start_ti2ti:end_ti2ti]
            attention_bias_ti2ti = (attn_mask_ti2ti[:, :, None] & attn_mask_ti2ti[:, None, :]).unsqueeze(1)
            attention_bias[start_ti2ti:end_ti2ti] = attention_bias_ti2ti

        # 2. Model Forward Pass (no changes)
        logits = self(input_ids, inputs_embeds=inputs_embeds, attention_bias=attention_bias).logits
        self.output_size = logits.shape[-1]
        
        # 3. Loss Calculation
        device = input_ids.device
        zero_loss = torch.tensor(0.0, device=device)
        
        # Calculate masked indices for the entire batch
        masked_indices = (input_ids == self.config.mask_token_id)
        
        text_vocab_size = text_vocab_size_override
        image_vocab_size = getattr(self.config, "codebook_size", 0)
        image_vocab_start = text_vocab_size
        image_vocab_end = min(image_vocab_start + image_vocab_size, logits.shape[-1])
        current_idx = 0

        # T2I Loss
        if batch_size_t2i > 0:
            logits_t2i = logits[current_idx:current_idx + batch_size_t2i, max_seq_length + 1:]
            labels_t2i = labels[current_idx:current_idx + batch_size_t2i, max_seq_length + 1:]
            if image_vocab_size <= 0:
                warnings.warn("t2i encountered non-positive image vocab size; skipping loss.")
                loss_t2i = zero_loss
            else:
                effective_vocab = image_vocab_end - image_vocab_start
                if effective_vocab <= 0:
                    warnings.warn("t2i effective image vocab is invalid; skipping loss.")
                    loss_t2i = zero_loss
                else:
                    logits_slice = logits_t2i[..., image_vocab_start:image_vocab_end]
                    labels_relative = torch.full_like(labels_t2i, -100)
                    valid_mask = (labels_t2i >= image_vocab_start) & (labels_t2i < image_vocab_end)
                    if not valid_mask.any():
                        warnings.warn("t2i labels contain no valid image tokens; skipping loss.")
                        loss_t2i = zero_loss
                    else:
                        labels_relative[valid_mask] = labels_t2i[valid_mask] - image_vocab_start
                        logits_slice_fp32 = logits_slice.to(torch.float32)
                        loss_t2i = F.cross_entropy(
                            logits_slice_fp32.contiguous().view(-1, effective_vocab),
                            labels_relative.contiguous().view(-1),
                            ignore_index=-100,
                        ).to(logits_slice.dtype)
                        loss_t2i_check = loss_t2i.to(torch.float32)
                        if (not torch.isfinite(loss_t2i_check)) or (loss_t2i_check < 0) or (loss_t2i_check > 10000):
                            label_vals = labels_t2i[valid_mask]
                            warn_msg = (
                                "t2i loss became non-finite. label_min={} label_max={} valid_count={}"
                                .format(
                                    label_vals.min().item() if label_vals.numel() > 0 else -1,
                                    label_vals.max().item() if label_vals.numel() > 0 else -1,
                                    int(valid_mask.sum().item()),
                                )
                            )
                            logger.warning("[t2i warn] %s", warn_msg)
                            warnings.warn(warn_msg)
                            loss_t2i = zero_loss
        else:
            loss_t2i = zero_loss
        current_idx += batch_size_t2i

        # I2I Loss
        if batch_size_i2i > 0:
            if image_vocab_size <= 0:
                warnings.warn("i2i encountered non-positive image vocab size; skipping loss.")
                loss_i2i = zero_loss
            else:
                start, end = current_idx, current_idx + batch_size_i2i
                logits_i2i = logits[start:end]
                labels_i2i = labels[start:end]
                effective_vocab = image_vocab_end - image_vocab_start
                if effective_vocab <= 0:
                    warnings.warn("i2i effective image vocab is invalid; skipping loss.")
                    loss_i2i = zero_loss
                else:
                    logits_slice = logits_i2i[..., image_vocab_start:image_vocab_end]
                    labels_relative = torch.full_like(labels_i2i, -100)
                    image_mask = (labels_i2i >= image_vocab_start) & (labels_i2i < image_vocab_end)
                    if not image_mask.any():
                        warnings.warn("i2i labels contain no valid image tokens; skipping loss.")
                        loss_i2i = zero_loss
                    else:
                        labels_relative[image_mask] = labels_i2i[image_mask] - image_vocab_start
                        loss_i2i = F.cross_entropy(
                            logits_slice.contiguous().view(-1, effective_vocab),
                            labels_relative.contiguous().view(-1),
                            ignore_index=-100,
                        )
                        if (not torch.isfinite(loss_i2i)) or (loss_i2i < 0):
                            label_vals = labels_i2i[image_mask]
                            warn_msg = (
                                "i2i loss became non-finite. label_min={} label_max={} valid_count={}"
                                .format(
                                    label_vals.min().item() if label_vals.numel() > 0 else -1,
                                    label_vals.max().item() if label_vals.numel() > 0 else -1,
                                    int(image_mask.sum().item()),
                                )
                            )
                            warnings.warn(warn_msg)
                            loss_i2i = zero_loss
        else:
            loss_i2i = zero_loss
        current_idx += batch_size_i2i

        # TI2TI Loss (handles both text + image tokens)
        if batch_size_ti2ti > 0:
            start, end = current_idx, current_idx + batch_size_ti2ti
            logits_ti2ti = logits[start:end]
            labels_ti2ti = labels[start:end]
            loss_text = zero_loss
            loss_img = zero_loss

            # text part: labels in [0, text_vocab_size)
            if text_vocab_size is not None and text_vocab_size > 0:
                vocab_text = int(min(text_vocab_size, logits_ti2ti.shape[-1]))
                logits_text = logits_ti2ti[..., :vocab_text]
                labels_text = labels_ti2ti.clone()
                invalid_text = (labels_text >= vocab_text) | (labels_text < 0)
                labels_text[invalid_text] = -100
                if (labels_text != -100).any():
                    # Masked positions only; mirror LM-style masking (p_mask style)
                    per_token_loss = F.cross_entropy(
                        logits_text.contiguous().view(-1, vocab_text),
                        labels_text.contiguous().view(-1),
                        ignore_index=-100,
                        reduction="none",
                    ).view_as(labels_text)
                    mask = labels_text != -100
                    if mask.any():
                        # p_mask-like scaling (here effectively 1.0 on masked positions)
                        denom = mask.float().clamp(min=1e-3)
                        per_token_loss = per_token_loss / denom
                        loss_text = per_token_loss[mask].mean()
                    else:
                        loss_text = zero_loss

            # image part: labels in [image_vocab_start, image_vocab_end)
            if image_vocab_size > 0:
                logits_img = logits_ti2ti[..., image_vocab_start:image_vocab_end]
                labels_relative = torch.full_like(labels_ti2ti, -100)
                image_mask = (labels_ti2ti >= image_vocab_start) & (labels_ti2ti < image_vocab_end)
                if image_mask.any():
                    labels_relative[image_mask] = labels_ti2ti[image_mask] - image_vocab_start
                    effective_vocab = image_vocab_end - image_vocab_start
                    loss_img = F.cross_entropy(
                        logits_img.contiguous().view(-1, effective_vocab),
                        labels_relative.contiguous().view(-1),
                        ignore_index=-100,
                    )

            loss_ti2ti = loss_text + loss_img
        else:
            loss_ti2ti = zero_loss
        current_idx += batch_size_ti2ti
        
        # LM Loss
        if batch_size_lm > 0:
            start, end = current_idx, current_idx + batch_size_lm
            logits_lm, labels_lm = logits[start:end], labels[start:end]
            masked_indices_lm = masked_indices[start:end]
            selected_logits_lm = logits_lm[masked_indices_lm]
            if selected_logits_lm.numel() == 0:
                loss_lm = zero_loss
            else:
                effective_vocab_lm = selected_logits_lm.shape[-1]
                if text_vocab_size and text_vocab_size < self.output_size:
                    effective_vocab_lm = min(text_vocab_size, selected_logits_lm.shape[-1])
                    selected_logits_lm = selected_logits_lm[:, :effective_vocab_lm]
                per_token_loss = F.cross_entropy(
                    selected_logits_lm.contiguous().view(-1, effective_vocab_lm),
                    labels_lm[masked_indices_lm].contiguous().view(-1),
                    ignore_index=-100,
                    reduction='none',
                )
                p_mask_vals = (
                    p_mask_lm.to(device)[masked_indices_lm]
                    .clamp(min=1e-4)
                    .to(per_token_loss.dtype)
                )
                per_token_loss = per_token_loss / p_mask_vals

                if answer_lengths_lm is not None:
                    lengths = (
                        answer_lengths_lm.to(device)[masked_indices_lm]
                        .clamp(min=1)
                        .to(per_token_loss.dtype)
                    )
                    loss_lm = (per_token_loss / lengths).sum() / logits_lm.shape[0]
                else:
                    loss_lm = per_token_loss.mean()
        else:
            loss_lm = zero_loss
        current_idx += batch_size_lm
        
        # MMU Loss
        if batch_size_mmu > 0:
            start, end = current_idx, current_idx + batch_size_mmu
            loss_mmu = calculate_mmu_style_loss(
                logits[start:end], labels[start:end], masked_indices[start:end],
                p_mask_mmu, answer_lengths_mmu, self.output_size, device,
            )
        else:
            loss_mmu = zero_loss
        current_idx += batch_size_mmu

        # VID (V2T) Loss
        if batch_size_v2t > 0:
            start, end = current_idx, current_idx + batch_size_v2t
            loss_vid = calculate_mmu_style_loss(
                logits[start:end], labels[start:end], masked_indices[start:end],
                p_mask_vid, answer_lengths_vid, self.output_size, device,
            )
        else:
            loss_vid = zero_loss
        current_idx += batch_size_v2t

        # V2S Loss
        if batch_size_v2s > 0:
            start, end = current_idx, current_idx + batch_size_v2s
            if (
                t2s_vocab_start is None
                or t2s_codebook_size is None
                or t2s_special_token_ids is None
            ):
                warnings.warn("v2s missing t2s vocab configuration; skipping loss.")
                loss_v2s = zero_loss
            elif answer_lengths_v2s is None or not (answer_lengths_v2s > 0).any():
                warnings.warn("v2s encountered empty answer lengths; skipping loss.")
                loss_v2s = zero_loss
            else:
                eoa_id = t2s_special_token_ids.get('eoa')
                eos_id = t2s_special_token_ids.get('eos')
                loss_v2s = calculate_t2s_loss(
                    logits[start:end],
                    labels[start:end],
                    masked_indices[start:end],
                    p_mask_v2s,
                    answer_lengths_v2s,
                    t2s_vocab_start,
                    t2s_codebook_size,
                    eoa_id,
                    eos_id,
                    device,
                    ignore_index=-100,
                )
        else:
            loss_v2s = zero_loss
        current_idx += batch_size_v2s

        # S2T Loss
        if batch_size_s2t > 0:
            start, end = current_idx, current_idx + batch_size_s2t
            loss_s2t = calculate_mmu_style_loss(
                logits[start:end], labels[start:end], masked_indices[start:end],
                p_mask_s2t, answer_lengths_s2t, self.output_size, device,
            )
        else:
            loss_s2t = zero_loss
        current_idx += batch_size_s2t

        # S2S Loss
        if batch_size_s2s > 0:
            start, end = current_idx, current_idx + batch_size_s2s
            if (
                t2s_vocab_start is None
                or t2s_codebook_size is None
                or t2s_special_token_ids is None
                or p_mask_s2s is None
                or answer_lengths_s2s is None
            ):
                warnings.warn("s2s missing t2s vocab configuration or masks; skipping loss.")
                loss_s2s = zero_loss
            elif not (answer_lengths_s2s > 0).any():
                warnings.warn("s2s encountered empty answer lengths; skipping loss.")
                loss_s2s = zero_loss
            else:
                eoa_id = t2s_special_token_ids.get('eoa')
                eos_id = t2s_special_token_ids.get('eos')
                loss_s2s = calculate_t2s_loss(
                    logits[start:end],
                    labels[start:end],
                    masked_indices[start:end],
                    p_mask_s2s,
                    answer_lengths_s2s,
                    t2s_vocab_start,
                    t2s_codebook_size,
                    eoa_id,
                    eos_id,
                    device,
                    ignore_index=-100,
                )
        else:
            loss_s2s = zero_loss
        current_idx += batch_size_s2s

        # T2S Loss
        if batch_size_t2s > 0:
            start, end = current_idx, current_idx + batch_size_t2s
            if (
                t2s_vocab_start is not None
                and t2s_codebook_size is not None
                and t2s_special_token_ids is not None
            ):
                eoa_id = t2s_special_token_ids.get('eoa')
                eos_id = t2s_special_token_ids.get('eos')
            else:
                eoa_id = eos_id = None

            if eoa_id is not None and eos_id is not None:
                loss_t2s = calculate_t2s_loss(
                    logits[start:end],
                    labels[start:end],
                    masked_indices[start:end],
                    p_mask_t2s,
                    answer_lengths_t2s,
                    t2s_vocab_start,
                    t2s_codebook_size,
                    eoa_id,
                    eos_id,
                    device,
                    ignore_index=-100,
                )
            else:
                loss_t2s = calculate_mmu_style_loss(
                    logits[start:end], labels[start:end], masked_indices[start:end],
                    p_mask_t2s, answer_lengths_t2s, self.output_size, device
                )
        else:
            loss_t2s = zero_loss
        current_idx += batch_size_t2s
        
        return logits, loss_t2i, loss_i2i, loss_ti2ti, loss_lm, loss_mmu, loss_vid, loss_v2s, loss_s2t, loss_s2s, loss_t2s  

    def forward_process_with_r2i(
            self,
            input_ids, 
            labels,
            t2i_masks=None,
            max_seq_length=128,
            batch_size_t2i=0,
            batch_size_lm=0,
            batch_size_mmu=0,
            batch_size_r2i=0,
            p_mask_lm=None,
            p_mask_mmu=None,
            p_mask_r2i=None,
            answer_lengths=None,
            answer_lengths_lm=None,
            answer_lengths_r2i=None,
            ):
        # attention bias, True for batch_size, 1, seq_len, seq_len  
        attention_bias = torch.ones(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1])
        attention_bias_t2i = (t2i_masks[:, :, None] & t2i_masks[:, None, :]).bool().unsqueeze(1)
        attention_bias[:batch_size_t2i] = attention_bias_t2i
        logits = self(input_ids, attention_bias=attention_bias).logits 
        # logits = self(input_ids).logits
        self.output_size = logits.shape[-1]

        if batch_size_t2i == 0:
            loss_t2i = torch.tensor(0.0, device=input_ids.device)
        else:
            # t2i loss
            loss_t2i = F.cross_entropy(
                logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.output_size),
                labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
                )
        
        # llada loss  
        start_lm = batch_size_t2i
        end_lm = start_lm + batch_size_lm
        start_mmu = end_lm
        end_mmu = start_mmu + batch_size_mmu
        start_r2i = end_mmu
        end_r2i = start_r2i + batch_size_r2i

        masked_indices = input_ids == self.config.mask_token_id 
        masked_indices_lm = masked_indices[start_lm:end_lm]
        masked_indices_mmu = masked_indices[start_mmu:end_mmu]
        masked_indices_r2i = masked_indices[start_r2i:end_r2i]

        p_mask_lm = p_mask_lm.to(masked_indices_lm.device)
        p_mask_mmu = p_mask_mmu.to(masked_indices_mmu.device)
        p_mask_r2i = p_mask_r2i.to(masked_indices_r2i.device)

        answer_lengths = answer_lengths.to(masked_indices_mmu.device) 
        answer_lengths_lm = answer_lengths_lm.to(masked_indices_lm.device)
        answer_lengths_r2i = answer_lengths_r2i.to(masked_indices_r2i.device)

        loss_lm = F.cross_entropy(
            logits[start_lm:end_lm][masked_indices_lm].contiguous().view(-1, self.output_size),
            labels[start_lm:end_lm][masked_indices_lm].contiguous().view(-1), ignore_index=-100, reduction='none'
            )/p_mask_lm[masked_indices_lm]

        if answer_lengths_lm is not None:
            loss_lm = torch.sum(loss_lm / answer_lengths_lm[masked_indices_lm]) / (logits[start_lm:end_lm].shape[0]) 
        else:
            loss_lm = loss_lm.sum() / (logits[start_lm:end_lm].shape[0] * logits[start_lm:end_lm].shape[1])

        loss_mmu = F.cross_entropy(
            logits[start_mmu:end_mmu][masked_indices_mmu].contiguous().view(-1, self.output_size),
            labels[start_mmu:end_mmu][masked_indices_mmu].contiguous().view(-1), ignore_index=-100, reduction='none'
            )/p_mask_mmu[masked_indices_mmu]
        loss_mmu = torch.sum(loss_mmu/answer_lengths[masked_indices_mmu]) / (logits[start_mmu:end_mmu].shape[0])
        
        loss_r2i = F.cross_entropy(
            logits[start_r2i:end_r2i][masked_indices_r2i].contiguous().view(-1, self.output_size),
            labels[start_r2i:end_r2i][masked_indices_r2i].contiguous().view(-1), ignore_index=-100, reduction='none'
            )/p_mask_r2i[masked_indices_r2i]
        loss_r2i = torch.sum(loss_r2i/answer_lengths_r2i[masked_indices_r2i]) / (logits[start_r2i:end_r2i].shape[0])
        
        return logits, loss_t2i, loss_lm, loss_mmu, loss_r2i

    def forward_t2i(
            self,
            input_ids, 
            labels,
            batch_size_t2i=0,
            max_seq_length=128,
            t2i_masks=None
            ):
        # attention bias, True for batch_size, 1, seq_len, seq_len  
        attention_bias = torch.ones(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1])
        attention_bias_t2i = (t2i_masks[:, :, None] & t2i_masks[:, None, :]).bool().unsqueeze(1)
        attention_bias[:batch_size_t2i] = attention_bias_t2i
        logits = self(input_ids, attention_bias=attention_bias).logits 
        # logits = self(input_ids).logits
        self.output_size = logits.shape[-1]

        loss_t2i = F.cross_entropy(
            logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.output_size),
            labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
            )
        
        return loss_t2i
    
    # Temp
    def forward_i2i(self, input_ids, attention_mask, labels):
        """
        Forward pass for the I2I task.
        """
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )

        return logits, loss

    # Temp
    def forward_s2t(
            self,
            input_ids, 
            labels,
            batch_size_s2t=0,
            max_seq_length=128,
            p_mask_s2t=None,
            answer_lengths=None,
            ):
        # attention bias, True for batch_size, 1, seq_len, seq_len  
        attention_bias = torch.ones(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1], device=input_ids.device)
        logits = self(input_ids, attention_bias=attention_bias).logits 
        self.output_size = logits.shape[-1]

        masked_indices = input_ids == self.config.mask_token_id 
        masked_indices_s2t = masked_indices[-batch_size_s2t:]
        p_mask_s2t = p_mask_s2t.to(masked_indices_s2t.device)       
        answer_lengths = answer_lengths.to(masked_indices_s2t.device) 

        loss_s2t = F.cross_entropy(
            logits[-batch_size_s2t:][masked_indices_s2t].contiguous().view(-1, self.output_size),
            labels[-batch_size_s2t:][masked_indices_s2t].contiguous().view(-1), ignore_index=-100, reduction='none'
            )/p_mask_s2t[masked_indices_s2t]
        loss_s2t = torch.sum(loss_s2t/answer_lengths[masked_indices_s2t]) / (logits[-batch_size_s2t:].shape[0])
        
        return logits, loss_s2t

    def forward_t2s(
        self,
        input_ids,
        labels,
        batch_size_t2s=0,
        max_seq_length=128,
        p_mask_t2s=None,
        answer_lengths=None,
    ):
        """
        Forward pass for text-to-speech (T2S) diffusion LM training.

        Args:
            input_ids: (B, L) Input token IDs (text + [MASK]*len(speech)).
            labels:    (B, L) Target speech codebook token IDs.
            batch_size_t2s:   Batch size for t2s task (for multitask batches).
            max_seq_length:   Prompt(text) 길이
            p_mask_t2s:       (B, L) Mask probability per position (optional).
            answer_lengths:   (B,)  각 row별 target length (optional).
        Returns:
            logits, loss_t2s
        """
        attention_bias = torch.ones(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1], device=input_ids.device)
        logits = self(input_ids, attention_bias=attention_bias).logits
        self.output_size = logits.shape[-1]

        masked_indices = input_ids == self.config.mask_token_id 
        masked_indices_t2s = masked_indices[-batch_size_t2s:]
        p_mask_t2s = p_mask_t2s.to(masked_indices_t2s.device)
        answer_lengths = answer_lengths.to(masked_indices_t2s.device)

        loss_t2s = F.cross_entropy(
            logits[-batch_size_t2s:][masked_indices_t2s].contiguous().view(-1, self.output_size),
            labels[-batch_size_t2s:][masked_indices_t2s].contiguous().view(-1),
            ignore_index=-100, reduction='none'
        ) / p_mask_t2s[masked_indices_t2s]
        loss_t2s = torch.sum(loss_t2s / answer_lengths[masked_indices_t2s]) / logits[-batch_size_t2s:].shape[0]

        return logits, loss_t2s
        
    def forward_v2t(
        self,
        input_ids,
        labels,
        batch_size_v2t=0,
        max_seq_length=128,
        p_mask_v2t=None,
        answer_lengths=None,
    ):
        """
        video-to-text (V2T) diffusion LM training.
        """
        attention_bias = torch.ones(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1], device=input_ids.device)
        logits = self(input_ids, attention_bias=attention_bias).logits
        self.output_size = logits.shape[-1]
        
        masked_indices = input_ids == self.config.mask_token_id
        masked_indices_v2t = masked_indices[:batch_size_v2t]
        p_mask_v2t = p_mask_v2t.to(masked_indices_v2t.device)
        answer_lengths = answer_lengths.to(masked_indices_v2t.device)
        
        loss_v2t = F.cross_entropy(
            logits[:batch_size_v2t][masked_indices_v2t].contiguous().view(-1, self.output_size),
            labels[:batch_size_v2t][masked_indices_v2t].contiguous().view(-1), 
            ignore_index=-100, 
            reduction='none'
        ) / p_mask_v2t[masked_indices_v2t]
        loss_v2t = torch.sum(loss_v2t / answer_lengths[masked_indices_v2t]) / (logits[:batch_size_v2t].shape[0])
        return logits, loss_v2t
    
    def forward_v2t_encoder(
        self,
        input_ids,
        labels,
        batch_size_v2t=0,
        max_seq_length=128,
        p_mask_v2t=None,
        answer_lengths=None,
    ):
        """
        video-to-text (V2T) diffusion LM training.
        """
        attention_bias = torch.ones(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1], device=input_ids.device)
        input_embeddings = super().model.transformer.wte(input_ids)
        
        
        logits = self(input_ids, attention_bias=attention_bias).logits
        self.output_size = logits.shape[-1]
        
        masked_indices = input_ids == self.config.mask_token_id
        masked_indices_v2t = masked_indices[:batch_size_v2t]
        p_mask_v2t = p_mask_v2t.to(masked_indices_v2t.device)
        answer_lengths = answer_lengths.to(masked_indices_v2t.device)
        
        loss_v2t = F.cross_entropy(
            logits[:batch_size_v2t][masked_indices_v2t].contiguous().view(-1, self.output_size),
            labels[:batch_size_v2t][masked_indices_v2t].contiguous().view(-1), 
            ignore_index=-100, 
            reduction='none'
        ) / p_mask_v2t[masked_indices_v2t]
        loss_v2t = torch.sum(loss_v2t / answer_lengths[masked_indices_v2t]) / (logits[:batch_size_v2t].shape[0])
        return logits, loss_v2t
        
    def forward_v2s(
        self,
        input_ids,
        labels,
        batch_size_v2s=0,
        max_seq_length: int = 128,
        p_mask_v2s=None,
        answer_lengths=None,
        t2s_vocab_start: Optional[int] = None,
        t2s_codebook_size: Optional[int] = None,
        t2s_special_token_ids: Optional[Dict[str, int]] = None,
    ):
        """
        # video-to-speech (V2S) diffusion LM training.
        """
        attention_bias = torch.ones(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1], device=input_ids.device)
        logits = self(input_ids, attention_bias=attention_bias).logits
        self.output_size = logits.shape[-1]
        
        masked_indices = input_ids == self.config.mask_token_id
        masked_indices_v2s = masked_indices[:batch_size_v2s]
        if batch_size_v2s == 0:
            return logits, torch.tensor(0.0, device=input_ids.device)

        p_mask_v2s = p_mask_v2s.to(masked_indices_v2s.device)
        answer_lengths = answer_lengths.to(masked_indices_v2s.device)

        if (
            t2s_vocab_start is not None
            and t2s_codebook_size is not None
            and t2s_special_token_ids is not None
        ):
            eoa_id = t2s_special_token_ids.get('eoa')
            eos_id = t2s_special_token_ids.get('eos')
        else:
            eoa_id = eos_id = None

        loss_v2s = calculate_t2s_loss(
            logits[:batch_size_v2s],
            labels[:batch_size_v2s],
            masked_indices_v2s,
            p_mask_v2s,
            answer_lengths,
            t2s_vocab_start,
            t2s_codebook_size,
            eoa_id,
            eos_id,
            input_ids.device,
            ignore_index=-100,
        )
        return logits, loss_v2s

    @torch.no_grad()
    def mmu_generate(self, idx=None, input_embeddings=None, max_new_tokens=128, steps=128,block_length=128, temperature=0.0, top_k=None, eot_token=None, cfg_scale=0.0, remasking='low_confidence', mask_id=126336, attention_mask=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        if attention_mask is not None and 0.0 in attention_mask:
            attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
        else:
            attention_bias = None
        try:
            device = idx.device
        except:
            device = input_embeddings.device

        result = []
        batch_size = idx.shape[0]
        x = torch.full((batch_size, idx.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(self.device)
        x[:, :idx.shape[1]] = idx.clone()
        prompt_index = (x != mask_id)
        
        
        assert max_new_tokens % block_length == 0
        num_blocks = max_new_tokens // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks
        
        # num_transfer_tokens = get_num_transfer_tokens(prompt_index, steps)
        for num_block in range(num_blocks):
            block_mask_index = (x[:, idx.shape[1] + num_block * block_length: idx.shape[1] + (num_block + 1) * block_length:] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            # num_transfer_tokens = get_num_transfer_tokens(prompt_index, steps)
            for i in range(steps):
                mask_index = (x == mask_id) 
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self(x, attention_bias=attention_bias).logits

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
                if remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, idx.shape[1] + (num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]

        return x

    @torch.no_grad()
    def s2t_generate(self, idx=None, input_embeddings=None, max_new_tokens=128, steps=128,block_length=128, temperature=0.0, top_k=None, eot_token=None, cfg_scale=0.0, remasking='low_confidence', mask_id=126336, attention_mask=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        if attention_mask is not None and 0.0 in attention_mask:
            attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
        else:
            attention_bias = None
        try:
            device = idx.device
        except:
            device = input_embeddings.device

        result = []
        batch_size = idx.shape[0]
        x = torch.full((batch_size, idx.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(self.device)
        x[:, :idx.shape[1]] = idx.clone()
        prompt_index = (x != mask_id)
        
        
        assert max_new_tokens % block_length == 0
        num_blocks = max_new_tokens // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks
        
        for num_block in range(num_blocks):
            block_mask_index = (x[:, idx.shape[1] + num_block * block_length: idx.shape[1] + (num_block + 1) * block_length:] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            
            for i in range(steps):
                mask_index = (x == mask_id) 
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self(x, attention_bias=attention_bias).logits
                
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
                if remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, idx.shape[1] + (num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]

        return x

    @torch.no_grad()
    def mmu_generate_fast(self, idx=None, input_embeddings=None, max_new_tokens=128, steps=128,block_length=128, temperature=0.0, top_k=None, eot_token=None, cfg_scale=0.0, remasking='low_confidence', mask_id=126336, attention_mask=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        if attention_mask is not None and 0.0 in attention_mask:
            attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
        else:
            attention_bias = None
        try:
            device = idx.device
        except:
            device = input_embeddings.device

        result = []
        batch_size = idx.shape[0]
        x = torch.full((batch_size, idx.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(self.device)
        x[:, :idx.shape[1]] = idx.clone()
        prompt_index = (x != mask_id)
        
        
        assert max_new_tokens % block_length == 0
        num_blocks = max_new_tokens // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks
        
        for num_block in range(num_blocks):
            block_mask_index = (x[:, idx.shape[1] + num_block * block_length: idx.shape[1] + (num_block + 1) * block_length:] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            for i in range(steps):
                mask_index = (x == mask_id) 
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self(x, attention_bias=attention_bias).logits
                
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
                if remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, idx.shape[1] + (num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]
            if eot_token is not None:
                last_token_index_in_current_block = idx.shape[1] + (num_block + 1) * block_length - 1
                if last_token_index_in_current_block < x.shape[1]:
                    tokens_at_block_end = x[:, last_token_index_in_current_block]
                    if torch.all(tokens_at_block_end == eot_token):
                        break
        return x

    @torch.no_grad()
    def mmu_generate_fastdllm_v1(
        self,
        idx=None,
        input_embeddings=None,
        max_new_tokens=128,
        steps=128,
        block_length=128,
        temperature=0.0,
        top_k=None,
        eot_token=None,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=126336,
        attention_mask=None,
        use_cache=True,
        threshold=None,
        factor=None,
    ):
        """
        Fast-dLLM v1 decoding path (isolated from default eval).
        """
        if _mmu_generate_fastdllm_v1 is None:
            raise ImportError(
                "fastdllm_v1 is not installed. Install it to use "
                "DyninOmniModelLM.mmu_generate_fastdllm_v1()."
            )
        return _mmu_generate_fastdllm_v1(
            model=self,
            idx=idx,
            max_new_tokens=max_new_tokens,
            steps=steps,
            block_length=block_length,
            temperature=temperature,
            top_k=top_k,
            eot_token=eot_token,
            cfg_scale=cfg_scale,
            remasking=remasking,
            mask_id=mask_id,
            attention_mask=attention_mask,
            use_cache=use_cache,
            threshold=threshold,
            factor=factor,
        )

    @torch.no_grad()
    def t2i_generate_decoding_stepwise(
            self,
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            uncond_attention_mask=None,
            temperature=1.0,
            timesteps=18,  # ideal number of steps is 18 in maskgit paper
            guidance_scale=0,
            noise_schedule=cosine_schedule,
            generator: torch.Generator = None,
            config=None,
            seq_len=1024,
            mask_token_id = 126336,
            resolution = 512,
            codebook_size = 8192,
            vq_model = None,
            **kwargs,
    ):
        """
        Generate 1:1 similar to the original MaskGit repo
        https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
        """

        # begin with all image token ids masked
        mask_count = (input_ids == mask_token_id).sum().item()
        num_vq_tokens = seq_len
        num_new_special_tokens = 0
        uni_prompting = kwargs.get("uni_prompting", None)
        input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()
        input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id, mask_token_id, input_ids_minus_lm_vocab_size - len(uni_prompting.text_tokenizer) - num_new_special_tokens)

        # for classifier-free guidance
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, :resolution + 1]

        for step in range(timesteps):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, resolution + 1:]], dim=1)
                model_input = torch.cat([input_ids, uncond_input_ids])
                attention_mask = torch.cat([attention_mask, uncond_attention_mask], dim=0)
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(model_input, attention_bias=attention_bias).logits 
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                # it seems that muse has a different cfg setting
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]
            else:
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(input_ids, attention_bias=attention_bias).logits
                logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]

            # logits: 1, 1024, 8192
            probs = logits.softmax(dim=-1)
            sampled = probs.reshape(-1, logits.size(-1))
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1]) # 1, 1024

            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
            # Defines the mask ratio for the next round. The number to mask out is
            current_image_vq_indices = sampled_ids.clone()
            current_image_vq_indices = torch.clamp(current_image_vq_indices, 0, 8192 - 1)
            current_image = vq_model.decode_code(current_image_vq_indices)
            images = torch.clamp((current_image + 1.0) / 2.0, min=0.0, max=1.0)
            images *= 255.0
            images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            pil_images = Image.fromarray(images[0]) 
            yield pil_images, f"Step {step + 1}/{timesteps}"
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))
            # Computes the probabilities of each selected tokens.
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
            selected_probs = selected_probs.squeeze(-1)

            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
            # Gets mask lens for each sample in the batch according to the mask ratio.
            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            mask_len = torch.max(
                torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )
            # Adds noise for randomness
            temperature = temperature * (1.0 - ratio)
            masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
            # Masks tokens with lower confidence.
            input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id,
                                                          sampled_ids + len(uni_prompting.text_tokenizer)
                                                          + num_new_special_tokens)
            input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)

        return sampled_ids
    

AutoConfig.register("dynin_omni", DyninOmniConfig)
AutoModelForCausalLM.register(DyninOmniConfig, DyninOmniModelLM)
AutoModel.register(DyninOmniConfig, DyninOmniModelLM)
