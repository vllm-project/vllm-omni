# coding=utf-8
# Copyright 2026 Dynin-Omni Team, AIDAS Lab, Seoul National University.
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

reserved_token_mapping = {
    '<|soi|>': 126084,  
    '<|eoi|>': 126085,
    '<|sov|>': 126086,
    '<|eov|>': 126087,
    '<|t2i|>': 126088,
    '<|mmu|>': 126089,
    '<|t2v|>': 126090,
    '<|v2v|>': 126091,
    '<|lvg|>': 126092,
    '[iPAD]': 126093,
    '<|r2i|>': 126094,
    '<|i2i|>': 126095,
    '<|s2t|>': 126096,
    '<|soa|>': 126097,
    '<|eoa|>': 126098,
    '<|t2s|>': 126099,
    '<|v2t|>': 126100,
    '<|s2s|>': 126101,
    '<|v2s|>': 126102,
    '<|ti2ti|>': 126103,
    '<|t2ti|>': 126104,
    '<think>': 126105,
    '</think>': 126106
}


import torch
from typing import Union, Optional

class UniversalPrompting():
    def __init__(self, text_tokenizer,
                 special_tokens=(
                    "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
                    "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>", 
                    "<|i2i|>", "<|ti2ti|>", "<|v2t|>", "<|v2s|>", "<|s2t|>", "<|t2s|>", "<|s2s|>", "<|soa|>", "<|eoa|>", 
                    "<think>", "</think>",
                 ),
                 max_text_len=8000, max_audio_len = 384, max_audio_len_short = 256, max_seq_len=377, max_image_len=1024, ignore_id=-100, cond_dropout_prob=0.1, use_reserved_token=False, mask_token_id=126336):
        """
        :param text_tokenizer: original text tokenizer
        """
        if not use_reserved_token:
            self.text_tokenizer = text_tokenizer
            self.text_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.text_tokenizer.add_tokens(list(special_tokens))
            self.sptids_dict = {token: torch.tensor(
                self.text_tokenizer.convert_tokens_to_ids([token])) for token in special_tokens} 
            self.sptids_dict['<|sot|>'] = torch.tensor([self.text_tokenizer.bos_token_id])
            self.sptids_dict['<|eot|>'] = torch.tensor([self.text_tokenizer.eos_token_id])
            self.sptids_dict['<|pad|>'] = torch.tensor([self.text_tokenizer.pad_token_id])
        else:
            self.text_tokenizer = text_tokenizer
            self.sptids_dict = {}
            
            for token, token_id in reserved_token_mapping.items():
                self.sptids_dict[token] = torch.tensor([token_id])
            
            self.sptids_dict['<|sot|>'] = torch.tensor([self.text_tokenizer.bos_token_id])
            self.sptids_dict['<|eot|>'] = torch.tensor([self.text_tokenizer.eos_token_id])    
            end_header_tokens = self.text_tokenizer.convert_tokens_to_ids(['<|end_header_id|>'])
            
            if end_header_tokens and len(end_header_tokens) > 0 and end_header_tokens[0]:
                self.sptids_dict['<|end_header_id|>'] = torch.tensor(end_header_tokens)
                self.sptids_dict['<|eot_id|>'] = torch.tensor(self.text_tokenizer.convert_tokens_to_ids(['<|eot_id|>']))
                self.sptids_dict['<|start_header_id|>'] = torch.tensor(self.text_tokenizer.convert_tokens_to_ids(['<|start_header_id|>']))
            else:
                special_tokens_dict = {
                    'additional_special_tokens': [
                        '<|start_header_id|>',
                        '<|end_header_id|>',
                        '<|eot_id|>'
                    ]
                }
                
                num_added = self.text_tokenizer.add_special_tokens(special_tokens_dict)
                new_token_id = self.text_tokenizer.convert_tokens_to_ids(['<|end_header_id|>'])
                self.sptids_dict['<|end_header_id|>'] = torch.tensor(new_token_id)
                self.sptids_dict['<|eot_id|>'] = torch.tensor(self.text_tokenizer.convert_tokens_to_ids(['<|eot_id|>']))
                self.sptids_dict['<|start_header_id|>'] = torch.tensor(self.text_tokenizer.convert_tokens_to_ids(['<|start_header_id|>']))
        
        self.max_text_len = max_text_len + 1
        self.max_image_len = max_image_len
        self.max_audio_len = max_audio_len
        self.max_audio_len_short = max_audio_len_short
        self.pad_id = reserved_token_mapping['[iPAD]']
        self.mask_token_id = mask_token_id
        self.ignore_id = ignore_id
        self.cond_dropout_prob = cond_dropout_prob

    def t2i_prompt(self, text_ids, image_ids, labels, max_text_len=None):

        device = image_ids.device
        max_text_len = max_text_len if max_text_len is not None else self.max_text_len
        sequence_ids = []
        attention_masks = []
        label_ids = []
        probs = torch.rand(len(text_ids))
        eos_id = self.text_tokenizer.eos_token_id

        for i in range(len(text_ids)):

            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]

            temp_ids = [int(self.sptids_dict['<|t2i|>'])] + text_ids[i] + [self.text_tokenizer.eos_token_id]

            if probs[i] < self.cond_dropout_prob:
                temp_ids = [int(self.sptids_dict['<|t2i|>']), self.text_tokenizer.bos_token_id, self.text_tokenizer.eos_token_id]

            if max_text_len >= len(temp_ids):
                old_len = len(temp_ids)
                pad_len = max_text_len - old_len
                temp_ids = [self.pad_id] * pad_len + temp_ids
                temp_masks = [0] * pad_len + [1] * (old_len + image_ids.shape[-1] + 2)
            else:
                temp_ids = temp_ids[:max_text_len - 1] + [self.text_tokenizer.eos_token_id]
                temp_masks = [1] * (len(temp_ids) + image_ids.shape[-1] + 2)  # +2 for two special tokens
                pad_len = 0
            
            # prompting [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_label_ids = torch.cat([
                # should we predict text tokens when doing image reconstruction?
                torch.tensor(temp_ids).to(device),
                self.sptids_dict['<|soi|>'].to(device),
                labels[i],
                self.sptids_dict['<|eoi|>'].to(device)
            ], dim=0)
            if pad_len > 0:
                temp_label_ids[:pad_len] = self.ignore_id


            temp_ids = torch.cat([
                torch.tensor(temp_ids).to(device),
                self.sptids_dict['<|soi|>'].to(device),
                image_ids[i],
                self.sptids_dict['<|eoi|>'].to(device)
            ], dim=0)

            # sequence_ids: [pad]...[pad] <|t2i|> <bos> text_1 ... text_n <eos> <|soi|> image_1 ... image_m <|eoi|> 
            temp_masks = torch.tensor(temp_masks).to(device)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
            label_ids.append(temp_label_ids.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0), torch.cat(label_ids, dim=0)

    def i2i_prompt(self, prompts, original_image_tokens, masked_edited_tokens, labels, max_text_len=None):
        """
        Constructs the input sequence for the Image-to-Image task.
        Final sequence structure:
        <|i2i|> <|soi|> [source_image] <|eoi|> <bos> text_1 ... text_n <eos> <|soi|> [masked_target_image] <|eoi|>
        """
        device = original_image_tokens.device
        batch_size = len(prompts)
        max_text_len = max_text_len if max_text_len is not None else self.max_text_len
        
        sequence_ids = []
        attention_masks = []
        label_ids = []
        
        tokenized_prompts = self.text_tokenizer(prompts, add_special_tokens=False, return_tensors=None).input_ids

        for i in range(batch_size):
            
            # 1. Process text prompts with <bos> and <eos>
            temp_text_ids = [self.text_tokenizer.bos_token_id] + tokenized_prompts[i] + [self.text_tokenizer.eos_token_id]
            
            if torch.rand(1) < self.cond_dropout_prob:
                temp_text_ids = [self.text_tokenizer.bos_token_id, self.text_tokenizer.eos_token_id]
            
            if max_text_len >= len(temp_text_ids):
                pad_len = max_text_len - len(temp_text_ids)
                padded_text_ids = [self.pad_id] * pad_len + temp_text_ids
            else:
                padded_text_ids = temp_text_ids[:max_text_len-1] + [self.text_tokenizer.eos_token_id]
            
            padded_text_ids_tensor = torch.tensor(padded_text_ids, device=device)
            
            # 2. Construct the full input sequence (input_ids)
            temp_ids = torch.cat([
                self.sptids_dict['<|i2i|>'].to(device),
                self.sptids_dict['<|soi|>'].to(device),
                original_image_tokens[i],
                self.sptids_dict['<|eoi|>'].to(device),
                padded_text_ids_tensor,
                self.sptids_dict['<|soi|>'].to(device), # Using <|soi|> for target image
                masked_edited_tokens[i],
                self.sptids_dict['<|eoi|>'].to(device)  # Using <|eoi|> for target image
            ], dim=0)
            
            sequence_ids.append(temp_ids.unsqueeze(0))

            # 3. Construct the labels
            len_prefix_ignore = (
                1 + # <|i2i|>
                1 + len(original_image_tokens[i]) + 1 + # <|soi|>...<|eoi|>
                len(padded_text_ids_tensor) + # text
                1 # <|soi|> for target
            )
            
            ignore_labels = torch.full((len_prefix_ignore,), self.ignore_id, device=device)
            
            temp_label_ids = torch.cat([
                ignore_labels,
                labels[i],
                torch.tensor([self.ignore_id], device=device) # Ignore final <|eoi|>
            ], dim=0)
            
            label_ids.append(temp_label_ids.unsqueeze(0))

            # 4. Construct the attention mask
            text_attention_mask = (padded_text_ids_tensor != self.pad_id).long()
            
            # All non-padding tokens should have attention
            len_prefix = 1 + 1 + len(original_image_tokens[i]) + 1 
            len_suffix = 1 + len(masked_edited_tokens[i]) + 1
            
            prefix_mask = torch.ones(len_prefix, device=device)
            suffix_mask = torch.ones(len_suffix, device=device)

            temp_masks = torch.cat([
                prefix_mask,
                text_attention_mask,
                suffix_mask
            ], dim=0)

            attention_masks.append(temp_masks.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0), torch.cat(label_ids, dim=0)

    def ti2ti_prompt(self, prompts, source_tokens, masked_target_tokens, labels_img, target_texts, target_mask_bools=None, task_token: str = "<|ti2ti|>"):
        """
        Builds a text+image -> text+image sequence:
        [pads] <|ti2ti|> <|soi|> [src_img] <|eoi|> [src text (padded)] <|soi|> [masked tgt img] <|eoi|> [tgt text (padded)]
        Prompt (src text) is conditioning-only; target text + target image are supervised.
        """
        device = source_tokens.device
        batch_size = len(prompts)
        task_id = int(self.sptids_dict[task_token])
        soi_id = int(self.sptids_dict['<|soi|>'])
        eoi_id = int(self.sptids_dict['<|eoi|>'])
        pad_id = int(self.pad_id)
        ignore_id = int(self.ignore_id)
        max_text_len = self.max_text_len

        tokenized_prompts = self.text_tokenizer(
            prompts,
            truncation=True,
            max_length=max_text_len,
        )['input_ids']
        tokenized_targets = self.text_tokenizer(
            target_texts,
            truncation=True,
            max_length=max_text_len,
        )['input_ids']

        sequence_ids = []
        attention_masks = []
        label_ids = []

        for i in range(batch_size):
            prompt_ids = tokenized_prompts[i]
            target_ids = tokenized_targets[i]
            if len(target_ids) == 0 or target_ids[0] != self.text_tokenizer.bos_token_id:
                target_ids = [self.text_tokenizer.bos_token_id] + target_ids
            target_ids = target_ids + [self.text_tokenizer.eos_token_id]

            def _pad(ids):
                if len(ids) < max_text_len:
                    pad_len = max_text_len - len(ids)
                    mask = [1] * len(ids) + [0] * pad_len
                    ids = ids + [pad_id] * pad_len
                else:
                    ids = ids[: max_text_len - 1] + [self.text_tokenizer.eos_token_id]
                    mask = [1] * len(ids)
                return ids, mask

            # src text (conditioning; ignore labels), no padding here
            if len(prompt_ids) == 0 or prompt_ids[0] != self.text_tokenizer.bos_token_id:
                prompt_ids = [self.text_tokenizer.bos_token_id] + prompt_ids
            prompt_ids = prompt_ids + [self.text_tokenizer.eos_token_id]
            if torch.rand(1) < self.cond_dropout_prob:
                prompt_ids = [self.text_tokenizer.bos_token_id, self.text_tokenizer.eos_token_id]

            target_ids, target_mask = _pad(target_ids)

            prompt_tensor = torch.tensor(prompt_ids, device=device, dtype=torch.long)
            prompt_mask_tensor = torch.ones(len(prompt_tensor), device=device, dtype=torch.long)
            target_tensor = torch.tensor(target_ids, device=device, dtype=torch.long)
            target_ids_tensor = torch.tensor(target_ids, device=device, dtype=torch.long)
            loss_mask = torch.ones_like(target_tensor, dtype=torch.bool)
            if target_mask_bools is not None and i < len(target_mask_bools):
                mask_bool_raw = target_mask_bools[i]
                if mask_bool_raw.shape[0] < target_tensor.shape[0]:
                    pad_extra = target_tensor.shape[0] - mask_bool_raw.shape[0]
                    mask_bool_raw = torch.cat(
                        [mask_bool_raw, torch.zeros(pad_extra, device=device, dtype=torch.bool)]
                    )
                mask_bool_raw = mask_bool_raw.to(device=device, dtype=torch.bool)
                mask_bool = mask_bool_raw[: target_tensor.shape[0]] & (target_tensor != pad_id)
                mask_value = torch.full_like(target_tensor, self.mask_token_id)
                target_tensor = torch.where(mask_bool, mask_value, target_tensor)
                loss_mask = mask_bool
            else:
                loss_mask = target_tensor != pad_id
            target_mask_tensor = torch.tensor(target_mask, device=device, dtype=torch.long)

            labels_prompt = torch.full((len(prompt_tensor),), ignore_id, device=device, dtype=torch.long)
            labels_text = torch.full_like(target_tensor, ignore_id, dtype=torch.long)
            labels_text = torch.where(
                (loss_mask & (target_ids_tensor != pad_id)),
                target_ids_tensor,
                labels_text,
            )

            seq = torch.cat([
                torch.tensor([task_id], device=device),
                torch.tensor([soi_id], device=device),
                source_tokens[i],
                torch.tensor([eoi_id], device=device),
                prompt_tensor,
                torch.tensor([soi_id], device=device),
                masked_target_tokens[i],
                torch.tensor([eoi_id], device=device),
                target_tensor,
            ], dim=0)
            sequence_ids.append(seq)

            prefix_len = 1 + 1 + len(source_tokens[i]) + 1
            label_prefix = torch.full((prefix_len,), ignore_id, device=device, dtype=torch.long)
            labels_combined = torch.cat([
                label_prefix, # task + src image (ignore)
                labels_prompt, # prompt (ignore)
                torch.tensor([ignore_id], device=device, dtype=torch.long), # soi before tgt img
                labels_img[i], # masked target image supervision
                torch.tensor([ignore_id], device=device, dtype=torch.long), # eoi after tgt img
                labels_text, # target text supervision
            ], dim=0)
            label_ids.append(labels_combined)

            attn_prefix = torch.ones(prefix_len, device=device, dtype=torch.long)
            attn_tgt_img = torch.ones(len(masked_target_tokens[i]) + 1, device=device, dtype=torch.long)  # tgt img + eoi
            attn = torch.cat([
                attn_prefix,
                prompt_mask_tensor,
                torch.ones(1, device=device, dtype=torch.long), # soi before tgt img
                attn_tgt_img,
                target_mask_tensor,
            ], dim=0)
            attention_masks.append(attn)

        # left-pad sequences in batch to uniform length
        max_len = max(seq.size(0) for seq in sequence_ids)
        padded_seqs = []
        padded_masks = []
        padded_labels = []
        for seq, att, lab in zip(sequence_ids, attention_masks, label_ids):
            pad_len = max_len - seq.size(0)
            if pad_len > 0:
                pad_seq = torch.full((pad_len,), self.pad_id, device=device, dtype=torch.long)
                pad_mask = torch.zeros(pad_len, device=device, dtype=torch.long)
                pad_lab = torch.full((pad_len,), ignore_id, device=device, dtype=torch.long)
                seq = torch.cat([pad_seq, seq], dim=0)
                att = torch.cat([pad_mask, att], dim=0)
                lab = torch.cat([pad_lab, lab], dim=0)
            padded_seqs.append(seq.unsqueeze(0))
            padded_masks.append(att.unsqueeze(0))
            padded_labels.append(lab.unsqueeze(0))

        return torch.cat(padded_seqs, dim=0), torch.cat(padded_masks, dim=0), torch.cat(padded_labels, dim=0)

    def t2i_gen_prompt(self, text_ids, image_ids):

        device = image_ids.device
        sequence_ids = []
        attention_masks = []
        for i in range(len(text_ids)):
            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]
            # note that, llama3 tokenizer automatically add the bot token at first but without eot
            temp_ids = [int(self.sptids_dict['<|t2i|>'])] + text_ids[i] + [self.text_tokenizer.eos_token_id]
            if self.max_text_len >= len(temp_ids):
                old_len = len(temp_ids)
                temp_ids = [self.pad_id] * (self.max_text_len - len(temp_ids)) + temp_ids
                temp_masks = [0] * (self.max_text_len - old_len) + [1] * (old_len + image_ids.shape[-1] + 2)
            else:
                # should add the eos token
                temp_ids = temp_ids[:self.max_text_len - 1] + [self.text_tokenizer.eos_token_id]
                temp_masks = [1] * (len(temp_ids) + image_ids.shape[-1] + 2)  # +2 for two special tokens

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_ids = torch.cat([
                torch.tensor(temp_ids).to(device),
                self.sptids_dict['<|soi|>'].to(device),
                image_ids[i],
                self.sptids_dict['<|eoi|>'].to(device)
            ], dim=0)

            temp_masks = torch.tensor(temp_masks).to(device)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0)

    def ti2ti_gen_prompt(self, prompts, target_texts, source_tokens, placeholder_tokens):
        """
        Generation prompt for TI2TI:
        [ti2ti][soi]src[eoi][cond text][soi]placeholder[eoi][tgt text (masked by caller)]
        """
        device = source_tokens.device
        ti2ti_id = int(self.sptids_dict['<|ti2ti|>'])
        soi_id = int(self.sptids_dict['<|soi|>'])
        eoi_id = int(self.sptids_dict['<|eoi|>'])
        pad_id = int(self.pad_id)
        max_text_len = self.max_text_len

        tokenized_prompts = self.text_tokenizer(prompts, truncation=True, max_length=max_text_len,)['input_ids']
        tokenized_targets = self.text_tokenizer(target_texts, truncation=True, max_length=max_text_len,)['input_ids']

        seq_list = []
        attn_list = []
        for i in range(len(prompts)):
            prompt_ids = tokenized_prompts[i]
            tgt_ids = tokenized_targets[i]
            if len(prompt_ids) == 0 or prompt_ids[0] != self.text_tokenizer.bos_token_id:
                prompt_ids = [self.text_tokenizer.bos_token_id] + prompt_ids
            prompt_ids = prompt_ids + [self.text_tokenizer.eos_token_id]
            if len(tgt_ids) == 0 or tgt_ids[0] != self.text_tokenizer.bos_token_id:
                tgt_ids = [self.text_tokenizer.bos_token_id] + tgt_ids
            tgt_ids = tgt_ids + [self.text_tokenizer.eos_token_id]

            def _pad(ids):
                if len(ids) < max_text_len:
                    pad_len = max_text_len - len(ids)
                    mask = [0] * pad_len + [1] * len(ids)
                    ids = [pad_id] * pad_len + ids
                else:
                    ids = ids[: max_text_len - 1] + [self.text_tokenizer.eos_token_id]
                    mask = [1] * len(ids)
                return ids, mask

            prompt_ids, prompt_mask = _pad(prompt_ids)
            tgt_ids, tgt_mask = _pad(tgt_ids)

            prompt_tensor = torch.tensor(prompt_ids, device=device, dtype=torch.long)
            prompt_mask_tensor = torch.tensor(prompt_mask, device=device, dtype=torch.long)
            tgt_tensor = torch.tensor(tgt_ids, device=device, dtype=torch.long)
            tgt_mask_tensor = torch.tensor(tgt_mask, device=device, dtype=torch.long)

            seq = torch.cat([
                torch.tensor([ti2ti_id, soi_id], device=device),
                source_tokens[i],
                torch.tensor([eoi_id], device=device),
                prompt_tensor,
                torch.tensor([soi_id], device=device),
                placeholder_tokens[i],
                torch.tensor([eoi_id], device=device),
                tgt_tensor,
            ], dim=0)
            attn = torch.cat([
                torch.ones(1 + 1 + len(source_tokens[i]) + 1, device=device, dtype=torch.long),
                prompt_mask_tensor,
                torch.ones(1 + len(placeholder_tokens[i]) + 1, device=device, dtype=torch.long),
                tgt_mask_tensor,
            ], dim=0)
            seq_list.append(seq.unsqueeze(0))
            attn_list.append(attn.unsqueeze(0))

        # Left-pad to uniform length
        max_len = max(s.size(1) for s in seq_list)
        padded_seq = []
        padded_attn = []
        for seq, attn in zip(seq_list, attn_list):
            pad_len = max_len - seq.size(1)
            if pad_len > 0:
                pad_seq = torch.full((1, pad_len), pad_id, device=device, dtype=torch.long)
                pad_attn = torch.zeros((1, pad_len), device=device, dtype=torch.long)
                seq = torch.cat([pad_seq, seq], dim=1)
                attn = torch.cat([pad_attn, attn], dim=1)
            padded_seq.append(seq)
            padded_attn.append(attn)

        return torch.cat(padded_seq, dim=0), torch.cat(padded_attn, dim=0)
    
    def i2i_gen_prompt(self, texts, input_image_tokens, output_image_placeholder):
        device = input_image_tokens.device
        
        if isinstance(texts, str):
            texts = [texts]
        
        batch_size = len(texts)
        
        sequence_ids_batch = []
        attention_masks_batch = []

        for i in range(batch_size):
            text_item = texts[i]
            input_img_item = input_image_tokens[i]
            output_img_placeholder_item = output_image_placeholder[i]

            text_ids_list = self.text_tokenizer(text_item)['input_ids']

            if not text_ids_list: 
                text_ids_list = [self.text_tokenizer.bos_token_id]
            elif text_ids_list[0] != self.text_tokenizer.bos_token_id:
                text_ids_list = [self.text_tokenizer.bos_token_id] + text_ids_list
            text_ids_list.append(self.text_tokenizer.eos_token_id)

            max_text_len = self.max_text_len
            if max_text_len >= len(text_ids_list):
                pad_len = max_text_len - len(text_ids_list)
                padded_text_ids = [self.pad_id] * pad_len + text_ids_list
                text_attention_mask_list = [0] * pad_len + [1] * len(text_ids_list)
            else:
                padded_text_ids = text_ids_list[:max_text_len - 1] + [self.text_tokenizer.eos_token_id]
                text_attention_mask_list = [1] * max_text_len

            # [TASK][CONDITION_IMG][CONDITION_TEXT][START_GEN][TARGET_IMG][END_GEN]
            temp_ids = torch.cat([
                self.sptids_dict['<|t2i|>'].to(device),                   
                self.sptids_dict['<|soi|>'].to(device),                  
                input_img_item,
                self.sptids_dict['<|eoi|>'].to(device),
                self.sptids_dict['<|sot|>'].to(device),                   
                torch.tensor(padded_text_ids, dtype=torch.long, device=device),
                self.sptids_dict['<|eot|>'].to(device),
                self.sptids_dict['<|soi|>'].to(device),                   
                output_img_placeholder_item,
                self.sptids_dict['<|eoi|>'].to(device)                   
            ], dim=0)

            temp_masks = torch.cat([
                torch.ones(1, dtype=torch.long, device=device),                            
                torch.ones(1, dtype=torch.long, device=device),                            
                torch.ones_like(input_img_item, dtype=torch.long),                         
                torch.ones(1, dtype=torch.long, device=device),                            
                torch.ones(1, dtype=torch.long, device=device),                            
                torch.tensor(text_attention_mask_list, dtype=torch.long, device=device),   
                torch.ones(1, dtype=torch.long, device=device),                            
                torch.ones(1, dtype=torch.long, device=device),                            
                torch.ones_like(output_img_placeholder_item, dtype=torch.long),            
                torch.ones(1, dtype=torch.long, device=device)                             
            ], dim=0)

            sequence_ids_batch.append(temp_ids.unsqueeze(0))
            attention_masks_batch.append(temp_masks.unsqueeze(0))

        return torch.cat(sequence_ids_batch, dim=0), torch.cat(attention_masks_batch, dim=0)

    def t2s_gen_prompt(self, text_ids, audio_ids):

        device = audio_ids.device
        sequence_ids = []
        attention_masks = []
        for i in range(len(text_ids)):
            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]
            # note that, llama3 tokenizer automatically add the bot token at first but without eot
            temp_ids = [int(self.sptids_dict['<|t2s|>'])] + text_ids[i] + [self.text_tokenizer.eos_token_id]
            if self.max_text_len >= len(temp_ids):
                old_len = len(temp_ids)
                temp_ids = [self.pad_id] * (self.max_text_len - len(temp_ids)) + temp_ids
                temp_masks = [0] * (self.max_text_len - old_len) + [1] * (old_len + audio_ids.shape[-1] + 1)
            else:
                # should add the eos token
                temp_ids = temp_ids[:self.max_text_len - 1] + [self.text_tokenizer.eos_token_id]
                temp_masks = [1] * (len(temp_ids) + audio_ids.shape[-1] + 1)  # +1 for SOA

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [audio tokens] [eoi]
            temp_ids = torch.cat([
                torch.tensor(temp_ids).to(device),
                self.sptids_dict['<|soa|>'].to(device),
                audio_ids[i],
                # self.sptids_dict['<|eoa|>'].to(device)
            ], dim=0)

            temp_masks = torch.tensor(temp_masks).to(device)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0)

    def t2s_fixed_gen_prompt(self, text_ids, audio_ids):

        device = audio_ids.device
        sequence_ids = []
        attention_masks = []
        for i in range(len(text_ids)):
            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]
            # note that, llama3 tokenizer automatically add the bot token at first but without eot
            temp_ids = [int(self.sptids_dict['<|t2s|>'])] + text_ids[i] + [self.text_tokenizer.eos_token_id]
            if self.max_text_len >= len(temp_ids):
                old_len = len(temp_ids)
                temp_ids = [self.pad_id] * (self.max_text_len - len(temp_ids)) + temp_ids
                temp_masks = [0] * (self.max_text_len - old_len) + [1] * (old_len + audio_ids.shape[-1] + 2)
            else:
                # should add the eos token
                temp_ids = temp_ids[:self.max_text_len - 1] + [self.text_tokenizer.eos_token_id]
                temp_masks = [1] * (len(temp_ids) + audio_ids.shape[-1] + 2)  # +1 for SOA and EOA

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [audio tokens] [eoi]
            temp_ids = torch.cat([
                torch.tensor(temp_ids).to(device),
                self.sptids_dict['<|soa|>'].to(device),
                audio_ids[i],
                self.sptids_dict['<|eoa|>'].to(device)
            ], dim=0)

            temp_masks = torch.tensor(temp_masks).to(device)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0)

    # language modeling
    def lm_prompt(self, text_ids, max_seq_len):
        sequence_ids = []
        attention_masks = []
        label_ids = []
        for i in range(len(text_ids)):
            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]

            temp_ids = text_ids[i] + [self.text_tokenizer.eos_token_id]

            if max_seq_len >= len(temp_ids):
                pad_len = max_seq_len - len(temp_ids)
                eos_id = self.text_tokenizer.eos_token_id
                temp_ids = temp_ids + [eos_id] * pad_len
                temp_labels_ids = temp_ids
                temp_masks = [1] * len(temp_ids)  # already max_seq_len long
            else:
                pad_len = 0
                # In language modeling, we only process text tokens. We do not add the eos token if the text length
                # exceeds the max sequence length
                temp_labels_ids = temp_ids[:max_seq_len]
                temp_ids = temp_ids[:max_seq_len]
                temp_masks = [1] * len(temp_ids)  # +2 for two special tokens

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_ids = torch.tensor(temp_ids)
            temp_masks = torch.tensor(temp_masks)
            temp_labels_ids = torch.tensor(temp_labels_ids)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
            label_ids.append(temp_labels_ids.unsqueeze(0))

        # input_ids, masks, labels
        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0), torch.cat(label_ids, dim=0)

    # language modeling
    def lm_chat_prompt(self, text_ids, max_seq_len):
        sequence_ids = []
        prompt_masks = []
        label_ids = []

        for i in range(len(text_ids)):
            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]

            temp_ids = text_ids[i] + [self.text_tokenizer.eos_token_id]

            if max_seq_len >= len(temp_ids):
                pad_len = max_seq_len - len(temp_ids)
                eos_id = self.text_tokenizer.eos_token_id
                temp_ids = temp_ids + [eos_id] * pad_len
                temp_labels_ids = temp_ids
            else:
                # In language modeling, we only process text tokens. We do not add the eos token if the text length
                # exceeds the max sequence length
                temp_labels_ids = temp_ids[:max_seq_len]
                temp_ids = temp_ids[:max_seq_len]

            end_header_id = int(self.sptids_dict['<|end_header_id|>'])
            end_header_pos = -1
            for pos in range(len(temp_ids) - 1, -1, -1):    # 尝试从文本序列中寻找<|end_header_id|>
                if temp_ids[pos] == end_header_id:
                    end_header_pos = pos
                    break
            if end_header_pos != -1:
                prompt_length = end_header_pos + 1
            else:
                prompt_length = 0
            temp_masks = [1] * prompt_length + [0] * (len(temp_ids) - prompt_length)

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_ids = torch.tensor(temp_ids)
            temp_masks = torch.tensor(temp_masks)
            temp_labels_ids = torch.tensor(temp_labels_ids)
            sequence_ids.append(temp_ids.unsqueeze(0))
            prompt_masks.append(temp_masks.unsqueeze(0))
            label_ids.append(temp_labels_ids.unsqueeze(0))

        # input_ids, masks, labels
        return torch.cat(sequence_ids, dim=0), torch.cat(prompt_masks, dim=0), torch.cat(label_ids, dim=0)

    def s2s_gen_prompt(
        self,
        audio_usr_ids: list[torch.Tensor],
        audio_asst_placeholders: list[torch.Tensor],
        image_ids: Optional[list[Optional[torch.Tensor]]] = None,
    ):
        if len(audio_usr_ids) != len(audio_asst_placeholders):
            raise ValueError("audio_usr_ids and audio_asst_placeholders must have the same length")
        if image_ids is None:
            image_ids = [None] * len(audio_usr_ids)
        elif len(image_ids) != len(audio_usr_ids):
            raise ValueError("image_ids length must match user audio list")

        device = audio_usr_ids[0].device

        task_token = self.sptids_dict['<|s2s|>'].to(device).view(-1)
        soa_token = self.sptids_dict['<|soa|>'].to(device).view(-1)
        eoa_token = self.sptids_dict['<|eoa|>'].to(device).view(-1)
        soi_token = self.sptids_dict['<|soi|>'].to(device).view(-1)
        eoi_token = self.sptids_dict['<|eoi|>'].to(device).view(-1)

        user_header = "<|start_header_id|>user<|end_header_id|>\n"
        asst_header = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        u_tokens = self.text_tokenizer(user_header, return_tensors="pt").input_ids.to(device).view(-1)
        a_tokens = self.text_tokenizer(asst_header, return_tensors="pt").input_ids.to(device).view(-1)

        sequences: list[torch.Tensor] = []
        attention_masks: list[torch.Tensor] = []

        for usr_tokens, asst_placeholder, img_tokens in zip(audio_usr_ids, audio_asst_placeholders, image_ids):
            usr_tokens = usr_tokens.to(device).long()
            if usr_tokens.dim() > 1:
                usr_tokens = usr_tokens.view(-1)

            asst_placeholder = asst_placeholder.to(device).long()
            if asst_placeholder.dim() > 1:
                asst_placeholder = asst_placeholder.view(-1)

            seq_parts = [task_token, u_tokens]

            if isinstance(img_tokens, (list, tuple)):
                for seg in img_tokens:
                    if seg is None:
                        continue
                    seg = seg.to(device).long()
                    if seg.dim() > 1:
                        seg = seg.view(-1)
                    seq_parts.extend([soi_token, seg, eoi_token])
            elif img_tokens is not None:
                seg = img_tokens.to(device).long()
                if seg.dim() > 1:
                    seg = seg.view(-1)
                seq_parts.extend([soi_token, seg, eoi_token])

            seq_parts.extend([soa_token, usr_tokens, eoa_token, a_tokens, soa_token, asst_placeholder])

            seq = torch.cat(seq_parts, dim=0)
            attn_mask = torch.ones_like(seq, dtype=torch.long)

            sequences.append(seq.unsqueeze(0))
            attention_masks.append(attn_mask.unsqueeze(0))

        return torch.cat(sequences, dim=0), torch.cat(attention_masks, dim=0)

    def v2s_gen_prompt(
        self,
        video_ids: Union[list[torch.Tensor], torch.Tensor],
        text_ids: list[list[int]],
        audio_placeholders: list[torch.Tensor],
    ):
        if len(text_ids) != len(audio_placeholders):
            raise ValueError("text_ids and audio_placeholders must have the same length")

        if isinstance(video_ids, torch.Tensor):
            video_list = [video_ids[i] for i in range(video_ids.shape[0])]
        else:
            video_list = video_ids

        device = audio_placeholders[0].device

        v2s_token = self.sptids_dict['<|v2s|>'].to(device).view(-1)
        soi_token = self.sptids_dict['<|soi|>'].to(device).view(-1)
        eoi_token = self.sptids_dict['<|eoi|>'].to(device).view(-1)
        soa_token = self.sptids_dict['<|soa|>'].to(device).view(-1)
        eoa_token = self.sptids_dict['<|eoa|>'].to(device).view(-1)

        max_text_len = self.max_text_len - 1
        max_audio_len = self.max_audio_len_short
        eos_id = self.text_tokenizer.eos_token_id

        sequences: list[torch.Tensor] = []
        attention_masks: list[torch.Tensor] = []

        for vid_tokens, txt_ids, audio_placeholder in zip(video_list, text_ids, audio_placeholders):
            if len(txt_ids) == 0:
                txt_ids = [self.text_tokenizer.bos_token_id]
            elif txt_ids[0] != self.text_tokenizer.bos_token_id:
                txt_ids = [self.text_tokenizer.bos_token_id] + txt_ids

            temp_text = txt_ids + [eos_id]
            if len(temp_text) < max_text_len:
                temp_text = temp_text + [eos_id] * (max_text_len - len(temp_text))
            else:
                temp_text = temp_text[:max_text_len - 1] + [eos_id]
            text_tensor = torch.tensor(temp_text, dtype=torch.long, device=device)

            if isinstance(vid_tokens, torch.Tensor):
                vid_tensor = vid_tokens.to(device).long()
            else:
                vid_tensor = torch.tensor(vid_tokens, dtype=torch.long, device=device)
            if vid_tensor.dim() > 1:
                vid_tensor = vid_tensor.view(-1)

            audio_placeholder = audio_placeholder.to(device).long()
            if audio_placeholder.dim() > 1:
                audio_placeholder = audio_placeholder.view(-1)

            audio_block = torch.cat([soa_token, audio_placeholder], dim=0)
            if audio_block.numel() > max_audio_len:
                audio_block = audio_block[:max_audio_len]
            elif audio_block.numel() < max_audio_len:
                pad_len = max_audio_len - audio_block.numel()
                if pad_len > 0:
                    pad_value = audio_placeholder.new_full((pad_len,), audio_placeholder[-1].item())
                    audio_block = torch.cat([audio_block, pad_value], dim=0)

            seq = torch.cat([v2s_token, soi_token, vid_tensor, eoi_token, text_tensor, audio_block], dim=0)
            attn_mask = torch.ones_like(seq, dtype=torch.long)

            sequences.append(seq.unsqueeze(0))
            attention_masks.append(attn_mask.unsqueeze(0))

        return torch.cat(sequences, dim=0), torch.cat(attention_masks, dim=0)

    def mmu_prompt(self, batch_image_ids_list, batch_text_ids, max_text_len_override=None):
        """
        Args:
            batch_image_ids_list: List[List[Tensor]] where each inner list is multiple images (per sample)
            batch_text_ids: List[List[int]] token ids for text
            max_text_len_override: optional max text length
        """
        # Flatten multiple images per sample into one sequence (concatenate tokens)
        image_seqs = []
        for imgs in batch_image_ids_list:
            if len(imgs) == 0:
                raise ValueError("mmu_prompt expects at least one image token sequence per sample.")
            # concatenate multiple images back-to-back
            seq = torch.cat(imgs, dim=0)
            image_seqs.append(seq)

        device = image_seqs[0].device
        sequence_ids = []
        prompt_masks = []
        label_ids = []
        max_text_len = (max_text_len_override if max_text_len_override is not None else self.max_text_len) - 1
        eos_id = self.text_tokenizer.eos_token_id

        for text_ids, image_ids in zip(batch_text_ids, image_seqs):
            if len(text_ids) == 0:
                text_ids = [self.text_tokenizer.bos_token_id]
            elif text_ids[0] != self.text_tokenizer.bos_token_id:
                text_ids = [self.text_tokenizer.bos_token_id] + text_ids

            temp_ids = text_ids + [self.text_tokenizer.eos_token_id]

            if max_text_len >= len(temp_ids):
                pad_len = max_text_len - len(temp_ids)
                temp_ids = temp_ids + [eos_id] * pad_len
                temp_masks = [1] * (len(temp_ids) + image_ids.shape[-1] + 3) + [0] * pad_len
                # ignore padded eos in labels
                pad_labels = torch.full((pad_len,), self.ignore_id, device=device, dtype=torch.long)
            else:
                temp_ids = temp_ids[:max_text_len - 1] + [self.text_tokenizer.eos_token_id]
                temp_masks = [1] * (len(temp_ids) + image_ids.shape[-1] + 3)
                pad_labels = torch.empty(0, device=device, dtype=torch.long)

            temp_label_ids = torch.cat([
                torch.tensor([self.ignore_id], device=device),
                torch.tensor([self.ignore_id], device=device),
                torch.ones_like(image_ids, device=device) * self.ignore_id,
                torch.tensor([self.ignore_id], device=device),
                torch.tensor(temp_ids, device=device),
            ], dim=0)
            if pad_labels.numel() > 0:
                temp_label_ids[-pad_len:] = pad_labels

            return_temp_ids = torch.cat([
                self.sptids_dict['<|mmu|>'].to(device),
                self.sptids_dict['<|soi|>'].to(device),
                image_ids,
                self.sptids_dict['<|eoi|>'].to(device),
                torch.tensor(temp_ids, device=device),
            ], dim=0)

            end_header_id = int(self.sptids_dict['<|end_header_id|>'])
            end_header_pos = -1
            for pos in range(len(temp_ids) - 1, -1, -1):
                if temp_ids[pos] == end_header_id:
                    end_header_pos = pos
                    break
            if end_header_pos != -1:
                prompt_length = len(return_temp_ids) - len(temp_ids) + end_header_pos + 1
            else:
                prompt_length = len(return_temp_ids) - len(temp_ids)
            predict_length = len(return_temp_ids) - prompt_length
            prompt_mask = torch.tensor([1] * prompt_length + [0] * predict_length, device=device)

            sequence_ids.append(return_temp_ids.unsqueeze(0))
            prompt_masks.append(prompt_mask.unsqueeze(0))
            label_ids.append(temp_label_ids.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0), torch.cat(prompt_masks, dim=0), torch.cat(label_ids, dim=0)
    
    def mmu_mult_prompt(self, batch_image_ids_list, batch_text_ids, max_text_len_override=None):
        """
        Multi-image prompt builder (strict whole-image fit).

        INPUTS
        -------
        batch_image_ids_list : List[List[torch.LongTensor]]
            Length = B.
            For sample i, a list of K_i encoded image token tensors, each shape (L_img_k,).
            Example for one sample: [img_ids_0, img_ids_1, ...]
            IMPORTANT: Images are already *encoded* to discrete token IDs.

        batch_text_ids : List[List[int]]
            Length = B.
            For sample i, raw tokenized text IDs (no BOS/EOS added here).
            IMPORTANT: Text is *tokenized*, not encoded beyond text tokenizer IDs.

        RETURNS
        -------
        sequence_ids : torch.LongTensor, shape (B, 1 + max_image_len + max_text_len)
            The model input IDs:
            [<mmu>] + [<soi> image_0 <eoi> ... <soi> image_m <eoi>] + [text_ids (BOS...EOS padded)]
        prompt_masks : torch.LongTensor, shape (B, 1 + max_image_len + max_text_len)
            1 = prompt/context, 0 = generation region
            (split determined by last <|end_header_id|> inside the text segment; if none, entire text is generation)
        label_ids    : torch.LongTensor, shape (B, 1 + max_image_len + max_text_len)
            self.ignore_id for specials & image tokens; text positions = text token IDs
        """

        B = len(batch_text_ids)
        device = (
            batch_image_ids_list[0][0].device
            if (batch_image_ids_list and batch_image_ids_list[0])
            else torch.device("cpu")
        )

        max_text_len  = (max_text_len_override if max_text_len_override is not None else self.max_text_len) - 1
        max_image_len = self.max_image_len

        # Text tokenizer ids
        bos_id = self.text_tokenizer.bos_token_id
        eos_id = self.text_tokenizer.eos_token_id

        # Specials (stored in self.sptids_dict as 1D LongTensors)
        mmu_tok       = self.sptids_dict['<|mmu|>'].to(device)   # task token, shape (1,)
        soi_tok       = self.sptids_dict['<|soi|>'].to(device)   # start of image
        eoi_tok       = self.sptids_dict['<|eoi|>'].to(device)   # end of image
        end_header_id = int(self.sptids_dict['<|end_header_id|>'])

        sequence_rows = []
        mask_rows     = []
        label_rows    = []

        for i in range(B):
            # 1. Build image block under the image budget
            img_parts = []
            used_img_len = 0

            for img_ids in batch_image_ids_list[i]:
                img_ids = img_ids.to(device)
                need = 2 + img_ids.numel()  # <soi> + tokens + <eoi>
                if used_img_len + need <= max_image_len:
                    img_parts.extend([soi_tok, img_ids, eoi_tok])
                    used_img_len += need
                else:
                    continue

            image_block = (
                torch.cat(
                    [p if isinstance(p, torch.Tensor) else torch.tensor([p], device=device) for p in img_parts],
                    dim=0
                )
                if img_parts else torch.empty((0,), dtype=torch.long, device=device)
            )

            # 2. Prepare text to fill the remaining budget
            # Target per-sample total length:
            #   1 (mmu) + max_image_len + max_text_len
            # Prefix currently uses: 1 (mmu) + used_img_len
            # So text must fill: text_budget = max_text_len + (max_image_len - used_img_len)
            text_budget = max_text_len + (max_image_len - used_img_len)

            text_ids = batch_text_ids[i]
            # Ensure BOS at start
            if len(text_ids) == 0:
                text_ids = [bos_id]
            elif text_ids[0] != bos_id:
                text_ids = [bos_id] + text_ids

            # Append EOS and pad/truncate to text_budget
            tmp = text_ids + [eos_id]
            if len(tmp) < text_budget:
                tmp = tmp + [eos_id] * (text_budget - len(tmp))
            else:
                tmp = tmp[:max(text_budget - 1, 0)] + ([eos_id] if text_budget > 0 else [])

            temp_ids = torch.tensor(tmp, dtype=torch.long, device=device)  # text segment

            # 3. Build final sequence 
            prefix = torch.cat([mmu_tok, image_block], dim=0)  # length = 1 + used_img_len
            return_ids = torch.cat([prefix, temp_ids], dim=0)  # length = 1 + used_img_len + text_budget

            # Enforce exact total length
            expected_len = 1 + max_image_len + max_text_len
            assert return_ids.numel() == expected_len, f"got {return_ids.numel()}, want {expected_len}"

            # 4. Labels (ignore for specials/images; supervise text)
            ignore_prefix = torch.full((prefix.numel(),), self.ignore_id, dtype=torch.long, device=device)
            temp_label_ids = torch.cat([ignore_prefix, temp_ids], dim=0)  # same length as return_ids

            # 5. Prompt mask
            # Find last <|end_header_id|> in the TEXT region; generation starts after it.
            end_header_pos = -1
            for pos in range(temp_ids.numel() - 1, -1, -1):
                if temp_ids[pos].item() == end_header_id:
                    end_header_pos = pos
                    break

            if end_header_pos != -1:
                prompt_len = prefix.numel() + (end_header_pos + 1)
            else:
                # Match original behavior: if not found, prompt is only the prefix (images+mmu),
                # and the entire text region is the generation target.
                prompt_len = prefix.numel()

            predict_len = return_ids.numel() - prompt_len
            prompt_mask = torch.cat([
                torch.ones(prompt_len, dtype=torch.long, device=device),
                torch.zeros(max(predict_len, 0), dtype=torch.long, device=device)
            ], dim=0)

            # 6. Collect rows (keep original return structure)
            sequence_rows.append(return_ids.unsqueeze(0))
            mask_rows.append(prompt_mask.unsqueeze(0))
            label_rows.append(temp_label_ids.unsqueeze(0))

        # cat along dim=0
        return torch.cat(sequence_rows, dim=0), torch.cat(mask_rows, dim=0), torch.cat(label_rows, dim=0)

    def v2t_prompt(self, image_ids, text_ids, max_text_len=None):
        device = image_ids.device
        sequence_ids = []
        prompt_masks = []
        label_ids = []
        max_text_len = (max_text_len if max_text_len is not None else self.max_text_len) - 1
        eos_id = self.text_tokenizer.eos_token_id
        for i in range(len(text_ids)):
            # note that, llama3 tokenizer automatically add the bot token at first but without eot
            # for empty list []

            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]

            temp_ids = text_ids[i] + [self.text_tokenizer.eos_token_id]

            if max_text_len >= len(temp_ids):
                pad_len = max_text_len - len(temp_ids)
                temp_ids = temp_ids + [eos_id] * pad_len
                temp_masks = [1] * (len(temp_ids) + image_ids.shape[-1] + 3) + [0] * pad_len
                pad_labels = torch.full((pad_len,), self.ignore_id, device=device, dtype=torch.long)
            else:
                temp_ids = temp_ids[:max_text_len - 1] + [self.text_tokenizer.eos_token_id]
                temp_masks = [1] * (len(temp_ids) + image_ids.shape[-1] + 3)  # +2 for two special tokens
                pad_labels = torch.empty(0, device=device, dtype=torch.long)

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_label_ids = torch.cat([
                torch.tensor([self.ignore_id]).to(device),
                torch.tensor([self.ignore_id]).to(device),
                torch.ones_like(image_ids[i]) * self.ignore_id,
                torch.tensor([self.ignore_id]).to(device),
                torch.tensor(temp_ids).to(device),
            ], dim=0)
            if pad_labels.numel() > 0:
                temp_label_ids[-pad_len:] = pad_labels


            return_temp_ids = torch.cat([
                self.sptids_dict['<|v2t|>'].to(device),  # task token
                self.sptids_dict['<|soi|>'].to(device),
                image_ids[i],
                self.sptids_dict['<|eoi|>'].to(device),
                torch.tensor(temp_ids).to(device),
            ], dim=0)
            end_header_id = int(self.sptids_dict['<|end_header_id|>'])
            end_header_pos = -1
            for pos in range(len(temp_ids) - 1, -1, -1):
                if temp_ids[pos] == end_header_id:
                    end_header_pos = pos
                    break
            if end_header_pos != -1:
                prompt_length = len(return_temp_ids) - len(temp_ids) + end_header_pos + 1
            else:
                prompt_length = len(return_temp_ids) - len(temp_ids)
            predict_length = len(return_temp_ids) - prompt_length
            prompt_mask = [1] * prompt_length + [0] * predict_length
            prompt_mask = torch.tensor(prompt_mask).to(device)
            sequence_ids.append(return_temp_ids.unsqueeze(0))
            prompt_masks.append(prompt_mask.unsqueeze(0))
            label_ids.append(temp_label_ids.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0), torch.cat(prompt_masks, dim=0), torch.cat(label_ids, dim=0)

    def _v2s_prompt_impl(
        self,
        image_ids,
        text_ids,
        audio_ids,
        supervise_padding: bool = True,
        max_text_len=None,
    ):
        """
        image_ids: list[torch.Tensor] or Tensor[B, L_img]
        text_ids : list[list[int]]
        audio_ids: list[torch.Tensor]  # each shaped (1, L_audio)
        """
        device = (image_ids[0].device if isinstance(image_ids, list) else image_ids.device)
        sequence_ids, prompt_masks, label_ids = [], [], []

        max_text_len  = (max_text_len if max_text_len is not None else self.max_text_len) - 1
        max_audio_len = self.max_audio_len_short
        eos_id        = self.text_tokenizer.eos_token_id
        ignore_id     = self.ignore_id

        B = len(text_ids)
        for i in range(B):
            # ---- Text normalize ----
            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]

            temp_text_ids = text_ids[i] + [eos_id]
            if len(temp_text_ids) < max_text_len:
                temp_text_ids = temp_text_ids + [eos_id] * (max_text_len - len(temp_text_ids))
            else:
                temp_text_ids = temp_text_ids[:max_text_len - 1] + [eos_id]
            text_tensor = torch.tensor(temp_text_ids, dtype=torch.long, device=device)

            # ---- Audio block with <soa>/<eoa>, clamp/pad to max_audio_len ----
            soa = self.sptids_dict['<|soa|>'].to(device).unsqueeze(0)  # (1,1)
            eoa = self.sptids_dict['<|eoa|>'].to(device).unsqueeze(0)  # (1,1)
            audio_tensor = audio_ids[i].to(device)  # keep concat on a single device
            audio_block = torch.cat([soa, audio_tensor, eoa], dim=1)   # (1, L+2)
            pre_pad_len = audio_block.shape[1]
            actual_len = min(pre_pad_len, max_audio_len)

            audio_block = audio_block[:, :actual_len]
            if pre_pad_len > max_audio_len and audio_block[0, -1] != eoa[0]:
                audio_block[0, -1] = eoa[0]

            pad_len = max_audio_len - audio_block.shape[1]
            if pad_len > 0:
                pad = torch.full((1, pad_len), eos_id, dtype=torch.long, device=device)
                audio_block = torch.cat([audio_block, pad], dim=1)

            # ---- Sequence: <|v2s|>, <|soi|>, image, <|eoi|>, text, <|soa|>, audio..., <|eoa|>, [pads...] ----
            v2s = self.sptids_dict['<|v2s|>'].to(device).to(torch.long)
            soi = self.sptids_dict['<|soi|>'].to(device).to(torch.long)
            eoi = self.sptids_dict['<|eoi|>'].to(device).to(torch.long)

            img_tokens = image_ids[i] if isinstance(image_ids, list) else image_ids[i]
            img_tokens = img_tokens.to(device).to(torch.long)

            seq = torch.cat([
                v2s, soi, img_tokens, eoi, text_tensor, audio_block.squeeze(0).to(torch.long)
            ], dim=0)

            # ---- Prompt mask: 1 through and including <soa>, then 0 over audio targets ----
            prompt_length = 1 + 1 + img_tokens.shape[-1] + 1 + len(text_tensor) + 1  # +1 for <soa>
            total_length  = seq.shape[0]
            predict_length = actual_len - 1  # exclude <soa>
            padding_region = total_length - prompt_length - predict_length

            tail_mask_value = 0 if supervise_padding else 1
            prompt_mask = torch.tensor(
                [1]*prompt_length +
                [0]*predict_length +
                [tail_mask_value]*padding_region,
                dtype=torch.long,
                device=device,
            )

            # ---- Labels: ignore prompt, then audio after <soa> ----
            audio_targets = audio_block.squeeze(0)[1:actual_len].to(torch.long)
            if supervise_padding:
                padding_labels = audio_block.squeeze(0)[actual_len:].to(torch.long)
            else:
                padding_labels = torch.full((padding_region,), ignore_id, dtype=torch.long, device=device)

            label = torch.cat([
                torch.full((prompt_length,), ignore_id, dtype=torch.long, device=device),
                audio_targets,
                padding_labels,
            ], dim=0)

            # ---- Sanity checks ----
            assert audio_block.shape[1] == max_audio_len
            assert total_length == prompt_length + (max_audio_len - 1)  # targets exclude <soa>
            assert label.shape[0] == total_length
            assert prompt_mask.shape[0] == total_length

            sequence_ids.append(seq.unsqueeze(0))
            prompt_masks.append(prompt_mask.unsqueeze(0))
            label_ids.append(label.unsqueeze(0))

        return (
            torch.cat(sequence_ids, dim=0),
            torch.cat(prompt_masks, dim=0),
            torch.cat(label_ids, dim=0),
        )

    def v2s_prompt(self, image_ids, text_ids, audio_ids, max_text_len=None):
        return self._v2s_prompt_impl(image_ids, text_ids, audio_ids, supervise_padding=True, max_text_len=max_text_len)

    def v2s_prompt_ignore_padding(self, image_ids, text_ids, audio_ids, max_text_len=None):
        return self._v2s_prompt_impl(image_ids, text_ids, audio_ids, supervise_padding=False, max_text_len=max_text_len)

    def s2t_prompt(self, audio_ids, text_ids, max_text_len=None):

        device = audio_ids[0].device
        
        sequence_ids, prompt_masks, label_ids = [], [], []
        effective_text_len = max_text_len if max_text_len is not None else (self.max_text_len - 1)
        max_audio_len = self.max_audio_len + 1
        eos_id = self.text_tokenizer.eos_token_id

        for i in range(len(text_ids)):
            task_tensor = self.sptids_dict['<|s2t|>'].to(device).unsqueeze(0)
            soa_tensor = self.sptids_dict['<|soa|>'].to(device).unsqueeze(0)
            eoa_tensor = self.sptids_dict['<|eoa|>'].to(device).unsqueeze(0)
            current_audio_tokens = audio_ids[i]

            # (<|s2t|>, <|soa|>, <|eoa|>) 
            effective_max_audio = max_audio_len - 3
            if current_audio_tokens.shape[1] > effective_max_audio:
                current_audio_tokens = current_audio_tokens[:, :effective_max_audio]
            
            audio_block = torch.cat([task_tensor, soa_tensor, current_audio_tokens, eoa_tensor], dim=1)
            
            num_padding = max_audio_len - audio_block.shape[1]
            if num_padding > 0:
                padding_tensor = torch.full((1, num_padding), self.pad_id, dtype=torch.long, device=device)
                padded_audio_block = torch.cat([padding_tensor, audio_block], dim=1)
            else:
                padded_audio_block = audio_block
            
            padded_audio_block_len = padded_audio_block.shape[1] 

            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]

            temp_ids = text_ids[i] + [self.text_tokenizer.eos_token_id]
            
            if effective_text_len >= len(temp_ids):
                temp_ids = temp_ids + [eos_id] * (effective_text_len - len(temp_ids))
            else:
                temp_ids = temp_ids[:effective_text_len - 1] + [self.text_tokenizer.eos_token_id]

            return_temp_ids = torch.cat([
                padded_audio_block.squeeze(0),
                torch.tensor(temp_ids, device=device),
            ], dim=0)


            prompt_length = padded_audio_block_len 
            temp_label_ids = torch.cat([
                torch.full((prompt_length,), self.ignore_id, device=device),
                torch.tensor(temp_ids, device=device),
            ], dim=0)
            
            end_header_id = int(self.sptids_dict['<|end_header_id|>'])
            end_header_pos = -1
            for pos in range(len(temp_ids) - 1, -1, -1):
                if temp_ids[pos] == end_header_id:
                    end_header_pos = pos; break
            
            if end_header_pos != -1:
                final_prompt_length = prompt_length + end_header_pos + 1
            else:
                final_prompt_length = prompt_length
                
            prompt_mask = torch.tensor([1] * final_prompt_length + [0] * (len(return_temp_ids) - final_prompt_length), device=device)

            sequence_ids.append(return_temp_ids.unsqueeze(0))
            prompt_masks.append(prompt_mask.unsqueeze(0))
            label_ids.append(temp_label_ids.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0), torch.cat(prompt_masks, dim=0), torch.cat(label_ids, dim=0)

    def t2s_prompt(self, text_ids, audio_ids, max_text_len=None):
        """
        text_ids: list[list[int]]
        audio_ids: list[torch.Tensor]
        """

        device = audio_ids[0].device
        max_text_len = max_text_len if max_text_len is not None else self.max_text_len

        audio_pad_token = self.text_tokenizer.eos_token_id
        max_audio_len = self.max_audio_len 

        sequence_ids, prompt_masks, label_ids = [], [], []
        probs = torch.rand(len(text_ids))

        for i in range(len(text_ids)):
            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]
            temp_ids = [int(self.sptids_dict['<|t2s|>'])] + text_ids[i] + [self.text_tokenizer.eos_token_id]

            if probs[i] < self.cond_dropout_prob:
                temp_ids = [int(self.sptids_dict['<|t2s|>']), self.text_tokenizer.bos_token_id, self.text_tokenizer.eos_token_id]

            if max_text_len >= len(temp_ids):
                temp_ids = [self.pad_id] * (max_text_len- len(temp_ids)) + temp_ids
            else:
                temp_ids = temp_ids[:max_text_len - 1] + [self.text_tokenizer.eos_token_id]
            text_part_len = len(temp_ids)
            
            soa_tensor = self.sptids_dict['<|soa|>'].to(device).unsqueeze(0)
            eoa_tensor = self.sptids_dict['<|eoa|>'].to(device).unsqueeze(0)
            audio_block = torch.cat([soa_tensor, audio_ids[i], eoa_tensor], dim=1)

            if audio_block.shape[1] > max_audio_len:
                audio_block = audio_block[:, :max_audio_len]
                if audio_block[0, -1] != eoa_tensor[0]:
                    audio_block[0, -1] = eoa_tensor[0]

            num_padding = max_audio_len - audio_block.shape[1]
            if num_padding > 0:
                padding_tensor = torch.full((1, num_padding), audio_pad_token, dtype=torch.long, device=device)
                padded_audio_ids = torch.cat([audio_block, padding_tensor], dim=1)
            else:
                padded_audio_ids = audio_block

            seq = torch.cat([
                torch.tensor(temp_ids, device=device),
                padded_audio_ids.squeeze(0)
            ], dim=0)

            prompt_part_len = text_part_len + 1     # add 1 for <soa>
            audio_part_len = max_audio_len - 1      # subsitude 1 for <soa>
            prompt_mask = torch.tensor([1] * prompt_part_len + [0] * audio_part_len, device=device)
            
            label = torch.cat([
                torch.full((prompt_part_len,), self.ignore_id, device=device),      # ignore up to <soa>
                padded_audio_ids.squeeze(0)[1:]                                     # delete <soa> token
            ], dim=0)
            
            sequence_ids.append(seq.unsqueeze(0))
            prompt_masks.append(prompt_mask.unsqueeze(0))
            label_ids.append(label.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0), torch.cat(prompt_masks, dim=0), torch.cat(label_ids, dim=0)


    def t2s_prompt_ignore_padding(self, text_ids, audio_ids, max_text_len=None):

        device = audio_ids[0].device
        max_text_len = max_text_len if max_text_len is not None else self.max_text_len

        audio_pad_token = self.text_tokenizer.eos_token_id
        max_audio_len = self.max_audio_len 

        sequence_ids, prompt_masks, label_ids = [], [], []
        probs = torch.rand(len(text_ids))

        for i in range(len(text_ids)):
            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]
            temp_ids = [int(self.sptids_dict['<|t2s|>'])] + text_ids[i] + [self.text_tokenizer.eos_token_id]

            if probs[i] < self.cond_dropout_prob:
                temp_ids = [int(self.sptids_dict['<|t2s|>']), self.text_tokenizer.bos_token_id, self.text_tokenizer.eos_token_id]

            if max_text_len >= len(temp_ids):
                temp_ids = [self.pad_id] * (max_text_len- len(temp_ids)) + temp_ids
            else:
                temp_ids = temp_ids[:max_text_len - 1] + [self.text_tokenizer.eos_token_id]
            text_part_len = len(temp_ids)
            
            soa_tensor = self.sptids_dict['<|soa|>'].to(device).unsqueeze(0)
            eoa_tensor = self.sptids_dict['<|eoa|>'].to(device).unsqueeze(0)
            audio_block = torch.cat([soa_tensor, audio_ids[i], eoa_tensor], dim=1)

            if audio_block.shape[1] > max_audio_len:
                audio_block = audio_block[:, :max_audio_len]
                if audio_block[0, -1] != eoa_tensor[0]:
                    audio_block[0, -1] = eoa_tensor[0]

            num_padding = max_audio_len - audio_block.shape[1]
            if num_padding > 0:
                padding_tensor = torch.full((1, num_padding), audio_pad_token, dtype=torch.long, device=device)
                padded_audio_ids = torch.cat([audio_block, padding_tensor], dim=1)
            else:
                padded_audio_ids = audio_block

            # Full input sequence
            seq = torch.cat([
                torch.tensor(temp_ids, device=device),
                padded_audio_ids.squeeze(0)
            ], dim=0)

            # Compute lengths for masking/labels
            prompt_part_len = text_part_len + 1  # include <soa>
            actual_audio_target_len = audio_block.shape[1] - 1  # exclude <soa>, include audio VQ codes and <eoa>
            padded_region_len = num_padding  # trailing padding tokens after audio_block

            # Prompt mask: do NOT mask text, <soa>, or trailing padding; mask only real audio targets
            prompt_mask = torch.tensor(
                [1] * prompt_part_len + [0] * actual_audio_target_len + [1] * padded_region_len,
                device=device,
            )

            # Labels: ignore prompt and padding regions; supervise only real audio targets
            label = torch.cat([
                torch.full((prompt_part_len,), self.ignore_id, device=device),
                audio_block.squeeze(0)[1:],  # drop <soa>
                torch.full((padded_region_len,), self.ignore_id, device=device),
            ], dim=0)
            
            sequence_ids.append(seq.unsqueeze(0))
            prompt_masks.append(prompt_mask.unsqueeze(0))
            label_ids.append(label.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0), torch.cat(prompt_masks, dim=0), torch.cat(label_ids, dim=0)

    def _s2s_prompt_impl(
        self,
        audio_usr_ids: list[torch.Tensor],
        audio_asst_ids: list[torch.Tensor],
        image_ids: Optional[list[Optional[torch.Tensor]]] = None,
        supervise_padding: bool = False,
    ):
        if len(audio_usr_ids) != len(audio_asst_ids):
            raise ValueError("audio_usr_ids and audio_asst_ids must have the same length")

        if len(audio_usr_ids) == 0:
            raise ValueError("s2s_prompt requires at least one sample")

        device = audio_usr_ids[0].device

        task_tensor = self.sptids_dict['<|s2s|>'].to(device).unsqueeze(0)
        soa_tensor = self.sptids_dict['<|soa|>'].to(device).unsqueeze(0)
        eoa_tensor = self.sptids_dict['<|eoa|>'].to(device).unsqueeze(0)

        user_header = "<|start_header_id|>user<|end_header_id|>\n"
        asst_header = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        u_tokens = self.text_tokenizer(user_header, return_tensors="pt").input_ids.to(device)
        a_tokens = self.text_tokenizer(asst_header, return_tensors="pt").input_ids.to(device)

        left_pad_id = self.pad_id
        right_pad_id = self.text_tokenizer.eos_token_id
        max_audio_len = self.max_audio_len

        sequence_ids = []
        prompt_masks = []
        label_ids = []

        task_len = task_tensor.shape[1]
        soa_len = soa_tensor.shape[1]
        eoa_len = eoa_tensor.shape[1]
        user_header_len = u_tokens.shape[1]
        asst_header_len = a_tokens.shape[1]

        for usr_tokens, asst_tokens in zip(audio_usr_ids, audio_asst_ids):
            if usr_tokens.device != device:
                usr_tokens = usr_tokens.to(device)
            if asst_tokens.device != device:
                asst_tokens = asst_tokens.to(device)
            if usr_tokens.dim() == 1:
                usr_tokens = usr_tokens.unsqueeze(0)
            if asst_tokens.dim() == 1:
                asst_tokens = asst_tokens.unsqueeze(0)
            usr_tokens = usr_tokens.long()
            asst_tokens = asst_tokens.long()

            max_usr_audio = max_audio_len - (task_len + user_header_len + soa_len + eoa_len)
            max_usr_audio = max(0, max_usr_audio)
            if usr_tokens.shape[1] > max_usr_audio:
                usr_tokens = usr_tokens[:, :max_usr_audio]

            usr_parts = [task_tensor, u_tokens]
            usr_parts.extend([soa_tensor, usr_tokens, eoa_tensor])
            usr_block = torch.cat(usr_parts, dim=1)

            target_usr_len = max_audio_len
            num_usr_pad = target_usr_len - usr_block.shape[1]
            if num_usr_pad < 0:
                num_usr_pad = 0
            if num_usr_pad > 0:
                usr_block = torch.cat([
                    torch.full((1, num_usr_pad), left_pad_id, dtype=torch.long, device=device),
                    usr_block
                ], dim=1)

            max_asst_audio = max_audio_len - (asst_header_len + soa_len + eoa_len)
            max_asst_audio = max(0, max_asst_audio)
            if asst_tokens.shape[1] > max_asst_audio:
                asst_tokens = asst_tokens[:, :max_asst_audio]

            asst_block = torch.cat([a_tokens, soa_tensor, asst_tokens, eoa_tensor], dim=1)
            target_asst_len = max_audio_len
            num_asst_pad = target_asst_len - asst_block.shape[1]
            if num_asst_pad < 0:
                num_asst_pad = 0
            if num_asst_pad > 0:
                asst_block = torch.cat([
                    asst_block,
                    torch.full((1, num_asst_pad), right_pad_id, dtype=torch.long, device=device)
                ], dim=1)

            seq = torch.cat([usr_block, asst_block], dim=1)

            prefix_len = usr_block.shape[1] + asst_header_len + soa_len
            target_len = asst_tokens.shape[1] + eoa_len
            padding_len = asst_block.shape[1] - (asst_header_len + soa_len + target_len)

            mask_segments = [
                torch.ones((prefix_len,), device=device, dtype=torch.long),
                torch.zeros((target_len,), device=device, dtype=torch.long)
            ]
            if padding_len > 0:
                pad_mask_value = 0 if supervise_padding else 1
                mask_segments.append(torch.full((padding_len,), pad_mask_value, device=device, dtype=torch.long))
            prompt_mask = torch.cat(mask_segments, dim=0)

            labels = seq.clone()
            mask_bool = prompt_mask.bool().unsqueeze(0)
            labels[mask_bool] = self.ignore_id

            sequence_ids.append(seq)
            prompt_masks.append(prompt_mask.unsqueeze(0))
            label_ids.append(labels)

        if len(sequence_ids) > 1:
            max_seq_len = max(seq.shape[1] for seq in sequence_ids)
            if any(seq.shape[1] != max_seq_len for seq in sequence_ids):
                padded_sequences = []
                padded_masks = []
                padded_labels = []
                for seq, mask, labels in zip(sequence_ids, prompt_masks, label_ids):
                    seq_pad = max_seq_len - seq.shape[1]
                    if seq_pad > 0:
                        seq = torch.nn.functional.pad(seq, (0, seq_pad), value=right_pad_id)
                        mask = torch.nn.functional.pad(mask, (0, seq_pad), value=1)
                        pad_labels = torch.full((labels.shape[0], seq_pad), self.ignore_id, dtype=labels.dtype, device=device)
                        labels = torch.cat([labels, pad_labels], dim=1)
                    padded_sequences.append(seq)
                    padded_masks.append(mask)
                    padded_labels.append(labels)
                sequence_ids = padded_sequences
                prompt_masks = padded_masks
                label_ids = padded_labels

        return (
            torch.cat(sequence_ids, dim=0),
            torch.cat(prompt_masks, dim=0),
            torch.cat(label_ids, dim=0),
        )

    def s2s_prompt(
        self,
        audio_usr_ids: list[torch.Tensor],
        audio_asst_ids: list[torch.Tensor],
        image_ids: Optional[list[Optional[torch.Tensor]]] = None,
    ):
        return self._s2s_prompt_impl(audio_usr_ids, audio_asst_ids, image_ids, supervise_padding=False)

    def s2s_prompt_eos(
        self,
        audio_usr_ids: list[torch.Tensor],
        audio_asst_ids: list[torch.Tensor],
        image_ids: Optional[list[Optional[torch.Tensor]]] = None,
    ):
        return self._s2s_prompt_impl(audio_usr_ids, audio_asst_ids, image_ids, supervise_padding=True)

    def s2s_prompt_ignore_padding(self, audio_usr_ids: list[torch.LongTensor], audio_asst_ids: list[torch.LongTensor]):
        """
        Args:
            audio_usr_ids: list[torch.LongTensor], each elem is of shape (1, S), S is seq_len
            audio_asst_ids: list[torch.LongTensor], each elem is of shape (1, S), S is seq_len
        Returns:
            sequence_ids: torch.LongTensor, of shape (B, L)
            prompt_masks: torch.LongTensor, of shape (B, L)
            label_ids: torch.LongTensor, of shape (B, L)
        """
        device = audio_usr_ids[0].device
        sequence_ids, prompt_masks, label_ids = [], [], []

        # Pad tokens
        left_pad_id = self.pad_id
        right_pad_id = self.text_tokenizer.eos_token_id

        # Task and special tokens
        task_tensor = self.sptids_dict['<|s2s|>'].to(device).unsqueeze(0)
        soa_tensor = self.sptids_dict['<|soa|>'].to(device).unsqueeze(0)
        eoa_tensor = self.sptids_dict['<|eoa|>'].to(device).unsqueeze(0)

        # Headers for instruction tuning
        u = "<|start_header_id|>user<|end_header_id|>\n"
        a = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        u_tokens = self.text_tokenizer(u, return_tensors="pt")['input_ids']
        a_tokens = self.text_tokenizer(a, return_tensors="pt")['input_ids']

        # Maximum lengths
        max_usr_len = self.max_audio_len
        max_asst_len = self.max_audio_len

        for i in range(len(audio_usr_ids)):

            # User tokens (truncation and left padding)
            current_usr_tokens = audio_usr_ids[i]

            effective_max_audio = max_usr_len - (task_tensor.shape[1] + u_tokens.shape[1] + soa_tensor.shape[1] + eoa_tensor.shape[1])
            if current_usr_tokens.shape[1] > effective_max_audio:
                current_usr_tokens = current_usr_tokens[:, :effective_max_audio]
            
            usr_block = torch.cat([task_tensor, u_tokens, soa_tensor, current_usr_tokens, eoa_tensor], dim=1)

            num_padding = max_usr_len - usr_block.shape[1]
            if num_padding > 0:
                padding_tensor = torch.full((1, num_padding), left_pad_id, dtype=torch.long, device=device)
                padded_usr_block = torch.cat([padding_tensor, usr_block], dim=1)
            else:
                padded_usr_block = usr_block

            # Assistant tokens (truncation and right padding)
            asst_block = torch.cat([a_tokens, soa_tensor, audio_asst_ids[i], eoa_tensor], dim=1)

            if asst_block.shape[1] > max_asst_len:
                asst_block = asst_block[:, :max_asst_len]
                asst_block[0, -1] = eoa_tensor[0]

            num_padding = max_asst_len - asst_block.shape[1]
            if num_padding > 0:
                padding_tensor = torch.full((1, num_padding), right_pad_id, dtype=torch.long, device=device)
                padded_asst_block = torch.cat([asst_block, padding_tensor], dim=1)
            else:
                padded_asst_block = asst_block

            # Full sequence
            seq = torch.cat([
                padded_usr_block,
                padded_asst_block,
            ], dim=1)

            # Mask and labels
            prefix_mask_len = max_usr_len + (a_tokens.shape[1] + soa_tensor.shape[1])  # padded usr block + asst headers + <soa>
            actual_audio_target_len = asst_block.shape[1] - (a_tokens.shape[1] + soa_tensor.shape[1])  # exclude asst headers and <soa> but incldue audio tokens and <eoa>
            prompt_mask = torch.tensor(
                [1] * prefix_mask_len + [0] * actual_audio_target_len + [1] * (seq.shape[1] - prefix_mask_len - actual_audio_target_len),
                device=device, dtype=torch.long).unsqueeze(0)

            labels = seq.clone()
            labels[prompt_mask.bool()] = self.ignore_id

            # Append to lists
            sequence_ids.append(seq)
            prompt_masks.append(prompt_mask)
            label_ids.append(labels)

        return torch.cat(sequence_ids, dim=0), torch.cat(prompt_masks, dim=0), torch.cat(label_ids, dim=0)


    def mmu_gen_prompt(self, image_ids, text_ids):
        device = image_ids.device
        sequence_ids = []
        prompt_masks = []
        max_text_len = self.max_text_len - 1
        for i in range(len(text_ids)):

            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]

            temp_ids = text_ids[i] + [self.text_tokenizer.eos_token_id]

            if max_text_len >= len(temp_ids):
                # minus 1 because task token was prepended to the former image tokens
                temp_ids = temp_ids + [self.pad_id] * (max_text_len - len(temp_ids))
            else:
                # should add the eos token
                temp_ids = temp_ids[:max_text_len - 1] + [self.text_tokenizer.eos_token_id]

            return_temp_ids = torch.cat([
                self.sptids_dict['<|mmu|>'].to(device),  # task token
                self.sptids_dict['<|soi|>'].to(device),
                image_ids[i],
                self.sptids_dict['<|eoi|>'].to(device),
                torch.tensor(temp_ids).to(device),
            ], dim=0)
            
            end_header_id = int(self.sptids_dict['<|end_header_id|>'])
            end_header_pos = -1
            for pos in range(len(temp_ids) - 1, -1, -1):
                if temp_ids[pos] == end_header_id:
                    end_header_pos = pos
                    break
            if end_header_pos != -1:
                prompt_length = len(return_temp_ids) - len(temp_ids) + end_header_pos + 1
            else:
                prompt_length = len(return_temp_ids) - len(temp_ids)
            predict_length = len(temp_ids) - prompt_length
            prompt_mask = [1] * prompt_length + [0] * predict_length
            prompt_mask = torch.tensor(prompt_mask).to(device)
            sequence_ids.append(return_temp_ids.unsqueeze(0))
            prompt_masks.append(prompt_mask.unsqueeze(0))
        return torch.cat(sequence_ids, dim=0), torch.cat(prompt_masks, dim=0)

    def r2i_prompt(self, image_ids, text_ids):
        device = image_ids.device
        sequence_ids = []
        prompt_masks = []
        label_ids = []
        r2i_id = int(self.sptids_dict['<|r2i|>'])
        soi_id = int(self.sptids_dict['<|soi|>'])
        eoi_id = int(self.sptids_dict['<|eoi|>'])
        max_text_len = self.max_text_len - 1    # 512，include BOS text EOS
        for i in range(len(text_ids)):
            # note that, llama3 tokenizer automatically add the bot token at first but without eot
            # for empty list []
            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0]!= self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]
            text_ids_with_bos_eos = text_ids[i] + [self.text_tokenizer.eos_token_id]
            if max_text_len >= len(text_ids_with_bos_eos):
                # minus 1 because task token was prepended to the former image tokens
                text_ids_full_len = text_ids_with_bos_eos + [self.text_tokenizer.eos_token_id] * (max_text_len - len(text_ids_with_bos_eos))
            else:
                # should add the eos token
                text_ids_full_len = text_ids_with_bos_eos[:max_text_len - 1] + [self.text_tokenizer.eos_token_id]
            
            sequence_ids.append(torch.cat([
                torch.tensor([r2i_id]).to(device),  # task token
                torch.tensor(text_ids_full_len).to(device),
                torch.tensor([soi_id]).to(device),
                image_ids[i],
                torch.tensor([eoi_id]).to(device),
            ], dim=0).unsqueeze(0))

            end_header_id = int(self.sptids_dict['<|end_header_id|>'])
            end_header_pos = -1
            for pos in range(len(text_ids_full_len) - 1, -1, -1):
                if text_ids_full_len[pos] == end_header_id:
                    end_header_pos = pos
                    break
            prompt_mask = torch.zeros(sequence_ids[i].size(1)).to(device)
            prompt_mask[0] = 1  # task_id
            if end_header_pos != -1:
                prompt_mask[1:end_header_pos+2] = 1
            else:
                prompt_mask[1:len(text_ids_full_len)+1] = 1
            prompt_mask[len(text_ids_full_len)+1] = 1
            prompt_mask[len(text_ids_full_len)+2+len(image_ids[i])] = 1
            prompt_masks.append(prompt_mask.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0), torch.cat(prompt_masks, dim=0), torch.cat(sequence_ids, dim=0)

    def mask_prompt(self):
        pass

    def __call__(self, input, task, padding=True, config=None):
        """
        input (tuple) : data pairs contain text(str), image(tensor), or videos(tensor).
        task (str) : a flag indicates the current task.
        """
        cfg = config or {}
        max_text_len_override = cfg.get("max_text_len_override")
        effective_max_text_len = max_text_len_override if max_text_len_override is not None else self.max_text_len

        if task == "t2i":
            text_ids = self.text_tokenizer(
                input[0],
                truncation=True,
                max_length=effective_max_text_len,
            )['input_ids']  # (B, max_len)
            image_ids = input[1]  # (B, #tokens)
            sequence_ids_with_masks = self.t2i_prompt(text_ids, image_ids, input[2], max_text_len=effective_max_text_len)
        
        elif task == "i2i":
            text_ids = input[0]
            original_image_ids = input[1]  # (B, #tokens)
            edited_image_ids = input[2]  # (B, #tokens)
            sequence_ids_with_masks = self.i2i_prompt(
                text_ids,
                original_image_ids,
                edited_image_ids,
                input[3],
                max_text_len=effective_max_text_len,
            )

        elif task == "s2t":
            image_ids = input[0]
            text_ids = self.text_tokenizer(
                input[1],
                truncation=True,
                max_length=effective_max_text_len,
            )['input_ids']
            sequence_ids_with_masks = self.s2t_prompt(image_ids, text_ids, max_text_len=effective_max_text_len)
        
        elif task == "t2s":
            audio_ids = input[1]
            text_ids = self.text_tokenizer(
                input[0],
                truncation=True,
                max_length=effective_max_text_len,
            )['input_ids']
            sequence_ids_with_masks = self.t2s_prompt(text_ids, audio_ids, max_text_len=effective_max_text_len)

        elif task == "t2s_ip":
            audio_ids = input[1]
            text_ids = self.text_tokenizer(
                input[0],
                truncation=True,
                max_length=effective_max_text_len,
            )['input_ids']
            sequence_ids_with_masks = self.t2s_prompt_ignore_padding(text_ids, audio_ids, max_text_len=effective_max_text_len)

        elif task == "s2s_ip":
            audio_user_ids = input[0]
            audio_asst_ids = input[1]
            image_ids = input[2] if len(input) > 2 else None
            sequence_ids_with_masks = self.s2s_prompt(audio_user_ids, audio_asst_ids, image_ids)

        elif task == "s2s":
            audio_user_ids = input[0]
            audio_asst_ids = input[1]
            image_ids = input[2] if len(input) > 2 else None
            sequence_ids_with_masks = self.s2s_prompt_eos(audio_user_ids, audio_asst_ids, image_ids)

        elif task == "t2v":
            text_ids = self.text_tokenizer(
                input[0],
                truncation=True,
                max_length=effective_max_text_len,
            )['input_ids']  # (B, max_len)
            image_ids = input[1]  # (B, #tokens)
            sequence_ids_with_masks = self.t2v_prompt(text_ids, image_ids, input[2])

        elif task == "t2i_plus_lm":
            text_ids = self.text_tokenizer(
                input[0],
                truncation=True,
                max_length=effective_max_text_len,
            )['input_ids']  # (B, max_len)
            image_ids = input[1]  # (B, #tokens)
            sequence_ids_with_masks = self.t2i_prompt(
                text_ids[:config.training.batch_size],
                image_ids,
                input[2],
                max_text_len=effective_max_text_len,
            )
            sequence_ids_with_masks_lm = self.lm_prompt(text_ids[config.training.batch_size:], input[3])
            return sequence_ids_with_masks, sequence_ids_with_masks_lm

        elif task == "t2i_gen":
            text_ids = self.text_tokenizer(
                input[0],
                truncation=True,
                max_length=effective_max_text_len,
            )['input_ids']  # (B, max_len)
            image_ids = input[1]  # (B, #tokens)
            sequence_ids_with_masks = self.t2i_gen_prompt(text_ids, image_ids)

        elif task == "t2v_gen":
            text_ids = self.text_tokenizer(
                input[0],
                truncation=True,
                max_length=effective_max_text_len,
            )['input_ids']  # (B, max_len)
            image_ids = input[1]  # (B, #tokens)
            sequence_ids_with_masks = self.t2v_gen_prompt(text_ids, image_ids)

        elif task == "ti2ti_gen":
            prompts = input[0]
            target_texts = input[1]
            source_tokens = input[2]
            placeholder_tokens = input[3]
            sequence_ids_with_masks = self.ti2ti_gen_prompt(prompts, target_texts, source_tokens, placeholder_tokens)

        elif task == "lm":
            text_ids = self.text_tokenizer(input[0], truncation=True)['input_ids']  # (B, max_len)
            sequence_ids_with_masks = self.lm_prompt(text_ids, input[1])

        elif task == "lm_chat":
            text_ids = self.text_tokenizer(input[0], truncation=True)['input_ids']  # (B, max_len)
            sequence_ids_with_masks = self.lm_chat_prompt(text_ids, input[1])

        elif task == "mmu":                  
            text_ids = self.text_tokenizer(
                input[1],
                truncation=True,
                max_length=effective_max_text_len,
            )['input_ids']   # (B, max_len)

            sequence_ids_with_masks = self.mmu_mult_prompt(
                batch_image_ids_list=input[0],
                batch_text_ids=text_ids,
                max_text_len_override=max_text_len_override,
            )

        elif task == "v2t":
            video_ids = input[0]
            text_ids = self.text_tokenizer(
                input[1],
                truncation=True,
                max_length=effective_max_text_len,
            )['input_ids']
            sequence_ids_with_masks = self.v2t_prompt(video_ids, text_ids, max_text_len=effective_max_text_len)
        
        elif task == 'v2s':
            video_ids = input[0]
            text_ids = self.text_tokenizer(
                input[1],
                truncation=True,
                max_length=effective_max_text_len,
            )['input_ids']
            audio_ids = input[2]
            sequence_ids_with_masks = self.v2s_prompt(video_ids, text_ids, audio_ids, max_text_len=effective_max_text_len)

        elif task == 'v2s_ip':
            video_ids = input[0]
            text_ids = self.text_tokenizer(
                input[1],
                truncation=True,
                max_length=effective_max_text_len,
            )['input_ids']
            audio_ids = input[2]
            sequence_ids_with_masks = self.v2s_prompt_ignore_padding(video_ids, text_ids, audio_ids, max_text_len=effective_max_text_len)
        
        elif task == "r2i":
            image_ids = input[0]
            text_ids = self.text_tokenizer(input[1])['input_ids']
            sequence_ids_with_masks = self.r2i_prompt(image_ids, text_ids)

        elif task == "i2i_gen":
            text_ids = input[0]
            input_image_ids = input[1]
            output_image_ids = input[2]
            sequence_ids_with_masks = self.i2i_gen_prompt(text_ids, input_image_ids, output_image_ids)

        elif task == "t2s_gen":
            text_ids = self.text_tokenizer(input[0])['input_ids']  # (B, max_len)
            audio_ids = input[1]  # (B, #tokens)
            sequence_ids_with_masks = self.t2s_gen_prompt(text_ids, audio_ids)

        elif task == "t2s_fixed_gen":
            text_ids = self.text_tokenizer(input[0])['input_ids']  # (B, max_len)
            audio_ids = input[1]  # (B, #tokens)
            sequence_ids_with_masks = self.t2s_fixed_gen_prompt(text_ids, audio_ids)

        elif task == "s2s_gen":
            audio_user_ids = input[0]
            audio_placeholders = input[1]
            image_ids = input[2] if len(input) > 2 else None
            sequence_ids_with_masks = self.s2s_gen_prompt(audio_user_ids, audio_placeholders, image_ids)

        elif task == "v2s_gen":
            video_ids = input[0]
            text_ids = self.text_tokenizer(
                input[1],
                truncation=True,
                max_length=self.max_text_len,
            )['input_ids']
            audio_ids = input[2]
            sequence_ids_with_masks = self.v2s_gen_prompt(video_ids, text_ids, audio_ids)

        else:
            raise NotImplementedError

        return sequence_ids_with_masks


if __name__ == '__main__':
    pass
