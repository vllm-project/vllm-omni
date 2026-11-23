# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from Tencent-Hunyuan/Hy-Nomas (Copyright 2026 Tencent).

from types import SimpleNamespace

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

_VIDEO_SYSTEM = (
    "You are a helpful assistant. Describe the video by detailing the following aspects:         "
    "1. The main content and theme of the video.         "
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.         "
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.         "
    "4. background environment, light, style and atmosphere.         "
    "5. camera angles, movements, and transitions used in the video."
)


class TextEncoder(nn.Module):
    def __init__(self, text_encoder_path, tokenizer_path, *, device):
        super().__init__()
        model = AutoModel.from_pretrained(text_encoder_path, low_cpu_mem_usage=True)
        if hasattr(model, "language_model"):
            model = model.language_model
        self.model = model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="right")
        self.prompt_template_video = {
            "template": [{"role": "system", "content": _VIDEO_SYSTEM}, {"role": "user", "content": "{}"}],
            "crop_start": -1,
        }
        self.text2tokens("a photo of a cat", data_type="video")

    def text2tokens(self, text, data_type="video", max_length=300):
        if data_type != "video":
            raise ValueError("Hy-Nomas text encoding supports video prompts.")
        texts = [text] if isinstance(text, str) else text
        if not isinstance(texts, (tuple, list)) or not texts or not all(isinstance(item, str) for item in texts):
            raise TypeError("Hy-Nomas text must be a string or a nonempty sequence of strings.")
        chats = [
            [{"role": "system", "content": _VIDEO_SYSTEM}, {"role": "user", "content": item or " "}] for item in texts
        ]
        crop = self.prompt_template_video["crop_start"]
        if crop == -1:
            temporary = self.tokenizer.apply_chat_template(
                chats,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                truncation=True,
                max_length=256,
                padding="max_length",
                return_tensors="pt",
            )
            ids = temporary["input_ids"][0].tolist()
            marker = self.tokenizer("<|im_start|>user\n", add_special_tokens=False)["input_ids"]
            crop = next(
                (
                    index + len(marker)
                    for index in range(len(ids) - len(marker) + 1)
                    if ids[index : index + len(marker)] == marker
                ),
                0,
            )
            self.prompt_template_video["crop_start"] = crop
        return self.tokenizer.apply_chat_template(
            chats,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            truncation=True,
            max_length=max_length + max(crop, 0),
            padding="max_length",
            return_tensors="pt",
        )

    def encode(self, tokens, *, data_type="video", device=None, is_uncond=False):
        if data_type != "video":
            raise ValueError("Hy-Nomas text encoding supports video prompts.")
        target = self.model.device if device is None else device
        mask = tokens["attention_mask"].to(target)
        outputs = self.model(input_ids=tokens["input_ids"].to(target), attention_mask=mask, output_hidden_states=True)
        crop = max(0, self.prompt_template_video["crop_start"])
        return SimpleNamespace(hidden_state=outputs.hidden_states[-3][:, crop:], attention_mask=mask[:, crop:])
