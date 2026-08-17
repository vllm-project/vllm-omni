# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma-2-2b-it text encoder for PiD.

Loads via :func:`from_pretrained_with_prefetch` (HF-cache-tolerant, parallel
prefetch) and inherits :class:`torch.nn.Module` so the encoder is visible in
the pipeline's module tree (component discovery, CPU-offload classification).

Replicates the chi-prompt prefixing + max_length padding + select_index slicing
from PixelDiTModel._encode_text_raw so the caption embedding fed to PidNet
matches the training distribution exactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch

# Chi-prompt prefix used by the SFT-distill experiments. Every caption is
# prefixed with this prompt-engineering string before encoding so the Gemma
# hidden states match the training distribution. MUST stay in sync with
# PiD/pid/_src/configs/pid/experiment/shared_config.py::_CHI_PROMPT.
_CHI_PROMPT = [
    'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:',  # noqa: E501
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",  # noqa: E501
    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",  # noqa: E501
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",  # noqa: E501
    "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",  # noqa: E501
    "User Prompt: ",
]

# Matches PidNet's txt_max_length (see config.py _SHARED_BACKBONE).
_MODEL_MAX_LENGTH = 300

# Mapping PidDecodeConfig.precision -> torch dtype.
_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class GemmaTextEncoder(nn.Module):
    def __init__(
        self,
        model_id: str = "Efficient-Large-Model/gemma-2-2b-it",
        precision: str = "bfloat16",
    ):
        super().__init__()
        if precision not in _DTYPE_MAP:
            raise ValueError(f"precision must be one of {list(_DTYPE_MAP)}, got {precision!r}")
        dtype = _DTYPE_MAP[precision]

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_side = "right"

        # Use the vllm-omni cache-tolerant loader. Gemma's weights live at the
        # repo root (no subfolder); pass subfolder="" and prefetch_list=().
        decoder = from_pretrained_with_prefetch(
            AutoModelForCausalLM.from_pretrained,
            model_id,
            subfolder="",
            prefetch_list=(),
            torch_dtype=dtype,
        ).get_decoder()
        self.model = decoder.to(get_local_device())
        self.model.eval()
        self.model.requires_grad_(False)

        # Chi-prompt prefix joined into a single string, matching
        # PixelDiTModel.__init__ which does "\n".join(config.chi_prompt).
        self._chi_prompt_str = "\n".join(_CHI_PROMPT)
        self._num_chi_tokens = len(self.tokenizer.encode(self._chi_prompt_str))

    @torch.no_grad()
    def encode(self, captions: list[str]) -> torch.Tensor:
        """Encode captions -> (B, model_max_length, 2304) hidden states.

        Replicates PixelDiTModel._encode_text_raw: prepend chi-prompt, pad to
        max_length, run Gemma decoder, then slice via select_index to keep BOS +
        the last (model_max_length - 1) tokens. The PiD student was trained with
        this fixed 300-token layout (chi-prompt tail + caption + right-padding)
        and no attention mask is applied downstream, so matching it is required
        for correct image quality.
        """
        prompts_all = [self._chi_prompt_str + cap for cap in captions]
        max_length_all = self._num_chi_tokens + _MODEL_MAX_LENGTH - 2

        caption_token = self.tokenizer(
            prompts_all,
            max_length=max_length_all,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.model.device)

        caption_embs = self.model(
            caption_token.input_ids,
            caption_token.attention_mask,
        )[0]

        select_index = [0] + list(range(-_MODEL_MAX_LENGTH + 1, 0))
        caption_embs = caption_embs[:, select_index]
        return caption_embs
