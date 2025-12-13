# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import PIL.Image


def pil_img2rgb(image: PIL.Image.Image) -> PIL.Image.Image:
    """Match Bagel's behavior: ensure RGB, handling alpha by compositing on white."""
    if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
        image = image.convert("RGBA")
        white = PIL.Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
        white.paste(image, mask=image.split()[3])
        return white
    return image.convert("RGB")


def add_special_tokens(tokenizer) -> tuple[Any, dict[str, int], int]:
    """Resolve Bagel's required special token IDs.

    IMPORTANT: Do not mutate the tokenizer (no add_tokens) during inference.
    Adding tokens changes token IDs but the checkpoint embedding matrix is not
    resized/updated, which will cause out-of-bounds embedding lookups on GPU.
    """

    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}

    def _tok_id(tok: str) -> int:
        if tok in vocab:
            return int(vocab[tok])
        tid = tokenizer.convert_tokens_to_ids(tok)
        # Many tokenizers return unk_token_id for unknown tokens.
        if tid is None or (hasattr(tokenizer, "unk_token_id") and tid == tokenizer.unk_token_id):
            raise ValueError(
                f"Bagel tokenizer is missing required token {tok!r}. "
                "Please use the tokenizer shipped with the Bagel checkpoint."
            )
        return int(tid)

    new_token_ids = dict(
        bos_token_id=_tok_id("<|im_start|>"),
        eos_token_id=_tok_id("<|im_end|>"),
        start_of_image=_tok_id("<|vision_start|>"),
        end_of_image=_tok_id("<|vision_end|>"),
    )
    return tokenizer, new_token_ids, 0


@dataclass
class BagelGenParams:
    # Default inference knobs (mirrors Bagel inferencer defaults)
    cfg_text_scale: float = 4.0
    cfg_img_scale: float = 1.5
    cfg_interval: tuple[float, float] = (0.4, 1.0)
    cfg_renorm_min: float = 0.0
    cfg_renorm_type: str = "global"
    num_timesteps: int = 50
    timestep_shift: float = 3.0
