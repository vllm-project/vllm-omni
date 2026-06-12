# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Alpamayo multimodal processor.

Two responsibilities, exactly as the SGLang reference processor:

1. **Add the new Alpamayo tokens** to the tokenizer — the ``traj_vocab_size``
   discrete trajectory tokens ``<i0>..<i(N-1)>`` plus the Alpamayo special tokens
   (``<|traj_history|>``, ``<|traj_future_start|>``, ...). This is done in
   :meth:`AlpamayoProcessingInfo.get_hf_processor`, so the tokenizer the engine
   uses to encode prompts knows these tokens (and their ids match the checkpoint
   vocab, 155697).

2. **Reuse Qwen3-VL's preprocess** — ``AlpamayoMultiModalProcessor`` /
   ``AlpamayoDummyInputsBuilder`` subclass the base Qwen3-VL classes so the
   multi-camera image processing (smart-resize, patching, grid_thw, vision
   placeholders) is inherited unchanged.

The raw ego history used to *decode* trajectories travels separately via
``extra_args["robot_obs"]`` (GR00T-style) and is handled by the Stage-1 pipeline,
not here.
"""

from __future__ import annotations

import logging

from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)

from vllm_omni.model_executor.models.alpamayo.processing import add_alpamayo_tokens

logger = logging.getLogger(__name__)


class AlpamayoProcessingInfo(Qwen3VLProcessingInfo):
    """Qwen3-VL processing info that injects the Alpamayo trajectory tokens.

    The Alpamayo-1.5 checkpoint ships only model weights + config — no tokenizer
    or image processor files. The user supplies them via ``--tokenizer
    /path/to/Qwen3-VL-8B-Instruct``; we override ``get_hf_processor`` so the
    processor is loaded from that tokenizer path (where the processor files
    actually live), not from ``model_config.model`` (which super() uses and
    where the files are absent).
    """

    def get_hf_processor(self, **kwargs: object):
        from transformers import AutoProcessor
        from vllm.transformers_utils.processor import get_processor

        model_config = self.ctx.model_config
        tok_path = getattr(model_config, "tokenizer", None) or model_config.model
        # Match super()'s behavior (use_fast=True default) and trust_remote_code.
        use_fast = bool(kwargs.pop("use_fast", True))
        # Pin the image-pixel band to upstream NVIDIA Alpamayo 1.5's
        # helper.get_processor defaults (see alpamayo1.5 helper.py:23-25).
        # Without this, vllm-omni inherits the 8B-Instruct preprocessor's
        # default range [65536, 16_777_216] and feeds the VLM ~10× more
        # visual tokens per image than the model was trained on, which
        # measurably hurts CoT quality and trajectory ADE.
        kwargs.setdefault("min_pixels", 163840)
        kwargs.setdefault("max_pixels", 196608)
        try:
            processor = get_processor(
                tok_path,
                trust_remote_code=getattr(model_config, "trust_remote_code", False),
                use_fast=use_fast,
                **kwargs,
            )
        except Exception:
            # Last-resort plain AutoProcessor (older configs that miss the
            # registered class hook).
            processor = AutoProcessor.from_pretrained(
                tok_path,
                trust_remote_code=True,
                use_fast=use_fast,
                min_pixels=kwargs["min_pixels"],
                max_pixels=kwargs["max_pixels"],
            )
        self._ensure_traj_tokens(processor)
        return processor

    def _ensure_traj_tokens(self, processor) -> None:
        # Idempotent: ctx caches the processor, so only add once.
        if getattr(processor, "_alpamayo_tokens_added", False):
            return
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            return
        if "<i0>" in tokenizer.get_vocab():
            processor._alpamayo_tokens_added = True
            return

        hf_config = self.get_hf_config()
        info = add_alpamayo_tokens(
            tokenizer,
            traj_vocab_size=int(getattr(hf_config, "traj_vocab_size", 4000)),
            add_special_tokens=bool(getattr(hf_config, "add_special_tokens", True)),
        )
        processor._alpamayo_tokens_added = True
        logger.info(
            "Alpamayo processor: added trajectory tokens (start_idx=%s, future_start=%s, vocab=%d)",
            info["traj_token_start_idx"],
            info["traj_token_ids"].get("future_start"),
            len(tokenizer),
        )


class AlpamayoMultiModalProcessor(Qwen3VLMultiModalProcessor):
    """Reuses Qwen3-VL's full image/video preprocess unchanged."""


class AlpamayoDummyInputsBuilder(Qwen3VLDummyInputsBuilder):
    """Reuses Qwen3-VL's dummy-input generation."""


__all__ = [
    "AlpamayoProcessingInfo",
    "AlpamayoMultiModalProcessor",
    "AlpamayoDummyInputsBuilder",
]
