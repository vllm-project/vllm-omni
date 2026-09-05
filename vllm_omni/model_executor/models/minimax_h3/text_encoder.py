# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax H3 text encoder on the vLLM Qwen3-VL model runner."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import regex as re
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_omni.model_executor.models.minimax_h3.conditioning import (
    MINIMAX_H3_CONDITION_LABELS_KEY,
    MINIMAX_H3_PRESENTATION_TASK_KEY,
)
from vllm_omni.model_executor.models.minimax_h3.preprocessing import (
    IMAGE_PAD,
    VIDEO_PAD,
    VISION_END,
    VISION_START,
    build_minimax_h3_presentation,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput


def _build_minimax_h3_presentation(
    tokenizer: Any,
    *,
    prompt: str,
    task: str,
    condition_labels: list[tuple[str, int]],
    image_grid_thw: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None,
    video_timestamps: Sequence[Sequence[float]] | None,
    merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the same token stream consumed by the fused H3 encoder."""
    return build_minimax_h3_presentation(
        tokenizer,
        prompt=prompt,
        task=task,
        condition_labels=condition_labels,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        video_timestamps=video_timestamps,
        merge_size=merge_size,
    )


class MiniMaxH3MultiModalProcessor(Qwen3VLMultiModalProcessor):
    """Qwen3-VL media processing with H3's exact segmented presentation."""

    @staticmethod
    def _base_processor_kwargs(
        kwargs: Mapping[str, object],
    ) -> dict[str, object]:
        return {
            key: value
            for key, value in kwargs.items()
            if key
            not in {
                MINIMAX_H3_PRESENTATION_TASK_KEY,
                MINIMAX_H3_CONDITION_LABELS_KEY,
            }
        }

    @staticmethod
    def _condition_labels(value: object) -> list[tuple[str, int]]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return []
        labels: list[tuple[str, int]] = []
        for item in value:
            if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
                raise ValueError("MiniMax H3 condition labels must be (type, index) pairs")
            labels.append((str(item[0]), int(item[1])))
        return labels

    def _cached_apply_hf_processor(self, inputs: Any, timing_ctx: Any):
        # H3 presentation IDs depend on the processed grids for every media
        # item. On a sender-cache hit, the generic processor passes only cache
        # misses to _apply_hf_processor_main, so the full presentation cannot
        # be reconstructed there. Reprocess H3 media to preserve exact IDs.
        return self._apply_hf_processor(inputs, timing_ctx)

    def _apply_hf_processor_main(
        self,
        prompt: str | list[int],
        mm_items: Any,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
        *,
        enable_hf_prompt_update: bool,
    ):
        task_value = hf_processor_mm_kwargs.get(MINIMAX_H3_PRESENTATION_TASK_KEY)
        if task_value is None:
            return super()._apply_hf_processor_main(
                prompt,
                mm_items,
                hf_processor_mm_kwargs,
                tokenization_kwargs,
                enable_hf_prompt_update=enable_hf_prompt_update,
            )
        if not isinstance(prompt, str):
            raise ValueError("MiniMax H3 presentation requires the original prompt text")

        task = str(task_value)
        condition_labels = self._condition_labels(hf_processor_mm_kwargs.get(MINIMAX_H3_CONDITION_LABELS_KEY))
        base_kwargs = self._base_processor_kwargs(hf_processor_mm_kwargs)
        image_count = sum(kind == "image" for kind, _ in condition_labels)
        video_count = sum(kind == "video" for kind, _ in condition_labels)
        image_slot = f"{VISION_START}{IMAGE_PAD}{VISION_END}"
        hf_processor = self.info.get_hf_processor(**base_kwargs)
        video_slot = (
            VIDEO_PAD if self._expands_only_video_token(hf_processor) else f"{VISION_START}{VIDEO_PAD}{VISION_END}"
        )
        processing_prompt = image_slot * image_count + video_slot * video_count
        if not processing_prompt:
            processing_prompt = prompt

        _, processed_data, _ = super()._apply_hf_processor_main(
            processing_prompt,
            mm_items,
            base_kwargs,
            tokenization_kwargs,
            enable_hf_prompt_update=enable_hf_prompt_update,
        )
        image_processor = self.info.get_image_processor(**base_kwargs)
        ids, _ = _build_minimax_h3_presentation(
            self.info.get_tokenizer(),
            prompt=prompt,
            task=task,
            condition_labels=condition_labels,
            image_grid_thw=processed_data.get("image_grid_thw"),
            video_grid_thw=processed_data.get("video_grid_thw"),
            video_timestamps=processed_data.get("timestamps"),
            merge_size=int(image_processor.merge_size),
        )
        return ids.tolist(), processed_data, True

    def _get_prompt_updates(
        self,
        mm_items: Any,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: Any,
    ):
        return super()._get_prompt_updates(
            mm_items,
            self._base_processor_kwargs(hf_processor_mm_kwargs),
            out_mm_kwargs,
        )


class _ResidualMerge(nn.Module):
    """Return the decoder residual stream without the final RMSNorm."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, None]:
        if residual is not None:
            hidden_states = hidden_states + residual
        return hidden_states, None


@MULTIMODAL_REGISTRY.register_processor(
    MiniMaxH3MultiModalProcessor,
    info=Qwen3VLProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class MiniMaxH3TextEncoder(Qwen3VLForConditionalGeneration):
    """Qwen3-VL encoder used by MiniMax H3.

    MiniMax H3 consumes the residual stream after decoder layer 50.  The
    upstream vLLM Qwen3-VL implementation supplies the vision tower,
    multimodal processor, M-RoPE, DeepStack injection, paged attention, and
    tensor-parallel decoder layers.  This class changes only the two parts of
    the contract that differ from causal generation: decoder depth and output
    normalization.
    """

    have_multimodal_outputs = True
    omni_pooler_payload_include_hidden = False
    requires_full_prefix_cached_hidden_states = False
    num_encoder_layers = 50

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("MiniMax H3 text encoding supports tensor parallelism, not pipeline parallelism")

        hf_config = copy.deepcopy(vllm_config.model_config.hf_config)
        hf_config.text_config.num_hidden_layers = self.num_encoder_layers
        # Reuse the token embedding as the unused LM head, avoiding a second
        # vocab-sized allocation in the upstream constructor.  Set the flag
        # on both levels because VllmConfig.with_hf_config propagates the
        # multimodal parent value into its nested text config.
        hf_config.tie_word_embeddings = True
        hf_config.text_config.tie_word_embeddings = True
        encoder_config = vllm_config.with_hf_config(hf_config)
        super().__init__(vllm_config=encoder_config, prefix=prefix)

        self.language_model.model.norm = _ResidualMerge()
        self._token_tags: torch.Tensor | None = None

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Qwen3-VL expands every visual block to exactly
        # vision_start + image/video tokens + vision_end.  Classifying each
        # token directly makes tags independent of prefill chunk boundaries.
        visual_token = (
            (input_ids == self.config.vision_start_token_id)
            | (input_ids == self.config.vision_end_token_id)
            | (input_ids == self.config.image_token_id)
            | (input_ids == self.config.video_token_id)
        )
        token_tags = (~visual_token).long()
        self._token_tags = None
        embeddings = super().embed_input_ids(
            input_ids,
            multimodal_embeddings,
            is_multimodal=is_multimodal,
        )
        self._token_tags = token_tags
        return embeddings

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | OmniOutput,
        **_: object,
    ) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        if self._token_tags is None:
            raise RuntimeError("token tags were not prepared before MiniMax H3 text encoding")
        token_tags = self._token_tags
        self._token_tags = None
        padding = model_outputs.shape[0] - token_tags.shape[0]
        if padding < 0:
            raise RuntimeError(
                "MiniMax H3 token tags exceed the model output length: "
                f"tags={token_tags.shape[0]}, outputs={model_outputs.shape[0]}"
            )
        if padding:
            # Multimodal embedding runs on scheduled tokens, while the AR
            # forward may use a CUDA-graph-padded token buffer.  Match that
            # buffer so both conditioning tensors are recognized as
            # per-token prefix-cache entries; runner slicing removes these
            # padding rows before the stage payload is emitted.
            token_tags = torch.cat((token_tags, token_tags.new_ones(padding)))
        # Keep the stage-wire representation two-dimensional.  The Omni
        # runner treats tensors with shape [tokens, features] as per-token
        # data, so they are concatenated across chunked-prefill steps and
        # reconstructed on prefix-cache hits.  The Stage 1 adapter restores
        # H3's semantic [tokens] representation.
        token_tags = token_tags.unsqueeze(-1)
        return OmniOutput(
            text_hidden_states=model_outputs,
            multimodal_outputs={
                "hidden_states": {"output": model_outputs},
                "meta": {"token_role_ids": token_tags},
            },
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        **_: object,
    ) -> torch.Tensor:
        """Emit EOS immediately; the useful result is the prefill hidden state."""
        logits = hidden_states.new_full(
            (hidden_states.shape[0], self.config.text_config.vocab_size),
            torch.finfo(hidden_states.dtype).min,
        )
        logits[:, int(self._tokenizer.eos_token_id)] = 0
        return logits

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        def encoder_weights():
            for name, weight in weights:
                layer_match = re.match(r"model\.language_model\.layers\.(\d+)\.", name)
                if layer_match and int(layer_match.group(1)) >= self.num_encoder_layers:
                    continue
                if name == "model.language_model.norm.weight" or name.startswith("lm_head."):
                    continue
                yield name, weight

        loader = AutoWeightsLoader(self)
        return loader.load_weights(
            encoder_weights(),
            mapper=self.hf_to_vllm_mapper,
        )
