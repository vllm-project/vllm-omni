# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native Intent decoder with Torchtitan's pre-final-norm H50 sidecar.

Keep the full decoder and stock vision, attention, logits and weight loaders.
The payload contains *scheduled* rows, including graph padding: the runner must
slice it to num_tokens_unpadded before storing or exporting it. Selecting shared
and assistant-body rows, cache generation ownership and resumable MRoPE are
separate integration gates; this adapter alone does not implement streaming.
"""

from __future__ import annotations

from pathlib import Path

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .h50_connector import H50_SIDECAR_KEYS, attach_scheduled_metadata


class DuoIntentProcessingInfo(Qwen3VLProcessingInfo):
    """Frame-v5 observes each temporal patch at its LAST frame, not midpoint.

    Keep native resident vision preprocessing. Do not alter frame indices to
    trick the stock midpoint clock: those indices describe the actual media.
    """

    def _calculate_timestamps(self, indices, video_fps, merge_size):
        indices = indices if isinstance(indices, list) else indices.tolist()
        if not indices or video_fps <= 0 or merge_size < 1:
            raise ValueError("Invalid frame-v5 temporal clock")
        if len(indices) % merge_size:
            indices = indices + [indices[-1]] * (merge_size - len(indices) % merge_size)
        return [indices[i] / video_fps for i in range(merge_size - 1, len(indices), merge_size)]


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=DuoIntentProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class DuoIntentForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Export layer-50 residual-summed rows without altering final logits.

    vLLM's layer index 50 means after decoder layer index 49. Its auxiliary
    collector sums the fused residual representation, matching the HF layer
    output *before* the final language-model RMSNorm. Capture is internal, not
    runner EAGLE output: the runner receives a regular OmniOutput.
    """

    have_multimodal_outputs = True
    supports_eagle = False
    supports_eagle3 = False
    accepts_runner_sampling_extra_args = True
    H50_LAYER = 50
    omni_pooler_payload_include_hidden = False
    omni_client_multimodal_output_keys = H50_SIDECAR_KEYS

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        # Fail before allocating weights for combinations not audited for H50.
        if vllm_config.speculative_config is not None:
            raise ValueError("Duo H50 does not support speculative decoding")
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("Duo H50 currently requires pipeline_parallel_size=1")
        if vllm_config.quant_config is not None:
            raise ValueError("Duo H50 quantization requires a separate parity gate")
        if getattr(getattr(vllm_config, "cache_config", None), "enable_prefix_caching", False):
            raise ValueError("Realization H50 APC requires a validated auxiliary sidecar; disable APC")
        text_config = vllm_config.model_config.hf_config.text_config
        if text_config.num_hidden_layers < self.H50_LAYER:
            raise ValueError("Duo requires the output of decoder layer 50")
        self._duplex_adapter_path = getattr(vllm_config.model_config.hf_config, "intent_adapter_path", None)
        self._duplex_wide_rows = getattr(vllm_config.model_config.hf_config, "duo_wide_rows", False)
        self._duplex_intent_tp = getattr(vllm_config.model_config.hf_config, "duo_intent_tp", False)
        if (
            self._duplex_adapter_path
            and vllm_config.parallel_config.tensor_parallel_size != 1
            and not self._duplex_intent_tp
        ):
            raise ValueError("Fixed FP32 Intent adapter TP>1 requires a separate parity gate")
        if self._duplex_adapter_path and vllm_config.lora_config is not None:
            raise ValueError("Do not combine fixed duplex adapter with dynamic vLLM LoRA")
        self._duplex_forbidden_ids = getattr(vllm_config.model_config.hf_config, "duo_forbidden_token_ids", None)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if self._duplex_forbidden_ids is not None:
            vocab = text_config.vocab_size
            if not isinstance(self._duplex_forbidden_ids, list) or any(
                type(i) is not int or not 0 <= i < vocab for i in self._duplex_forbidden_ids
            ):
                raise ValueError("Invalid model-specific forbidden vocabulary")
            self.register_buffer(
                "_duplex_mask_indices",
                torch.tensor(
                    self._duplex_forbidden_ids, dtype=torch.long, device=self.language_model.lm_head.weight.device
                ),
                persistent=False,
            )
            self._duplex_mask_vocab = vocab
        if self.is_multimodal_pruning_enabled:
            raise ValueError("Duo H50 requires unpruned multimodal row order")
        # Do not enable runner auxiliary/EAGLE mode. Wrap the auxiliary result
        # here, so extract_multimodal_outputs sees structured payload, not a
        # nested (final_hidden, aux_hidden) tuple.
        self.language_model.model._set_aux_hidden_state_layers((self.H50_LAYER,))

    def prepare_runner_inputs(
        self,
        *,
        input_ids,
        positions,
        inputs_embeds,
        req_ids,
        num_computed_tokens,
        num_scheduled_tokens,
        input_ids_buffer,
        sampling_extra_args=(),
        discard_mask=None,
    ):
        # The official hook runs once per real batch, outside CUDA graphs.
        # Freeze scheduled inputs before bookkeeping can reuse the buffers.
        self._realization_scheduled_metadata = attach_scheduled_metadata(
            {}, input_ids_buffer, positions, int(num_scheduled_tokens.sum())
        )
        # CPU scheduling metadata; token-dependent decisions stay on the GPU.
        # Logits are outside the captured decoder, so no graph-key expansion.
        configs = [v.get("realization_action") for v in sampling_extra_args]
        self._realization_action_rows = None
        self._realization_repeat_rows = None
        if any(c is not None for c in configs):
            ends = num_scheduled_tokens.cumsum()
            self._realization_action_rows = (
                input_ids_buffer[torch.as_tensor(ends - 1, device=input_ids_buffer.device)],
                torch.tensor(
                    [
                        (int(done + size) - c["start"], c["limit"], c["wait"], c["speak"], c["eos"])
                        if c
                        else (-1, 0, 0, 0, 0)
                        for done, size, c in zip(num_computed_tokens, num_scheduled_tokens, configs)
                    ],
                    device=input_ids_buffer.device,
                ),
            )
            pairs, penalties = [], []
            for row, (done, size, config) in enumerate(zip(num_computed_tokens, num_scheduled_tokens, configs)):
                if not config or "output_repeat" not in config:
                    continue
                if not 1 < int(done + size) - config["start"] < config["limit"]:
                    continue
                penalty, bigrams = config["output_repeat"]
                pairs.extend((row, prefix, target) for prefix, target in bigrams)
                penalties.extend([penalty] * len(bigrams))
            if pairs:
                self._realization_repeat_rows = (
                    torch.tensor(pairs, device=input_ids_buffer.device),
                    torch.tensor(penalties, dtype=torch.float32, device=input_ids_buffer.device),
                )
        return input_ids, positions

    def post_sample_multimodal_outputs(self, *, req_ids, invalid_req_indices, multimodal_outputs):
        # Sampled tails are not H50 rows; only the captured forward inputs join
        # the native per-request scheduled-row slicing/transport path.
        return {**multimodal_outputs, **self._realization_scheduled_metadata}

    def compute_logits(self, hidden_states):
        logits = super().compute_logits(hidden_states)
        if self._duplex_forbidden_ids is not None:
            from .token_mask import apply_forbidden_mask

            logits = apply_forbidden_mask(logits, self._duplex_mask_indices, self._duplex_mask_vocab)
        action = getattr(self, "_realization_action_rows", None)
        if action is not None and logits is not None:
            from .token_mask import normalize_action_logits

            normalize_action_logits(logits, *action)
            if self._realization_repeat_rows is not None:
                from .token_mask import apply_assistant_bigram_penalty

                apply_assistant_bigram_penalty(logits, action[0], *self._realization_repeat_rows)
        return logits

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> OmniOutput:
        output = super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("Native Qwen3-VL did not return the requested H50 auxiliary output")
        final_hidden, auxiliary = output
        if not isinstance(auxiliary, list) or len(auxiliary) != 1:
            raise RuntimeError("Expected exactly one pre-final-norm Intent H50 tensor")
        h50 = auxiliary[0]
        if not isinstance(final_hidden, torch.Tensor) or not isinstance(h50, torch.Tensor):
            raise TypeError("Intent final hidden states and H50 must be tensors")
        if h50.shape != final_hidden.shape:
            raise RuntimeError("Intent H50 row/width geometry differs from final hidden states")
        return OmniOutput(
            text_hidden_states=final_hidden,
            multimodal_outputs={"hidden_states": {"layers": {self.H50_LAYER: h50}}},
        )

    def apply_duplex_kv_reanchor(self, runner, scheduler_output):
        from .native_window import apply_window

        apply_window(runner, scheduler_output)

    def load_weights(self, weights):
        if not self._duplex_adapter_path:
            return super().load_weights(weights)
        from .intent_adapter import install_fixed_intent_adapter, load_intent_adapter

        config = self.config.text_config.to_dict()
        adapter = load_intent_adapter(Path(self._duplex_adapter_path), config)
        if config["vocab_size"] != adapter.effective_vocab_size:
            raise ValueError("Set native text_config.vocab_size to adapter tokenizer length before model creation")

        def resized_weights():
            for name, value in weights:
                if name in ("model.language_model.embed_tokens.weight", "lm_head.weight"):
                    if value.shape[0] < adapter.effective_vocab_size:
                        raise ValueError("Base vocabulary expansion requires explicit initialization parity")
                    value = value[: adapter.effective_vocab_size]
                yield name, value

        loaded = super().load_weights(resized_weights())
        install_fixed_intent_adapter(self, adapter, experimental_tp=self._duplex_intent_tp)
        return loaded
