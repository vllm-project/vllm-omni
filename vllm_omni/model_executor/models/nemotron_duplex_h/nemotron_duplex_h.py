# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inference-only NemotronDuplexH model for vLLM-Omni.

A minimal extension of the upstream :class:`NemotronHForCausalLM` that:

1. Accepts pre-computed acoustic encoder embeddings per step via
   ``additional_information["acoustic_embedding"]`` (one row per
   scheduled token). The prefill step accepts a pre-computed full
   combined embedding via
   ``additional_information["prefill_combined_embeddings"]``; on this
   path the model's :meth:`preprocess` returns the tensor as-is. The
   producer is expected to send
   ``prefill_combined_embeddings=None`` on every subsequent decode
   chunk so the runner's streaming-input merge overwrites the prefill
   tensor with ``None`` and the decode branch is taken.
2. Embeds an additional per-step ASR token id stream. The id is fed
   **autoregressively from the model itself** via the per-request
   ``input_asr_ids`` buffer that ``postprocess`` keeps populated after
   every step (initial seed = 0, taken from ``additional_information``
   only on the very first prefill chunk if provided).
3. Combines all three signals into the input embedding fed to the
   NemotronH backbone:

       hidden_in = embed_tokens(input_ids)
                 + embed_asr_tokens(input_asr_ids)
                 + acoustic_embedding

4. Adds a parallel ``asr_head`` that produces an ASR token at every
   decoding step. The ``asr_head`` matmul and the ``argmax`` run in
   :meth:`make_omni_output` (which the runner invokes *outside* the
   CUDA-graph wrapper) on the full-batch ``hidden_states`` returned by
   :meth:`forward`. ASR tokens are exposed under
   ``OmniOutput.multimodal_outputs["asr_tokens"]``, and
   :meth:`postprocess` stashes the request's last ASR id back into
   ``input_asr_ids`` so the next step's :meth:`preprocess` can read it
   as the autoregressive ASR id input.

   Returning a dict-with-tensor directly from :meth:`forward` is
   unsafe under FULL CUDA graphs: ``weak_ref_tensors`` cannot weak-ref
   tensors nested inside dicts, and the wrapper coerces ``NamedTuple``
   to a plain ``tuple`` on replay. Routing the multimodal output
   through :meth:`make_omni_output` keeps every cudagraph-replayed
   value a plain ``Tensor``.

Text token sampling is unchanged and goes through the standard
``compute_logits`` -> sampler path of the parent class.
"""

from collections.abc import Iterable
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput


class NemotronDuplexHForCausalLM(NemotronHForCausalLM):
    """NemotronH + per-step ASR token embedding & ASR head."""

    have_multimodal_outputs = True
    has_preprocess = True
    has_postprocess = True

    # Keys whose values stay GPU-resident across decode steps (set by
    # ``postprocess`` and read by the next ``preprocess``).
    gpu_resident_buffer_keys: set[str] = {"input_asr_ids"}

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # NemotronH backbone weights live under
            # `stt_model.llm.backbone.*` in the duplex checkpoint and need to
            # land under our `model.*`.
            "stt_model.llm.backbone": "model",
            "stt_model.llm": "model",
            "stt_model.embed_tokens": "model.embed_tokens",
            "stt_model.embed_asr_tokens": "embed_asr_tokens",
            "stt_model.lm_head": "lm_head",
            "stt_model.asr_head": "asr_head",
            # Bare-NemotronH naming, kept as a fallback.
            "backbone": "model",
        },
        orig_to_new_substr={"A_log": "A", "embeddings": "embed_tokens"},
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        config = vllm_config.model_config.hf_config

        self.embed_asr_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        self.asr_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE,
            prefix=maybe_prefix(prefix, "asr_head"),
        )

    # ------------------------------------------------------------------ #
    #  preprocess                                                        #
    # ------------------------------------------------------------------ #

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Combine text/asr/speech embeddings into a single per-token vector.

        Two paths:

        * **Prefill short-circuit.** When
          ``additional_information["prefill_combined_embeddings"]`` is
          a tensor, the caller has pre-computed the
          ``(T_prefill, hidden_size)`` combined embedding offline. We
          return it as-is, bypassing the text+asr+acoustic sum. The
          producer must send ``prefill_combined_embeddings=None`` on
          every decode chunk to let the streaming-input merge in the
          runner overwrite this slot.

        * **Decode (single-token step).** Builds the combined embedding
          per scheduled token from:

          - ``input_ids``           – per-step text token id (one per
                                      scheduled token; standard vLLM
                                      autoregressive feedback).
          - ``input_asr_ids``       – per-step ASR token id, written
                                      back by :meth:`postprocess` on
                                      every step.
          - ``acoustic_embedding``  – per-step acoustic encoder
                                      embedding, sourced from
                                      ``additional_information``.

          ``input_embeds`` is the runner's pre-allocated scratch buffer
          on this path and its contents are ignored.
        """
        device = input_ids.device
        n = int(input_ids.shape[0])

        # Prefill vs decode is detected directly on the value of
        # ``prefill_combined_embeddings``. The producer sends a tensor
        # on the prefill chunk and ``None`` on every decode chunk; the
        # scheduler's streaming-session update propagates ``None`` to
        # the runner, which then writes it into the per-request
        # intermediate buffer.
        prefill_combined = info_dict.get("prefill_combined_embeddings")
        if isinstance(prefill_combined, torch.Tensor):
            target_dtype = self.model.embed_tokens.weight.dtype
            prefill_combined = prefill_combined.to(
                device=device, dtype=target_dtype
            )
            assert prefill_combined.dim() == 2, (
                f"prefill_combined_embeddings must be 2D, got "
                f"shape {tuple(prefill_combined.shape)}"
            )
            assert prefill_combined.shape[0] == n, (
                f"prefill_combined_embeddings length {prefill_combined.shape[0]} "
                f"does not match scheduled token count {n}"
            )
            return input_ids, prefill_combined, {}

        text_emb = self.model.embed_tokens(input_ids)

        asr_ids = info_dict.get("input_asr_ids")
        assert isinstance(asr_ids, torch.Tensor), "input_asr_ids is not a tensor"
        asr_ids = asr_ids.to(device=device, dtype=torch.long).reshape(-1)

        assert asr_ids.numel() == n, (
            f"input_asr_ids length {asr_ids.numel()} does not match "
            f"scheduled token count {n}"
        )

        asr_emb = self.embed_asr_tokens(asr_ids)

        combined = text_emb + asr_emb

        # Per-step acoustic encoder embedding, sourced from
        # ``additional_information["acoustic_embedding"]``.
        acoustic = info_dict.get("acoustic_embedding")
        assert isinstance(acoustic, torch.Tensor), (
            "acoustic_embedding is required in additional_information "
            "on every decode step"
        )
        acoustic = acoustic.to(device=device, dtype=combined.dtype)
        assert acoustic.dim() == 2, (
            f"acoustic_embedding must be 2D, got "
            f"shape {tuple(acoustic.shape)}"
        )
        assert acoustic.shape[0] == n, (
            f"acoustic_embedding length {acoustic.shape[0]} does not "
            f"match scheduled token count {n}"
        )
        combined = combined + acoustic

        return input_ids, combined, {}

    # ------------------------------------------------------------------ #
    #  postprocess - autoregressive feedback for ASR ids                 #
    # ------------------------------------------------------------------ #

    def postprocess(
        self,
        hidden_states: torch.Tensor,
        multimodal_outputs: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Stash the last asr_token of this request as the next step's input.

        ``hidden_states`` is a slice of the full-batch hidden_states tensor.
        ``multimodal_outputs["asr_tokens"]`` is the full-batch asr_tokens
        tensor produced by :meth:`forward`. We pick the asr token aligned
        with the last position of this request's slice.
        """
        assert multimodal_outputs
        asr_tokens = multimodal_outputs.get("asr_tokens")
        assert isinstance(asr_tokens, torch.Tensor)
        start = hidden_states.storage_offset() // hidden_states.stride(0)
        last_idx = start + hidden_states.shape[0] - 1
        last_asr = asr_tokens[last_idx : last_idx + 1].detach().to(torch.long)
        return {"input_asr_ids": last_asr}

    # ------------------------------------------------------------------ #
    #  forward                                                           #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        """Run the backbone and return its hidden states.

        ASR tokens are produced by :meth:`make_omni_output`, which the
        runner invokes *outside* the CUDA-graph wrapper. This keeps the
        captured graph's output a plain ``Tensor`` (or
        ``IntermediateTensors``) — both types that ``weak_ref_tensors``
        handles correctly. Returning a ``NamedTuple`` containing a
        ``dict[str, Tensor]`` directly here would corrupt the dict's
        tensors on FULL graph replay (the wrapper coerces
        ``NamedTuple`` -> plain ``tuple`` and cannot weak-ref tensors
        nested in dicts).

        IMPORTANT — cudagraph mode requirement
        --------------------------------------
        This model must be run with ``cudagraph_mode="PIECEWISE"`` (or
        ``enforce_eager=True``). The streaming-input pattern used here
        keeps extending each request's prompt with every audio chunk,
        so ``num_computed_tokens < num_prompt_tokens`` is permanently
        true and Mamba's metadata builder always classifies the request
        as a *prefill* (because
        :func:`split_decodes_and_prefills` is called with
        ``treat_short_extends_as_decodes=False`` in
        ``Mamba2AttentionMetadataBuilder._compute_common_metadata``).

        With FULL cudagraph mode, the persistent
        ``state_indices_tensor_d`` buffer is only updated when
        ``num_prefills == 0``, so for streaming it stays at the
        capture-time dummy value (0) while the FULL decode graph is
        still dispatched (the dispatcher only checks ``query_len``).
        The captured Mamba kernel then reads slot 0 of ``mamba_cache``
        instead of the real slot, producing garbage hidden states.
        PIECEWISE side-steps this because the Mamba layer runs eagerly
        and reads the freshly-computed metadata tensor, and the prefill
        code path correctly *writes* the chunk into Mamba state on
        every step (which is essential — there is no separate "prefill"
        phase in this streaming setup).
        """
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    # ------------------------------------------------------------------ #
    #  make_omni_output - runs eagerly outside the CUDA graph wrapper    #
    # ------------------------------------------------------------------ #

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | IntermediateTensors | OmniOutput,
        **_: Any,
    ) -> OmniOutput:
        """Wrap backbone hidden states with ASR tokens.

        Invoked by :class:`OmniGPUModelRunner._model_forward` after the
        CUDA-graph wrapper has returned, so the ``asr_head`` matmul +
        ``argmax`` here run eagerly. They operate on the full-batch
        ``hidden_states`` tensor in a single GEMM, so the cost is
        negligible relative to the backbone forward.
        """
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        if isinstance(model_outputs, IntermediateTensors):
            return OmniOutput(
                text_hidden_states=model_outputs,
                intermediate_tensors=model_outputs,
            )

        hidden = model_outputs
        asr_logits = self.logits_processor(self.asr_head, hidden)
        asr_tokens = torch.argmax(asr_logits, dim=-1).to(torch.long)

        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={"asr_tokens": asr_tokens},
        )

    # ------------------------------------------------------------------ #
    #  weight loading                                                    #
    # ------------------------------------------------------------------ #

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=["mtp"])
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
