# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audio8 TTS Preview 0.1b -- hybrid-Mamba2 slow AR (``slow_backbone = "falcon_h1"``).

The 0.1b checkpoint shares ``model_type = "arktts"`` and the codec / fast-DualAR
shape with the 0.6b, but its slow branch is a ``FalconH1Model`` (hybrid Mamba2 +
attention) with muP multipliers, so it cannot reuse the dense Qwen2 path in
``audio8_tts_slow_ar.py``.

This stage wraps vLLM's native ``FalconH1ForCausalLM`` -- the same pattern
``nemotron_voicechat_thinker`` uses for ``NemotronHForCausalLM`` -- and declares
the hybrid-Mamba interfaces so vLLM sizes a mamba state cache for the SSM layers
alongside the paged KV cache for the attention layers.

SCAFFOLD STATUS (see the design note on issue #6495):
  * Implemented and CPU-verified: the flat-config -> ``FalconH1Config`` mapping
    (``configuration_audio8_tts.Audio8TTSHybridSlowARConfig``), registry entry,
    pipeline resolver, and the hybrid-state classmethod delegation below.
  * NOT implemented yet -- these need a GPU + vLLM environment to develop
    against, and are deliberately left as explicit TODOs:
      - the compact semantic head (``semantic_output``: hidden -> 4096 + 1 EOS)
        with ``lm_head_multiplier`` applied, replacing the dense full-vocab head;
      - the codebook / fast-AR / codec-code wiring shared with the dense stage
        (requires extracting ``_build_backbone`` / ``_build_heads`` hooks from
        ``Audio8TTSSlowARForGeneration`` instead of duplicating ~500 lines);
      - ``load_weights`` prefix mapping for the 0.1b checkpoint;
      - the Qwen2-vs-FalconH1 RoPE style check and the parity run against the
        reference ``modeling_arktts.py``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix

from .configuration_audio8_tts import Audio8TTSConfig, Audio8TTSHybridSlowARConfig

logger = init_logger(__name__)

#: vLLM-native hybrid backbone (``vllm/model_executor/models/falcon_h1.py``).
_LM_ARCHITECTURE = "FalconH1ForCausalLM"


class Audio8TTSHybridSlowARForConditionalGeneration(nn.Module, HasInnerState, IsHybrid):
    """Stage 0 for the 0.1b checkpoint: text -> semantic tokens + codec codes."""

    prefer_model_sampler = True

    # ---- hybrid-Mamba interfaces -------------------------------------------
    # Delegation mirrors nemotron_voicechat_thinker.py: the state shapes come
    # from the wrapped backbone, which reads the same FalconH1Config we build in
    # ``Audio8TTSHybridSlowARConfig``.

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig) -> tuple[torch.dtype, torch.dtype]:
        from vllm.model_executor.models.falcon_h1 import FalconH1ForCausalLM

        return FalconH1ForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig) -> tuple[tuple[int, int], tuple[int, int, int]]:
        """Size the mamba caches from the hybrid text config.

        ``FalconH1ForCausalLM.get_mamba_state_shape_from_config`` reads
        ``model_config.hf_config`` directly, which here is the outer
        ``Audio8TTSConfig`` -- it carries no ``mamba_*`` fields (they live on
        ``get_text_config()``), so delegating would raise ``AttributeError``.
        Mirrors vLLM's own body with the config swapped for the text config.
        """
        from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
        # Copied from vLLM 0.32.0 FalconH1ForCausalLM (only the config source
        # differs), so re-check this body when the mamba2_state_shape
        # signature changes on a vLLM bump.

        hf_config = vllm_config.model_config.hf_config.get_text_config()
        intermediate_size = (
            int(hf_config.mamba_expand * hf_config.hidden_size)
            if hf_config.mamba_d_ssm is None
            else hf_config.mamba_d_ssm
        )
        return MambaStateShapeCalculator.mamba2_state_shape(
            intermediate_size=intermediate_size,
            tp_world_size=vllm_config.parallel_config.tensor_parallel_size,
            n_groups=hf_config.mamba_n_groups,
            num_heads=hf_config.mamba_n_heads,
            head_dim=hf_config.mamba_d_head,
            state_size=hf_config.mamba_d_state,
            conv_kernel=hf_config.mamba_d_conv,
            num_spec=vllm_config.num_speculative_tokens,
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        from vllm.model_executor.models.falcon_h1 import FalconH1ForCausalLM

        return FalconH1ForCausalLM.get_mamba_state_copy_func()

    # ---- construction ------------------------------------------------------

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        config: Audio8TTSConfig = vllm_config.model_config.hf_config  # type: ignore[assignment]
        self.config = config
        text_config = config.get_text_config()
        if not isinstance(text_config, Audio8TTSHybridSlowARConfig):
            raise ValueError(
                "Audio8TTSHybridSlowARForConditionalGeneration requires a checkpoint with "
                f"slow_backbone='falcon_h1' (got {getattr(config, 'slow_backbone', None)!r} with "
                f"text_config={type(text_config).__name__})."
            )
        self.text_config: Audio8TTSHybridSlowARConfig = text_config
        self.fast_ar_config = config.fast_ar_config

        # muP scalars: absent (=1.0) on the 0.6b, non-trivial on the 0.1b. The
        # MLP / SSM multipliers are already applied inside vLLM's FalconH1
        # (``_init_mup_vector`` / ``FalconH1MLP``); these two are ours.
        self.embedding_multiplier = float(getattr(text_config, "embedding_multiplier", 1.0))
        self.lm_head_multiplier = float(getattr(text_config, "lm_head_multiplier", 1.0))

        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = True
        self.mtp_hidden_size = int(text_config.hidden_size)
        self.talker_mtp_output_key = ("codes", "audio")
        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {("hidden_states", "last")}
        self.talker_mtp_graph_safe = True

        # FalconH1 hybrid backbone on vLLM native layers. ``hf_config`` is a
        # real FalconH1Config subclass, so the wrapped model reads it unchanged.
        self.model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=text_config,
            prefix=maybe_prefix(prefix, "model"),
            architectures=[_LM_ARCHITECTURE],
        )

        # TODO(#6495): compact semantic head -- Linear(hidden, codebook_size + 1,
        # bias=False) matching the reference ``semantic_output``, with
        # ``lm_head_multiplier`` applied to its output. Deliberately not the
        # dense stage's full-vocab ParallelLMHead + allowed-mask path.
        # TODO(#6495): codebook_embeddings / fast AR / codec-code wiring, shared
        # with the dense stage via extracted hooks (see module docstring).

        logger.info(
            "Audio8 TTS 0.1b (hybrid slow AR): hidden=%d layers=%d mamba(d_ssm=%d n_heads=%d d_state=%d) "
            "muP(embed=%s lm_head=%s)",
            text_config.hidden_size,
            text_config.num_hidden_layers,
            getattr(text_config, "mamba_d_ssm", -1),
            getattr(text_config, "mamba_n_heads", -1),
            getattr(text_config, "mamba_d_state", -1),
            self.embedding_multiplier,
            self.lm_head_multiplier,
        )

    # ---- vLLM hooks (surface only; see module docstring for TODOs) ---------

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ):
        # vLLM's ``FalconH1Model.forward`` already multiplies by
        # ``config.embedding_multiplier`` on both the token-id and the
        # ``inputs_embeds`` path, so scaling here would apply it twice
        # (0.1089^2) on the embeddings path. Nothing to do in the wrapper.
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor, **_: Any) -> torch.Tensor:
        raise NotImplementedError(
            "Audio8 TTS 0.1b is a work-in-progress scaffold: the compact semantic head "
            "is not implemented yet (see issue #6495 design note §5)."
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        raise NotImplementedError(
            "Audio8 TTS 0.1b is a work-in-progress scaffold: the checkpoint prefix map "
            "is not implemented yet (slow.* / semantic_output.* / fast_*.* -- see issue #6495)."
        )


__all__ = ["Audio8TTSHybridSlowARForConditionalGeneration"]
