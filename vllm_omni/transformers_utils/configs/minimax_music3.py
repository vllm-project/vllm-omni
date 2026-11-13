# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HuggingFace config for the MiniMax Music 3 composite repository.

The checkpoint root carries only::

    {"architectures": ["MiniMaxMusic3ForConditionalGeneration"],
     "model_type": "minimax_music3"}

and ships no modelling code, so ``trust_remote_code`` has nothing to load and
``AutoConfig`` rejects the repository outright. Stage 0 sidesteps this by
declaring ``model_subdir="language_model"`` and reading the plain
``Qwen3ForCausalLM`` config, but the acoustic stage must open the root to reach
``condition_encoder/``, ``transformer/`` and ``vocoder/``. Registering this
class is what lets that stage build a ``ModelConfig`` at all.

The generic transformer fields describe the AR backbone, because the composite
root has no transformer of its own: they are metadata for vLLM's config
plumbing, not a description of the acoustic stage. The acoustic stage holds no
vLLM attention layers, so no KV cache is derived from them; it reads its real
geometry from ``models/minimax_music3/constants.py`` and its weights from the
component folders.
"""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig


class MiniMaxMusic3Config(PretrainedConfig):
    """Root config for the multi-component MiniMax Music 3 repository."""

    model_type = "minimax_music3"

    def __init__(self, **kwargs: Any) -> None:
        # AR backbone geometry (language_model/config.json). Kept in sync with
        # the checkpoint so max_model_len validation sees the real context.
        kwargs.setdefault("hidden_size", 4096)
        kwargs.setdefault("num_hidden_layers", 36)
        kwargs.setdefault("num_attention_heads", 32)
        kwargs.setdefault("num_key_value_heads", 8)
        kwargs.setdefault("head_dim", 128)
        kwargs.setdefault("intermediate_size", 12288)
        kwargs.setdefault("vocab_size", 200000)
        kwargs.setdefault("max_position_embeddings", 10240)
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)

        # Audio contract, mirroring models/minimax_music3/constants.py. These
        # are fixed by the checkpoint; they are exposed here only so a caller
        # inspecting the config sees the same numbers the model uses.
        self.audio_frame_rate = 25
        self.num_codebooks = 8
        self.audio_vocab_size = 1024
        self.c0_vocab_size = 16384
        self.sample_rate = 32000
        self.audio_channels = 2


AutoConfig.register("minimax_music3", MiniMaxMusic3Config)

__all__ = ["MiniMaxMusic3Config"]
