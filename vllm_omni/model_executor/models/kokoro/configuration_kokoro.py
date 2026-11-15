# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Kokoro-82M config.

The checkpoint's ``config.json`` names neither ``model_type`` nor
``architectures``; ``OmniEngineArgs`` injects both from the pipeline's
``model_arch`` (see ``_ARCH_TO_MODEL_TYPE``). The Kokoro fields are kept as
shipped. The top-level LM fields below only size the AR scheduler that drives
the per-request chunk generator: the network has no KV cache and no vocabulary
logits, so they do not describe the model itself.
"""

from transformers import PretrainedConfig

# Token ids of the scheduler-only "vocabulary" compute_logits() emits:
# KOKORO_CONTINUE_TOKEN_ID keeps a request alive for its next chunk,
# KOKORO_STOP_TOKEN_ID finishes it.
KOKORO_STOP_TOKEN_ID = 0
KOKORO_CONTINUE_TOKEN_ID = 1


class KokoroConfig(PretrainedConfig):
    model_type = "kokoro"

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        n_token: int = 178,
        hidden_dim: int = 512,
        style_dim: int = 128,
        n_layer: int = 3,
        max_dur: int = 50,
        text_encoder_kernel_size: int = 5,
        plbert: dict | None = None,
        istftnet: dict | None = None,
        sample_rate: int = 24000,
        # One AR step synthesizes one <=510-phoneme chunk, so this caps the
        # chunks per request (prompt is a single token).
        max_position_embeddings: int = 4096,
        **kwargs,
    ):
        kwargs.setdefault("eos_token_id", KOKORO_STOP_TOKEN_ID)
        super().__init__(**kwargs)
        self.vocab = dict(vocab or {})
        self.n_token = n_token
        self.hidden_dim = hidden_dim
        self.style_dim = style_dim
        self.n_layer = n_layer
        self.max_dur = max_dur
        self.text_encoder_kernel_size = text_encoder_kernel_size
        self.plbert = dict(plbert or {})
        self.istftnet = dict(istftnet or {})
        self.sample_rate = sample_rate

        self.max_position_embeddings = max_position_embeddings
        self.vocab_size = n_token
        self.hidden_size = hidden_dim
        self.num_attention_heads = 1
        self.num_key_value_heads = 1
        self.num_hidden_layers = 1


__all__ = ["KOKORO_CONTINUE_TOKEN_ID", "KOKORO_STOP_TOKEN_ID", "KokoroConfig"]
