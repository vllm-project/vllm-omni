# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

# ---------------------------------------------------------------------------
# Token IDs (confirmed from tokenizer_config.json)
# ---------------------------------------------------------------------------

AUDIO_DUMMY_TOKEN_ID = 151705  # <audioPatch>
AUDIO_START_TOKEN_ID = 151706  # <audio>
AUDIO_END_TOKEN_ID = 151707  # </audio>
AUDIO_EOS_TOKEN_ID = 151704  # <end_of_audio>
VISION_START_TOKEN_ID = 151652  # <|vision_start|>

TEXT_EOS_TOKEN_ID = 151669  # <text_eos>


# ---------------------------------------------------------------------------
# MoE (bailing_moe / Ming-omni-tts-16.8B-A3B) token IDs
# (confirmed from the bailing tokenizer_config.json — different vocab than the
# dense Qwen2 tokenizer above). The bailing tokenizer has NO <text_eos>; the
# audio AR loop terminates purely on the stop-head (see upstream
# modeling_bailingmm.sample), so <end_of_audio> doubles as the AR stop token.
# ---------------------------------------------------------------------------

MOE_AUDIO_DUMMY_TOKEN_ID = 126357  # <audioPatch>
MOE_AUDIO_START_TOKEN_ID = 126358  # <audio>
MOE_AUDIO_END_TOKEN_ID = 126359  # </audio>
MOE_AUDIO_EOS_TOKEN_ID = 126356  # <end_of_audio>
MOE_TEXT_EOS_TOKEN_ID = 126356  # no <text_eos> in bailing; reuse <end_of_audio> as AR stop
MOE_SPK_TOKEN_ID = 126368  # <spk> — speaker-embedding placeholder (dense uses <|vision_start|>)


# ---------------------------------------------------------------------------
# Architectural constants (confirmed from original config.json)
# ---------------------------------------------------------------------------

LATENT_DIM = 64
PATCH_SIZE = 4
HISTORY_PATCH_SIZE = 32
LLM_HIDDEN_SIZE = 896
LLM_VOCAB_SIZE = 151936
AGGREGATOR_HIDDEN_SIZE = 1024
VAE_PATCH_SIZE = 4
SAMPLE_RATE = 44100

# AudioVAE frame/hop geometry (confirmed)
AUDIO_FRAME_HOP = 882  # enc input_dim / hop_size / dec output_dim

# stop_head defaults
STOP_HEAD_MIN_STEPS = 3
STOP_HEAD_THRESHOLD = 0.5

# FlowLoss sampling defaults
DEFAULT_CFG = 2.0
DEFAULT_SIGMA = 0.25
DEFAULT_TEMPERATURE = 0.0

# Connector / Stage-2 streaming defaults (runtime tuning)
LATENT_CHUNK_SIZE = 25
LATENT_LEFT_CONTEXT = 0
MAX_DECODE_STEPS = 200


# ---------------------------------------------------------------------------
# seq_data.extra_data keys
# ---------------------------------------------------------------------------

KEY_LATENT_HISTORY = "ming_latent_history"
KEY_DECODE_STEP = "ming_decode_step"
KEY_LAST_STOP_PROB = "ming_last_stop_prob"
KEY_NEXT_EMBEDS = "ming_next_embeds"
KEY_PROMPT_LATENTS = "ming_prompt_latents"
KEY_SPEAKER_EMBEDDING = "ming_speaker_embedding"
KEY_REQUEST_ID = "ming_request_id"
KEY_CHUNK_ID = "ming_chunk_id"
KEY_CFG = "ming_cfg"
KEY_SIGMA = "ming_sigma"
KEY_TEMPERATURE = "ming_temperature"
KEY_MAX_DECODE_STEPS = "ming_max_decode_steps"
KEY_MIN_DECODE_STEPS = "ming_min_decode_steps"
KEY_TEXT_MODE = "ming_text_mode"
