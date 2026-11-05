# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

AUDIO_PATCH_TOKEN_ID = 157164
AUDIO_START_TOKEN_ID = 157165
AUDIO_END_TOKEN_ID = 157166
SPEAKER_START_TOKEN_ID = 157175
SPEAKER_END_TOKEN_ID = 157176
TEXT_EOS_TOKEN_ID = 156892

LATENT_DIM = 64
FLOW_PATCH_SIZE = 1
HISTORY_SIZE = 1
SPEAKER_EMBEDDING_DIM = 192
SAMPLE_RATE = 44100
AUDIO_FRAME_HOP = 882
VAE_PATCH_SIZE = 4

# Ming's cond + cfg * (cond - uncond) uses cfg=1 for antoc's uncond + 2 * (cond - uncond).
DEFAULT_CFG = 1.0
DEFAULT_SIGMA = 0.25
DEFAULT_TEMPERATURE = 0.0
MIN_DECODE_STEPS = 5
MAX_DECODE_STEPS = 800
STOP_THRESHOLD = 0.5

LATENT_CHUNK_SIZE = 25
INITIAL_LATENT_CHUNK_SIZE = 4
LATENT_LEFT_CONTEXT = 0

KEY_LAST_STOP_PROB = "ming3_last_stop_prob"
KEY_SPEAKER_EMBEDDING = "ming3_speaker_embedding"
KEY_SPEAKER_WAVEFORM = "ming3_speaker_waveform"
KEY_SPEAKER_WAVEFORM_LENGTHS = "ming3_speaker_waveform_lengths"
KEY_SPEAKER_SAMPLE_RATES = "ming3_speaker_sample_rates"
KEY_REQUEST_ID = "ming3_request_id"
KEY_CFG = "ming3_cfg"
KEY_SIGMA = "ming3_sigma"
KEY_TEMPERATURE = "ming3_temperature"
KEY_MAX_DECODE_STEPS = "ming3_max_decode_steps"
KEY_MIN_DECODE_STEPS = "ming3_min_decode_steps"
KEY_STOP_REASON = "ming3_stop_reason"
KEY_SPEAKER_POSITIONS = "ming3_speaker_positions"
KEY_PROMPT_AUDIO_START = "ming3_prompt_audio_start"
