# Copyright 2025 vLLM-Omni Team
"""Centralized constants for Kimi Audio models.

These values are tied to the Kimi Audio tokenizer / model configuration and
should be imported by all Kimi Audio modules instead of duplicating magic
numbers.
"""

# Audio comprehension / input sample rate (Whisper encoder expects 16kHz)
KIMI_AUDIO_SAMPLE_RATE = 16000

# Special token IDs in the unified Kimi Audio vocabulary
KIMI_AUDIO_BOS_TOKEN_ID = 151643  # [BOS] - tokenizer prepends this; strip for prompts/targets
KIMI_AUDIO_EOS_TOKEN_ID = 151644  # [EOS] - engine-level stop token
KIMI_AUDIO_TEXT_EOS_TOKEN_ID = 151667  # <|im_kimia_text_eos|>
KIMI_AUDIO_BLANK_TOKEN_ID = 151666  # <|im_kimia_text_blank|>

# Audio stream end-of-data markers
KIMI_AUDIO_AUDIO_EOD_TOKEN_IDS = {151645, 151663}  # <|im_msg_end|>, <|im_media_end|>

# Message / media boundary markers used when building dual-stream prompts
KIMI_AUDIO_USER_MSG_START_TOKEN_ID = 151670  # <|im_kimia_user_msg_start|>
KIMI_AUDIO_ASSISTANT_MSG_START_TOKEN_ID = 151671  # <|im_kimia_assistant_msg_start|>
KIMI_AUDIO_MEDIA_BEGIN_TOKEN_ID = 151661  # <|im_media_begin|>
KIMI_AUDIO_MEDIA_END_TOKEN_ID = 151663  # <|im_media_end|>
KIMI_AUDIO_MSG_END_TOKEN_ID = 151645  # <|im_msg_end|>

# Continuation tokens that tell the model which modality to generate next
KIMI_AUDIO_SPEECH_CT_ID_TOKEN_ID = 151675  # <|im_kimia_speech_ct_id|> -> generate text
KIMI_AUDIO_SPEECH_CTD_ID_TOKEN_ID = 151676  # <|im_kimia_speech_ctd_id|> -> generate audio

# Vocabulary layout
KIMI_AUDIO_TOKEN_OFFSET = 152064  # First audio semantic token ID
KIMI_AUDIO_TEXT_VOCAB_SIZE = 152064  # Number of text tokens
KIMI_AUDIO_SEMANTIC_VOCAB_SIZE = 16384  # Number of audio semantic tokens
KIMI_AUDIO_TOTAL_VOCAB_SIZE = 168448  # text + audio tokens

# Dual-stream audio generation
KIMI_AUDIO_DELAY = 6  # First N audio tokens are forced to BLANK

# Detokenizer output sample rate
KIMI_AUDIO_OUTPUT_SAMPLE_RATE = 24000

# Stage-transfer chunking for audio semantic tokens
CODEC_CHUNK_FRAMES = 50
CODEC_LEFT_CONTEXT_FRAMES = 0
