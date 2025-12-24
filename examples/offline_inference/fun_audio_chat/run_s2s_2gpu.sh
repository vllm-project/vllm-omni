#!/bin/bash
# Run Fun-Audio-Chat in S2S mode with 2 GPUs (recommended)
# GPU 0: Main model (Stage 0)
# GPU 1: CRQ Decoder + CosyVoice (Stages 1-2)

# The default fun_audio_chat_s2s.yaml config already uses 2 GPUs
python end2end.py --mode s2s --output-dir output_s2s
