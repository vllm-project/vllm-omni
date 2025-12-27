#!/bin/bash
# Launch Fun-Audio-Chat server in S2T (Speech-to-Text) mode

vllm serve FunAudioLLM/Fun-Audio-Chat-8B \
    --omni \
    --port 8091 \
    --trust-remote-code
