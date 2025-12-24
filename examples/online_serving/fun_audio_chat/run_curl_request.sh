#!/bin/bash
# Send request via curl to Fun-Audio-Chat API
# Usage: bash run_curl_request.sh [text|use_audio]

QUERY_TYPE="${1:-use_audio}"
API_BASE="http://localhost:8091/v1"
MODEL="FunAudioLLM/Fun-Audio-Chat-8B"

if [ "$QUERY_TYPE" == "text" ]; then
    # Text-only request
    curl -X POST "${API_BASE}/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"${MODEL}"'",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful voice assistant."
                },
                {
                    "role": "user",
                    "content": "Hello! How are you today?"
                }
            ],
            "max_tokens": 2048,
            "temperature": 0.7
        }' | jq .
else
    # Audio request (using test audio URL)
    AUDIO_URL="https://upload.wikimedia.org/wikipedia/commons/d/d5/English_spoken_voice.ogg"

    curl -X POST "${API_BASE}/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"${MODEL}"'",
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful voice assistant."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": "'"${AUDIO_URL}"'"}},
                        {"type": "text", "text": "What did you hear in the audio?"}
                    ]
                }
            ],
            "max_tokens": 2048,
            "temperature": 0.7
        }' | jq .
fi
