#!/usr/bin/env bash
set -euo pipefail

SEED=42

thinker_sampling_params='{
  "temperature": 0.0,
  "top_p": 1.0,
  "top_k": -1,
  "max_tokens": 2048,
  "seed": 42,
  "detokenize": true,
  "repetition_penalty": 1.1
}'

talker_sampling_params='{
  "temperature": 0.9,
  "top_p": 0.8,
  "top_k": 40,
  "max_tokens": 2048,
  "seed": 42,
  "detokenize": true,
  "repetition_penalty": 1.05,
  "stop_token_ids": [8294]
}'

code2wav_sampling_params='{
  "temperature": 0.0,
  "top_p": 1.0,
  "top_k": -1,
  "max_tokens": 2048,
  "seed": 42,
  "detokenize": true,
  "repetition_penalty": 1.1
}'
# Above is optional, it has a default setting in stage_configs of the corresponding model.

output=$(curl -sS -X POST http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @- <<EOF
{
  "model": "Qwen/Qwen2.5-Omni-7B",
  "extra_body": {
    "sampling_params_list": [
      $thinker_sampling_params,
      $talker_sampling_params,
      $code2wav_sampling_params
    ]
  },
  "messages": [
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Explain the system architecture for a scalable audio generation pipeline. Answer in 15 words."
        }
      ]
    }
  ]
}
EOF
)

# Here it only shows the text content of the first choice. Audio content has many binaries, so it's not displayed here.
echo "Output of request: $(echo "$output" | jq '.choices[0].message.content')"
