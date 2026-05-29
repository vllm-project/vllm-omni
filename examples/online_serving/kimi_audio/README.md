# Kimi-Audio-7B online serving

vLLM-Omni exposes Kimi-Audio through the OpenAI-compatible
`/v1/chat/completions` endpoint. All three task modes — audio-in /
text-out (`audio2text`), audio-in / audio-out (`audio2audio`), and
text-in / audio-out (`text2audio`) — flow over chat.

## Launch the server

```bash
vllm serve moonshotai/Kimi-Audio-7B-Instruct --omni --port 8091 \
    --stage-configs-path vllm_omni/deploy/kimi_audio.yaml
```

Single-GPU sync by default. To enable multi-GPU async-chunk streaming
for sub-second TTFB, edit the YAML per the comments at its top.

The flow-matching detokenizer + BigVGAN vocoder is vendored in-tree
under `vllm_omni/model_executor/models/kimi_audio/`.

## Send a request

`run_curl.sh` covers all three tasks:

```bash
TASK=audio2text  bash run_curl.sh
TASK=audio2audio bash run_curl.sh   # writes response.wav
TASK=text2audio  bash run_curl.sh   # writes response.wav
```

Overrides: `PORT`, `HOST`, `OUT_FILE`, `QUESTION`.

## Endpoint summary

| Task          | Response field                                         |
|---------------|--------------------------------------------------------|
| `audio2text`  | `choices[0].message.content`                           |
| `audio2audio` | `choices[0].message.audio.data` (WAV b64, 24 kHz mono) |
| `text2audio`  | `choices[0].message.audio.data` (WAV b64, 24 kHz mono) |

For audio-out responses the request must set `modalities: ["text", "audio"]`.

## Multi-GPU (TP)

To shard the fused thinker, set `tensor_parallel_size: 2`,
`devices: "0,1"`, and `distributed_executor_backend: "mp"` on stage 0
in the deploy YAML; move stage 1 onto a free device (e.g.
`devices: "2"`). The Qwen2 backbone shards via vLLM's standard TP path;
the 6-layer MIMO branch is replicated on every rank (greedy sampling
keeps ranks in sync without extra collectives).
