# Kimi-Audio-7B for audio2text and audio chat

## Summary

- Vendor: moonshotai
- Model: `moonshotai/Kimi-Audio-7B-Instruct`
- Tasks: `audio2text` (audio in, text out), `audio2audio` (audio in,
  text + spoken audio out, including multi-turn), and `text2audio`
  (text in, audio out)
- Mode: Online serving with the OpenAI-compatible API
- Maintainer: Community

## When to use this recipe

Use this recipe when you want a known-good starting point for serving
`moonshotai/Kimi-Audio-7B-Instruct` with vLLM-Omni in one of three modes:

- **`audio2text`** — transcribe an audio clip with `"modalities": ["text"]`.
- **`audio2audio`** — respond to an audio prompt with text and
  synthesized speech using `"modalities": ["text", "audio"]`.
  Multi-turn audio chat uses the same pipeline.
- **`text2audio`** — synthesize speech from a text prompt with
  `"modalities": ["text", "audio"]`.

The pipeline runs the fused thinker (Whisper-large-v3 + VQ-Adaptor +
Qwen2-7B + 6-layer MIMO branch) on stage 0 and a flow-matching DiT +
BigVGAN code2wav module on stage 1 that turns MIMO-emitted semantic
codes into a 24 kHz waveform.

## References

- Upstream model: [`MoonshotAI/Kimi-Audio`](https://github.com/MoonshotAI/Kimi-Audio)
- For offline inference (including the multi-turn case that requires a
  custom `prompt_token_ids` builder) and a Python streaming client, see
  `examples/offline_inference/kimi_audio/` and
  `examples/online_serving/kimi_audio/`.

## Hardware Support

This recipe documents reference GPU configurations for the two-stage
audio-out deployment. Other hardware and configurations are welcome
as community validation lands.

## GPU

### 1x L4 24GB — single-GPU sync audio chat

The bundled `kimi_audio.yaml` ships in single-GPU sync mode: both
stages pin to `cuda:0`, stage 0 reserves 0.55 of GPU memory, and stage
1 fits alongside it. To enable multi-GPU async-chunk streaming for
sub-second TTFB, edit the YAML per the comments at its top
(`async_chunk: true`, move stage 1 to `devices: "1"`, add a
`SharedMemoryConnector`).

#### Environment

- OS: Linux
- Python: 3.10+
- CUDA driver / runtime: 12.4+
- vLLM: 0.20.0
- vLLM-Omni: 0.20.0rc1 or built from this checkout (`pip install -e .`)

#### Command

```bash
vllm serve moonshotai/Kimi-Audio-7B-Instruct \
    --omni \
    --port 8091 \
    --stage-configs-path vllm_omni/deploy/kimi_audio.yaml
```

For text-out-only (`audio2text`) deployments where you want to skip
loading the MIMO audio branch and save ~4 GB of weights, override
`hf_overrides.kimia_generate_audio: false` on stage 0 in the YAML and
request `"modalities": ["text"]` from the client.

#### Verification

ASR — transcribe a clip:

```bash
curl http://localhost:8091/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "moonshotai/Kimi-Audio-7B-Instruct",
      "modalities": ["text"],
      "sampling_params_list": [
        {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "max_tokens": 512, "seed": 42}
      ],
      "messages": [
        {"role": "user", "content": [
          {"type": "audio_url", "audio_url": {"url": "https://raw.githubusercontent.com/MoonshotAI/Kimi-Audio/master/test_audios/asr_example.wav"}},
          {"type": "text", "text": "请将音频内容转换为文字。"}
        ]}
      ]
    }' | jq -r '.choices[0].message.content'
```

Audio chat — spoken response (saves to `kimi_response.wav`):

```bash
curl http://localhost:8091/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "moonshotai/Kimi-Audio-7B-Instruct",
      "modalities": ["text", "audio"],
      "sampling_params_list": [
        {"temperature": 0.6, "top_p": 0.95, "top_k": 50, "max_tokens": 2048, "seed": 42},
        {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "max_tokens": 16384, "seed": 42, "detokenize": false}
      ],
      "messages": [
        {"role": "user", "content": [
          {"type": "audio_url", "audio_url": {"url": "https://raw.githubusercontent.com/MoonshotAI/Kimi-Audio/master/test_audios/qa_example.wav"}}
        ]}
      ]
    }' | jq -r '.choices[0].message.audio.data' | base64 -d > kimi_response.wav
```

The reply should be ~5 s of clean speech (rms ≈ 0.045, peak ≈ 0.35).

Spoken response from text input ("text-to-audio" mode):

```bash
curl http://localhost:8091/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "moonshotai/Kimi-Audio-7B-Instruct",
      "modalities": ["text", "audio"],
      "sampling_params_list": [
        {"temperature": 0.6, "top_p": 0.95, "top_k": 50, "max_tokens": 2048, "seed": 42},
        {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "max_tokens": 16384, "seed": 42, "detokenize": false}
      ],
      "messages": [
        {"role": "user", "content": [
          {"type": "text", "text": "Please say the following in audio: \"Hello, my name is Kimi.\""}
        ]}
      ]
    }' | jq -r '.choices[0].message.audio.data' | base64 -d > kimi_tts.wav
```

#### Notes

- **One stage YAML ships**: `kimi_audio.yaml`. Default is single-GPU
  sync; flip `async_chunk` and rewire devices for multi-GPU streaming
  per the comments at the top of the YAML.
- **`max_tokens=2048`** on the thinker is the right setting for
  audio-out tasks — the audio head needs room to reach its natural
  EOD (`<|im_msg_end|>` / `<|im_media_end|>`). Smaller caps truncate
  the speech mid-sentence.
- **Audio sampling caveat**: the MIMO audio head samples via greedy
  argmax inside the model and does not flow through vLLM's sampler.
  The thinker's `temperature` / `top_p` / `top_k` only affect the
  text path; audio tokens are deterministic given the text prefix.
  Changing audio quality requires editing
  `_run_mimo_branch` in `kimi_audio_thinker.py`.
- **Multi-turn audio history** (where a prior assistant turn includes
  spoken audio) is not yet wired through the OpenAI chat-completions
  path. For multi-turn, use the offline Python entry point under
  `examples/offline_inference/kimi_audio/` (see `--task multiturn`),
  which builds the prompt manually with GLM-4-Voice tokenization of
  prior assistant audio.
