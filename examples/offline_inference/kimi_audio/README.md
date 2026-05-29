# Kimi-Audio-7B offline inference

Unified `end2end.py` supports four task modes via the `--task` flag:

  * `audio2text`  — audio in, text out (ASR over upstream `asr_example.wav`)
  * `audio2audio` — audio in, audio + text out (spoken QA over upstream
    `qa_example.wav`, audio-only user turn — no text instruction)
  * `multiturn`   — multi-turn `audio2audio` (q1 → assistant audio+text
    a1 → q2). Uses a custom prompt builder that pre-tokenizes prior
    assistant audio with the GLM-4-Voice tokenizer.
  * `text2audio`  — text in, audio + text out (TTS-style)

All tasks share `vllm_omni/deploy/kimi_audio.yaml`.
Per-stage sampling params come from that file; `--stage-configs-path` /
`--output-dir` / `--question` override the defaults.

## Run

Defaults pull audio from
[upstream Kimi-Audio's `test_audios/`](https://github.com/MoonshotAI/Kimi-Audio/tree/master/test_audios).

```bash
cd examples/offline_inference/kimi_audio

# Audio2text over upstream asr_example.wav (Chinese transcription)
python end2end.py --task audio2text

# Audio2text over your own clip
python end2end.py --task audio2text \
    --audio-path /path/to/clip.wav \
    --question "Summarize what the speaker is saying."

# Spoken QA (audio2audio) over upstream qa_example.wav
python end2end.py --task audio2audio

# Multi-turn audio2audio (3 upstream multiturn URLs)
python end2end.py --task multiturn

# Text in, audio out (TTS-style)
python end2end.py --task text2audio
```

Outputs land under the per-task default directory (`./output_<task>`)
or under `--output-dir` if provided.

## Deploy config

`vllm_omni/deploy/kimi_audio.yaml` is a 2-stage
audio-out pipeline. Stage 0 is the fused thinker (Whisper-large-v3 +
VQ-Adaptor + Qwen2-7B + 6-layer MIMO branch). Stage 1 is `code2wav`
(PrefixStreamingFlowMatchingDetokenizer + BigVGAN). The detokenizer is
vendored in-tree under `vllm_omni/model_executor/models/kimi_audio/`
(see `detokenizer.py`, `flow_matching.py`, `modeling_bigvgan.py`) and
uses vLLM's bundled `vllm.vllm_flash_attn` for attention — no separate
`flash-attn` install is needed.

Defaults to single-GPU sync (both stages on `cuda:0`). To enable
multi-GPU async-chunk streaming for sub-second TTFB, edit the YAML per
the comments at its top: set `async_chunk: true`, move stage 1 to
`devices: "1"`, and add a `SharedMemoryConnector` block. To save
~4 GB on `audio2text` runs, override
`hf_overrides.kimia_generate_audio: false` on stage 0.

## Audio sampling caveat

The MIMO audio head samples via greedy argmax inside the model and does
not go through vLLM's sampler (vLLM expects one distribution per step,
and text+audio heads run in parallel). So the `default_sampling_params`
in the deploy YAML — `temperature`, `top_p`, `top_k` — only affect the
text path. Audio tokens are deterministic for a given text prefix.
Changing generation quality of the audio requires editing
`_run_mimo_branch` in `kimi_audio_thinker.py`.
