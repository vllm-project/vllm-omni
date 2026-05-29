# Kimi-Audio

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/kimi_audio>.

For a hardware-tagged step-by-step guide, see the
[Kimi-Audio recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/moonshotai/Kimi-Audio-7B-Instruct.md).

## Setup

The pipeline runs as two stages: a fused thinker (Whisper-large-v3 +
VQ-Adaptor + Qwen2-7B + 6-layer MIMO branch) and a code2wav stage
(flow-matching DiT + BigVGAN). See the
[stage configuration documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/stage_configs/)
to size GPU memory for your setup.

## Run examples

```bash
cd examples/offline_inference/kimi_audio
```

`audio2text` (audio in, text out):

```bash
python end2end.py --task audio2text
```

`audio2audio` (audio in, text + spoken audio out):

```bash
python end2end.py --task audio2audio
```

Multi-turn `audio2audio` (q1 audio → a1 audio + text → q2 audio):

```bash
python end2end.py --task multiturn
```

`text2audio` (text in, audio out — TTS-style):

```bash
python end2end.py --task text2audio
```

All four tasks share the single-GPU sync `kimi_audio.yaml`. To enable
multi-GPU async-chunk streaming for sub-second TTFB, edit the YAML per
the comments at its top.

Outputs land in `output_<task>/` (one `.txt` transcript and one `.wav`
per request, where applicable).
