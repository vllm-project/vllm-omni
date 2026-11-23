# MiMo-Audio for omni audio generation and understanding

> Offline + online TTS, ASR, audio understanding, and multi-turn dialogue

## Summary

- Vendor: Xiaomi MiMo
- Model: `XiaomiMiMo/MiMo-Audio-7B-Instruct`
- Task: Text-to-speech, speech-to-text, audio understanding, spoken dialogue
- Mode: Offline end-to-end scripts and online OpenAI-compatible chat serving
- Maintainer: Community

## When to use this recipe

Use this recipe when you want a known-good starting point for running
MiMo-Audio on a single **32 GB Blackwell consumer GPU** (RTX 5090 / 5090D).
The bundled deploy profile [`vllm_omni/deploy/mimo_audio_5090d.yaml`](../../vllm_omni/deploy/mimo_audio_5090d.yaml)
splits VRAM between the AR thinker (stage 0) and code2wav (stage 1), avoids
FlashAttention-2 on SM 12.x, and keeps the audio tokenizer on CPU to reduce
Stage-1 GPU pressure.

## References

- Upstream model card: <https://huggingface.co/XiaomiMiMo/MiMo-Audio-7B-Instruct>
- Audio tokenizer: <https://huggingface.co/XiaomiMiMo/MiMo-Audio-Tokenizer>
- Offline example:
  [`examples/offline_inference/mimo_audio/end2end.py`](../../examples/offline_inference/mimo_audio/end2end.py)
- Online example:
  [`examples/online_serving/mimo_audio/README.md`](../../examples/online_serving/mimo_audio/README.md)
- Default deploy config (generic single GPU):
  [`vllm_omni/deploy/mimo_audio.yaml`](../../vllm_omni/deploy/mimo_audio.yaml)
- 5090D-tuned deploy config:
  [`vllm_omni/deploy/mimo_audio_5090d.yaml`](../../vllm_omni/deploy/mimo_audio_5090d.yaml)
- Pipeline / deploy docs:
  [Stage configs](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/stage_configs/)
- Related issue or discussion:
  [RFC: add recipes folder](https://github.com/vllm-project/vllm-omni/issues/2645)
- Tokenizer device alignment fix (required for CPU tokenizer on stage 1):
  [PR #6539](https://github.com/vllm-project/vllm-omni/pull/6539)

## Hardware Support

## GPU

### 1x NVIDIA GeForce RTX 5090 / 5090D 32GB

Two-stage async-chunk pipeline (thinker + code2wav) on a single GPU. Tested
offline with `tts_sft`.

#### Environment

- OS: Ubuntu (Linux x86_64)
- Python: 3.12
- PyTorch: 2.11.0+cu129
- Driver / runtime: NVIDIA driver 570.x, CUDA 12.9 runtime (Blackwell SM 12.x)
- GPU: NVIDIA GeForce RTX 5090 D, 32 GB
- vLLM: 0.26.0+cu129
- vLLM-Omni: `main` (tested while [#6539](https://github.com/vllm-project/vllm-omni/pull/6539) was open for CPU-tokenizer support)

Install vLLM-Omni from source and align the vLLM wheel with your CUDA stack.
On Blackwell, prefer the **cu129** vLLM build over cu13 defaults when using
PyTorch 2.11+cu129.

Download or cache:

- `XiaomiMiMo/MiMo-Audio-7B-Instruct`
- `XiaomiMiMo/MiMo-Audio-Tokenizer` (or a local checkout of the tokenizer repo)

Export the runtime knobs below before offline or online runs:

```bash
export MIMO_AUDIO_TOKENIZER_PATH="${MIMO_AUDIO_TOKENIZER_PATH:-XiaomiMiMo/MiMo-Audio-Tokenizer}"
export MIMO_AUDIO_TOKENIZER_DEVICE=cpu
export MIMO_AUDIO_TOKENIZER_CUDA_GRAPH=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Optional (mirror / download stability):

```bash
export HF_HUB_DISABLE_XET=1
export HF_ENDPOINT=https://hf-mirror.com   # if you use a Hugging Face mirror
```

Do **not** set `VLLM_ATTENTION_BACKEND` on vLLM 0.26.x; configure
`attention_backend: TRITON_ATTN` in the deploy YAML instead.

#### Command

Offline TTS validation from the repository root:

```bash
cd examples/offline_inference/mimo_audio

python3 -u end2end.py \
  --deploy-config vllm_omni/deploy/mimo_audio_5090d.yaml \
  --model-name XiaomiMiMo/MiMo-Audio-7B-Instruct \
  --query-type tts_sft \
  --text "The weather is so nice today."
```

Online serving (requires MiMo chat template):

```bash
export MIMO_AUDIO_TOKENIZER_PATH="XiaomiMiMo/MiMo-Audio-Tokenizer"

vllm serve XiaomiMiMo/MiMo-Audio-7B-Instruct --omni \
  --deploy-config vllm_omni/deploy/mimo_audio_5090d.yaml \
  --served-model-name "MiMo-Audio-7B-Instruct" \
  --port 18091 \
  --chat-template ./examples/online_serving/mimo_audio/chat_template.jinja
```

Pass the same environment exports to the server process when using a CPU
tokenizer or disabling FlashInfer sampling.

#### Verification

Offline:

```bash
ls -lh output_audio/tts_sft/*.wav
```

Expect a non-empty `.wav` under `output_audio/tts_sft/` and a companion `.txt`
with the decoded text. A successful run completes both stages without a
Stage-1 `decode_vq` device mismatch.

Online:

```bash
curl -s -o /tmp/mimo_health.txt -w '%{http_code}' http://127.0.0.1:18091/health
```

Then use the client under
[`examples/online_serving/mimo_audio/`](../../examples/online_serving/mimo_audio/).

#### Notes

- Memory usage (observed offline): Stage-0 weights ~16.6 GiB; KV cache ~4.9 GiB
  with `gpu_memory_utilization: 0.78` on stage 0 and `0.12` on stage 1.
- Key flags:
  - `--deploy-config vllm_omni/deploy/mimo_audio_5090d.yaml` for 5090D memory
    and `TRITON_ATTN`.
  - `MIMO_AUDIO_TOKENIZER_PATH` is mandatory.
  - `MIMO_AUDIO_TOKENIZER_DEVICE=cpu` saves VRAM; requires
    [#6539](https://github.com/vllm-project/vllm-omni/pull/6539) (or equivalent
    fix) so code2wav decode uses the tokenizer device.
  - `VLLM_USE_FLASHINFER_SAMPLER=0` avoids FlashInfer top-p/top-k sampler
    issues on consumer GPUs.
- Known limitations:
  - Blackwell may log `SM 12.x requires CUDA >= 12.9` from some optional
    tooling; the tested path uses TRITON attention and cu129 wheels.
  - Without a working `flash-attn` build, audio quality may be slightly
    metallic; install FlashAttention when a compatible wheel is available.
  - First request triggers Triton JIT for rotary / attention kernels (latency
    spike); subsequent requests are faster.
  - For longer outputs or batching, you may need to raise `max_model_len` in
    the deploy YAML and match `max_position_embeddings` in the model config
    (see the offline example README).
