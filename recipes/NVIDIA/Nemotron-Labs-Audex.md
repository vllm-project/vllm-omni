# Nemotron-Labs Audex (2B / 30B-A3B)

> Speech suite on 1×H100: TTS, text-to-audio, audio understanding, and cascaded speech-to-speech

## Summary

- Vendor: NVIDIA
- Model: `nvidia/Nemotron-Labs-Audex-2B`, `nvidia/Nemotron-Labs-Audex-30B-A3B`
- Task: Text-to-speech (English), text-to-audio (sound effects), audio understanding (ASR / audio QA), cascaded speech-to-speech
- Mode: Online serving (`/v1/audio/speech`, `/v1/chat/completions`) and offline batch inference
- Maintainer: Community

## When to use this recipe

Use this recipe to serve either Audex checkpoint through one of its four
deployment pipelines. Both sizes share the same repo layout, codec token
space, and streaming causal speech decoder (16 kHz output); they differ in
the thinker: the 2B is a dense LM, the 30B-A3B is a hybrid Mamba + MoE
NemotronH (~3B active parameters). Pick the pipeline by task:

| pipeline (`vllm_omni/deploy/<name>[_30b].yaml`) | audio in | text out | speech out | general audio out | endpoint |
|---|---|---|---|---|---|
| `audex_tts` | ❌ | ❌ | ✅ | ❌ | `/v1/audio/speech` |
| `audex_tta` | ❌ | ❌ | ❌ | ✅ | `/v1/audio/speech` |
| `audex_thinker_only` | ✅ | ✅ | ❌ | ❌ | `/v1/chat/completions` |
| `audex_s2s` | ✅ | ✅ | ✅ | ❌ | both |

## References

- Model cards: [Nemotron-Labs-Audex-2B](https://huggingface.co/nvidia/Nemotron-Labs-Audex-2B), [Nemotron-Labs-Audex-30B-A3B](https://huggingface.co/nvidia/Nemotron-Labs-Audex-30B-A3B)
- Offline examples: [`examples/offline_inference/audex/`](../../examples/offline_inference/audex/)
- Online examples: [`examples/online_serving/audex/`](../../examples/online_serving/audex/)

## Hardware Support

## GPU

### 1×H100 80GB — Audex-2B

#### Environment

- OS: Linux
- Python: 3.12+
- CUDA: 12.x
- vLLM version: 0.24.0
- vLLM-Omni version or commit: PR #4976 (branch `audex`)

#### Command

**Online serving** (TTS is the default pipeline; the repo root's
`model_type` auto-resolves `audex_tts.yaml`):

```bash
vllm-omni serve nvidia/Nemotron-Labs-Audex-2B \
    --host 0.0.0.0 --port 8097 \
    --trust-remote-code --omni
# other modes: add --stage-configs-path vllm_omni/deploy/audex_{tta,thinker_only,s2s}.yaml
# or use the launcher: MODE=s2s PORT=8098 examples/online_serving/audex/run_server.sh
```

**Offline batch inference** (one script per task, each defaults to its
correct deploy yaml):

```bash
python examples/offline_inference/audex/text_to_speech.py \
    --texts "Hello world." --cfg-scale 1.5 --output-dir results/audex_wavs
python examples/offline_inference/audex/text_to_audio.py     # sound effects
python examples/offline_inference/audex/audio_qa.py          # ASR / audio QA
python examples/offline_inference/audex/speech_to_speech.py  # 3-pass cascade
```

#### Verification

```bash
curl -X POST http://localhost:8097/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "model": "nvidia/Nemotron-Labs-Audex-2B",
        "input": "Hello, how are you?",
        "response_format": "wav",
        "extra_params": {"cfg_scale": 1.5}
    }' --output hello.wav
# or the universal client (all four modes):
python examples/online_serving/audex/client.py --mode tts --output hello.wav
```

#### Notes

- Memory usage: TTS deploy — stage 0 (thinker) 0.4, stage 1 (decoder) 0.25
  GPU-memory fraction; both stages share GPU 0.
- Key flags: `--trust-remote-code` and `--omni` are required.
- Classifier-free guidance: `extra_params.cfg_scale` (1.0 = off; 1.5 is the
  official quality setting; TTA effectively requires its default 3.0).
  Measured en-24 self-transcribed CER: 6.87% guided vs 7.24% unguided.
- Output: 16 kHz mono WAV. English, single built-in voice (no cloning).
- TTA decodes through the external XCodec1 checkpoint
  (`hf-audio/xcodec-hubert-general-balanced`, auto-downloaded; override
  with `XCODEC1_PATH`).

### 1×H100 80GB — Audex-30B-A3B

#### Environment

Same as the 2B row above.

#### Command

The 30B REQUIRES its explicit deploy yaml (the shared HF `model_type`
auto-resolves to the 2B-tuned config otherwise):

```bash
vllm-omni serve nvidia/Nemotron-Labs-Audex-30B-A3B \
    --host 0.0.0.0 --port 8097 \
    --trust-remote-code --omni \
    --stage-configs-path vllm_omni/deploy/audex_tts_30b.yaml
# or: SIZE=30b MODE=tts examples/online_serving/audex/run_server.sh
```

Offline: same four scripts with
`--model nvidia/Nemotron-Labs-Audex-30B-A3B --deploy-config vllm_omni/deploy/audex_<mode>_30b.yaml`.

#### Verification

Same curl / `client.py` as the 2B with
`"model": "nvidia/Nemotron-Labs-Audex-30B-A3B"` (the server validates the
model id). First launch downloads ~60 GB.

#### Notes

- Memory usage: ~61 GiB weights on one H100 80 GB (stage 0 at 0.85 GPU
  fraction + decoder at 0.08 on the same card); verified with healthy KV
  headroom. If long sequences OOM, set `tensor_parallel_size: 2` on
  stage 0 (documented in the yaml headers).
- `enable_prefix_caching: false` on stage 0 — hybrid Mamba constraint.
- CUDA graphs stay ON for the thinker: the NemotronH stage captures
  cleanly and eager mode costs ~7× single-stream TTS RTF (review-measured
  0.30 vs 2.08).
- Quality (en-24 self-transcribed CER): 0.72% unguided baseline; 0.72% at
  cfg 1.5 with guided temperature 0.1.
- Known limitation: the S2S chat pass needs the official
  formatting-instruction prompt (the bundled examples/client send it);
  a bare transcript turn makes the 30B answer in speech-codec tokens.
