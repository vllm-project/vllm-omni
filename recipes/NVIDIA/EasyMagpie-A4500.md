# EasyMagpie TTS — RTX A4500

## Summary

- Vendor: NVIDIA.
- Model: a converted EasyMagpie/Nemotron-TTS checkpoint with architecture `EasyMagpieTTSForConditionalGeneration`.
- Task and mode: speaker-conditioned text-to-speech through `/v1/audio/speech`, including streaming PCM.
- Hardware: one NVIDIA RTX A4500, 20 GB, PCIe.
- Maintainer: vLLM-Omni community.

## Supported model contract

| Item | Contract |
| --- | --- |
| Input | Complete text in `input`; a bundled speaker name in `voice` |
| Conditioning | Optional `extra_params.context_text`; use the language/speaker context expected by the checkpoint |
| Output | Mono audio at 22,050 Hz; streaming PCM is signed 16-bit little-endian |
| Pipeline | Nemotron-H plus local codebook transformer, followed by a stateful spectral codec |
| Preset voices | Files under `speaker_embeddings/<voice>.pt`; default name `eng` |
| Reference audio | Requires conversion with both audio encoders bundled; not qualified by the preset-voice comparison |
| Streaming text / WebSocket | Not qualified by this integration |

| Profile | Capacity | Precision | Codec chunk sizes |
| --- | --- | --- | --- |
| [`easymagpie_a4500.yaml`](../../vllm_omni/deploy/easymagpie_a4500.yaml) | 32 requests per stage | FP16 talker and codec; FP32 SSM state | 2, then 6, then 8 stacked frames |
| [`easymagpie.yaml`](../../vllm_omni/deploy/easymagpie.yaml) | 32 requests per stage | FP16 talker; FP32 codec and SSM state | 6, then 6, then 8 stacked frames; configuration-only |

The optimized profile reserves 1 GiB for talker cache and 128 MiB for codec
state. These are cache budgets, not total GPU-memory usage. Both workers run
on device 0 with TP=PP=DP=1. Recompute preemption is unsupported; size the
cache and admission capacity for the workload.

## Checkpoint setup

The model directory must contain the converted `config.json`, safetensors
weights, text tokenizer assets, phoneme tokenizer, speaker embeddings, and
`codec_native/{config.json,model.safetensors}`. A NeMo archive alone is not a
serving checkpoint. Conversion and checkpoint distribution are external to
this integration; retain the converter's architecture and tokenizer metadata.

Perth watermarking follows the codec configuration and is enabled by default.
It requires `resemble-perth` with the `perth.perth_net` package and its pretrained
weights, either installed with Perth or bundled at the configured
`watermark_checkpoint` path. The reference implementation uses
`resemble-perth @ git+https://github.com/yhayarannvidia/Perth.git@c12bf20dbf37965f869207efc135547cea52c34d`.
To reproduce the comparison without watermarking, copy the deployment YAML
and add `watermark_enabled: false` to stage 1's `engine_extras.hf_overrides`.

## Software environment

The local comparison uses Ubuntu 22.04, Python 3.12, NVIDIA driver 580.126.20,
PyTorch 2.13.0+cu130, vLLM 0.32.0, and vLLM-Omni based on `6b0efc84f` plus this
integration. It uses one GPU with no competing server. Other accelerator
models need independent qualification.

## Serve and verify

Run from the repository root with the matching vLLM-Omni environment installed:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0 \
vllm serve /path/to/converted-model --omni \
    --served-model-name easymagpie \
    --deploy-config vllm_omni/deploy/easymagpie_a4500.yaml \
    --trust-remote-code --port 8091
```

`VLLM_USE_FLASHINFER_SAMPLER=0` avoids requiring FlashInfer's sampler JIT toolchain;
EasyMagpie samples acoustic codes in its own local transformer. The generic
sampler only handles the two-symbol continue/stop vocabulary.

Use a speaker actually bundled in the checkpoint and its matching context:

```bash
curl --fail-with-body http://localhost:8091/v1/audio/speech \
    -H 'Content-Type: application/json' \
    -d '{"model":"easymagpie","input":"Welcome to the demonstration.","voice":"eng","response_format":"pcm","stream":true,"stream_format":"audio","max_new_tokens":1024,"extra_params":{"context_text":"[EN]"}}' \
    --output speech.pcm
ffplay -f s16le -ar 22050 -ac 1 -autoexit speech.pcm
```

Expect nonempty PCM received in successive chunks, ending at the model's stop
condition or generation limit. Set `stream` to `false` and `response_format`
to `wav` for a complete WAV response. The captured local-transformer sampling path retains the reference
implementation's defaults (temperature 0.7 and top-k 80); request overrides and
`seed` do not guarantee greedy or bitwise-reproducible acoustic codes. Use the
same settings in both deployments when comparing performance.

## Qualification scope

C16/C32 comparisons use the same converted checkpoint, preset voice, context,
eight English prompts, generation limit, and chunk schedule in both servers.
Each run warms 64 requests before measuring 128 requests. Measure aggregate
RTFx as generated audio seconds divided by wall time, and TTFA from HTTP request
to first PCM bytes. Record failures and playback underruns as well as latency.
Warmup, profiler runs, and server startup are excluded from performance claims.
These runs do not qualify voice cloning, other languages, H100 C256, or perceptual
quality across the model's full distribution.

## Regression tests

Install the repository's test requirements in the matching vLLM-Omni environment.
CPU coverage is collected by the existing L1 suite:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q -o addopts='' \
    tests/model_executor/models/easymagpie -m 'core_model and cpu' \
    --run-level=core_model
```

The opt-in serving tests need a CUDA GPU with at least 20 GB and the converted
checkpoint. The test uses an L4 resource marker for CI routing; it also runs
on an A4500. Set `EASYMAGPIE_MODEL`, `EASYMAGPIE_VOICE`,
`EASYMAGPIE_CONTEXT_TEXT`, and optionally `EASYMAGPIE_DEPLOY_CONFIG`, then run:

```bash
python -m pytest -q -o addopts='' tests/e2e/online_serving/test_easymagpie.py \
    -m 'core_model and cuda' --run-level=core_model
```

The tests cover complete WAV responses and four simultaneous PCM streams.
They skip without `EASYMAGPIE_MODEL`; no private checkpoint path is embedded
in CI. A CI-owned checkpoint is needed before enabling a dedicated GPU job.
Local offline validation also checks that `Omni.generate` returns the complete
22,050 Hz waveform rather than only the final codec chunk.

## References

- [Model implementation](../../vllm_omni/model_executor/models/easymagpie/easymagpie.py)
- [Speech API](../../docs/serving/speech_api.md)
- [Shared online TTS clients](../../examples/online_serving/text_to_speech/README.md)
- [Supported models](../../docs/models/supported_models.md)
- [H100 configuration](EasyMagpie-H100.md)
