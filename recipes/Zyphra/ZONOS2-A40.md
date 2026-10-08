# ZONOS2 speech synthesis — NVIDIA A40

## Summary and model contract

Zyphra `Zyphra/ZONOS2` is an 8B-A0.9B MoE TTS model. This native integration
uses an AR talker followed by DAC. Maintainer:
[Xiaotangyuan-xty](https://github.com/Xiaotangyuan-xty).

| Input/control | Contract |
| --- | --- |
| Text | Nonempty; NeMo TN followed by UTF-8 bytes |
| Language | English, Chinese, French, German, Spanish, Italian, Portuguese, Japanese, Korean; explicit aliases; no automatic detection |
| Voice | `default`, omitted, one `ref_audio`, or finite 2048D `speaker_embedding` |
| Reference | MediaConnector data URI/URL/permitted file; one reference; no transcript |
| Speed/quality | Native rate buckets and six quality features; [API controls](../../docs/serving/speech_api.md#zonos2) |
| Output | Mono 44100Hz PCM; float32 internally, 512 samples per frame |
| Streaming | Raw PCM/WAV or OpenAI speech SSE; min16 frames, overlap4 OLA |
| Budget | Default/maximum 1024 frames; budget termination is an incomplete-generation error |
| Unsupported | CFG except 1, style instructions, prefix cache, async AR scheduling, AR graph, quantization/distributed profiles |

## Hardware and profiles

One NVIDIA A40 48GB, PCIe, approximately 503GB host RAM. CPU staging is used
for reference speaker extraction and output PCM. Other accelerators and
multi-GPU topologies are not qualified by this recipe.

| Profile | Status |
| --- | --- |
| One visible GPU, TP1/PP1, BF16 eager talker, FP32 DAC, B1, async codec | Recommended; offline/HTTP/streaming and fixed-set quality qualified |
| Explicit B4/B8 overrides | Experimental: routing tested, but batching changes logits; fixed-set C4 `zh_01` hit the frame cap |

Incoming concurrency may queue behind the B1 default. Codec chunk transport
is independent of unsupported async AR scheduling.

## References and software

- [Model](https://huggingface.co/Zyphra/ZONOS2), snapshot
  `65f1e80f94b599d474bb6af9094a803dc52f60bd`.
- [Official code](https://github.com/Zyphra/ZONOS2), commit
  `194c0a3ab67b90383a67646289f28d4ecb1c1f64`.
- [Tracking issue](https://github.com/vllm-project/vllm-omni/issues/7862).
- [Offline hub](../../examples/offline_inference/text_to_speech/README.md),
  [online hub](../../examples/online_serving/text_to_speech/README.md),
  [supported models](../../docs/models/supported_models.md).

| Item | Environment |
| --- | --- |
| OS/Python | Ubuntu 24.04 / Python 3.12.13 |
| Driver | 580.95.05 |
| Rebased runtime | vLLM 0.30.0, Torch 2.13.0+cu130, FlashInfer 0.6.18.post1 |
| Historical measurements | Native vLLM 0.29.0/Torch 2.13.0+cu130; official Torch 2.9.1+cu128 |
| Rebase target | `527982d886db4da88f205a576cbc0333f9b8eac5` |

Historical timings below predate rebase. Post-rebase validation is a bounded
regression run, not a replacement controlled performance matrix.

## Checkpoint and dependencies

The original checkpoint is `model.pth` plus `params.json`. Convert the trusted
frozen source using `tools/convert_zonos2_to_safetensors.py --help`; serve the
converted safetensors output. Its manifest records 507 tensors/source hashes.
Only the explicit conversion tool deserializes the original pickle checkpoint.

```bash
uv pip install -r requirements/zonos2.txt --overrides requirements/zonos2-overrides.txt
export VLLM_ZONOS2_DAC_PATH=/owned/assets/weights_44khz_8kbps_0.0.1.pth
export VLLM_ZONOS2_SPEAKER_PATH=/owned/assets/qwen3-voice-embedding-snapshot
export VLLM_ZONOS2_TN_CACHE_DIR=/owned/cache/zonos2-tn
export HF_MODULES_CACHE=/owned/cache/hf-modules
```

Use torchaudio matching torch. The speaker snapshot includes its trusted local
configuration/modeling Python files. Speaker extraction is lazy, CPU-only and
local-only. DAC uses restricted tensor loading, not the package's general
module/package loader. Reference formats require the matching ffmpeg/audio
backend. Assets are provisioned before serving; runtime does not download them.

The model-specific protobuf override preserves 6.33.6 instead of audiotools'
obsolete `<3.20` metadata restriction, which conflicts with Ray. See
[dependency notes](../../vllm_omni/model_executor/models/zonos2/README.md).

## Online commands

Check the selected physical GPU is idle. Both stages use logical device 0
inside the single-card visibility mapping:

```bash
export CUDA_VISIBLE_DEVICES=2
vllm serve /owned/zonos2-safetensors --omni \
  --deploy-config vllm_omni/deploy/zonos2.yaml \
  --served-model-name zonos2 --host 127.0.0.1 --port 8091 --trust-remote-code

curl --fail-with-body http://127.0.0.1:8091/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"zonos2","input":"Hello, this is a speech test.","language":"English","seed":42,"response_format":"wav"}' \
  --output output.wav

curl --fail-with-body http://127.0.0.1:8091/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"zonos2","input":"Hello, this is a streaming test.","seed":42,"stream":true,"stream_format":"audio","response_format":"pcm"}' \
  --output output.pcm
```

Expect nonempty mono WAV at 44100Hz or PCM16 stream bytes. The generic curl
interface supplies online inference; no dedicated Python example is added.
See the online hub for data URI references. A frame-budget termination returns
an error; a stream may already have emitted partial audio before its terminal
error. Seeded/explicitly budgeted requests are not silently retried.

## Offline API and qualification

The public offline contract is `Zonos2Processor.build_prompt(...)` and
`AsyncOmni.generate(...)` with talker/decoder SamplingParams. The generic
diffusion text-to-audio CLI does not accept this AR frame contract. Use the
public API or the regression entrypoints; do not reuse another model's prompt
builder or add a new model-specific Python example.

```bash
export ZONOS2_TEST_MODEL_PATH=/owned/zonos2-safetensors
export ZONOS2_TEST_OUTPUT_DIR=/owned/qualification-output
pytest -sv tests/e2e/zonos2 -m 'advanced_model and cuda and cards_1' --run-level advanced_model
pytest -sv tests/e2e/zonos2 -m 'full_model and cuda and cards_1' --run-level full_model
pytest -q tests/model_executor/models/zonos2 -m cpu
```

Real-weight tests fail without local assets. Vendored references use data URIs;
CI never downloads external audio fixtures. `hardware_test` CI H100/B200 marks
do not extend the hardware qualification of this A40 recipe.

## Evidence and limitations

Frozen pre-rebase gates: L0 consumed 507/507 tensors; L1 covered 4520 frames/
40680 predictions with 100% top-1 agreement; neutral greedy L2 covered 8346
frames/75114 codes with 100% agreement. Default RNG is request-local and does
not promise the official RNG bitstream or cross-batch bit identity.

The fixed 10-case pre-rebase A40 B1 study measured EN WER 2.74%, ZH CER 0,
native cosine proxy .97265 and UTMOS 3.3647. Streaming first-PCM P50 was 1.03s
versus sync 15.34s; RTF P50 was 3.70, so this is not a real-time performance
claim. Peak sampled allocation was approximately 34GiB including KV reserve.

The B4 Chinese cap failure is retained as a batch quality limitation. B1 and
the explicit budget error protect recommended serving behavior. DAC compile
failed the numerical gate; DAC graph passed only a component experiment.
Full AR graph remains disabled. Reproduce measurements with
[benchmarks/zonos2](../../benchmarks/zonos2/README.md).

## Optional router replay and CI preparation

Set `VLLM_ZONOS2_ROUTER_REPLAY=1` before the existing serve/benchmark
command to opt into the qualified BF16 single-row router component.
`enforce_eager` stays true; this does not capture full AR execution or
request state. Default is off. See the [model contract](../../vllm_omni/model_executor/models/zonos2/README.md#optional-single-row-router-replay).

Real-weight CI explicitly prepares pinned assets before offline testing:

```bash
python tools/prepare_zonos2_ci.py --asset-root /owned/zonos2-ci --env-file /owned/zonos2-ci/asset.env
. /owned/zonos2-ci/asset.env
```

An owned reusable cache/preprovisioned paths are supported. Fresh original
plus converted weights need approximately 35GB and additional result space.
Buildkite requires an authorized trigger or the repository CI labels;
local A40 runs do not establish a hosted H100/B200 pass.

Router replay A/B on one A40 GPU1 used the same ten frozen inputs, two
warmups and three measured rounds per variant (30 requests each).
E2E P50: 15.584s eager / 13.731s replay; RTF P50: 3.710 / 3.324;
first PCM P50: 1.027s / .933s. Sampled peak: 28817 / 29369MiB.
All 30 paired code histories and waveforms were bit-identical. Round-zero
quality was identical: EN WER 2.74%, ZH CER 0, native cosine .97265,
UTMOS 3.3647, no failed samples. These small-corpus A40 results do not
qualify other hardware or realtime performance. Default replay remains off.

The warmed 32-forward trace retained 40685 GPU kernels, while CPU
`cudaLaunchKernel` calls fell from 35347 to 25737 with 744 graph launches.
Replay reduces host dispatch; it does not replace attention or sampling.
