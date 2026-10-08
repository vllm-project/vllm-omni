# ZONOS2 P5 acceptance tests

These tests use the production talker/sampler and DAC. The observers in
`trace.py` record lifecycle, conditioning, codes and decoder outputs without
replacing logits, weights or sampling. Only the GPU core test constructs a
small random DAC checkpoint; it checks codec contracts rather than speech quality.

## Local assets and a single GPU

Provision the converted frozen safetensors model and DAC/speaker assets locally:

```bash
export CUDA_VISIBLE_DEVICES=2  # Check this physical GPU is idle before each run.
export ZONOS2_TEST_MODEL_PATH=/path/to/zonos2-safetensors
export VLLM_ZONOS2_DAC_PATH=/path/to/weights_44khz_8kbps_0.0.1.pth
export VLLM_ZONOS2_SPEAKER_PATH=/path/to/local-qwen3-speaker-snapshot
export VLLM_ZONOS2_TN_CACHE_DIR=/path/to/owned/tn-cache
uv pip install -r requirements/zonos2.txt --overrides requirements/zonos2-overrides.txt
```

`ZONOS2_TEST_MODEL_PATH` must contain `config.json` and the converted model
weights. Missing assets fail real-weight tests explicitly; they are not
downloaded at test time. Both stages map logical device 0 to the one visible
GPU. The fixture uses an operator-owned temporary HF module cache and forces
Hugging Face/Transformers offline mode. Reference audio comes from existing
vendored `tests/assets/qwen3_tts/clone_2.wav` and `tests/assets/glm_tts/jiayan_zh.wav`.
HTTP requests embed the reference as a data URI.

## Levels and commands

```bash
# CPU acceptance oracles: reject empty, silent, invalid or incomplete outputs.
CUDA_VISIBLE_DEVICES='' pytest tests/model_executor/models/zonos2/test_acceptance_checks.py -m cpu

# GPU core: tiny real DAC, boundaries, OLA, duplicates and cleanup; no model download.
pytest tests/model_executor/models/zonos2 -m 'core_model and cuda and cards_1'

# Advanced: real offline request and HTTP WAV/raw-PCM/SSE with reference audio.
pytest tests/e2e/zonos2 -m 'advanced_model and cuda and cards_1' --run-level advanced_model

# Full/nightly: mixed text lengths, languages, two speakers and four seeds.
pytest tests/e2e/zonos2 -m 'full_model and cuda and cards_1' --run-level full_model
```

GPU marks are attached with `hardware_test`. The core job targets L4/B200;
real-weight jobs target H100/B200 with one card. They can also be run manually
on a sufficiently large CUDA card with the same one-card visibility setting.
The dedicated core job installs codec extras; broader GPU suites skip that
optional test when DAC is absent. Real-weight jobs explicitly prepare the pinned original model, conversion,
DAC and speaker assets with `tools/prepare_zonos2_ci.py` before starting
offline tests. Source and tensor hashes are verified. Agents may set an
owned `ZONOS2_CI_ASSET_ROOT` and the three existing asset-path variables to
reuse verified local assets. A fresh preparation needs approximately 35GB
for the original and converted model, plus room for retained test evidence.
The preparation step may use the network; inference stays offline.

Buildkite GPU jobs require the repository's `ready` (core), `merge-test`
(advanced), or `nightly-test` (full) labels/authorized build trigger.
Changing a Draft PR to ready for review does not substitute for a CI label.
Contributors without label permissions need a maintainer to trigger it. Ready, merge and nightly pipelines add only ZONOS2
jobs, selected through model-specific source dependencies.

## Assertions and evidence

The offline test requires nonempty 44.1kHz float32 audio, finite samples,
reasonable duration, nonzero RMS, aligned EOS, exact `frames * 512` sample
count and both talker/DAC cleanup notifications. It also checks that its
own GPU workers have exited after subprocess shutdown.

The HTTP test exercises `/v1/audio/speech` through a real loopback server:
non-streaming WAV, raw PCM streaming and SSE. It verifies HTTP status,
multiple audio chunks, one terminal SSE event, sample count and lifecycle.
An unsupported emotion CFG request must return a client error. No reference
fixture is fetched from an external URL.

The concurrency test runs four fixed-budget greedy requests individually,
then together, followed by four concurrent default-sampling requests.
Text/language lengths, speaker embeddings and seeds differ across requests.
Default requests must finish by EOS rather than the token budget.
It matches each request's conditioning fingerprints and decoder input codes
to its own history, then independently reconstructs its exact DAC/OLA waveform
and compares it with the delivered audio. This detects routing mixups,
missing/duplicate chunks and output contamination. Both state maps must be
empty at the end.

Individual and batched greedy trajectories are diagnostic observations;
they are not required to be bit-identical across different numerical batch
paths. The first divergent step records equal prior histories and differing
adjusted logits. The waveform reconstruction oracle uses the actual codes
of each request, rather than equating a batch-dependent token difference
with cross-request audio routing.

By default evidence lives under pytest temporary directories. Set
`ZONOS2_TEST_OUTPUT_DIR` to an owned directory to retain logs, traces, codes,
WAV/NumPy audio and JSON summaries. Each pytest run uses a fresh output folder.
Subprocess cleanup signals only the process group created by that test.

These tests do not measure WER/CER, speaker similarity, UTMOS or performance;
those remain P6 work.

## Asset preparation

```bash
python tools/prepare_zonos2_ci.py --asset-root /owned/zonos2-ci --env-file /owned/zonos2-ci/asset.env
. /owned/zonos2-ci/asset.env
# To verify preprovisioned paths without any download:
python tools/prepare_zonos2_ci.py --offline --asset-root /owned/zonos2-ci --env-file /owned/zonos2-ci/asset.env
```
