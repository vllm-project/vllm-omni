# ZONOS3-Mini Speech API — RTX 3090

## Summary

- Vendor/model: Zyphra, [`Zyphra/ZONOS3-Mini`](https://huggingface.co/Zyphra/ZONOS3-Mini).
- Task: text-to-speech through `POST /v1/audio/speech`.
- Runtime: vLLM-Omni MRv2, single-stage AR + stateful packed ZAC decoder (`keep2_plain`).
- Hardware: one NVIDIA GeForce RTX 3090, 24 GiB; one worker on GPU 0.
- Maintainer: vLLM-Omni contributors.

## Supported model contract

| Item | This recipe |
| --- | --- |
| Input | Non-empty text; omit `voice` for unconditioned synthesis |
| Output | Mono WAV at 44,100 Hz |
| Provided profile | `vllm_omni/deploy/zonos3_mini.yaml`, 32 request slots, AR context 2048 tokens |
| Precision | AR BF16; speaker encoder and Codec weights FP32 |
| Execution | Piecewise AR prefill/mixed, FULL AR decode, model-owned Codec graphs |
| Qualification | Historical two-stage measurements below; single-stage validation is reported separately |
| Voice cloning / other topologies | Not qualified by this recipe |

For generic API fields, see the [Speech API](../../docs/serving/speech_api.md).
Model-specific requests should follow this recipe rather than supplying a preset
`voice`: Zonos3 has no built-in preset voices in this integration.

## Setup

Run the following commands from the vLLM-Omni repository root in the CUDA
environment described by the [installation guide](../../docs/getting_started/installation/README.md).
This checkout targets vLLM **0.33.0**.

```bash
pip install -e '.[zonos3]'
export FLA_TILELANG=0
```

The optional dependency installs `fla-core==0.5.2` and
`zyphra-textnorm==0.1.0`. `FLA_TILELANG=0` selects the GDN Triton path used by
this integration.

## Download only the required files

Keep your existing `HF_HOME` / `HF_HUB_CACHE` values, or choose them before
downloading. Do not use `--local-dir` when you want the standard HF cache layout.
The pinned release below is the revision used for qualification.

```bash
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export ZONOS3_MODEL=Zyphra/ZONOS3-Mini
export ZONOS3_REVISION=f68e6e470fd14d6d1bc50e8461ee4d4f64d6c513

hf download "$ZONOS3_MODEL" \
  lm/zonos3_mini/params.json \
  lm/zonos3_mini/model.pth \
  lm/zonos3_mini/tokenizer/superbpe8192.json \
  autoencoder/generator.pt \
  autoencoder/heads/keep2_plain/config_full_unfreezeall_mr_drop30_4x10.json \
  autoencoder/heads/keep2_plain/step_0050000.pt \
  speaker_encoders/ownmect_s2/ckpt_0037.pth \
  --revision "$ZONOS3_REVISION" \
  --cache-dir "$HF_HUB_CACHE"

export ZONOS3_MODEL_PATH="$HF_HUB_CACHE/models--Zyphra--ZONOS3-Mini/snapshots/$ZONOS3_REVISION"
```

These seven files cover AR weights/config/tokenizer, the default Codec generator
and head, and the speaker encoder. The current loader also loads the speaker
encoder for unconditioned requests. GGUF, Web, teacher and `full_mr` files are
not needed for this profile. The complete `generator.pt` is required on disk
even though only the selected decoder network is retained on the GPU.

Use the **snapshot root** as `ZONOS3_MODEL_PATH`, not its `lm/zonos3_mini`
subdirectory: the loader resolves `autoencoder/` and `speaker_encoders/` relative
to this root. Keep the snapshot's symlinks and corresponding cache blobs intact.

## Start the server

After downloading, this command uses the pinned local snapshot and disables
further HF downloads for the server:

```bash
HF_HUB_OFFLINE=1 FLA_TILELANG=0 \
vllm serve "$ZONOS3_MODEL_PATH" \
  --served-model-name Zyphra/ZONOS3-Mini \
  --omni \
  --deploy-config vllm_omni/deploy/zonos3_mini.yaml \
  --port 8000
```

The deploy reserves **128 MiB** for AR KV. GDN and Codec persistent state
are separately preallocated. One native scheduler owns both AR and Codec slots;
sampled delayed RVQ stays on the GPU, without an inter-stage connector.

AR decode uses native FULL graphs and mixed/prefill uses PIECEWISE graphs.
After sampling, the Codec stream updates the RVQ ring and tests chunk readiness
and logical EOS for the whole batch. Device conditionals skip the complete
packed Codec when nothing is ready and select a token-capacity bucket on GPU. Ready requests share one packed forward;
inactive lanes use zero-length scratch slots. Remaining frames and synthesis
tail are flushed once at logical EOS or the request length limit.

The Codec graph has fixed capacity for each request-count bucket. Actual lengths,
slots and final flags stay on device. PCM is copied into an owned step snapshot
before replay can overwrite graph storage. The native output materializer waits
for Codec completion, then copies only valid PCM before delivering logical tokens. Cancellation invalidates the original output lease,
and slot reuse resets Codec state on its serial stream.

The default per-request capacity is 31 RVQ frames, including silence priming on
the first packet. Custom chunk settings must fit it; increase
`hf_overrides.zonos3_codec_max_chunk` at startup for larger packets. This also
increases graph workspace and padding cost.

## Decoder variants and launch adjustments

| Decoder | Additional files under the snapshot root | Qualification |
| --- | --- | --- |
| `keep2_plain` | The two `autoencoder/heads/keep2_plain/` files listed above | Default; single-GPU online smoke passed |
| `full_mr` | `autoencoder/heads/full_mr/config_full_unfreezeall_mr.json` and `autoencoder/heads/full_mr/step_0050000.pt` | Loader/config support; this recipe has not GPU-qualified this variant |

### Select the full-size `full_mr` decoder

After the base download above, fetch only the alternate decoder config and
checkpoint. Both variants share the LM, tokenizer, generator and speaker encoder;
those files do not need another download.

```bash
hf download "$ZONOS3_MODEL" \
  autoencoder/heads/full_mr/config_full_unfreezeall_mr.json \
  autoencoder/heads/full_mr/step_0050000.pt \
  --revision "$ZONOS3_REVISION" \
  --cache-dir "$HF_HUB_CACHE"

HF_HUB_OFFLINE=1 FLA_TILELANG=0 \
vllm serve "$ZONOS3_MODEL_PATH" \
  --served-model-name Zyphra/ZONOS3-Mini \
  --omni \
  --deploy-config vllm_omni/deploy/zonos3_mini.yaml \
  --stage-overrides '{"0":{"hf_overrides":{"zonos3_decoder":"full_mr"}}}' \
  --port 8000
```

Selection happens at startup, not in the Speech API request. Returning to
`keep2_plain` requires removing the override or setting it to `keep2_plain` and
restarting. The larger network may need more device memory and capture time;
the measured memory figures below apply only to `keep2_plain`.

### Enforce eager execution

Set Stage 0 `enforce_eager: true` to disable both AR and Codec graphs. Codec
uses the same packed core, with zero-length scratch lanes for requests that are
not ready. This avoids a host readiness check but still executes padding work;
it does not promise that dummy lanes cost no compute.

```bash
HF_HUB_OFFLINE=1 FLA_TILELANG=0 \
vllm serve "$ZONOS3_MODEL_PATH" \
  --served-model-name Zyphra/ZONOS3-Mini \
  --omni \
  --deploy-config vllm_omni/deploy/zonos3_mini.yaml \
  --stage-overrides '{"0":{"enforce_eager":true}}' \
  --port 8000
```

## Generate speech

In another terminal:

```bash
curl --fail-with-body http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Zyphra/ZONOS3-Mini",
    "input": "The weather is nice today, perfect for a walk in the park.",
    "seed": 42,
    "response_format": "wav"
  }' \
  --output zonos3.wav
```

Expected result: a non-empty, decodable 44.1 kHz mono WAV containing the requested
sentence. Do not pass `sample_rate` to this model; its native output is 44.1 kHz.

The default `first_chunk_codebooks=1` sends the first Codec chunk as soon as
CB0 arrives, omitting unavailable residual codebooks. Only the first chunk uses
this partial policy; subsequent chunks wait for all codebooks. Frames already
emitted are not replayed with later residuals. Set
`"extra_params": {"first_chunk_codebooks": 10}` to request a full-codebook first
chunk. This affects first-audio latency and can affect the initial audio.

The default `silence_prefill_frames=3` adds delayed silence rows to the AR prompt
and primes the Codec; the corresponding prefill samples are cropped from output.
This conditioning can also lengthen the silence generated before speech. To
disable it for a request, add the following field to the JSON body:

```json
"extra_params": {"silence_prefill_frames": 0}
```

This changes generation conditioning, rather than trimming audio afterward.
Keep the same `seed` when comparing the two settings.

## Verification

With the repository's test and ASR dependencies installed, the same online smoke
can be run as follows. Export `ZONOS3_MODEL_PATH` from the download section in
the test terminal as well.

```bash
FLA_TILELANG=0 python -m pytest -sv tests/e2e/online_serving/test_zonos3_mini.py \
  -m 'advanced_model and cuda' --run-level advanced_model
```

The test starts its own server and checks the synthesized text through ASR.
It uses real weights: the `core_model` harness's dummy-weight path is not
supported by the strict Zonos3 companion loaders.

## Performance and WER measurement

Start the server with the default YAML above. With Seed-TTS English data and
the benchmark's ASR dependencies installed, run the shared TTS benchmark:

```bash
export VLLM_OMNI_BENCH_AUDIO_SAMPLE_RATE=44100
export SEED_TTS_EVAL_DEVICE=cuda:0
export SEED_TTS_UTMOS_EVAL=0
export SEED_TTS_HF_WHISPER_MODEL=/path/to/whisper-large-v3

python benchmarks/tts/bench_tts.py \
  --host 127.0.0.1 --port 8000 \
  --model Zyphra/ZONOS3-Mini --task default_voice --locale en \
  --dataset-path /path/to/seedtts_testset \
  --num-prompts 128 --num-warmups 32 --request-seed 42 \
  --concurrency 1 4 8 16 32 --wer-eval \
  --output-dir results/zonos3-packed \
  -- --tokenizer /path/to/benchmark-tokenizer
```

The tokenizer is used for benchmark input-token accounting, not Zonos3 speech
generation. Keep the dataset, tokenizer and request seed fixed when comparing
runs. Report audio throughput, first-packet latency, RTF, request failures and
WER separately for each concurrency. Warmup requests are excluded from the
timed workload. ASR evaluation on the serving GPU is a separate phase; distinguish
its memory usage from the server's steady-state and request-processing peaks.

## Historical two-stage qualification evidence

| Environment | Validated value |
| --- | --- |
| Accelerator | 1 × RTX 3090, 24 GiB; no inter-GPU communication |
| OS / Python | Ubuntu 22.04.5 LTS / Python 3.12.3 |
| PyTorch / CUDA build | 2.14.0+cu130 / 13.0 |
| NVIDIA driver | 595.71.05 |
| vLLM / Omni source | 0.33.0 / uncommitted packed-only working tree, measured 2026-11-15 |
| Checkpoint | Revision pinned above, default `keep2_plain` decoder |
| Workload | Seed-TTS English default voice; seed 42; 32 warmups and 128 timed requests per concurrency |
| Quality evaluation | Whisper large-v3; mean per-request WER; 128 evaluated outputs per concurrency |
| Codec graph capture | 30 packed keys: B=1/2/4/8/16/32, sumT=B × {1,2,4,8,15}; captured before serving |
| Process GPU memory | After startup, before ASR loading: Stage 0 2684 MiB; Stage 1 6044 MiB (NVML) |
| Device steady-state memory | Approximately 8755 MiB before ASR loading; excludes Whisper; a serving-only peak was not isolated |

| Concurrency | Audio seconds / wall second | TTFP p50 (ms) | TTFP p99 (ms) | Mean RTF | Mean WER |
| --- | --- | --- | --- | --- | --- |
| 1 | 6.915 | 252 | 331 | 0.149 | 1.904% |
| 4 | 9.842 | 597 | 1021 | 0.425 | 1.666% |
| 8 | 11.468 | 851 | 2042 | 0.711 | 1.018% |
| 16 | 13.031 | 1079 | 3800 | 1.241 | 0.940% |
| 32 | 15.722 | 2053 | 6711 | 2.059 | 2.601% |

All 640 timed requests and their ASR evaluations succeeded, with no missing PCM.
These measurements cover the default packed-only YAML, AR BF16 and Codec FP32,
with both stages on the same GPU, before the single-stage migration. They are one bounded measurement run, not a
statistical no-regression claim. Warmups and subsequent ASR evaluation are outside
the timed serving workload. High-concurrency RTF exceeds 1, so successful request
completion does not imply uninterrupted real-time playback. Graph replay was
observed, but aggregate eager-fallback counts were not collected.

The measured Codec source SHA-256 is
`d3973c718a45f8cde4b476abbb7fee8b4223cd0110fcbb3aa5c2af813fe14a93`;
the graph wrapper SHA-256 is
`94e0b55fdd1c19eed2c3d12cef2bff4105ab7b5fc9cbbc579edca3f497c74d0f`.
Process memory includes runtime state and graph pools, not only weights.
See the
[upstream source](https://github.com/Zyphra/ZONOS3-Mini) for model details.
