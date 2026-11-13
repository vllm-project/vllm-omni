# ZONOS3-Mini Speech API — RTX 3090

## Summary

- Vendor/model: Zyphra, [`Zyphra/ZONOS3-Mini`](https://huggingface.co/Zyphra/ZONOS3-Mini).
- Task: text-to-speech through `POST /v1/audio/speech`.
- Runtime: vLLM-Omni MRv2, AR → stateful ZAC decoder (`keep2_plain`).
- Hardware: one NVIDIA GeForce RTX 3090, 24 GiB; both stages on GPU 0.
- Maintainer: vLLM-Omni contributors.

## Supported model contract

| Item | This recipe |
| --- | --- |
| Input | Non-empty text; omit `voice` for unconditioned synthesis |
| Output | Mono WAV at 44,100 Hz |
| Provided profile | `vllm_omni/deploy/zonos3_mini.yaml`, 32 request slots, AR context 2048 tokens |
| Precision | AR BF16; speaker encoder and Codec weights FP32 |
| Execution | Eager AR prefill, graph AR decode, model-owned Codec graphs |
| Qualified scope | Single-GPU startup and one online WAV case with ASR content validation |
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

The deploy sets Stage 0 KV cache to **128 MiB**. GDN recurrent/conv/window state
is separately preallocated. Stage 1 `enforce_eager: false` enables complete Codec
graphs; set it to `true` for eager Codec execution. Graph buckets derive from
`max_num_seqs` and connector chunk sizes; no separate graph-bucket configuration
is needed.

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
  --stage-overrides '{"1":{"hf_overrides":{"zonos3_decoder":"full_mr"}}}' \
  --port 8000
```

Selection happens at startup, not in the Speech API request. Returning to
`keep2_plain` requires removing the override or setting it to `keep2_plain` and
restarting. The larger network may need more device memory and capture time;
the measured memory figures below apply only to `keep2_plain`.

### Disable Codec graphs

To keep AR decode graphs but run the Codec eagerly, override **Stage 1 only**:

```bash
HF_HUB_OFFLINE=1 FLA_TILELANG=0 \
vllm serve "$ZONOS3_MODEL_PATH" \
  --served-model-name Zyphra/ZONOS3-Mini \
  --omni \
  --deploy-config vllm_omni/deploy/zonos3_mini.yaml \
  --stage-overrides '{"1":{"enforce_eager":true}}' \
  --port 8000
```

The two overrides can be combined in one Stage 1 object:

```json
{"1":{"enforce_eager":true,"hf_overrides":{"zonos3_decoder":"full_mr"}}}
```

Do not apply a global `--enforce-eager` to this profile: Stage 0 currently requires
`FULL_DECODE_ONLY`. To adjust Codec bucket shapes, edit connector
`codec_chunk_frames` / `codec_chunk_ramp` and the Codec stage's `max_num_seqs` in
a copy of the deploy YAML. The graph keys are derived from those values.

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

## Qualification evidence

| Environment | Validated value |
| --- | --- |
| Accelerator | 1 × RTX 3090, 24 GiB; no inter-GPU communication |
| OS / Python | Ubuntu 22.04.5 LTS / Python 3.12.3 |
| PyTorch / CUDA build | 2.14.0+cu130 / 13.0 |
| NVIDIA driver | 595.71.05 |
| vLLM / Omni revision | 0.33.0 / `a3a8759dc` |
| Checkpoint | Revision pinned above, default `keep2_plain` decoder |
| Online case | One English WAV request, seed 42; ASR text matched the input |
| Graph capture | All 30 default Codec keys captured: B=1/2/4/8/16/32 × frames=1/2/4/8/15 |
| Process GPU memory | After startup, NVML reported Stage 0: 2674 MiB; Stage 1: 5078 MiB |

These are bounded startup/smoke observations, not peak-memory or throughput
benchmarks. They include runtime state and graph pools, not only weights.
The verification command above covers the online case. See the
[upstream source](https://github.com/Zyphra/ZONOS3-Mini) for model details.
