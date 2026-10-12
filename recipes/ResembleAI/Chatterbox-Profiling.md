# Chatterbox H100 profiling

These measurements used the combined model and runtime implementation. The
shared runtime changes are now reviewed separately in
[PR #8842](https://github.com/vllm-project/vllm-omni/pull/8842); the model branch
requires that prerequisite. Splitting the PRs did not produce new GPU measurements.

Measurements on October 12, 2026 used one H100 SXM 80 GB, PyTorch
2.13.0/CUDA 13.0, vLLM 0.31.0 and Transformers 5.14.1. Serving runtime revision
`26e4d3910` removes GPU-to-CPU scalar synchronization from flow-mask repair.
The subsequent `39f251fb0` changes only decoder benchmark OOM reporting.

## Bottlenecks and changes

Flow accounted for approximately 90–95% of isolated FP32 decoder time. Native
BF16 autocast applies only to flow; HiFT weights, compute, and returned audio
remain FP32. Selecting native cuDNN SDPA improves the BF16 flow further without
changing its masks. Unsupported cuDNN shapes fail explicitly.

At batch 64, isolated steady chunks contain 2.4 seconds of output audio:

| Model | FP32 decoder seconds | BF16 seconds | BF16 + cuDNN seconds |
| --- | --- | --- | --- |
| Turbo | 0.840 | 0.385 | 0.325 |
| Original | 4.502 | 1.623 | 1.331 |

These replay real checkpoint conditioning and codes, but exclude generation,
conditioning, transport and competing GPU stages. They cannot establish serving
latency. Original performs ten flow steps and internal classifier-free guidance;
Turbo performs two steps without decoder CFG.

Flow-mask repair also performed a host scalar read on every attention layer.
The repair now uses an equivalent Boolean tensor expression, and pad masks use
the already-known padded length. Real-GPU traces reduced decoder scalar reads
from 31 to 2 for Turbo and 143 to 2 for Original. Synchronization wait times
overlap GPU execution; their sum is not a predicted wall-clock speedup.

## Fully warmed Turbo batch-size sweep

The HTTP streaming workload uses 64 concurrent clients, 128 full-load warmup
requests, then 128 timed requests. Each request reads the same approximately
65-word English narration with a different request number and uses one cached
reference recording. Seed is zero and the output allowance is 1000 speech tokens.
This is a fixed-workload test, not a general latency guarantee.

| Decoder slots | MPS | p95 RTF | p95 first audio (s) | p95 playback deficit (s) | p99 deficit (s) |
| --- | --- | --- | --- | --- | --- |
| 8 | No | 0.552 | 2.443 | 0.612 | 0.615 |
| 16 | No | 0.435 | 2.077 | 0.195 | 0.773 |
| 32 | No | 0.373 | 1.713 | 0.076 | 0.077 |
| 64 | No | 0.383 | 1.782 | 0.062 | 0.124 |
| 32 | Yes | 0.332 | 1.444 | 0 | 0 |
| 64 | Yes | 0.332 | 1.395 | 0 | 0 |

Every row completed all 128 timed requests with zero failures. RTF is complete
request latency divided by generated audio duration. Playback begins at the
first chunk; the continuity metric allows 100 ms of scheduling jitter. Zero
reported deficit means no underrun beyond that tolerance. First-audio latency
still matters even when RTF is below one. Two small warmups do not warm kernels
for a 64-row flow batch and produce materially different tails.

The selected CUDA overlay retains 64 decoder slots: 32 and 64 have similar RTF,
and 64 has slightly lower first-audio latency in this run. Each scheduled flow
batch is decoded together; there is no internal four-request decoder limit.
HiFT groups incompatible stream lengths/cache states to preserve correctness.

## Matched Qwen3-TTS target

Qwen3-TTS 12Hz 1.7B Base used its existing
`qwen3_tts_high_concurrency_mrv2_single_gpu.yaml` profile on the same H100, with
the same references, target text, seed, allowance, 128 full-load warmups and
128 timed HTTP requests. Both rows completed all requests without failures:

| Model, 64 clients | p95 RTF | p95 first audio (s) | p95 playback deficit (s) | p99 deficit (s) |
| --- | --- | --- | --- | --- |
| Qwen3-TTS Base | 0.185 | 0.775 | 0.279 | 0.309 |
| Chatterbox Turbo CUDA | 0.332 | 1.395 | 0 | 0 |

Qwen generates this workload faster and starts audio sooner. Its smaller early
chunks still produce playback deficits under load; Turbo's larger initial
buffer avoids measured deficits in this warm workload. This does not establish
overall Qwen parity or a universal real-time service guarantee.

## Original concurrency

With the same BF16/cuDNN/MPS settings and 15-token initial chunks, Original
completed every request but did not sustain continuous playback at 64 clients.
Each row uses 128 warmups and 128 timed requests:

| Clients | p95 RTF | p95 first audio (s) | p95 playback deficit (s) | p99 deficit (s) |
| --- | --- | --- | --- | --- |
| 8 | 0.346 | 1.376 | 0 | 0 |
| 16 | 0.385 | 1.473 | 0 | 0.008 |
| 32 | 0.589 | 2.043 | 0.327 | 0.327 |
| 64 | 1.096 | 4.021 | 1.873 | 1.874 |

The Original CUDA profile preserves the 64-request admission contract. The
table distinguishes that capacity from a suitable interactive operating point.
It does not reduce the flow step count or disable either classifier-free
guidance path to make a performance number look better.

Increasing only the initial chunk setting to 30 tokens reduces repeated early
flow work. With the ten-second reference, alignment emits approximately
1.84 seconds in the first chunk instead of 0.64 seconds. The maximum remains
60 tokens. The same fully warmed protocol measured:

| Clients | p95 RTF | p95 first audio (s) | p95 playback deficit (s) | p99 deficit (s) |
| --- | --- | --- | --- | --- |
| 32 | 0.560 | 2.130 | 0 | 0 |
| 64 | 1.016 | 3.529 | 0.531 | 0.531 |

All 128 timed requests completed in each run. The opt-in Original CUDA profile
uses this 30-token initial setting; the ordinary profiles retain 15. At 32
clients it sustained continuous playback on this workload. At 64 it still
did not. A separate 64-output transcription check measured 0.19% mean WER
with zero ASR failures for the 30-token setting.

## Conditioning and quality boundaries

On an A5000, encoding 64 real speech references in batches of 8/16/32/64 took
3.189/3.074/3.055/3.031 seconds, with peak allocated memory of
0.83/1.15/1.78/3.05 GiB. All matched serial conditioning within the existing
tensor tolerances without changing encoder buffers. Retaining batches of eight
trades approximately 5% encoding time for substantially lower memory. The
64-request admission bound and duplicate-miss coalescing remain independent of
this encoder batch size.

Whisper-large-v3 measured mean WER of 0.22% on 64 Turbo BF16/cuDNN outputs and
0.27% on 64 Original outputs from this narration. A Qwen3-TTS Base run on the
same narration measured 0.11%. All three had zero ASR failures.
These checks establish narrow intelligibility evidence, not speaker similarity
or broad quality parity. BF16 changes numerical audio values. Cold-reference
variants derived from one recording stress cache misses, not distinct speakers.

The final fresh-server conditioning-inclusive HTTP and sentence-streaming
WebSocket checks of the committed CUDA profile were interrupted when Runpod
stopped the H100 for insufficient account balance. They did not produce complete
reports and are not passing validation. Earlier capacity results in the recipe
predate the full-batch decoder; the new warm performance runs are separate from
proof of 64 simultaneously active requests with cold conditioning.

Profiling matrices, traces, quality logs, 64-output quality WAV sets, and the
Original concurrency reports were saved locally before shutdown. The final hot
Qwen JSON was not copied before the shutdown; its table above uses the completed
report's metrics captured from the pod before it stopped. Do not represent that
capture as a complete raw report.

## Reproduction

Start Turbo with the CUDA overlay, and start Qwen3-TTS Base separately with
`qwen3_tts_high_concurrency_mrv2_single_gpu.yaml` on the same GPU. Use identical
Seed-TTS metadata and reference audio for both model-specific `voice_clone`
tasks. Restart between models and report cold conditioning separately.

The fixed target text was:

```text
Request number {index}. The river moves quietly past the old stone bridge today.
In the morning, people walk along the bank and listen to the birds in the trees.
A small boat carries fresh bread to the market across the water. By evening,
the streets become quiet again, and warm lights shine through the windows of the houses.
```

The reference is `tests/assets/qwen3_tts/clone_2.wav`. Warm metadata reuses the
same file for every row; cold metadata uses distinct gain variants of that
recording. The benchmark uses the same GPT-2 tokenizer only to count prompt
tokens for both models, not to tokenize model input inside the server.

```bash
vllm-omni serve ResembleAI/chatterbox-turbo --omni --port 8091 \
  --deploy-config vllm_omni/deploy/chatterbox_turbo_high_concurrency_cuda.yaml

uv run benchmarks/tts/bench_tts.py \
  --model ResembleAI/chatterbox-turbo --task voice_clone --locale en \
  --concurrency 64 --num-prompts 128 --num-warmups 128 \
  --dataset-path /path/to/workload --output-len 1000 --request-seed 0 \
  --output-dir /path/to/results --port 8091 -- \
  --tokenizer /path/to/common-gpt2-tokenizer \
  --save-detailed --metric-percentiles 50,95,99
```

The metadata reference transcript is supplied for Qwen's Base task; Chatterbox
does not require it. Add `--wer-eval` for a separate quality run. Inspect the
quality log and saved WAVs as well as the report: the current wrapper can omit
WER from its summary JSON. Use the recipe's separate public-API capacity harness
to measure active scheduler occupancy; client concurrency alone does not prove
64 simultaneously running model requests.
