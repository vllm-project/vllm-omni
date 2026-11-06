# LFM3-Omni-2B on Ascend and CUDA

## Model and execution

[LiquidAI/LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B)
accepts text, images, audio, and silent video frames. The temporal model and
MaskGIT depth decoder run in the same autoregressive stage: each generated
audio frame becomes the temporal model's next input. A second stage runs
LiquidCodec and returns 24 kHz audio. Text-only deployment omits that stage.

The checkpoint contains custom Transformers code and requires
`--trust-remote-code`. The reference used during integration is revision
`21d75123f86c1f7ffb4bef1ea202710c4e7ebe67` with Transformers 5.20.2.
Pin both model and code revisions when reproducing comparisons. The reference
model uses `attn_implementation="eager"` for temporal, encoder, and depth
attention; LiquidCodec retains its checkpoint-native dtype and attention backend.
The serving integration also runs with Transformers 5.19.0, within the
repository's supported dependency range; upgrading the shared installation
to the reference environment's version is not required.

## Deployment

The integration is validated on one Ascend 910B1 with 64 GiB HBM and on one
NVIDIA RTX 4090 with 24 GB.

From the repository root, in an installed vLLM-Omni Ascend environment:

```bash
vllm serve LiquidAI/LFM3-Omni-2B --omni --trust-remote-code \
  --revision 21d75123f86c1f7ffb4bef1ea202710c4e7ebe67 \
  --deploy-config vllm_omni/deploy/lfm3_omni.yaml --async-chunk \
  --hf-overrides '{"lfm3_depth_graph":true,"lfm3_vectorized_greedy":true}' \
  --port 8091
```

On CUDA, use the same command without `--hf-overrides`.
`lfm3_depth_graph` captures NPU graphs only. On the RTX 4090,
`lfm3_vectorized_greedy` measured within run-to-run noise; see the measured CUDA
performance below.

For text output, including image/audio/video understanding, select
`vllm_omni/deploy/lfm3_omni_text.yaml` and omit `--async-chunk` and the audio
optimization overrides. The speech command above transfers incremental audio
codes to LiquidCodec. Its cache and overlap-add state are retained per request
and flushed on the final chunk.

Use the shared OpenAI-compatible speech endpoint:

```bash
curl http://localhost:8091/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"LiquidAI/LFM3-Omni-2B","input":"Hello from LFM3.","instructions":"Perform TTS in American English using a female voice.","response_format":"wav"}' \
  --output speech.wav
```

Voice descriptions belong in `instructions`; this adapter has no named voice
table or reference-audio voice-cloning interface. The speech adapter appends
the checkpoint's audio-start marker. Raw token prompts for audio or interleaved
generation must end in `<|audio_start|>` or `<|mixed_start|>`, respectively.

## Configuration and limits

- Tensor and pipeline parallel sizes are both one. Both stages share device 0
  in the bundled deployment, with a maximum of four concurrent requests and
  a 4,096-token context.
- Prefix caching, chunked prefill, recompute preemption, and whole-model graph
  execution are disabled because convolution and audio feedback are owned by
  individual requests. Finished and aborted requests release this state.
- Reference attention is the default. It retains request-local KV tensors and
  reports their bounded capacity through `model_local_kv_specs`. Native paged
  attention can be selected with `hf_overrides.lfm3_attention_backend=paged`,
  but changes BF16 reduction behavior; revalidate quality for the workload.
- `hf_overrides.lfm3_depth_graph=true` enables fixed-shape NPU graph replay of
  the depth decoder. Sampling and the per-request random generator remain
  outside capture. Benchmark after warmup and compare generated codes before
  interpreting latency differences. The command above enables it together with
  `lfm3_vectorized_greedy`, which groups the greedy codebook selection. Both
  switches default to false when omitted; stochastic sampling preserves its
  original scalar random-draw order.
- On CUDA, the width-7 convolutions use the checkpoint's own Triton kernels and
  its seven-input cache, as the reference does on CUDA. NPU and CPU keep the
  torch convolution.
- Audio sampling defaults to temperature 0.4, repetition penalty 1.2, and a
  64-frame repetition window. `SamplingParams.extra_args` accepts
  `audio_temperature`, `audio_top_k`, `audio_repetition_penalty`, and
  `audio_repetition_window`. Temperature zero selects greedy audio sampling.
- Videos require timestamps from frame indices and FPS. Extraction of a video
  soundtrack, duplex input streaming, quantization, and distributed execution
  are outside this integration's supported scope.
- LiquidCodec loads its own checkpoint dtype and attention backend, matching
  `AutoProcessor.audio_tokenizer`; the temporal model uses BF16. Stage shutdown
  has a 30-second grace period for NPU worker cleanup.

## Validation

Compare the same checkpoint revision, prompts, media preprocessing, sampling
parameters, and seed with `AutoModelForMultimodalLM`. Check text token IDs,
valid audio codes, decoded sample count, and waveform error separately. The
internal audio-EOS placeholder is a scheduler step and is not a valid codec
frame. An interleaved text-end marker starts the remaining audio tail; it must
not terminate the full two-stage pipeline.

On the pinned checkpoint and environment above, six fixed understanding cases
(text, arithmetic, a prompt longer than the sliding window, image, audio, and
video) matched generated text token IDs. Greedy TTS (46 frames), seeded
temperature-0.4 TTS (42 frames), and greedy interleaved generation (63 frames)
matched every code across all 12 codebooks. The optimized full greedy waveform
was bitwise equal to the reference's 88,320 float samples. Nine online requests
at concurrency 1, 2, and 4 matched the four-frame streaming reference within one
PCM16 quantization unit, with normal worker shutdown. These fixed cases are
regression evidence, not a broad quality benchmark.

On one RTX 4090, with the same reference revision run in Transformers 5.20.2 on
CUDA, nine cases matched. Prompt token IDs and generated text token IDs were
identical for all nine:

- Understanding: text, arithmetic, a 500-token prompt beyond the sliding window,
  image, audio, and eight-frame video.
- Speech: greedy TTS (42 frames), seeded temperature-0.4 TTS (42 frames), and
  greedy interleaved generation (59 frames). Their waveforms were bitwise
  identical with full payloads.
- With `--async-chunk`, the waveforms were bitwise identical to the reference's
  four-frame streaming decode, except for the final-chunk limitation below.
- With `VLLM_BATCH_INVARIANT=1`, a concurrent mix of 18 requests matched each
  request's single-request output. Without it, concurrent greedy speech can
  differ from a single request's, because the GEMM shapes change.

A stream whose frame count is a multiple of four is missing its 480-sample
(20 ms) overlap-add tail. Its last chunk ends at a codec boundary, and the
empty final payload is not decoded. This was measured on CUDA; Ascend uses the
same scheduling path.

For performance measurements, reserve the NPU, warm both stages, and run paired
baseline/optimized measurements on the same card at each reported concurrency.
Report completed and failed requests, first-audio latency, total latency, audio
duration, real-time factor, and clean worker shutdown. A depth-decoder
microbenchmark does not establish endpoint throughput or overall optimality.

The checked-in benchmark retains per-request timings and audio tensors:

```bash
python benchmarks/lfm3_omni/benchmark.py --mode eager --output results/eager
python benchmarks/lfm3_omni/benchmark.py --mode depth --output results/depth
python benchmarks/lfm3_omni/benchmark.py --mode optimized --output results/optimized
```

Reserve one card for the comparison and repeat the chosen pair in A/B/B/A
order. Each invocation warms concurrency 1, 2, and 4 before three measured
batches at each concurrency, for both greedy audio and temperature 0.4.
`depth` enables only depth-decoder graph replay;
`optimized` also groups greedy codebook selection while preserving the MaskGIT
reveal order and repetition penalty. Stochastic sampling retains the original
scalar random-draw order. Use `--audio-temperature 0.4` to measure that path.
Compare output lengths and retained audio before reporting a speedup; inspect
worker logs for forced shutdown even when the benchmark process exits normally.

## Measured NPU performance

One reserved Ascend 910B1 (64 GiB), BF16, one short English TTS sentence, seed 42,
96-token limit, four-frame streaming codec chunks. A disables depth graphs and
grouped greedy sampling; B enables both. Both run the same model implementation.
Upstream main has no runnable LFM3 baseline; A is the functional eager adapter.
Order: A1/B1/B2/A2, with a fresh engine process for each arm. Each concurrency
and temperature has one warmup and three measured batches per process.
Startup and warmup are excluded. Timing covers the offline Omni API, including
orchestration and codec; it is not HTTP network latency.

Each reported observation averages three batches within one process; means and
sample standard deviations below use the two independent processes per arm.
This is a descriptive comparison, not a statistical significance claim.

### Audio temperature 0.0

| Concurrency | First audio, s (A → B) | Latency, s (A → B) | RTF (A → B) | Requests/s (A → B, mean ± SD) | Throughput change |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.490 → 0.331 | 5.494 → 3.617 | 1.493 → 0.983 | 0.182 ± 0.007 → 0.277 ± 0.002 | +51.8% |
| 2 | 0.759 → 0.453 | 8.391 → 4.675 | 2.280 → 1.270 | 0.238 ± 0.003 → 0.425 ± 0.007 | +78.9% |
| 4 | 1.275 → 0.678 | 14.026 → 6.708 | 3.811 → 1.823 | 0.285 ± 0.009 → 0.593 ± 0.011 | +108.4% |

### Audio temperature 0.4

| Concurrency | First audio, s (A → B) | Latency, s (A → B) | RTF (A → B) | Requests/s (A → B, mean ± SD) | Throughput change |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.505 → 0.360 | 5.190 → 3.599 | 1.545 → 1.071 | 0.193 ± 0.005 → 0.278 ± 0.005 | +44.2% |
| 2 | 0.779 → 0.504 | 7.860 → 4.878 | 2.339 → 1.452 | 0.253 ± 0.005 → 0.407 ± 0.009 | +60.7% |
| 4 | 1.337 → 0.771 | 13.321 → 7.324 | 3.964 → 2.180 | 0.300 ± 0.009 → 0.544 ± 0.007 | +81.4% |

All 168 measured requests completed (84 per configuration), with zero failures
in this completed paired run. Every retained waveform was bitwise identical
across concurrency, repeats, and A/B for its sampling temperature: 88,320
samples at temperature 0 and 80,640 at temperature 0.4. All four engine
processes and both stages exited normally, without forced worker termination.

Device process-memory peaks sampled every two seconds were 6,835/6,835 MB for A1/A2 and 6,851/6,851 MB for B1/B2. These are sampled process sums over startup, warmup and measurement, not allocator high-water marks.

The initial experiment was discarded after the benchmark consumer failed on a
metadata-only DELTA update; the corrected consumer was used for the complete
four-arm run reported here. No partial or failed-run timings are mixed in.

These results apply to the stated workload and are the best of the measured
configurations. They do not establish global optimality, broad quality
equivalence, or realtime latency for arbitrary speech lengths and concurrency.

## Measured CUDA performance

These are smoke measurements on one RTX 4090 (24 GB) with the checked-in
benchmark, for one English sentence (42 frames, 80,640 samples at
concurrency 1). Setup: BF16, four-frame streaming codec chunks, a warmed
engine, and three measured batches per concurrency. Timing covers the offline
Omni API, including orchestration and the codec.

```bash
python benchmarks/lfm3_omni/benchmark.py --mode eager --output results/cuda-eager
```

| Audio temperature | Concurrency | First audio, s | Latency, s | RTF | Requests/s |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 1 | 0.252 | 2.421 | 0.720 | 0.413 |
| 0.0 | 2 | 0.401 | 3.871 | 1.136 | 0.516 |
| 0.0 | 4 | 0.644 | 6.635 | 1.861 | 0.600 |
| 0.4 | 1 | 0.264 | 2.531 | 0.753 | 0.395 |
| 0.4 | 2 | 0.411 | 3.686 | 1.159 | 0.542 |
| 0.4 | 4 | 0.683 | 6.366 | 1.966 | 0.628 |

- With `--mode optimized`, the results were within 5% of these. At
  concurrency 1, its waveforms were bitwise identical to eager mode.
- Concurrent batches change the GEMM shapes, so greedy sample counts vary
  across concurrency.
