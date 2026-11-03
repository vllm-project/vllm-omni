# LFM3-Omni-2B on Ascend

## Model and execution

[LiquidAI/LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B)
accepts text, images, audio, and silent video frames. The temporal model and
MaskGIT depth decoder run in the same autoregressive stage: each generated
audio frame becomes the temporal model's next input. A second stage runs
LiquidCodec and returns 24 kHz audio. Text-only deployment omits that stage.

The checkpoint contains custom Transformers code and requires
`--trust-remote-code`. The reference used during integration is revision
`21d75123f86c1f7ffb4bef1ea202710c4e7ebe67` with Transformers 5.20.2.
Pin both model and code revisions when reproducing comparisons.
The serving integration also runs with Transformers 5.19.0, within the
repository's supported dependency range; upgrading the shared installation
to the reference environment's version is not required.

## Deployment

The integration is validated on one Ascend 910B1 with 64 GiB HBM. GPU execution
has not been validated by this recipe.

From the repository root, in an installed vLLM-Omni Ascend environment:

```bash
vllm serve LiquidAI/LFM3-Omni-2B --omni --trust-remote-code \
  --revision 21d75123f86c1f7ffb4bef1ea202710c4e7ebe67 \
  --deploy-config vllm_omni/deploy/lfm3_omni.yaml \
  --hf-overrides '{"lfm3_depth_graph":true,"lfm3_vectorized_greedy":true}' \
  --port 8091
```

For text output, including image/audio/video understanding, select
`vllm_omni/deploy/lfm3_omni_text.yaml`. The full deployment supports
`--async-chunk` to transfer incremental audio codes to LiquidCodec. Its cache
and overlap-add state are retained per request and flushed on the final chunk.

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
