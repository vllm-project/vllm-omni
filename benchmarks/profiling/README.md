# MiniCPM-o multimodal encoder profiling

`benchmark_multimodal_encoder.py` measures client-observed text TTFT, total
latency, steady/peak GPU memory, and per-process GPU-memory peaks against a
running vLLM-Omni server. NVML sampling starts before warmup so the report keeps
the first-modality allocation peak (`warmup_memory_summary`) separate from the
post-warmup request peak (`memory_summary`). Each request result retains output
token IDs and the latest per-stage metrics snapshot returned by the server. The
benchmark can also bracket measured requests with the server's stage profiler
endpoints.

The model server should use text-only output while profiling the thinker
encoder. This avoids mixing talker and Token2Wav costs into encoder TTFT.

Example image run:

```bash
python benchmarks/profiling/benchmark_multimodal_encoder.py \
  --base-url http://127.0.0.1:8099 \
  --model openbmb/MiniCPM-o-4_5 \
  --modality image \
  --asset /path/to/fossil.png \
  --profile-stage 0 \
  --server-max-model-len 8192 \
  --server-max-num-seqs 1 \
  --server-limit-mm-per-prompt '{"image":1}' \
  --output results/image-mml8192-seqs1.json
```

Repeat `--asset` to test multiple items in one prompt. Use `--concurrency` to
match the request pressure intended for `max_num_seqs`. The JSON records the
server configuration as metadata; it does not change server settings.

For image, audio, and video baselines, suitable model-repository assets are:

- `assets/fossil.png`
- `assets/Trump_WEF_2018_10s.mp3`
- `assets/Skiing.mp4`

Start the server with torch profiler memory capture enabled before passing
`--profile-stage 0`. Without server profiling, omit that option; external TTFT
and NVML memory sampling still work.

For an omni LLM stage, profiler settings belong in the stage entry of a copied
deploy YAML (not the diffusion-only CLI example). A stage-0 configuration for
one active request can include:

```yaml
max_model_len: 8192
max_num_seqs: 1
limit_mm_per_prompt:
  image: 1
  audio: 1
  video: 1
profiler_config:
  profiler: torch
  torch_profiler_dir: /path/to/results/profiles
  torch_profiler_record_shapes: true
  torch_profiler_with_stack: true
  torch_profiler_with_memory: true
```

When using a ModelScope snapshot, keep the public API model name stable while
loading the local path:

```bash
vllm serve /path/to/OpenBMB/MiniCPM-o-4_5 --omni \
  --served-model-name openbmb/MiniCPM-o-4_5 \
  --deploy-config /path/to/profile-deploy.yaml \
  --trust-remote-code --host 127.0.0.1 --port 8099
```

For reproducible comparisons on the test system, both versions were started
with `VLLM_USE_FLASHINFER_SAMPLER=0`. FlashAttention 2 remained enabled; only
the FlashInfer top-k/top-p sampler was disabled because it failed independently
on this Blackwell/CUDA 13 environment.

### Benchmark environment

| Component | Version or configuration |
| --- | --- |
| Baseline source | vLLM-Omni `d09f549e` |
| Changed source | `d09f549e` plus the MiniCPM-o encoder candidate diff |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB |
| NVIDIA driver / reported CUDA | 590.44.01 / 13.1 |
| Python | 3.12.3 |
| PyTorch / packaged CUDA runtime | 2.11.0+cu130 / 13.0 |
| vLLM | 0.25.0 |
| Transformers | 5.14.1 |
| Triton | 3.6.0 |

All before/after server pairs used the same checkpoint, virtual environment,
GPU, sampler setting, deploy configuration, and media preprocessing. Direct
encoder timings use CUDA events after one warmup. Server latency statistics use
independent cache-miss media variants rather than repeated cache hits.

## Initial baseline matrix

Run text output only so stage-1 talker latency and memory do not contaminate
the thinker encoder measurements. Record the matching server settings on each
invocation.

| Dimension | Initial values |
| --- | --- |
| Modality | text control, image, audio, video |
| `max_model_len` | 4096, 8192, 16384 |
| Items per prompt | 1, then 2/4 where `limit_mm_per_prompt` permits |
| `max_num_seqs` and client concurrency | 1, 2, 4 on stage 0 |
| Repetitions | 1 warmup, at least 5 measured requests |

The shipped full MiniCPM-o pipeline keeps stage 1 at `max_num_seqs: 1` for
correctness. Higher-concurrency measurements therefore apply only to text
output ending at stage 0 unless the talker limitation is fixed separately.

For memory accounting, use the post-warmup NVML baseline for weights, KV cache,
and persistent allocations. The measured peak minus that baseline is the
request-time delta. Both device-wide and per-process baselines, peaks, and
deltas are retained. A torch-profiler memory snapshot is still required to
split that delta into allocator blocks, activations, copies, and workspaces.

## Encoder call paths to inspect

The first profiling pass should focus on these functions in
`minicpmo_4_5_omni_llm.py`:

- `get_multimodal_embeddings`: modality dispatch and output lifetime.
- `get_vision_hidden_states`: dense `all_pixel_values` padding buffer,
  `patch_attn_mask`, SigLIP `vpm`, and `resampler` boundaries. Image and
  video-as-image both use this path. The baseline active path sends all
  frames/slices to `vpm` at once and does not apply `config.vision_batch_size`
  (16), although the older `_process_image_input` helper contains chunking.
- `SiglipVisionTransformer.forward`, `SiglipEncoder.forward`, and
  `Resampler.forward`: encoder activations and attention workspace.
- `get_audio_hidden_states`: padded `wavforms`, `seq_range`, expanded masks,
  materialized quadratic `audio_attention_mask`, Whisper `apm`, projection,
  pooling, and final concatenation. The deployed configuration selects
  `audio_encoder_layer=-1`, but the call still uses
  `output_hidden_states=True`, retaining every Whisper layer output until the
  call returns even though only the final hidden state is consumed.
- `MiniCPMWhisperEncoder.forward`: per-layer activation and attention costs.
- `MiniCPMO45OmniLLMForConditionalGeneration.forward`: the second text
  embedding construction and per-item `torch.zeros_like` materialization used
  for `text_inputs_embeds`.

Do not change these paths before collecting the first trace. If operator stacks
are insufficiently attributable, add narrowly scoped `record_function` ranges
around the listed boundaries and rerun the same inputs/configuration.

The bounded video fix applies the existing `vision_batch_size` limit to the
active VPM path. It concatenates the VPM outputs and invokes the resampler once,
matching the older helper's operation ordering.

For any chunking change, compare the pre-merge multimodal embeddings from the
same inputs and seed, reporting shape/dtype plus maximum absolute error, maximum
relative error, and cosine similarity. The external benchmark also stores
response text hashes and token IDs, but those end-to-end checks do not replace
direct numerical parity at the encoder boundary.

The full-checkpoint comparison can use either synthetic patch tensors or the
exact frames and image preprocessing produced for a real video:

```bash
python benchmarks/profiling/compare_minicpmo_vision_batching.py \
  --model /path/to/OpenBMB/MiniCPM-o-4_5 \
  --video /path/to/Skiing.mp4 \
  --vision-batch-size 16 \
  --latency-repetitions 10 \
  --output results/vision-batching.json
```

Pass `--image /path/to/fossil.png` instead of `--video` to exercise the exact
image slicing output. Whisper hidden-state retention can be measured with the
matching full-checkpoint audio tool:

```bash
python benchmarks/profiling/compare_minicpmo_audio_hidden_states.py \
  --model /path/to/OpenBMB/MiniCPM-o-4_5 \
  --audio /path/to/Trump_WEF_2018_10s.mp3 \
  --latency-repetitions 20 \
  --output results/audio-hidden-states.json
```

## Cross-modality memory pressure

The standalone allocator measurements isolate the encoder modules from the
17.46 GiB serialized checkpoint, the CUDA runtime, and the KV cache. They show
that the 32-frame video batch is the dominant transient on this input set;
Whisper and a three-slice image are much smaller but still have avoidable or
measurable allocations.

| Encoder input | Original standalone peak allocated delta | Main source |
| --- | ---: | --- |
| Image: 600x390, three slices | 583,159,296 B | SigLIP activations and attention |
| Audio: 10.006 s, 1,001 mel frames | 56,558,592 B | Whisper activations plus retained layer outputs |
| Video: 32 frames, 1,032 patch tokens/frame | 5,958,894,592 B | Batched SigLIP attention and activations |

For comparison, the original 4K server startup leaves 28.96 GiB for KV cache
(210,864 tokens). The unbatched video's 6,262 MiB server request delta is about
21% of that available KV allocation and directly reduces the cache capacity
reserved during startup profiling. The image and 10-second audio transients do
not materially change startup KV sizing in this single-item configuration.

## RTX PRO 6000 Blackwell image result

`fossil.png` produces three VPM items, with 1,008 or 1,040 patch tokens each.
That is below `vision_batch_size=16`, so the changed code executes the same
single VPM call as the original. Five content variants were used for server
cache misses.

| Metric | Original | Changed | Change |
| --- | ---: | ---: | ---: |
| Server request peak delta | 0 MiB | 0 MiB | no sampled change |
| Standalone peak allocated delta | 583,159,296 B | 573,512,704 B | -1.65% |
| Standalone peak reserved delta | 715,128,832 B | 692,060,160 B | -3.23% |
| Direct vision encoder mean, 20 runs | 49.69 ms | 49.05 ms | -1.28% |
| Image TTFT mean, 5 content variants | 262.18 ms | 265.08 ms | +1.10% |
| Image end-to-end mean, 5 variants | 413.66 ms | 420.37 ms | +1.62% |

The direct encoder ranges were 49.24-50.38 ms and 48.77-49.29 ms. Across the
five server variants, TTFT was 262.18 +/- 23.55 ms and 265.08 +/- 17.28 ms
(sample standard deviation); end-to-end latency was 413.66 +/- 25.92 ms and
420.37 +/- 18.01 ms.

The direct post-resampler output has shape `[3, 64, 4096]` and is bit-identical
in BF16. Since both measurements execute the same call shape, the small direct
memory and latency differences are allocator/timing noise rather than an image
optimization. This is the independent image regression result: the bounded
batching branch only changes inputs larger than 16 VPM items.

## RTX PRO 6000 Blackwell video result

This result compares commit `d09f549e` with the VPM-only batching change on a
single 96 GiB NVIDIA RTX PRO 6000 Blackwell Server Edition. The checkpoint is a
local ModelScope snapshot of `OpenBMB/MiniCPM-o-4_5`; both runs use BF16,
`max_model_len=4096`, `max_num_seqs=1`, `limit_mm_per_prompt` of one per
modality, eager execution, and the same sampler setting described above.

`Skiing.mp4` is uniformly decoded to 32 frames. MiniCPM preprocessing produces
32 VPM items with a 24x43 grid, or 1,032 patch tokens per frame. The original
trace materializes a `[32, 16, 1032, 1032]` attention tensor in every SigLIP
layer. The changed trace has two groups of `[16, 16, 1032, 1032]`; the final
resampler still runs once over all 32 VPM outputs.

| Metric | Unbatched VPM | VPM batch 16 | Change |
| --- | ---: | ---: | ---: |
| Server request peak delta | 6,262 MiB | 2,606 MiB | -58.4% |
| Standalone peak allocated delta | 5,958,894,592 B | 3,032,646,656 B | -49.1% |
| Standalone peak reserved delta | 7,136,608,256 B | 3,598,712,832 B | -49.6% |
| Available KV cache at startup | 28.96 GiB | 36.35 GiB | +25.5% |
| GPU KV cache capacity | 210,864 tokens | 264,672 tokens | +25.5% |
| Direct vision encoder mean, 10 runs | 458.98 ms | 464.39 ms | +1.18% |
| Video TTFT mean, 5 content variants | 1,328.18 ms | 1,289.92 ms | -2.88% |
| Video end-to-end mean, 5 variants | 1,522.95 ms | 1,502.35 ms | -1.35% |

The direct encoder ranges were 458.02-461.96 ms and 464.29-464.63 ms. Across
the five server variants, TTFT was 1,328.18 +/- 102.22 ms and
1,289.92 +/- 87.03 ms; end-to-end latency was 1,522.95 +/- 101.22 ms and
1,502.35 +/- 86.06 ms.

The direct encoder timing is a CUDA-event measurement after one warmup using
the full vision checkpoint and the same preprocessed video tensors. The server
latency rows use five content-modified videos so every request is an encoder
cache miss. The first video also retains cold modality allocation behavior.

The profiler shows that cumulative allocation and compute do not disappear:
the 27 VPM layers become 54 half-batch calls. The reduction comes from bounding
simultaneously live attention and activation tensors. Profiler-observed VPM
`bmm` dimensions change from a leading dimension of 512 to 256, while external
latency remains effectively neutral.

### 16K deployment matrix

Two additional single-GPU configurations separate multimodal item pressure
from request concurrency:

- `minicpmo_4_5_single_gpu_16k.yaml`: `max_num_seqs=1`, up to four videos in
  one prompt, and `gpu_memory_utilization=0.60`.
- `minicpmo_4_5_single_gpu_16k_seqs4.yaml`: four active sequences with one
  video each and `gpu_memory_utilization=0.75`.

| 16K configuration | Unbatched VPM | VPM batch 16 |
| --- | ---: | ---: |
| One sequence, four-video limit: startup | Fails, 0 GiB KV left | Starts |
| One sequence, four-video limit: KV cache | 0 tokens | 187,680 tokens / 25.78 GiB |
| Four sequences, one-video limit: KV cache | 74,528 tokens / 10.24 GiB | 291,408 tokens / 40.02 GiB |
| Four sequences: maximum 16K concurrency | 4.55x | 17.79x |
| Four concurrent cache-miss videos: mean TTFT | 2,875.55 ms | 2,891.75 ms |
| Four concurrent cache-miss videos: mean end-to-end | 3,552.64 ms | 3,590.54 ms |

For the four concurrent requests, TTFT sample standard deviation was 550.08 ms
before and 543.86 ms after; end-to-end standard deviation was 6.38 ms and
16.54 ms, respectively.

The final code also completed a single 8,496-token prompt containing four
different 32-frame videos. Its cold TTFT was 4,740.49 ms, end-to-end latency
was 4,959.87 ms, and its NVML request peak delta was 6,212 MiB. All four
concurrent requests succeeded on both versions; the changed latency is +0.56%
for TTFT and +1.07% end-to-end in this small stress sample.

The full cross-product (`max_num_seqs=4` and four videos per prompt) is not a
viable 0.60-utilization single-GPU configuration. Startup profiles 16 maximum
video items: the original path OOMs while trying to allocate one 32 GiB
attention tensor, and the batched path finishes encoder profiling but leaves
no KV blocks. Splitting item-limit and concurrency stress as above is the
practical configuration guidance for a 96 GiB card.

### Numerical parity

The comparison uses the exact 32 decoded skiing frames and compares the
post-resampler tensor of shape `[32, 64, 4096]` directly.

| Dtype | Cosine | Normalized RMSE | p99 absolute error | Max absolute error |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 0.999999999992 | 3.97e-6 | 1.23e-5 | 3.57e-4 |
| BF16 deployment | 0.999499875 | 3.16e-2 | 1.02e-1 | 2.50 |

FP32 establishes the algorithmic equivalence. BF16 changes the matrix batch
shape, so backend kernel selection and rounding differ and the error compounds
through 27 layers. Maximum relative error is not used as the primary criterion
because reference values near zero make it unstable; cosine, normalized RMSE,
p99 absolute error, and the reference scale are retained in the JSON report.
All five video requests, the image regression, and the text control completed
successfully. The BF16 video generations remained semantically equivalent but
were not token-identical: one of five token-ID sequences and response hashes
matched exactly, while four differed in short continuations such as "mountain"
versus "mountain slope". This numerical effect of changing the BF16
matrix-batch shape is reported explicitly rather than treated as bit parity;
the FP32 comparison establishes algorithmic equivalence.

Targeted CPU validation reports seven passing tests under the repository pytest
configuration, covering latency statistics, payload construction, memory
summarization, stage metrics, exact fake-VPM ordering, and final/non-final
Whisper layer selection. `ruff`, `py_compile`, and `git diff --check` also pass.

## RTX PRO 6000 Blackwell audio result

The deployed configuration always selects `audio_encoder_layer=-1`, but the
original call asks Whisper for every hidden state and indexes the last element.
The bounded fix requests only `last_hidden_state` for `-1`; non-final layer
selection retains the original `output_hidden_states=True` behavior.

`Trump_WEF_2018_10s.mp3` is resampled to 16 kHz and produces an input tensor of
shape `[1, 80, 1001]`. After convolution and pool step 5, the returned audio
embedding contains 100 tokens of width 4,096. The direct comparison uses the
full Whisper and projection checkpoint in BF16 with eager Whisper attention.

| Metric | Retain all 24 layer outputs | Final state only | Change |
| --- | ---: | ---: | ---: |
| Standalone peak allocated delta | 56,558,592 B | 23,391,232 B | -58.6% |
| Standalone peak reserved delta | 77,594,624 B | 29,360,128 B | -62.2% |
| Direct audio encoder mean, 20 runs | 10.19 ms | 10.10 ms | -0.85% |
| Cold server request peak delta | 30 MiB | 8 MiB | -73.3% |
| Audio TTFT mean, 5 content variants | 210.82 ms | 183.01 ms | -13.2% |
| Audio end-to-end mean, 5 variants | 597.63 ms | 577.87 ms | -3.31% |

The direct encoder ranges were 10.10-10.33 ms and 9.97-10.34 ms. Across the
five server variants, TTFT was 210.82 +/- 30.89 ms and 183.01 +/- 35.98 ms;
end-to-end latency was 597.63 +/- 29.11 ms and 577.87 +/- 35.57 ms.

The direct output is bit-identical (`max_absolute_error=0`, cosine effectively
1.0). All five end-to-end token-ID sequences and response hashes also match.
The torch-profiler traces retain the same 24 Whisper attention calls and tensor
shapes; the changed stage-0 context CUDA duration is 101.44 ms versus 120.36 ms
in the original trace. The direct CUDA-event result is the more isolated
encoder latency comparison, while the server result includes preprocessing and
LLM prefill.

Raw audio on commit `d09f549e` otherwise fails with 100 encoder embeddings for
250 placeholders. These measurements apply upstream PR #5125 only in separate
experimental worktrees to align processor `pool_step=5`. That unrelated patch
is deliberately excluded from this change; the candidate diff contains only
the encoder memory fixes. Audio validation should be repeated without the
temporary dependency once #5125 lands.

## Checkpoint weight baseline

The ModelScope and Hugging Face safetensor shards have identical byte sizes.
Reading only their headers gives this BF16 checkpoint breakdown:

| Prefix | Parameters | Bytes |
| --- | ---: | ---: |
| `llm` | 8,189,195,264 | 16,378,390,528 |
| `vpm` (SigLIP) | 417,792,240 | 835,584,480 |
| `apm` (Whisper) | 307,216,384 | 614,432,768 |
| `resampler` | 88,907,776 | 177,815,552 |
| `audio_projection_layer` | 20,979,712 | 41,959,424 |
| `tts` | 347,696,290 | 695,392,580 |
| Total | 9,371,787,666 | 18,743,575,332 |

These are serialized tensor bytes, not runtime process memory. Stage 1 lazily
constructs portions of the TTS/Token2Wav path in float32, and CUDA contexts,
allocator fragmentation, KV cache, activations, and workspaces must be measured
separately.

The thinker has 36 layers, 8 KV heads, and a 128-wide head dimension. A dense
BF16 KV cache therefore requires 147,456 bytes per cached token per sequence,
before block metadata and alignment:

| Tokens per sequence | Theoretical BF16 KV bytes |
| ---: | ---: |
| 4,096 | 603,979,776 (576 MiB) |
| 8,192 | 1,207,959,552 (1.125 GiB) |
| 16,384 | 2,415,919,104 (2.25 GiB) |
| 40,960 (model maximum) | 6,039,797,760 (5.625 GiB) |

vLLM normally allocates KV blocks from the memory left by weights and its
`gpu_memory_utilization` budget, so startup logs and runtime measurements remain
the source of truth.
