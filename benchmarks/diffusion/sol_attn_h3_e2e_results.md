# Original MiniMax-H3: Sol-Attn generation benchmark

Measured on 2026-11-17 using the original MiniMax-H3 FL2VA checkpoint in
text-to-video-and-audio mode and the unmodified upstream SOL kernel.
This is a matched dense/SOL comparison, not a FastH3 or adapted-weight result.

## Revisions and environment

| Component | Version |
| --- | --- |
| vLLM-Omni runtime | `e1c527896f8e58523de86d4fcaa9c48749fa2555` |
| Base revision | `e58c990da9ac75721e1f0ee9b42b632d5cc86bd6` |
| `MiniMaxAI/MiniMax-H3` | `0919fb684102ef115126612f3fbb84d1027e7317` |
| NVlabs/Sana SOL source | `f4035e0a976d828ed41c7b0b799e9712b47cd60d`, `sol-attn` 0.5.0 |
| PyTorch / CUDA / cuDNN | 2.13.0 / 13.0 / 92000 |
| vLLM / transformers | 0.33.0 / 5.19.0 |
| Hardware | Four CUDA SM120 GPUs, 73,415 MiB each, driver 580.95.05 |

The driver reports `NVIDIA Graphics Device`; no commercial GPU SKU is inferred.
Other GPUs on the host ran unrelated workloads. All 81 official FL2VA files
were verified using their upstream LFS SHA256 or Git blob hashes. The environment
emits a vLLM 0.33.0 / vLLM-Omni 0.28.0 metadata warning; this is not a clean-release
compatibility certification. Subsequent report/test-only changes do not change
the benchmarked runtime.

## Matched workload

- BF16, diffusion TP4 and text-encoder TP4, eager execution, tiled VAE decoding.
- One request at a time: 1344x768, 24 FPS, requested duration 5 seconds,
  20 denoising steps, video/audio flow shifts 12.0/3.0.
- Dense baseline: `CUDNN_ATTN`. Sparse backend: `SOL_ATTN`, `tau=1.0`,
  `thresh_type=diag`, `dense_steps=10`, `dense_layers="0,1"`,
  `prefix_sink=true`, `kv_splits=1`.
- Both backends override `minimax_h3.token_refiner` with `TORCH_SDPA`.
- No adapters, quantization or additional cache acceleration.
- One complete warmup per backend, then three timed repeats at seed 1101.
  Seeds 1102 and 1103 each have one additional generation for quality comparison.

Prompt:

> At night, while their owner sleeps, three cats march into a bedroom playing
> tiny brass instruments, freeze, and quietly march out.

Timing covers wall-clock offline `Omni.generate`: encoding, denoising, VAE
decoding and output transfer. It excludes model startup, warmup, downloads and
MP4 muxing. **This is not HTTP-serving latency or a kernel-only benchmark.**

## Latency and memory

| Backend | Three repeats (s) | Median (s) | Sample standard deviation (s) |
| --- | --- | --- | --- |
| Dense cuDNN | 148.7167 / 148.7591 / 148.8526 | 148.7591 | 0.0695 |
| SOL | 135.3220 / 135.1698 / 135.1515 | 135.1698 | 0.0936 |

The median ratio is **1.10053x**, or **9.13506% lower generation latency**.
Warmups took 153.661 s and 143.528 s respectively and are excluded.
Sampled peak total device memory was approximately **39.235 GiB per GPU** for
both backends: no peak-memory benefit was observed. Memory was sampled through
NVML every 100 ms; this is not an exact allocator or per-process peak.

## Output differences

Both backends produced 124 video frames (5.1667 seconds) and 165600 stereo audio
samples at 32 kHz (5.175 seconds), with matching shapes and finite values.
All six saved MP4s decoded successfully. These checks do not establish AV sync.

| Seed | Mean SSIM | Mean frame PSNR (dB) | Mean LPIPS | Audio SNR (dB) | Audio correlation |
| --- | --- | --- | --- | --- | --- |
| 1101 | 0.90479 | 32.23 | 0.04713 | 18.51 | 0.9944 |
| 1102 | 0.92620 | 27.54 | 0.03140 | 9.36 | 0.9428 |
| 1103 | 0.92210 | 30.13 | 0.03971 | 3.43 | 0.7697 |

Metrics compare SOL against the same-seed dense output, not against ground
truth. SSIM/PSNR cover all 124 native-resolution RGB frames. AlexNet LPIPS v0.1
uses 16 evenly spaced frames, bilinearly resized to 256x256 and scaled to
[-1, 1]. Neither backend clipped the audio waveforms in these samples.

**Sparse attention is not lossless.** Seed 1103 has appreciable audio waveform
differences. Human listening, temporal video review and AV synchronization
assessment remain outstanding; the metrics are not perceptual acceptance.

## Evidence limits

- One prompt, three quality seeds, one preset and one hardware layout do not
  establish general speed or quality. No concurrency-throughput improvement
  is claimed.
- Ref2VA, adapted weights, other-model E2E, full-model compilation/HSDP and
  other CUDA architectures were not benchmarked.
- An initial profiling smoke recorded 384 SOL SM120 kernel calls but later
  failed in locally modified checkpoint VAE code. That trace is dispatch
  evidence only. The successful runs above used a clean official checkpoint,
  without runtime or kernel changes.
- The local benchmark harness and raw media are not checked into this report;
  this is recorded validation evidence, not a standalone benchmark runner.
- The earlier [SM120 sweep](sol_attn_sm120_results.md) predates the current
  prefix/padding/custom-op implementation and is historical evidence only.

Backend configuration and regression commands are in the
[Sol-Attn guide](../../docs/user_guide/diffusion/attention_backends/sol_attn.md).
