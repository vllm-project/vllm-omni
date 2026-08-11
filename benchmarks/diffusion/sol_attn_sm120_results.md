# MiniMax-H3 Sol-Attn speed and quality sweep on 4x RTX PRO 5000 Blackwell

This report records an end-to-end BF16 speed/quality sweep of the Sol-Attn
backend added by this pull request. A canonical copy is also retained in the
[vLLM-Omni rankings repository](https://github.com/lishunyang12/vllm-omni-rankings/tree/main/scripts/minimax_h3_sol_attn_sm120).

## Test configuration

- Hardware: 4x NVIDIA RTX PRO 5000 Blackwell (SM120), 73,415 MiB per GPU
- Physical GPU order: `4,6,5,7`
- Driver / CUDA reported by `nvidia-smi`: 580.95.05 / 13.0
- Model: MiniMax-H3 `FL2VA`, BF16
- Parallelism: diffusion TP4, text-encoder TP4
- Output: 1344x768, 24 FPS, 5 seconds, 16:9
- Denoise: 20 requested steps (19 progress-bar updates)
- Flow shifts: video 12.0, audio 3.0
- Prompt: `At night, while their owner sleeps, three cats march into a bedroom playing tiny brass instruments, freeze, and quietly march out.`
- Seed: 1101
- Timing: one warmup followed by three measured runs; the table reports median generation latency
- Visual quality: measured run 1 against the same-seed dense cuDNN output
- SSIM / PSNR: every decoded video frame at native resolution
- LPIPS: up to 16 uniformly sampled frames resized to 256x256, AlexNet weights

The dense reference uses `CUDNN_ATTN`. Unless a row says otherwise, Sol-Attn
uses `tau=1.0`, `thresh_type=diag`, `sink_tokens=951`, `sink_start=0`,
`dense_steps=10`, `dense_layers="0,1"`, and `kv_splits=1`.

## Main result

The quality-first preset reduced median latency from 142.49 seconds to 129.91
seconds: a 1.097x speedup and an 8.83% latency reduction. It passed all visual
quality gates with 0.93344 SSIM, 33.28 dB PSNR, and 0.08563 LPIPS.

The balanced preset reached 1.140x while passing every gate. The fastest tested
configuration that passed every gate was `sol_dense_steps_5` at 1.161x. The
more aggressive preset reached 1.205x, but failed the LPIPS gate.

| Configuration | Median (s) | Speedup | Latency reduction | SSIM | PSNR (dB) | LPIPS | Quality |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense cuDNN | 142.49 | 1.000x | 0.00% | 1.00000 | inf | 0.00000 | PASS |
| Sol-Attn recommended (`tau=1.0`, dense steps 10) | 129.91 | 1.097x | 8.83% | 0.93344 | 33.28 | 0.08563 | PASS |
| Sol-Attn balanced (`tau=1.5`, dense steps 8) | 124.95 | 1.140x | 12.31% | 0.91780 | 31.49 | 0.11771 | PASS |
| Sol-Attn fastest passing (`tau=1.0`, dense steps 5) | 122.75 | 1.161x | 13.86% | 0.90985 | 30.58 | 0.11525 | PASS |
| Sol-Attn aggressive (`tau=2.0`, dense steps 5) | 118.29 | 1.205x | 16.99% | 0.87586 | 27.00 | 0.22556 | FAIL |

## Dense-step speed/quality curve

Reducing the dense guard increases the fraction of denoise work handled by
the sparse Sol-Attn kernel. Five dense steps was the fastest point that still
passed all configured quality gates in this sweep.

| Dense steps | Median (s) | Speedup | SSIM | PSNR (dB) | LPIPS | Quality |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 142.73 | 0.998x | 0.95825 | 35.60 | 0.05441 | PASS |
| 15 | 136.73 | 1.042x | 0.95129 | 35.10 | 0.05834 | PASS |
| 8 | 127.13 | 1.121x | 0.93170 | 32.66 | 0.08395 | PASS |
| 5 | 122.75 | 1.161x | 0.90985 | 30.58 | 0.11525 | PASS |
| 0 | 115.35 | 1.235x | 0.75299 | 20.90 | 0.54201 | FAIL |

## Full sweep

| Case | Group | Median (s) | Speedup | Latency reduction | SSIM | PSNR (dB) | LPIPS | Quality |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `dense_cudnn` | baseline | 142.49 | 1.000x | 0.00% | 1.00000 | inf | 0.00000 | PASS |
| `sol_recommended` | preset | 129.91 | 1.097x | 8.83% | 0.93344 | 33.28 | 0.08563 | PASS |
| `sol_medium` | preset | 124.95 | 1.140x | 12.31% | 0.91780 | 31.49 | 0.11771 | PASS |
| `sol_aggressive` | preset | 118.29 | 1.205x | 16.99% | 0.87586 | 27.00 | 0.22556 | FAIL |
| `sol_tau_0p0` | tau | 135.17 | 1.054x | 5.14% | 0.94565 | 34.28 | 0.08346 | PASS |
| `sol_tau_0p5` | tau | 132.16 | 1.078x | 7.25% | 0.94117 | 33.87 | 0.08399 | PASS |
| `sol_tau_1p5` | tau | 128.24 | 1.111x | 10.01% | 0.93126 | 32.92 | 0.08067 | PASS |
| `sol_tau_2p0` | tau | 127.14 | 1.121x | 10.78% | 0.92328 | 31.81 | 0.10230 | PASS |
| `sol_dense_steps_0` | dense steps | 115.35 | 1.235x | 19.05% | 0.75299 | 20.90 | 0.54201 | FAIL |
| `sol_dense_steps_5` | dense steps | 122.75 | 1.161x | 13.86% | 0.90985 | 30.58 | 0.11525 | PASS |
| `sol_dense_steps_8` | dense steps | 127.13 | 1.121x | 10.78% | 0.93170 | 32.66 | 0.08395 | PASS |
| `sol_dense_steps_15` | dense steps | 136.73 | 1.042x | 4.05% | 0.95129 | 35.10 | 0.05834 | PASS |
| `sol_dense_steps_20` | dense steps | 142.73 | 0.998x | -0.16% | 0.95825 | 35.60 | 0.05441 | PASS |
| `sol_dense_layers_none` | dense layers | 129.26 | 1.102x | 9.29% | 0.94281 | 34.06 | 0.06195 | PASS |
| `sol_dense_layers_0` | dense layers | 129.42 | 1.101x | 9.18% | 0.94273 | 34.01 | 0.06271 | PASS |
| `sol_dense_layers_0_3` | dense layers | 130.55 | 1.091x | 8.38% | 0.94310 | 34.09 | 0.06346 | PASS |
| `sol_sink_0` | sink tokens | 129.49 | 1.100x | 9.13% | 0.94340 | 33.94 | 0.06579 | PASS |
| `sol_sink_256` | sink tokens | 129.42 | 1.101x | 9.18% | 0.94189 | 34.00 | 0.06479 | PASS |
| `sol_sink_512` | sink tokens | 129.56 | 1.100x | 9.08% | 0.94225 | 33.99 | 0.06500 | PASS |
| `sol_kv_splits_auto` | KV splits | 129.76 | 1.098x | 8.93% | 0.94262 | 34.03 | 0.06458 | PASS |
| `sol_thresh_exact` | threshold | 128.69 | 1.107x | 9.69% | 0.92055 | 32.26 | 0.11685 | PASS |

## Quality gates

- SSIM >= 0.82
- PSNR >= 20.0 dB
- LPIPS <= 0.20

## Benchmark artifacts

The benchmark harness, raw tables, and generated report are archived in the
[vLLM-Omni rankings repository](https://github.com/lishunyang12/vllm-omni-rankings/tree/main/scripts/minimax_h3_sol_attn_sm120).
They are intentionally not part of the runtime feature patch.

## Limitations and incomplete cases

- Visual quality is measured on one prompt and one seed. The latency numbers
  use three measured repeats, but preset changes should receive a multi-seed
  confirmation before becoming defaults.
- SSIM, PSNR, and LPIPS cover video frames only; this run does not measure
  audio quality.
- `sol_kv_splits_2` and `sol_kv_splits_4` failed and are excluded from the
  table. `sol_kv_splits_auto` completed successfully.
- These measurements predate the review follow-up that recomputes sink query
  rows densely. Rebenchmark the final PR head before treating the reported
  latency and quality values as release claims.
- The report demonstrates the measured latency/quality tradeoff on this
  hardware and workload; it is not a cross-model or cross-hardware claim.
