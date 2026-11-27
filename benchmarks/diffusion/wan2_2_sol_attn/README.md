# Wan2.2 TI2V-5B Sol-Attn on Ascend 910B3

On a 50-step text-to-video request, Sol-Attn reduced DiT time from 385.05 s to 218.04 s (1.77× speedup) and generation request time from 419.22 s to 251.83 s (1.66×). Relative to FlashAttention, mean per-frame PSNR was 21.38 dB and SSIM was 0.8207.

## Integration

Wan uses the existing per-role attention configuration to opt video self-attention into `SOL_ATTN`. The NPU backend routes dense steps and layers to FlashAttention and calls `mindiesd.sol_attn` for the remaining layers. The tested provider implements diagonal thresholding. Set `tau` to adjust the threshold. [MindIE-SD PR 929](https://gitcode.com/Ascend/MindIE-SD/pull/929)

The Sol backend caches Morton indices by video grid and device. When a forward uses Sol, Wan reorders the video tokens, rotary embeddings and per-token timestep conditioning together before the DiT block stack, then restores the original token order after the stack. Each block gets its own attention metadata.

## 50-step comparison

Both runs used one Ascend 910B3, identical BF16 weights and prompt, 1280 × 704 resolution, 121 frames, 24 fps, UniPC, CFG 5, flow shift 5 and seed 1024. With `tau=1.0`, `dense_steps=4` and `dense_layers=1`, steps 1–4 used FlashAttention throughout. On steps 5–50, layer 1 used FlashAttention and layers 2–30 used Sol. Each timing is one observation on a shared host. [Reproduction commands](../../../docs/user_guide/diffusion/attention_backends/sol_attn.md)

| Stage | FlashAttention | Sol diag |
| --- | ---: | ---: |
| Load weights | 19.68 s | 22.87 s |
| Encode prompt and negative prompt | 0.137 s | 0.138 s |
| DiT (100 forwards) | 385.05 s | 218.04 s |
| Denoising loop including DiT | 385.47 s | 218.43 s |
| VAE decode | 30.22 s | 30.07 s |
| Generation request after warm-up | 419.22 s | 251.83 s |
| MP4 export | 6.06 s | 6.61 s |
| Launch through MP4 export, including engine warm-up | 528.06 s | 371.79 s |

PSNR and SSIM use the 121 uncompressed RGB frames. A separate run collected routing counts so device synchronization for those counters did not affect the reported timings. Of 3,000 self-attention calls, 2,668 used Sol and 332 used FlashAttention. The fraction of query/key block pairs using the pooled approximation was **68.98% within Sol calls** and **61.34% across all calls**. [Full prompt, timings, quality metrics and routing counts](https://github.com/Ea510chan/vllm-omni/blob/55cbafe7dda38f9a41d12595b993faabc8e1c9ac/review/wan2.2-sol-attn/morton-50step/summary.json) · [Generated video comparison (dense left, Sol right)](https://github.com/Ea510chan/vllm-omni/blob/55cbafe7dda38f9a41d12595b993faabc8e1c9ac/review/wan2.2-sol-attn/dense-vs-diag.mp4)

## Dense-step sweep and full dense fallback

The 11-point sweep varied the dense prefix from 0 to 50 in five-step increments. With 30 dense steps, the measured request took 365.23 s with mean PSNR 37.32 dB and SSIM 0.9822 against the sweep's 414.77 s Flash reference. With all 50 steps dense, the request took 415.15 s and the uncompressed frames matched Flash exactly. The 10-, 30- and 45-step timings were repeated on card 6 after other NPU processes were observed during the original runs. The default four-step measurement above used its own 419.22 s Flash reference.

[Complete results, stage timings, occupancy samples, plots and generated videos](https://github.com/Ea510chan/vllm-omni/tree/c78b9bcbbfac6c6daa0bc2c6fddc0e8e89248b9d/review/wan2.2-sol-attn/dense-step-sweep)

## Validation

The focused suite passed 168 tests at feature revision `758442d34` with MindIE-SD revision `69852419`. Unit tests cover opt-in configuration, step and layer boundaries, provider arguments, Morton cache reuse, the inverse permutation and per-token alignment. The two-step offline NPU case crosses the dense-to-Sol boundary and checks the shape, finite values and pixel variation of the generated 17-frame video. [Unit tests](../../../tests/diffusion/models/wan2_2/test_wan22_sol_attn.py) · [Offline NPU test](../../../tests/e2e/offline_inference/test_wan22_t2v.py) · [Test results](https://github.com/Ea510chan/vllm-omni/blob/55cbafe7dda38f9a41d12595b993faabc8e1c9ac/review/wan2.2-sol-attn/morton-move-tests.log)
