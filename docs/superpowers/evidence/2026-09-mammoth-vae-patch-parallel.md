# MammothModa2 VAE patch-parallel: RTX-3090 validation

Date: 2026-09-19. This includes real-weight VAE-only measurements, a two-rank stage smoke test, and complete AR→DiT→VAE requests with PP=1 and PP=2 on a four-GPU host. The complete requests establish functionality, not an end-to-end speedup.

## Source and hardware

- Base: vLLM-Omni `4c7a98c26f6167a219b008263773c959a65ac0f2` (contains #7134), plus the `codex/mammoth-vae-pp` registry/Mammoth changes now committed as `c424f8aec` after a DCO sign-off rewrite (formerly `129a4d7258040229442205a544bcce1b7e58df3d`). The rewrite changed commit metadata, not source content. Older benchmark JSONs were captured before that commit and record the then-current modified-file SHA-256 values; the base commit alone does not identify the tested code.
- VAE config and weights: `bytedance-research/MammothModa2-Preview`, config revision `ef5a5e41dbf0de1ef6275586b7580f0d4248b4c6`; all 244 `gen_vae.*` tensors were loaded from `model-00006-of-00008.safetensors` with a strict state-dict check. The AR and DiT weights were not loaded.
- 2 × GeForce RTX 3090, 24 GiB, SM 8.6; driver 570.124.04. `nvidia-smi topo -m` reports `SYS` between the GPUs (cross-NUMA, no NVLink).
- PyTorch 2.13.0+cu129, vLLM 0.29.0+cu129, Diffusers 0.40.0, Transformers 5.14.1. FP16 unless a row says otherwise.
- Latents are deterministic Gaussian samples (seed 7), transformed by the checkpoint's `scaling_factor=0.3611` and `shift_factor=0.1159`. They are not sampled DiT latents; image outputs therefore cannot establish semantic image quality.

## 2048 × 2048 output, FP16, real VAE weights

Runs 1 and 3 used two warmups and five timed iterations; runs 2 and 4 used one warmup and three timed iterations. Run 4 predates the tile-boundary instrumentation below. Timing is the maximum synchronized per-rank wall time and includes distributed collectives and rank-0 stitching, but excludes weight loading and warmup.

| Run | 1 GPU untiled median | 1 GPU tiled median | 2 GPU tiled median | 2 GPU vs 1 GPU tiled | Relative L2 vs 1 GPU tiled |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1227.6 ms | 1713.4 ms | 966.3 ms | 1.77× | 0.0528% |
| 2 | 1229.0 ms | 1722.0 ms | 979.4 ms | 1.76× | 0.1092% |
| 3 | 1230.4 ms | 1705.1 ms | 970.3 ms | 1.76× | 0.0590% |
| 4 | 1230.4 ms | 1702.7 ms | 1013.0 ms | 1.68× | 0.0593% |

Run 1 peak allocated memory: untiled single GPU 10.41 GB; tiled single GPU 2.81 GB; two-GPU tiled 2.82 GB on rank 0 and 2.74 GB on rank 1. Thus the large per-GPU memory reduction versus untiled decode is caused by tiling, **not** by patch parallelism. The evidence for PP itself is approximately 1.68–1.77× faster decode at comparable per-GPU tiled memory across four launches; warmup/iteration counts differ.

The PP=2 output differs from single-GPU tiled output by a mean absolute value of `3.8e-5` to `1.1e-4` across runs. The older raw records label a center stripe as a "join"; this is **not** a valid seam test at 2048px because the executor uses multiple tiles rather than one two-half split. No seam-quality claim is made from those records. Repeat decodes within a mode also showed nonzero FP16 numerical variation on some runs, so cross-mode error should be interpreted against that noise floor. PP=2 versus *untiled* output has larger relative L2 error (~0.57%), almost entirely matching the single-GPU tiled-versus-untiled difference (~0.57%).

## Actual tile-boundary check, FP16, real VAE weights

The updated benchmark obtains the grid from `vae.tile_split(latents)` and the tile-to-rank mapping from the executor's `_balance_tasks`; it does not infer joins from the image center. Each horizontal/vertical output boundary is `(index + 1) × row_limit`, clipped to the output shape. A ±16-pixel strip around each boundary segment is classified as cross-rank or same-rank according to the two adjacent tiles; overlapping strip pixels count as cross-rank. The remaining pixels are interior. Both new runs used one warmup and three timed iterations with the same 244 VAE weight tensors and seed 7. The benchmark SHA-256 is `7b42018a…` in both raw records.

| Latent → output | Tiled PP=1 / PP=2 median | PP=2 vs tiled relative L2 | Cross-rank / same-rank / interior mean abs error vs tiled PP=1 |
| --- | ---: | ---: | ---: |
| 256² → 2048² | 1720.6 / 980.4 ms (1.76×) | 0.0580% | `4.37e-5` / `4.96e-5` / `4.17e-5` |
| 257² → 2056² | 1731.0 / 977.8 ms (1.77×) | 0.0849% | `8.98e-5` / `5.48e-5` / `7.94e-5` |

Both shapes produced the expected full output size. The actual tile grid is 3×3 with `row_limit=768`, so the horizontal and vertical joins are at output pixels 768 and 1536; the rank assignment is recorded in each JSON file. These two random-latent samples do **not** prove natural-image quality or statistical seam equivalence; the strips measure PP=2 versus the same tiled algorithm at PP=1, not tiled versus untiled reconstruction quality.

## Smaller shapes and BF16 investigation

- 1024 × 1024 FP16, real weights, two warmups/seven iterations: untiled PP=1 median 252.4 ms; PP=2 patch path median 189.7 ms. Peak allocated on rank 0: 2.74 GB → 2.11 GB. Relative L2 versus untiled: 0.277%. The vertical-join stripe error (`0.000938`) matched the away-region error (`0.000929`).
- The initial 1024² BF16 single-GPU control drifted by 1.2–1.8% relative L2 between repeated decodes. At latent size 128, Diffusers does not enter tiled decode (`height > tile_latent_min_size` is false). A separate four-mode diagnostic using the original Diffusers `AutoencoderKL` and the distributed subclass showed identical outputs when the CUDA allocator cache was retained, but drift in **both** classes when `torch.cuda.empty_cache()` was called between modes. Removing the peak-memory reset alone did not remove it. This implicates allocator-sensitive BF16 execution on this RTX 3090 setup, not a patch-parallel-only code path. It does not prove that every BF16 deviation has the same cause.
- The benchmark now retains the cache by default, while `--clear-cache-between-modes` enables the old behavior for allocator studies. At 1024² BF16, PP=1 untiled and PP=1 `use_tiling=True` were bit-identical and internally repeatable. PP=2 was 192.5 ms versus 255.4 ms (1.33×), but its repeated output still had 0.362% relative L2 drift and differed from PP=1 by 0.679% relative L2. Thus 1024² BF16 is **not** a clean parity result.
- At 2048² BF16 with cache retained, PP=1 tiled and PP=2 tiled were bit-identical in this launch and each repeated exactly. Medians were 1733.7 ms and 985.4 ms (1.76×); all measured cross-rank/same-rank/interior tile-boundary errors versus tiled PP=1 were zero. PP=1 untiled was not internally repeatable (0.835% relative L2), so comparison to untiled is not a patch-parallel acceptance metric. Peak *allocated* memory was 2.81 GB for tiled PP=1 and 2.79/2.77 GB for PP=2. Peak *reserved* memory cannot be compared across retained-cache modes.

## Two-rank stage integration smoke

With `sequence_parallel_size=ulysses_degree=vae_patch_parallel_size=2`, the actual registry constructed `MammothModa2DiTPipeline` and a `DistributedAutoencoderKL`, enabled VAE tiling, and strictly loaded all 244 real `gen_vae.*` tensors. A tiny synthetic DiT forward with random/unloaded DiT weights ran on both ranks under the real forward context and produced finite `[1,16,8,8]` outputs (maximum rank-to-rank absolute difference `9.77e-4`). A real-weight VAE decode produced a finite `[1,3,1024,1024]` result on rank 0 and the expected empty dummy result on rank 1. The smoke also executed the **complete DiT-stage `pipeline.forward`** for one denoising step using synthetic AR hidden states: rank 0 returned a finite `[1,3,64,64]` output, rank 1 returned the expected empty dummy. It does not load real AR or DiT weights or establish image quality. Peak allocation here includes both DiT and VAE and is not a VAE-only memory measurement.

The integration run exposed two configuration hazards. Mammoth has no DiT `_sp_plan`, yet the two-rank stage needs an SP group to coordinate VAE patch parallelism. Its shared attention must therefore opt out of sequence-parallel collectives; otherwise full, replicated DiT inputs could be treated as sequence shards. A dedicated regression test failed before `skip_sequence_parallel=True` was added to Mammoth attention and passed afterward. Also, different ranks can finish denoising with slightly different latents: the random/unloaded tiny DiT smoke produced a `9.77e-4` maximum absolute rank difference. The pipeline now broadcasts rank 0's final latent before tile decode. In a two-rank test deliberately perturbing rank 1's VAE latent, the maximum inter-rank difference fell from `0.5039` to exactly zero before the real-weight distributed decode. The registry still logs the generic "SP hooks not applied" warning; for this VAE-only use of the group, that warning is expected.

After adding native output shape/dtype/finiteness checks to the benchmark, a final FP16 2048² run produced `torch.float16` `[1,3,2048,2048]` on each measured mode, with PP=1 tiled median 1721.2 ms and PP=2 tiled median 1009.3 ms (1.71×). Tiled PP=1 and PP=2 outputs were bit-identical in this launch. This run includes a SHA-256 of the Mammoth attention implementation alongside the registry, pipeline and benchmark hashes.

## Three-GPU full-model and PXB-link checks

On a second host with 3 × RTX 3090 24 GiB (driver 595.71.05, PyTorch 2.13.0+cu130, vLLM 0.29.0), all eight pinned Preview model shards were downloaded and their SHA-256 values checked against their upstream LFS OIDs. AR TP=2 on GPUs 0+1 and DiT/VAE on GPU 2 loaded the full real checkpoint. The repository's offline T2I example generated a valid 512×512 RGB image and then a 1280×1024 RGB image with the same prompt, seed, FP16 weights, and two denoising steps. The 1280×1024 request passed `--vae-use-tiling` and completed in 169.84 seconds end to end, including 166.89 seconds of AR generation and 2.80 seconds of DiT/VAE generation. A later registry audit found that the PP=1 `gen_vae` did **not** receive this explicit tiling setting in that source revision, so this is an *untiled* functional smoke, not tiled-decode evidence. Two denoising steps are not an image-quality validation. Experiment-only stage placement is in [the baseline YAML](mammoth-3gpu-baseline.yaml). This host required `DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA` because its installed FA3 wheel rejected FP16, and `VLLM_USE_FLASHINFER_SAMPLER=0` because its CUDA 11.8 `nvcc` cannot compile the installed FlashInfer sampler. These environment workarounds do not change the proposed VAE code.

On the same host's GPU 1↔2 `PXB` pair, a separate real-weight 2048² VAE-only benchmark (one warmup, three timed iterations) measured untiled PP=1 median 1229.4 ms, tiled PP=1 median 1751.2 ms, and tiled PP=2 median 1001.9 ms: **1.75× versus the like-for-like tiled baseline**. Native output shape/dtype/finiteness checks passed, and tiled PP=1 versus tiled PP=2 were bit-identical in this run, including all actual tile-boundary strips. Raw evidence is [here](mammoth-vae-2048-3gpu-pxb.json). The benchmark now tolerates a source-copy host without `.git`, records `VLLM_OMNI_SOURCE_COMMIT` when supplied, and still records SHA-256 of the relevant source files. The focused suite passed with 88 tests on this host.

Full-request PP=2 was attempted with AR TP=2 on GPUs 0+1 and DiT/VAE SP=2 on GPUs 1+2, using [the experimental PP=2 YAML](mammoth-3gpu-pp2.yaml). It failed during AR's second TP rank weight loading on shared GPU 1: the DiT process held 7.02 GiB, the AR process held 16.52 GiB, only 3.69 MiB remained, and a 130 MiB allocation failed. Repeating with the stage-1 `gpu_memory_utilization` lowered from 0.25 to 0.20 produced the same residency and failure. This is a 24 GiB colocation limit, not a demonstrated distributed VAE decode failure; no full-request PP=2 timing or image is claimed. The separate two-rank registry/DiT-forward/VAE-decode smoke above remains the PP=2 integration evidence.

## Four-GPU complete-request validation

The four-GPU host has 4 × RTX 3090 24 GiB, driver 570.124.04, PyTorch 2.13.0+cu129 and vLLM 0.29.0+cu129. GPUs 0+1 ran AR TP=2; GPU 2 ran PP=1 DiT/VAE, while GPUs 2+3 ran SP=2 DiT and PP=2 VAE. `nvidia-smi topo -m` reports `PXB` for the 2↔3 pair. The PP=2 request used source `565dd993532bebfa89b5efd2f9a30d2b5a022d08` (now `33324604e` after a sign-off-only rewrite). The corrected PP=1 request used the subsequent registry change committed as `ed1637810`; that change only enables explicit single-rank `gen_vae` tiling and leaves the PP=2 path unchanged. Pinned Preview weights, FP16, seed 42, prompt `A red fox in snow`, guidance 4, and VAE tiling were shared in the **corrected** comparison. Both PP=2 logs confirm tile splitting in both distributed DiT ranks. The focused suite passed 89 tests in the final environment. These are single paired requests, not statistically significant benchmarks.

| Request | PP=1 tiled total / AR / stage 1 | PP=2 tiled total / AR / stage 1 | Output comparison |
| --- | ---: | ---: | --- |
| 2048×2048, 20 denoise steps | 555.343 / 447.834 / 107.200 s | 543.023 / 435.624 / 107.123 s | AR token IDs identical; decoded RGB image bit-identical |

Both 2048² requests generated all 16,513 expected AR image tokens with `max_model_len=18432`; their identical AR token-ID sequence has SHA-256 `c7842b0278be06211ca6dc512d288f14cc1c85a134ee61bdeafd5999fa2ef4c1`. The `finish_reason='length'` here reflects the explicitly requested 16,513 tokens, not context truncation. Both output image files have SHA-256 `5cdde82c5c04e95de431ffa8c1b9dc772a3be5b2b6610dad4931aaf32f67e9d1`; direct RGB comparison has zero differing pixels. They are valid 2048×2048 images. AR hidden states were not persisted, so equal token IDs alone would not prove equal DiT inputs; the final-image parity is an observed result for this one request, not a general quality guarantee.

The matched 2048² configurations are [PP=1](mammoth-4gpu-2048-pp1.yaml) and [PP=2](mammoth-4gpu-2048-pp2.yaml). Earlier PP=1 runs passed `--vae-use-tiling` but, before the registry fix, `gen_vae.use_tiling` remained false. Thus the old 1280×1024 PP=1 versus PP=2 image difference and the old 2048² PP=1 image SHA-256 `176b1002…` compare **untiled versus tiled** output; they are excluded from the matched table. Separately, 2048² runs with `max_model_len=6144` stopped AR after 6,110 tokens and are invalid as full-grid comparisons. The corrected 2048² stage-1 times differ by only 0.077 s, while AR differs by 12.210 s. This paired run does **not** establish request-level latency improvement. Omni reported primary diffusion-worker peak **reserved** memory of 11,612 MiB (PP=1 tiled) and 11,674 MiB (PP=2 tiled); this is a stage-level observation and not necessarily the maximum across all workers. The 1280×1024 PP=2 smoke configuration is [here](mammoth-4gpu-pp2.yaml).

On the same four-GPU host's 2↔3 PXB pair, the final-code real-weight FP16 VAE-only 2048² run (one warmup, three timed iterations) measured untiled PP=1 median 1217.8 ms, tiled PP=1 1691.6 ms (range 1689.8–1698.6 ms), and tiled PP=2 957.5 ms (range 955.7–959.3 ms): **1.77× versus like-for-like tiled decode**. The two tiled outputs were bit-identical, including actual cross-rank tile boundaries; both modes repeated exactly. PP=1 tiled versus PP=2 peak *allocated* memory was approximately 2.78 GB on rank 0 versus 2.77/2.74 GB on ranks 0/1. The [final raw JSON](mammoth-vae-2048-4gpu-pxb-cu129-final.json) records source commit `ed1637810`, SHA-256 of the tested source files, timings, and boundary mapping. The [pre-fix run](mammoth-vae-2048-4gpu-pxb-cu129.json), which was not affected because the standalone benchmark sets VAE tiling directly, measured 1693.1 versus 950.5 ms (1.78×). Allocator-cache-retained peak *reserved* memory is not comparable across modes. This isolated decode benefit is compatible with the flat complete-request stage-1 result: the latter is dominated by 20 DiT steps and includes SP=2 overhead.

## Reproduction

On the two-GPU host, with the isolated environment and checkpoint shard already present:

```bash
cd /root/autodl-tmp/vllm-omni-vae-pp
PYTHONDONTWRITEBYTECODE=1 /root/autodl-tmp/vllm-omni-vae-pp-venv/bin/python \
  -m torch.distributed.run --standalone --nproc-per-node=2 \
  --module benchmarks.diffusion.bench_mammoth_vae_patch_parallel \
  --model-config /root/autodl-tmp/mammoth-vae-weights/config.json \
  --weights-shard /root/autodl-tmp/mammoth-vae-weights/model-00006-of-00008.safetensors \
  --latent-size 256 --warmup 1 --iterations 3 \
  --output /root/autodl-tmp/mammoth-vae-weights/real-2048-tile-boundaries.json
```

The exact [two-rank integration smoke script](mammoth-registry-vae-smoke.py) is also preserved here. Its model paths are specific to the experiment host; copy it there and run with `PYTHONPATH` pointing at this checkout and `python -m torch.distributed.run --standalone --nproc-per-node=2`.

Raw JSON copies are preserved beside this note: [run 1](mammoth-vae-2048-run1.json), [run 2](mammoth-vae-2048-run2.json), [run 3](mammoth-vae-2048-run3.json), [run 4](mammoth-vae-2048-current-source.json), [2048² tile-boundary run](mammoth-vae-2048-tile-boundaries.json), [2056² tile-boundary run](mammoth-vae-2056-tile-boundaries.json), [initial BF16 deterministic run](mammoth-vae-1024-bf16-deterministic.json), [1024² BF16 retained-cache run](mammoth-vae-1024-bf16-retain-cache.json), [2048² BF16 retained-cache run](mammoth-vae-2048-bf16-retain-cache.json), [final FP16 validator run](mammoth-vae-2048-fp16-final-validator.json), [three-GPU PXB-link FP16 run](mammoth-vae-2048-3gpu-pxb.json), [four-GPU PXB-link FP16 run](mammoth-vae-2048-4gpu-pxb-cu129.json), [final-code four-GPU run](mammoth-vae-2048-4gpu-pxb-cu129-final.json), and [two-rank stage smoke](mammoth-registry-vae-smoke.json). The older raw runs emitted non-standard `Infinity` for exact-match PSNR; their local copies normalize that value to JSON `null`. The BF16 retained-cache records use the preceding benchmark SHA-256 (`c824b64a…`); the final FP16 record uses the validator version (`e1296b52…`). Comparison images and full-model images remain on the experiment host and are not uploaded.

## Remaining gates

1. Do not claim general BF16 parity from one clean 2048² launch; 1024² PP=2 still has measurable drift. Repeat on another host/GPU if maintainers require a strict BF16 tolerance.
2. Complete Preview AR→DiT generation with PP=2 is now validated on four 3090s, but neither the 1280×1024 nor the 2048² single-pair stage timing establishes end-to-end acceleration. For a performance claim, collect repeated matched requests and isolate VAE decode within the stage, or optimize the DiT/SP path separately.
3. Review checkpoint-compatible weight loading and stage-level rank-0 output handling with maintainers. Keep standalone VAE slicing/tiling (#7435) outside this patch.
