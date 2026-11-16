# H3 reshape: batch, temporal chunks and tile-count coverage

This local campaign checks 30 representative configurations of the balanced atomic native-group repair against archived upstream main. It does not claim an exhaustive proof over all possible inputs.

Baseline: `b7bbfbdb4f2b3c070d9632043ea6cd2b1c601270`. Published repair: `5ba12c664a610761268f1b1c34904aea28414a70`. Starting local candidate module SHA256: `9995e9a6f8b023ac58a85655591b3a71d230b7bb0f4c7ec20fcc803ecc420231`. The earlier fixed-case native-group result is separate; no previous timing sample is imported into this campaign.

Same verified nightly image software and original full Ref2VA Video VAE checkpoint. Inputs are derived reproducibly from the fixed captured Float32 latent by periodic extension/crop and a fixed per-sample offset. They are identical between paired arms. This is resident VAE decode validation; DiT, encode, HTTP and media codec execution are excluded.

Candidate wrappers are installed and restored around each arm on the same pinned main adapter and resident weights. Two independent eight-worker launches use reversed per-case arm order. Each arm has a complete Float32/stream-RGB byte gate, five warmups and five formal synchronized requests. Plain-decode RGB is saved outside the timed interval and compared. Every byte, including signed zero, is checked. No formal timing is accepted after a failed numerical gate. CPU comparison and regression helpers do not overlap formal NPU timing.

Optimization activity and precision are separate requirements: a fallback with exact output is recorded as fallback, and a configuration without a measurable gain is recorded without inventing a speedup.

## Matrix

| Configuration | Sample batch | Latent T | Chunks | Latent H×W | Tile size | Tile limit | Native stack |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| s-b1-c2-n15 | 1 | 12 | 2 | 54×30 | 256 | 4 | True |
| s-b1-c5-n15 | 1 | 27 | 5 | 54×30 | 256 | 4 | True |
| s-b2-c2-n15 | 2 | 12 | 2 | 54×30 | 256 | 4 | True |
| s-b2-c5-n15 | 2 | 27 | 5 | 54×30 | 256 | 4 | True |
| s-b3-c2-n15 | 3 | 12 | 2 | 54×30 | 256 | 4 | True |
| s-b3-c5-n15 | 3 | 27 | 5 | 54×30 | 256 | 4 | True |
| s-b4-c2-n15 | 4 | 12 | 2 | 54×30 | 256 | 4 | True |
| s-b4-c5-n15 | 4 | 27 | 5 | 54×30 | 256 | 4 | True |
| s-b1-c3-n8 | 1 | 17 | 3 | 100×12 | 256 | 4 | True |
| s-b1-c3-n9 | 1 | 17 | 3 | 32×32 | 256 | 4 | True |
| s-b1-c3-n16 | 1 | 17 | 3 | 44×44 | 256 | 4 | True |
| s-b1-c3-n28 | 1 | 17 | 3 | 54×30 | 192 | 4 | True |
| s-b1-c3-n32 | 1 | 17 | 3 | 100×44 | 256 | 4 | True |
| s-b2-c3-n9 | 2 | 17 | 3 | 32×32 | 256 | 4 | True |
| s-b3-c3-n16 | 3 | 17 | 3 | 44×44 | 256 | 4 | True |
| s-b4-c3-n28 | 4 | 17 | 3 | 54×30 | 192 | 4 | True |
| s-b1-c1-n15 | 1 | 7 | 1 | 54×30 | 256 | 4 | True |
| s-b1-c3-n15 | 1 | 17 | 3 | 54×30 | 256 | 4 | True |
| s-b1-c17-n15 | 1 | 87 | 17 | 54×30 | 256 | 4 | True |
| s-b1-c21-n15 | 1 | 107 | 21 | 54×30 | 256 | 4 | True |
| s-b1-c5-n15-k2 | 1 | 27 | 5 | 54×30 | 256 | 2 | True |
| s-b1-c5-n15-k3 | 1 | 27 | 5 | 54×30 | 256 | 3 | True |
| s-b1-c5-n15-k8 | 1 | 27 | 5 | 54×30 | 256 | 8 | True |
| s-b1-t18-n15-padding | 1 | 18 | 4 | 54×30 | 256 | 4 | True |
| u-b1-c3-n9 | 1 | 17 | 3 | 32×32 | 256 | 4 | False |
| u-b2-c3-n15 | 2 | 17 | 3 | 54×30 | 256 | 4 | False |
| u-b3-c2-n16 | 3 | 12 | 2 | 44×44 | 256 | 4 | False |
| u-b4-c2-n15 | 4 | 12 | 2 | 54×30 | 256 | 4 | False |
| u-b1-c17-n15 | 1 | 87 | 17 | 54×30 | 256 | 4 | False |
| u-b1-c3-n28 | 1 | 17 | 3 | 54×30 | 192 | 4 | False |

CPU checks: 260 planning/projection tests and 2 distributed native-group identity/order checks passed. Exact logs are under `logs/`; successful exit artifacts are under `state/`.

All final numerical and timing checks completed; see the paired results below.

## Local repair and executed precision evidence

The starting w1-only balanced native-group prototype failed 14 of the 30 new configurations. The second revision, preserving attention output projection shapes and native temporal padding, failed 11. Adding the original final output projection calls reduced failures to one. The final revision also preserves QKV and FFN contraction calls in the unstacked path; all 30 complete Float32 and streamed-RGB byte gates passed before formal timing.

Full-weight same-card traces identify shape-dependent outputs with identical inputs at `attn.to_out`, then `decoder.proj_out`. The remaining unstacked 192px tile sequence first differs at `attn.to_qkv`, then `ff.w2` after restoring QKV. See `results/trace-v1.json`, `results/trace-v2.json`, and `results/trace-unstack-v3.json`. These are numerical observations; no claim is made about a particular closed CANN kernel's internal accumulation algorithm.

For a native group G with n_G tiles and original sample batch B, the protected Linear call shape is `(n_G*B, S, D)` when native stacking is enabled, and `(B, S, D)` for each tile when native stacking is disabled.

In real arithmetic, with L(X) = X W^T + b:

L(cat_b(X_1,...,X_K)) = cat_b(L(X_1),...,L(X_K)).

The repair preserves the right-hand-side native GEMM group shapes and member order. FFN expansion and attention output projectors write independent results into disjoint output slices in the native-stack path; the final output projection retains its original Linear calls and concatenates their outputs. No output correction, dtype widening, quantization change, or tolerance is used. In the unstacked path, w1, w2, QKV, attention output, and final output projections retain their original sample batches.

The time plan copies the same last latent token as native `decode_temporal`: p = (-(T + token_drop)) mod tokens_chunk_size, T_padded = T + p. Native output frame trimming and temporal blending remain. No floating-point arithmetic is added by the padding copies.

Native stack plans without fewer waves or a lower maximum tile load retain the original path. This is a structural guard, not a runtime speedup guarantee for every other plan. Every observed positive or negative formal sample will be retained.

Final candidate module SHA256: `5f5a46633c84803f9cfcfcd998a6a4bbaedf5d11f41587da7e1b5997d2ea8425`. 278 CPU planning, projection, padding, group-member/order and restoration tests pass. `native-groups-coverage.patch` replays on public repair 5ba12c6 with forward GNU patch, fuzz=0, producing the exact candidate bytes.

## Final paired results

All 30 full configurations are byte-exact in both independent launches. Float32 video, streaming RGB and plain-decode RGB match the paired baseline; every arm also matches its own independent repeat, including signed zero. All differing-byte counts, maximum numerical errors and RMSE values in the full-output comparisons are zero.

Each arm has ten formal samples per configuration: five from each independent launch, after five warmups. 600 formal samples are retained. Timing is synchronized maximum-rank resident video decode. Positive percentages mean less latency; negative percentages mean slower execution. No sample is excluded.

24 configurations execute the optimization and six keep the original native path. Nine active configurations have a positive observed pooled reduction; the remaining fifteen active configurations have nonpositive observed reduction. Small fractions of a percent are reported as observations rather than a universal speedup claim.

| Case | Baseline mean ± std (s) | Candidate mean ± std (s) | Latency reduction | Active | Bit-exact |
| --- | ---: | ---: | ---: | --- | --- |
| s-b1-c2-n15 | 0.449156 ± 0.000614 | 0.452229 ± 0.000826 | -0.684% | True | True |
| s-b1-c5-n15 | 1.088280 ± 0.000630 | 1.091419 ± 0.000571 | -0.288% | True | True |
| s-b2-c2-n15 | 0.867777 ± 0.001808 | 0.888028 ± 0.000942 | -2.334% | True | True |
| s-b2-c5-n15 | 2.129977 ± 0.001904 | 2.202465 ± 0.003085 | -3.403% | True | True |
| s-b3-c2-n15 | 1.337537 ± 0.001117 | 1.338204 ± 0.002003 | -0.050% | True | True |
| s-b3-c5-n15 | 3.315209 ± 0.003422 | 3.319810 ± 0.002250 | -0.139% | True | True |
| s-b4-c2-n15 | 1.759370 ± 0.001198 | 1.790928 ± 0.001085 | -1.794% | True | True |
| s-b4-c5-n15 | 4.369395 ± 0.006148 | 4.439178 ± 0.001730 | -1.597% | True | True |
| s-b1-c3-n8 | 0.285033 ± 0.000403 | 0.241886 ± 0.000401 | 15.138% | True | True |
| s-b1-c3-n9 | 0.650462 ± 0.000393 | 0.446995 ± 0.000178 | 31.280% | True | True |
| s-b1-c3-n16 | 0.667484 ± 0.001819 | 0.669115 ± 0.000237 | -0.244% | True | True |
| s-b1-c3-n28 | 0.602168 ± 0.000702 | 0.602915 ± 0.000774 | -0.124% | False | True |
| s-b1-c3-n32 | 1.309789 ± 0.001748 | 1.311550 ± 0.001557 | -0.134% | False | True |
| s-b2-c3-n9 | 1.270398 ± 0.001515 | 0.896399 ± 0.000741 | 29.440% | True | True |
| s-b3-c3-n16 | 2.012614 ± 0.000680 | 2.003265 ± 0.001378 | 0.465% | True | True |
| s-b4-c3-n28 | 2.435896 ± 0.002647 | 2.439313 ± 0.004021 | -0.140% | False | True |
| s-b1-c1-n15 | 0.236464 ± 0.000460 | 0.237620 ± 0.000525 | -0.489% | False | True |
| s-b1-c3-n15 | 0.662554 ± 0.000413 | 0.664882 ± 0.000270 | -0.351% | True | True |
| s-b1-c17-n15 | 3.646732 ± 0.001946 | 3.452430 ± 0.007681 | 5.328% | True | True |
| s-b1-c21-n15 | 4.495679 ± 0.001197 | 4.302758 ± 0.010329 | 4.291% | True | True |
| s-b1-c5-n15-k2 | 1.087582 ± 0.001303 | 1.089131 ± 0.001386 | -0.142% | False | True |
| s-b1-c5-n15-k3 | 1.088332 ± 0.000571 | 1.088960 ± 0.000694 | -0.058% | False | True |
| s-b1-c5-n15-k8 | 1.087603 ± 0.001343 | 1.118381 ± 0.001540 | -2.830% | True | True |
| s-b1-t18-n15-padding | 0.875191 ± 0.000778 | 0.881323 ± 0.001091 | -0.701% | True | True |
| u-b1-c3-n9 | 0.719690 ± 0.000473 | 0.468768 ± 0.000256 | 34.865% | True | True |
| u-b2-c3-n15 | 1.300141 ± 0.000334 | 1.407694 ± 0.003435 | -8.272% | True | True |
| u-b3-c2-n16 | 1.268926 ± 0.001830 | 1.420022 ± 0.000900 | -11.907% | True | True |
| u-b4-c2-n15 | 1.719683 ± 0.000420 | 1.894326 ± 0.000704 | -10.156% | True | True |
| u-b1-c17-n15 | 4.075196 ± 0.012014 | 3.640769 ± 0.009075 | 10.660% | True | True |
| u-b1-c3-n28 | 0.834774 ± 0.002309 | 0.634133 ± 0.002830 | 24.035% | True | True |

### Important performance limits

The fixed native-stack B=1, 17-chunk, 15-tile case retains a 5.328% resident-decode reduction. Its 21-chunk variant retains 4.291%. Three-chunk native-stack cases with eight and nine tiles reduce latency 15.138% and 31.280%; B=2 with nine tiles reduces 29.440%. Unstacked B=1 cases retain 34.865% at nine tiles, 24.035% at 28 tiles, and 10.660% on the 17-chunk 15-tile input.

A general positive latency result is not established. Unstacked B=2/3/4 cases in this matrix increase latency by 8.272%, 11.907%, and 10.156%. Some native-stack short inputs increase latency by roughly 0-3.4%. Keeping reference GEMM groups protects precision but can reduce batching efficiency, and requests without enough imbalance or batching benefit need not accelerate. The feature remains opt-in.

A routed native case is exact but is not an active optimization success. This includes fully packed 28/32-tile native stacks, the single-chunk 15-tile case, and the native-stack tile limits two and three for the five-chunk fixture. Their small measured deviations from baseline do not establish a gain.

### Coverage and reproducibility

This is a representative matrix, not a proof over infinitely many sizes. NPU-tested original sample batches are 1,2,3,4; tile counts are 8,9,15,16,28,32; chunk counts include 1,2,3,5,17,21; tile batch limits are 2,3,4,8; tile sizes are 192/256 pixels, including nonsquare 256×192 effective tiles. One unaligned latent temporal length exercises original native token padding and trimming. Each request uses a fixed spatial grid and equal temporal clip shapes after native padding. Ragged resolutions inside one request, training, isolated first/last frames and internal decoder spatial parallelism are not established by this matrix.

Software: eight Ascend 910B4-1 64GB cards, driver 25.5.0; verified quay.io/ascend/vllm-omni:nightly software, Python 3.12.16, torch 2.10.0+cpu, torch_npu 2.10.0.post7, vLLM 0.33.0, CANN 9.2.1. Same image digest as state/contract.json. Original weights: the original full MiniMax-H3 Ref2VA checkpoint. Captured input SHA256 was independently rechecked. Image software runs natively with the host NPU driver, as in the prior campaign.

The matrix uses the pinned main VAE adapter and identical resident weights. Candidate wrappers are installed/restored request-locally around each arm. This avoids weight or process-loading differences; arm order is reversed in the second independent worker launch. No CPU regression, precision comparison, repository scanner or profile helper overlaps formal NPU timing. Precision copies, file I/O and metadata collection are outside the timed interval.

All operator parameters, original input bits and output coordinates are preserved. No MP4 codec equivalence is claimed. Audio and DiT are outside this video-only comparison.

### Delivered code and evidence

`sources/candidate/vae_reshape.py` is the exact runtime-tested source; `native-groups-coverage.patch` replays its bytes on public repair 5ba12c6. `sources/review/vllm_omni/diffusion/models/minimax_h3/vae_reshape.py` is a Ruff-formatted review copy with identical executable AST; `native-groups-coverage-formatted.patch` replays that copy with GNU patch fuzz=0. Their separate SHA256 values and AST identity are recorded in state/review-source.json. No model behavior changes during formatting.

`results/summary.json` contains all per-rank activity counters, complete comparisons, per-launch rows, all 600 timing samples, source hashes and configuration parameters; `results/summary.csv` contains the compact table. Exact launches and successful exits are in state/job-*.json. `scripts/matrix.py`, `scripts/controller.py`, and `scripts/analyze_matrix.py` reproduce the experiment. Failed initial variants and their numerical traces are retained, and no timing from them is included in the final result.

This report describes the validated native-group repair replacing the older w1-only precision repair in PR #10256.

## Frozen OPT compatibility and repair cost

The same formatted ordinary-Linear repair was separately tested under frozen OPT 0.26, source 9693e4d3805720eb9fcc53ee6b17c3c21a5bf92f, Omni 0.26.1.dev0+g0e4b28f, vLLM 0.26.0+empty, Python 3.12.13, torch 2.10.0+cpu, torch-npu 2.10.0.post2, CANN 9.0.1, driver 25.5.0, eight 910B4-1 64GB cards. All 144 W8A8 projectors retained NZ format 29; RMSNorm/RoPE/SwiGLU fusion counts were 144/36/36 and packed staging stayed enabled.

Both fixed real avatar12s inputs use sample B=1, native stack enabled, tile batching enabled, K=4 and cross-chunk requested. 480P is latent [1,24,87,54,30], output 864x480, 17 chunks and 15 tiles. 768P is [1,24,87,86,48], output 1376x768, 17 chunks and 28 tiles. Two independent launches reverse arm/resolution order; five warmups and ten formals per arm/launch, twenty samples per cell. Precision requests are untimed.

| Resolution | Before mean ± std (s) | After mean ± std (s) | ΔT (s) | Latency increase |
| --- | ---: | ---: | ---: | ---: |
| 480p | 2.657499480 ± 0.000810766 | 2.653005970 ± 0.002914625 | -0.004493511 | -0.169088% |
| 768p | 5.000924373 ± 0.000989389 | 5.653384447 ± 0.002668109 | +0.652460074 | +13.046789% |

480P retains eight forwards and is approximately unchanged. 768P changes from 15 cross-chunk waves to 17 native forwards (groups of 4/3), reports native-group-no-gain, and forfeits about 0.65 seconds of the original cross-chunk benefit. This regression is intentional native-group preservation, not a universal speedup. The OPT fixed inputs are already bit-exact before repair; all complete pre-codec Float32/RGB outputs and independent repeats are bit-exact, including signed zero.

A separate W8A8-aware research variant expanded projector selection and called the original quantized forward for each group; it did not replace INT8 arithmetic with Float16 addmm. This variant is not included in this commit. Its separate paired results were:

| Resolution | Before mean ± std (s) | After mean ± std (s) | ΔT (s) | Latency increase |
| --- | ---: | ---: | ---: | ---: |
| 480p | 2.658858109 ± 0.001927589 | 2.816367102 ± 0.002671543 | +0.157508993 | +5.923934% |
| 768p | 5.004974318 ± 0.004006088 | 5.653223968 ± 0.003270042 | +0.648249650 | +12.952107% |

The ordinary repair selects only the remaining final nn.Linear under OPT; block projectors remain W8A8Linear. The expanded variant costs about 0.15751 seconds at 480P. All 200 OPT formal samples across both variants and the separate native reference were retained; all six successful NPU jobs exited zero. No E2E or encoded-media equivalence is claimed.

Normal OPT serving imports reproduced corrupted-size teardown failures of unestablished native allocator cause. Successful component tests used frozen genuine VAE/quantization/fusion/staging code with actual Torch NPU stream/event and HCCL interfaces and minimal serving-only bridges. Failed probes are excluded. Full serving startup, E2E performance, CUDA and peak-memory claims are outside this evidence.

## Public reproduction

Use the checked-in `minimax_h3_vae_decode.py` benchmark and `minimax_h3_vae_precision.py` full-byte comparator with the original task-partition checkpoint and saved `(video_latent, audio_latent)` inputs. Set `stack_tiling` in an isolated copy of the checkpoint configuration for the stacked versus unstacked experiment; preserve the selected tile geometry and sample batch. Run each configuration in two independent launches, reverse arm order, and compare complete outputs before accepting performance. The matrix and OPT campaigns use separate direct component runners and their resident streamed-decode timings must not be equated with this benchmark's plain-decode timing.

```bash
H3_VIDEO_DECODER_CROSS_CHUNK=0 python3 -m torch.distributed.run --nproc_per_node=8 benchmarks/diffusion/minimax_h3_vae_decode.py --arm baseline --model /path/to/Ref2VA --inputs /path/to/inputs.pt --output-dir baseline-results --warmups 5 --formals 5 --precision
H3_VIDEO_DECODER_CROSS_CHUNK=1 H3_VIDEO_DECODER_TILE_BATCHING=1 H3_VIDEO_DECODER_TILE_BATCH_SIZE=4 python3 -m torch.distributed.run --nproc_per_node=8 benchmarks/diffusion/minimax_h3_vae_decode.py --arm candidate --model /path/to/Ref2VA --inputs /path/to/inputs.pt --output-dir candidate-results --warmups 5 --formals 5 --precision
python3 benchmarks/diffusion/minimax_h3_vae_precision.py --baseline-dir baseline-results --candidate-dir candidate-results --output-json precision.json
python3 -m pytest -q tests/diffusion/models/minimax_h3/test_minimax_h3_vae_cross_chunk_tiling.py tests/diffusion/models/minimax_h3/test_minimax_h3_vae_native_groups.py
```

AI assistance: TRAE CLI implemented the repair, regression coverage, local NPU validation and report preparation.
