# LingBot-World v2 NPU optimization series

Subsequent timestep cleanup removes the seven ordinary compute AICPU types
using FP64 host-prepared frequencies. Both optimized presets enable
`lingbot_npu_host_timestep`; HCCL is unchanged. See
[single-NPU precision and Level1 verification](AICPU_TIMESTEP_NPU.md).

This branch collects the measured NPU paths on base `7afd00e98` rather than
all experimental implementations. Nothing changes the default precision policy.
The checkpoint is the causal-fast 14B diffusers package. The measured workload
is case 03, 464x832, four DMD steps, seed 42, 16 chunks / 189 frames at 16 fps,
TP1 / Ulysses4, eager. Measurements use one warmup and three unprofiled sessions
per configuration plus an independent steady-state profile.

## Commit order and scope

| Order | Change | Control | Evidence and limits |
| --- | --- | --- | --- |
| 1 | Pass paged attention lengths from host metadata | `lingbot_npu_host_lengths` | Avoids device-to-host length reads. Its small standalone latency change was within run-to-run variation. |
| 2 | Native Q/K projection RMSNorm | `lingbot_npu_fused_rmsnorm` | TP1 only; normalizes the full 5120-D projection, not each 128-D head. Latest incremental whole-model latency reduction 5.5%, steady interval reduction 2.4%; output drift, quality unvalidated. |
| 3 | Cache block camera preparation and RoPE | `lingbot_cache_block_conditions` | Reuses conditioning within a block; resets with block state. Later isolated cache/padding campaign preserved the quantized control output. |
| 4 | Reuse staged history and refresh the partial boundary | KV `reuse_history_staging`, gather environment below | Reuses the unchanged prefix with session/signature invalidation. Later isolated whole-model latency reduction about 1.2%, including the same MXFP8 control. |
| 5 | Indexed KV writes | NPU dispatch, other devices unchanged | `IndexPutV2` becomes `index_copy_` / `ScatterUpdate` plus slicing. Whole-model RTF 2.5844 to 2.3964; all 189 decoded frames and MP4 bytes identical. |
| 6 | Selective online MXFP8 Linear | `lingbot_npu_online_quant: mxfp8` | Attention Q/K/V/O and FFN parallel linears only. Gates and camera-conditioning linears remain unquantized. Checks all 320 expected linears materialize. Earlier isolated whole-model latency reduction about 9.6%; quality unvalidated. |
| 7 | Fold spatial padding into VAE Conv3d | `lingbot_npu_vae_spatial_padding` | Unsharded VAE only, temporal cache semantics retained. Shape-specific equality check and fallback; earlier isolated whole-model reduction about 0.9%, identical output to its MXFP8 control. |
| 8 | Dense FP8 EagleFIA | `VLLM_OMNI_AR_DIFFUSION_FP8_FIA_MODE` | C8V16 tiling512, no sparse block selection. Latest incremental RTF 2.2642 to 2.0830, steady interval reduction 11.6%; includes online quantization and layout cost, quality unvalidated. |
| 9 | vLLM 0.32 scheduler compatibility | No performance switch | Keyword-only cached request data fixes the observed import failure with the tested vLLM version. |

The sequential four-card experiment has these unprofiled means:

| Configuration | Overall RTF | Steady interval (ms) |
| --- | ---: | ---: |
| MXFP8/cache/padding control, indexed writes off, native Q/K RMSNorm off | 2.584408 | 1803.68 |
| + indexed KV writes | 2.396355 | 1726.51 |
| + native Q/K RMSNorm | 2.264198 | 1685.36 |
| + FP8 EagleFIA | 2.082999 | 1489.90 |

Those four historical rows used the original Wan normalization plus an
experimental VAE epilogue, held fixed throughout. That numerical VAE restoration
and its fusion implementations are **not** included here. Thus the table
documents prior operator ablations, not a newly measured baseline for this
cleaned source snapshot. Do not add percentages from different campaigns.

## Profiles and output accuracy

The profiling window confirms indexed writes (800 `ScatterUpdate` calls),
native DiT RMSNorm (1200 calls), and dense EagleFIA (400 calls plus 1200
`DynamicBlockQuant` calls per rank). Conventional FA falls from 806 to 406
calls because cross attention and other unchanged attention paths remain BF16.
Quantized Linear has 2400 `QuantBatchMatmulV3` calls in the same window.
These are device-window counts, not end-to-end critical-path timings.

The first 16 rank captures yielded ten complete `kernel_details.csv` files.
FP8 host API correlation was incomplete, including in a four-rank recapture;
all 20 captures still had usable `op_statistic.csv` and `task_time.csv`.
The default-Wan VAE follow-up produced four complete kernel tables.

Indexed writes were bit exact. Native Q/K RMSNorm versus its control had video
PSNR 16.83 dB / SSIM 0.589; FP8 versus the preceding native-RMSNorm control
had PSNR 17.58 dB / SSIM 0.641. These compare decoded fixed-seed MP4 frames,
include codec effects, and are not ground-truth quality acceptance scores.
Online MXFP8 also changes output. No accuracy-changing option is enabled in
the conservative preset; the performance preset requires separate quality review.

## Default Wan VAE

This series keeps the framework's default Wan VAE implementation. It does not
restore a different normalization expression, replace the VAE checkpoint, or
install the experimental original-norm fusion. With the full DiT optimization
set, default Wan VAE gave steady intervals 1437.86 / 1450.26 ms in two cohorts
of three sessions. Overall RTF was 2.15731 / 2.03889 because mean first-chunk
latency changed from 4001 to 2436 ms. All six sessions aggregate to RTF 2.09810.
Steady improvement was 2.7%-3.5%; a stable overall speedup was not established.
The cleanup source itself has CPU contract checks, not a new whole-model NPU run.

## Running

Start from `configs/npu_usp4_conservative.yaml` for precision-preserving options.
For measured speed candidates use `configs/npu_usp4_fp8_experimental.yaml`;
it enables MXFP8 Linear and native Q/K RMSNorm, both accuracy-changing.
The deployment API does not treat extra model config keys as validated switches:
do not reuse the old all-experiments YAMLs with options excluded from this series.

```bash
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7  # choose reserved physical devices
export VLLM_OMNI_AR_DIFFUSION_KV_GATHER=1
# Optional, accuracy-changing dense FP8 attention; otherwise BF16 FA remains:
export VLLM_OMNI_AR_DIFFUSION_FP8_FIA_MODE=C8V16_TILING512
vllm-omni serve /path/to/lingbot-world-v2-14b-causal-fast-diffusers \
  --omni --deploy-config benchmarks/lingbot_world/configs/npu_usp4_fp8_experimental.yaml
```

For the conservative preset, unset the FP8 attention environment variable.
All presets are opt-in; the original main deployment configurations are unchanged.
Single-session gathered noncausal block attention is required for EagleFIA.
The provider must expose `FP8FAMode.C8V16_TILING512` and matching native libraries.
The verified MindIE-SD source is official GitCode `dev` commit
`62a91eb77ccaefa810fdea97a2043f930925b7a3` (source version 3.1.0), with matching
Python, Torch2.10/Python3.12 plugin and Ascend950 FIA library builds. Native
quantization modes are Q/K/V 7/7/12, `inner_precise=4`, Q128/K256 sequence
blocks and V512x64 blocks. An older isolated provider may lack the V5 symbol.

The measured hardware is Atlas350PR / Ascend950PR with 112 GB per NPU.
Eight-card target acceptance remains untested. Memory bandwidth, unsharded VAE,
host work and communication do not necessarily scale with peak compute.
There is no demonstrated stable RTF<1 or completed quality acceptance.

## Excluded experiments

- EQBSA/CDF selection: no same-configuration whole-model positive result;
  short-window overhead and sparse-selection accuracy still need assessment.
- Fused AdaLN and packed QKV/RoPE prototypes: numerical or performance concerns.
- Original Wan norm restoration, native VAE norm experiments and epilogue/full
  norm-SiLU kernels: different semantics or no established overall advantage
  over the framework default VAE. Keep these on the earlier experiment branch.
- KV-cache 8-bit storage, DistGN and INT8 Linear: not implemented by this series.

Existing implementation and regression provenance is retained in individual
commits. New source cleanup is verified separately from historical NPU results.

## Cleaned-source checks

- Ruff check, Ruff format check and `git diff --check`: passed.
- CPU contracts: 315 passed, 21 hardware cases deselected. The independent
  test process used a CPU platform with no-op synchronization for synchronous
  CPU tensors; it did not alter production code or normalization mathematics.
- In the default NPU-detected test environment, 310 passed and five real-VAE
  CPU cases failed because the pre-existing global Wan patch dispatches CPU
  tensors to `npu_rms_norm`. All five failures reproduced on `7afd00e98` with
  only the common vLLM 0.32 scheduler compatibility correction applied.
- Coverage includes paged host lengths/staging, slot writes, MXFP8 selection,
  FP8 provider ABI, VAE padding/fallback, transformer and pipeline CPU contracts.
  No new cleaned-snapshot whole-model or eight-card NPU measurement is claimed.
