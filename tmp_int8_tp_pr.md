<!-- markdownlint-disable -->
<!-- Proposed title: [WIP][Bugfix] Fix diffusion INT8 tensor-parallel scales and enable ROCm -->

Fixes #6584

## Purpose

Online diffusion INT8 quantized each input-sharded tensor-parallel weight independently. Each rank derived its per-output-row scale from only its local input shard, so TP2 did not use the same quantized weight representation as TP1. BAGEL also stores secondary expert weights on a plain module, and MiniMax-H3 can use a text encoder TP group that is independent from the DiT TP group.

This change follows the scale-sharing design in upstream vLLM PR [#49764](https://github.com/vllm-project/vllm/pull/49764). It records the global and local input dimensions, detects input sharding from those dimensions, computes the row maximum across the correct process group, and packs each local shard with the shared row scale. Standard layers use vLLM's default TP group. MiniMax-H3 provides an override only for its independent text encoder group.

ROCm now uses vLLM's existing INT8 linear kernel selection. vLLM-Omni does not add or fork a matrix multiplication kernel. The linear wrapper only flattens diffusion hidden states to the rank-2 input required by the upstream Triton and AITER kernels, then restores the original leading dimensions.

The files under `rocm_int8_verification/` are a temporary test-only harness included in the draft PR so maintainers can reproduce the AMD checks. They keep model downloads, compiler caches, temporary files, and results under that folder. They are not runtime code and will be removed before the PR is marked ready for review or merged. The current draft branch is squashed to one commit with the harness included. The branch will receive one final cleanup squash after maintainers finish using it.

This PR does not claim BF16 versus INT8 output-quality parity. The MiniMax-H3 and BAGEL runs are execution smoke tests. The exact quantized-weight tests cover the TP scale correctness claim.

## Test Plan

```bash
python -m pytest \
  tests/diffusion/quantization/test_int8_config.py \
  tests/diffusion/models/minimax_h3/test_minimax_h3_quantization.py \
  tests/diffusion/models/bagel/test_bagel_quantization.py \
  --run-level advanced_model \
  -m core_model
```

On a two-GPU ROCm container:

```bash
ROCM_VERIFY_RUN_ID=<run-id> \
VLLM_TEST_MINIMAX_H3_FL2VA_MODEL="$PWD/rocm_int8_verification/models/MiniMax-H3/FL2VA" \
RUN_AITER=1 \
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_USE_AITER_LINEAR=1 \
bash "$PWD/rocm_int8_verification/run_all.sh"
```

The `rocm_int8_verification/` command is available while the PR is in draft. After maintainers finish reproducing the results, remove that temporary folder, rerun the remaining checks, and perform the final cleanup squash without the harness.

### Help requested for CUDA and NPU validation

Upstream vLLM PR [#49764](https://github.com/vllm-project/vllm/pull/49764) tested the same TP-invariant packing rule by replacing the collective with the full unsharded `amax`, then checking exact packed-weight and scale equality for dense and MoE shards. The PR reports `6 passed, 16 warnings` on an NVIDIA GB200, followed by TP4 CUDA model-load and generation smoke tests. It did not report NPU validation.

The equivalent vLLM-Omni CUDA checks use the real upstream INT8 kernel and a real two-process NCCL collective. Help running this exact command on two NVIDIA GPUs would be appreciated:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python -m pytest \
  tests/diffusion/quantization/test_int8_config.py::TestGPUInt8Smoke \
  tests/diffusion/quantization/test_int8_config.py::test_shared_quantizer_matches_native_kernel_on_two_gpus \
  --run-level advanced_model \
  -m 'core_model and cuda' \
  -vv -s -rs
```

Expected result: the three real CUDA kernel smoke tests and the two-GPU exact TP parity test pass. The parity test must not skip.

The current NPU checks compare the shared quantizer with the real `torch_npu.npu_dynamic_quant` result and run a real `npu_quant_matmul` forward. Help running this exact command on an Ascend NPU would be appreciated:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
python -m pytest \
  tests/diffusion/quantization/test_int8_config.py::TestNPUInt8Smoke \
  --run-level advanced_model \
  -m core_model \
  -vv -s -rs
```

Expected result: all four real NPU smoke tests pass. This covers native packing equivalence and the vendor matmul, but not a real two-NPU collective. The existing two-accelerator parity worker is intentionally CUDA and ROCm only and uses NCCL. Guidance or access from an NPU maintainer is requested for an equivalent two-NPU HCCL parity run. Until that is completed, NPU TP2 remains unverified.

Run the diff-scoped pre-commit hooks after the final rebase and squash:

```bash
pre-commit run --files \
  docs/user_guide/quantization/int8.md \
  rocm_int8_verification/.gitignore \
  rocm_int8_verification/README.md \
  rocm_int8_verification/minimax_int8_tp2.py \
  rocm_int8_verification/run_all.sh \
  tests/diffusion/models/bagel/test_bagel_quantization.py \
  tests/diffusion/models/minimax_h3/test_minimax_h3_quantization.py \
  tests/diffusion/quantization/test_int8_config.py \
  vllm_omni/diffusion/models/minimax_h3/encoder.py \
  vllm_omni/quantization/int8_config.py
```

**vLLM Version:** `0.27.0`

**vLLM-Omni Commit:** `ef5324668a75400cbe2cb5cd826582eed9f3daf3`

## Test Result

Local targeted suite:

```text
45 passed, 8 skipped, 14 warnings in 17.85s
```

Two-GPU ROCm results with PyTorch `2.11.0+gitd0c8b1f`, HIP `7.2.53211`, vLLM `0.27.0`, and vLLM-Omni `e20bcd5231f2f8ffb10fe6e337c303305cde9118`:

```text
Affected INT8, BitsAndBytes, MiniMax-H3, and BAGEL tests:
54 passed, 10 skipped, 14 warnings in 55.74s

Triton INT8 kernel:
3 passed, 14 warnings in 0.76s

AITER INT8 kernel:
3 passed, 14 warnings in 1.10s

Two-GPU exact TP1 versus TP2 quantized-weight parity:
1 passed, 14 warnings in 32.52s

MiniMax-H3 INT8 with DiT TP2 and text encoder TP2:
Passed and saved nonempty video and audio arrays

BAGEL INT8 TP2:
Passed in a standalone rerun and saved a nonempty PNG
```

The AMD runs were completed before the history-only rebase and squash. The contents of every file changed by this PR are identical between tested commit `e20bcd5231f2f8ffb10fe6e337c303305cde9118` and current commit `ef5324668a75400cbe2cb5cd826582eed9f3daf3`.

CUDA reviewer validation: requested and pending.

NPU reviewer validation: requested and pending. Real one-NPU packing and matmul tests exist; real two-NPU HCCL TP parity is not yet covered.

The diff-scoped pre-commit run passed every available hook. The shellcheck hook could not run because the local machine does not have the `shellcheck` binary. This must be resolved before submission.

**BEFORE SUBMITTING:** read [CONTRIBUTING.md](https://github.com/vllm-project/vllm-omni/blob/main/CONTRIBUTING.md) and run the [precheck-pr skill](https://github.com/vllm-project/vllm-omni/blob/main/.claude/skills/precheck-pr/SKILL.md) with the code agent for a self-check against project conventions.
(anything written below this line will be removed by GitHub Actions)
