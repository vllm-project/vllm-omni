# SPDX-License-Identifier: Apache-2.0
"""BDE DreamZero validation harness — accuracy metrics, artifacts, parity check.

Implements the plan in ``BDE_doc/dreamzero_kv_phase1_profiling_accuracy.md``:
compare the BDE KV path (enabled) against the model-local path (disabled) on
weighted DreamZero, save the generated videos + comparisons, and gate on
PSNR/SSIM/LPIPS. The model run itself is delegated to the existing offline
export script; the reusable pieces here are unit-tested on synthetic data.
"""
