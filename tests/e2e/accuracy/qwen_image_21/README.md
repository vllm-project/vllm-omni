# Qwen-Image 2.1 accuracy

`test_qwen_image_21.py` compares native eager HTTP text-to-image generation
against the independent Diffusers implementation from
[PR #14804](https://github.com/huggingface/diffusers/pull/14804), pinned to
`8d3c30bfda9b511c00992f40cff4170a5502814d`. Both paths use the same
`Qwen/Qwen-Image-2.1` snapshot at `790c92633540aa0cb11d9abf19eb46d861714758`.
Missing imports, reference code or weights fail rather than skip the test.

## Comparison contract

| Field | Reference | Native | Alignment evidence |
| --- | --- | --- | --- |
| Input | `A ceramic teapot on a wooden table`; no image/negative prompt | Same | Serialized request |
| Output | One 1024 x 1024 RGBA PNG | Same | Assert count, size, mode and format; no resizing/compositing |
| Sampling | 50 steps, true CFG 1.0 | Same | Pipeline call / image request |
| RNG | Seed 42, BF16 CUDA noise | Same | Both `prepare_latents` use the same shape and `randn_tensor` draw before T2I denoising |
| Scheduler | Checkpoint FlowMatchEulerDiscreteScheduler, linear sigma schedule and dynamic shift | Same | Same config, steps and latent geometry; reference records timesteps/sigmas |
| DiT attention | cuDNN SDPA | `CUDNN_ATTN` | Piecewise prefill and prefix-cache decode |
| Cache | `use_kv_cache=True` | Checkpoint `causal_condition` prefix cache | BF16 KV, no cache quantization |
| Execution | One GPU, BF16, no offload | Same; eager, no compilation/CUDA graphs | Worker metadata and server arguments/log |

The pinned upstream processor uses `backend=None` during prefill. The worker
therefore wraps only the transformer's forward with
`sdpa_kernel(CUDNN_ATTENTION)` and selects Diffusers' `native` backend for
decode. Text encoder and VAE keep their ordinary loading/attention paths.
Native fused QKV and reference separate projections can round differently.
This tests native eager T2I fidelity; editing, CFG > 1, multi-image input,
compilation, CUDA graphs, parallelism and quantization are outside its scope.

Every invocation creates a fresh artifact directory. Before loading either
model, the reference worker verifies its actual imports and Git revision,
processor/tokenizer, configs and every indexed weight shard (nonempty file
and safetensors header). Metric dependencies import in the pytest process.
The reference generates twice, with a reset generator per call; RGBA pixels
must repeat exactly. Its process exits before the native server starts.

## Prepare and run

Requires the repository's installed runtime/test dependencies (including
native `diffusers==0.40.0` and torchmetrics), Git, one sufficiently large CUDA
GPU, and access to the complete pinned checkpoint. The nightly lane uses
one H100. Local validation on a different SKU does not calibrate H100.

From the repository root, prepare a separate reference source and venv:

```bash
git init .ci-references/qwen-image-21-diffusers
git -C .ci-references/qwen-image-21-diffusers fetch --depth=1 \
  https://github.com/huggingface/diffusers.git \
  8d3c30bfda9b511c00992f40cff4170a5502814d
git -C .ci-references/qwen-image-21-diffusers checkout --detach FETCH_HEAD
export QWEN_IMAGE_21_REFERENCE_ROOT="$PWD/.ci-references/qwen-image-21-diffusers"
python3 -m venv --system-site-packages .ci-references/qwen-image-21-env
.ci-references/qwen-image-21-env/bin/python -m pip install --no-deps torchao==0.16.0
export QWEN_IMAGE_21_REFERENCE_PYTHON="$PWD/.ci-references/qwen-image-21-env/bin/python"
```

The pinned reference needs torchao's `FqnToConfig` import even for BF16.
Only its subprocess receives the reference source in `PYTHONPATH`; native
serving keeps the repository's Diffusers dependency. A local checkpoint can
be selected with `QWEN_IMAGE_21_MODEL=/absolute/frozen/snapshot/path`. Its
directory name must equal the pinned revision, or explicitly set
`QWEN_IMAGE_21_MODEL_REVISION=790c92633540aa0cb11d9abf19eb46d861714758`.
Both processes read that same snapshot. Local provisioning is responsible
for matching its asserted revision; the test records config hashes and
weight sizes, but does not rehash 34 GB of weights on every invocation.

Local single-case command:

```bash
python -m pytest -sv \
  tests/e2e/accuracy/qwen_image_21/test_qwen_image_21.py::test_qwen_image_21_matches_diffusers \
  --run-level full_model
```

CI-equivalent command:

```bash
python -m pytest -sv tests/e2e/accuracy/qwen_image_21/test_qwen_image_21.py \
  -m "full_model and diffusion and H100 and cards_1" --run-level full_model \
  --junitxml=tests/e2e/accuracy/artifacts/Qwen-Image-2_1/pytest.xml
```

Add `--collect-only -q` for selection checks. Collection is not an accuracy
pass. Nightly X2I explicitly selects this file, uses `h100_1`, and retains
the complete artifact subtree plus pytest log/JUnit on success and failure.
`pipefail` preserves pytest's status when tee writes the CI log. Scheduled
main builds require `NIGHTLY=1`; PR builds require `nightly-test` and a
matching dependency path. Rendering does not execute Buildkite.

## Threshold proposal and artifacts

Status: **threshold proposal awaiting selection**. The bounds below are
engineering targets, not a user-selected or H100-calibrated gate. Results
meeting them are provisional until selection. Do not loosen a bound merely
to accept a mismatch.

| Metric | Definition / aggregation | Proposed bound |
| --- | --- | --- |
| SSIM | Shared torchmetrics full-image SSIM, separately over RGB and RGBA; higher is better, [-1, 1] | Both >= 0.99 |
| PSNR | Shared torchmetrics full-image PSNR over normalized RGB/RGBA channels; higher is better, dB | Both >= 40 dB (RMSE <= 0.01) |
| Alpha MAE | Mean absolute alpha-channel error divided by 255; lower is better, [0, 1] | <= 1/255 |

`tests/e2e/accuracy/artifacts/Qwen-Image-2_1/t2i-*/` retains the request,
source hashes/revisions/paths, packages, preflight checks, reference log and
scheduler/dtype/backend metadata, reference/repeat/native PNGs, repeatability
and metric summaries. Both outputs and all metrics are saved before the
numerical assertions. The raw alpha channel is compared directly.

### Measured evidence (2026-10-10)

One fresh candidate run on **NVIDIA L20X** used base
`a43cdcdeee0278dce55878b5a0a5f13697ab19a4` plus this suite and CI patch.
The exact CI selector collected one node; execution reported **1 passed,
0 failed, 0 skipped, 0 deselected**, process exit **0**, in **239.32 s**.
This is a pass against the proposed bounds, pending user selection.

| Metric | Observed RGB | Observed RGBA | Headroom to proposed bound |
| --- | --- | --- | --- |
| SSIM | 0.995479405 | 0.996481776 | 0.005479405 / 0.006481776 |
| PSNR (dB) | 43.848800659 | 45.080787659 | 3.848800659 / 5.080787659 dB |
| Alpha MAE | — | 0.000126577 | 0.003794992 |

Both reference calls in this run produced bit-identical RGBA pixels and
PNG hashes. There is one independent candidate measurement: its observed
range has a single value per metric, so the worst values are those above.
The proposed bounds preserve a 1% normalized pixel RMSE target and catch
larger structural/alpha drift. The measured gap supports this initial
proposal, but does **not** establish cross-run stability or H100 tolerances.
Other prompts, seeds, workloads, GPU SKUs and numerical environments need
their own evidence; do not transfer the margin as an established guarantee.

The native runtime imported Diffusers **0.40.0** from
`/tmp/diffusers040-system/diffusers/__init__.py` (the existing provisioned
package), without modifying the shared installation. Both paths used
Torch **2.13.0+cu130**, Transformers **5.14.1** and CUDA **13.0**; native
vLLM was **0.31.0**, torchmetrics **1.9.0**. Reference source/installed
distribution versions are recorded separately because `PYTHONPATH` selects
the pinned Git source. Reference-only torchao **0.16.0** was installed into
this task's venv. Optional DeepEP import warnings did not prevent the run.

Evidence locations:

- Remote log: `/data/dyy/logs/qwen21-accuracy-20261010-1621/run1/run.log`
- Remote media: `/data/dyy/code/vllm-omni-codex-qwen21-accuracy-20261010-1621/tests/e2e/accuracy/artifacts/Qwen-Image-2_1/t2i-healzdxs/`
- Downloaded report/media/logs: workspace `outputs/qwen21-accuracy-20261010-1621/`

Ruff, marker validation, exact-node collection, full pipeline rendering and
dependency-filter rendering passed. CI routing uses one H100 and retains
the images, metrics, metadata, pytest log and JUnit report. Bootstrap/group
activation conditions were inspected; actual Buildkite execution and
H100/H200 numerical calibration remain **unvalidated**. GPU execution is
not repeated for documentation or CI dependency-only follow-up edits.
