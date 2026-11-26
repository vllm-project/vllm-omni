# H3-Image regression coverage

The serving tests use release weights, including at `core_model`: there is no
H3-Image tiny builder. A single GPU uses layerwise offload for both the DiT and
text encoder. Cache the image checkpoint and the matching base H3 text encoder;
plan for about 135 GB of weights and at least 150 GB of free host RAM for offload.

From the repository root in the installed serving/test environment:

```bash
pytest -sv tests/e2e/online_serving/test_minimax_h3_image.py \
  -m 'core_model and diffusion and cuda and H100 and cards_1' \
  --run-level core_model
pytest -sv tests/e2e/online_serving/test_minimax_h3_image.py \
  -m 'advanced_model and diffusion and cuda and H100 and cards_1' \
  --run-level advanced_model
```

CUDA Ready runs generation and single-reference editing on one H100. CUDA
Merge runs two-reference editing on one H100 and generation and single-reference
editing on two H100 GPUs with resident DiT TP2 and text-encoder TP2. Select the
two-GPU cases with
`advanced_model and cuda and H100 and cards_2`.

For NPU Ready on Atlas 800T A3, use `npu and A3 and cards_1` in place of
`cuda and H100 and cards_1` in the `core_model` command. The eager DLO server
uses one die of an A3 card for generation and single-reference editing.

Set `VLLM_TEST_MINIMAX_H3_IMAGE_MODEL` to use a local complete checkpoint.
The existing external-server 4K transfer regression remains separately selectable
with `cards_4` and requires `H3_IMAGE_TEST_BASE_URL`.

## Performance

Nightly measures 2048² text-to-image and single-reference editing at six steps,
concurrency one, two warmups, and ten measured requests. The benchmark records
latency and throughput in Buildkite artifacts. Profiling and baseline assertions
are disabled.

| CI device | Deployment | Pytest device count |
| --- | --- | ---: |
| H100, 1 card | DiT and text-encoder layer offload | 1 |
| H100, 2 cards | Resident DiT TP2 and text-encoder TP2 | 2 |
| H100, 4 cards | Resident DiT TP4 and text-encoder TP4 | 4 |
| Atlas 800T A3, 1 card | Ulysses SP2, text-encoder TP2, DiT layer offload with AllGather | 2 |
| Atlas 800T A3, 2 cards | Ulysses SP4, text-encoder TP4, DiT layer offload with AllGather | 4 |
| Atlas 800T A3, 4 cards | Resident DiT TP8 and text-encoder TP8 | 8 |

A3 exposes two dies per card. Its pytest selector and `--num-gpus` count dies.
Each performance job has a 60-minute timeout. The benchmark harness can wait
up to 40 minutes for the server, leaving 20 minutes for requests, teardown,
and artifact upload when startup reaches its limit.

```bash
export BENCHMARK_DIR=tests/dfx/perf/results
pytest -sv tests/dfx/perf/scripts/run_benchmark.py \
  --test-config-file tests/dfx/perf/tests/test_minimax_h3_image_vllm_omni.json \
  -m 'full_model and diffusion and cuda and H100 and cards_1' --run-level full_model
```

Use the matching device and count for the other rows. Cache the GPT-2 tokenizer
and model weights before running offline. Editing uses a synthetic reference.

## Diffusers accuracy calibration

On one H100, the pinned Modular Diffusers pipeline generates reference images in PNG format
with component-level CPU offload. The vLLM-Omni server uses single-GPU
layerwise offload. The test resolves the model's `main` once and passes the
same local snapshot to both implementations. It records the resolved commit in
the accuracy results. Both implementations use the same text encoder and fixed
2048² generation and editing requests. The test retains two draws per
implementation and task, plus cross-implementation and repeatability SSIM/PSNR.

From the repository root, source
`tests/e2e/accuracy/minimax_h3_image/setup_reference.sh` and run the accuracy
test with `--run-level full_model`. The first run needs access to the pinned
Diffusers source and both weight formats. For offline runs, pre-cache `main`
with the native and converted weights. The CUDA Nightly job retains images,
metrics, and standard `result_acc_*.json` accuracy summaries. The test allocates
up to 3 minutes for reference preflight, 20 for CUDA reference generation, 10
for server startup, and 3 per image request. These limits leave at
least 10 minutes of the 60-minute job for setup, cleanup, and artifacts.

`H3_IMAGE_CALIBRATE` defaults to `1`. Successful calibration reports SKIP
after retaining artifacts with `calibration_only=true` because it does not
enforce numerical accuracy floors.
Runtime errors still fail the test. To enforce thresholds, set `H3_IMAGE_CALIBRATE=0`
after selecting device-specific `ssim` and `psnr` floors for both `t2i` and
`edit` in `tests/e2e/accuracy/minimax_h3_image/thresholds.json` and changing its
status to `selected`. The shared accuracy helpers resolve the device profile.
All floors currently remain `null`; an unselected or invalid gate fails before
model startup. Both draws for each task must meet both floors.

A3 accuracy calibration is paused in Nightly while the Diffusers reference
failure is investigated. Manual A3 runs use one die for each implementation.
The Diffusers reference uses leaf-level offload for the text encoder and one
block per group for the DiT, with a 25-minute generation timeout. For a manual
run, initialize the CANN environment, source `setup_reference.sh`, and select
`full_model and npu and A3 and cards_1`.
