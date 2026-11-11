# LingBot World accuracy

These tests belong to the existing `tests/e2e/accuracy` suite:

- `test_precision.py::test_numeric_precision`: fixed-input Wan VAE tensors and full decoded video/window/content comparisons.
- `test_wbench.py::test_wbench_quality`: native WBench per-dimension and binary-activity checks.
- `test_vbench.py::test_vbench_quality`: native VBench per-dimension checks and full-frame background coverage.

A session-scoped fixture generates reference/candidate inputs once. It has no
semantic acceptance gate: each test validates its own authenticated inputs and
writes its own report and `result_acc_*.json` before asserting. A missing NPZ
therefore fails precision without hiding complete videos from either evaluator.
The reports preserve generation hardware/source/model/workload separately from
the CUDA hardware running the evaluators.

## Reference and resources

The one supported reference path checks out `e96e3beeb6fa0842ad5f4a74b4d409edca39a0d5`
and uses the same installed vLLM 0.33 runtime and pinned model/input assets as the
candidate. Three fresh reference and candidate starts supply the existing pixel
repeatability and semantic activity/spread checks. They are accuracy inputs,
not performance baselines. Automatically created reference checkouts are removed
on success or failure; an explicit `--lingbot-reference-root` / `--root` remains
caller-owned. There is no recorded-golden or auto-refresh mode.

CUDA uses two H100 GPUs with Ulysses/SP=2. NPU uses the existing four-device A2
configuration; it does not reserve eight devices or silently switch to A3.
Each case records one 40-chunk video per start, using the common realtime-video
benchmark transport. Native evaluator dependencies/weights remain locked in a
separate CUDA environment, provisioned lazily by the semantic tests.

CUDA runs the three tests together, sharing their capture:

```bash
python -m pytest -sv --run-level full_model \
  tests/e2e/accuracy/lingbot_world/test_precision.py \
  tests/e2e/accuracy/lingbot_world/test_wbench.py \
  tests/e2e/accuracy/lingbot_world/test_vbench.py \
  --lingbot-profile=tests/e2e/accuracy/lingbot_world/configs/cuda.json \
  --lingbot-output=lingbot-artifacts/cuda-accuracy
```

NPU captures on four A2 devices, then the same three tests consume its artifacts
on the CUDA evaluator worker. Loading captured data does not prepare a model or
assert the evaluator worker has NPU devices:

```bash
python -m tests.e2e.accuracy.lingbot_world.prepare capture \
  --profile tests/e2e/accuracy/lingbot_world/configs/npu.json \
  --output lingbot-artifacts/npu-accuracy
# Download the capture directory on the CUDA worker, then run:
python -m pytest -sv --run-level full_model \
  tests/e2e/accuracy/lingbot_world/test_precision.py \
  tests/e2e/accuracy/lingbot_world/test_wbench.py \
  tests/e2e/accuracy/lingbot_world/test_vbench.py \
  --lingbot-profile=tests/e2e/accuracy/lingbot_world/configs/npu.json \
  --lingbot-generation=lingbot-artifacts/npu-accuracy \
  --lingbot-output=lingbot-artifacts/npu-accuracy-results
```

Partial capture retains its manifest and available artifacts. CI allows the
accuracy job to inspect them even when capture failed, while each selected test
still fails on its own missing, changed, incomplete or nonfinite inputs. A local
run without `--lingbot-profile` is opt-in; selected CI tests cannot skip.

Performance belongs to `tests/dfx/perf` and its normal recorded JSON baseline
workflow. There is no live paired performance calibration or long-stability
lane here. CPU checks and test collection validate these contracts without
claiming real-model accuracy or two-GPU capacity.
