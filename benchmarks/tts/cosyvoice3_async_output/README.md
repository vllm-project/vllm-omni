# CosyVoice3 B2 validation

This directory contains the reproduction code and results for
[RFC #6870, B2](https://github.com/vllm-project/vllm-omni/issues/6870).
The implementation and its validation material belong to the same PR branch;
a separate published validation branch is not needed.

## B2 completion

The [completion report](completion/README.md) includes the complete design
contract, expanded regression coverage, current-main C=1/C=4 repeated GPU
acceptance, and diagnostics of the historical 0.28 waveform discrepancy.
Use its current-main results and scope when reviewing the submitted B2 change.
The earlier studies below remain historical reproduction evidence.

## Current payload-only cleanup

Following the maintainer's updated B2 scope, CosyVoice3 keeps output
materialization inline and omits unused Talker hidden payloads in async-chunk
mode. The processor still receives sampled tokens on payload-free steps.
The shared GPU runner has no B2 changes.

The [cleanup validation](cleanup/README.md) records the isolated single-GPU
follow-up and validation against current main. The study below retains the
previous async-materialization implementation and measurements for provenance.

## Expanded measurements

The [follow-up study](followup/README.md) adds five paired C=1 rounds,
natural and fixed-length C=4 comparisons, component ablations, and separate
profiling. It observes about 0.50% mean C=1 AR improvement and no consistent
C=4 benefit. Enabling per-step async materialization while retaining hidden
payloads regressed AR latency by about 7.9% in four ablation rounds.
The results below retain the earlier two-round observation for provenance.

## Initial measurements

These are historical measurements on a fixed vLLM 0.28.0-compatible source
baseline, `3204a0b2ade0f05424b7f11e3bcc03cb3542e0c6`, plus the B2 patch.
Current-main validation with vLLM 0.30.0 is reported separately in the cleanup study.

| Round | AR ITL baseline → B2 (ms/token) | TTFA median baseline → B2 (ms) |
| --- | ---: | ---: |
| 1 | 4.852 → 4.805 | 458.10 → 454.78 |
| 2 | 4.831 → 4.798 | 449.70 → 451.18 |

Mean AR ITL decreased by **0.040 ms/token (0.83%)**. This is a small local
signal from two rounds, not a statistically established or general speedup.
TTFA improved in one round and regressed in the other. Sampled peak device
memory was 24,760.44 MiB for every run (NVML sampled at 200 ms intervals).
No stable TTFA, concurrency-scaling, or 0.45 ms/step improvement is claimed.

On the patched 0.28-compatible checkout, 295 selected regression tests passed
(21 warnings, 5.23 s), 64 CUDA snapshot/output checks passed, and all 16 paired
generated waveforms and streaming chunk boundaries matched the baseline.
The generated audio is regression-test output, not a ground-truth recording
or an independent speech-quality evaluation.

Machine-readable results are in [results.json](results.json), and exact source
hashes and versions are in [manifest.json](manifest.json). Raw logs and example
audio are in the separately attached evidence archive.

## Environment

- One RTX 4090, 49,140 MiB reported by NVML; driver 550.127.05.
- Python 3.12.3, vLLM 0.28.0+cu129, PyTorch 2.13.0+cu129,
  Transformers 5.14.1, NCCL 2.29.7, s3tokenizer 0.3.0.
- `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`, revision
  `29e01c4e8d000f4bcd70751be16fa94bf3d85a18`.
- Torch flow backend (`COSYVOICE3_TRT=0`); TensorRT is not part of this run.
- Concurrency 1, two rounds ordered baseline/B2/B2/baseline, with two warmups
  and eight measured requests per run. The English/Chinese inputs and reference
  transcript are fixed in `benchmark.py`; the reference WAV is the repository
  asset `tests/assets/cosyvoice3/zero_shot_prompt.wav`.
- Both variants use seed 0, `codec_chunk_frames=25`, `max_num_seqs=8`, async
  chunk and AR async scheduling, and disabled prefix caching. Stage placement,
  sampling, and memory configuration otherwise come from the same baseline YAML.

Use a working CUDA environment matching these versions. No driver or package
installation is performed by these scripts. The original local image setup is
documented in the evidence archive; it is not a clean-install recipe.

## Prepare two local checkouts

Run from the PR repository. These detached worktrees are local directories,
not additional branches to publish. Choose a new empty validation directory:

```bash
B2_SCRIPTS="$PWD/benchmarks/tts/cosyvoice3_async_output"
B2_RUN=/tmp/cosyvoice3-b2-validation
mkdir -p "$B2_RUN"
git worktree add --detach "$B2_RUN/baseline" 3204a0b2ade0f05424b7f11e3bcc03cb3542e0c6
git worktree add --detach "$B2_RUN/patched" 3204a0b2ade0f05424b7f11e3bcc03cb3542e0c6
git -C "$B2_RUN/patched" apply --unidiff-zero --check "$B2_SCRIPTS/v028.patch"
git -C "$B2_RUN/patched" apply --unidiff-zero "$B2_SCRIPTS/v028.patch"
```

`v028.patch` is the measured B2 implementation and regression coverage exported
with zero context (`git diff --unified=0`). Its source hashes are verified by
`run_matrix.py`. It is a reproduction artifact, not another copy imported by
the runtime. Apply it only to the pinned baseline.

Make the official model available under
`$B2_RUN/models/Fun-CosyVoice3-0.5B-2512`, for example:

```bash
mkdir -p "$B2_RUN/models"
ln -s /path/to/Fun-CosyVoice3-0.5B-2512 "$B2_RUN/models/Fun-CosyVoice3-0.5B-2512"
```

The model is not bundled. Preserve the same model revision and S3Tokenizer
cache for both variants. When using Docker, make these paths available inside
the container and use the container paths in the commands below.

## Regression and CUDA checks

Select the patched source through both working directory and `PYTHONPATH`.
This avoids importing current main into the vLLM 0.28.0 environment.

```bash
cd "$B2_RUN/patched"
export PYTHONPATH="$B2_RUN/patched"
export COSYVOICE3_TRT=0
export XDG_CACHE_HOME="$B2_RUN/cache"
python3 -m pytest \
  tests/model_executor/models/cosyvoice3/test_cosyvoice3_model_helpers.py \
  tests/model_executor/stage_input_processors/test_cosyvoice3_stage_input_processors.py \
  tests/worker/test_gpu_ar_model_runner.py \
  tests/core/sched/test_omni_ar_scheduler_*.py \
  tests/distributed/omni_connectors/test_chunk_transfer_adapter.py \
  --run-level=core_model -q --disable-warnings
python3 "$B2_SCRIPTS/gpu_snapshot.py" --output "$B2_RUN/gpu-snapshot.json"
```

The unit cases are L1 CPU tests; the snapshot script needs a real CUDA device.
Unit tests additionally need pytest, pytest-mock, pytest-asyncio and pytest-xdist.

## Performance and streaming comparison

Run the matrix sequentially on one GPU:

```bash
python3 "$B2_SCRIPTS/run_matrix.py" \
  --baseline "$B2_RUN/baseline" \
  --patched "$B2_RUN/patched" \
  --validation-dir "$B2_RUN"
```

The matrix selects the correct source for each subprocess and saves per-run
logs, metrics, deployment YAML, generated WAV/NumPy arrays, memory observations,
and pairwise comparisons under `$B2_RUN/results`. It refuses to overwrite an
existing run. The timing workload is preserved from the recorded experiment;
the checked-in wrappers add path arguments and source/output validation.

`compare.py` computes AR ITL from Stage-0 `vllm_itls_ms` rows, weighted by token
interval count after excluding warmups. The matrix does not enable profiling.
The underlying iterator is used to keep the same engine alive across warmups
and measured requests; the public generator wrapper closes it on exhaustion.

AI assistance: Codex assisted with implementation, tests, validation scripts,
benchmark analysis, and this documentation. Review these results within the
stated version and workload limits.
