# MammothModa2 startup benchmark

Measures `Omni()` initialization and first/subsequent image requests for
MammothModa2-Preview. This is a measurement baseline for
[#7075](https://github.com/vllm-project/vllm-omni/issues/7075); it changes no runtime code.

## Run

From the repository root, with the model downloaded and vLLM-Omni installed:

```bash
set -o pipefail
python benchmarks/mammoth_moda2/bench_startup.py \
    --model /path/MammothModa2-Preview \
    --deploy-config vllm_omni/deploy/mammoth_moda2.yaml \
    --height 512 --width 512 --seed 42 \
    --extra-body '{"text_guidance_scale": 4.0, "cfg_range": [0.0, 1.0], "num_inference_steps": 8}' \
    --repeat 1 --label baseline --output-json baseline.json --save-image baseline.png \
    2>&1 | tee baseline.log
python benchmarks/mammoth_moda2/parse_startup_log.py baseline.log --require-complete --markdown
```

Use `OMP_NUM_THREADS=16` to set CPU threads and `--parallel-stage-init` to
initialize stages concurrently. For first/subsequent requests, use
`--height 1024 --width 1024 --repeat 3` and 50 inference steps. Memory polling
is opt-in (`--sample-memory`); collect it in separate runs to avoid perturbing
the timing comparison. `--save-image` saves only the first request's image.
Compare that image across runs; a fixed seed alone does not guarantee equality
across execution modes or software versions.

## Measurement boundaries

| Metric | Boundary / limitation |
| --- | --- |
| Imports | Before torch import → benchmark imports complete; excludes interpreter startup |
| Time to ready | `Omni()` call → return; excludes client imports and requests |
| Time to first image | Benchmark clock start → first `generate()` return, including request setup |
| First / subsequent request | Each `generate()` call → return; same prompt and settings |
| Stage spawn / device init | Log timestamps: launch → `world_size=` → `Starting to load model`; 1 s resolution |
| Weight load | Engine-reported `Loading weights took`; read/cast/copy are merged |
| Model setup estimate | Reported model-load time minus weight-load time; not independently instrumented |
| Profile + KV + warmup | One merged interval after loading; compilation/capture may be nested within it |
| Compile / graph capture | Explicit engine log durations when present; N/A for eager execution |
| Memory | Optional 1 s samples: sum of process-tree RSS and maximum device memory used across GPUs reported by `nvidia-smi`; includes other GPU users and can miss transient peaks |

Stage timelines overlap with parallel initialization; do not sum them to
calculate time to ready. Missing log fields stay unmeasured. The parser marks
logs without `BENCH_JSON` as incomplete; `--require-complete` makes this an error.

## Reference results

A800-SXM4-80GB; Ubuntu 22.04; 16-CPU cgroup quota on a 128-core host;
vLLM 0.28.0+cu129, torch 2.13.0+cu129, vLLM-Omni `34aa1d2a`.
Checkpoint: 34.5 GiB, 8 shards, mostly fp32, on local NVMe.
Committed eager deploy config, GPU memory fractions 0.5 / 0.3.

The 2026-09-10 rerun used the corrected benchmark: one warm-up process,
then three rounds of four configurations with order rotated between rounds.
The checkpoint page cache was warmed before the matrix; no cache eviction
was requested during it. Each fresh process sent one 512×512, 8-step request
(guidance 4.0, seed 42). Memory polling was disabled. All 12 measured runs
are retained in [the results](reference_results.json).
Values below are `Omni()` seconds, mean ± sample standard deviation, n=3;
client imports took approximately another 10 s.

| CPU threads | Serial init | Parallel init |
| --- | --- | --- |
| Default (64) | 60.1 ± 0.2 | 41.1 ± 1.0 |
| 16 | 56.5 ± 0.9 | 36.7 ± 0.2 |

On this configuration, parallel initialization reduced time to ready by about
32%; combining it with 16 threads reduced it by about 39%. This supports
trying these settings on similar deployments, not changing defaults for all
models or hardware. Fixed 512×512 samples matched across all 12 measured runs;
this is a smoke check, not a broad quality evaluation.

For context, in the earlier serial baseline AR / DiT spent approximately 11 / 11 s in
spawn and imports, 6 / 6 s in device setup, 6 / 4 s loading weights, and
5 / 2 s in profile + KV + warmup. About 6 s followed the last stage becoming
ready. These coarse log intervals locate work; they are not independent
operator timings.

Separate observations from the earlier experiment (not rerun in this matrix):

- Requested page-cache eviction: 80.8 s; FlashInfer JIT cache removed:
  129.1 s (means, n=3 each). These are separate cache conditions, not guaranteed
  cold-machine startup or isolated disk/JIT costs.
- Three 1024×1024, 50-step eager requests in one process took
  95.2 / 94.9 / 94.6 s. No large first-request penalty was observed in that run.
- With AR `enforce_eager: false` and serial init, time to ready was
  62.5 ± 0.1 s (warm caches, n=3); logged graph capture was 4 s / 0.20 GiB.
  The model reported torch.compile unsupported. KV capacity fell from about
  15.7 to 2.0 GiB, and one parallel-init trial failed. The reported non-KV
  profiling difference is not a measurement of graph allocation. Graph-mode
  samples differed from eager. This needs separate memory and quality analysis
  before recommending the configuration.

## Loading diagnostic and storage scenarios

```bash
OMP_NUM_THREADS=16 python benchmarks/mammoth_moda2/raw_load_bench.py \
    --model /path/MammothModa2-Preview --shards 6,7,8 --mode cpu_cast
# Compare --mode gpu_cast in a fresh process with the same cache conditions.

MODEL_LOCAL=/local/MammothModa2-Preview OUT_DIR=./startup-bench \
    bash benchmarks/mammoth_moda2/bench_storage_scenarios.sh
# Optional MODEL_NFS=/nfs/MammothModa2-Preview; SCENARIOS="local-cold local-warm".
```

The diagnostic clones each mmap-backed tensor into owned CPU memory before
casting. `materialize_s` includes page faults, CPU allocation and copying;
`cast_s` measures dtype conversion; `h2d_s` includes GPU allocation and copy.
CUDA context/operation warmup is outside the timer. Total also includes file
opening and bookkeeping. This extra CPU copy differs from the engine loader,
so these numbers do not decompose its loading time or measure disk bandwidth.
The revised diagnostic was run three times each for CPU cast with 64 / 16
threads and GPU cast with 16 threads; per-run values are in the results file.
Both CPU materialization and conversion were faster with 16 threads here; this
does not isolate how much of the engine-level improvement comes from either.

Storage scenarios explicitly read shards before warm runs and request
`posix_fadvise(DONTNEED)` before cold runs. Eviction is not guaranteed, and
server-side NFS caches are uncontrolled. The script stops on failure and
summarizes only logs from the current invocation.
