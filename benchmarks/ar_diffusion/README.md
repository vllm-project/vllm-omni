# Noisy PP: CausVid Wan DiT benchmark

This opt-in harness runs the native `StageWanTransformer`, cell executor,
Latest KV selection, experimental FlowEuler sampler and paged FlashAttention from
[PR #8282](https://github.com/vllm-project/vllm-omni/pull/8282).
It publishes KV per transformer block using CUDA IPC on the same host and
NCCL between hosts. Native attention, model weights and sampling stay in use.
The fused variant also caches request-local text K/V projections and
preserves eager BF16 rounding in gated residual kernels. Normalization and
modulation use the native AdaLayerNorm, including its CUDA fusion.

These are benchmark-scoped adapters, installed and restored by context
managers. The diffusion frontend does not gain a multinode serving executor.
Each process handles one request at a time; this harness does not measure
concurrent serving or VAE/end-to-end video throughput.

## Reproduce

The native cell APIs from PR #8282 are merged into upstream `main` (merge
commit `959b483cfb6e3cb72b35ca3992f46717d22a66e5`). This draft branch
includes the merged dependency; no dependency cherry-pick is needed.
From a local clone, fetch and check out the benchmark branch:

```bash
git fetch https://github.com/vllm-project/vllm-omni.git pull/8728/head
git switch -c codex/wan-benchmark-reproduce FETCH_HEAD
```

Use a Linux/CUDA environment compatible with this checkout, with vLLM,
PyTorch, paged FlashAttention, Triton, `safetensors` and `cuda.bindings` installed.
The tested environments and raw measurements are linked under
[Recorded validation](#recorded-validation).

`OMNI_MODEL` must point to the complete `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` root,
including `transformer/config.json` and transformer safetensors. Passing only
the transformer directory is unsupported. The model configuration must have
30 blocks, 12 attention heads and head size 128.

Set `CAUSVID_CKPT` to the locally downloaded `tianweiy/CausVid`
`autoregressive_checkpoint/model.pt`. The native remapper overlays the student
after loading the Diffusers DiT shell. Both variants use identical weights.
Runtime module paths and pipeline registration follow the pinned dependency.
The default is T=3 plus one clean stage, shift=8. This uses the dependency's
experimental FlowEuler path; it does not implement official DMD sampling parity.

The compressed fixture contains BF16 T5 embeddings for `a cat walking on grass`,
shape `[1,512,4096]`. It excludes T5 loading and encoding from DiT measurements.
Expand it to an absolute path visible from both nodes:

```bash
gzip -dc benchmarks/ar_diffusion/fixtures/conditioning.pt.gz > /tmp/omni-condition.pt
sha256sum /tmp/omni-condition.pt
# 198358abb9eb6e80296d18bb78f82924a4fef1fa1a44ff26af34e75cfda9c5af
```

For two nodes, set `OMNI_NODE_RANK` to `0` or `1`, `OMNI_MASTER_ADDR` to the
first node's reachable address, `OMNI_CONDITION` to the expanded fixture path,
and `OMNI_OUTPUT` to an empty shared results directory. Run on both nodes:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
NCCL_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=0 \
python -m torch.distributed.run \
  --nnodes=2 --nproc-per-node=4 --node-rank="$OMNI_NODE_RANK" \
  --master-addr="$OMNI_MASTER_ADDR" --master-port=29879 \
  -m benchmarks.ar_diffusion.benchmark_noisy_pp \
  --model "$OMNI_MODEL" --causvid-ckpt "$CAUSVID_CKPT" \
  --condition "$OMNI_CONDITION" --out "$OMNI_OUTPUT" \
  --groups 2 --steps 3 --shift 8 --chunks 128 --skip 64 --measure 32 \
  --variants baseline hybrid_fused --warmup 1 --repeat 3
```

Adjust the network interface to the actual deployment. Each node uses four
GPUs; the stage-major placement has four stages with two block partitions per
stage (`K=15`). CUDA IPC requires distinct physical GPUs for local ranks.

For one node (`K=30`), use:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m benchmarks.ar_diffusion.benchmark_noisy_pp \
  --model "$OMNI_MODEL" --causvid-ckpt "$CAUSVID_CKPT" \
  --condition "$OMNI_CONDITION" --out "$OMNI_OUTPUT" \
  --groups 1 --steps 3 --shift 8 --chunks 128 --skip 64 --measure 32 \
  --variants baseline hybrid_fused --warmup 1 --repeat 3
```

Run from the checkout root and use a distinct output directory for each
topology or source revision. `hybrid` enables the transport and text K/V
projection cache without pointwise fusion; `cached` enables only that cache.
Condition embeddings use the native per-step cache from PR #8282.
Only Latest-KV with one fixed denoise/clean stage per rank is supported.
Each scheduled cell and its attention context refer to one transformer block;
the hybrid ring indexes the native plan without expanding the schedule again.

## Timing and output contract

Each request has 128 chunks of three latent frames at BF16 shape
`[1,16,3,60,104]`, seed zero, history six chunks and no sink. The complete latent
is `[1,16,384,60,104]`; decoding it would produce 1533 frames at 832x480.

Steady FPS is a pixel-frame equivalent of pure DiT throughput. The final
denoise rank records one CUDA completion event per chunk. After 64 skipped
chunks, 32 measured chunks contribute 384 frame equivalents:

```text
FPS = 384000 / (completion_ms[95] - completion_ms[63])
```

The remaining 32 chunks complete the request. Text encoding, model loading,
VAE, hashing and video encoding are excluded. A full request warms each
variant, followed by three full measured requests; the report uses medians.
The harness retains all latents, checks finiteness and hashes the complete
output after timing. All variants and repeats must match the supplied hash
or the first request's hash. Latest KV semantics depend on `K`, so hashes are
compared within a topology rather than across `K=15` and `K=30`.

The default hybrid ring has `(history+1) * stages = 28` version positions per layer;
it owns the only KV pool in hybrid runs. The baseline allocates its native
version pool separately. `kv_capacity` reports version positions per layer
for the selected pool, identified by `kv_pool`; `kv_reserved_bytes` includes
the hybrid ticket storage when applicable.
Buffers use producer-ready and consumer-release tickets to prevent a reused
ring position from overwriting KV that attention still reads. Failed IPC
requests keep mappings alive until the worker job exits.

Source snapshots, per-request events and hashes, GPU memory statistics and
median summaries are written under the output directory. Per-block transport
rounds and read labels derive directly from the native `CellPlan`; no second
scheduler or general KV manager is included. Original transport
source attribution and adaptation hashes are listed in
[provenance.json](provenance.json).

## Tests

CPU tests cover native KV labels, last-reader release identities, physical
pool addresses, allocation boundaries, exported-buffer lifetime on failure,
cache expiration and restoration of modules after normal and failed requests.
CUDA tests compare fused kernels against separate eager BF16 operations,
including strided residual tensors. They do not require model weights:

```bash
python -m pytest -o addopts='' \
  tests/diffusion/ar_diffusion/test_wan_hybrid_kv.py \
  tests/diffusion/ar_diffusion/test_wan_native_optimizations.py \
  -m 'core_model and cpu' --run-level=core_model -q
python -m pytest -o addopts='' \
  tests/diffusion/ar_diffusion/test_wan_native_pointwise.py \
  -m 'core_model and cuda' --run-level=core_model -q
```

The repository-wide test plugins also require a vLLM version compatible with
the selected Omni branch. For isolated unit/kernel checks when those unrelated
plugins cannot load, add `--noconftest` and omit `--run-level`.

## Recorded validation

### Post-merge compatibility (October 17)

With upstream main `b164fea797531c1fa10f8b87829632cad006b9fc` (including
merged PR #8282), 20 CPU checks and 3 CUDA BF16 kernel checks passed. A real
four-H200 T3/S4/G1, shift8, eight-chunk smoke ran all four variants with clean
process exits. All complete latent hashes matched
`4e34a83a4592b47ed2341f31e61455635c0973efa98bad901065f2ec246e49d4`.
This confirms compatibility after branch synchronization; it is not a new
steady-throughput measurement. Runtime benchmark code is unchanged from
`a93668047`. Evidence is under
`/data/dxw/outputs/pr8728-postmerge-20261017/results/` on the validation host.

### CausVid/cell validation (October 15)

On four H200 GPUs with dependency `80d6d53` and benchmark `a54c13cd1`, BF16,
T3/S4/G1, shift8, q3/H6,
128 chunks and skip64/measure32, one warmup plus three measured requests gave:

| Variant | Median DiT FPS | KV reservation per rank |
| --- | ---: | ---: |
| Native baseline | 129.320568 | 8.03 GiB |
| `hybrid_fused` | 138.301577 (+6.94%) | 22.49 GiB |

All eight complete latent hashes matched. The 23 CPU contracts and 3 CUDA
kernel checks passed. Eight-chunk smokes also passed for all four variants
at T3/G1 and for baseline/hybrid_fused at T1/G2 (four GPUs in both cases).
The fused block now reuses native AdaLayerNorm; bypassing its CUDA path had
caused a hash mismatch. This is single-node DiT validation, not E2E or DMD parity.
Raw events, source snapshots and the independent timing/hash audit are under
`/data/dxw/outputs/pr8728-causvid-cell-20261015/results/` on the validation host.

A subsequent cleanup removed the redundant condition-embedding wrapper:
20 CPU checks and a two-GPU T1/G1 before/after smoke passed, with all eight
complete outputs matching across the four variants. The performance table
remains bound to `a54c13cd1`.

### Historical RF validation (October 10)

The following measurements used the earlier RF checkpoint, T=4, shift=5 and
chunk APIs, rather than the current CausVid benchmark.

Independent five-H200 validation measured baseline **120.589275 DiT FPS** and
`hybrid_fused` **133.678281 DiT FPS** (+10.85%), with matching full latent hashes
across one warmup and three measured requests per variant. The four-variant
smoke, 23 CPU contracts and 3 CUDA kernel checks also passed. Performance is
bound to dependency `86490bab` plus benchmark `ef5f8c8c`; the later import fixes
were checked with another five-GPU smoke. These are single-node DiT results.

The complete evidence and audit scripts are preserved at the fixed archive
commit `72d0a1a0ae92157d3aa1d5156eadad5dfb2696fe`, outside the active benchmark:

- [Independent H200 report and raw evidence](https://github.com/Dong1017/vllm-omni/tree/72d0a1a0ae92157d3aa1d5156eadad5dfb2696fe/benchmarks/ar_diffusion/results/h200-validation-20261010)
- [Original contributor's H800 report and raw evidence](https://github.com/Dong1017/vllm-omni/tree/72d0a1a0ae92157d3aa1d5156eadad5dfb2696fe/benchmarks/ar_diffusion/results/h800-q3-20261009)
- [H800 archive audit script](https://github.com/Dong1017/vllm-omni/blob/72d0a1a0ae92157d3aa1d5156eadad5dfb2696fe/benchmarks/ar_diffusion/audit_results.py)

The first baseline request establishes a reference hash, which every variant
and repeat must match. Record that hash and optionally pass `--expected-sha`
for subsequent runs in the same environment. Archived H800 hashes are not
cross-environment references.

## End-to-end video requests

`benchmark_e2e.py` measures synchronous, single-request `Omni.generate` wall
time. Each request includes text encoding, DiT denoising, VAE decoding,
postprocessing and return to the client. Model initialization and warmup are
excluded. Output is measured as actual returned RGB frames divided by request
wall time. `--fps` specifies playback speed; it is not generation throughput.

The E2E entrypoint uses the selected model's pipeline and deployment config.
It does not install the DiT harness's four optimization variants. The original
`benchmark_noisy_pp.py` retains its pre-encoded-text, DiT-only measurement scope.

For an installed Physis model integration, set `OMNI_MODEL` to its full local
model snapshot or repository ID, then run from the repository root:

```bash
python benchmarks/ar_diffusion/benchmark_e2e.py \
  --model "$OMNI_MODEL" \
  --class-name PhysisWanPipeline \
  --deploy-config vllm_omni/deploy/physis_wan.yaml \
  --frames 81 --chunks 19 --fps 16 \
  --steps 4 --height 480 --width 832 \
  --prompt 'A corgi running along the beach at sunset, waves rolling in, golden light.' \
  --seed 0 --kv-history-chunks 18 --warmup 1 --repeats 3 \
  --output-json results/wan21-1p3b-e2e.json
```

Keep the scheduler's `CUDA_VISIBLE_DEVICES` allocation. The script reads and
records it; it does not assign GPUs. `--pipeline-parallel-size` optionally
overrides the deploy configuration's PP degree. PP is the number of processes
in the model pipeline; Physis's `stage_parallel_size` independently selects
how the denoising steps are distributed.

Use Omni's existing `--stage-overrides` mapping to adjust logical device IDs
and model configuration without copying deployment files. For example, a
four-process Same-Step topology based on `physis_wan.yaml` adds:

```bash
--pipeline-parallel-size 4 \
--stage-overrides '{"0":{"devices":"0,1,2,3","model_config":{"physis_profile":"wan21_1p3b","num_denoise_steps":4,"ar_diffusion_stage_config":{"stage_parallel_size":4,"max_batch_size":1,"max_history_chunks":18}}}}'
```

Logical device IDs are relative to the scheduler's visible allocation. For
two processes, use devices `0,1`, PP=2 and `stage_parallel_size=1`; for one
process, use device `0`, PP=1 and `stage_parallel_size=1`. Set the complete
`model_config` override explicitly, since it replaces that configuration.

The following fixed workloads all use four denoising steps, 480x832 output,
guidance scale 1.0, seed 0 and 18 history chunks:

| Model | Class | Deploy config | Wan profile | Frames | Chunks | Playback FPS |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Wan 2.1 1.3B Same-Step | `PhysisWanPipeline` | `physis_wan.yaml` | `wan21_1p3b` | 81 | 19 | 16 |
| Wan 2.1 14B Same-Step | `PhysisWanPipeline` | `physis_wan.yaml` | `wan21_14b` | 81 | 19 | 16 |
| Wan 2.2 5B Same-Step | `PhysisWanPipeline` | `physis_wan.yaml` | `wan22_5b` | 121 | 29 | 24 |
| LTX 2.5 22B Same-Step | `PhysisLTXPipeline` | `physis_ltx.yaml` | — | 121 | 14 | 24 |
| MiniMax H3 33B Same-Step | `PhysisH3Pipeline` | `physis_h3.yaml` | — | 162 | 48 | 24 |

For the two other Wan checkpoints, replace `physis_profile` in the stage
override. LTX and H3 use their respective deploy configurations and retain
their own model configuration fields when constructing an override. These
pipelines and deploy files are provided by the model integration, separately
from this benchmark change.

The JSON includes every measured request plus median and aggregate E2E
throughput, benchmark and imported runtime source HEADs and working tree status, model input and
resolved local path, optional `--revision`, device identity and visible device
allocation. Use a pinned local snapshot or pass `--revision` to identify model
weights. `physis.dit` and `physis.chunk_*` are retained in seconds under
`physis_stage_durations_seconds`; raw `stage_durations` is also retained because
other engine keys may have different units. DiT FPS is derived only when a
`physis.dit` duration is present. Those model timings describe denoising and
chunk completion, while E2E FPS measures the entire completed request.
