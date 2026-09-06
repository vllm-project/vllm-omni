# Host weight dependency memory diagnostic

`safetensors_retention.py` isolates repeated CPU `get_tensor()` calls from
access to one cached tensor view. It imports neither vLLM nor HWR and uses a
synthetic 16-element float32 payload. HWR itself calls `get_tensor()` when
acquiring a lease, then exposes cached views; this probe does not measure HWR
requests or repeated lease acquisition.

Run one mode per fresh process in the same Python environment and CPU affinity:

```bash
git rev-parse HEAD > revision.txt
git status --short > working-tree.txt
git diff > working-tree.patch
# Select an allowed CPU from your process affinity; 0 is only an example.
CUDA_VISIBLE_DEVICES='' taskset -c 0 timeout 120s python \
  benchmarks/host_weight_runtime/safetensors_retention.py \
  --mode get_tensor --iterations 10000 --sample-every 5000 > feasibility.json
```

For a comparison, run each mode twice in fresh processes with the same controls:

```bash
for repetition in 1 2; do
  for mode in get_tensor reuse; do
    CUDA_VISIBLE_DEVICES='' taskset -c 0 timeout 120s python \
      benchmarks/host_weight_runtime/safetensors_retention.py \
      --mode "$mode" > "$mode-$repetition.json"
  done
done
```

The diagnostic sets Torch to one CPU thread, warms up before the baseline,
and reports preparation separately from timed loop work. Checkpoints run GC
and report Linux private/anonymous memory, RSS, and open descriptors. It checks
that transient tensor objects and the final cached view are released. Linux
`/proc/self/smaps_rollup` must be readable; invalid arguments and missing proc
support fail explicitly. Keep raw JSON and the repository snapshot with results.

Compare the final loop sample with the warmed baseline, then inspect the closed
sample. Private-memory growth after Python tensor objects disappear is evidence
of retention in the dependency/allocator path, not proof of a leak's root cause.
RSS can include file-backed pages; it is not a private-memory metric. Samples
also include small diagnostic bookkeeping allocations. Do not assert a portable
memory threshold or turn this synthetic loop into a per-request service estimate.

Do not drop shared caches, trim allocators, or add serving-time GC to make the
numbers look smaller. A dependency replacement or version change requires its
own reproducible comparison and correctness/lifecycle validation.
