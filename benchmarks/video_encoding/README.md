# Multiview video encoding benchmark

`benchmark_multiview_encoding.py` compares the sequential Diffusers exporter,
the new serial and parallel streaming offline exporters, the existing HTTP
monolithic encoder, and parallel HTTP camera encoding plus remuxing. It loads
one prerecorded `(views, frames, height, width, 3)` NumPy array, performs one
warm-up, alternates variant order, and runs every measurement in a fresh
process.

Primary runs use seven 1280×720 cameras with 217 and 425 frames per camera and
a fourteen-CPU affinity mask:

```bash
taskset -c 0-13 python benchmarks/video_encoding/benchmark_multiview_encoding.py \
  --input generated_7x217x720p.npy --fps 30 --runs 5

taskset -c 0-13 python benchmarks/video_encoding/benchmark_multiview_encoding.py \
  --input generated_7x425x720p.npy --fps 30 --runs 5
```

Use `--http-requests 2` for the fair-sharing run. Repeat with `--preset medium`
to exercise B-frame remuxing. The report contains median and p95 wall time,
median queue wait and CPU time, output bytes, and peak process-tree RSS above
the resident input-buffer baseline. Use actual captured/generated output for
performance claims; synthetic camera patterns are intended only for correctness
checks.
