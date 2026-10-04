# SenseNova-U1.5 mixed warmup: committed H20 run evidence

These reports were recorded with the benchmark script at commit
`ed5c7a3c8f2daeecac24c8d786b128cfaa0a6cbf` on one NVIDIA H20-3e
(GPU 0, BF16, TP=1), using model snapshot
`9feeeab8a2792514d109cd34589342a2cc1d4ab2`.
The default-warmup servers ran baseline commit
`a038b38179e9c788d3af6a8e73f94652afb247e4`; the mixed-warmup
servers ran committed warmup code `ed5c7a3c8f2daeecac24c8d786b128cfaa0a6cbf`.
All four servers used separate empty CUDA, Triton, Inductor, and vLLM
cache directories. They ran sequentially on the same GPU.

Each report contains eight serial requests: two repetitions of
`t2i:1024x1024`, `t2t`, `i2t`, `t2i:1536x1536`.
Image requests used seed 42; the distilled LoRA runs used snapshot
`f33b8fe0216e6f10e5aed58ebbfdd05f02826732`.

| Run | Original report | Start (Unix s) | First `/health` (Unix s) | Startup | GPU process memory at readiness |
| --- | --- | ---: | ---: | ---: | ---: |
| base BF16, default warmup | [base2.json](base2.json) | 1790960890.655597925 | 1790960917.201091528 | 26.5 s | 34432 MiB (pid 1719655) |
| base BF16, mixed warmup | [warm.json](warm.json) | 1790960989.715407610 | 1790961078.307873011 | 88.6 s | 35600 MiB (pid 1721308) |
| distilled LoRA, default warmup | [base-distill.json](base-distill.json) | 1790961091.699863911 | 1790961127.247184753 | 35.5 s | 34432 MiB (pid 1722904) |
| distilled LoRA, mixed warmup | [warm-distill.json](warm-distill.json) | 1790961204.148144960 | 1790961296.750504494 | 92.6 s | 35600 MiB (pid 1724628) |

The readiness memory column is the raw result of
`nvidia-smi -i 0 --query-compute-apps=pid,used_gpu_memory --format=csv,noheader`
at first successful `/health`; it is process allocation, not peak memory.
The reports count graph captures and recompiles after the benchmark starts.
These are single runs of eight serial requests, not throughput or confidence
interval measurements. Regional `torch.compile` was skipped for this model.

## Server log excerpts

The lines below are copied from each run's server log. In default-warmup
runs, paged decode graph capture follows readiness; in mixed-warmup runs
it precedes readiness.

### base BF16, default warmup

```text
(DiffusionWorker pid=1719655) INFO:     Application startup complete.
(DiffusionWorker pid=1719655) DEBUG 10-03 01:09:37 [paged_decode.py:357] Captured decode graph for bucket=512 generation=0
```

### base BF16, mixed warmup

```text
(DiffusionWorker pid=1721308) INFO 10-03 01:10:05 [pipeline_sensenova_u1.py:626] SenseNova mixed warmup profile armed: resolutions=[(1024, 1024), (1536, 1536)] text_to_text=True image_to_text=True
(DiffusionWorker pid=1721308) DEBUG 10-03 01:11:15 [paged_decode.py:357] Captured decode graph for bucket=512 generation=0
(DiffusionWorker pid=1721308) INFO 10-03 01:11:15 [pipeline_sensenova_u1.py:1394] SenseNova mixed warmup text_to_text took 58.65 s
(DiffusionWorker pid=1721308) INFO 10-03 01:11:15 [pipeline_sensenova_u1.py:1394] SenseNova mixed warmup image_to_text took 0.06 s
(DiffusionWorker pid=1721308) INFO 10-03 01:11:16 [pipeline_sensenova_u1.py:1394] SenseNova mixed warmup text_to_image_1024x1024 took 0.80 s
(DiffusionWorker pid=1721308) INFO 10-03 01:11:17 [pipeline_sensenova_u1.py:1394] SenseNova mixed warmup text_to_image_1536x1536 took 1.02 s
(DiffusionWorker pid=1721308) INFO:     Application startup complete.
```

### distilled LoRA, default warmup

```text
(DiffusionWorker pid=1722904) INFO:     Application startup complete.
(DiffusionWorker pid=1722904) DEBUG 10-03 01:13:07 [paged_decode.py:357] Captured decode graph for bucket=512 generation=0
```

### distilled LoRA, mixed warmup

```text
(DiffusionWorker pid=1724628) INFO 10-03 01:13:39 [pipeline_sensenova_u1.py:626] SenseNova mixed warmup profile armed: resolutions=[(1024, 1024), (1536, 1536)] text_to_text=True image_to_text=True
(DiffusionWorker pid=1724628) DEBUG 10-03 01:14:54 [paged_decode.py:357] Captured decode graph for bucket=512 generation=0
(DiffusionWorker pid=1724628) INFO 10-03 01:14:54 [pipeline_sensenova_u1.py:1394] SenseNova mixed warmup text_to_text took 58.12 s
(DiffusionWorker pid=1724628) INFO 10-03 01:14:54 [pipeline_sensenova_u1.py:1394] SenseNova mixed warmup image_to_text took 0.08 s
(DiffusionWorker pid=1724628) INFO 10-03 01:14:54 [pipeline_sensenova_u1.py:1394] SenseNova mixed warmup text_to_image_1024x1024 took 0.68 s
(DiffusionWorker pid=1724628) INFO 10-03 01:14:55 [pipeline_sensenova_u1.py:1394] SenseNova mixed warmup text_to_image_1536x1536 took 0.60 s
(DiffusionWorker pid=1724628) INFO:     Application startup complete.
```

No `Recompiling function` line appeared during any benchmark sequence.
