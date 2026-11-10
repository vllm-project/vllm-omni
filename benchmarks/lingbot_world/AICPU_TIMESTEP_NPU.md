# Eliminate timestep compute AICPU on NPU

Enable `model_config.lingbot_npu_host_timestep: true` for eager NPU execution.
Both optimized NPU presets enable it. The default for an unspecified deployment
remains false; the original device expression is retained for other callers.
HCCL communication is outside this change.

## Source and precision contract

The complete four-rank profile previously contained 70 non-communication
`AI_CPU` calls per rank: Range, Neg, RealDiv, Pow, Cast, Cos and Sin, ten each.
They arise from `_sinusoidal_embedding`: a FLOAT timestep becomes DOUBLE,
followed by a 128-frequency basis and a `[3,128]` phase at `freq_dim=256`.

The DMD runner already has the scalar timestep on the host. The new path:

1. Rounds that host scalar through FP32, exactly as the existing device
   `torch.full(..., dtype=torch.float32)` does.
2. Evaluates the **same** `_sinusoidal_embedding` expression in FP64 on CPU.
3. Transfers the result once at the model dtype and caches it by FP32 bits,
   geometry, dtype and device, with at most 32 entries.
4. Passes the prepared frequencies explicitly to the transformer. The learned
   `time_embedding` and `time_projection` still execute normally on the NPU.

No timestep is copied back from the device. No FP32 approximation of the
sinusoid or learned-projection caching is introduced. Both the four noisy DMD
probes and the clean-x0 commit at timestep zero use this path. Request-local
flow-shift changes enter the cache key through their actual FP32 timestep.
Training/grad-enabled and non-eager/non-NPU use are rejected when the option is
selected. Direct transformer callers without prepared frequencies keep the
original path, including support for nonuniform per-frame timestep tensors.

## Single-NPU verification

On one Ascend950PR, tested actual checkpoint time-MLP weights with batch 1,
three frames, frequency width 256, hidden width 5120, BF16. Four flow shifts
(0.5, 1, 5, 8) each tested all four DMD labels plus the clean zero timestep.
All 20 cases had identical frequency vectors, time embeddings and projections
(max absolute error zero).

Level1 profiles contain ten timestep calls, including learned MLP execution:

| Path | Device AI_CPU calls | Total device kernels |
| --- | ---: | ---: |
| Original FP64 device expression | 70 | 170 |
| Host-prepared cached frequencies | 0 | 60 |

The seven original AICPU types each appeared ten times; none appeared after
the change. A first Level0 capture reported `N/A` core labels, so it was not
used to claim zero AICPU. Level1 supplied the explicit `AI_CPU` classification.

Three unprofiled batches of ten calls had median time per call 0.5483 ms before
and 0.2598 ms after. This is a local timestep-plus-time-MLP measurement, **not**
an end-to-end video speedup; do not add its percentage to earlier model gains.
The CPU contracts cover FP32 timestep rounding, BF16/FP16/FP32 output dtype,
geometry, bounded LRU reuse, tensor-metadata rejection, prepared time-MLP
outputs and both noisy/clean DMD routes. The full selected CPU suite passed
329 cases with 21 hardware cases deselected.

No four-card profiling or full video generation was run for this change.
This validates removal in the isolated production-shape timestep subgraph,
not a newly measured all-operator-zero whole-model trace. It intentionally
does not alter or count HCCL's AICPU communication kernels.
