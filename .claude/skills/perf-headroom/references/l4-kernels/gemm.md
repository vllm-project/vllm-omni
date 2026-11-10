# GEMM and linear diagnosis

## Inventory

Record M/N/K, batch count, operand strides/transposes, weight residency, input and
accumulation dtype, bias/activation/epilogue, selected library/kernel, and graph
mode. Distinguish separate Q/K/V or FFN projections from already-fused projections.
Partition by shape rather than reporting one aggregate "linear MFU".

## Procedure

1. Confirm the linear actually dispatches to the expected implementation and
   precision. Match source operation to kernel(s), including casts and packing.
2. Compute useful work with the convention in [roofline](roofline.md). Compare
   to the correct dense precision capacity and a matched operator reference.
3. For small/skinny shapes inspect grid size, waves, tails and memory cost. For
   large shapes inspect Tensor throughput, instruction mix and operand movement.
4. Inspect alignment, padding and layout. A padded kernel may have high executed
   throughput but poor useful throughput; padding is a cost, not free work.
5. Include wrappers, epilogues and transposes in the region comparison. Repeated
   weight packing may be a structural issue even if GEMM itself is near reference.
6. Use source/resources/counters to discriminate backend selection, memory,
   parallelism and launch-limit hypotheses. Keep unexplained discrepancies open.

## Recommendation branches

| Evidence | Possible next investigation | Reason to reject/deprioritize |
| --- | --- | --- |
| Wrong fallback or non-Tensor path for supported operands | Dispatch eligibility and compatible installed library path | Semantics/dtype/shape may require fallback |
| Large gap to same-shape operator reference | Library/algorithm or layout selection | Reference omits packing/epilogue, uses different cache/precision |
| Small grid, partial waves | Compatible grouping/batching/decomposition | Batching changes latency objective or lacks independent work |
| Fast GEMM with expensive wrappers | L3 epilogue/projection/layout fusion region | Already fused, extra consumers, spills or conversion cost |
| Close to matched reference and necessary work | Report mature current execution | Does not close other shapes or structural work |

Do not recommend quantization merely to raise apparent efficiency under a fixed
precision contract. If a permitted quantized regime is separately analyzed,
include scale/cast/dequant work and its quality acceptance.

## Output

Provide a shape-weighted table with current/reference cost, match caveats,
useful throughput, hardware mechanism and unsampled population. An efficient
representative linear supports only its declared population. Recommend full
region and unprofiled request validation; local kernel gains are not E2E gains.
