# L5: communication, readiness and residency

## When applicable

Use for intra-stage collectives, inter-stage payload/KV transport, host/device
copies or offload. A single-GPU deployment may still have IPC, output and offload
costs. Mark collectives N/A only from the resolved configuration, not from absent
events in one rank's trace.

Read [communication and residency](communication-and-residency.md) when any such
path is present. Use the L1 request map and L0 hardware/runtime contract.

## Procedure

1. Resolve rank groups, placement, replicas and actual topology. Keep TP/SP/CFG/VAE
   collectives distinct from stage connectors and request replication.
2. Capture all relevant participants and correlate producer-ready, transfer,
   completion and consumer-start events. Align clocks or bound synchronization
   error before interpreting cross-host gaps.
3. Separate readiness skew from data-exchange time. A long communication kernel
   may wait for another rank; its duration is not a wire-bandwidth measurement.
4. Identify exposed dependency delay and overlap with independent work. Check
   resource competition: overlap can make compute and communication slower.
5. Compare matched message sizes/operations/topology with measured references.
   Include consumer-visible completion, not only sender enqueue/return.
6. Inspect storage/residency and capacity constraints for offload or staging copies.
7. Propose the smallest differentiating experiment; feed local kernel-shape changes
   and exposed/hidden costs into the joint L6 scenario.

## Output

Provide a rank/transfer table with payload/operation, bytes, endpoints, readiness
skew, elapsed span, exposure assumptions, reference, unknowns and source owner.
Do not sum rank durations into request latency or equate aggregate NCCL duration
with removable time. Single-rank evidence supports local observations only.

Near-reference transfer efficiency does not close unnecessary transfers. A
balanced run at one size does not close scaling behavior at another size/load.
