# Distributed Layerwise Offload

This document describes distributed layerwise offload (DLO) for diffusion
models. DLO keeps only a small number of DiT blocks on the accelerator and
streams the remaining blocks from host memory. The distributed backend can
either shard those host-side weights across an existing parallel group or keep
the standard loader's rank-local weights and avoid an additional collective.

For user-facing commands, see the
[distributed layerwise offloading guide](../../../user_guide/diffusion/offloader/distributed_layerwise_offload.md)
and the [Cosmos3 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/cosmos3/Cosmos3-DistOffload.md).

## Status

DLO is implemented for multi-device diffusion execution. The default
AllGather path is the primary path for DP and SP deployments. The
`--dlo-no-use-allgather` path is a rank-local compatibility mode: it is useful
for standard-loader sharding, workstation bring-up, and systems where an
additional DLO collective is undesirable, but it does not reduce host weight
storage across ranks.

The compatibility matrix below describes the current implementation. The
unit-level guards are covered, but not every parallelism combination has a
full model-and-hardware end-to-end test.

## Design

### DLO consumes the existing parallel topology

DLO does not create a new DP, TP, or SP topology. It reads the configured
`DiffusionParallelConfig` and attaches offload hooks to the DiT blocks after the
standard distributed groups have been initialized.

The DLO weight-sharding group is selected as follows:

1. Use the existing DP group when `data_parallel_size > 1`.
2. When DP is one and SP is greater than one, use the SP group.
3. Otherwise, run rank-locally without a DLO process group.

TP is deliberately not used as DLO's AllGather group. HSDP has its own
parameter-sharding lifecycle and is not allowed to be sharded a second time by
DLO's AllGather path.

### AllGather path

With the default `dlo_use_allgather=True`, each rank stores approximately
`1 / group_size` of each streamable block in pinned host memory. The next
block's shard is copied to a device buffer and reconstructed with
`all_gather_into_tensor` on a communication stream while the current block is
executing.

```text
Compute:    [Block N]             [Block N+1]          [Block N+2]
H2D:                      [shard N+1]           [shard N+2]
AllGather:                [full N+1]             [full N+2]
Buffers:    [current slot]       [prefetch slot]       [current slot]
```

![DLO double-buffer prefetch pipeline](../../figures/dlo/dlo_pipeline.gif)

The backend uses two shared device buffers, so accelerator weight residency is
bounded by the largest streamed blocks rather than the complete model.

When DP is greater than one, the engine can process one request per DP rank in
the same denoising wave. Because AllGather is a collective, all participating
requests must take the same execution path at every denoising step.

### Rank-local path without DLO AllGather

With `--dlo-no-use-allgather`, DLO forces its internal offload shard size to
one. The regular model loader remains responsible for preparing each rank's
weights, including TP-local tensors or HSDP-managed parameters. DLO then
streams those rank-local tensors block by block using H2D copies only.

This mode means:

- DP still provides independent replicas, but DLO does not shard weights
  across DP ranks.
- SP still performs its normal activation/attention collectives, but DLO does
  not shard weights across SP ranks.
- TP/HSDP/SP collectives, if configured, are not disabled by this flag; only
  DLO's additional weight AllGather is disabled.
- Pure DP deployments keep a full host-side model copy per rank, subject to
  shared OS page-cache behavior.
- The scheduler does not require a synchronized DP request wave for DLO.

This path is intentionally not implemented by reusing the AllGather mmap
loader. It relies on the standard loader so model-specific TP/HSDP transforms
remain intact.

## Parallelism compatibility

| Parallelism | DLO + AllGather | DLO without AllGather |
|---|---|---|
| **DP** | Supported primary path. DLO shards host weights across the DP group and can run DP multi-concurrency. | Supported rank-local path. DP replicas remain independent; no DLO weight sharding or cross-DP DLO collective. |
| **SP** | Supported in the implementation. With DP=1, DLO uses the SP group for host-weight sharding; SP still shards sequence/activation work. | SP remains active, but DLO keeps standard-loader rank-local weights and adds no SP weight collective. |
| **TP > 1** | Unsupported in the DLO mmap path. TP-aware loader callbacks are bypassed, so the backend rejects this configuration when it enters that path. | Standard loading is retained and TP-local tensors can be streamed. This is the intended compatibility path, but it still needs broader model and hardware validation. |
| **HSDP** | Rejected. HSDP has already sharded parameters, so DLO AllGather would double-shard them. | Accepted by configuration. HSDP owns parameter sharding and its own gathers; DLO only stages rank-local parameters. End-to-end coverage is limited. |

### Combined dimensions

- **DP + SP:** DLO uses the DP group for weight sharding when DP is greater
  than one; SP continues to use its own sequence-parallel group. If DP is one,
  the SP group becomes DLO's sharding group in AllGather mode.
- **DP + TP/SP without AllGather:** standard model loading defines the
  rank-local tensor layout. DLO adds no cross-DP, cross-TP, or cross-SP weight
  collective.
- **HSDP + SP:** the general parallel configuration permits HSDP over SP, but
  DLO must use `--dlo-no-use-allgather`. HSDP remains responsible for weight
  materialization and synchronization.
- **HSDP + DP or TP:** rejected independently by the diffusion parallel
  configuration.

## Request and loading constraints

AllGather DP multi-concurrency requires:

- explicit `num_inference_steps`;
- the same `num_inference_steps` for all requests in a wave; and
- identical request arguments that affect the collective execution path.

The no-AllGather path does not impose these DLO-specific synchronized-wave
requirements.

The mmap loader is used only by the supported DLO+AllGather path when the
model has a compatible checkpoint layout. Online quantization is incompatible
with that sharded mmap path; use `--dlo-no-use-allgather` or disable online
quantization. Models that do not meet the mmap requirements use the regular
loader path.

## Validation coverage

Current source-level validation includes:

- HSDP + DLO + AllGather rejection;
- HSDP + DLO without AllGather acceptance at configuration level;
- TP rejection in the DLO+AllGather mmap path;
- resident-layer requests requiring no-AllGather;
- DP request-wave validation for denoising-step compatibility;
- sharding, double-buffer, AllGather-size, and heterogeneous-block regression
  tests.

The highest-value missing coverage is end-to-end numerical comparison against
ordinary layerwise offload for DP+SP, TP+no-AllGather, and HSDP+SP+no-AllGather
on the target CUDA/NCCL or CANN/HCCL hardware.

## Recommendations

- Use **DP + DLO AllGather** for the supported throughput and host-memory
  scaling path.
- Use **SP + DLO AllGather** for long-sequence workloads when DP concurrency is
  not the goal.
- Use **no-AllGather** to bring up TP or workstation PCIe configurations, with
  the expectation of higher host-memory use and lower validation confidence.
- Prefer **HSDP alone** for production HSDP deployments until the combined
  HSDP + DLO no-AllGather path has broader end-to-end coverage.
