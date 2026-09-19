# NPU

vLLM-Omni supports NPU through the vLLM Ascend Plugin (vllm-ascend). This is a community maintained hardware plugin for running vLLM on NPU.

## Requirements

- OS: Linux
- Python: 3.12

!!! note
    vLLM-Omni is currently not natively supported on Windows.

=== "NPU"

    --8<-- "docs/getting_started/installation/npu/npu.inc.md:requirements"

## Installation

### Set up using Docker

=== "NPU"

    --8<-- "docs/getting_started/installation/npu/npu.inc.md:pre-built-images"

### Build wheel from source

=== "NPU release"

    --8<-- "docs/getting_started/installation/npu/npu.inc.md:installation-release"

=== "NPU from main"

    --8<-- "docs/getting_started/installation/npu/npu.inc.md:installation-main"

## Configure MoE communication for diffusion stages

Ascend NPU deployments can select the MoE communication method for each
diffusion stage through `VllmConfig.additional_config`. Set
`npu_moe_comm_method` in the stage configuration:

```yaml
stages:
  - stage_id: 0
    additional_config:
      npu_moe_comm_method: ALLTOALL
```

The accepted values are case-insensitive:

| Value | Requirement | Description |
|-------|-------------|-------------|
| `ALLGATHER` | None | Select the AllGather MoE communication method. |
| `ALLTOALL` | Expert parallelism enabled with an effective EP world size greater than one | Select the AllToAll MoE communication method. |

Enable expert parallelism and configure a multi-rank EP topology before using
`ALLTOALL`. An incompatible `ALLTOALL` override fails during worker runtime
preparation. See the [Expert Parallelism Guide](../../user_guide/diffusion/parallelism/expert_parallel.md)
for EP configuration.

For a single diffusion deployment, the same setting can be passed through the
CLI:

```bash
vllm serve MODEL --omni \
    --enable-expert-parallel \
    --tensor-parallel-size 2 \
    --additional-config '{"npu_moe_comm_method":"ALLTOALL"}'
```

When `npu_moe_comm_method` is omitted, vLLM-Omni keeps the automatic selection
based on the Ascend device type and EP topology. The override is local to each
diffusion stage, so stages with different parallel topologies can use different
methods.
