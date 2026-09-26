# Expert Parallel

This section describes how to add Expert Parallel (EP) to a diffusion transformer that uses Mixture-of-Experts (MoE) layers.
We use **HunyuanImage3.0** as the reference implementation.

---

## Table of Contents

- [Overview](#overview)
- [Step-by-Step Implementation](#step-by-step-implementation)
- [Testing](#testing)
- [Reference Implementations](#reference-implementations)
- [Summary](#summary)

---

## Overview

### What is Expert Parallel?

**Expert Parallel** is a parallelism strategy in Mixture-of-Experts (MoE) models that distributes different expert networks across distinct computational devices. Each device holds and computes only a subset of experts (local experts), with tokens dispatched to and gathered from remote devices via collective communication operations (e.g., All-to-All, All-Gather).

| Backend | Description |
|---------|-------------|
| `allgather_reducescatter` | Default backend based on allgather/reducescatter primitives, suitable for general EP+DP deployments.|

## Configuration

Enable EP by setting the `--enable-expert-parallel` flag. The EP size is automatically calculated as:

```text
EP_SIZE = TP_SIZE × SP_SIZE × CFG_SIZE × DP_SIZE
```


Where:

- `TP_SIZE`: Tensor parallel size
- `SP_SIZE`: Sequence parallel size
- `CFG_SIZE`: Classifier-free guidance parallel size
- `DP_SIZE`: Data parallel size
- `EP_SIZE`: Expert parallel size (computed automatically)

Note:
- Expert parallelism is only applicable to Mixture-of-Experts (MoE) models.
- The EP group is created **per pipeline stage**, meaning it includes all ranks that participate in model parallelism except pipeline parallelism.
- The underlying communication pattern for expert parallelism is **All-to-All** among the ranks in the EP group.

For example, consider a configuration with `TP=2`, `SP=1`, `CFG=2`, and `DP=4` (total 2×1×2×4 = 16 GPUs).

- Expert layers are handled by an EP group of size 16.

- Attention layers use tensor parallelism of size 2 within each of the 8 DP groups (because `DP×CFG×SP = 4×2×1 = 8` groups, each containing the 2 TP ranks). Inside each such group, the attention weights are sharded across the 2 GPUs.


## Step-by-Step Implementation

### Step 1: Configure Expert Parallelism Settings

Calculate local experts per rank:

```
ep_size = 8  # Expert Parallel size (typically equals TP size)
num_experts = 64
num_local_experts = num_experts // ep_size  # 8 experts per card

# Check divisibility
assert num_experts % ep_size == 0, "Experts must be divisible by EP size"
```

### Step 2: Use Sparse MoE Block to enable EP routing.

Example:
```
from vllm.model_executor.layers.linear import ReplicatedLinear
class HunYuanSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = -1,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.n_routed_experts = config.num_experts  # 64

        # Calculate local experts per rank (key for EP)
        if self.tp_size > self.n_routed_experts:
            raise ValueError(f"TP size {self.tp_size} > experts {self.n_routed_experts}")

        # Routing gate (replicated on all ranks, computes scores for all tokens to all experts)
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        # EP expert layer (factory loads platform-specific implementation)
        self.experts = HunyuanFusedMoE(...)
```
**Key Points:**
- gate is **ReplicatedLinear** (replicated on all ranks)
- experts is created via **HunyuanFusedMoE factory**, which automatically handles EP dispatch

### Step 3: Initialize EP Runtime

Initialize the EP communication context before model loading.
```
from vllm.utils.import_utils import resolve_obj_by_qualname
# Call during __init__ or model loading
op_name = "hunyuan_fused_moe"

# Prepare EP runtime: establish communication groups, assign local expert indices, init _expert_map
current_omni_platform.prepare_diffusion_op_runtime(op_name)

# Factory automatically resolves platform implementation (GPU: FusedMoE / NPU: AscendFusedMoE)
impl = resolve_obj_by_qualname(
    current_omni_platform.get_diffusion_model_impl_qualname(op_name)
)
```

### Step 4: Expert Weight Mapping & Loading

Each rank loads only the expert weights assigned to its local allocation.
```
# Get expert parameter mapping (different per rank)
expert_mapping = HunyuanFusedMoE.make_expert_params_mapping(
    model=self,
    ckpt_gate_proj_name="gate_proj",
    ckpt_down_proj_name="down_proj",
    ckpt_up_proj_name="up_proj",
    num_experts=64,
    num_redundant_experts=0,
)
# Returns: [(param_name, weight_name, expert_id, shard_id), ...]
# Note: Each rank only contains mappings for its local expert_ids

# Filter non-local experts during loading
for name, loaded_weight in weights:
    if "mlp.experts" in name:
        # Parse expert_id from weight name (implementation needed)
        expert_id = parse_expert_id_from_name(name)
        local_expert_start = (ep_rank) * num_local_experts
        local_expert_end = (ep_rank + 1) * num_local_experts

        if not (local_expert_start <= expert_id < local_expert_end):
            continue  # Skip non-local expert weights
```
### Step 5: Forward Pass with EP

Example (MoE Forward):
```
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    orig_shape = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

    # 1. Global routing computation (all tokens, all expert scores)
    # hidden_states: [num_tokens, hidden_dim] (full tensor)
    router_logits, _ = self.gate(hidden_states)  # [num_tokens, num_experts]

    # 2. EP dispatch and compute (HunyuanFusedMoE handles all_to_all internally)
    # - Dispatch: Send tokens to target ranks based on router_logits
    # - Local Compute: Each rank processes only its num_local_experts
    # - Combine: Results returned to original token positions
    final_hidden_states = self.experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
    )

    # 3. Add shared expert output (not EP, computed on all ranks)
    if self.shared_mlp is not None:
        shared_out = self.shared_mlp(hidden_states)
        final_hidden_states = final_hidden_states + shared_out

    # 4. Tensor Parallel All-Reduce (synchronize across TP group)
    if self.tp_size > 1:
        final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
            final_hidden_states
        )

    return final_hidden_states.view(orig_shape)
```

## Testing
After adding Expert Parallel support, test via command line:
```bash
cd examples/offline_inference/text_to_image
python text_to_image.py \
    --model Your-org/your-model \
    --prompt "a cup of coffee on the table" \
    --output "ep_enabled.png" \
    --num-inference-steps 50 \
    --guidance-scale 5.0 \
    --tensor-parallel-size 8 \
    --seed 1234 \
    --enable-expert-parallel
```

vLLM‑Omni currently focuses on core diffusion model inference acceleration, so the Expert Parallel implementation includes only the basic multi‑GPU expert sharding functionality (enabled via --enable-expert-parallel). Advanced features such as communication backend selection (--all2all-backend), load balancing (--enable-eplb and its configuration), and multi‑node deployment belong to the extended capabilities of the main vLLM project and have not yet been integrated into Omni.

## Reference Implementations

Complete examples in the codebase:

| Model | Path | Pattern | Notes |
|-------|------|---------|-------|
| **HunyuanImage3.0** | `vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py` | Standard EP | Full implementation with validation |
| **EP Tests** | `vllm-omni/tests/e2e/offline_inference/test_expert_parallel.py` | E2E testing | EP correctness and performance |
| **Constraint Tests** | `vllm-omni/tests/diffusion/models/hunyuan_image3/test_hunyuan_fused_moe.py` | Unit testing | Validation logic |

---
## Summary

Adding Expert Parallel support to diffusion model:

1. **Identify MoE layers** - Locate the router and expert networks in each transformer block.
2. **Validate EP constraints** – Ensure num_experts is divisible by expert_parallel_size.
3. **Test** - Run with enable-expert-parallel, check memory reduction, speedup, and output quality against single‑GPU baseline.
