# GPU to NPU Translation Patterns

This document provides a quick reference for translating GPU code patterns to NPU equivalents when porting omni-specific logic.

## Import Translations

### Forward Context
```python
# GPU
from vllm.forward_context import set_forward_context

# NPU
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
```

### Graph Wrapper
```python
# GPU
from vllm.compilation.cuda_graph import CUDAGraphWrapper

# NPU
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
```

### Attention State
```python
# GPU (no equivalent - uses FlashAttention states directly)

# NPU
from vllm_ascend.attention.attention_v1 import AscendAttentionState
```

### Utilities
```python
# GPU
# (directly use torch.cuda functions)

# NPU
from vllm_ascend.utils import enable_sp, lmhead_tp_enable
from vllm_ascend.ops.rotary_embedding import update_cos_sin
```

## Context Manager Translations

### Forward Context Setup
```python
# GPU
with set_forward_context(
    attn_metadata,
    self.vllm_config,
    num_tokens=num_tokens_padded,
    num_tokens_across_dp=num_tokens_across_dp,
    cudagraph_runtime_mode=cudagraph_mode,
    batch_descriptor=batch_desc,
):
    # forward pass

# NPU
with set_ascend_forward_context(
    attn_metadata,
    self.vllm_config,
    num_tokens=num_tokens_padded,
    num_tokens_across_dp=num_tokens_across_dp,
    aclgraph_runtime_mode=cudagraph_mode,  # Note: 'aclgraph' not 'cudagraph'
    batch_descriptor=batch_desc,
    num_actual_tokens=scheduler_output.total_num_scheduled_tokens,
    model_instance=self.model,
):
    # forward pass
```

### Graph Capture Context
```python
# GPU
from vllm.compilation.cuda_graph import graph_capture as cuda_graph_capture
with cuda_graph_capture(self.device):
    # capture

# NPU
from vllm_ascend.worker.model_runner_v1 import graph_capture
with graph_capture(self.device):
    # capture
```

## Graph Wrapper Usage

### Creating Graph Wrapper
```python
# GPU
if cudagraph_mode.has_full_cudagraphs() and has_separate_talker:
    self.talker_mtp = CUDAGraphWrapper(
        talker_mtp,
        self.vllm_config,
        runtime_mode=CUDAGraphMode.FULL
    )

# NPU
if cudagraph_mode.has_full_cudagraphs() and has_separate_talker:
    self.talker_mtp = ACLGraphWrapper(
        talker_mtp,
        self.vllm_config,
        runtime_mode=CUDAGraphMode.FULL
    )
```

### Checking Graph Wrapper Type
```python
# GPU
if not isinstance(self.talker_mtp, CUDAGraphWrapper):
    _cudagraph_mode = CUDAGraphMode.NONE

# NPU
if not isinstance(self.talker_mtp, ACLGraphWrapper):
    _cudagraph_mode = CUDAGraphMode.NONE
```

## Device Operations

### Synchronization
```python
# GPU
torch.cuda.synchronize()

# NPU
torch.npu.synchronize()
```

### Stream Operations
```python
# GPU
stream = torch.cuda.Stream(device=device)
torch.cuda.current_stream()

# NPU
stream = torch.npu.Stream(device=device)
torch.npu.current_stream()
```

## Attention Metadata

### State Setting (NPU-specific)
```python
# GPU - handled internally by attention backends

# NPU - explicit state setting required
self.attn_state = AscendAttentionState.DecodeOnly
if self.speculative_config and self.speculative_config.method == "mtp":
    if self.vllm_config.model_config.use_mla:
        self.attn_state = AscendAttentionState.SpecDecoding
    else:
        self.attn_state = AscendAttentionState.ChunkedPrefill
```

### Building Attention Metadata
```python
# GPU - uses vllm attention builders

# NPU - may need additional parameters
(attn_metadata, spec_decode_common_attn_metadata) = self._build_attention_metadata(
    num_tokens=num_tokens_unpadded,
    num_tokens_padded=num_tokens_padded,
    num_reqs=num_reqs,
    num_reqs_padded=num_reqs_padded,
    max_query_len=max_num_scheduled_tokens,
    ubatch_slices=ubatch_slices_attn,
    logits_indices=logits_indices,
    use_spec_decode=use_spec_decode,
    num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
    num_scheduled_tokens_np=num_scheduled_tokens_np,
    cascade_attn_prefix_lens=cascade_attn_prefix_lens,
)
```

## Rotary Embedding

### Update Cos/Sin Cache
```python
# GPU - typically handled inside attention

# NPU - explicit update required before forward
from vllm_ascend.ops.rotary_embedding import update_cos_sin
update_cos_sin(positions)
```

## Sequence Parallelism

### Enable SP Check
```python
# GPU - use vllm distributed utilities

# NPU - use vllm-ascend wrapper
from vllm_ascend.utils import enable_sp

if enable_sp():
    # sequence parallelism enabled
```

## Sampler

### Sampler Type
```python
# GPU - uses vllm sampler
self.sampler = Sampler()

# NPU - uses AscendSampler
from vllm_ascend.sample.sampler import AscendSampler
self.sampler = AscendSampler()
```

## Input Batch

### Batch Class
```python
# GPU
from vllm.v1.worker.gpu_input_batch import InputBatch

# NPU
from vllm_ascend.worker.npu_input_batch import NPUInputBatch
```

## Graph Parameter Updates

### Full Graph Params Update (NPU-specific)
```python
# GPU - not needed

# NPU - required for FULL graph mode
from vllm_ascend.compilation.acl_graph import update_full_graph_params

forward_context = get_forward_context()
if (
    forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
    and not forward_context.capturing
    and not self.use_sparse
):
    update_full_graph_params(
        self.attn_backend,
        self.update_stream,
        forward_context,
        num_tokens_padded,
        self.vllm_config,
        self.speculative_config,
        positions.shape[0],
    )
```

## Paged Attention Check

```python
# GPU - not typically needed

# NPU
from vllm_ascend.attention.utils import using_paged_attention

if is_graph_capturing and using_paged_attention(num_tokens, self.vllm_config):
    seq_lens = SEQ_LEN_WITH_MAX_PA_WORKSPACE
```

## Common Method Signature Differences

### _dummy_run Parameters
```python
# GPU (v0.17.0)
def _dummy_run(
    self,
    num_tokens: int,
    cudagraph_runtime_mode: CUDAGraphMode | None = None,
    force_attention: bool = False,
    uniform_decode: bool = False,
    allow_microbatching: bool = True,
    skip_eplb: bool = False,
    is_profile: bool = False,
    create_mixed_batch: bool = False,
    remove_lora: bool = True,
    is_graph_capturing: bool = False,
    num_active_loras: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:

# NPU (v0.17.0) - adds with_prefill, activate_lora
def _dummy_run(
    self,
    num_tokens: int,
    with_prefill: bool = False,
    cudagraph_runtime_mode: CUDAGraphMode | None = None,
    force_attention: bool = False,
    uniform_decode: bool = False,
    is_profile: bool = False,
    create_mixed_batch: bool = False,
    allow_microbatching: bool = True,
    skip_eplb: bool = False,
    remove_lora: bool = True,
    activate_lora: bool = False,
    is_graph_capturing: bool = False,
    num_active_loras: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
```

### _model_forward Parameters
```python
# GPU - no num_tokens_padded
def _model_forward(
    self,
    input_ids: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    **model_kwargs: dict[str, Any],
):

# NPU - has num_tokens_padded as first parameter
def _model_forward(
    self,
    num_tokens_padded: int,
    input_ids: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    **model_kwargs: dict[str, Any],
):
```

## Quick Reference Table

| Feature | GPU | NPU |
|---------|-----|-----|
| Graph wrapper | `CUDAGraphWrapper` | `ACLGraphWrapper` |
| Forward context | `set_forward_context` | `set_ascend_forward_context` |
| Runtime mode param | `cudagraph_runtime_mode` | `aclgraph_runtime_mode` |
| Device sync | `torch.cuda.synchronize()` | `torch.npu.synchronize()` |
| Stream | `torch.cuda.Stream` | `torch.npu.Stream` |
| Current stream | `torch.cuda.current_stream()` | `torch.npu.current_stream()` |
| Input batch | `InputBatch` | `NPUInputBatch` |
| Sampler | `Sampler` | `AscendSampler` |
| Attention state | N/A | `AscendAttentionState` |
| RoPE update | N/A | `update_cos_sin()` |
