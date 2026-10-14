---
name: vllm-omni-npu-model-runner-upgrade
description: "Upgrade vllm-omni NPU model runners (OmniNPUModelRunner, NPUARModelRunner, NPUGenerationModelRunner) to align with the latest vllm-ascend NPUModelRunner while preserving omni-specific logic."
---

# vLLM-Omni NPU Model Runner Upgrade Skill

## Overview

This skill guides the process of upgrading vllm-omni's NPU model runners to align with the latest vllm-ascend codebase while preserving omni-specific enhancements. The NPU runners are designed to run omni multimodal models (like Qwen3-Omni, Bagel, MiMoAudio) on Ascend NPUs.

## File Structure

### NPU Model Runner Files
```
vllm-omni/vllm_omni/platforms/npu/worker/
├── __init__.py
├── npu_model_runner.py           # OmniNPUModelRunner (base class)
├── npu_ar_model_runner.py        # NPUARModelRunner (autoregressive)
├── npu_ar_worker.py              # AR worker
├── npu_generation_model_runner.py # NPUGenerationModelRunner (diffusion/non-AR)
└── npu_generation_worker.py      # Generation worker
```

### GPU Reference Files (for omni-specific logic sync)
```
vllm-omni/vllm_omni/worker/
├── __init__.py
├── gpu_model_runner.py           # OmniGPUModelRunner
├── gpu_ar_model_runner.py        # GPUARModelRunner
├── gpu_ar_worker.py
├── gpu_generation_model_runner.py
├── gpu_generation_worker.py
├── mixins.py
├── base.py
└── gpu_memory_utils.py
```

### vllm-ascend Reference Files
```
vllm-ascend/vllm_ascend/worker/
├── model_runner_v1.py            # NPUModelRunner (base class to copy from)
├── npu_input_batch.py
├── block_table.py
├── pcp_utils.py
└── worker.py
```

## Inheritance Hierarchy

```
                    GPUModelRunner (vllm)
                         |
        +----------------+----------------+
        |                                 |
  OmniGPUModelRunner              NPUModelRunner (vllm-ascend)
  (vllm_omni/worker)              (vllm_ascend/worker)
        |                                 |
        +----------- OmniNPUModelRunner --+
                     (multiple inheritance)
                            |
            +---------------+---------------+
            |                               |
    NPUARModelRunner            NPUGenerationModelRunner
    (autoregressive)            (non-autoregressive/diffusion)
```

## Omni-Specific Comment Markers

Omni-specific logic is marked with comment blocks:
```python
# -------------------------------------- Omni-new -------------------------------------------------
# ... omni-specific code ...
# -------------------------------------- Omni-new -------------------------------------------------
```

Or simpler variations:
```python
#  -------------------------------------- Omni-new -------------------------------------------------
#  ------------------------------------------------------------------------------------------------
```

**Important**:
- Always preserve and add these markers when modifying code.
- **The reference documents (`references/omni-specific-blocks.md`) may not be up-to-date.** Always grep for `Omni-new` in the GPU implementations to find the authoritative list of omni-specific blocks.
- When you discover new omni-specific code that is not documented in the references, please update the reference files.

## Key Methods Requiring Attention

### OmniNPUModelRunner (npu_model_runner.py)

| Method | Description | Omni-Specific Logic |
|--------|-------------|---------------------|
| `load_model` | Load model and initialize talker_mtp | Uses `ACLGraphWrapper` instead of `CUDAGraphWrapper`, initializes talker buffers |
| `_dummy_run` | Warmup/profiling run | talker_mtp dummy forward, `extract_multimodal_outputs` |
| `_model_forward` | Forward pass wrapper | Injects `model_kwargs_extra`, wraps with `OmniOutput`, NPU-specific graph updates |
| `_talker_mtp_forward` | Talker MTP forward for Qwen3-Omni | Uses `set_ascend_forward_context` |

### NPUARModelRunner (npu_ar_model_runner.py)

| Method | Description | Omni-Specific Logic |
|--------|-------------|---------------------|
| `__init__` | Initialize with KV transfer manager | `OmniKVTransferManager` setup |
| `execute_model` | Main inference entry | KV transfer handling, `_update_states` override, `extract_multimodal_outputs` |
| `sample_tokens` | Token sampling | Hidden states extraction, multimodal outputs processing, `OmniModelRunnerOutput` |
| `_resolve_global_request_id` | Request ID resolution | For disaggregated inference |

### NPUGenerationModelRunner (npu_generation_model_runner.py)

| Method | Description | Omni-Specific Logic |
|--------|-------------|---------------------|
| `_update_request_states` | Update request states for async chunk | async_chunk handling |
| `execute_model` | Generation forward | async_chunk, `seq_token_counts`, `_run_generation_model` |
| `sample_tokens` | Output processing | multimodal output packaging to `OmniModelRunnerOutput` |
| `_dummy_run` | Dummy run override | model_kwargs initialization, multimodal extraction |
| `_run_generation_model` | Run generation model | Calls `_model_forward` with sampler |

## Upgrade Workflow

### Step 1: Preparation

1. **Identify target versions**(Use gh cli to check):
   - We're using vllm-omni main branch
   - Check the last release of vllm-omni
   - Target vllm-ascend version(Just directly use the local latest vllm-ascend code)

2. **Check GPU-side changes** (since last release):
   ```bash
   cd /root/vllm-workspace/vllm-omni
   git log --oneline --since="<last-release-date>" -- vllm_omni/worker/
   ```

3. **Read latest vllm-ascend code**:
   - We don't track vllm-ascend changes - just directly use the latest code from `/root/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py`
   - Copy the relevant methods and re-insert omni-specific blocks

### Step 2: Analyze Omni-Specific Logic

For each NPU model runner file:

1. **Extract existing omni-specific blocks**:
   ```bash
   grep -n "Omni-new" vllm_omni/platforms/npu/worker/npu_model_runner.py
   ```

2. **Document each omni block**:
   - Which method it belongs to
   - What functionality it provides
   - Dependencies on other omni code

### Step 3: Update Base Class (OmniNPUModelRunner)

**Note**: Always check the GPU implementation `gpu_model_runner.py` for any new omni logic not yet documented in references.

1. **Read the latest vllm-ascend `NPUModelRunner.load_model`**
2. **Copy the method, keeping the structure**
3. **Re-insert omni-specific logic** (check GPU `gpu_model_runner.py` for authoritative list):
   - Replace `CUDAGraphWrapper` with `ACLGraphWrapper`
   - Keep talker_mtp initialization
   - Preserve buffer allocations for talker
   - Check for any new omni blocks added since last sync

4. **Update `_dummy_run`**:
   - Copy from vllm-ascend
   - Compare with GPU `_dummy_run` for omni-specific blocks
   - Re-insert all `Omni-new` marked code from GPU version

5. **Update `_model_forward`**:
   - Keep the omni wrapper logic
   - Update NPU-specific parts (graph params, SP all-gather)
   - Check GPU version for any new omni logic

### Step 4: Update AR Model Runner

1. **Compare with GPU `gpu_ar_model_runner.py`** for any new omni features
2. **Copy `execute_model` from vllm-ascend**
3. **Re-insert omni blocks** (reference `references/omni-specific-blocks.md`, but note it may be incomplete):
   - **IMPORTANT**: Always check the GPU implementation `gpu_ar_model_runner.py` for all `Omni-new` marked code blocks
   - The reference doc may not include newly added omni logic - treat it as a starting point, not exhaustive
   - When discovering new omni code blocks, please update `references/omni-specific-blocks.md`
   - Common omni blocks include but are not limited to: KV transfer, multimodal outputs, sampling_metadata handling, etc.

4. **Update `sample_tokens`** (also compare with GPU implementation):
   - Compare with `gpu_ar_model_runner.py`'s `sample_tokens` method
   - Identify all `Omni-new` marked code blocks
   - Ensure NPU version includes all omni-specific logic

### Step 5: Update Generation Model Runner

**Note**: Generation model runner may have unique omni logic for diffusion/non-AR models.

1. **Compare with GPU `gpu_generation_model_runner.py`** - grep for all `Omni-new` blocks
2. **Update `execute_model`**:
   - Check GPU version for all omni-specific blocks
   - Keep async_chunk handling
   - Keep `seq_token_counts` injection
   - Update forward/context setup from vllm-ascend
   - Look for any new omni logic not documented in references

3. **Update `_dummy_run`**:
   - Copy from vllm-ascend base
   - Compare with GPU `_dummy_run` if exists
   - Re-insert all omni-specific logic

### Step 6: Update Imports

Check and update imports at the top of each file:

```python
# Common vllm-ascend imports
from vllm_ascend.ascend_forward_context import get_forward_context, set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import using_paged_attention
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper, update_full_graph_params
from vllm_ascend.ops.rotary_embedding import update_cos_sin
from vllm_ascend.utils import enable_sp, lmhead_tp_enable
from vllm_ascend.worker.model_runner_v1 import SEQ_LEN_WITH_MAX_PA_WORKSPACE, NPUModelRunner

# Omni-specific imports
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import OmniKVTransferManager
```

### Step 7: Sync GPU-Side Omni Changes

1. **Check recent GPU worker changes**:
   ```bash
   git diff <from-tag>..<to-tag> -- vllm_omni/worker/gpu_model_runner.py
   git diff <from-tag>..<to-tag> -- vllm_omni/worker/gpu_ar_model_runner.py
   ```

2. **Identify new omni features** that need to be ported to NPU

3. **Apply corresponding changes** to NPU runners

### Step 8: Validation

1. **Run type checking**:
   ```bash
   cd /root/vllm-workspace/vllm-omni
   python -m py_compile vllm_omni/platforms/npu/worker/npu_model_runner.py
   python -m py_compile vllm_omni/platforms/npu/worker/npu_ar_model_runner.py
   python -m py_compile vllm_omni/platforms/npu/worker/npu_generation_model_runner.py
   ```

2. **Run import test**:
   ```bash
   python -c "from vllm_omni.platforms.npu.worker import *"
   ```

3. **Run model serving test** (if hardware available):
   ```bash
   vllm serve <model-path> --trust-remote-code
   ```

## Common Pitfalls

### 1. Forward Context Differences
- GPU uses `set_forward_context`
- NPU uses `set_ascend_forward_context`
- Parameters may differ slightly

### 2. Graph Wrapper Differences
- GPU: `CUDAGraphWrapper`
- NPU: `ACLGraphWrapper`
- Constructor parameters may differ

### 3. Buffer Creation
- GPU: `_make_buffer` returns different structure
- NPU: May need numpy=True/False parameter

### 4. Attention Metadata
- GPU: Uses vllm attention metadata builders
- NPU: Uses `AscendCommonAttentionMetadata`

### 5. Sampling
- GPU: Uses vllm sampler
- NPU: Uses `AscendSampler`

## Checklist Before Commit

- [ ] All omni-specific comment markers preserved
- [ ] New omni logic from GPU side synced
- [ ] Imports updated to latest vllm-ascend
- [ ] No `CUDAGraphWrapper` references in NPU code
- [ ] `set_ascend_forward_context` used instead of `set_forward_context`
- [ ] `ACLGraphWrapper` used for talker_mtp wrapping
- [ ] Type hints match vllm-ascend signatures
- [ ] No duplicate code blocks
- [ ] Python syntax valid (py_compile passes)

## Reference Files for Comparison

When upgrading, keep these files open for reference:

1. **vllm-ascend NPUModelRunner**: `/root/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py`
2. **vllm GPUModelRunner**: `/root/vllm-workspace/vllm/vllm/v1/worker/gpu_model_runner.py`
3. **vllm-omni OmniGPUModelRunner**: `/root/vllm-workspace/vllm-omni/vllm_omni/worker/gpu_model_runner.py`
