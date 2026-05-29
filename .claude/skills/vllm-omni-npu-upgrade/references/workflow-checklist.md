# NPU Model Runner Upgrade Workflow Checklist

> **Note**: Reference documents (`omni-specific-blocks.md`) may not be complete. Always grep for `Omni-new` in GPU implementations to find all omni-specific code blocks. Update the reference docs when discovering new blocks.

## Pre-Upgrade Preparation

### 1. Version Information
- [ ] Identify current vllm-omni version: `_________`
- [ ] Identify target vllm-ascend version: `_________`
- [ ] Identify target vllm version: `_________`
- [ ] Last release date for GPU worker changes: `_________`

### 2. Gather Git History
```bash
# GPU-side omni changes since last release
cd /root/vllm-workspace/vllm-omni
git log --oneline --since="YYYY-MM-DD" -- vllm_omni/worker/

# vllm-ascend NPUModelRunner changes
cd /root/vllm-workspace/vllm-ascend
git log --oneline <from-tag>..<to-tag> -- vllm_ascend/worker/model_runner_v1.py
```

### 3. Backup Current Files
- [ ] Create backup of current NPU runners:
  ```bash
  cp -r vllm_omni/platforms/npu/worker vllm_omni/platforms/npu/worker.backup
  ```

---

## OmniNPUModelRunner (npu_model_runner.py)

### Read and Understand
- [ ] Read current `npu_model_runner.py`
- [ ] Read latest `vllm_ascend/worker/model_runner_v1.py`
- [ ] Read latest `vllm_omni/worker/gpu_model_runner.py`

### Method: load_model
- [ ] Document existing omni-specific logic
- [ ] Copy latest NPUModelRunner.load_model structure
- [ ] Re-insert: `enable_sp(self.vllm_config)` call
- [ ] Re-insert: talker_mtp detection and setup
- [ ] Replace: `CUDAGraphWrapper` → `ACLGraphWrapper`
- [ ] Re-insert: Buffer allocations (talker_mtp_input_ids, etc.)

### Method: _dummy_run
- [ ] Document existing omni-specific logic locations
- [ ] Copy latest NPUModelRunner._dummy_run
- [ ] Re-insert: talker_mtp dummy forward block (inside context)
- [ ] Re-insert: `extract_multimodal_outputs` call
- [ ] Verify: Comment markers are present

### Method: _model_forward
- [ ] Copy latest NPUModelRunner._model_forward structure
- [ ] Re-insert: `_build_model_kwargs_extra()` call
- [ ] Re-insert: OmniOutput wrapping logic
- [ ] Re-insert: `_omni_last_model_output` caching
- [ ] Keep: NPU graph params update
- [ ] Keep: SP all-gather logic

### Method: _talker_mtp_forward
- [ ] Verify: Uses `set_ascend_forward_context`
- [ ] Verify: Uses `ACLGraphWrapper` check
- [ ] Sync any changes from GPU `_talker_mtp_forward`

### Imports
- [ ] Update vllm-ascend imports to latest paths
- [ ] Verify all omni imports are present
- [ ] Remove any deprecated imports

---

## NPUARModelRunner (npu_ar_model_runner.py)

### Read and Understand
- [ ] Read current `npu_ar_model_runner.py`
- [ ] Read latest `vllm_ascend/worker/model_runner_v1.py` execute_model
- [ ] Read latest `vllm_omni/worker/gpu_ar_model_runner.py`

### Method: __init__
- [ ] Sync any new initialization from GPU side
- [ ] Keep: `OmniKVTransferManager` setup
- [ ] Keep: Custom buffer allocations

### Method: execute_model
- [ ] Document all omni blocks with line numbers
- [ ] Copy latest NPUModelRunner.execute_model structure
- [ ] Re-insert: KV transfer handling (beginning)
- [ ] Re-insert: Custom `_update_states` call
- [ ] Re-insert: `extract_multimodal_outputs`
- [ ] Re-insert: `compute_logits` with sampling_metadata try/except
- [ ] Update: ExecuteModelState to include multimodal_outputs

### Method: sample_tokens
- [ ] Document all omni blocks
- [ ] Copy latest NPUModelRunner.sample_tokens structure
- [ ] Re-insert: `kv_extracted_req_ids` handling
- [ ] Re-insert: Hidden states CPU copy
- [ ] Re-insert: `_process_additional_information_updates`
- [ ] Re-insert: `OmniModelRunnerOutput` construction

### ExecuteModelState
- [ ] Verify: `multimodal_outputs` field is present
- [ ] Verify: Imported/used correctly in execute_model

### Imports
- [ ] Update all vllm-ascend imports
- [ ] Keep omni-specific imports

---

## NPUGenerationModelRunner (npu_generation_model_runner.py)

### Read and Understand
- [ ] Read current `npu_generation_model_runner.py`
- [ ] Read latest GPU `gpu_generation_model_runner.py`

### Method: _update_request_states
- [ ] Verify: async_chunk handling is correct
- [ ] Sync any changes from GPU side

### Method: execute_model
- [ ] Document all omni blocks
- [ ] Copy latest NPUModelRunner.execute_model base structure
- [ ] Re-insert: async_chunk update logic
- [ ] Re-insert: `seq_token_counts` injection
- [ ] Re-insert: `_run_generation_model` call
- [ ] Re-insert: `extract_multimodal_outputs`
- [ ] Use: ExecuteModelState from npu_ar_model_runner

### Method: sample_tokens
- [ ] Keep: Entire omni multimodal output processing
- [ ] Update: Any new output fields needed
- [ ] Keep: `OmniModelRunnerOutput` construction

### Method: _run_generation_model
- [ ] Sync any changes from GPU side
- [ ] Keep: `_model_forward` call with sampler

### Method: _dummy_run
- [ ] Copy latest NPUModelRunner._dummy_run
- [ ] Re-insert: `model_kwargs = self._init_model_kwargs()`
- [ ] Re-insert: `extract_multimodal_outputs` at end

### Imports
- [ ] Import ExecuteModelState from npu_ar_model_runner
- [ ] Update vllm-ascend imports

---

## Post-Upgrade Validation

### Syntax Validation
- [ ] `python -m py_compile vllm_omni/platforms/npu/worker/npu_model_runner.py`
- [ ] `python -m py_compile vllm_omni/platforms/npu/worker/npu_ar_model_runner.py`
- [ ] `python -m py_compile vllm_omni/platforms/npu/worker/npu_generation_model_runner.py`

### Import Validation
- [ ] `python -c "from vllm_omni.platforms.npu.worker.npu_model_runner import OmniNPUModelRunner"`
- [ ] `python -c "from vllm_omni.platforms.npu.worker.npu_ar_model_runner import NPUARModelRunner"`
- [ ] `python -c "from vllm_omni.platforms.npu.worker.npu_generation_model_runner import NPUGenerationModelRunner"`

### Comment Markers
- [ ] Grep for "Omni-new" in all three files
- [ ] Verify all omni blocks have closing markers

### Code Review
- [ ] No `CUDAGraphWrapper` references
- [ ] All `set_forward_context` replaced with `set_ascend_forward_context`
- [ ] Parameter names correct (`aclgraph_runtime_mode` not `cudagraph_runtime_mode`)
- [ ] No duplicate code blocks
- [ ] No missing imports

---

## Git Commit

### Commit Message Template
```
[NPU] Upgrade model runners to align with vllm-ascend vX.Y.Z

- Update OmniNPUModelRunner with latest NPUModelRunner base
- Update NPUARModelRunner execute_model and sample_tokens
- Update NPUGenerationModelRunner for async_chunk changes
- Sync GPU-side omni changes from vX.Y.Z release
- Preserve all omni-specific logic (marked with Omni-new comments)

Changes from vllm-ascend:
- <list key changes>

Changes synced from GPU:
- <list key GPU-side omni changes>
```

### Files to Stage
- [ ] `vllm_omni/platforms/npu/worker/npu_model_runner.py`
- [ ] `vllm_omni/platforms/npu/worker/npu_ar_model_runner.py`
- [ ] `vllm_omni/platforms/npu/worker/npu_generation_model_runner.py`
- [ ] Any other modified files

---

## Troubleshooting

### Import Errors
- Check if vllm-ascend module paths have changed
- Verify PYTHONPATH includes both vllm-ascend and vllm-omni

### Type Errors
- Check method signatures match between GPU and NPU
- Verify NamedTuple fields match expected structure

### Runtime Errors
- Enable debug logging: `export VLLM_LOGGING_LEVEL=DEBUG`
- Check graph capture issues: try `--enforce-eager`
- Check attention issues: verify AscendAttentionState usage

### Performance Regression
- Compare with previous version on same model
- Check if graph capture is working: look for ACLGraph logs
- Verify SP/EP configurations are correct
