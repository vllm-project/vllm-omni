# Specific Code Fixes for PR #17

This document provides specific code changes needed to address the review comments.

## Fix 1: Remove/Fix Hardcoded Debugging Code

### File: `vllm_omni/worker/gpu_model_runner.py`

**Current Code (Lines ~544-545)**:
```python
# Get the valid generated tokens.
import os
sampled_token_ids = sampler_output.sampled_token_ids if os.environ.get("model_stage") != "code2wav" else torch.tensor([[8294]]).to(torch.int32).cuda()
max_gen_len = sampled_token_ids.shape[-1]
```

**Option 1 - Remove Debugging Code (Recommended)**:
```python
# Get the valid generated tokens.
sampled_token_ids = sampler_output.sampled_token_ids
max_gen_len = sampled_token_ids.shape[-1]
```

**Option 2 - If Code is Required, Properly Implement**:
```python
# At module level (after other imports)
import os

# Define constant at module level
# Special token ID used for code2wav model stage
CODE2WAV_END_TOKEN = 8294

# In the function (around line 544)
# Get the valid generated tokens.
# TODO: Investigate why code2wav stage needs special handling
# This is a temporary workaround - should be fixed in the model itself
if os.environ.get("model_stage") == "code2wav":
    logger.debug("Using special token handling for code2wav stage")
    sampled_token_ids = torch.tensor(
        [[CODE2WAV_END_TOKEN]], dtype=torch.int32, device="cuda"
    )
else:
    sampled_token_ids = sampler_output.sampled_token_ids
max_gen_len = sampled_token_ids.shape[-1]
```

---

## Fix 2 & 3: Remove Redundant NumPy Imports

### File: `vllm_omni/worker/gpu_model_runner.py`

**Current Code (Line ~126)**:
```python
try:
    if getattr(new_req_data, "prompt_embeds", None) is not None:
        payload = new_req_data.prompt_embeds
        import numpy as np  # <-- REMOVE THIS LINE
        dtype = getattr(np, payload.dtype)
        # ... rest of code
```

**Fixed Code**:
```python
try:
    if getattr(new_req_data, "prompt_embeds", None) is not None:
        payload = new_req_data.prompt_embeds
        # np is already imported at module level
        dtype = getattr(np, payload.dtype)
        # ... rest of code
```

**Current Code (Line ~150)**:
```python
else:
    from vllm.v1.engine import AdditionalInformationPayload
    if isinstance(payload_info, AdditionalInformationPayload):
        import numpy as np  # <-- REMOVE THIS LINE
        for k, entry in payload_info.entries.items():
            # ... rest of code
```

**Fixed Code**:
```python
else:
    from vllm.v1.engine import AdditionalInformationPayload
    if isinstance(payload_info, AdditionalInformationPayload):
        # np is already imported at module level
        for k, entry in payload_info.entries.items():
            # ... rest of code
```

---

## Fix 4: Fix Return Type Annotation

### File: `vllm_omni/worker/gpu_model_runner.py`

**Current Code (Line ~616)**:
```python
@torch.inference_mode()
def extract_multimodal_outputs(self, hidden_states: torch.Tensor) -> dict:
    if hasattr(self.model, "have_multimodal_outputs") and self.model.have_multimodal_outputs:
        text_hidden_states = hidden_states.text_hidden_states
        multimodal_outputs = hidden_states.multimodal_outputs

    else:
        text_hidden_states = hidden_states
        multimodal_outputs = {}
    return text_hidden_states, multimodal_outputs
```

**Fixed Code**:
```python
@torch.inference_mode()
def extract_multimodal_outputs(
    self, hidden_states: torch.Tensor
) -> tuple[torch.Tensor, dict]:
    """Extract text hidden states and multimodal outputs from model output.
    
    Args:
        hidden_states: Model output containing hidden states
        
    Returns:
        A tuple of (text_hidden_states, multimodal_outputs)
    """
    if hasattr(self.model, "have_multimodal_outputs") and self.model.have_multimodal_outputs:
        text_hidden_states = hidden_states.text_hidden_states
        multimodal_outputs = hidden_states.multimodal_outputs
    else:
        text_hidden_states = hidden_states
        multimodal_outputs = {}
    return text_hidden_states, multimodal_outputs
```

---

## Fix 5: Improve Warning Message

### File: `vllm_omni/worker/gpu_model_runner.py`

**Current Code (Line ~739)**:
```python
logger.warning(f"Multimodal outputs are not returned in the dummy run, need to double check the implementation!")
text_hidden_states, multimodal_outputs = self.extract_multimodal_outputs(hidden_states)
```

**Option 1 - If this is expected behavior**:
```python
logger.info("Multimodal outputs are not returned in dummy runs. This is expected behavior.")
text_hidden_states, multimodal_outputs = self.extract_multimodal_outputs(hidden_states)
```

**Option 2 - If implementation is incomplete**:
```python
# TODO: Implement multimodal output handling for dummy runs
logger.debug("Multimodal outputs extraction skipped in dummy run")
text_hidden_states, multimodal_outputs = self.extract_multimodal_outputs(hidden_states)
```

**Option 3 - Remove the message if it adds no value**:
```python
text_hidden_states, multimodal_outputs = self.extract_multimodal_outputs(hidden_states)
```

---

## Optional Fix: List Initialization (if desired)

### File: `vllm_omni/worker/model_runner.py`

**Current Code (Lines ~26-28)**:
```python
# Combine and flatten intermediate data.
input_tokens = list[int]()
inputs_embeds_list = list[torch.Tensor]()
token_types = list[int]()
```

**Alternative (Standard Python)**:
```python
# Combine and flatten intermediate data.
input_tokens: list[int] = []
inputs_embeds_list: list[torch.Tensor] = []
token_types: list[int] = []
```

**Note**: The PR author stated this follows vLLM's original implementation style. Only change if you want to deviate from vLLM's style for better clarity.

---

## Summary of Changes

| Fix # | File | Priority | Lines | Description |
|-------|------|----------|-------|-------------|
| 1 | gpu_model_runner.py | HIGH | 544-545 | Remove/fix hardcoded debugging code |
| 2 | gpu_model_runner.py | MEDIUM | 126 | Remove redundant numpy import |
| 3 | gpu_model_runner.py | MEDIUM | 150 | Remove redundant numpy import |
| 4 | gpu_model_runner.py | LOW | 616 | Fix return type annotation |
| 5 | gpu_model_runner.py | LOW | 739 | Improve warning message |
| 6 | model_runner.py | OPTIONAL | 26-28 | Modernize list initialization |

---

## Testing After Fixes

After making these changes, ensure:

1. Run linting:
   ```bash
   python -m flake8 vllm_omni/worker/gpu_model_runner.py
   python -m flake8 vllm_omni/worker/model_runner.py
   python -m mypy vllm_omni/worker/gpu_model_runner.py
   python -m mypy vllm_omni/worker/model_runner.py
   ```

2. Run existing tests:
   ```bash
   pytest tests/ -v
   ```

3. Test the specific functionality:
   - Test multimodal model execution
   - Test prompt embeddings handling
   - Test additional_information payload processing
