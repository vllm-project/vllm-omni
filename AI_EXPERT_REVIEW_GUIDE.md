# AI Expert Review Guidelines for vLLM-Omni

## Overview

This document provides expert-level guidance for reviewing PRs in the vLLM-omni project from an AI/ML systems perspective. It focuses on domain-specific concerns that general software engineers might overlook.

## 1. Deep Learning Systems Expertise

### 1.1 Tensor Operations
When reviewing code that manipulates tensors:

**Key Questions:**
- Are tensor shapes validated before operations?
- Is broadcasting behavior explicit and correct?
- Are reshape operations safe (no hidden bugs)?
- Is memory layout (contiguous vs. non-contiguous) handled properly?

**Common Issues:**
```python
# ❌ Bad: Assumes shape without validation
logits = model_output.view(-1, vocab_size)  # What if batch_size=0?

# ✅ Good: Validates and handles edge cases
batch_size, seq_len, hidden_dim = model_output.shape
if batch_size == 0:
    return empty_result()
logits = model_output.view(batch_size * seq_len, vocab_size)
```

**Red Flags:**
- Hardcoded tensor dimensions
- Implicit assumptions about batch size
- Missing shape checks
- Unnecessary `.contiguous()` calls (performance)

### 1.2 Numerical Precision
Modern ML uses mixed precision (fp32, fp16, bf16, int8). Review carefully:

**Precision Considerations:**
- **fp16**: Faster, less memory, but limited range (6e-8 to 65,504)
- **bf16**: Better range than fp16, preferred for training
- **fp32**: Full precision, needed for numerical stability
- **int8**: Quantization, 4x memory reduction

**Review Checklist:**
```python
# Check for precision-sensitive operations
- [ ] Loss calculations in fp32?
- [ ] Softmax numerically stable?
- [ ] Layer norm in higher precision?
- [ ] Gradients not underflowing?
- [ ] Quantization/dequantization correct?
```

**Example Issue:**
```python
# ❌ Problematic: Softmax in fp16 can overflow/underflow
logits_fp16 = logits.half()
probs = F.softmax(logits_fp16, dim=-1)  # Risk of NaN

# ✅ Better: Softmax in fp32
probs = F.softmax(logits.float(), dim=-1).half()

# ✅ Best: Use built-in numerical stability
probs = F.softmax(logits, dim=-1)  # PyTorch handles this
```

### 1.3 Memory Management
GPU memory is precious. Review memory efficiency:

**Memory Patterns:**
```python
# ❌ Bad: Creates unnecessary copies
x_copy = x.clone()
x_copy = x_copy + 1
x_copy = x_copy * 2

# ✅ Good: In-place operations
x.add_(1)
x.mul_(2)

# ❌ Bad: Keeps gradients unnecessarily
with torch.no_grad():  # But still creates computation graph sometimes
    y = model(x)

# ✅ Good: Explicit gradient management
torch.set_grad_enabled(False)
y = model(x)
torch.set_grad_enabled(True)
```

**Review Points:**
- Are temporary tensors freed promptly?
- Is `torch.no_grad()` used for inference?
- Are large tensors kept on CPU when not needed?
- Is gradient checkpointing considered for large models?

## 2. Inference Engine Expertise

### 2.1 KV-Cache Management (Autoregressive Models)
The KV-cache is critical for transformer inference performance:

**Key Aspects:**
```python
# Cache structure
k_cache: (batch_size, num_heads, max_seq_len, head_dim)
v_cache: (batch_size, num_heads, max_seq_len, head_dim)

# Review checklist
- [ ] Cache size bounded (prevent OOM)?
- [ ] Cache reuse logic correct?
- [ ] Cache eviction policy appropriate?
- [ ] Multi-request batching handled?
- [ ] Cache invalidation on error?
```

**Common Bugs:**
```python
# ❌ Bug: Off-by-one in cache indexing
cache[:, :, position] = new_kv  # Wrong if position is length

# ✅ Correct:
cache[:, :, position:position+1] = new_kv

# ❌ Bug: Cache not cleared between requests
def generate(input_ids):
    # Forgets to reset cache from previous request
    output = model(input_ids, past_key_values=self.cache)

# ✅ Correct:
def generate(input_ids):
    self.cache = None  # or appropriate reset
    output = model(input_ids, past_key_values=self.cache)
```

### 2.2 Batching Strategy
Efficient batching is crucial for throughput:

**Dynamic Batching:**
```python
# Review considerations
- [ ] Padding minimized (wasted computation)?
- [ ] Attention masks correct?
- [ ] Different sequence lengths handled?
- [ ] Batch size determination optimal?
- [ ] Timeout for batch assembly?
```

**Continuous Batching (PagedAttention style):**
```python
# Advanced batching for vLLM
- [ ] Request scheduling fair?
- [ ] Memory fragmentation minimized?
- [ ] Preemption handled correctly?
- [ ] Mixed-length batches optimized?
```

### 2.3 Diffusion Model Specifics (DiT)
For non-autoregressive diffusion models:

**Timestep Handling:**
```python
# ❌ Common mistake: Incorrect noise scheduling
timesteps = torch.linspace(0, 1, num_steps)  # Linear is often wrong!

# ✅ Better: Use appropriate schedule
timesteps = get_schedule(num_steps, schedule_type='cosine')

# Review checklist for DiT
- [ ] Timestep embedding correct?
- [ ] Noise schedule appropriate?
- [ ] Denoising steps sufficient?
- [ ] Classifier-free guidance implemented correctly?
- [ ] Latent space dimensions correct?
```

**Iterative Refinement:**
```python
# Key points
- [ ] Number of inference steps reasonable?
- [ ] Early stopping criteria defined?
- [ ] Intermediate results cached if needed?
- [ ] Batch processing across timesteps?
```

## 3. Multi-Modal ML Expertise

### 3.1 Cross-Modal Alignment
When models process multiple modalities:

**Alignment Considerations:**
```python
# Text-Image alignment
- [ ] Embeddings in same dimension space?
- [ ] Projection layers appropriate?
- [ ] Attention masks account for different lengths?
- [ ] Positional encodings compatible?

# Example
text_embed: (batch, text_len, hidden_dim)
image_embed: (batch, num_patches, hidden_dim)  # Must match hidden_dim

# ❌ Wrong: Concatenate without consideration
combined = torch.cat([text_embed, image_embed], dim=1)  
# Loses information about modality boundaries!

# ✅ Better: Add modality indicators
text_with_type = text_embed + text_type_embedding
image_with_type = image_embed + image_type_embedding
combined = torch.cat([text_with_type, image_with_type], dim=1)
```

### 3.2 Modality-Specific Preprocessing
Each modality has unique requirements:

**Image Processing:**
```python
- [ ] Resolution handling (resize vs. crop)?
- [ ] Normalization (ImageNet stats or custom)?
- [ ] Channel order (RGB vs. BGR)?
- [ ] Patch embedding correct?
- [ ] Aspect ratio preservation?
```

**Audio Processing:**
```python
- [ ] Sample rate conversion handled?
- [ ] Spectrogram parameters appropriate?
- [ ] Window/hop length correct?
- [ ] Mel-scale transformation if needed?
- [ ] Audio length padding/truncation?
```

**Video Processing:**
```python
- [ ] Frame sampling strategy (uniform vs. adaptive)?
- [ ] Temporal information preserved?
- [ ] Frame rate normalization?
- [ ] Spatial-temporal encoding?
```

## 4. Performance Optimization Expertise

### 4.1 Kernel Fusion
Efficient GPU utilization requires kernel fusion:

```python
# ❌ Inefficient: Multiple kernel launches
x = x + bias
x = F.relu(x)
x = F.dropout(x, p=0.1)

# ✅ Better: Fused operations (if available)
x = F.fused_bias_relu_dropout(x, bias, p=0.1)

# Review
- [ ] Custom CUDA kernels justified?
- [ ] Fusion opportunities identified?
- [ ] Memory access patterns optimized?
```

### 4.2 Parallelization
For multi-GPU/distributed inference:

```python
# Model parallelism considerations
- [ ] Layer placement strategy clear?
- [ ] Communication overhead minimized?
- [ ] Pipeline parallelism implemented correctly?
- [ ] Gradient synchronization (if training)?

# Data parallelism
- [ ] Balanced data distribution?
- [ ] Efficient all-reduce operations?
- [ ] Gradient accumulation correct?
```

### 4.3 Quantization
Lower precision can dramatically improve performance:

```python
# Post-training quantization
- [ ] Calibration data representative?
- [ ] Per-channel vs. per-tensor quantization?
- [ ] Quantization-aware operations used?
- [ ] Accuracy degradation acceptable?

# Example
# ✅ Proper quantization workflow
model_fp32 = load_model()
calibration_data = get_calibration_data()
model_int8 = quantize(model_fp32, calibration_data)
validate_accuracy(model_int8, threshold=0.99)  # Check degradation
```

## 5. Architecture-Specific Concerns

### 5.1 Transformer Variants
Different transformer architectures have different requirements:

**Decoder-Only (GPT-style):**
```python
- [ ] Causal masking correct?
- [ ] Position embeddings appropriate (absolute/relative/RoPE)?
- [ ] KV-cache structure matches architecture?
```

**Encoder-Decoder (T5-style):**
```python
- [ ] Cross-attention implemented correctly?
- [ ] Separate caches for encoder/decoder?
- [ ] Prefix caching for encoder?
```

**Vision Transformer (ViT):**
```python
- [ ] Patch extraction correct?
- [ ] CLS token handling?
- [ ] Positional embeddings for 2D structure?
```

### 5.2 Non-Transformer Architectures
vLLM-omni supports various architectures:

**Diffusion Transformers (DiT):**
```python
- [ ] Conditional generation handled?
- [ ] Timestep embedding injection points correct?
- [ ] Noise prediction vs. data prediction clear?
```

**Hybrid Models (AR + DiT):**
```python
- [ ] Stage transitions well-defined?
- [ ] Intermediate representations formatted correctly?
- [ ] Resource allocation per stage appropriate?
```

## 6. Testing from AI Perspective

### 6.1 Numerical Tests
```python
def test_numerical_stability():
    """Test model with edge case inputs"""
    # Test with zeros
    output = model(torch.zeros_like(input))
    assert not torch.isnan(output).any()
    
    # Test with large values
    output = model(torch.ones_like(input) * 1000)
    assert torch.isfinite(output).all()
    
    # Test precision
    output_fp32 = model_fp32(input)
    output_fp16 = model_fp16(input)
    assert torch.allclose(output_fp32, output_fp16, rtol=1e-2)
```

### 6.2 Correctness Tests
```python
def test_inference_correctness():
    """Validate inference against known outputs"""
    # Test with reference implementation
    ref_output = reference_model(input)
    our_output = our_model(input)
    assert torch.allclose(ref_output, our_output, rtol=1e-5)
    
    # Test determinism
    torch.manual_seed(42)
    output1 = model(input)
    torch.manual_seed(42)
    output2 = model(input)
    assert torch.equal(output1, output2)
```

### 6.3 Performance Tests
```python
def test_performance():
    """Benchmark latency and throughput"""
    # Warmup
    for _ in range(10):
        _ = model(sample_input)
    
    # Measure
    start = time.time()
    for _ in range(100):
        output = model(sample_input)
    latency = (time.time() - start) / 100
    
    assert latency < target_latency
    
    # Memory
    peak_memory = torch.cuda.max_memory_allocated()
    assert peak_memory < memory_budget
```

## 7. Security & Safety (AI-Specific)

### 7.1 Input Validation
```python
# Prevent adversarial inputs
- [ ] Input size limits enforced?
- [ ] Token limits respected?
- [ ] Image resolution bounded?
- [ ] Audio length limited?

# Example
def validate_input(text):
    if len(text) > MAX_INPUT_LENGTH:
        raise ValueError(f"Input too long: {len(text)} > {MAX_INPUT_LENGTH}")
    if contains_special_tokens(text):
        raise ValueError("Input contains special tokens")
```

### 7.2 Output Safety
```python
# Prevent harmful outputs
- [ ] Content filtering applied?
- [ ] Output length limited?
- [ ] Rate limiting per user?
- [ ] Logging for audit?
```

## 8. Documentation Requirements

For AI/ML PRs, documentation should cover:

```markdown
### Model Changes
- Architecture modifications and rationale
- Performance impact (latency, throughput, memory)
- Accuracy impact (if applicable)
- Compatibility with existing models

### Algorithm Changes
- Mathematical formulation (if complex)
- Hyperparameter choices and tuning
- Trade-offs made
- Edge cases handled

### Example
Provide runnable examples showing:
- Basic usage
- Advanced configurations
- Performance benchmarks
- Comparison with alternatives
```

## 9. Quick Reference Checklist

When reviewing AI/ML PRs in vLLM-omni:

**Critical Items:**
- [ ] No numerical instability (NaN, Inf checks)
- [ ] Memory bounds enforced (no OOM)
- [ ] Precision handling appropriate (fp32/fp16/bf16)
- [ ] Cache logic correct (KV-cache, DiT cache)
- [ ] Batching efficient (minimal padding)
- [ ] Tests cover edge cases
- [ ] Performance benchmarked
- [ ] Documentation updated

**Domain-Specific:**
- [ ] Multi-modal alignment correct
- [ ] Modality-specific preprocessing appropriate
- [ ] Non-AR model handling (if applicable)
- [ ] Hybrid pipeline transitions clear
- [ ] Quantization accuracy acceptable
- [ ] Distributed execution correct

**Best Practices:**
- [ ] Type hints used
- [ ] Error messages actionable
- [ ] Code follows ML conventions
- [ ] Reproducibility ensured (seeds)
- [ ] Determinism when required

---

## Conclusion

This guide provides deep technical expertise for reviewing vLLM-omni PRs. Use it alongside the general review framework to ensure changes are not only functionally correct but also optimized for ML workloads.

For PR #19 specifically, focus on areas that the PR actually touches, using this guide as a reference for domain-specific concerns.
