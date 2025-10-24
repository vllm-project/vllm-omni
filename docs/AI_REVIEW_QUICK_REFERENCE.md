# Quick Reference: AI Expert PR Review for vLLM-omni

## 🎯 Critical Review Points for Multi-Modal AI Systems

### ⚡ Quick Checks (5 minutes)
1. **Does it handle all modalities correctly?** (text, image, audio, video)
2. **Is tensor shape management correct?** (Check all `.shape`, `.view()`, `.reshape()`)
3. **Are memory leaks prevented?** (Check for proper `del`, context managers, `torch.no_grad()`)
4. **Is error handling comprehensive?** (Try/except blocks, validation)
5. **Do tests exist and pass?** (Check test coverage)

### 🔍 Deep Dive Points (30-60 minutes)

#### 1. Multi-Modal Input Processing
```python
# ✅ GOOD: Proper validation and type checking
def process_input(self, input_data: Union[str, Image, Audio]) -> Tensor:
    if isinstance(input_data, str):
        return self.text_encoder(input_data)
    elif isinstance(input_data, Image):
        return self.image_encoder(input_data)
    else:
        raise ValueError(f"Unsupported input type: {type(input_data)}")

# ❌ BAD: No validation, assumes type
def process_input(self, input_data) -> Tensor:
    return self.encoder(input_data)  # Will crash on wrong type
```

#### 2. Diffusion Model Implementation
```python
# ✅ GOOD: Proper noise scheduling and guidance
def denoise_step(self, latents: Tensor, t: int, 
                 guidance_scale: float = 7.5) -> Tensor:
    # Expand latents for classifier-free guidance
    latent_model_input = torch.cat([latents] * 2)
    
    # Predict noise residual
    with torch.no_grad():
        noise_pred = self.unet(latent_model_input, t)
    
    # Perform guidance
    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_pred_cond - noise_pred_uncond
    )
    
    # Compute previous noisy sample
    return self.scheduler.step(noise_pred, t, latents).prev_sample

# ❌ BAD: Missing guidance, incorrect tensor handling
def denoise_step(self, latents, t):
    noise_pred = self.unet(latents, t)  # No CFG
    return latents - noise_pred  # Wrong denoising formula
```

#### 3. KV Cache Management (AR Models)
```python
# ✅ GOOD: Proper cache initialization and cleanup
class KVCache:
    def __init__(self, max_batch_size: int, max_seq_len: int):
        self.cache = {}
        self.max_size = max_batch_size * max_seq_len
    
    def store(self, key: str, value: Tensor) -> None:
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        self.cache[key] = value.detach()  # Detach from computation graph
    
    def clear(self) -> None:
        self.cache.clear()
        torch.cuda.empty_cache()  # Free GPU memory

# ❌ BAD: No size limits, no cleanup, memory leak
class KVCache:
    def __init__(self):
        self.cache = {}
    
    def store(self, key, value):
        self.cache[key] = value  # Keeps gradient history, unbounded growth
```

#### 4. Tensor Shape Validation
```python
# ✅ GOOD: Explicit shape checking
def forward(self, x: Tensor) -> Tensor:
    expected_shape = (self.batch_size, self.seq_len, self.hidden_dim)
    if x.shape != expected_shape:
        raise ValueError(
            f"Expected shape {expected_shape}, got {x.shape}"
        )
    return self.layer(x)

# ❌ BAD: No validation, silent failures
def forward(self, x):
    return self.layer(x)  # Will crash with cryptic CUDA error
```

---

## 🏗️ Architecture Review Checklist

### vLLM Integration
- [ ] Uses `LLMEngine` or `AsyncLLM` correctly
- [ ] Implements `SchedulerInterface` if custom scheduler
- [ ] Extends `GPUModelRunner` for model execution
- [ ] Returns `ModelRunnerOutput` with correct structure
- [ ] Handles `SchedulerOutput` properly

### Multi-Stage Pipeline
- [ ] Stages are properly configured (`OmniStageConfig`)
- [ ] Output from one stage feeds correctly to next
- [ ] Error in one stage is handled gracefully
- [ ] Stage outputs are properly cached if needed
- [ ] Pipeline supports both sync and async execution

### Configuration Management
- [ ] Uses dataclasses for configuration
- [ ] Validates all required parameters
- [ ] Provides sensible defaults
- [ ] Supports environment variable overrides
- [ ] Configuration is serializable

---

## ⚠️ Common Pitfalls & Red Flags

### Memory Issues
```python
# 🚨 RED FLAG: Large tensor in loop without cleanup
for i in range(1000):
    large_tensor = torch.randn(1000, 1000, device='cuda')
    result = process(large_tensor)
    # Missing: del large_tensor, torch.cuda.empty_cache()

# ✅ FIX: Proper cleanup
for i in range(1000):
    large_tensor = torch.randn(1000, 1000, device='cuda')
    result = process(large_tensor)
    del large_tensor
    if i % 100 == 0:
        torch.cuda.empty_cache()
```

### Numerical Stability
```python
# 🚨 RED FLAG: Division without stability check
def normalize(x: Tensor) -> Tensor:
    return x / x.sum()  # Can divide by zero

# ✅ FIX: Add epsilon for stability
def normalize(x: Tensor, eps: float = 1e-8) -> Tensor:
    return x / (x.sum() + eps)
```

### Type Safety
```python
# 🚨 RED FLAG: No type hints
def process(data, config):
    return model(data, config)

# ✅ FIX: Proper type hints
def process(
    data: Union[Tensor, List[Tensor]], 
    config: ModelConfig
) -> ModelOutput:
    return model(data, config)
```

### Error Handling
```python
# 🚨 RED FLAG: Silent failure
try:
    result = risky_operation()
except:
    pass  # Silently swallows all errors

# ✅ FIX: Specific exception handling
try:
    result = risky_operation()
except ModelLoadError as e:
    logger.error(f"Failed to load model: {e}")
    raise
except RuntimeError as e:
    logger.warning(f"Runtime error, using fallback: {e}")
    result = fallback_operation()
```

---

## 🔒 Security Quick Checks

### Input Validation
- [ ] All external inputs are validated
- [ ] File uploads have size limits
- [ ] Input formats are strictly checked
- [ ] No arbitrary code execution (no `eval()`, `exec()`)
- [ ] Prompt injection protections in place

### Model Loading
- [ ] Models loaded from trusted sources only
- [ ] Path traversal protections
- [ ] Safe pickle usage (use `torch.load` with `weights_only=True`)
- [ ] Model checksums verified if applicable

### Data Privacy
- [ ] No PII in logs
- [ ] Secure caching of sensitive data
- [ ] Proper data retention policies
- [ ] Memory cleared after processing sensitive data

---

## 📊 Performance Quick Checks

### GPU Utilization
```python
# Check GPU memory usage
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Profile GPU operations
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    output = model(input)
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Optimization Checklist
- [ ] `torch.no_grad()` used for inference
- [ ] `torch.cuda.amp` used for mixed precision if applicable
- [ ] Batch operations instead of loops
- [ ] In-place operations where safe (`add_`, `mul_`, etc.)
- [ ] Proper dtype usage (fp16/bf16 for large models)

---

## 📝 Documentation Standards

### Docstring Template
```python
def process_multimodal_input(
    text: str,
    image: Optional[Image] = None,
    audio: Optional[Audio] = None,
    config: ProcessConfig = None,
) -> MultiModalOutput:
    """Process multi-modal inputs and generate embeddings.
    
    Args:
        text: Input text string
        image: Optional PIL Image
        audio: Optional audio data as numpy array
        config: Processing configuration
        
    Returns:
        MultiModalOutput containing embeddings for each modality
        
    Raises:
        ValueError: If no valid input is provided
        ModelError: If model processing fails
        
    Example:
        >>> output = process_multimodal_input(
        ...     text="A cat",
        ...     image=Image.open("cat.jpg")
        ... )
        >>> print(output.text_embedding.shape)
        torch.Size([1, 768])
    """
```

---

## ✅ Final Review Checklist

### Before Approving
- [ ] All critical issues addressed
- [ ] Tests pass and cover new code
- [ ] No security vulnerabilities
- [ ] No memory leaks
- [ ] Performance is acceptable
- [ ] Documentation is complete
- [ ] Breaking changes are documented
- [ ] Code follows project style

### Sign-off Criteria
1. **Correctness**: All functionality works as intended
2. **Safety**: No security or stability issues
3. **Quality**: Code is maintainable and well-documented
4. **Performance**: No significant performance regression
5. **Testing**: Adequate test coverage

---

## 📚 Resources

- [Full AI Expert PR Review Guide](./AI_EXPERT_PR_REVIEW_GUIDE.md)
- [vLLM Documentation](https://docs.vllm.ai/)
- [PyTorch Best Practices](https://pytorch.org/tutorials/beginner/best_practices.html)
- [Diffusion Models Guide](https://huggingface.co/docs/diffusers/)
- [vLLM-omni Implementation Architecture](./architecture/implementation_architecture.md)

---

## 💡 Tips for Efficient Reviews

1. **Start with tests**: Read tests first to understand expected behavior
2. **Check CI/CD**: Look at automated checks before manual review
3. **Focus on changes**: Use git diff to focus on what changed
4. **Run locally**: Clone and test locally for complex changes
5. **Use profiling**: Profile performance-critical changes
6. **Check dependencies**: Verify new dependencies are necessary and safe
7. **Review documentation**: Ensure docs match implementation

---

**Remember**: The goal is to maintain high code quality while enabling innovation. Be thorough but constructive!
