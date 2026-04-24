# Troubleshooting Reference

## Common Errors When Adding a Diffusion Model

### ImportError / ModuleNotFoundError

**Cause**: Missing or incorrect registration.

**Fix checklist**:
1. Model registered in `vllm_omni/diffusion/registry.py` `_DIFFUSION_MODELS` dict
2. `__init__.py` exports the pipeline class
3. Pipeline file exists at the correct path: `vllm_omni/diffusion/models/{folder}/{file}.py`
4. Class name in registry matches the actual class name in the file

### Shape Mismatch in Attention

**Symptom**: `RuntimeError: shape mismatch` or `expected 4D tensor`

**Cause**: QKV tensors not reshaped to `[batch, seq_len, num_heads, head_dim]`.

**Fix**: Before calling `self.attn(q, k, v, ...)`, ensure:
```python
q = q.view(batch, seq_len, self.num_heads, self.head_dim)
k = k.view(batch, kv_seq_len, self.num_kv_heads, self.head_dim)
v = v.view(batch, kv_seq_len, self.num_kv_heads, self.head_dim)
```

After attention, reshape back:
```python
out = out.reshape(batch, seq_len, -1)
```

### Weight Loading Failures

**Symptom**: `RuntimeError: size mismatch for parameter ...` or missing keys

**Debugging**:
1. Print diffusers weight names: `safetensors.safe_open(path, "pt").keys()`
2. Print model parameter names: `dict(model.named_parameters()).keys()`
3. Compare and add name remappings in `load_weights()`

**Common remappings needed**:
- `ff.net.0.proj` → `ff.net_0.proj` (PyTorch Sequential indexing)
- `.to_out.0.` → `.to_out.` (Sequential unwrapping)
- `scale_shift_table` → moved to a wrapper module

### Black/Blank/Noisy Output

**Possible causes**:
1. **Wrong latent normalization**: Check VAE expects latents scaled by `vae.config.scaling_factor`
2. **Wrong scheduler**: Using the wrong scheduler class or wrong `flow_shift`
3. **Missing CFG**: Some models require `guidance_scale > 1.0` with negative prompt
4. **Wrong timestep format**: Some schedulers expect float, others expect int/long
5. **Missing post-processing**: Raw VAE output may need denormalization

**Quick test**: Run with diffusers directly using the same seed and compare latents at each step.

### OOM (Out of Memory)

**Solutions** (in order of preference):
1. `--enforce-eager` to disable torch.compile (saves compile memory)
2. `--enable-cpu-offload` for model-level offload
3. `--enable-layerwise-offload` for block-level offload (better for large models)
4. `--vae-use-slicing --vae-use-tiling` for VAE memory reduction
5. Reduce resolution: `--height 480 --width 832`
6. Use TP: `--tensor-parallel-size 2`

### Different Output vs Diffusers Reference

**Common causes**:
1. **Attention backend difference**: FlashAttention vs SDPA may produce slightly different results. Set `DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA` to match diffusers
2. **Float precision**: vLLM-Omni may use bfloat16 where diffusers uses float32 for some operations
3. **Missing normalization**: Check all LayerNorm/RMSNorm are preserved
4. **Scheduler rounding**: Some schedulers have numerical sensitivity

### Tensor Parallel Errors

**Symptom**: `AssertionError: not divisible` or incorrect output with TP>1

**Fix**:
1. Verify `num_heads % tp_size == 0` and `num_kv_heads % tp_size == 0`
2. Ensure `ColumnParallelLinear` / `RowParallelLinear` are used correctly
3. Check that norms between parallel layers use distributed norm if needed
4. Verify `load_weights` handles TP sharding for norm weights
5. Use `self.to_qkv.num_heads` (local heads per GPU) for QKV split sizes, not total heads

**Missing `input_is_parallel=True`**:

`RowParallelLinear` expects sharded input from `ColumnParallelLinear`:
```python
self.w1 = ColumnParallelLinear(dim, hidden_dim, return_bias=False)
self.w2 = RowParallelLinear(hidden_dim, dim, input_is_parallel=True, return_bias=False)
```

### Sequence Parallel Errors

**Symptom**: Incorrect output or crashes with `--ulysses-degree N` or `--usp N`

**Possible causes**:
1. **Inline operations between shard/gather points**: `torch.cat()`, `pad_sequence()` etc. not at `nn.Module` boundaries. Fix: extract into submodule.
2. **Wrong `split_dim`**: Check the tensor shape at the shard point. Sequence dimension is typically `dim=1` for `[B, S, D]` tensors.
3. **RoPE not sharded**: If RoPE is computed separately, add it to `_sp_plan` with `split_output=True`.
4. **Sequence not divisible by SP degree**: Use `auto_pad=True` in `SequenceParallelInput` or switch to `ulysses_mode="advanced_uaa"`.

**Debugging**: Add `expected_dims=N` to `SequenceParallelInput`/`Output` for shape validation at runtime.

### CFG Parallel Errors

**Symptom**: CFG parallel not activating, no speedup

**Fix checklist**:
1. Pipeline inherits `CFGParallelMixin`
2. `guidance_scale > 1.0`
3. Negative prompt provided (even if empty string)
4. `--cfg-parallel-size 2` specified
5. `diffuse()` method calls `predict_noise_maybe_with_cfg()` and `scheduler_step_maybe_with_cfg()`

**Symptom**: Different output with CFG parallel vs sequential

**Possible cause**: Non-deterministic scheduler. Fix: pass `generator=torch.Generator(device).manual_seed(seed)` to `scheduler_step_maybe_with_cfg()`.

### HSDP Errors

**Symptom**: HSDP not activating or errors during weight loading

**Fix checklist**:
1. Transformer defines `_hsdp_shard_conditions` class attribute
2. Shard condition functions return `True` for correct modules (test with `model.named_modules()`)
3. Not combining with TP (HSDP and TP are incompatible)
4. For standalone HSDP, `hsdp_shard_size` is specified explicitly

**Verify**: Check logs for "HSDP Inference: replicate_size=..., shard_size=..." and "Sharded N modules + root".

### Cache-DiT Not Applied

**Symptom**: No speedup, no cache-related log messages

**Fix checklist**:
1. Model not in `_NO_CACHE_ACCELERATION` in `registry.py`
2. Pipeline class name matches `CUSTOM_DIT_ENABLERS` key (if using custom enabler)
3. `cache_backend="cache_dit"` specified
4. Check logs for "Cache-dit enabled successfully on xxx"

**Verify pipeline name**: `print(pipeline.__class__.__name__)` — must match registry key.

### Cache-DiT Quality Degradation

**Symptom**: Artifacts or lower quality with cache-dit

**Fix**: Reduce aggressiveness:
```python
cache_config={
    "residual_diff_threshold": 0.12,      # Lower from 0.24
    "max_warmup_steps": 6,                # Increase from 4
    "max_continuous_cached_steps": 2,      # Reduce if higher
}
```

If quality is still poor, the model may need a custom enabler with per-block-list `ParamsModifier` tuning.

### Model Not Detected / Wrong Pipeline Class

**Symptom**: `ValueError: Model class ... not found in diffusion model registry`

**Cause**: The model's `model_index.json` has a `_class_name` for the pipeline that doesn't match registry keys.

**Fix**: The registry key must match the diffusers pipeline class name from `model_index.json`. If using a different name, map it in the registry:
```python
"DiffusersPipelineClassName": ("your_folder", "your_file", "YourVllmClassName"),
```

## Debugging Workflow

1. **Add verbose logging**: Use `logger.info()` to print tensor shapes at each stage
2. **Compare step-by-step**: Run diffusers and vllm-omni side by side, comparing tensors after each major operation
3. **Use small configs**: Reduce `num_inference_steps=2`, small resolution for fast iteration
4. **Test transformer isolation**: Feed the same input to both diffusers and vllm-omni transformers, compare outputs
5. **Binary search for bugs**: Comment out blocks/layers to isolate where divergence starts
