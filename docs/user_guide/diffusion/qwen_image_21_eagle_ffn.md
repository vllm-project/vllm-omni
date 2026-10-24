# Qwen-Image-2.1 optional Eagle FFN

Qwen-Image-2.1 can opt into the public MindIE-SD `eagle_ffn_linear` API for
its complete SwiGLU feed-forward sublayer. The default remains `native`.
This is an experimental execution option; a performance improvement or
image-quality equivalence is not established for this integration.

## Configuration

Use the existing supplementary configuration:

```python
additional_config={"qwen21_mlp_backend": "eagle"}
```

The eligible configuration is Ascend A5, BF16, `enforce_eager=True`, TP1/SP1,
unquantized resident weights, and no HSDP or LoRA. Install a MindIE-SD build
that provides both the public `mindiesd.layers.eagle_ffn_linear` interface
and its compiled Eagle FFN operator. A Python interface alone does not prove
the compiled operator is present. Unsupported configurations or a missing
Python interface warn and retain the native path; selected operator execution
errors propagate. Invalid backend values raise `ValueError`.

The `native` setting does not probe the optional dependency. Set
`qwen21_mlp_backend` to `native` (or omit it) and restart the engine to roll back.
After changing the MindIE-SD installation, restart the process to refresh API
resolution.

## Weight and execution boundaries

Gate/up weights are packed once after checkpoint loading in `[gate, up]`
order. The original parameters become views of that storage, retaining the
checkpoint names and avoiding persistent duplicate gate/up weights. Packing
temporarily requires an additional gate/up allocation. No weights are packed
in the denoising loop. Reloading weights through `load_weights` repacks them.

The complete fused FFN bypasses the three linear module calls, so it is
restricted to TP1 and excludes LoRA, offload, and quantization. Its MLP
projections keep ND weights; automatic NZ conversion for other linears remains
unchanged. Compilation, training, gradient-enabled execution, replaced
projection modules, and invalidated packed storage use the existing native MLP.
The outer modulation, residual gate, and attention path are unchanged.

Eagle can have different intermediate rounding from the native chain.
Compare real-model outputs and measure profiler-free request latency before
adopting it for a workload. Kernel fusion alone does not establish a speedup.

## Validation commands

With the repository's test dependencies installed, run CPU contract checks:

```bash
python -m pytest -q tests/diffusion/models/qwen_image_21/test_mlp_fusion.py \
  tests/diffusion/layers/test_fused_swiglu.py -m cpu --run-level core_model
```

With an exclusive A5 card, a compatible torch-npu environment, and the complete
MindIE-SD operator build, run the real-shape adapter check:

```bash
python -m pytest -sv tests/diffusion/layers/test_fused_swiglu_npu.py \
  -m 'core_model and npu and A5 and cards_1' --run-level core_model
```

This operator test is not full-checkpoint quality or end-to-end performance
acceptance. A skipped test does not validate an installed operator.
