# NPU FlexAttention prefill

`FLEX_ATTN` is an experimental, non-quantized NPU backend. Qwen-Image-2.1's
joint attention selects it by default on NPU when no backend was explicitly
configured. Other models and platforms retain their existing defaults.

The first implementation uses compiled FlexAttention for homogeneous,
unpadded full-prefill calls carrying `full_attn_spans`, with FP16/BF16 Q/K/V,
equal Q/K/V head counts, and head dimension 128. TP, SP and HSDP configurations
use the existing FA implementation. Scheduler-managed paged KV is not supported.

## Execution

1. Qwen's existing `attn_path="auto"` produces image spans.
2. Spans define the exact keep rule `(q >= kv) OR same_image_block`.
3. A bounded, 16-entry cache reuses BlockMask objects across layers and matching
   requests. The key includes all image boundaries, sequence length, device and
   block size; the initial block size is 128.
4. The backend converts BSND Q/K/V to BNSD and calls compiled FlexAttention
   once per layer. It restores BSND output afterward.
5. Decode, explicit dense masks and unsupported prefill shapes delegate to the
   existing FA paths. A span-based fallback keeps the PR's piecewise FA calls.

Only the Flex callable is compiled with `torch.compile(..., backend="inductor",
fullgraph=True)`. The NPU platform's whole-model compilation setting is unchanged.
Model eager mode therefore does not disable the compilation required by this
explicitly selected attention implementation. First-use compilation and shape
specialization must be measured separately from warmed inference.

## Dependencies and selection

Use matching PyTorch, torch_npu, CANN and Ascend Triton builds with NPU Flex
Inductor support, plus MindIE-SD for the FA paths. Startup validates imports;
this does not establish that code generation works on a particular device.
Missing dependencies raise an actionable error. Compilation/execution failures
propagate instead of silently turning a Flex experiment into an FA run.

Explicit selection, using the existing serving config:

```bash
--diffusion-attention-config '{"default":{"backend":"FLEX_ATTN"}}'
```

To return to the current piecewise FA implementation:

```bash
--diffusion-attention-config '{"default":{"backend":"FLASH_ATTN"}}'
```

In Python, pass the corresponding dictionary as `diffusion_attention_config`
to `Omni`. Explicit global/per-role configuration overrides the model default.
The current pipeline does not expose `attn_path` as a request option; to produce
a dense-mask Golden in a separate experiment checkout, set its
`attention_kwargs` to `{"attn_path": "mask"}`.

## Validation status

CPU tests cover exact visibility, final partial blocks, mask reuse and routing.
The NPU test executes real compiled Flex and compares against an FP32 dense
reference for text-to-image and multiple image spans. It requires no weights.
An unavailable dependency produces a skip, which is not a successful NPU test.

```bash
# CPU logic tests; requires the repository's vLLM/Omni test dependencies
pytest -q tests/diffusion/attention/test_flex_block_mask.py tests/diffusion/attention/test_flex_attn.py -m 'core_model and cpu'

# Real NPU test; requires a compatible NPU Flex compiler stack
pytest -sv tests/diffusion/attention/test_flex_attn.py -m 'core_model and npu and A5' --run-level=core_model
```

Full-model NPU correctness and performance are not yet validated. Compare
identical Q/K/V first, then same-prompt/seed T2I and multi-reference I2I against
the explicit dense-mask Golden. The pre-PR default omitted span rules and is
not a mathematical reference. Include two denoising steps to exercise prefix
cache reuse, followed by the intended full inference schedule.

Do not infer speedup from skipped blocks alone: piecewise FA already excludes
many invisible connections. Measure BlockMask creation, compilation, warmed
attention, end-to-end generation and peak device memory separately.

Attention quantization is deliberately rejected, including FP8 modes inherited
by the FA implementation. FP8 Flex needs separate kernel/scale validation;
storing prefix KV in FP8 is not proof of quantized Flex computation.
