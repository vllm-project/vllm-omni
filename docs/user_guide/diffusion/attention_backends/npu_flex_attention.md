# NPU FlexAttention

MindIE-SD provides CATLASS DSL FlexAttention on Ascend NPUs. vLLM-Omni exposes
it through a shared adapter in the `FLASH_ATTN` backend. The adapter consumes
block-causal attention metadata; each model integration supplies its own
sequence geometry and selects when to use this path.

## Dependencies and current adapter limits

Install a MindIE-SD build exporting `flex_attention`, `make_block_causal_mask`
and `check_flex_attention_device`, together with its matching FA4 and CATLASS
DSL dependencies. The current kernel path requires Ascend 950 and head
dimension 128.

The current vLLM-Omni adapter accepts one BF16 sample in BSND layout, with
complete local Q/K/V, equal query/key sequence lengths and unquantized
attention. It requires block-causal spans rather than a caller-provided dense
mask, and runs under `torch.no_grad()` or `torch.inference_mode()` outside an
outer `torch.compile` region. Sequence parallelism is disabled for this path.
These are the current Omni adapter's limits, not the full MindIE-SD API contract.

## Integration boundary

Model code supplies ordered, non-overlapping bidirectional spans through
`AttentionMetadata.full_attn_spans` and opts in with the `npu_flex_attention`
metadata flag. The rule allows causal visibility plus bidirectional attention
within each span. Optional `npu_flex_valid_tokens` metadata excludes invalid
tokens.

The shared adapter owns the request-local `_npu_flex_cache`, which model
integrations initialize before copying metadata across attention layers.
MindIE-SD constructs the sparse metadata, compiles the kernel and executes
local attention through `flex_attention`. Model code owns the transition to
other attention paths when its execution phase or input contract changes.

## Model integrations

Qwen-Image-2.1 is the current model integration. Its source-level selection,
prefill/decode behavior and model validation procedure are described in the
[Qwen-Image-2.1 recipe](../../../../recipes/Qwen/Qwen-Image-2.1.md#optional-npu-flexattention).

## Validation

CPU dependency and request-isolation contracts:

```bash
pytest -q tests/diffusion/attention/test_npu_flex_attention.py -m 'core_model and cpu' --run-level=core_model
```

Device numerics with synthetic block-causal inputs, requiring Ascend 950 and
the dependencies above:

```bash
pytest -sv tests/diffusion/attention/test_npu_flex_attention_numerics.py -m 'core_model and npu and A5' --run-level=core_model
```
