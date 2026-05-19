# Optimization Playbook

Use this reference when turning trace evidence into prioritized vLLM Omni optimizations.

## Parallel Strategy

Start from workload shape:

- Long video tokens, high resolution, many frames: self-attention and FFN dominate. USP/SP is usually valuable.
- CFG enabled with positive and negative prompts: transformer forward count doubles. CFG parallel may beat USP on small/moderate sequence lengths.
- Cross-attention: skip SP/USP when condition tokens are much smaller than latent tokens; extra all-to-all can exceed compute saved.
- HSDP/FSDP: use to fit memory first. If memory allows, compare without HSDP because all-gathers and lazy init add overhead.
- VAE parallel: useful when encode/decode is material. Check gather/merge/broadcast and whether every rank needs final decoded output.

Candidate matrices:

- 2 cards: `USP=2`, `CFG=2`, HSDP on/off, VAE PP on/off.
- 4 cards: `USP=4`, `CFG=2 x USP=2`, `USP=2 x HSDP`, VAE PP on/off.
- 8 cards: add pipeline/stage parallel only after measuring orchestration and inter-stage transfer overhead.

## Host And Free Bubble Analysis

Look for device intervals where no kernel, memcpy, or memset is active.

Common causes:

- First request FSDP/HSDP lazy init.
- Python preprocessing: image load, resize, `to_tensor`, tokenizer, request marshaling.
- `torch.cuda.empty_cache()` or explicit synchronizations.
- Scheduler small linear algebra such as tiny `torch.linalg.solve`.
- VAE tile split/merge Python loops.
- Profiler overhead such as CUPTI command buffer flush/full.

Also inspect trace-external time:

- API request parsing and multipart upload.
- Engine/orchestrator dispatch.
- Worker RPC and queue wait.
- Output serialization, video write, base64, HTTP response.

## Operator Optimization Candidates

Evaluate necessity before implementation:

- **FA to LA / LA for head:** high potential only if attention is the main CUDA hotspot and target shapes fit LA well.
- **Cross-attention BSND/no transpose:** do only if transpose/copy around cross-attn is measurable.
- **AdaLayerNorm fusion:** fuse norm, scale/shift, gate, residual, and dtype conversion if repeated elementwise kernels are significant.
- **RMSNorm fusion:** useful for Q/K norms and VAE norms when small-kernel launch count is high.
- **FP32 LayerNorm to bf16:** quality-sensitive; run visual regression.
- **LA preprocess:** integrate only if preprocess overhead is cached or amortized.
- **RoPE move/cache/fuse:** useful when RoPE small kernels are visible around every self-attn block.
- **VAE RMSNorm replacement:** only after VAE internal trace shows norm as a real component of decode/encode.
- **Scheduler fusion/cache:** cache coefficient solves and fuse CFG combine with scheduler arithmetic when possible.

Before writing a custom kernel, check:

- Local vLLM Omni implementations.
- Upstream vLLM kernels and attention backends.
- SGLang, FlashInfer, FlashAttention, and Triton examples.
- Shape constraints and layout expectations.

## Quantization And Sparsity

Treat these as high-risk optimizations:

- Rainfusion/sparse diffusion can be high reward but needs model-specific quality validation.
- VAE decode quantization can reduce decode time but usually has smaller end-to-end upside than transformer attention/FFN work.
- FP8 or approximate kernels require seed-stable visual regression, temporal flicker checks, and prompt diversity.

## Output Format For Recommendations

Return a table with:

- Priority: P0/P1/P2.
- Optimization.
- Evidence from trace or code.
- Expected benefit.
- Risk/quality impact.
- Implementation path.
- Validation command or A/B test.

State clearly when a recommendation is a hypothesis requiring new trace with shapes.
