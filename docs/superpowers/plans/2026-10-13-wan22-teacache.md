# Wan 2.2 TeaCache Distributed Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Wan 2.2 TeaCache preserve native single-device, PP, and SP semantics.

**Architecture:** Decompose the existing Wan forward without replacing its parallel helpers: retain native preprocessing, run only stage-local blocks, and make TeaCache state local to the stage and shard.

**Tech Stack:** PyTorch, pytest, vLLM-Omni TeaCache and parallel helpers.

---

### Task 1: Test and restore the native Wan single-device contract

**Files:**
- Modify: `vllm_omni/diffusion/cache/teacache/extractors.py:401-515`
- Create: `tests/diffusion/cache/test_wan22_teacache.py`

- [ ] **Step 1: Write failing T2V and TI2V extractor tests**

```python
context = extract_wan2_2_context(module, hidden_states, timestep, text_states)
assert module.blocks[0].norm1.call_args.args[1:] == (scale_msa, shift_msa)
assert context.postprocess(context.hidden_states).sample.shape == hidden_states.shape
```

- [ ] **Step 2: Verify failure**

Run: `pytest tests/diffusion/cache/test_wan22_teacache.py -k 'single_device or ti2v' -v`

Expected: FAIL because the extractor omits AdaLayerNorm scale and shift.

- [ ] **Step 3: Implement the native calls**

```python
norm_hidden_states = block.norm1(hidden_states, scale_msa, shift_msa).type_as(hidden_states)
h = module.norm_out(h, scale, shift).type_as(h)
```

Retain native condition embedding, I2V/TI2V image-state concatenation, output unpatchification, and `return_dict`.

- [ ] **Step 4: Verify pass**

Run: `pytest tests/diffusion/cache/test_wan22_teacache.py -k 'single_device or ti2v' -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vllm_omni/diffusion/cache/teacache/extractors.py tests/diffusion/cache/test_wan22_teacache.py
git commit -m "fix: preserve Wan TeaCache single-device forward"
```

### Task 2: Test and restore pipeline-parallel ownership

**Files:**
- Modify: `vllm_omni/diffusion/cache/teacache/extractors.py:401-515`
- Modify: `tests/diffusion/cache/test_wan22_teacache.py`

- [ ] **Step 1: Write failing first/middle/last PP tests**

```python
assert patch_embedding.called is first_stage
assert block_calls == list(range(module.start_layer, module.end_layer))
assert isinstance(context.postprocess(h), IntermediateTensors) is (not last_stage)
```

- [ ] **Step 2: Verify failure**

Run: `pytest tests/diffusion/cache/test_wan22_teacache.py -k pipeline_parallel -v`

Expected: FAIL because every stage patchifies, runs all blocks, and projects output.

- [ ] **Step 3: Implement native PP branches**

```python
if is_pipeline_first_stage():
    hidden_states = module.patch_embedding(hidden_states).flatten(2).transpose(1, 2)
else:
    if intermediate_tensors is None:
        raise RuntimeError("intermediate_tensors must be provided for non-first PP stages")
    hidden_states = intermediate_tensors["hidden_states"]
for block in module.blocks[module.start_layer : module.end_layer]:
    h = block(h, encoder_hidden_states, timestep_proj, rotary_emb, hidden_states_mask,
              vsa_dit_seq_shape, module.preserve_vsa_all_blocks)
```

Return `IntermediateTensors({"hidden_states": h})` from non-last stage postprocessing.

- [ ] **Step 4: Verify and commit**

Run: `pytest tests/diffusion/cache/test_wan22_teacache.py -k pipeline_parallel -v`

Expected: PASS.

```bash
git add vllm_omni/diffusion/cache/teacache/extractors.py tests/diffusion/cache/test_wan22_teacache.py
git commit -m "fix: preserve Wan TeaCache pipeline parallelism"
```

### Task 3: Test and restore sequence-parallel and VSA propagation

**Files:**
- Modify: `vllm_omni/diffusion/cache/teacache/extractors.py:401-515`
- Modify: `tests/diffusion/cache/test_wan22_teacache.py`

- [ ] **Step 1: Write failing SP forwarding tests**

```python
module._sp_shard_point.assert_called_once()
assert block.call_args.args[4] is local_padding_mask
assert block.call_args.args[5] == expected_vsa_grid
assert block.call_args.args[6] is module.preserve_vsa_all_blocks
```

- [ ] **Step 2: Verify failure**

Run: `pytest tests/diffusion/cache/test_wan22_teacache.py -k sequence_parallel -v`

Expected: FAIL because the extractor does not shard, mask, or forward VSA data.

- [ ] **Step 3: Implement the native SP sequence**

After first-stage patchification call `_sp_shard_point`; under the native forward-context condition call `build_local_sp_padding_mask`; pass the local mask, post-patch `(T, H, W)` grid, and `preserve_vsa_all_blocks` to every local block.

- [ ] **Step 4: Verify and commit**

Run: `pytest tests/diffusion/cache/test_wan22_teacache.py -k sequence_parallel -v`

Expected: PASS.

```bash
git add vllm_omni/diffusion/cache/teacache/extractors.py tests/diffusion/cache/test_wan22_teacache.py
git commit -m "fix: preserve Wan TeaCache sequence parallelism"
```

### Task 4: Verify TeaCache miss/hit behavior and documentation

**Files:**
- Modify: `tests/diffusion/cache/test_wan22_teacache.py`
- Modify: `docs/user_guide/diffusion_features.md:190`

- [ ] **Step 1: Add hook miss/hit coverage**

```python
config = TeaCacheConfig(transformer_type="WanTransformer3DModel")
first = hooked_transformer(**inputs)
second = hooked_transformer(**next_inputs)
assert first.sample.shape == second.sample.shape
assert first.sample.dtype == second.sample.dtype
```

Assert a miss executes local blocks and an allowed hit applies the local residual.

- [ ] **Step 2: Run the complete validation suite**

Run:

```bash
pytest tests/diffusion/cache/test_wan22_teacache.py -v
pytest tests/diffusion/cache/test_teacache.py tests/diffusion/cache/test_cache_backends.py -v
git diff --check
```

Expected: all tests PASS and no whitespace errors.

- [ ] **Step 3: Keep docs aligned and commit**

Retain the Wan2.2 TeaCache support marker only after the tests pass; do not claim TeaCache for Wan2.2-S2V or Wan2.1-VACE.

```bash
git add tests/diffusion/cache/test_wan22_teacache.py docs/user_guide/diffusion_features.md
git commit -m "test: cover Wan TeaCache parallel execution"
```

## Plan self-review

- Tasks 1–4 cover T2V, I2V, TI2V, PP, SP, miss/hit behavior, output contracts, and documentation.
- All code references use native Wan argument order and `IntermediateTensors` handoff.
