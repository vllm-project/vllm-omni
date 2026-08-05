# Qwen3-TTS Code2Wav NPU dtype (FP16/BF16) Design

**Date:** 2026-08-05  
**Status:** Approved for implementation planning  
**Scope:** NPU 910B/C Stage1 Code2Wav decoder precision  
**Non-goals:** CUDA load_weights change, 310P behavior change, custom kernels, making ~1s savings a CI gate

## Problem

On NPU 910, Qwen3-TTS Stage1 Code2Wav decoder runs in **FP32** by default. ConvTranspose / Conv-heavy upsample work dominates Stage1 latency. Half precision is expected to help, but must stay opt-in until A/B proves quality and kernel speedup.

## Current state

| Path | Behavior |
|------|----------|
| Common `Qwen3TTSCode2Wav.load_weights` | Hardcodes `decoder.to(..., dtype=torch.float32)` |
| NPU shared patch `platforms/npu/models/qwen3_tts_code2wav.py` | After load, moves decoder using `_npu_decoder_runtime_dtype`; **defaults to FP32** when missing |
| 310P patch | Subclass forces `torch.float16` via `_npu_decoder_runtime_dtype` |
| Deploy `qwen3_tts.yaml` Stage1 | No dtype field |

NPU weight packing (Linear NZ, group-1 Conv/ConvTranspose Fractal-Z) already runs on 910; only runtime dtype is still FP32.

## Goals

1. Make Stage1 decoder dtype configurable: `fp32` | `fp16` | `bf16`.
2. **Default remains `fp32`** (today’s behavior) until A/B justifies changing it.
3. Keep weights + intermediate compute in the selected dtype; restore final waveform to FP32 at the Stage1 output boundary.
4. Do not delete the FP32 path.
5. Re-profile Stage1 focusing on ConvTranspose total/max when comparing dtypes.

## Non-goals

- Changing CUDA / ROCm Code2Wav default dtype in this work.
- Changing 310P hard-coded FP16 override (310P subclass continues to win).
- Hand-written kernels or ACL graph fusion as part of this change.
- Treating an expected “~1s save” as an acceptance requirement.

## Configuration surface

Use existing **stage `dtype`**, flowing through `VllmConfig.model_config.dtype`.

Example (deploy yaml):

```yaml
stages:
  - stage_id: 1
    dtype: bf16   # or fp16 / fp32; omit for default fp32 behavior
```

Resolution order for NPU Code2Wav runtime dtype:

1. If the instance implements `_npu_decoder_runtime_dtype` as a 310P (or other platform) override that hard-codes a dtype → **that override wins** (preserve 310P).
2. Else read `vllm_config.model_config.dtype`.
3. Else `torch.float32`.

Invalid / unsupported values: log warning and fall back to `float32`.

## Implementation design

### 1. NPU shared hook (primary)

In `vllm_omni/platforms/npu/models/qwen3_tts_code2wav.py` (or via a small method patched onto `Qwen3TTSCode2Wav`):

- Provide a default `_npu_decoder_runtime_dtype(self, device) -> torch.dtype` that maps `model_config.dtype` → `torch.float32` / `float16` / `bfloat16`.
- Keep existing `load_weights` wrapper behavior:
  - `self.decoder.to(device=device, dtype=runtime_dtype)`
  - `_prepare_npu_decoder_weights(self.decoder)`
  - if `runtime_dtype != float32`: `precompute_snake_caches()`

### 2. Common CUDA path

Leave `Qwen3TTSCode2Wav.load_weights` FP32 hardcode **unchanged** in this iteration so CUDA baselines stay stable.

### 3. 310P

Keep `_Qwen3TTSCode2Wav310P._npu_decoder_runtime_dtype` returning `float16`. No requirement to honor stage dtype on 310P in this work.

### 4. Output boundary

Ensure Stage1 audio tensors exposed to connectors / clients are FP32 (`.float()` / `.to(torch.float32)` at the waveform return site if not already). Downstream audio packaging must not depend on half-precision buffers.

### 5. Tests

- Unit: dtype resolver maps `fp32`/`fp16`/`bf16` (and torch dtypes / string aliases) correctly; unknown → FP32 + warning.
- Unit: NPU patch `load_weights` path casts decoder parameters to the resolved dtype and calls snake cache precompute when non-FP32 (extend `tests/platforms/npu/test_310p_patches.py` patterns or a 910-focused unit).
- Manual / device A/B checklist (not necessarily CI): see Acceptance.

## Data flow (Stage1)

```
Talker codec tokens (int64)
        │
        v
Qwen3TTSTokenizerV2Decoder  @ runtime_dtype (fp32|fp16|bf16)
  - embedding / transformer / ConvNeXt
  - ConvTranspose upsample (profile focus)
  - SnakeBeta (caches must match weight dtype)
        │
        v
waveform → cast to float32 → multimodal_output / client
```

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| NaN/Inf in Snake / LN / softmax under FP16 | Prefer BF16 for first 910 trials; hard fail on non-finite outputs in A/B harness |
| Fractal-Z + half dtype hits slower kernels | A/B ConvTranspose total/max; keep FP32 default |
| Chunk-boundary quality drift in streaming | Compare first-chunk and steady-chunk spectrograms vs FP32 |
| Accidental CUDA behavior change | Do not touch common `load_weights` dtype line |

## Acceptance (A/B, single variable = dtype)

For fixed prompt set, concurrency, and compile/graph flags:

1. **Perf:** Stage1 latency (ms); ConvTranspose total and max kernel time.
2. **Stability:** RTF, buffer underruns, NaN/Inf count (must be 0 for pass).
3. **Quality:** amplitude / spectrum delta vs FP32; WER; timbre similarity; human listening.
4. **Expectation:** any “~1s” figure from exploratory notes is informational only; pass/fail is based on measured A/B.

Default product config stays FP32 until a recorded A/B supports changing Stage1 default on 910.

## Rollout

1. Land config + resolver + tests (default FP32).
2. Run FP32 vs BF16 vs FP16 A/B on target 910 hardware.
3. Optionally document recommended Stage1 `dtype` in NPU deploy notes; change yaml default only after sign-off.

## Open items resolved in review

- Target platform: **NPU 910** (primary).
- Default: **configurable, FP32 default**.
- Config surface: **stage `dtype`**.
- 310P / CUDA: **out of scope** for behavior change.
