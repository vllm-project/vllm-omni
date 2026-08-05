# [Diffusion][Quantization] SVDQuant W4A4 (Nunchaku) for Z-Image-Turbo

## Summary

Integrate **SVDQuant W4A4 NVFP4** (the [Nunchaku](https://github.com/nunchaku-ai/nunchaku) family) as an offline-quantized backend for diffusion transformers, validated on Z-Image-Turbo on RTX 5090 (consumer Blackwell, SM_120).

**Headline (1024×1024, 20 steps, seed=42, batch=1):**
- **2.24× speedup** vs BF16 (11.07s → 4.94s)
- **-29% peak VRAM** (24.26 → 17.14 GiB), **-34% weights** (20.87 → 13.74 GiB)

## Why this is a new PR (not duplicate of #1986)

The closing comment on #1986 explained the three things that changed materially:

1. **On-disk format pivoted to canonical row-major NVFP4.** Offline converter (`vllm_omni/quantization/tools/convert_nunchaku_to_svdquant.py`) emits a backend-agnostic checkpoint; the nunchaku PTX-MMA tile fragment layout is now produced at load time inside the backend, not baked into the on-disk weights. No more runtime `MergedColumnParallelLinear` output-half swap.
2. **Config + LinearMethod + backend live in vllm-omni**, not vllm. The earlier plan was a split (config in vllm, glue in vllm-omni), but per review on the in-vllm PR (`vllm-project/vllm#43471`, now closed) the consensus was that SVDQuant — being a Python wrapper around `nunchaku`'s W4A4 CUDA kernels with a consumer-GPU-only envelope — fits vllm-omni's "diffusion-side caller" pattern alongside `DiffusionInt8Config` / `DiffusionMXFP4Config` / `DiffusionMXFP8Config` rather than vllm proper. This PR is now the single home for SVDQuant in the vllm ecosystem.
3. **Reviewer concerns from #1986 are addressed declaratively**, not papered over (declarative `quantization_config` in `transformer/config.json`, strict missing-weight validation preserved via `ComponentQuantizationConfig` returning `UnquantizedLinearMethod` for non-quantized components, model-side key remapping reduced to one trailing-dot fix in `stacked_params_mapping`).

## What's in this PR

`svdquant-converter` branch, 7 commits:

| commit | scope |
|---|---|
| `ea7325bf` | **Config + LinearMethod + nunchaku backend infrastructure** (migrated from closed vllm-project/vllm#43471). New files: `vllm_omni/quantization/svdquant_config.py` (`DiffusionSVDQuantConfig` + LinearMethod, backend-agnostic), `svdquant_dispatch.py` (hardware gate + `select_backend`), `svdquant_nunchaku.py` (nunchaku capability detection + lazy importlib wrappers + `prepare_weights` / `apply`). Promotes `tools/svdquant_nvfp4_layout.py` from re-export shim to real implementation. Factory registers `"svdquant"`. 12 tests in `tests/diffusion/quantization/test_svdquant_config.py`. Converter also drops nunchaku's unused `smooth_factor_orig` suffix at group time (nunchaku itself marks it `(Unused)` in `nunchaku/models/linear.py:54`). |
| `3d7ff30a` | Pre-commit (ruff format) cleanup |
| `762caa48` | Bench: use `nvidia-smi` for GPU memory (spawn-mode worker isolation) + zero-division guard |
| `1aa18b06` | Bench: `--baseline-model` + `--quantization auto` sentinel, for offline-quantized checkpoints where BF16 and quant trees are separate paths |
| `605b342a` | Converter: `modules_to_not_convert: ["lm_head"]` so Qwen3 text-encoder's top-level LM head doesn't fall through to SVDQuant |
| `a8e637d3` | Bench: emit Memory Profiling table (PR #1470 layout) |
| `edad5f35` | Offline converter (`convert_nunchaku_to_svdquant.py`, 549 lines) + per-component config wiring + Z-Image transformer trailing-dot key fix + example label resolution |

The 4 `[Benchmark]` commits are bench scaffolding. Happy to split them into a separate `[Benchmark]` PR if reviewers prefer (the nvidia-smi memory fix in particular is a latent bug that hits every offline-quant PR going through spawn-mode workers).

## Quantized checkpoints

Both produced by the included converter from the original nunchaku-published merged safetensors:

- **HuggingFace**: https://huggingface.co/ultranationalism/nunchaku-z-image-turbo-svdq
- **ModelScope** (CN mirror): https://www.modelscope.cn/models/ultranationalism/Z-Image-Turbo-SVDQuant-NVFP4

## Test Plan

```bash
# Bench command — produces all 3 tables + 16 side-by-side PNGs
python benchmarks/diffusion/quantization_quality.py \
  --baseline-model Tongyi-MAI/Z-Image-Turbo \
  --model ultranationalism/nunchaku-z-image-turbo-svdq \
  --task t2i \
  --quantization auto \
  --prompts \
    "a close-up portrait of an elderly fisherman with weathered skin and a thick gray beard, soft natural light" \
    "an aerial view of a coral reef with crystal clear turquoise water" \
    "extreme close-up of a dewdrop on a red rose petal, morning sunlight" \
    "a bustling night market in Tokyo with neon signs, rain-slicked streets, and crowds with umbrellas" \
    "a vintage bookstore storefront with the sign CLASSICS AND RARE EDITIONS in elegant gold lettering" \
    "a campfire in a dark forest with sparks rising into a starry sky" \
    "a ballet dancer in mid-leap on an empty theater stage, dramatic spotlight from above" \
    "a cup of coffee on a wooden table, morning light" \
  --height 1024 --width 1024 \
  --num-inference-steps 20 \
  --seed 42 \
  --lpips-net alex \
  --output-dir ./svdquant_bench_output
```

## Test Result

- **GPU**: NVIDIA RTX 5090 D (Blackwell SM_120, 32 GiB) — primary bench target
- **Backend dispatched**: `nunchaku` (the only in-tree backend in this PR; see Roadmap below for SM_100/103 + Ascend plans)
- **Stack**: vLLM + this PR's `svdquant-converter` branch + nunchaku `1.2.1+cu12.8torch2.11` + PyTorch 2.11.0+cu128

### Summary

| Config | Avg Time | Speedup | Memory (GiB) | Mem Reduction | Mean LPIPS |
|--------|----------|---------|--------------|---------------|------------|
| BF16 baseline | 11.07s | 1.00× | 24.26 | — | (ref) |
| SVDQuant W4A4 NVFP4 (nunchaku backend) | 4.94s | **2.24×** | 17.14 | **-29%** | 0.2324 |

### Memory Profiling

First-prompt snapshot at 1024×1024, 20 steps, TP=1. Memory read via `nvidia-smi --query-gpu=memory.used` (vllm-omni spawn-mode workers have their own CUDA contexts; the bench driver process sees 0 GiB allocated).

| Config | Weights | Activations | Peak | Total Reduction |
|--------|---------|-------------|------|-----------------|
| BF16, TP=1 | 20.87 GiB | 3.39 GiB | 24.26 GiB | — |
| SVDQuant, TP=1 | 13.74 GiB | 3.40 GiB | 17.14 GiB | **-29%** |

Activations are unchanged (text-encoder activations + diffusion latents are not quantized). The 7.13 GiB weights delta is the entire SVDQuant win.

Spot-checked on a smaller box: **RTX 5060 Ti (SM_120, 16 GiB)** with `--enable-cpu-offload` runs 512×512 / 8 steps in **5.9 s**, peak VRAM **8.5 GiB**. Confirms backend dispatch + nunchaku weight-prep + apply path work under model CPU-offload, not just dense GPU residency.

### Per-Prompt LPIPS (alex backbone)

| Prompt | LPIPS |
|--------|------:|
| a close-up portrait of an elderly fisherman with weathered skin... | 0.276 |
| an aerial view of a coral reef with crystal clear turquoise water | 0.210 |
| extreme close-up of a dewdrop on a red rose petal, morning sunlight | **0.394** |
| a bustling night market in Tokyo with neon signs, rain-slicked streets... | **0.312** |
| a vintage bookstore storefront with the sign CLASSICS AND RARE EDITIONS... | 0.161 |
| a campfire in a dark forest with sparks rising into a starry sky | 0.192 |
| a ballet dancer in mid-leap on an empty theater stage, dramatic spotlight... | 0.160 |
| a cup of coffee on a wooden table, morning light | 0.155 |
| **mean** | **0.232** |

### Quality trade-off (honest framing)

W4A4 is significantly more aggressive than the W8A8 baselines that other offline-quant PRs (ModelOpt FP8 #2913, MXFP8 #3140) report on. LPIPS scores reflect that:

- **Simple subjects** (coffee, ballet, bookstore signage): LPIPS ~0.16, on par with #1470's int8 Z-Image mean of 0.1597.
- **Complex multi-object scenes** (Tokyo night market) and **extreme close-ups with fine texture** (dewdrop on rose petal): LPIPS 0.31–0.39.

These higher LPIPS values are inherent to W4A4 with SVD low-rank correction (rank=128) — the algorithm trades quality for compression more aggressively than W8A8. **The PR's value proposition is 2.24× speedup + 34% weights compression**, not strict pixel parity with BF16. Users who need higher fidelity should stay on the BF16 or int8/FP8 paths.

### Visual Gallery

<details>
<summary>Z-Image-Turbo — BF16 vs SVDQuant W4A4 NVFP4 (8 prompts, same seed)</summary>

| # | Prompt | BF16 | SVDQuant |
|:-:|--------|:----:|:--------:|
| 0 | elderly fisherman portrait | <img src="https://github.com/user-attachments/assets/afd3910c-5ce6-4634-b5d7-2e0ab497e9e0" width="350"> | <img src="https://github.com/user-attachments/assets/f7ac8a8e-90d7-48e7-94a0-229b45cb3ff2" width="350"> |
| 1 | aerial coral reef | <img src="https://github.com/user-attachments/assets/bbb25892-f58f-48cc-aa52-063d87fbf63f" width="350"> | <img src="https://github.com/user-attachments/assets/64210347-1eba-405e-a8d4-66c366cc53db" width="350"> |
| 2 | dewdrop macro (worst LPIPS) | <img src="https://github.com/user-attachments/assets/7722ade3-d2d0-4f71-8ddc-9838151ecd8a" width="350"> | <img src="https://github.com/user-attachments/assets/9f399a96-2847-4bd1-beec-3461bfb540a7" width="350"> |
| 3 | Tokyo night market | <img src="https://github.com/user-attachments/assets/1cd7ec86-a9f1-4e1e-b1e9-4584ef3c451b" width="350"> | <img src="https://github.com/user-attachments/assets/aa61df2c-d4bc-4b7e-9838-849428cc2d4f" width="350"> |
| 4 | vintage bookstore (text rendering) | <img src="https://github.com/user-attachments/assets/9ad049b8-b119-4362-a1c1-22bca1c9e848" width="350"> | <img src="https://github.com/user-attachments/assets/ae0a7f94-0a40-4f60-8a9f-3e2a692eceb2" width="350"> |
| 5 | campfire in dark forest | <img src="https://github.com/user-attachments/assets/e1294824-66b9-4b08-a839-512b47a55670" width="350"> | <img src="https://github.com/user-attachments/assets/7d974d4c-3a86-42fd-b65d-44665da1014b" width="350"> |
| 6 | ballet dancer mid-leap | <img src="https://github.com/user-attachments/assets/93d66c5f-bec8-489a-867a-f6c18830fb7a" width="350"> | <img src="https://github.com/user-attachments/assets/6e4a3a45-1d29-4565-84e6-51bb761cef57" width="350"> |
| 7 | coffee on wooden table (best LPIPS / #1986 reference) | <img src="https://github.com/user-attachments/assets/85cdb881-ec98-4b24-b6bb-3abe81566cb4" width="350"> | <img src="https://github.com/user-attachments/assets/4d63f568-54c6-4242-b05e-38f81c2049ed" width="350"> |

</details>

## Roadmap (forward-looking; not in this PR)

The dispatch architecture in `svdquant_dispatch.py` is built so new backends drop in as siblings — each backend is a single module exposing three functions (`supports(cap, precision) -> bool`, `prepare_weights(layer, precision)`, `apply(layer, x, bias)`), and `select_backend()` returns the first one that claims the active platform. Adding a new backend requires zero changes to `DiffusionSVDQuantLinearMethod` or the on-disk format. The on-disk canonical row-major NVFP4 (or INT4-nibble) layout is the explicit cross-backend contract — one checkpoint serves all of them.

| Status | Backend | Hardware | Module | Notes |
|---|---|---|---|---|
| ✅ Shipped (this PR) | `nunchaku` | Consumer NVIDIA: SM_75 Turing, SM_80/86/89 Ampere/Ada, SM_120 consumer Blackwell | `svdquant_nunchaku.py` | PTX-MMA fragment layout; `prepare_weights` repacks row-major → fragment at load time. Hopper SM_90 deliberately excluded (no validated kernel family). |
| 🛠️ Planned | `flashinfer` | Datacenter Blackwell: SM_100 (B200), SM_103 (GB300) | `svdquant_flashinfer.py` (TBD) | Native CuTe DSL W4A4 kernel landing in [FlashInfer](https://github.com/flashinfer-ai/flashinfer) so SGLang and vllm-omni share the same primitive. Consumes the on-disk canonical row-major NVFP4 directly — no second checkpoint needed. |
| 🛠️ Planned | `npu` (Ascend 910x) | Huawei Ascend 910 / 910B / 910C via `torch_npu` | `svdquant_npu.py` (TBD) | Mirrors the existing `Diffusion{Int8,MXFP4,MXFP8}Config` NPU path: capability detection via `current_omni_platform.is_npu()`, kernel call via `torch_npu.npu_*`-family ops. Requires Huawei's W4A4 SVDQuant-equivalent primitive to ship in their CANN release; the on-disk format and dispatcher entry are ready ahead of time. |

Each future backend reuses the converter, the LinearMethod, the factory registration, the test suite, and the hardware gate. Only the GEMM/quantize call sites are new.

## Other follow-ups (not in this PR)

- **Per-component quantization config** ([RFC #1044](https://github.com/vllm-project/vllm-omni/issues/1044) `OmniDiffusionConfig.quantization_targets`): the current `modules_to_not_convert: ["lm_head"]` skip is a substring escape hatch. The cleaner long-term fix mirrors PR #2702 (Qwen3-Omni encoder fix): pass `quant_config=None` explicitly to non-quantized subcomponents at pipeline construction in `pipeline_z_image.py`. That requires plumbing `quant_config` through `create_transformers_model` and is orthogonal to this PR.
- **Bench scaffolding extraction**: if reviewers prefer, the 4 `[Benchmark]` commits in this PR can move to a separate fix-only PR.

## Closes / Refs

- Closes #1986 (the closed nunchaku integration PR — this re-opens with the post-pivot architecture)
- Supersedes closed PR vllm-project/vllm#43471 (in-vllm proposal, abandoned in favor of consolidating here)
- Supersedes closed RFC vllm-project/vllm#37908 (per the closing comment, the SVDQuant stack lands in vllm-omni; the FlashInfer SM_100/103 plan from that RFC still stands as a Roadmap item above)
- Tracker row in #1854: Z-Image | Nunchaku SVDQuant W4A4 | S | Blackwell

---

**AI assistance**: this PR's commits and PR description were produced with Claude Code assistance. Every change was reviewed and validated end-to-end on RTX 5090 + RTX 5060 Ti by the human submitter before push.
