# LVSA Long-Video Showcase

**Training-free block-sparse attention for long-video diffusion**, running on
vLLM-Omni via the public plugin entry points (no core changes). LVSA speeds up
extended-horizon generation and prevents the freeze/loop failure mode dense
attention exhibits beyond a model's training length — with no fine-tuning and no
weight changes.

Upstream: **[LongVideoSparseAttention](https://github.com/JiusiServe/LongVideoSparseAttention)**
(algorithm, all model adapters, full docs).

## Results

Wan 2.1 1.3B, single A100 80 GB, mean over 5 prompts (Dense vs LVSA-FlashInfer):

| Horizon | 1× | 2× | 3× | 4× | 5× | 6× |
|---|---|---|---|---|---|---|
| **Speedup** | 0.9× | 1.4× | 1.9× | 2.5× | 3.0× | **3.5×** |
| **VQeval composite** (Dense→LVSA) | 63→63 | 59→63 | 61→63 | 61→64 | 59→64 | 59→63 |

The speedup **grows with length** (LVSA is the dense regime at 1× by design, so
the win starts at 2×), and quality *improves* at extension — LVSA's rotating
keyframes prevent the looping/static output dense produces beyond its horizon
(VQeval loop-quality +27 to +34 at ≥3×). At long horizons dense **OOMs on 80 GB**
(HunyuanVideo ≥2×, Wan2.1-14B and Cosmos at 6×) where LVSA still generates.

→ Full sweep (5 models × 6 horizons, VQeval + VBench-Long) and the
quality-metric methodology:
**[benchmarks/](https://github.com/JiusiServe/LongVideoSparseAttention/tree/main/benchmarks)**.

## Quickstart

```bash
# 1. vLLM-Omni 0.22.0 (stable) + matching vLLM
pip install "vllm==0.22.0"
pip install --no-build-isolation \
  "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@v0.22.0"

# 2. LVSA core + plugin
git clone https://github.com/JiusiServe/LongVideoSparseAttention.git
( cd LongVideoSparseAttention && pip install -e . && pip install -e lvsa-vllm-omni/ )

# 3. this showcase's client dep
pip install requests
```

FlashInfer (used by the default `flashinfer` backend) **ships with vLLM** — step 1
already pulls the pinned `flashinfer-python` + `flashinfer-cubin`, so the fused
backend works with no extra install. Two optional extras:

```bash
# (optional) faster startup — prebuilt JIT cache, matched to your CUDA + flashinfer
CU=cu128   # match your CUDA: cu121 / cu124 / cu128 / …
FIVER=$(python -c "import importlib.metadata as m; print(m.version('flashinfer-python'))")
pip install --extra-index-url "https://flashinfer.ai/whl/${CU}" \
  "flashinfer-jit-cache==${FIVER}+${CU}"

# (fallback) no FlashInfer in your environment? run the SDPA backend instead:
export LVSA_BACKEND=sdpa
```

```bash
# Serve an LVSA-enabled endpoint (Wan 2.1 1.3B at 2× horizon):
MODEL=/path/to/Wan2.1-T2V-1.3B-Diffusers MODEL_FAMILY=wan FRAMES=161 \
  bash serve_lvsa.sh

# In another shell — generate:
python generate.py --frames 161 --height 480 --width 832 \
  --prompt "A dog running through a sunlit forest." --out wan_lvsa.mp4
```

`serve_lvsa.sh` sets the LVSA env vars for the chosen `MODEL_FAMILY`
(`wan` / `hunyuan`). Wan and HunyuanVideo run through the LVSA **attention
backend** by default — it stays sparse even under sequence-parallel, unlike the
monkey-patch hooks.

## Configuration

The one knob you must get right is the model's training horizon
(`LVSA_REFERENCE_LATENT_FRAMES`: Wan = 21, HunyuanVideo = 33); `serve_lvsa.sh`
sets it per family. For the full environment-variable reference, the multi-GPU
support matrix, and tuning (`sparsity_scale`, rotating keyframes, non-standard
resolutions), see the upstream docs:

- **[lvsa-vllm-omni plugin reference](https://github.com/JiusiServe/LongVideoSparseAttention/blob/main/lvsa-vllm-omni/README.md)** — every `LVSA_*` env var
- **[docs/parallelism.md](https://github.com/JiusiServe/LongVideoSparseAttention/blob/main/docs/parallelism.md)** — TP / Ulysses / HSDP support per model
- **[docs/tuning.md](https://github.com/JiusiServe/LongVideoSparseAttention/blob/main/docs/tuning.md)** — picking knobs for your model and horizon
