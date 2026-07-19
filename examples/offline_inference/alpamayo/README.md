# Alpamayo — Offline Inference (engine-side ADE eval)

Drives the vllm-omni engine in-process (no HTTP) and computes minADE /
meanADE / minFDE against a clip's ground-truth future. Useful as a
parity check against the HTTP path — same model + seed should reproduce
the HTTP client's numbers within ~10 mm (AR sampling noise).

For the production HTTP path, see
[`examples/online_serving/alpamayo/`](../../online_serving/alpamayo/).

---

## 0. Prerequisites

Export once:

```bash
export ALPAMAYO_REPO=/path/to/vllm-omni-alpamayo              # this repo
export ALPAMAYO_MODEL=nvidia/Alpamayo-1.5-10B                 # HF id or local dir
export ALPAMAYO_VLM_BASE=Qwen/Qwen3-VL-8B-Instruct            # HF id or local dir
export ALPAMAYO_CLIP_PKL=/path/to/clip.pkl                    # see "clip format" below
```

Also:
- `vllm-omni` installed (`pip install -e .` from the repo root).
- A free GPU with ≥ 90 GB free (`nvidia-smi`).
- This offline eval uses the Qwen3-VL tokenizer directly via
  `ALPAMAYO_VLM_BASE` — no separate tokenizer bake step is required (the
  online-serving path needs it; see that README).

**Clip format.** `ALPAMAYO_CLIP_PKL` is a `pickle.load`-able dict with
keys `image_frames` (tensor `[C, F, 3, H, W]`),
`camera_indices` (length-C int list), `ego_history_xyz` (tensor
`[n_traj, T, 3]`), `ego_history_rot` (`[n_traj, T, 3, 3]`) and
`ego_future_xyz` (`[1, 1, T_future, 3]` GT used for ADE).

---

## 1. Run

```bash
CUDA_VISIBLE_DEVICES=0 \
ALPAMAYO_N_SAMPLES=4 \
python3 $ALPAMAYO_REPO/examples/offline_inference/alpamayo/eval_ade.py
```

Expected output:
```
[gen] ~10s
[actions] shape=(4, 64, 2) dtype=torch.bfloat16
clip=<clip-id>  n_samples=4
  minADE@4  ≈ 0.44 m
  meanADE@4 ≈ 2–3 m
  minFDE@4  ≈ 1 m
```

This spins up its own in-process engine (does NOT talk to a running
server). Use it to verify the HTTP path is producing the same numbers
as a direct-engine call.

The script's vLLM/engine init prints a wall of `INFO`/`WARNING` lines.
Filter to the relevant lines with:

```bash
... 2>&1 | grep -E '\[gen\]|\[actions\]|clip=|minADE|meanADE|minFDE|ERROR'
```

### Tunables (env vars)

| Var | Default | Why |
|---|---|---|
| `ALPAMAYO_MODEL` | `nvidia/Alpamayo-1.5-10B` | Model weights — HF id or local dir. |
| `ALPAMAYO_VLM_BASE` | `Qwen/Qwen3-VL-8B-Instruct` | Base tokenizer — HF id or local dir. |
| `ALPAMAYO_CLIP_PKL` | (required) | Clip pickle (images + ego history + GT future). |
| `ALPAMAYO_N_SAMPLES` | `4` | Per-request flow-matching noise rolls → `(N, 64, 2)` actions → minADE@N. Passed to engine as `sampling_params.extra_args["n_samples"]`. |
