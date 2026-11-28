# CAE-FastWAM LIBERO policy — MI355X

> World-action model policy serving over the OpenPI realtime WebSocket

## Summary

- Vendor: AMD (checkpoint `amd/CAE-FastWAM`, Apache-2.0), built on FastWAM
  (yuantianyuan01/FastWAM, MIT) and Wan2.2-TI2V-5B
- Model: `amd/CAE-FastWAM`
- Task: LIBERO manipulation policy: two camera frames + 8-D end-effector state +
  a language instruction → a continuous `[32, 7]` action chunk
  (6 end-effector deltas + gripper)
- Mode: Online serving via the OpenPI realtime robot API
  (`/v1/realtime/robot/openpi`)
- Hardware: 1x AMD Instinct MI355X
- Recommended deployment: `vllm_omni/deploy/cae_fastwam_libero.yaml` (eager)
- Maintainer: Community

## When to use this recipe

Use this recipe to serve CAE-FastWAM as a LIBERO policy: a simulator or robot
client connects over the OpenPI WebSocket, sends one observation per replan and
receives a chunk of 32 actions (the LIBERO protocol executes the first 10).

`FastWAMPipeline` serves the FastWAM family. CAE-FastWAM is the default and the
validated variant. The original FastWAM LIBERO checkpoint (`yuanty/fastwam`,
`libero_uncond_2cam224.pt`) loads through the same pipeline with
`vllm_omni/deploy/fastwam_libero.yaml`; that configuration is configuration-only
(not validated).

## Supported model contract

| Item | Value |
| --- | --- |
| Inputs | `observation/image` and `observation/wrist_image`: uint8 `HxWx3`, upright (the OpenPI LIBERO client rotates LIBERO frames by 180°); `observation/state`: 8 floats (eef position, eef axis-angle, gripper qpos); `prompt`: task instruction |
| Suite | `task_suite` (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`) selects the suite's normalisation statistics and action-codec embedding. Required on every request unless `model_config.task_suite` in the deploy config pins one suite (needed for the stock OpenPI LIBERO client, which does not send it); requests without either are rejected |
| Optional inputs | `seed` (deploy default 42, so unseeded requests use the same noise); `sampling_params.num_inference_steps` |
| Image processing | Each camera scaled to cover 224x224 (bilinear) and centre-cropped; cameras concatenated side by side |
| Output | float32 `[32, 7]`, de-normalised; gripper binarised to LIBERO's convention (+1 close, -1 open) |
| Steps | 10 by default (CAE-FastWAM was also published at 4) |
| Batch | One observation per request (`max_num_seqs: 1`) |
| State | None across requests; `session_id` and `reset` are accepted and ignored |
| Determinism | The request seed seeds a CPU generator for the initial noise, as the reference evaluator does (deploy default `seed: 42`). On MI355X, one convolution of the Wan VAE encoder (MIOpen) is not run-to-run deterministic by default, in the reference implementation as well; repeated chunks for the same seed differ by up to about 4e-2 (normalised) / 3e-2 (env units). With `torch.use_deterministic_algorithms(True)` they are bitwise repeatable; this needs MIOpen's direct and GEMM convolution solvers, so unset `MIOPEN_DEBUG_CONV_DIRECT` and `MIOPEN_DEBUG_CONV_GEMM` first (the `vllm/vllm-openai-rocm:v0.34.0` image sets both to 0, and MIOpen then finds no deterministic solver for the VAE encoder convolution) |

| Profile | Deploy config | Status |
| --- | --- | --- |
| Eager, bfloat16 | `cae_fastwam_libero.yaml` | Qualified on MI355X: LIBERO 97.40 % (10 steps), 97.45 % (4 steps) |
| Regional `torch.compile`, bfloat16 | `cae_fastwam_libero.yaml` with `enforce_eager: false`, `diffusion_compile_granularity: regional`, `diffusion_compile_dynamic: false` | Qualified on MI355X: LIBERO 97.20 % (10 steps); 128 ms vs 240 ms eager per chunk |
| Original FastWAM checkpoint | `fastwam_libero.yaml` | Configuration-only |

## Weights

| Component | Params | Source |
| --- | ---: | --- |
| Video expert (Wan2.2-TI2V-5B DiT, fine-tuned) | 5.00 B | `amd/CAE-FastWAM/policy.pt` |
| Action expert (32-wide, 30 layers) + proprio encoder | 25.2 M | `amd/CAE-FastWAM/policy.pt` |
| Action codec (action-VAE decoder, float32) | 0.23 M | `amd/CAE-FastWAM/action_vae.pt` |
| Normalisation statistics | — | `amd/CAE-FastWAM/stats/<suite>.json` |
| Wan2.2 VAE encoder, umT5-XXL encoder, tokenizer | 5.7 B (umT5) | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (`vae/`, `text_encoder/`, `tokenizer/`) |

The `.pt` files are read with `torch.load(weights_only=True)`. Prompt contexts
are cached per unique instruction, so the umT5 encoder runs once per task; it
still stays resident on the device (about 11 GB in bfloat16). The Wan2.2 video
output head in `policy.pt` is not loaded (action-only inference never uses it).

## References

- Checkpoint: <https://huggingface.co/amd/CAE-FastWAM>
- Reference code: <https://github.com/AMD-AGI/faster-action-models>
  (`FastWAM/src`, `src/cae-fastwam`, `model_cards/fastwam-libero/README.md`)
- FastWAM: <https://github.com/yuantianyuan01/FastWAM>
- Pipeline: [`vllm_omni/diffusion/models/fastwam/`](../../vllm_omni/diffusion/models/fastwam/)
- OpenPI API: [`docs/serving/openpi_api.md`](../../docs/serving/openpi_api.md)
- Tests: [`tests/diffusion/models/fastwam/`](../../tests/diffusion/models/fastwam/),
  [`tests/e2e/online_serving/test_fastwam_openpi_expansion.py`](../../tests/e2e/online_serving/test_fastwam_openpi_expansion.py)

## Hardware

- Accelerator: AMD Instinct MI355X, 288 GB HBM3E
- Number of devices: 1
- Interconnect: not used (single device)
- Qualification scope: one MI355X (gfx950), batch 1, bfloat16, eager and regional `torch.compile`, 10 and 4 denoising steps

## Software environment

- OS: Ubuntu 22.04.5 (`vllm/vllm-openai-rocm:v0.34.0` image)
- Python: 3.12.16
- Runtime: ROCm SDK 10.0.0 (HIP 7.15), PyTorch 2.13.0+rocm10.0.0
- vLLM version: 0.34.0
- vLLM-Omni version or commit: `e71053ae` plus this change
- The 2000-episode LIBERO results and the latency numbers below were measured
  before the rebase, on `vllm/vllm-openai-rocm:v0.32.0` (ROCm 7.2.3, vLLM 0.32.0,
  vLLM-Omni `cbf35fb`). Parity, the GPU tests and a 400-episode LIBERO check
  were repeated on the v0.34.0 image (see Qualification evidence)

## Command

```bash
vllm serve amd/CAE-FastWAM --omni --host 0.0.0.0 --port 8091 \
  --deploy-config vllm_omni/deploy/cae_fastwam_libero.yaml
```

For regional `torch.compile`, set `enforce_eager: false` and
`diffusion_compile_dynamic: false` on the stage in a copy of the deploy config.
Every transformer layer of both experts is compiled; the umT5 and VAE encoders
stay eager. With `warmup: true` the engine's warm-up request runs a real forward
pass, so compilation happens at startup.

### Optional: AMD kernels from the Hugging Face Kernel Hub

An optional backend serves the action inference with AMD's CAE-FastWAM kernel
package (`amd/cae-fastwam-kernels`, pinned to version 1) instead of the stock
PyTorch path. It is **off by default** and enabled by one environment variable:

```bash
VLLM_OMNI_FASTWAM_KERNELS=hub vllm serve amd/CAE-FastWAM --omni --host 0.0.0.0 --port 8091 \
  --deploy-config vllm_omni/deploy/cae_fastwam_libero.yaml
```

- Keep the eager deploy config (`enforce_eager: true`). The package runs inside
  CAE-FastWAM's own compiled regions, so the runner does not add its own
  `torch.compile` on top of it.
- Requirements: an AMD Instinct MI355X (gfx950, the package's only validated
  architecture); the `kernels` library (already a vLLM-Omni dependency); access
  to the package on the Hub; and the reference implementation the package plugs
  into, AMD-AGI/faster-action-models `FastWAM/src` at commit `a5fa731`,
  importable as `fastwam` (for example `pip install --no-deps -e FastWAM` from a
  checkout of that commit). The package checks the reference files against its
  own sha256 pins and the checkpoint layout of `amd/CAE-FastWAM`.
- The backend builds the reference CAE-FastWAM model from the same assets as
  the stock path (`policy.pt` and `action_vae.pt`, plus the VAE and umT5 tensors
  of the Diffusers base repository, renamed to the reference layout). Observation
  parsing, normalisation and the LIBERO gripper post-processing are shared with
  the stock path.
- If anything is missing (another GPU architecture, no `kernels`, no Hub access,
  the reference package not installed), the server logs one warning naming the
  reason, for example `FastWAM kernel backend 'hub' requested
  (VLLM_OMNI_FASTWAM_KERNELS=hub) but not used: ... Serving with the stock
  PyTorch path.`, and serves the stock path. When the backend is active the log
  says `FastWAM kernel backend 'hub' active: amd/cae-fastwam-kernels version 1 ...`.
- Outputs are not bitwise equal to the stock path (fused kernels change the
  rounding); see Qualification evidence.

## Verification

```python
import numpy as np
from openpi_client import websocket_client_policy

policy = websocket_client_policy.WebsocketClientPolicy(
    host="ws://127.0.0.1:8091/v1/realtime/robot/openpi"
)
print(policy.get_server_metadata())
obs = {
    "observation/image": np.zeros((256, 256, 3), np.uint8),
    "observation/wrist_image": np.zeros((256, 256, 3), np.uint8),
    "observation/state": np.zeros(8, np.float32),
    "prompt": "put the bowl on the plate",
    "task_suite": "libero_goal",
}
actions = policy.infer(obs)
print(actions.shape, actions.dtype, np.unique(actions[:, -1]))
```

Expected: the metadata contains `action_horizon: 32`, `action_dim: 7`,
`image_resolution: [224, 224]`; the reply is `(32, 7) float32` with gripper
values in `{-1.0, 1.0}`.

## Qualification evidence

| Check | Result |
| --- | --- |
| Parity vs the reference eager path (CPU, bfloat16, seed 42, 10 steps) | Bitwise equal at every stage: VAE latents, umT5 context, all 10 velocity predictions, action chunk |
| Parity vs the reference eager path on MI355X (same PyTorch build, 4 suites x 4 observations x {10, 4} steps) | Bitwise equal (32/32 chunks) with `torch.use_deterministic_algorithms(True)`. With default kernels 17/32 are bitwise and the rest differ by up to 3.1e-2 (env units), because one Wan VAE encoder convolution is not run-to-run deterministic on ROCm; the reference implementation varies the same way |
| LIBERO success rate (4 suites x 10 tasks x 50 trials, seed 42, served over OpenPI, CPU MuJoCo), eager, 10 steps | 97.40 % (1948/2000): spatial 96.4, object 99.6, goal 96.2, libero_10 97.4 (published result with AMD's optimised kernel package: 97.28 ± 0.20 %) |
| Same, eager, 4 steps | 97.45 % (1949/2000): spatial 96.6, object 99.8, goal 97.0, libero_10 96.4 (published, AMD kernel package: 97.33 ± 0.26 %) |
| Same, regional `torch.compile`, 10 steps | 97.20 % (1944/2000): spatial 97.0, object 99.4, goal 95.8, libero_10 96.6 |
| Latency, B=1, `infer_action` with cached prompt context, median of 7 interleaved rounds x 20 calls | Eager 240 ms (10 steps) / 113 ms (4 steps); regional `torch.compile` 128 ms / 61 ms; reference eager 271 ms / 128 ms |
| OpenPI round trip, one client, idle server (256x256 frames) | Eager 263 ms (10 steps) / 136 ms (4 steps); regional `torch.compile` 148 ms / 82 ms |
| Peak device memory (allocated, one policy incl. umT5) | 20.4 GiB |
| Re-check on `vllm/vllm-openai-rocm:v0.34.0` after the rebase: parity (deterministic, MIOpen convolution variables unset on both sides) | Bitwise equal (32/32 chunks) |
| Same image: LIBERO success rate, 400 episodes per setting (4 suites x 10 tasks x 10 trials, init states 0-9, seed 42) | Eager 10 steps 97.75 % (391/400), eager 4 steps 97.00 % (388/400), regional `torch.compile` 10 steps 96.50 % (386/400); each within one standard error (about 0.85 pp) of the 2000-episode result above |
| Optional Kernel Hub backend (`VLLM_OMNI_FASTWAM_KERNELS=hub`, `amd/cae-fastwam-kernels` version 1) on MI355X, `vllm/vllm-openai-rocm:v0.34.0`: action parity vs the stock eager path (4 suites x 4 observations x {10, 4} steps, seed 42) | Not bitwise (0/32, fused kernels); max difference 0.042 env units (dims 0-5), gripper equal on 1023/1024 rows. For scale: the stock path against itself in a second process 0.023, stock eager vs regional `torch.compile` 0.036. Golden rows vs the CPU reference 0.0101 (test tolerance 2e-2), gripper equal |
| Same image: served latency through `vllm serve` + OpenPI (B=1, warm-up excluded, 4 interleaved rounds, fresh server per round; pipeline forward inside the server) | Eager 239.1 / 123.5 ms (10 / 4 steps); regional `torch.compile` 138.3 / 77.7 ms; Kernel Hub backend 40.0 / 34.1 ms (6.0x / 3.5x faster than eager / regional compile at 10 steps; round ranges do not overlap). OpenPI round trip on localhost, 10 steps: 242.5 / 141.7 / 43.5 ms |
| Same image: LIBERO success, stock eager vs the Kernel Hub backend, 10 steps, served, paired by episode (4 suites x 10 tasks x 10 initial states, seed 42) | Stock 96.25 % (385/400), Kernel Hub 97.00 % (388/400); difference +0.75 pp (95 % CI -1.15 to +2.65 pp) |

## Notes

- The default path has no custom kernels, fp8 or tuning tables: it is the plain
  PyTorch path. AMD's optimised kernels are available through the optional
  Kernel Hub backend above (`VLLM_OMNI_FASTWAM_KERNELS=hub`).
- Two ERROR blocks at server start are expected. `amd/CAE-FastWAM` has a root
  `config.json` with `model_type: fastwam` (it names the FastWAM preset and the
  checkpoint layout). vLLM's generic Hugging Face config probe (`repo_utils`)
  tries to parse it as a Transformers config, logs "The checkpoint you are
  trying to load has model type `fastwam` but Transformers does not recognize
  this architecture", retries once and logs the same block again. FastWAM does
  not use a Transformers config: the server then loads `FastWAMPipeline` and
  serves normally. The messages are harmless; upgrading Transformers, as they
  suggest, is not needed.
- Known limitations: one observation per forward; LIBERO two-camera layout only
  (RoboTwin three-camera checkpoints are not supported); the optional-IDM
  inference mode and future-video prediction are not implemented.

## Supported features

| Feature | Status |
| --- | --- |
| Eager execution | ✅︎ (default) |
| Regional `torch.compile` | ✅︎ (opt-in, see Command); qualified on MI355X |
| Request batching | ✗ (`max_num_seqs: 1`) |
| Cache acceleration (cache-dit / TeaCache) | ✗ (not applicable to the action loop) |
| Tensor / sequence / CFG parallelism | ✗ |
| Original FastWAM checkpoints | Configuration-only |
