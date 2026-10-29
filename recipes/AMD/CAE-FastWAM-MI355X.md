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
| Optional inputs | `seed` (deploy default 42, so unseeded requests are deterministic); `sampling_params.num_inference_steps` |
| Image processing | Each camera scaled to cover 224x224 (bilinear) and centre-cropped; cameras concatenated side by side |
| Output | float32 `[32, 7]`, de-normalised; gripper binarised to LIBERO's convention (+1 close, -1 open) |
| Steps | 10 by default (CAE-FastWAM was also published at 4) |
| Batch | One observation per request (`max_num_seqs: 1`) |
| State | None across requests; `session_id` and `reset` are accepted and ignored |
| Determinism | The request seed seeds a CPU generator for the initial noise, as the reference evaluator does (deploy default `seed: 42`) |

| Profile | Deploy config | Status |
| --- | --- | --- |
| Eager, bfloat16 | `cae_fastwam_libero.yaml` | TODO(validation): qualify on MI355X |
| Regional `torch.compile`, bfloat16 | `cae_fastwam_libero.yaml` with `enforce_eager: false`, `diffusion_compile_dynamic: false` | Configuration-only until qualified on MI355X (TODO(validation)) |
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
- Qualification scope: TODO(validation)

## Software environment

- OS: TODO(validation)
- Python: TODO(validation)
- Driver / runtime: TODO(validation) (ROCm version)
- vLLM version: TODO(validation)
- vLLM-Omni version or commit: TODO(validation)

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
| Parity vs the reference eager path on MI355X | TODO(validation) |
| LIBERO success rate (4 suites x 10 tasks x 50 trials, seed 42, 10 steps), eager | TODO(validation) (published GEAK-kernel result: 97.28 ± 0.20 %) |
| Latency, B=1, 10 / 4 steps, eager and regional compile | TODO(validation) |
| Peak device memory | TODO(validation) |

## Notes

- No custom kernels, fp8 or tuning tables: this is the plain PyTorch path. The
  published 12.82 ms (10 steps) latency used the GEAK kernels from the
  reference repository and is not comparable.
- Known limitations: one observation per forward; LIBERO two-camera layout only
  (RoboTwin three-camera checkpoints are not supported); the optional-IDM
  inference mode and future-video prediction are not implemented.

## Supported features

| Feature | Status |
| --- | --- |
| Eager execution | ✅︎ (default) |
| Regional `torch.compile` | Configuration-only: wired up (opt-in, see Command) but not yet verified on any device |
| Request batching | ✗ (`max_num_seqs: 1`) |
| Cache acceleration (cache-dit / TeaCache) | ✗ (not applicable to the action loop) |
| Tensor / sequence / CFG parallelism | ✗ |
| Original FastWAM checkpoints | Configuration-only |
