# Alpamayo 2 Super — H200

> NVIDIA Alpamayo 2 Super driving policy: chain-of-thought reasoning and ego
> trajectory prediction from multi-camera video on one H200

## Summary

- Vendor: NVIDIA
- Model: `nvidia/Alpamayo2-Super`
- Task: Ego trajectory prediction with chain-of-thought reasoning
- Mode: Offline inference (`Omni`)
- Hardware: 1x NVIDIA H200 141 GB
- Maintainer: Community

## When to use this recipe

Use this recipe to predict a 6.4-second ego trajectory, together with the
model's driving reasoning, from a few seconds of multi-camera driving video and
the recent ego motion.

## Supported model contract

The policy runs as two stages on one GPU. Stage 0 decodes the Qwen3-VL
reasoning natively in vLLM and ends the request after `<|traj_future_start|>`.
Its KV cache is handed to stage 1, a diffusion stage that runs the
flow-matching action expert over it and returns the trajectory. Requests that
reach stage 1 together share one expert batch.

| Item | Contract |
| --- | --- |
| Images | Up to 7 cameras x 4 frames, camera-major, cameras sorted by index (0 cross left, 1 front wide, 2 cross right, 3 rear left, 4 rear tele, 5 rear right, 6 front tele); resized to the checkpoint's 163,840–196,608 pixel budget |
| Ego history | 16 poses at 10 Hz ending at t0, in the ego frame at t0: `ego_history_xyz` `(16, 3)`, `ego_history_rot` `(16, 3, 3)` |
| Output | Stage 0: the reasoning tokens. Stage 1: `actions`, `(64, 3)` ego positions at 10 Hz in the same frame |
| Deployment | BF16, TP=1, both stages on one GPU, eager ([`alpamayo2_super.yaml`](../../vllm_omni/deploy/alpamayo2_super.yaml)) |

## References

- Model card: <https://huggingface.co/nvidia/Alpamayo2-Super> (license
  `openmdw-1.1`; validated at revision `00554695`)
- Reference implementation and data loader: <https://github.com/NVlabs/alpamayo2>
  (commit `9596749e`)
- Dataset: <https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles>
- Supported models table: [`docs/models/supported_models.md`](../../docs/models/supported_models.md)
- Reasoning stage: [`vllm_omni/model_executor/models/alpamayo2_super`](../../vllm_omni/model_executor/models/alpamayo2_super)
- Expert stage: [`vllm_omni/diffusion/models/alpamayo2_super`](../../vllm_omni/diffusion/models/alpamayo2_super)
- Tests: [`tests/model_executor/models/test_alpamayo2_super.py`](../../tests/model_executor/models/test_alpamayo2_super.py),
  [`tests/diffusion/models/alpamayo2_super`](../../tests/diffusion/models/alpamayo2_super),
  [`tests/e2e/offline_inference/test_alpamayo2_super_expansion.py`](../../tests/e2e/offline_inference/test_alpamayo2_super_expansion.py)

## Hardware

- Accelerator model and per-device memory: NVIDIA H200, 141 GB (SXM or NVL)
- Number of devices: 1
- Device interconnect: not used (single GPU)
- Qualification scope: BF16, both stages on one GPU

## Software environment

- OS: Linux (x86_64)
- Python: 3.12
- Driver / runtime: CUDA 13.0
- vLLM version: 0.32.0
- vLLM-Omni version or commit: `main` at `c319a81e` or later

## Command

### Input data

The example loads one scene from the PhysicalAI-AV dataset with the
NVlabs/alpamayo2 loader, which streams the clip from Hugging Face (accept the
dataset terms and run `hf auth login` first) and decodes four frames per camera
before `t0_us`:

```bash
pip install git+https://github.com/NVlabs/alpamayo2@9596749e
```

The package is only used to load data; the model runs entirely in vLLM-Omni.

### Offline inference

Save as a script and run it with `python`. The engine starts its stages as
separate processes, so the code runs under `if __name__ == "__main__":`.

```python
from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset
from vllm import SamplingParams

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.models.alpamayo2_super import Alpamayo2SuperConfig
from vllm_omni.model_executor.models.alpamayo2_super.processing import (
    build_alpamayo2_super_tokenizer,
    build_trajectory_prompt,
)

MODEL = "nvidia/Alpamayo2-Super"


def main() -> None:
    scene = load_physical_aiavdataset("030c760c-ae38-49aa-9ad8-f5650a545d26", t0_us=5_100_000)
    frames = scene["image_frames"]  # (cameras, 4, 3, H, W) uint8
    history_xyz = scene["ego_history_xyz"].reshape(16, 3)
    history_rot = scene["ego_history_rot"].reshape(16, 3, 3)

    config = Alpamayo2SuperConfig.from_pretrained(MODEL)
    tokenizer = build_alpamayo2_super_tokenizer(MODEL, config.history_vocab_size, config.future_vocab_size)
    prompt = {
        "prompt_token_ids": build_trajectory_prompt(
            tokenizer,
            config,
            frames=frames,
            camera_indices=scene["camera_indices"].tolist(),
            ego_history_xyz=history_xyz,
        ),
        "multi_modal_data": {"image": list(frames.flatten(0, 1))},
    }
    reasoning_params = SamplingParams(
        temperature=0.6,
        top_p=0.98,
        max_tokens=256,
        seed=42,
        # Trajectory tokens are outside the stage tokenizer; decode reasoning with `tokenizer`.
        detokenize=False,
        stop_token_ids=[config.traj_ids["future_end"]],
    )
    expert_params = OmniDiffusionSamplingParams(
        seed=42,
        num_inference_steps=10,
        extra_args={"robot_obs": {"ego_history_xyz": history_xyz.tolist(), "ego_history_rot": history_rot.tolist()}},
    )

    omni = Omni(model=MODEL)
    generated = omni.generate(prompt, [reasoning_params, expert_params], use_tqdm=False)
    outputs = {output.stage_id: output for output in generated}
    omni.close()
    print(tokenizer.decode(outputs[0].outputs[0].token_ids, skip_special_tokens=True))
    trajectory = outputs[1].multimodal_output["actions"]
    print(trajectory[-1])  # position 6.4 s ahead

    # Verification
    assert trajectory.shape == (64, 3) and trajectory.isfinite().all()


if __name__ == "__main__":
    main()
```

Each request returns one trajectory. To sample K trajectories, submit K
requests with different seeds; requests in flight decode together and share
expert batches.

## Verification

The script ends with its check: one `(64, 3)` trajectory of finite positions.
The decoded reasoning is one or two
short driving sentences, for example "Stop to keep distance to the lead
vehicle.", and the last position is a plausible distance ahead of the ego
vehicle for the scene (meters, x forward).

### Qualification evidence

One bounded run, not a benchmark:

- Checkpoint `nvidia/Alpamayo2-Super` at revision `00554695`, BF16, the
  deploy config above, both stages on 1x H200 SXM 141 GB (host CPU AMD
  EPYC 7413), vLLM 0.32.0, vLLM-Omni `main` at `c319a81e`.
- Workload: 10 PhysicalAI-AV events, cameras `[0, 1, 2, 6]`,
  six requests per event (seeds 42–47) submitted together, 10 expert steps,
  temperature 0.6, top-p 0.98; one warm-up event excluded.
- Correctness, on a separate run over 100 PhysicalAI-AV events (the same
  events for the reference), six requests per event with
  seeds that differ per event: mean minADE over the six trajectories
  0.623 ± 0.028 m over seeds 42–46, against 0.672 ± 0.017 m for the
  NVlabs/alpamayo2 reference (`9596749e`), within seed and event variation;
  mean ADE over all trajectories 1.42 m against 1.43 m. The expert's
  initial noise is fixed by the request seed, so reusing the same seeds on
  every event reuses the same noise and widens the spread. Run one request at
  a time, repeated runs return identical reasoning and trajectories.
- Peak GPU memory (`nvidia-smi`, both stages): 105.2 GiB.
- Timing, warm: median 2.41 s per event for its six requests submitted
  together (median 2.36 s per request), and median 1.39 s per request when
  requests run one at a time; measured from the `AsyncOmni.generate` call to
  the trajectory, with the prompt already built and the frames already
  resized to the checkpoint's pixel budget.
- Both stages run eagerly at small batch sizes, so timing depends on the
  host CPU as well as the GPU. On an H200 NVL node with a different host CPU,
  the same workload on vLLM 0.31.0 measured 2.20 s per event, 0.76 s per
  request one at a time, and a 98.5 GiB peak.

## Notes

- Memory usage: about 67 GiB of BF16 weights for stage 0, which uses
  `gpu_memory_utilization: 0.6` including its KV cache; stage 1 holds the
  2.4B-parameter expert and the prefixes of its current batch.
- KV handoff: the deploy config uses `NixlConnector`, which moves each
  request's KV prefix (about 0.8 GB for 3k tokens) GPU to GPU with UCX's CUDA
  IPC transport. It sets `UCX_CUDA_IPC_ENABLE_GET_ZCOPY=y` for both stages;
  without it UCX falls back to TCP on GPUs without NVLink, and each handoff
  becomes far slower. NIXL with its UCX backend is required; the vLLM-Omni
  CUDA image includes it, otherwise install it with `pip install nixl`.
- Known limitations: the navigation-instruction (classifier-free guidance) and
  text-task (VQA, meta-action, auto-labeling, grounding) modes of the
  reference implementation are not supported yet; support for them is
  planned.

## Supported features

| Feature | Status | Guide |
| --- | --- | --- |
| Offline inference | Supported | This recipe |
| Online serving | Not supported | [OpenPI API](../../docs/serving/openpi_api.md) |
| Tensor parallelism | Not supported (TP=1) | [Stage configs](../../docs/configuration/stage_configs.md) |
| CUDA graphs | Not supported (eager) | [Stage configs](../../docs/configuration/stage_configs.md) |
| Quantization | Not supported (BF16) | [Quantization](../../docs/user_guide/quantization/overview.md) |
