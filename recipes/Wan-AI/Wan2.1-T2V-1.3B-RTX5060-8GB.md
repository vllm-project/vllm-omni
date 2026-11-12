# Wan2.1 T2V 1.3B — RTX 5060 Laptop 8 GB

## Summary

- Vendor: Wan-AI
- Model: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
- Task: text-to-video
- Mode: online serving, one request at a time
- Hardware: 1x NVIDIA GeForce RTX 5060 Laptop, 8151 MiB
- Maintainer: community

## When to use this recipe

The BF16 UMT5 text encoder is larger than this GPU, and the DiT on top of it
does not fit either. The profile keeps the text encoder on CPU and streams both
it and the DiT block by block. 832×480 does not fit at any step count tried.

## Supported model contract

| Task | Input | Output | Profile |
| --- | --- | --- | --- |
| Text-to-video | one prompt | 640×368, 81 frames (5 s at 16 fps), 40 steps | `wan21_t2v_1_3b_8gb.yaml` |
| Text-to-video | one prompt | 768×432, 33 frames (2 s at 16 fps), 40 steps | same profile, request overrides the size |

The profile's own default is 20 steps. The 40-step ceiling below was measured
with the same offload config and the step count set per request.

## References

- Model card: [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers), commit `0fad780a534b6463e45facd96134c9f345acfa5b`
- Deploy profile: [`vllm_omni/deploy/wan21_t2v_1_3b_8gb.yaml`](../../vllm_omni/deploy/wan21_t2v_1_3b_8gb.yaml)
- Layerwise offload: [cpu_offload.md](../../docs/user_guide/diffusion/cpu_offload.md)
- Tracking issue: [vllm-omni#10008](https://github.com/vllm-project/vllm-omni/issues/10008)

## Hardware

- Accelerator: NVIDIA GeForce RTX 5060 Laptop, 8151 MiB, SM120 (Blackwell).
- Devices: 1. No interconnect.
- Host memory: 30 GB. Serving used 22.7–24.5 GiB RSS, so there is little room for a second model and the desktop.
- Qualification: one machine, one run per cell, measured 2026-11-10.

## Software environment

- OS: Linux, container `vllm/vllm-omni:nightly`
- Python: 3.12
- Driver / runtime: NVIDIA driver 595.99.02, CUDA 13.0
- vLLM: 0.32.0
- vLLM-Omni: 0.32.0rc2.dev352+g670aa7ec6, source `670aa7ec6043571e0a8a16448685322e41d93828`

## Command

```bash
vllm serve Wan-AI/Wan2.1-T2V-1.3B-Diffusers --omni \
  --deploy-config vllm_omni/deploy/wan21_t2v_1_3b_8gb.yaml --port 8091
```

The model is served through the `wan2_2` pipeline. The profile sets that
explicitly because the checkpoint name does not select it.

## Verification

Send one text-to-video request at 640×368 and confirm the returned mp4 is
640×368. The qualified runs used guidance 6.0, flow shift 8.0 and seed 42.

## Notes

Device peak is `nvidia-smi` after the request returned. Request peak is the
engine's own per-request maximum. Cold start is launch to `/health` and is disk
bound on the HDD the weights were read from.

| Geometry | Steps | Offload | Device peak | Request peak | Latency | Outcome |
| --- | --- | --- | --- | --- | --- | --- |
| 480×272×17 | 20 | none | 7679 MiB | — | — | OOM during load |
| 480×272×17 | 20 | text encoder only | 7631 MiB | — | — | OOM in the dummy run |
| 480×272×17 | 20 | text encoder + DiT | 5431 MiB | 5494 MB | 26.65 s | served |
| 640×368×17 | 20 | text encoder + DiT | 4203 MiB | 4094 MB | 38.59 s | served |
| 640×368×81 | 40 | text encoder + DiT | 7379 MiB | 7092 MB | 351.7 s | served |
| 768×432×33 | 40 | text encoder + DiT | 7423 MiB | 7400 MB | 190.3 s | served |
| 832×480×17 | 20, 30, 40 | text encoder + DiT | 7603–7609 MiB | — | — | OOM |

Frame count barely moves the peak: 17, 33, 49 and 81 frames at 640×368 all
served. 832×480 is the failure, at every step count tried.

`vae_use_tiling` is applied by the diffusion registry, which sets
`vae.use_tiling`, and this VAE tiles once a side passes 256, so 480×272 does
tile. A run with the flag set peaked 1206 MiB lower, but the two runs also
differed in cold-start timing and the log shows layerwise offload disabled
right after profiling, so that drop is not isolated to tiling. The profile
therefore does not claim tiling.

## Supported features

| Feature | Status on this hardware |
| --- | --- |
| [Layerwise offload](../../docs/user_guide/diffusion/cpu_offload.md) | Required for both `text_encoder` and `dit`. `pin_memory: false`. |
| VAE tiling | Wired, but not claimed. See the note above. |
| Quantization, cache backends, sleep/wake | Not measured. |
