# Open-H3-Regenerate-2K — H200

> Regenerate a 768P MiniMax-H3 draft at 2560x1440 in three denoising steps

## Summary

- Vendor: NVlabs (`Efficient-Large-Model`)
- Model: [`Efficient-Large-Model/Open-H3-Regenerate-2K`](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K)
  over the `MiniMaxAI/MiniMax-H3` FL2VA partition
- Task: video-to-video refinement of an H3 draft (`t2va` layout)
- Mode: OpenAI-compatible `/v1/videos` HTTP serving
- Hardware: 2x NVIDIA H200 141GB
- Maintainer: Community

## When to use this recipe

Generate a clip cheaply at 1344x768 with MiniMax-H3, then re-render the chosen
draft at 2560x1440. The release is an open replacement for the closed
H3-Regenerate-2K API: the draft latent is upscaled by the 3D-conv latent
upscaler, put back on the flow at sigma 0.909375 and denoised in three Euler
steps by MiniMax-H3 carrying a merged LoRA (Open-H3 refiner + LightX2V FL2VA
Turbo + LMS Sharp), then decoded.

## Supported model contract

| Item | Contract |
| ---- | -------- |
| Request | `/v1/videos` or `/v1/videos/sync` with `source_video` (the draft) and `extra_params` `{"task": "t2va", "regenerate": true}`; no noise masks |
| Draft canvas | `width`, `height`, `num_frames` / `fps` describe the draft. The draft is resampled to 24 fps, scaled to cover the canvas and centre-cropped, then cut or padded (last frame held) to the frame count. Validated: 1344x768, 124 frames |
| Prompt | The draft's prompt; it conditions the refine pass |
| Output size | `latent_upscale` in `extra_params`, default `{"width": 2560, "height": 1440, "align": 32}`; the same request formats as [latent super-resolution](../MiniMaxAI/MiniMax-H3.md#latent-super-resolution) |
| Steps | `num_inference_steps` 3 (default, sigmas 0.909375, 6/7, 2/3) or 2 (sigmas 0.7, 0.494, from the release's shift-6 rule) |
| Sampler | Euler, CFG 1, video flow shift 6; `flow_shift`, `sampler`, `latent_refine` and per-request `lora` are rejected |
| Audio | The refiner conditions on encoded silence, as the release does. The output carries the draft's soundtrack after an H3 audio-VAE round trip, or silence if the draft has none |

Every request on this server is a regenerate request; serve plain MiniMax-H3
from a separate server.

## References

- Release: <https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K>
  at revision `5809c33730c18b6b1c827b40974855a4288641b3`, and its reference code in
  [NVlabs/Sana `models/Open-H3-Regenerate-2K`](https://github.com/NVlabs/Sana/tree/6e4a44002149418a547434b91d7280a8bbdcace3/models/Open-H3-Regenerate-2K)
  at commit `6e4a44002149418a547434b91d7280a8bbdcace3`
- Base model: <https://huggingface.co/MiniMaxAI/MiniMax-H3> at revision
  `48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc` (`FL2VA/`)
- Base model and request API: [MiniMax H3 recipe](../MiniMaxAI/MiniMax-H3.md),
  [Videos API](../../docs/serving/videos_api.md#minimax-h3-regenerate)

## Checkpoints

`vllm-omni` serves the native MiniMax-H3 FL2VA checkpoint and needs only the
adapter files and the latent upscaler from the release (about 14 GB); the
release's `base_model/` copy is not used.

```bash
export REGEN_DIR=/path/to/Open-H3-Regenerate-2K
hf download Efficient-Large-Model/Open-H3-Regenerate-2K \
  --include 'lora/*' 'latent_upsampler/*' --local-dir "${REGEN_DIR}"
```

`lora/open_h3_regenerate_2k_lora.safetensors` is fused into the transformer
while the checkpoint loads. The LMS Sharp AdaLN curve branches in
`lora/sharp_curve_branches.safetensors` have all-zero up factors in this
release, so they contribute nothing; the loader verifies that and does not need
the release's separately fetched `adaln_t_table`.

The fusion runs on the accelerator, one parameter at a time, on top of the
weights already loaded. The largest parameter needs up to 1.72 GiB at that
moment; the server logs the figure when it claims the bundle and reports an
out-of-memory error during the fusion as such.

## Hardware

- Accelerator model and per-device memory: NVIDIA H200, 141 GB
- Number of devices: 2 (Ulysses SP2)
- Device interconnect: NVLink
- Host memory: the Qwen3-VL text encoder is layer-offloaded to host memory
- Qualification scope: online serving of one 1344x768 -> 2560x1440, 124-frame request

## Software environment

- OS: Ubuntu 22.04.5 LTS
- Python: 3.12.15
- Driver / runtime: NVIDIA 590.48.01, CUDA 13.2, PyTorch 2.13.0
- vLLM version: 0.32.0
- vLLM-Omni version or commit: `8d4b2789a` plus this change

## Command

```bash
CUDA_VISIBLE_DEVICES=0,1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve "${MODEL_ROOT}/FL2VA" --omni \
  --served-model-name H3-Regenerate --task-type fl2va --trust-remote-code \
  --lora-path "${REGEN_DIR}" \
  --num-gpus 2 --usp 2 --ring 1 \
  --vae-patch-parallel-size 2 --vae-parallel-mode tile --vae-use-tiling \
  --diffusion-offload-config '{"mode":"layer","components":["text_encoder"]}' \
  --additional-config '{"latent_refine_max_tokens_per_rank":70000}' \
  --enforce-eager --diffusion-attention-backend FLASH_ATTN \
  --port 8091
```

- `--lora-path` names the release directory; the LoRA is fused at load time,
  so the DiT cannot be offloaded. Offloading the text encoder alone is
  supported.
- A 2560x1440, 124-frame refine packs about 133,000 video rows, 66,600 per
  rank at SP2, above the default 65,536-row refine guard; the
  `latent_refine_max_tokens_per_rank` override admits it.
- The release's latent upscaler is used unless `latent_upscaler_path` names
  another checkpoint.

## Verification

```bash
curl -sS http://127.0.0.1:8091/v1/videos/sync \
  -F "prompt=The prompt the draft was generated from" \
  -F width=1344 -F height=768 -F aspect_ratio=16:9 -F num_frames=124 -F fps=24 \
  -F seed=1766580932 \
  -F 'extra_params={"task":"t2va","regenerate":true}' \
  -F "source_video=@draft.mp4;type=video/mp4" \
  -o regenerated.mp4
ffprobe -v error -show_entries stream=codec_name,width,height,nb_frames -of csv=p=0 regenerated.mp4
```

Expected: `h264,2560,1440,124` and an `aac` stream.

## Notes

Qualification evidence on this hardware (single bounded runs, not benchmarks):

- **DiT parity.** One transformer forward on real weights: vLLM-Omni (native
  FL2VA checkpoint at the base revision above, LoRA fused at load) against
  Diffusers 0.40.0 `MiniMaxH3Transformer3DModel` (the release's `base_model/`
  at the release revision above, LoRA merged as `cast(W_fp32 + B @ A)` like
  the release). Fixed random 7x16x32 video latent, 48 text rows and audio rows
  locked at timestep 1, all drawn from seed 1234, at the first and last
  sigmas of the default 3-step refine. Each run needs one GPU with about 80 GB
  free:

  ```bash
  T=tools/minimax_h3/regenerate_dit_parity.py
  PYTHONPATH=. python $T reference ref.pt
  PYTHONPATH=. python $T reference --no-adapter ref-base.pt
  PYTHONPATH=. python $T vllm-omni omni.pt
  PYTHONPATH=. python $T vllm-omni --no-adapter omni-base.pt
  PYTHONPATH=. python $T compare ref.pt omni.pt            # column 1
  PYTHONPATH=. python $T compare ref-base.pt omni-base.pt  # column 2
  PYTHONPATH=. python $T compare ref.pt ref-base.pt        # column 3
  ```

  | sigma | vLLM-Omni vs reference, with adapter | same, base H3 without adapter | adapter vs base (reference) |
  | ----- | ------------------------------------ | ----------------------------- | --------------------------- |
  | 0.909375 | rel. L2 1.2e-2, cosine 0.99993 | rel. L2 2.2e-2, cosine 0.99976 | rel. L2 0.44 |
  | 0.666667 | rel. L2 2.8e-2, cosine 0.9996 | rel. L2 1.8e-2, cosine 0.99984 | rel. L2 0.38 |

  The adapted outputs agree as closely as the two base implementations do
  (BF16, FlashAttention vs cuDNN SDPA), while the adapter itself moves the
  velocity by about 40%. The script downloads both models at the pinned
  revisions (`--release-dir` and `--base-dir` take local copies instead) and
  skips the Sharp curve branches after asserting they are zero.
- **End to end.** A MiniMax-H3 T2VA draft (1344x768, 124 frames, 50 steps)
  regenerated with the command above, first request after startup:

  | Steps | Denoise | Request (encode, text, upscale, refine, decode, MP4) | Output |
  | ----- | ------- | ---------------------------------------------------- | ------ |
  | 3 | 77 s (25.8 s/step) | 112 s | 2560x1440, 124 frames, H.264 + AAC |
  | 2 | 51 s | 77 s | 2560x1440, 124 frames, H.264 + AAC |

  Peak process memory was 87.3 GB per DiT rank (nvidia-smi, 2 s sampling).
  Composition and motion follow the draft; the mean squared Laplacian
  response of the output is 15.8x (range 14.4-19.3, 11 frames) that of the
  draft upscaled bicubically to the same size.
- **Known limitations.** One request at a time. Only the default 2560x1440
  output was run end to end; other `latent_upscale` sizes follow the upscaler's
  limits but are unqualified. The release's accelerated `sol_engine_fwd`
  (Sol-Attn, MXFP8, CUDA graphs) and its fast VAE decoder are not ported.

## Supported features

| Feature | Status |
| ------- | ------ |
| Ulysses sequence parallelism | Supported (SP2 validated) |
| Text-encoder layerwise offload | Supported |
| DiT offload | Not supported (the LoRA is fused while the checkpoint streams in) |
| Step execution, continuation, latent-mask editing, timeline guides | Not supported |
| Cache-DiT / TeaCache | Not supported (`quality=lossless` only) |
| Separate encoder stage | Not supported (the DiT stage encodes the silence the refiner conditions on) |
