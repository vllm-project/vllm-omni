# SeedVR2 3B video restoration

The native `SeedVR2Pipeline` restores an input video using the released 3B FP16
NaDiT and s8/c16/t4 causal VAE. It uses fixed checkpoint conditioning, CFG=1,
and one Euler step. Text prompts do not change conditioning.

## Model directory

Place these files together, retaining their names:

| File | SHA-256 of the validated release |
| --- | --- |
| `seedvr2_ema_3b_fp16.safetensors` | `2fd0e03a3dad24e07086750360727ca437de4ecd456f769856e960ae93e2b304` |
| `ema_vae_fp16.safetensors` | `20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1` |
| `pos_emb.pt` | `fa07a14844314772266b66c3b95deb0027696d8fe7065721263db5176f45d799` |

Add `model_index.json` containing:

```json
{"_class_name": "SeedVR2Pipeline"}
```

The loader checks every checkpoint tensor, including RoPE buffers, and rejects
missing, duplicate, unexpected, or incompatible tensors. Install PyAV for video
decoding. The validated environment uses PyAV 18.1, PyTorch 2.13/CUDA 13, and L20 GPUs.

## Serve and restore a video

```bash
vllm serve "$MODEL_DIR" --omni \
  --model-class-name SeedVR2Pipeline --dtype float16 --enforce-eager \
  --num-gpus 1 --host 127.0.0.1 --port 8098

curl --fail-with-body http://127.0.0.1:8098/v1/videos/sync \
  -F 'prompt= ' \
  -F 'input_references=@input.mp4;type=video/mp4' \
  -F 'size=224x128' -F 'num_inference_steps=1' \
  -F 'guidance_scale=1' -F 'seed=7723' \
  --output restored.mp4
```

For two-rank window sequence parallelism, expose two GPUs and replace `--num-gpus 1` with:

```bash
--num-gpus 2 --distributed-executor-backend mp \
  --stage-overrides '{"0":{"ulysses_degree":2}}'
```

Use degree 4 and four visible GPUs for SP4. Window-SP alone replicates the VAE.
To shard VAE activations across the same ranks, use:

```bash
--vae-use-tiling --num-gpus 2 --distributed-executor-backend mp \
  --stage-overrides '{"0":{"ulysses_degree":2,"vae_patch_parallel_size":2,"vae_parallel_mode":"spatial_shard_height"}}'
```

The VAE patch degree must match the window-SP degree.
For a 16-frame 1280×720 input restored at its original size, use four GPUs,
VAE tiling, and height sharding with both degrees set to four:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "$MODEL_DIR" --omni \
  --model-class-name SeedVR2Pipeline --dtype float16 --enforce-eager \
  --vae-use-tiling --num-gpus 4 --distributed-executor-backend mp \
  --stage-overrides '{"0":{"ulysses_degree":4,"vae_patch_parallel_size":4,"vae_parallel_mode":"spatial_shard_height"}}' \
  --host 127.0.0.1 --port 8098
```

Request `size=1280x720` for this landscape input or `size=720x1280` for a
portrait input of the same dimensions. Keep the requested output size equal to
the input size when testing restoration without upscaling.

The multipart API requires a prompt field; a single space supplies a blank
prompt. Nonblank text is rejected. Omit `fps` to retain the source frame rate;
an explicit rate must match the source. Output dimensions are explicit multiples
of 16. Input frames are resized with antialiased bicubic interpolation. The final
frame is repeated internally to reach 4n+1, and the decoded output is cropped back
to the original frame count. Five frames remain five; six frames are internally
padded to nine and return six.

## Colour correction

Restoration reproduces detail faithfully but shifts global colour, so the
resized input is used to transfer colour back onto the restored frames. This is
on by default; pass `color_correction_method` to choose how:

| Value | Behavior |
| --- | --- |
| `lab` (default) | Swaps the lowest frequency band, then histogram-matches CIELAB chroma. Most faithful colour. |
| `wavelet` | Swaps only the lowest frequency band, keeping every higher band from the restoration. |
| `adain` | Matches per-channel mean and standard deviation. Cheapest, and corrects only a global tint. |
| `none` | Skips the transfer and returns raw restored colour. |

```bash
curl --fail-with-body http://127.0.0.1:8098/v1/videos/sync \
  -F 'prompt= ' -F 'input_references=@input.mp4;type=video/mp4' \
  -F 'size=224x128' -F 'num_inference_steps=1' -F 'guidance_scale=1' \
  -F 'seed=7723' -F 'color_correction_method=wavelet' \
  --output restored.mp4
```

`lab` and `wavelet` retain the restored high-frequency detail; `adain` measurably
softens it because it rescales every frequency. The transfer runs in FP32 after
the VAE decode and costs a fraction of a second per frame. Unrecognized values
are rejected with 400 before the clip is admitted. Correction cannot recover
colour lost to 4:2:0 chroma subsampling in the returned MP4, which dominates the
remaining deviation once the transfer is applied.

## Request admission

The default 3B limit is 848×480 pixels per input or output frame and 2,035,200
pixels across the requested output clip. With VAE tiling and window SP of at
least four ranks whose `vae_patch_parallel_size` matches `ulysses_degree`, the
per-frame limit rises to 2560×1472 and the clip budget to 18,841,600 pixels.
Raising the degree never lowers the budget. Both budgets count temporal
padding to 4n+1 frames: the default admits up to five 848×480 frames or 93
192×112 frames; the sharded profile admits up to 45 848×480 frames or five
2560×1472 frames. A 16-frame 1280×720 or 720×1280 clip pads to 17 frames and
uses 15,667,200 of the sharded clip's 18,841,600 pixels. Both orientations passed
original-size HTTP restoration on four RTX 5090 GPUs with VAE tiling and height
sharding. An independent 257-frame cap bounds per-frame decoder work for tiny
inputs. The other longer combinations above are admission bounds, not completed
GPU validation. The decoder checks
declared duration, frame count, and input dimensions when available, then
enforces the limits as frames arrive. Requests outside these budgets return 400
before building the resized whole-clip tensor.

The per-frame cap is calibrated for the smallest qualified device. The clip cap
scales with the smallest visible device's memory: it stays at the calibrated
five 2560×1472 frames on a 32 GB device and grows on larger ones from a
per-rank memory model measured at SP4 (see `sharded_budget` in
`vllm_omni/diffusion/models/seedvr2/video.py`), which keeps 15% of the device
free and leaves room for allocator fragmentation. It stops at 2,118,057,984
padded pixels (513 frames at 1536×2688), the largest clip validated on four
B300 ranks; 80 GB devices get about 1.08 billion and 141 GB or larger devices
reach the cap. Clips at the cap in both memory regimes (561 frames at 2560×1472
and 2,049 frames at 768×1344) peaked at 54 GiB of activations per rank. Set
these before starting the server to override the caps; each must be a positive
integer, and validating an override on the target hardware is the operator's
responsibility:

| Variable | Default | Bounds |
| --- | --- | --- |
| `VLLM_OMNI_SEEDVR2_SHARDED_FRAME_PIXELS` | 3,768,320 | Pixels per frame on the sharded profile |
| `VLLM_OMNI_SEEDVR2_SHARDED_CLIP_PIXELS` | 18,841,600, scaled up with device memory | Padded pixels per clip on the sharded profile |
| `VLLM_OMNI_SEEDVR2_MAX_FRAMES` | 257 | Decoder-work frame cap, all profiles |

The sharded profile applies to any `ulysses_degree` of four or more, so eight
ranks inherit a budget calibrated on four. Raising these caps trades a 400 for a
CUDA OOM: the admission check passes and the request fails later inside the DiT
forward or the VAE decode, which can abort every in-flight request on the engine
rather than only the oversized one. Before serving with raised caps, restore the
largest clip they admit once and confirm it completes.

A 362-frame 1536×2688 2x upscale was restored on eight ranks with
`ulysses_degree=8`, VAE tiling and height sharding, using 4,300,000 for the
per-frame cap, the memory-scaled clip cap and 2,000 for the frame cap. The long-video
route sizes its windows from these caps, so the clip cap sets how many frames
each window covers.

## Long-video restoration

`POST /v1/seedvr2/restore-long` accepts one uploaded 24 FPS video, a blank
prompt, `size`, `num_frames` (up to 7,200), and optional `loop_input=true` and
`color_correction_method`, which it applies to every window.
An output frame must fit the sharded per-frame pixel cap. It returns a job ID; poll
`GET /v1/seedvr2/restore-long/{id}` and download the completed MP4 from
`GET /v1/seedvr2/restore-long/{id}/content`. `DELETE
/v1/seedvr2/restore-long/{id}` asks a running job to stop; it settles as
`cancelled` at the next window boundary. A job whose owning process is gone,
such as after a server restart, reports `failed` on the next poll.

Jobs live on the API server's local disk, so the route requires
`--api-server-count 1` and runs one job at a time. Storage is bounded by an
upload cap and by deleting settled jobs once they age out:

| Variable | Default | Bounds |
| --- | --- | --- |
| `VLLM_OMNI_SEEDVR2_LONG_OUTPUT_DIR` | system temp dir | Parent of the job directories |
| `VLLM_OMNI_SEEDVR2_LONG_MAX_UPLOAD_BYTES` | 8 GiB | Largest accepted upload |
| `VLLM_OMNI_SEEDVR2_LONG_JOB_TTL_SECONDS` | 3,600 | Age at which a settled job is deleted |
| `VLLM_OMNI_SEEDVR2_LONG_MAX_WINDOW` | 121 | Longest model window, in frames |

Download a completed job before its TTL expires; the sweep runs on each new
submission and removes the output with the job directory.

The service sends windows through the existing SeedVR2 endpoint, blends four
frames at each boundary, and writes one continuous MP4 encoder. Each window is
the longest 4n+1-frame clip that the sharded clip-pixel cap, the frame cap and
`VLLM_OMNI_SEEDVR2_LONG_MAX_WINDOW` admit at the output size. Every window pays
a fixed round trip through the serving stack and recomputes its overlap, and a
window of four latent frames gets no temporal attention in the DiT, so raise
the clip cap as far as the device allows. Windows carry source pixels, which
the model resizes on the device, and cross the endpoint losslessly; only the
final MP4 is compressed. The next window is restored while the previous one is
blended and encoded. It repeats source frames and audio only when
`loop_input=true`. The sharded serving profile above (VAE tiling, VAE height
sharding, Ulysses degree of at least four) is required. This route keeps a
window within the whole-clip pixel budget; it does not raise the
`/v1/videos/sync` 257-frame or pixel limits. The long route
uses fixed one-step, CFG=1 conditioning and requires `imageio[ffmpeg]` for
audio muxing. The 7,200-frame 768×1344 case is still undergoing full GPU E2E.

## Temporal and spatial VAE tiling

Add `--vae-use-tiling` to bound VAE intermediate activations along time. The
encoder processes nine frames first, then eight per chunk; the decoder processes
two latent frames per chunk. Each causal convolution carries its past inputs
within that encode/decode call, with temporal stride and first-frame upsampling
alignment preserved. GroupNorm and attention still see the entire spatial frame.
There is no overlap blend or independent restart at chunk boundaries.

Five-frame clips use one temporal chunk. On a single GPU, convolutions with
more than 256 input rows additionally process tiles of 128 output rows with
exact halos. GroupNorm and bottleneck attention retain full-frame statistics.
This bounds convolution workspace; full-frame activations and the DiT's
whole-clip activations still consume memory.

With VAE patch parallelism, each rank owns a band of latent rows and the
corresponding encoder/decoder rows. Neighbor exchanges supply convolution halos;
all-reduced centered statistics preserve full-frame GroupNorm. Bottleneck
attention gathers the low-resolution frame before splitting it again. Encoder
parameters are gathered before latent sampling, and decoded pixels are gathered
before returning the video. Unequal bands are supported; fewer latent rows than
ranks fall back to replicated execution. Width sharding and batch slicing are
unsupported. FP16 reduction and convolution order can change rounding, so
parallel outputs are numerically close rather than bitwise identical.

## Current integration scope

| Property | Contract |
| --- | --- |
| Weights / dtype | Released 3B DiT and VAE, FP16 |
| Reference semantics | C0 whole clip, colour correction on by default |
| Video input | One uploaded file, or offline TCHW RGB floats / PIL frames |
| Timing | Constant frame rate, increasing PTS; normalize the video origin to zero |
| Audio | First mono/stereo track, aligned by source PTS, cropped to the video interval, re-encoded as AAC |
| Randomness | Per-request generator; preserve reference latent strides when sampling noise |
| Sequence parallelism | Native engine SP1/2/4 correctness verified on five-frame L20 cases |
| VAE placement | Replicated by default; optional height sharding on the window-SP group |
| Unsupported | VFR, multichannel audio, 7B, other sampling schedules, quantization, cache acceleration, VAE width sharding / batch slicing, CPU offload, CFG/TP/PP parallelism, compiled execution, LoRA |

Unsupported engine modes are rejected before process hooks and worker creation.
`ulysses_degree` selects the model-owned SP group. This branch assigns whole
windows to ranks; the specialized head-sharded path is a dependent change.
Ring and AllGather-KV are unsupported.

The practical P0 reference's five-frame batching, overlap, and CPU swapping are
separate execution semantics. Temporal tiling and VAE patch
parallelism reduce activation peaks but do not bound the whole-clip DiT memory.
Large output frames may still exceed device capacity.

Invalid inputs return 400. A single-GPU model OOM fails that request; a
subsequent short request succeeds. With VAE patch parallelism, restart the
service after an OOM: another rank may still be waiting in a collective. Cancelling an in-progress video job removes it through the
existing DELETE endpoint; in-flight GPU work may drain before memory is reusable.
A lost SP worker fails the request and makes health return 503. Restart the
service to restore availability; automatic rank recovery is not provided.

## Reproducible deployment

See the [RTX 5090 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/ByteDance/SeedVR2-RTX-5090.md) for the
input/output contract, complete serving command, and media checks. The local
checkpoint test in `tests/diffusion/models/seedvr2/test_seedvr2_e2e.py` checks
3B transformer SP parity. The PR test result covers complete HTTP restoration.
