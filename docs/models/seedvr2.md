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

Window-local RoPE axes use zero-based unit-stride positions. Angle tables are
cached by axis length, device, and dtype, avoiding GPU-to-CPU reads to form a
cache key on repeated requests.

Grouped SDPA row indices are built once per window layout and reused across
layers. This removes per-layer GPU-to-CPU length reads; packed-varlen attention
is a separate opt-in path.

Shared text-prefix attention is opt-in through
`additional_config.seedvr2_shared_text_prefix=true` and requires FlashAttention.
It avoids repeating text K/V per window and caches fixed first-layer text QKV;
later text states remain video-dependent. A stable HTTP latency gain has not
been established.

The multipart API requires a prompt field; a single space supplies a blank
prompt. Nonblank text is rejected. Omit `fps` to retain the source frame rate;
an explicit rate must match the source. Output dimensions are explicit multiples
of 16. Input frames are resized with antialiased bicubic interpolation. The final
frame is repeated internally to reach 4n+1, and the decoded output is cropped back
to the original frame count. Five frames remain five; six frames are internally
padded to nine and return six.

## Request admission

The default 3B limit is 848×480 pixels per input or output frame and 2,035,200
pixels across the requested output clip. With four-rank window SP,
`vae_patch_parallel_size=4`, and VAE tiling, the per-frame limit rises to
2560×1472 and the clip budget to 18,841,600 pixels. Both budgets count temporal
padding to 4n+1 frames: the default admits up to five 848×480 frames or 93
192×112 frames; the SP4 profile admits up to 45 848×480 frames or five
2560×1472 frames. A 16-frame 1280×720 or 720×1280 clip pads to 17 frames and
uses 15,667,200 of the SP4 clip's 18,841,600 pixels. Both orientations passed
original-size HTTP restoration on four RTX 5090 GPUs with VAE tiling and height
sharding. An independent 257-frame cap bounds per-frame decoder work for tiny
inputs. The other longer combinations above are admission bounds, not completed
GPU validation. The decoder checks
declared duration, frame count, and input dimensions when available, then
enforces the limits as frames arrive. Requests outside these budgets return 400
before building the resized whole-clip tensor.

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
| Reference semantics | C0 whole clip, no color correction |
| Video input | One uploaded file, or offline TCHW RGB floats / PIL frames |
| Timing | Constant frame rate, increasing PTS; normalize the video origin to zero |
| Audio | First mono/stereo track, aligned by source PTS, cropped to the video interval, re-encoded as AAC |
| Randomness | Per-request generator; preserve reference latent strides when sampling noise |
| Sequence parallelism | SP1 whole-window path; SP2/4 head-sharded Ulysses window attention |
| VAE placement | Replicated by default; optional height sharding on the window-SP group |
| Unsupported | VFR, multichannel audio, 7B, other sampling schedules, quantization, VAE width sharding / batch slicing, CPU offload, CFG/TP/PP parallelism, compiled execution, LoRA |

Unsupported engine modes are rejected before process hooks and worker creation.
`ulysses_degree` selects the model-owned SP group. At SP>1, the DiT keeps MLP
rows sharded by sequence and exchanges only attention QKV into head shards.
Every head shard sees the complete regular or shifted window layout. Ring and
AllGather-KV are unsupported.

The practical P0 reference's five-frame batching, overlap, LAB correction, and
CPU swapping are separate execution semantics. Temporal tiling and VAE patch
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
