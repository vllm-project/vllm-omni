# VDN-H3 Hybrid Attention

`VDNH3_ATTN` serves [VDN-H3](https://github.com/OpenVDN/vdn-minimax-h3), a
MiniMax-H3 checkpoint that replaces the self-attention of every DiT block with
a hybrid of two branches:

| Branch | Computed by | What it covers |
| --- | --- | --- |
| Window softmax | `VDNH3_ATTN` | Exact softmax between each video frame and the frames of its neighboring chunks |
| Video DeltaNet | The model | A linear-attention summary of the frames outside that window, seeded with the prompt |

The two branches are projected separately and summed. Text, condition, and
audio rows keep dense attention, so only video-to-video attention outside the
window is summarized rather than computed exactly.

## Supported models

| Model / checkpoint | Required adapter | Tasks | Steps | Parallelism |
| --- | --- | --- | --- | --- |
| `MiniMaxAI/MiniMax-H3` (FL2VA partition) | [`OpenVDN/vdn-minimax-h3`](https://huggingface.co/OpenVDN/vdn-minimax-h3) (`stage-dmd-step-250`) | T2VA, FL2VA | 8 | Tensor parallelism |

## Quick start

Serve the FL2VA partition with the OpenVDN release as the adapter:

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve MiniMaxAI/MiniMax-H3 --omni \
  --trust-remote-code \
  --task-type fl2va \
  --lora-path OpenVDN/vdn-minimax-h3 \
  --diffusion-attention-backend VDNH3_ATTN \
  --tensor-parallel-size 2 \
  --text-encoder-tp-size 2 \
  --vae-patch-parallel-size 2
```

Only the release's `stage-dmd-step-250/` directory (about 5 GB) is downloaded;
the base weights come from `MiniMaxAI/MiniMax-H3`. `--lora-path` also accepts a
local copy of the release or of that directory.

To run the DiT and text-encoder linears in FP8, add `--quantization fp8`; see
[FP8](#fp8) for what it changes.

Requests use the
[MiniMax-H3 HTTP API](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md#http-api-examples)
with 8 inference steps:

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/videos/sync" \
  -F 'prompt=A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes wide with interest.' \
  -F 'width=1344' \
  -F 'height=768' \
  -F 'aspect_ratio=16:9' \
  -F 'fps=24' \
  -F 'num_inference_steps=8' \
  -F 'seed=1000' \
  -F 'extra_params={"task":"t2va","duration":5}' \
  -o vdn_t2va.mp4
```

FL2VA requests with a first frame, a last frame, or both use the same
fields as for base H3; see the
[FL2VA examples](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md#2-fl2va-first-frame-to-video-and-audio).

The checkpoint is a DMD student distilled at 8 steps with H3's default video
and audio flow shifts of 12 and 3 and the Euler sampler. Requests with a
different step count, shift, or sampler are rejected.

## Window layout

The packed DiT sequence is `[text | conditions | audio | video | padding]`.
Within the video segment:

- frame `t` belongs to chunk `t // 5` and attends to every frame of chunks
  `c - 1` through `c + 1`;
- the first and last frames are anchors: they attend to, and are attended by,
  every row;
- text, condition, and audio rows attend to every row;
- padding rows receive zeros.

`VDNH3_ATTN` runs this layout as one FlashAttention call per group of frames that
share a key set, over gathered keys. Attention calls without window metadata,
such as H3's token refiner, run exactly as `FLASH_ATTN`.

The linear branch runs a bidirectional delta-rule recurrence over frames,
starting from a state written by the prompt. Each frame reads the recurrent
state accumulated before and after its window, so every frame still sees the
whole clip.

## Checkpoint loading

At startup the release's two LoRA adapters are fused into the H3 weights as
they stream in, and the linear-branch weights are attached to the DiT blocks.
The log reports the result:

```text
VDN-H3 stage-dmd-step-250: fused 259 LoRA targets, loaded 800 branch tensors
```

Because the adapters are fused into the weights, requests cannot attach
another LoRA. `VDNH3_ATTN` and the VDN checkpoint must be selected together:
another backend would run the hybrid weights as dense attention, so a mismatch
is rejected at startup.

## Performance

torch.compile is on by default. In addition to the DiT blocks, it fuses the
linear branch's per-token work: the short convolutions, activations, and the
readout norm and gate. `--enforce-eager` runs every piece eagerly.

The first request at each new resolution or duration also compiles, because
H3 skips the engine's startup warm-up. Send one request at the serving geometry
before measuring latency.

Set `--vae-patch-parallel-size` to the tensor-parallel size so that every rank
decodes video VAE tiles.

Reference latency on H200 for a 1344x768, 345-frame (about 104k-token) T2VA
request, after warm-up, with VAE patch parallelism:

| GPUs | Configuration | DiT time per step | Request time |
| --- | --- | --- | --- |
| 2 | VDN-H3, BF16 | 8.2 s | 75 s |
| 2 | VDN-H3, FP8 | 7.2 s | 68 s |
| 2 | Dense MiniMax-H3, BF16 | 15.4 s | 132 s |
| 8 | VDN-H3, BF16 | 2.9 s | 29 s |
| 8 | VDN-H3, FP8 | 2.8 s | 28 s |
| 8 | Dense MiniMax-H3, BF16 | 4.4 s | 41 s |

The dense rows run the same 8 steps with `FLASH_ATTN`. Request time covers
text encoding, the 8 denoising steps, VAE decoding, and returning the frames.

### FP8

`--quantization fp8` quantizes the same linears as base MiniMax-H3 (see
[Online FP8 quantization](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md#online-fp8-quantization)),
plus the linear branch's output projection. The branch's narrow gate and decay
projections stay in BF16. FP8 shifts outputs by about as much as it does for
base MiniMax-H3, so compare it against BF16 with the same seed before you
deploy it.

## Verify the window

The first DiT forward after startup logs the window geometry once:

```text
VDNH3_ATTN window: frames=102 tokens/frame=1008 video_start=1170 used=103986 chunk=5 radius=1 anchors=both
```

`frames` is the number of latent frames. `video_start` and `used` give the
video segment's first row and the number of valid rows in the packed sequence.
