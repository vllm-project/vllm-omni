# MiniMax-H3

This example runs MiniMax-H3 offline: joint video + audio generation with
text, first-frame, image/audio, or multi-video conditions.

One Omni instance loads one checkpoint partition. Point `--model` at the
`FL2VA` directory for `t2va`/`fl2va`, or at the `Ref2VA` directory for
`ref2va`.

## Tasks

| `--task` | Conditions | Checkpoint partition | Output |
| :--- | :--- | :--- | :--- |
| `t2va` | Text only | `FL2VA` | Video + audio |
| `fl2va` | First-frame image + text | `FL2VA` | Video + audio |
| `ref2va` | One image + one audio, or one or more video references | `Ref2VA` | Video + audio |

`--task` is optional. When omitted it is auto-resolved from the checkpoint
partition and the provided conditions (image -> `fl2va`, ref2va partition ->
`ref2va`, otherwise `t2va`).

## Run Examples

Text to video + audio:

```bash
python examples/offline_inference/minimax_h3/end2end.py \
  --model /path/to/MiniMax-H3/FL2VA --task t2va \
  --prompts "A quiet cinematic night scene with matching ambient sound."
```

First-frame image to video + audio:

```bash
python examples/offline_inference/minimax_h3/end2end.py \
  --model /path/to/MiniMax-H3/FL2VA --task fl2va \
  --image-path first_frame.png \
  --prompts "The car drives away."
```

Image + audio reference to video + audio:

```bash
python examples/offline_inference/minimax_h3/end2end.py \
  --model /path/to/MiniMax-H3/Ref2VA --task ref2va \
  --image-path ref.png --audio-path ref.mp3 \
  --prompts "The cat lip-syncs."
```

Video reference(s) to video + audio (comma-separated for multiple videos; the
reference soundtracks are used and `--audio-path` is not accepted):

```bash
python examples/offline_inference/minimax_h3/end2end.py \
  --model /path/to/MiniMax-H3/Ref2VA --task ref2va \
  --video-path subject.mp4,background.mov \
  --prompts "Replace the background."
```

## Validated 8x Ascend NPU Configuration

Validated on 8 x Ascend910 (Atlas 800I A3) with the T2VA task at 768x1344,
duration 8.7s, using the BF16 dense configuration:

```bash
python examples/offline_inference/minimax_h3/end2end.py \
  --model /path/to/MiniMax-H3/FL2VA --task t2va \
  --usp 8 \
  --height 768 --width 1344 --duration 8.7 --seed 1101 \
  --text-encoder-tp-size 8 \
  --vae-patch-parallel-size 8 --vae-parallel-mode tile --vae-use-tiling \
  --enable-layerwise-offload \
  --diffusion-attention-backend FLASH_ATTN \
  --num-warmup 3 \
  --prompts "In a snowy blue-purple forest, Ori carefully walks past a sleeping giant..."
```

On NPU, use `--num-warmup 3` before collecting latency or profiler data: the
first generations pay kernel JIT/autotune, HCCL lazy-init, and offload
pipeline setup costs (~3.5x slower than steady state), and the engine prints
`Skipping dummy warmup run`, i.e. there is no built-in warmup.

## Key Arguments

| Argument | Description |
| :--- | :--- |
| `--model` | Path to one checkpoint partition directory (`FL2VA` for t2va/fl2va, `Ref2VA` for ref2va). Required. |
| `--task` | One of `t2va`, `fl2va`, `ref2va`. Default: auto-resolved from the partition and conditions. |
| `--image-path` | Condition image. First frame for fl2va; reference image for ref2va (combined with `--audio-path`). |
| `--audio-path` | Reference audio (wav/mp3/m4a) for image+audio ref2va. |
| `--video-path` | Reference video path(s) for video ref2va, comma-separated. Not combinable with `--image-path`/`--audio-path`. |
| `--height`, `--width` | Output video size (multiples of 32). |
| `--duration` | Output duration in seconds (decimal allowed). Overrides `--num-frames`. |
| `--num-frames` | Output frame count. Default: 209 for t2va/fl2va, 124 for ref2va. |
| `--steps` | Number of diffusion inference steps (default 50). |
| `--num-warmup` | Warmup generations before the measured/profiled run; outputs discarded, profiler starts after warmup. |
| `--seed` | Random seed. |
| `--flow-shift`, `--audio-flow-shift` | Video/audio sigma shifts (checkpoint defaults: 12 and 3). |
| `--usp`, `--ring` | Ulysses / ring sequence-parallel degrees. |
| `--text-encoder-tp-size` | Shard the Qwen3-VL text encoder across the first N DiT ranks (must divide 64 attention heads and 8 KV heads). |
| `--vae-patch-parallel-size`, `--vae-parallel-mode` | VAE patch parallelism. H3 supports the native `tile` mode only; size must be 1 or the full DiT group size. |
| `--vae-use-tiling` | Enable VAE tiling. |
| `--enable-cpu-offload` | Model-level CPU offload (single-device memory-first configuration). |
| `--enable-layerwise-offload` | Blockwise DiT offload to host memory. |
| `--enforce-eager` | Disable torch.compile. |
| `--diffusion-attention-backend` | Diffusion attention backend, e.g. `FLASH_ATTN` or `RAINFUSION_ATTN`. |
| `--profiler-config` | JSON object forwarded to `Omni` profiler_config; see the Profiling section. |
| `--output` | Output directory for the generated MP4 files (default `.`). |

## Notes

- Output is saved as `minimax_h3_<task>_<index>.mp4` with muxed video + audio
  (24 fps, 32 kHz audio).
- H3 is CFG-distilled, so CFG parallelism must remain 1.
- Ulysses carries all sequence parallelism on the validated NPU configuration
  (`--usp 8 --ring 1`): RainFusion's block-sparse kernel ranks key blocks over
  the whole sequence, so ring parallelism would split away the keys it needs.
- Image+audio ref2va accepts exactly one image and one audio reference; video
  ref2va accepts one or more videos but no standalone audio reference.
- Online serving deployment, RainFusion attention, and INT8 quantization are
  documented in the recipes: [recipes/MiniMaxAI/MiniMax-H3-NPU.md](../../../recipes/MiniMaxAI/MiniMax-H3-NPU.md)
  and [recipes/MiniMaxAI/MiniMax-H3.md](../../../recipes/MiniMaxAI/MiniMax-H3.md).
- Known harmless warnings on exit: `corrupted size vs. prev_size`,
  `resource_tracker leaked shared_memory`, and
  `Terminating diffusion worker ... after timeout` all occur after results
  are written and do not affect the outputs.
