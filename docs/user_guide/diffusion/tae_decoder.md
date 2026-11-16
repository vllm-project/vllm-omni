# MiniMax-H3 Tiny Decoder (TAEH3)

TAEH3 replaces the MiniMax-H3 video decoder with a tiny convolutional
autoencoder: about 10 million parameters, against about 2.4 billion (9 GiB in
FP32) for the original decoder. It is an approximation meant for previews and
for hosts that cannot hold the full decoder. Generated pixels differ from the
original decoder's, and fine detail is visibly softer. The video encoder and
the audio VAE are unchanged.

For a smaller decoder that stays close to the original's quality, use
[LiteVAE](light_vae.md) instead.

## Usage

Add the following to an existing [H3 serving command](gh-file:recipes/MiniMaxAI/MiniMax-H3.md):

```bash
--additional-config '{"tae_decoder_path":"/weights/taeh3.safetensors"}'
```

Download [`taeh3.safetensors`](https://github.com/madebyollin/taehv/blob/main/safetensors/taeh3.safetensors)
from [madebyollin/taehv](https://github.com/madebyollin/taehv) (MIT license).
The file must be available locally to every decoding process; it is not
downloaded automatically. The original native H3 VAE assets and
`--trust-remote-code` are still required: the checkpoint's VAE code is loaded
for its configuration and its encoder.

This is a service-startup setting, not a per-request parameter. To restore the
existing decoder selection, remove the key and restart the service.

## Decoder selection

| Configuration | Behavior |
| --- | --- |
| Key omitted | Preserve existing decoder selection, including bundled NVFP4 auto-discovery. |
| `tae_decoder_path` set | Load TAEH3, skip bundled NVFP4 auto-discovery, and drop the checkpoint's decode half so it is never placed on the device. |
| TAEH3 together with a non-null `video_vae_decoder` or `lightvae_decoder_path` | Reject the conflicting settings before loading. |
| Empty, null, missing, or incompatible checkpoint | Fail loading; do not silently fall back to the original decoder. |

## How it decodes

- The decoder takes the latents the DiT produces. The VAE's latent mean and
  standard deviation are not applied.
- A video is decoded one 17-frame clip (five latent frames) at a time. Only a
  few feature maps are carried from one clip to the next, so device memory
  does not grow with the video length.
- There is no tiling and no cross-device collective. `--vae-use-tiling` and
  `--vae-patch-parallel-size` have no effect on it. In a distributed VAE
  group every rank decodes the full video on its own device; with chunked MP4
  output only the output owner decodes.
- Weights are FP16 on accelerators and FP32 on CPU.

## Supported scope

- The regular H3 pipeline and the separate decoder pipeline accept this
  setting; encode-only stages ignore it.
- Video latents of `5n + 2` latent frames, which is every length the H3 video
  pipeline produces. Single-frame latents are rejected.
- The MiniMax-H3-Image pipeline uses a different VAE and is not covered.

## Performance and quality

No device measurements are published for this decoder yet. Compare it with the
original decoder on identical prompts, seeds, and sampling settings before
relying on it, and review faces, hands, and text in particular.
