# Cosmos3-Nano-Sim-Transfer

Use Sim-Transfer to generate a video conditioned on edge, blur, depth, or
segmentation control. Each request generates one complete clip offline; tick
and streaming transport are not supported.

## Artifact contract

Local Transfer checkpoints use the `Cosmos3NanoSimTransferPipeline` model
class and a `cosmos3_nano_sim_bimanual` manifest with
`conditioning.mode="control_video"` in `transformer/config.json`.
The CLI defaults to the
[dense deployment](../../vllm_omni/deploy/cosmos3_nano_sim_transfer.yaml).

## Input JSON

The Transfer adapter accepts `prompt` or `prompt_path`, optional `vision_path`,
`num_frames`, and exactly one of `edge`,
`blur`, `depth`, or `seg`. Edge and blur can be computed from `vision_path`;
depth and segmentation records normally provide `control_path` inside the
selected hint object. Relative caption and control paths are resolved against
the input JSON directory. Structured JSON captions are passed to the pipeline
as JSON. When `vision_path` is set, it must provide at least `num_frames`
frames.

The CLI uses the input vision clip to choose the aspect ratio when present.
Otherwise it uses the selected hint's `control` or `control_path`. Direct
pipeline requests can instead pass the control clip as `control_video`, with
either a named hint or `control_hint`; it cannot be combined with a hint's
`control` or `control_path`. Transfer snaps the source aspect ratio
to the requested bucket family unless `aspect_ratio` is supplied. CLI resolution
takes precedence over the JSON record, with 480 as the fallback. The resulting
dimensions are validated by the shared Cosmos3-Nano-Sim-Bimanual policy.

## Offline inference

For the depth example, save a record like this to `/data/sim_transfer_depth.json`
and place the control clip beside it.

```json
{"prompt":"A robot moves a block.","depth":{"control_path":"depth.mp4"},"num_frames":121}
```

```bash
python -m vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.offline_transfer \
  --model nvidia/Cosmos3-Nano-Sim-Transfer \
  --input-json /data/sim_transfer_depth.json \
  --resolution 480 \
  --num-frames 121 \
  --fps 30 \
  --kv-cache-inference-size 30 \
  --attention-sink-size 3 \
  --no-emphasize-control-in-prompt \
  --seed 42 \
  --output cosmos3_nano_sim_transfer.mp4
```

Window, sink, emphasis, and aspect-ratio CLI options override the same JSON
fields. Omitted window, sink, and emphasis settings come from the checkpoint.
`--prompt-path` overrides the record's caption.

The window counts temporal frames, including sinks and the current frame. Each
retained past frame holds both control and clean latents. Sinks require a
checkpoint chunk size of one. A direct pipeline request with
`kv_cache_inference_size: null` keeps full history and requires
`attention_sink_size=0`.

## Deployment

The default Transfer deployment uses eager execution with dense K/V. A custom
`--deploy-config` can enable regional compilation with `enforce_eager: false`
and `diffusion_compile_granularity: regional`. With compilation enabled, add
`model_config.use_cuda_graphs: true` to use native blockwise CUDA graphs.

Decode overlap is enabled by default for full-video CUDA requests. Set
`model_config.overlap_vae_decode: false` for serial decode. Control hints are
VAE-encoded up front, then consumed in chunks sized by the checkpoint
configuration.

## Verification

Check that the output opens and contains 121 frames at 30 FPS for the example
above. Review the generated video against the supplied depth sequence. To
compare runs, use the same checkpoint, input, seed, and request flags, save
pre-VAE outputs with `--output-type latent`, and compare the tensors before
comparing decoded videos.
