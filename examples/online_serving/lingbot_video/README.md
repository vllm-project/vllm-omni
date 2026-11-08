# LingBot-Video

LingBot-Video uses one `LingBotVideoPipeline` for text-to-image (T2I),
text-to-video (T2V), and text-image-to-video (TI2V) generation. Both the dense
and MoE checkpoints use the same request format.

## Start the server

```bash
MODEL=robbyant/lingbot-video-dense-1.3b bash run_server.sh
```

The MoE checkpoint uses the same server and request scripts, but requires
substantially more GPU memory:

```bash
MODEL=robbyant/lingbot-video-moe-30b-a3b bash run_server.sh
```

## Text to image

The image endpoint selects T2I mode and always generates one frame:

```bash
bash run_curl_text_to_image.sh
```

The script sends a `320x192`, two-step smoke request and writes
`lingbot_t2i.png`.

## Text or text-image to video

Run the video script without an image to select T2V mode:

```bash
bash run_curl_text_image_to_video.sh
```

Pass a first-frame image to the same script to select TI2V mode:

```bash
INPUT_IMAGE=/path/to/input.png bash run_curl_text_image_to_video.sh
```

The client scripts omit the optional `model` request field, so they target
whichever dense or MoE checkpoint the server loaded. The video example uses the
lightweight `320x192`, 9-frame, two-step configuration.

Until the shared `/v1/videos` reference-image resizing is removed, TI2V target
dimensions must be sent through `extra_params`, for example
`{"size":"320x192"}`. Do not use the top-level `size`, `width`, or `height`
fields for TI2V because the serving layer currently applies those dimensions
to the reference image before the model receives it. T2V requests can continue
to use the top-level dimension fields.

LingBot video frame counts use the causal VAE `4n+1` grid. The pipeline rounds
any requested frame count upward to the next valid value. An explicit
`num_frames` takes precedence over `seconds`; otherwise, the server first
resolves `seconds * fps` and the pipeline applies the same alignment.

Official `resolution`/`ratio` presets can be sent through `extra_params`, for
example `{"resolution":"720p","ratio":"16:9"}`. The `2k` and `4k` entries
only define output dimensions; whether they run successfully depends on the
checkpoint, GPU memory, and memory optimizations available in the deployment.

For `/v1/images/generations`, the server resolves these aliases to their final
pixel dimensions before applying `--max-generated-image-size`. Requests above
the configured limit return HTTP 400 before engine dispatch. LingBot produces
one output per prompt; image requests with `n>1` are also rejected with HTTP
400.

LingBot TI2V accepts exactly one image reference. Image editing, video
references, audio references, batching, and Refiner execution are not supported
by this pipeline mode.
