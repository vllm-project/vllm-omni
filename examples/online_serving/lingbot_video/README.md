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

## Prompt rewriter

LingBot-Video is trained on structured JSON captions. The optional rewriter
accepts a plain prompt and runs the official two-turn flow before text
encoding: the base VLM first expands the prompt, then the rewriter LoRA maps
the expansion to JSON. Prompts that already start with a JSON object pass
through unchanged. The rewriter is disabled by default.

The recommended deployment keeps the 27B rewriter outside the diffusion
worker. Serve the base and adapter-applied mapping models behind
OpenAI-compatible chat endpoints, then configure the diffusion stage:

```bash
MODEL=robbyant/lingbot-video-dense-1.3b bash run_server.sh --stage-overrides '{"0":{"model_config":{
    "rewriter_url":"http://127.0.0.1:30000",
    "rewriter_map_url":"http://127.0.0.1:30001",
    "rewriter_expand_model":"Qwen/Qwen3.6-27B",
    "rewriter_map_model":"lingbot-video-rewriter"
  }}}'
```

`rewriter_map_url` may be omitted when one endpoint serves both model names.
The mapping model must have
`robbyant/lingbot-video-rewriter-lora` applied or merged. The request fails
if the mapping turn does not produce parseable JSON; it does not silently run
the diffusion model with the out-of-distribution plain prompt.

For offline or tightly coupled deployments, the worker can lazily load the
base model and PEFT adapter on the first plain-text request:

```bash
MODEL=robbyant/lingbot-video-dense-1.3b bash run_server.sh --stage-overrides '{"0":{"model_config":{
    "rewriter_model_path":"Qwen/Qwen3.6-27B",
    "rewriter_adapter_path":"robbyant/lingbot-video-rewriter-lora",
    "rewriter_device_map":"auto"
  }}}'
```

This path requires `peft` and enough memory for the rewriter in addition to
LingBot. Configure only one of `rewriter_url` and `rewriter_model_path`.
In distributed execution, rank 0 runs the external or in-process rewriter and
broadcasts the resulting caption to the other ranks.

Set `"rewriter_auto_negative":true` in the same `model_config` to run one
additional base-model turn that removes negative-prompt terms which conflict
with the generated caption. This only edits LingBot's categorized JSON
negative prompt: free-text or malformed negative prompts pass through
unchanged, and the model cannot add or reorder terms.

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
