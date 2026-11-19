# MiniMax-H3-Image on one RTX 4090

## Validated scope

Text-to-image generation was reported successful on one NVIDIA GeForce RTX
4090 (24 GB), using GPUStack and layerwise offload of both the DiT and text
encoder. Requests for `1024x1024` and `6272x2688` (21:9) completed successfully.
A subsequent `1024x1024` image-editing request with two reference images also
returned HTTP 200 and a decoded image of the requested dimensions.
This is a community deployment report, not a benchmark or an image-quality
comparison. Concurrent requests were not tested.

| Item | Reported environment |
| --- | --- |
| GPU | 1x NVIDIA GeForce RTX 4090, 24 GB |
| vLLM | 0.33.0 |
| vLLM-Omni | 0.1.dev4171+gc958c8f5f, as printed in the startup log |
| Container | `docker.desaysv.com/vllm/vllm-omni:minimax-h3-image` (private registry) |
| Model | Local ModelScope download of `MiniMax/MiniMax-H3-Image` |
| Offload | Layerwise, both `dit` and `text_encoder` |

The contributor confirmed that the tested private-registry image was
`vllm/vllm-omni:minimax-h3-image`, downloaded, retagged and pushed without
modification. The public image is therefore covered by this DLO deployment
report. The image digest was not recorded. Driver, PyTorch, host RAM capacity,
peak VRAM and peak host memory
were not recorded for this run. Plan sufficient host memory for offloading:
the existing DLO profile keeps about 124 GiB of DiT and text-encoder weights
in pinned host memory, in addition to runtime overhead. That is a weight-size
estimate, not a measured RAM requirement for this RTX 4090 deployment.

## Prepare the downloaded model

Use a complete local download, including `text_encoder/config.json` and all
text-encoder weight shards. The reported deployment used these paths:

| Environment | Model directory |
| --- | --- |
| Host | `/data/cache/model_scope/MiniMax/MiniMax-H3-Image` |
| Inference container | `/var/lib/gpustack/cache/model_scope/MiniMax/MiniMax-H3-Image` |

Use the host path when editing files on the host and the mounted container
path in `vllm serve`. Adapt both paths to your deployment.

### Older-image text-encoder configuration workaround

The tested image failed at startup with:

```text
ValueError: model_index.json loads text_encoder from 'MiniMaxAI/MiniMax-H3', but ... also has text_encoder/
```

After downloading the model, back up `model_index.json` and manually remove
the remote-source object from its `text_encoder` array. That object contains
these four fields:

```json
{
  "pretrained_model_name_or_path": "MiniMaxAI/MiniMax-H3",
  "subfolder": "text_encoder",
  "variant": null,
  "revision": null
}
```

Remove the enclosing object and its preceding comma as well: keep only the
first two entries of the `text_encoder` array, unchanged. Do not leave an
empty `{}` as the third entry, and do not delete the local `text_encoder/`
directory. Validate the edited JSON with `python3 -m json.tool model_index.json`
before restarting the deployment.

This workaround applies only when the local encoder is complete and the
image raises the conflict above. Current upstream code prefers a bundled
`text_encoder/` directory automatically, so it does not require this edit.

## Start the server

Run inside the inference container, with the model directory mounted:

```bash
MODEL=/var/lib/gpustack/cache/model_scope/MiniMax/MiniMax-H3-Image

vllm serve "$MODEL" \
  --disable-access-log-for-endpoints /metrics \
  --enable-prompt-tokens-details \
  --num-gpus=1 \
  --usp=1 \
  --text-encoder-tp-size=1 \
  --diffusion-offload-config '{"mode":"layer","components":["dit","text_encoder"]}' \
  --omni \
  --tensor-parallel-size 1 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name h3-image
```

These are the reported GPUStack launch settings, with host and port normalized
for a local example. GPUStack can manage those networking arguments itself.
The two logging arguments are retained from the deployment; the memory-saving
setting is layerwise offload of **both** components.

## Generate a 21:9 image

Send one request at a time. Set `BASE_URL` to your service address when using
a gateway; add your own authorization header if the gateway requires one.

```bash
BASE_URL=http://localhost:8000

curl --fail-with-body --max-time 1800 \
  "$BASE_URL/v1/images/generations" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "h3-image",
    "prompt": "一只橘猫坐在窗边，午后的阳光洒在毛发上，窗外是绿色花园，真实摄影风格，细节丰富，柔和自然光，21:9超宽画幅，宽阔的环境构图",
    "n": 1,
    "size": "6272x2688",
    "response_format": "b64_json"
  }' \
  -o h3-image-response.json

jq -er '.data[0].b64_json' h3-image-response.json > h3-image.b64 && \
  base64 --decode h3-image.b64 > h3-image.png
```

Use `"size": "1024x1024"` for the smaller validated request. H3-Image normally
generates small requests on a canvas of at least 2048² pixels and then resizes
the output; the requested size is not always the internal canvas size.

The observed `1024x1024` curl transfer took approximately 20 seconds through
the deployment's API gateway, including response transfer. This was one
request, with warmup state unspecified, not model-only inference latency.
The `6272x2688` request was confirmed successful without a recorded latency.
No RTX 4090 memory peak or throughput result is claimed.

## References

- [H200 recipe and model contract](./MiniMax-H3-Image-H200.md)
- [Distributed layerwise offloading](../../docs/user_guide/diffusion/offloader/distributed_layerwise_offload.md)
- [Model card](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)
