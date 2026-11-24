# Invoker-Render 1.0

Invoker-Render renders a video from a structural proxy video, a reference image for appearance, and a frozen prompt.
The default configuration uses BF16, TP1, and Ulysses8 on a single eight-GPU node.

## Prepare Assets

Download the DiT configuration, index, and checkpoint shards from the root of the
[model repository](https://huggingface.co/inclusionAI/invoker-render-1.0). Follow the
[CodeWorldModel deployment guide](https://github.com/inclusionAI/CodeWorldModel/blob/86c952f6a37a82afbc391db2572b4d94ce332438/docs/INFERENCE.md#prepare-a-bundle)
to gather the matching H3 encoders, causal decoder, artifact manifest, and silence asset, then run
`invoker_render.py prepare`. Use `--deployment-profile reference`, `--no-fuse-first-denoise`, and
`--dit-norm-quality exact`, and select `omni-h3-cwm-diffusers-execution.json` as the execution config for the
released weights. Keep the asset paths referenced by the bundle accessible.

## Offline Inference

The reference image must be 1344×768. The bundle defines the proxy canvas size. The proxy video must use 24 FPS and
contain `17*N+5` frames, with a minimum of 39. Save `encoders.recipe.system_prompt` from the artifact in `prompt.txt`.

```python
from pathlib import Path

from PIL import Image

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(
    model="/models/invoker-render/bundle",
    deploy_config="vllm_omni/deploy/invoker_render.yaml",
    trust_remote_code=True,
)
try:
    outputs = omni.generate(
        {
            "prompt": Path("prompt.txt").read_text().strip(),
            "modalities": ["video"],
            "multi_modal_data": {"image": Image.open("first.png"), "video": "proxy.mkv"},
        },
        OmniDiffusionSamplingParams(seed=42),
    )
    print(outputs[0].images[0].shape)
finally:
    omni.close()
```

## Serve Video Files

```bash
vllm serve /models/invoker-render/bundle --omni --trust-remote-code \
  --deploy-config vllm_omni/deploy/invoker_render.yaml \
  --served-model-name invoker-render --port 8091
```

```bash
curl --fail-with-body http://127.0.0.1:8091/v1/videos/sync \
  -F 'model=invoker-render' -F 'seed=42' -F 'prompt=<prompt.txt' \
  -F 'input_references=@first.png;type=image/png' \
  -F 'input_references=@proxy.mkv;type=video/x-matroska' \
  -o output.mp4
```

`POST /v1/videos` accepts the same fields. Retrieve the output through the existing video job API.
The proxy video determines the output frame count.

## Stream Live Proxy Frames

Start the server with `vllm_omni/deploy/invoker_render_streaming.yaml` and connect to `WS /v1/realtime/video`.
Send the reference image, frozen prompt, and seed in `session.start`.

- `video.start.proxy` returns the canvas dimensions, `fps`, `packet_frames`, `tail_frames`, and initial `credit_frames`.
- `session.proxy.append` carries `input_seq`, `frame_start`, `num_frames`, `eos`, and a strictly increasing `capture_ms`
  timestamp for each frame. Follow the JSON immediately with an RGB24 binary message containing pixels in `T,H,W,3`
  order. Send at least two regular 17-frame packets before the final five-frame EOS packet.
- `session.proxy.credit` returns `accepted_frames`, `consumed_frames`, `next_input_seq`, and `credit_frames`. The client
  subtracts sent but unacknowledged frames from its credit. Playback time follows source frame indices on the 24 FPS timeline.
- `video.chunk_metadata` reports separate `generation_chunk_index` and `transport_chunk_index` values, along with the
  associated `source_frame_start` and `source_frame_count`. Binary messages carry fMP4 data. The server sends
  `session.done` after it drains the stream at EOS and finishes muxing. `session.stop` cancels the request.

See the [interactive racing example](../../examples/online_serving/invoker_render_game/README.md) for browser integration.
