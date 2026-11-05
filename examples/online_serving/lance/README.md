# Lance: Online serving

OpenAI-compatible chat completions API for Lance, served by `vllm-omni`.

## Start the server

```bash
bash examples/online_serving/lance/run_server.sh
# or, with overrides
MODEL=bytedance-research/Lance \
DEPLOY_CONFIG=vllm_omni/deploy/lance.yaml \
PORT=8091 \
    bash examples/online_serving/lance/run_server.sh
```

The default deploy config is the single-stage `lance.yaml`. For video
generation, set `MODEL` to the local `Lance_3B_Video` checkpoint directory
inside the downloaded Lance bundle (with its sibling VAE and ViT files):

```bash
MODEL=/path/to/Lance/Lance_3B_Video bash examples/online_serving/lance/run_server.sh

curl --fail-with-body http://127.0.0.1:8091/v1/videos/sync \
  -F 'prompt=a cute cat walking in the park, cinematic' \
  -F 'num_frames=25' -F 'width=768' -F 'height=480' -F 'fps=8' \
  -F 'num_inference_steps=30' -F 'seed=42' \
  -F 'extra_params={"cfg_text_scale":4.0,"timestep_shift":3.5}' \
  -o lance.mp4
```

## Send a request

```bash
# text-to-image
python examples/online_serving/lance/openai_chat_client.py \
    --prompt "A cute corgi astronaut on the moon, cinematic" \
    --modality text2img \
    --output corgi.png

# image edit
python examples/online_serving/lance/openai_chat_client.py \
    --prompt "Convert this into a vibrant cartoon-style illustration" \
    --modality img2img \
    --image-url path/to/photo.png \
    --output edited.png
```

The client is shared with BAGEL — same OpenAI message format, same
``modalities`` and ``num_inference_steps`` / ``seed`` / ``height`` /
``width`` knobs.
