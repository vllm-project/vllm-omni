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

The default deploy config is the single-stage `lance.yaml`; for `video_edit`
or `text2video` workloads, point ``DEPLOY_CONFIG`` at a video-checkpoint
config (or pass ``--model bytedance-research/Lance/Lance_3B_Video`` to
`run_server.sh`).

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
