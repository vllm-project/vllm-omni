# DeepSeek Janus (text-to-image)

DeepSeek Janus uses autoregressive image-token prediction plus VQ decode
instead of a classical DiT denoising loop. vLLM-Omni supports Janus through an
explicit single-stage deployment topology:

- `deepseek_janus_single_stage`: one diffusion stage runs the full Janus
  image-generation stack.

Pass the deploy config explicitly. Use `--deploy-config` for online serving;
the generic offline `text_to_image.py` entrypoint currently receives the same
YAML through `--stage-configs-path`.

## Dependencies

`addict` and `timm` are Janus-specific dependencies and are not installed from
`requirements/common.txt` by default:

```bash
pip install 'vllm-omni[janus-image]'
```

If installing from source, use `pip install -e '.[janus-image]'`.

## Offline Inference

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model deepseek-ai/Janus-1.3B \
  --stage-configs-path vllm_omni/deploy/deepseek_janus_single_stage.yaml \
  --prompt "A scenic mountain lake at sunset" \
  --output janus_out.png \
  --guidance-scale 5.0 \
  --tensor-parallel-size 1 \
  --height 384 \
  --width 384
```

For Janus-Pro-7B:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model deepseek-ai/Janus-Pro-7B \
  --stage-configs-path vllm_omni/deploy/deepseek_janus_single_stage.yaml \
  --prompt "A scenic mountain lake at sunset, photorealistic" \
  --output janus_pro_7b.png \
  --guidance-scale 5.0 \
  --height 384 \
  --width 384
```

## Online Serving

```bash
vllm serve deepseek-ai/Janus-Pro-7B --omni \
  --deploy-config vllm_omni/deploy/deepseek_janus_single_stage.yaml \
  --port 8091 \
  --tensor-parallel-size 1
```

Use the OpenAI-compatible image API:

```bash
curl -X POST http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A scenic mountain lake at sunset",
    "size": "384x384",
    "guidance_scale": 5.0,
    "seed": 42
  }'
```

## Text Rendering Prompts

These prompts exercise text embedded in the generated image:

```text
A clean white poster on a grass field that clearly reads "vLLM-Omni" in large colorful letters, centered composition
```

```text
A street cafe chalkboard sign that says "HELLO JANUS" in large white block letters, realistic lighting, centered composition
```

```text
A bakery display card with the words "OPEN SOURCE" written in bold icing-style letters, close-up product photo
```

## Notes

- Janus outputs fixed 384 x 384 images through a 24 x 24 VQ latent grid.
- `--num-inference-steps` does not affect Janus because image generation is a
  fixed 576-token AR loop.
- CPU offload and quantization are applicable to the Janus modules. TeaCache,
  Cache-DiT, tensor parallelism, CFG parallelism, VAE patch parallelism, and
  diffusion step execution are not wired for this single-stage implementation.
