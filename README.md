# vllm-omni

## Installation

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
uv pip install vllm==0.17.0 --torch-backend=auto
# Install Open-Sora in editable mode (https://github.com/reka-ai/moonvalley_ai/tree/main/open_sora)
# I ported this over from the original repo to avoid the dependency issues.
uv pip install -e ../moonvalley_ai/open_sora --no-deps
```

## Text to Video Usage with vllm-omni

You can launch the server through `vllm-omni serve`, you can find an example in [examples/online_serving/marey/run_server.sh](examples/online_serving/marey/run_server.sh)

```bash
# Launch the server on a GPU node
HF_HOME=/mnt/localdisk/vllm_omni_hf_cache/ \
VLLM_OMNI_STORAGE_PATH=/mnt/localdisk/vllm_omni_storage \
MODEL=/app/hf_checkpoints/marey-distilled-0100/ \
MOONVALLEY_AI_PATH=${PATH_TO_MOONVALLEY_AI} \
bash examples/online_serving/marey/run_server.sh

# On the same node
SEED=0 examples/online_serving/marey/run_curl_text_to_video.sh
```

*NOTE*: For some reason simultaneous loading of HF models from multiple ranks can cause issues when loading from the shared filesystem, if you run into errors like  `OSError: google/ul2 does not appear to have a file named pytorch_model-00001-of-00004.bin.`, set `HF_HOME` to a node-local path as in the example above.

## Marey 7B (Flux-7B) Serving

The pipeline also supports the smaller Marey 7B model (Flux-7B architecture). Key differences from the 30B (Flux-30B-control-v2):


|                       | 30B                                 | 7B                        |
| --------------------- | ----------------------------------- | ------------------------- |
| Architecture          | Flux-30B-control-v2                 | Flux-7B                   |
| Parameters            | ~30B                                | ~7.6B                     |
| hidden_size           | 5120                                | 3072                      |
| depth (double blocks) | 42                                  | 28                        |
| depth_single_blocks   | 28                                  | 0                         |
| num_heads             | 40                                  | 24                        |
| Text encoders         | UL2 + CLIP (vector) + ByT5 (quotes) | UL2 + MetaCLIP (sequence) |
| Vector conditioning   | CLIP pooled output                  | None                      |
| Control inputs        | Yes (skip_control_inputs=6)         | No                        |
| FLOW_SHIFT            | 3.0 (default)                       | 0.0 (required)            |


### Checkpoint preparation

A raw training checkpoint needs the following before it can be served:

1. `**model_index.json`** — required for model type detection:
  ```bash
   echo '{"_class_name": "MareyPipeline", "_diffusers_version": "0.29.0"}' > ${CHECKPOINT}/model_index.json
  ```
2. `**transformer/` scaffolding** — the weight loader expects `transformer/dummy.safetensors`:
  ```bash
   mkdir -p ${CHECKPOINT}/transformer
   echo '{"_class_name": "MareyTransformer"}' > ${CHECKPOINT}/transformer/config.json
   python3 -c "import torch, safetensors.torch; safetensors.torch.save_file({'__dummy__': torch.zeros(1)}, '${CHECKPOINT}/transformer/dummy.safetensors')"
  ```
3. `**config.yaml` edits** — add explicit architecture params and fix the VAE path:
  ```yaml
   model:
     depth: 28
     depth_single_blocks: 0
     num_heads: 24
     hidden_size: 3072

   text_encoder:
     byt5_max_length: 77   # must match metaclip_max_length

   vae:
     cp_path: vae.ckpt     # relative to checkpoint dir
  ```
4. `**vae.ckpt**` — place the VAE checkpoint in the same directory.

### Launch

```bash
# Launch the 7B server (8 GPUs, ULYSSES_DEGREE=8)
HF_HOME=/mnt/localdisk/vllm_omni_hf_cache/ \
VLLM_OMNI_STORAGE_PATH=/mnt/localdisk/vllm_omni_storage \
MODEL=/home/claudio/marey_checkpoints/epoch5-global_step70000 \
MOONVALLEY_AI_PATH=${PATH_TO_MOONVALLEY_AI} \
FLOW_SHIFT=0 \
bash examples/online_serving/marey/run_server.sh

# Test on the same node
SEED=0 bash examples/online_serving/marey/run_curl_text_to_video_7b.sh
```

## Marey Inference Usage

Matching test command in moonvalley_ai/inference-service/marey_inference.py
*NOTE*: Due to a bug/quirk in the way the cli params are implemented note that many flags actually disable the corresponding variable, eg --add-quality-guidance actually sets this variable to false. After talking to Igor and Adithya it seems this CLI isn’t the preferred way to launch commands on the moonvalley side so this did not interfere with their workloads. This command sets the recommended setup for marey inference that the vllm-omni implementation was designed against.

```bash
PYTHONPATH=${PATH_TO_MOONVALLEY_AI} \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run --project ${PATH_TO_MOONVALLEY_AI}/inference-service \
torchrun --nproc_per_node=8 ${PATH_TO_MOONVALLEY_AI}/inference-service/marey_inference.py infer \
  --num-seq-parallel-splits 8 \
  --offload-diffusion \
  --offload-vae \
  --offload-text-encoder \
  --model-folder "/app/wlam/models/checkpoints/marey/distilled-0001" \
  --checkpoint-folder "/app/wlam/models/checkpoints/marey/distilled-0001/epoch0-global_step5000_distilled" \
  --watermarker-path "/app/wlam/models/checkpoints/marey/videoseal/y_256b_img.jit" \
  --height 1080 \
  --width 1920 \
  --num-frames 128 \
  --fps 24 \
  --steps 33 \
  --guidance-scale 3.5 \
  --disable-caching \
  --use-negative-prompts \
  --negative-prompt "<synthetic> <scene cut> gopro, bright, contrast, static, overexposed, bright, vignette, artifacts, still, noise, texture, scanlines, videogame, 360 camera, VR, transition, flare, saturation, distorted, warped, wide angle, contrast, saturated, vibrant, glowing, cross dissolve, texture, videogame, saturation, cheesy, ugly hands, mutated hands, mutant, disfigured, extra fingers, blown out, horrible, blurry, worst quality, bad, transition, dissolve, cross-dissolve, melt, fade in, fade out, wobbly, weird, low quality, plastic, stock footage, video camera, boring, static" \
  --use-timestep-transform \
  --use-distilled-steps \
  --shift-value 3.0 \
  --use-guidance-schedule \
  --add-quality-guidance \
  --clip-value 10.0 \
  --seed 42 \
  --warmup-steps 4 \
  --cooldown-steps 18 \
  --output /home/aormazabal/wlam/wlam-inference//vllm-omni/marey_inference_test_output_$(date +%Y%m%d_%H%M%S).mp4 \
  "Detailed Description: A majestic, aged eagle with mottled golden-brown feathers soars gracefully through a vast, ancient indoor chamber. Its expansive wings barely flap, catching the air as it glides effortlessly between towering stone pillars adorned with glinting metallic accents. Beams of morning light pierce the gloom, filtering through a cracked skylight high above and illuminating swirling dust motes in their path. The camera pans smoothly, following the eagle's silent flight as it navigates the cavernous space, its sharp eyes scanning the stone floor below, creating a scene of serene power and timeless solitude. Background: The far reaches of the chamber fade into deep shadow, with the silhouettes of distant pillars barely visible. High above, a cracked skylight serves as the primary light source, its fractured glass creating distinct rays of light. Middleground: The aged eagle glides on a steady path, its mottled golden-brown wings spread wide. It passes through the dramatic beams of light, which highlight the intricate details of its feathers and the dust particles dancing in the air. Foreground: The camera looks up from a low angle, tracking the eagle's movement across the expansive stone floor, which is patterned with the bright shafts of light and deep shadows cast by the pillars."
```
