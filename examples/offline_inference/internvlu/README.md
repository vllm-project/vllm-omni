# InternVL-U

## Architecture

InternVL-U is a unified understanding-and-generation model served as a native
two-stage vLLM-Omni pipeline:

| Stage | Component | Role |
| :---- | :-------- | :--- |
| 0 | InternVision + Qwen3 VLM (`InternVLUChatModel`) | Understanding, chat, and generation conditioning (three CFG branches per image request) |
| 1 | MMDiT decoder + Qwen-Image VAE (`InternVLUPipeline`) | Dual-CFG denoising and image decoding |

| Feature | Description |
| :------ | :---------- |
| **Modalities** | text2img, img2img (editing), img2text (understanding), text2text |
| **CFG** | Dual CFG (conditional / partial / unconditional) with high-timestep delta normalization |
| **Think mode (text-then-image)** | Optional deployment (`internvlu_chat_think.yaml`) where the VLM writes a detailed description before generating the image |
| **Deployment** | Both stages share one GPU by default (`vllm_omni/deploy/internvlu_chat.yaml`) |

## Quick Start

```bash
cd examples/offline_inference/internvlu

# Text-to-image
python end2end.py --prompt "A cute cat"

# Image editing
python end2end.py --prompt "Add a red knitted scarf around the cat's neck" \
                  --image cat.png

# Think mode (text-then-image): deploys internvlu_chat_think.yaml
python end2end.py --prompt "A cute cat" --think

# Image understanding
python end2end.py --modality img2text \
                  --prompt "Describe this image in detail" --image photo.jpg

# Text chat
python end2end.py --modality text2text \
                  --prompt "What is the capital of France?"
```

> **Note**: The default deployment fits on a single 80GB GPU (A100/H100).

## Think Mode (Text-then-Image)

Think mode is configured **statically per deployment** (like BAGEL's think
mode), not per request. `--think` selects
`vllm_omni/deploy/internvlu_chat_think.yaml`, whose single
`mm_processor_kwargs.think` knob:

- formats generation prompts without a trailing `<img>` so the VLM first
  decodes a chain-of-thought description, and
- lets the model-local sampler enter image generation through an actual
  `<img>` token once the CoT finishes.

The deployment also budgets `max_tokens` for the reference CoT length (200)
plus two internal tokens, and Stage 1 automatically switches to the reference
CoT guidance defaults (3.5/1.5 instead of 4.5/2.0). The generated CoT text is
printed by `end2end.py` and surfaced as `cot_output` in online serving
responses.

## Parameter Reference

| Parameter | Default | Description |
| :-------- | :------ | :---------- |
| `--model` | `InternVL-U/InternVL-U` | HuggingFace model ID or local path |
| `--modality` | `auto` | `auto`, `text2img`, `img2img`, `img2text`, `text2text` |
| `--prompt` | (sample) | Prompt / editing instruction / question |
| `--image` | None | Reference image path(s) |
| `--height` / `--width` | 1024 | Generated image size (stride-16 aligned internally) |
| `--seed` | 42 | Random seed |
| `--num-steps` | 20 | Denoising steps |
| `--think` | off | Text-then-image deployment |
| `--max-tokens` | 512 | Max tokens for text outputs |

Model-specific knobs (`all_cfg_scale`, `part_cfg_scale`, `timestep_trunc`,
`flow_shift`) keep their checkpoint defaults; online they are declared
extra-body params (see the [online serving example](../../online_serving/internvlu/README.md)).

## Limitations

- One output image per prompt (`n=1`).
- Negative prompts are not supported (InternVL-U uses fixed partial and
  unconditional CFG prompts).
- Video understanding is not yet ported.
- In think mode the CoT is capped at the reference's 200 tokens: the
  sampler forces the `<img>` transition at the cap, and the think
  deployment budgets `max_tokens: 202` to fit it plus the two internal
  tokens.  A per-request `max_tokens` override too small for its CoT
  fails with a clear stage-bridge error instead of truncating.

## Reproducing the E2E Test

The weekly pixel-parity test serves the model and compares against goldens
generated with the official repository:

```bash
pytest -s -v tests/e2e/accuracy/test_internvlu.py
```

## Online Serving

For OpenAI-compatible API serving, see
[`examples/online_serving/internvlu/`](../../online_serving/internvlu/README.md).
