# Omni-Diffusion: Offline inference

This example runs all currently supported Omni-Diffusion tasks through the
synchronous `Omni` entrypoint:

| Task | Input | Output |
| --- | --- | --- |
| `t2i` | text | image |
| `vqa` | text + image | text |
| `asr` | audio | text |
| `tts` | text | audio |
| `s2i` | audio | image |
| `svqa` | audio + image | text |

Install the optional dependencies from the repository root:

```bash
uv pip install -e '.[omni-diffusion]'
```

## Component Models

The bundled deploy configs use the official Hugging Face repositories when a
component path is `null`:

| Config key | Default repository |
| --- | --- |
| `image_tokenizer_path` | `showlab/magvitv2` |
| `sensevoice_path` | `FunAudioLLM/SenseVoiceSmall` |
| `flow_path` | `THUDM/glm-4-voice-decoder` |

The first run downloads only the components needed by the selected task and
later runs reuse the Hugging Face cache. Set a key to an existing local model
directory in the deploy config to disable its automatic download.

## Run

```bash
# Use the Hugging Face model ID by default. Set MODEL to an existing local
# model directory to skip downloading the main checkpoint.
MODEL=${MODEL:-lijiang/Omni-Diffusion}

python examples/offline_inference/omni_diffusion/end2end.py \
  --task t2i --model "$MODEL" \
  --deploy-config vllm_omni/deploy/omni_diffusion_t2i.yaml

python examples/offline_inference/omni_diffusion/end2end.py \
  --task vqa --model "$MODEL" \
  --deploy-config vllm_omni/deploy/omni_diffusion_vqa.yaml

python examples/offline_inference/omni_diffusion/end2end.py \
  --task asr --model "$MODEL" \
  --deploy-config vllm_omni/deploy/omni_diffusion_asr.yaml

python examples/offline_inference/omni_diffusion/end2end.py \
  --task tts --model "$MODEL" \
  --deploy-config vllm_omni/deploy/omni_diffusion_tts.yaml

python examples/offline_inference/omni_diffusion/end2end.py \
  --task s2i --model "$MODEL" \
  --deploy-config vllm_omni/deploy/omni_diffusion_s2i.yaml

python examples/offline_inference/omni_diffusion/end2end.py \
  --task svqa --model "$MODEL" \
  --deploy-config vllm_omni/deploy/omni_diffusion_svqa.yaml
```

Like other multimodal examples, the commands use vLLM's cached
`cherry_blossom` image and `mary_had_lamb` audio assets by default. Use
`--image-path`, `--audio-path`, `--prompt`, and `--output` to override the
sample inputs, prompt, and output path.
