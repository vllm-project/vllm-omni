# Ming-flash-omni 2.0: Offline inference

[Ming-flash-omni-2.0](https://github.com/inclusionAI/Ming) is an omni-modal model supporting text, image, video, and audio understanding, with text and speech outputs.

vLLM-Omni supports three deployment modes:

| Mode | Deploy config | Output |
|------|--------------|--------|
| Thinker + Talker (omni-speech, default) | `vllm_omni/deploy/ming_flash_omni.yaml` | Text + Audio |
| Thinker only (multimodal understanding) | `vllm_omni/deploy/ming_flash_omni_thinker_only.yaml` | Text |
| Thinker + Imagegen (text-to-image / img2img) | `vllm_omni/deploy/ming_flash_omni_image.yaml` | Image (online-serving only at the moment) |

For standalone TTS (talker only), see the [Ming-flash-omni-TTS section in the Text-To-Speech hub](../text_to_speech/README.md#ming-flash-omni-tts).

## Setup

Please refer to the [stage configuration documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/stage_configs/) to configure memory allocation appropriately for your hardware setup.

When no `--deploy-config` is passed, the model registry auto-loads the full thinker+talker `vllm_omni/deploy/ming_flash_omni.yaml` (See [Omni-Speech](#omni-speech-thinker--talker)).

For text-only output without spinning up the talker, pass:

```bash
--deploy-config vllm_omni/deploy/ming_flash_omni_thinker_only.yaml
```

## Run examples

The end-to-end script defaults to built-in assets; pass `--image-path`, `--audio-path`, or `--video-path` to override.

### Multi-Modality Understanding (Standalone Thinker)

Here we pass thinker-only deploy yaml:

```bash
python examples/offline_inference/ming_flash_omni/end2end.py --deploy-config vllm_omni/deploy/ming_flash_omni_thinker_only.yaml --query-type text
python examples/offline_inference/ming_flash_omni/end2end.py --deploy-config vllm_omni/deploy/ming_flash_omni_thinker_only.yaml --query-type use_image
python examples/offline_inference/ming_flash_omni/end2end.py --deploy-config vllm_omni/deploy/ming_flash_omni_thinker_only.yaml --query-type use_audio
python examples/offline_inference/ming_flash_omni/end2end.py --deploy-config vllm_omni/deploy/ming_flash_omni_thinker_only.yaml --query-type use_video --num-frames 16
```

### Reasoning (Thinking Mode)

Reasoning ("detailed thinking on") is applied by the script when
`--query-type reasoning` is set. The default prompt matches Ming's cookbook
and expects the reference figure from the upstream repo — see
`get_reasoning_query` in `end2end.py`.

```bash
python examples/offline_inference/ming_flash_omni/end2end.py \
    --deploy-config vllm_omni/deploy/ming_flash_omni_thinker_only.yaml \
    --query-type reasoning \
    --image-path ./3_0.png
```

### Omni-Speech (Thinker + Talker) {#omni-speech-thinker--talker}

The default deploy YAML already runs thinker+talker, so spoken output only requires requesting `audio` (or `text,audio`) modalities.
The thinker processes your multimodal input, generates text, then the talker synthesises the response as speech.

**Audio-only output** (speech response, no text):
```bash
python examples/offline_inference/ming_flash_omni/end2end.py \
    --query-type text \
    --modalities audio \
    --output-dir output_ming_omni_speech
```

**Both text and audio output**:
```bash
python examples/offline_inference/ming_flash_omni/end2end.py \
    --query-type use_audio \
    --modalities text,audio \
    --output-dir output_ming_omni_speech
```

Generated `.wav` files are saved to `--output-dir` (default `output_ming`), one per request.

The default deploy YAML allocates thinker on GPUs 0–3 and talker on GPU 3 for a common device topology (4 rather than 5 devices that talker on its own device). Adjust `devices` in a copied YAML and pass it via `--deploy-config` to match your hardware or requirements.

### Modality control

| `--modalities` | Thinker output | Talker | Saved files |
|---------------|----------------|--------|-------------|
| `text` (default) | Text | Not run | `<id>.txt` |
| `audio` | Text (internal) | Runs | `<id>.wav` |
| `text,audio` | Text | Runs | `<id>.txt` + `<id>.wav` |

Pass `--deploy-config /path/to/your_deploy.yaml` to any of the commands above to override the bundled deploy config.

### Image generation (text-to-image / img2img)

Image generation is served through the standard task examples.
The diffusion-stage knobs are declared centrally in `vllm_omni/model_extras/ming_flash_omni.py` and routed via `--extra-body`. With deploy yaml assigned properly, the model can run through the shared example scripts like any other diffusion model:

Text-to-image (offline):

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
    --model Jonathan1909/Ming-flash-omni-2.0 \
    --deploy-config vllm_omni/deploy/ming_flash_omni_image.yaml \
    --prompt "Please draw a cute cat." \
    --height 1024 \
    --width 1024 \
    --extra-body '{"steps": 30, "cfg": 2.0, "seed": 42}' \
    --output ming_flash_omni_t2i.png
```

Image-to-image (offline):
```bash
# Reference image: figures/cases/person_gen_05.png from the upstream Ming repo
# https://github.com/inclusionAI/Ming/blob/3954fcb880ff5e61ff128bcf7f1ec344d46a6fe3/examples/vllm_demo.py
wget https://raw.githubusercontent.com/inclusionAI/Ming/3954fcb880ff5e61ff128bcf7f1ec344d46a6fe3/figures/cases/person_gen_05.png

python examples/offline_inference/image_to_image/image_edit.py \
    --model Jonathan1909/Ming-flash-omni-2.0 \
    --deploy-config vllm_omni/deploy/ming_flash_omni_image.yaml \
    --image person_gen_05.png \
    --prompt "Put a pair of sunglasses on the person." \
    --extra-args '{"steps": 30, "cfg": 2.0, "seed": 42}' \
    --output ming_flash_omni_i2i.png
```

For the online path and the full knob list
(`steps`/`cfg`/`seed`/`byte5_text`/`negative_prompt`/`height`/`width`), see the
[image-generation section in the recipe](../../../recipes/inclusionAI/Ming-flash-omni-2.0.md#image-generation-text-to-image--img2img).

## Online serving

For online serving via the OpenAI-compatible API, see [examples/online_serving/ming_flash_omni/README.md](../../online_serving/ming_flash_omni/README.md).
