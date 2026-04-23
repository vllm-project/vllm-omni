# Ming-flash-omni 2.0

[Ming-flash-omni-2.0](https://github.com/inclusionAI/Ming) is an omni-modal model supporting text, image, video, and audio understanding, with text and speech outputs.

vLLM-Omni supports two deployment modes:

| Mode | Stage config | Output |
|------|-------------|--------|
| Thinker only (multimodal understanding) | `ming_flash_omni_thinker.yaml` (default `--omni`) | Text |
| Thinker + Talker (omni-speech) | `ming_flash_omni.yaml` | Text + Audio |

For standalone TTS (talker only), see [`examples/offline_inference/ming_flash_omni_tts/`](../ming_flash_omni_tts/).

## Setup

Please refer to the [stage configuration documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/stage_configs/) to configure memory allocation appropriately for your hardware setup.

The default `--omni` flag runs thinker only.  For omni-speech, pass the two-stage config explicitly:

```bash
--stage-configs-path vllm_omni/model_executor/stage_configs/ming_flash_omni.yaml
```

## Run examples

The end-to-end script defaults to built-in assets; pass `--image-path`,
`--audio-path`, or `--video-path` to override.

```bash
# Text-only
python examples/offline_inference/ming_flash_omni/end2end.py --query-type text

# Image / audio / video / mixed understanding
python examples/offline_inference/ming_flash_omni/end2end.py --query-type use_image
python examples/offline_inference/ming_flash_omni/end2end.py --query-type use_audio
python examples/offline_inference/ming_flash_omni/end2end.py --query-type use_video --num-frames 16
python examples/offline_inference/ming_flash_omni/end2end.py --query-type use_mixed_modalities \
    --image-path /path/to/image.jpg --audio-path /path/to/audio.wav
```

#### Reasoning (Thinking Mode)

Reasoning ("detailed thinking on") is applied by the script when
`--query-type reasoning` is set. The default prompt matches Ming's cookbook
and expects the reference figure from the upstream repo — see
`get_reasoning_query` in `end2end.py`.

```bash
python examples/offline_inference/ming_flash_omni/end2end.py -q reasoning --image-path ./3_0.png
```

### Omni-speech (thinker + talker)

To enable spoken output, use the two-stage config and request `audio` (or `text,audio`) modalities.
The thinker processes your multimodal input, generates text, then the talker synthesises the response as speech.

**Audio-only output** (speech response, no text):
```bash
python examples/offline_inference/ming_flash_omni/end2end.py \
    --query-type text \
    --stage-configs-path vllm_omni/model_executor/stage_configs/ming_flash_omni.yaml \
    --modalities audio \
    --output-dir output_ming_omni_speech
```

**Both text and audio output**:
```bash
python examples/offline_inference/ming_flash_omni/end2end.py \
    --query-type use_audio \
    --stage-configs-path vllm_omni/model_executor/stage_configs/ming_flash_omni.yaml \
    --modalities text,audio \
    --output-dir output_ming_omni_speech
```

Generated `.wav` files are saved to `--output-dir` (default `output_ming`), one per request.

The stage config allocates thinker on GPUs 0–3 and talker on GPU 3 by default. Adjust `devices` in the YAML to match your hardware.

### Modality control

| `--modalities` | Thinker output | Talker | Saved files |
|---------------|----------------|--------|-------------|
| `text` (default) | Text | Not run | `<id>.txt` |
| `audio` | Text (internal) | Runs | `<id>.wav` |
| `text,audio` | Text | Runs | `<id>.txt` + `<id>.wav` |

Pass `--stage-configs-path /path/to/your_config.yaml` to any of the commands
above to override the stage config.

## Online serving

For online serving via the OpenAI-compatible API, see [examples/online_serving/ming_flash_omni/README.md](../../online_serving/ming_flash_omni/README.md).
