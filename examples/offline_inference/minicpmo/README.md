# MiniCPM-o 4.5: Offline inference

Two-stage pipeline: **thinker** (multimodal understanding) → **talker + Token2Wav**
(24 kHz speech). Deploy config auto-loads from
`vllm_omni/deploy/minicpmo_4_5.yaml` (2-GPU default).

## Setup

- Install talker deps: `pip install 'vllm-omni[minicpmo]'` (or `stepaudio2-minicpmo`)
- `--trust-remote-code` is always passed by `end2end.py` (required for MiniCPMO)
- See [stage configuration docs](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/stage_configs/) for memory tuning

## Run examples

### Single prompt

```bash
cd examples/offline_inference/minicpmo
bash run_single_prompt.sh
```

### Multiple prompts

```bash
bash run_multiple_prompts.sh
```

### Thinker tensor parallel (3-GPU)

```bash
bash run_single_prompt_tp.sh
```

### Modality control

Text-only (skip talker / no `<|tts_bos|>`):

```bash
python end2end.py --query-type use_audio --modalities text
```

Text + speech (default — appends `<|tts_bos|>`):

```bash
python end2end.py --query-type text --modalities text,audio
```

### Local media files

```bash
python end2end.py --query-type use_video --video-path /path/to/video.mp4
python end2end.py --query-type use_image --image-path /path/to/image.jpg
python end2end.py --query-type use_audio --audio-path /path/to/audio.wav
python end2end.py --query-type use_mixed_modalities \
    --video-path /path/to/video.mp4 \
    --image-path /path/to/image.jpg \
    --audio-path /path/to/audio.wav
```

Supported `--query-type` values:

| Query type | Inputs |
|---|---|
| `text` | Text only |
| `use_image` | Image + text |
| `use_audio` | Audio + text |
| `use_video` | Video + text |
| `use_multi_audios` | Two audio clips |
| `use_mixed_modalities` | Audio + image + video |

### Custom deploy config

```bash
python end2end.py --query-type text \
    --deploy-config /path/to/vllm_omni/deploy/minicpmo_4_5_8x4090.yaml
```

## Notes

- Speech requires `<|tts_bos|>` on the assistant prefix (offline equivalent of
  online `chat_template_kwargs.use_tts_template=true`). Without it, the talker
  gets an empty TTS span.
- Output WAV is **24 kHz mono**.
- Placeholders in the prompt are MiniCPM-style:
  `(<image>./</image>)`, `(<audio>./</audio>)`, `(<video>./</video>)`.
- Default layout needs **2 GPUs**. Async chunking is off in the bundled YAMLs.

## Online serving

See [`examples/online_serving/minicpmo/`](../../online_serving/minicpmo/) and
the recipe [`recipes/OpenBMB/MiniCPM-o-4_5.md`](../../../recipes/OpenBMB/MiniCPM-o-4_5.md).
