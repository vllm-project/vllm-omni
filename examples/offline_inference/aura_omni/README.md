# AURA Omni

This example runs the native AURA Omni pipeline offline:

```text
Qwen3-ASR -> AURA/Qwen3-VL -> Qwen3-TTS Talker -> Qwen3-TTS Code2Wav
```

The first stage consumes speech and produces a transcript. The AURA stage then
combines the transcript with the video frames from the original request and
returns text or `<|silent|>`. Non-silent text is passed to Qwen3-TTS.

## Run

```bash
cd examples/offline_inference/aura_omni
bash run_single_prompt.sh
```

Use local media:

```bash
python end2end.py \
  --audio-path /path/to/input.wav \
  --video-path /path/to/video.mp4 \
  --modalities text,audio
```

For local checkpoints, edit the stage `model` entries in
`vllm_omni/deploy/aura_omni.yaml` or pass a copied deploy config with
`--deploy-config`.

Generated text and audio are written to `--output-dir`
(default: `output_aura_omni`).
