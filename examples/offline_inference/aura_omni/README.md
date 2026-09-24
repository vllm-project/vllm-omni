# AURA Omni: Offline inference

This example runs the native AURA Omni pipeline offline:

```text
Qwen3-ASR -> AURA/Qwen3-VL -> Qwen3-TTS Talker -> Qwen3-TTS Code2Wav
```

The first stage consumes speech and produces a transcript. The AURA stage then combines the transcript with the video frames from the original request and returns text or `<|silent|>`. Non-silent responses are passed to Qwen3-TTS as AURA-generated token ids.

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

Base voice clone mode is the default. It uses the AURA reference audio unless
you override it:

```bash
python end2end.py \
  --tts-task-type Base \
  --tts-ref-audio /data/yrr/rein_test/shuhan.mp3 \
  --tts-x-vector-only-mode
```

CustomVoice mode requires stages 2 and 3 in `vllm_omni/deploy/aura_omni.yaml`
to point at a Qwen3-TTS CustomVoice checkpoint:

```bash
python end2end.py \
  --tts-task-type CustomVoice \
  --tts-speaker Vivian
```

For local checkpoints, edit the stage `model` entries in
`vllm_omni/deploy/aura_omni.yaml` or pass a copied deploy config with
`--deploy-config`.

## GPU Utilization Recommendation

Set `gpu_memory_utilization` in `vllm_omni/deploy/aura_omni.yaml` by stage.
Suggested starting point:

- Stage 0 (ASR): `0.10`
- Stage 1 (AURA): `0.4`
- Stage 2 (Qwen3-TTS Talker): `0.20`
- Stage 3 (Qwen3-TTS Code2Wav): `0.20`

Generated text and audio are written to `--output-dir`
(default: `output_aura_omni`).
