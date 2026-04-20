# Voxtream2 Offline Inference

`end2end.py` runs Voxtream2 text-to-speech through the vLLM-Omni `Omni`
engine with the single-stage Voxtream2 stage config.

Voxtream2 reads its generator settings from the Voxtream repo configs before
loading the model weights declared in `configs/generator.json`. Make sure the
`voxtream` package is importable, for example by installing it or by pointing
`--voxtream-root` at a local Voxtream checkout that contains:

- `configs/generator.json`
- `configs/speaking_rate.json`

```sh
git clone https://github.com/herimor/voxtream.git
uv pip install --no-deps -e voxtream
```

Example:

```sh
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python examples/offline_inference/voxtream2/end2end.py \
  --model herimor/voxtream2 \
  --voxtream-root voxtream \
  --ref-audio voxtream/assets/audio/english_male.wav \
  --text "This is a Voxtream2 example running through vLLM Omni." \
  --output output_audio/voxtream2.wav
```

Useful options:

| Option | Default | Description |
| --- | --- | --- |
| `--model` | `herimor/voxtream2` | HuggingFace repo ID or local model path. |
| `--stage-configs-path` | `vllm_omni/model_executor/stage_configs/voxtream2_1stage.yaml` | Voxtream2 stage config. |
| `--voxtream-root` | unset | Local Voxtream repo root containing `configs/*.json`. |
| `--ref-audio` | required | Reference audio path. |
| `--text` | required | Text to synthesize. |
| `--output` | required | Output WAV path. |
