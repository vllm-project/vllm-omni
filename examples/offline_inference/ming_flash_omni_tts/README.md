# Ming-flash-omni Standalone TTS (Offline)

This example runs **Ming-flash-omni-2.0 talker-only** offline inference with:

- `model`: `Jonathan1909/Ming-flash-omni-2.0`
- `stage config`: `vllm_omni/model_executor/stage_configs/ming_flash_omni_tts.yaml`

It follows the Ming cookbook parameter style:

- `prompt`: `"Please generate speech based on the following description.\n"`
- `max_decode_steps`: `200`
- `cfg`: `2.0`
- `sigma`: `0.25`
- `temperature`: `0.0`

## Quick Start

```bash
python examples/offline_inference/ming_flash_omni_tts/end2end.py --case style
```

## Cases

```bash
# Style
python examples/offline_inference/ming_flash_omni_tts/end2end.py --case style

# IP
python examples/offline_inference/ming_flash_omni_tts/end2end.py --case ip

# Basic (speed/pitch/volume control)
python examples/offline_inference/ming_flash_omni_tts/end2end.py --case basic
```

## Useful Arguments

- `--text`: override default text in the selected case
- `--output`: custom output wav path
- `--model`: local model path or HF repo id
- `--stage-configs-path`: custom talker stage config path
- `--log-stats`: enable runtime stats logs

## Notes

- This directory is for **standalone talker deployment (TTS)**.
- For Ming thinker multimodal understanding examples, see:
  `examples/offline_inference/ming_flash_omni/`.
