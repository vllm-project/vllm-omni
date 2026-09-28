# Spark-TTS-0.5B offline inference on RTX 4090

## Summary

- Vendor: SparkAudio
- Model: `SparkAudio/Spark-TTS-0.5B`
- Task: text-to-speech, with optional reference-audio voice cloning
- Mode: offline `Omni.generate`, complete waveform output
- Hardware: one NVIDIA RTX 4090 24 GB
- Maintainer: Community

## Supported model contract

| Input | Output | Current scope |
| --- | --- | --- |
| Text and discrete gender, pitch, speed labels | Complete 16 kHz mono waveform | Non-streaming, single GPU |
| Text and one reference WAV, with optional reference transcript | Complete 16 kHz mono waveform | Reference encoding runs on CPU |

The model directory contains `config.yaml`, `LLM/`, `BiCodec/`, and
`wav2vec2-large-xlsr-53/`. Keep the checkpoint directory intact. Stage 0
generates speech tokens with vLLM's Qwen2 implementation; stage 1 decodes the
complete token sequence with BiCodec inside Omni. The official `sparktts`
Python package supplies BiCodec and reference-encoder definitions. No separate
Spark HTTP service is needed.

## References

- [Model checkpoint](https://huggingface.co/SparkAudio/Spark-TTS-0.5B)
- [Official Spark-TTS source](https://github.com/SparkAudio/Spark-TTS)
- [Model request and first-PR scope](https://github.com/vllm-project/vllm-omni/issues/4295)

## Hardware and software environment

This profile was validated on one RTX 4090 24 GB with driver 550.67,
Python 3.12.14, PyTorch 2.13.0+cu129, vLLM 0.30.0+cu129, and
vLLM-Omni 0.30.1.dev15+g1b97ff118. The checkpoint revision was
`642071559bfc6346c2359d19dcb6be3f9dd8a05d`. These are validation
versions, not a general installation lockfile. Select a compatible vLLM/Omni
stack for your host before running this recipe.

The official source used in validation was
`2f1ea9082400547242641f5271b6f941c9f439d1`. Make it importable in the
Omni environment without installing its older pinned PyTorch over Omni's
PyTorch. The Spark-specific imports used by the CI job are `einops==0.8.2`,
`einx==0.4.3`, `omegaconf==2.3.1`, and `soxr==1.1.0`; install their
dependencies as needed and verify the environment with `pip check`.

## Offline command

Run from the vLLM-Omni repository root. Set `SPARK_TTS_MODEL_DIR` to a local
copy of the complete checkpoint and `SPARK_TTS_SOURCE_DIR` to the pinned
official source checkout. The example writes `spark_tts.wav` in the current
directory. Set `SPARK_TTS_REF_AUDIO` to a reference WAV for voice cloning;
`SPARK_TTS_REF_TEXT` is an optional transcript of that reference.

```bash
export CUDA_VISIBLE_DEVICES=0
export SPARK_TTS_MODEL_DIR=/path/to/Spark-TTS-0.5B
export SPARK_TTS_SOURCE_DIR=/path/to/Spark-TTS
export PYTHONPATH="$SPARK_TTS_SOURCE_DIR:$PWD:${PYTHONPATH:-}"

python - <<'PY'
import os

import numpy as np
import soundfile as sf
import torch

from vllm_omni import Omni
from vllm_omni.model_executor.models.spark_tts.prompt import clone_prompt, control_prompt

model = os.environ["SPARK_TTS_MODEL_DIR"]
ref_audio = os.environ.get("SPARK_TTS_REF_AUDIO")
if ref_audio:
    from sparktts.models.audio_tokenizer import BiCodecTokenizer

    encoder = BiCodecTokenizer(model, device=torch.device("cpu"))
    with torch.inference_mode():
        global_codes, semantic_codes = encoder.tokenize(ref_audio)
    prompt = clone_prompt(
        "Hello from Spark-TTS in Omni.",
        global_codes.reshape(-1).tolist(),
        semantic_codes.reshape(-1).tolist(),
        os.environ.get("SPARK_TTS_REF_TEXT"),
    )
    del encoder
else:
    prompt = control_prompt("Hello from Spark-TTS in Omni.")

engine = Omni(model=model, deploy_config="vllm_omni/deploy/spark_tts.yaml")
try:
    outputs = engine.generate([prompt])
    payload = outputs[0].outputs[0].multimodal_output
    value = payload.get("audio")
    if value is None:
        value = payload.get("model_outputs")
    if value is None:
        raise RuntimeError(f"BiCodec returned no audio: {list(payload)}")
    items = value if isinstance(value, list) else [value]
    audio = torch.cat([torch.as_tensor(item).float().reshape(-1).cpu() for item in items]).numpy()
finally:
    engine.close()

if audio.size == 0 or not np.isfinite(audio).all():
    raise RuntimeError("BiCodec returned empty or non-finite audio")
sf.write("spark_tts.wav", audio, 16000, format="WAV", subtype="PCM_16")
print(f"Wrote spark_tts.wav: {audio.size / 16000:.2f} s at 16 kHz")
PY
```

For voice cloning, set `SPARK_TTS_REF_AUDIO` before the same command. Use a
reference recording that you have permission to use.

## Verification and limits

The offline CUDA smoke test is
`tests/e2e/offline_inference/test_spark_tts.py`. It checks that one control
request produces nonempty, finite audio at 16 kHz. It passed with real weights
on an RTX 4090; the Buildkite L4 GPU run has not been observed yet. A valid
waveform does not establish transcript accuracy, naturalness, or speaker
similarity. Those require separate evaluation.

The provided deploy config uses `model_runner: v1`, eager execution,
`async_chunk: false`, and a 1024-token generation limit. The two stages both
use logical device `0`; `CUDA_VISIBLE_DEVICES` selects which physical GPU
that means. Raise concurrency or context limits only after measuring memory
and output quality on the target hardware.
