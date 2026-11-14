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

For the current checkout, follow the [CUDA installation guide](../../docs/getting_started/installation/README.md).
This checkout targets vLLM 0.33.0, PyTorch 2.14.0 and CUDA 13.0.
Verify that the driver can initialize CUDA in that environment before running the model.

The previous profile was validated on 2026-10-13 with one RTX 4090 24 GB,
driver 550.67, Python 3.12.14, PyTorch 2.13.0+cu129,
vLLM 0.31.0+cu129, and transformers 5.14.1. The Omni source checkout
was based on upstream `main` commit `b463516ff`. The checkpoint revision was
`642071559bfc6346c2359d19dcb6be3f9dd8a05d`. These are validation
versions, not a general installation lockfile.

The October validation host used `VLLM_USE_FLASHINFER_SAMPLER=0` and preloaded a
compatible `libstdc++.so.6`; these are host-specific runtime settings.

The official source used in validation was
`2f1ea9082400547242641f5271b6f941c9f439d1`. Make it importable in the
Omni environment without installing its older pinned PyTorch over Omni's
PyTorch. The Spark-specific imports used by the CI job are `einops==0.8.2`,
`einx==0.4.3`, `frozendict==2.5.1`, `omegaconf==2.3.1`, and `soxr==1.1.0`; install their
dependencies as needed and verify the environment with `pip check`.

The CUDA CI dependency image caches that pinned source at `SPARK_TTS_SOURCE`
and installs the codec dependencies during image construction. The offline
GPU test uses those installed files. Building this dependency layer still
requires access to `github.com/SparkAudio/Spark-TTS` and the pinned commit.
The build checks the resolved commit and imports BiCodec before accepting the
image. A failed fetch stops image construction; it does not fall back to an
unpinned revision. GPU test jobs do not clone or install Spark dependencies.

Stage 0 validates the selected tokenizer's global and semantic code ranges
and stop markers before constructing the model. A mismatched tokenizer raises
an error. Keep the checkpoint and its tokenizer together.

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

An empty or whitespace-only `SPARK_TTS_REF_TEXT` selects cloning without a
transcript. A nonempty transcript requires semantic codes from the reference
encoder. The official prompt concatenates the transcript and target verbatim at
the boundary. Include a trailing space in the transcript when the language
requires it (for example, `SPARK_TTS_REF_TEXT="Reference. "`). Chinese text
can use a punctuation boundary such as `参考音频。` directly. The decoder receives one scheduling placeholder per finished
request; all speech codes are delivered through the stage connector.

The Spark pipeline declares `shutdown_grace_s=15.0` for its BiCodec stage to
allow decoder cleanup to finish. Other stages keep the upstream shutdown
policy unless their pipeline declares a grace period. An explicit shutdown
timeout takes precedence over this stage default.

## Verification and limits

Both the CUDA ready and merge lanes run the short and long control-voice
cases in `tests/e2e/offline_inference/test_spark_tts.py` on one NVIDIA L4.
The long case checks for more than 21 seconds of audio; the merge lane also
transcribes it and checks that words from the final sentence are present.

The October RTX 4090 real-weight runs used Omni commit
`5234583f164536898db8c9a9d89c8dd1fbf2b234`, based on upstream `b463516ff`.
Control synthesis and cloning with and without a transcript produced finite
16 kHz audio. The stage processes exited with no forced-kill or leaked
semaphore warning in those runs.

A [reviewer on one H20](https://github.com/vllm-project/vllm-omni/pull/9889#discussion_r4486777275)
compared `062c582ab` against the official generator. The five short control
prompts and two clones had equal or lower Whisper large-v3 WER on Omni.
On a 99-word paragraph, the 1024-token limit cut Omni's output to 19.7 s
(WER 0.424); the official generator finished at 35.1 s (WER 0.000).

On a local RTX 5060 Ti (16 GB, WSL2), the offline pipeline passed the short
and long tests with real weights. On the separate 94-word regression
paragraph, a 1024-token cap produced 19.70 s of speech (Whisper large-v3
WER 18/94). With the new default of 3000, the model generated 1255 tokens
and 24.30 s of speech with WER 0/94. The Buildkite L4 run is still pending.

After rebasing onto upstream `b8f128171`, both real-weight offline tests passed
again on the local RTX 5060 Ti (16 GB, WSL2, driver 621.01), using vLLM 0.33.0,
PyTorch 2.14.0+cu130 and the default FlashInfer sampler. The advanced run
passed 2 tests in 133.74 s, including the Whisper small final-sentence check.
The long case generated 1255 tokens. Both stage processes completed shutdown.
This is local validation; a maintainer-admitted Buildkite L4 run is still needed.

The deploy config selects runner v1, eager execution, and `async_chunk: false`.
Stage 0 defaults to 3000 generated tokens, matching the official generator.
Its 4096-token context leaves 1096 input tokens when using the full output
budget. Count the complete tokenized prompt, including the reference codes
in clone mode. For longer inputs, shorten the reference or split the target
text so the prompt and desired output fit together. Both stages use logical
device 0; `CUDA_VISIBLE_DEVICES` selects the physical GPU.
