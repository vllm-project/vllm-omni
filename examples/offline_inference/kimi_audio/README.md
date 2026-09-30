# Kimi-Audio: Offline inference

`end2end.py` sends one text and/or audio request through `Omni.generate()`.
The AR stage returns text, and the acoustic stage produces audio when
`--output-type both` is selected. This example saves complete results;
online and streaming usage is described in the
[serving example](../../online_serving/kimi_audio/README.md).

## Setup

Follow the [installation guide](../../../docs/getting_started/installation/README.md)
and install vLLM-Omni with its `kimi-audio` extra. The model needs the full
Kimi-Audio-7B-Instruct checkpoint, including `whisper-large-v3`,
`audio_detokenizer` and `vocoder`, plus the GLM voice tokenizer snapshot.
Model IDs can be used directly; local snapshots can be selected with
`--model` and `--glm-tokenizer-path`.

The bundled deployment puts both stages on GPU 0 with BF16 and eager execution.
The underlying single-request path was exercised on an A800 80GB; other memory
budgets need an appropriate `--deploy-config`. This CLI example itself has not
been rerun on GPU after being extracted from the manual scripts.

## Run

Run these commands from the repository root. Text input defaults to a short
Chinese introduction request:

```bash
python examples/offline_inference/kimi_audio/end2end.py \
  --output-dir outputs/kimi-introduction
```

Use local weights and your own prompt:

```bash
python examples/offline_inference/kimi_audio/end2end.py \
  --model /path/to/Kimi-Audio-7B-Instruct \
  --glm-tokenizer-path /path/to/glm-4-voice-tokenizer \
  --text "你好，请介绍一下你自己。" \
  --output-type both --output-dir outputs/kimi-reply
```

Transcribe a local audio file:

```bash
python examples/offline_inference/kimi_audio/end2end.py \
  --audio-path /path/to/input.wav \
  --text "请将音频内容转换为文字。" \
  --output-type text --output-dir outputs/kimi-transcription
```

Answer an audio question with text and speech:

```bash
python examples/offline_inference/kimi_audio/end2end.py \
  --audio-path /path/to/question.wav \
  --output-type both --output-dir outputs/kimi-audio-reply
```

Input audio must be mono; the example reuses vLLM's resampler for 16 kHz input.
When `--audio-path` is provided without `--text`, only the audio message is sent.
Add the local-weight options above to either audio command when needed.

## Outputs and sampling

The example prints the text and its AR finish reason, writes `text.txt`, and
saves generated audio as `audio-0.wav` (mono, 24 kHz). Choose a separate output
directory for each run to keep results; existing files with those names are
overwritten. Text-only requests do not write a WAV, although the supplied
two-stage deployment still initializes both stages.

`--max-tokens` defaults to 512 and `--seed` to 42. Sampling otherwise comes
from the deployment: greedy text and audio temperature 0.8 / top-k 10.
The model generates a reply, rather than guaranteeing a verbatim reading of
the prompt. `finish_reason=length` means the AR step limit was reached;
audio response completion alone does not establish natural AR termination.
