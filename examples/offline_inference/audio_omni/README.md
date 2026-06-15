# Audio-Omni offline inference

Generate speech with the [Audio-Omni](https://huggingface.co/HKUSTAudio/Audio-Omni)
continuous-transformer diffusion pipeline (`AudioOmniPipeline`).

This example currently supports **`tts` (text-to-speech)** and **`voice_clone`
(TTS in a reference speaker's voice)**. Audio-Omni's full capability set will be
added in later PRs:

- **Generation**: Text-to-Speech (with voice cloning), Voice Conversion,
  Text-to-Audio (T2A), Text-to-Music (T2M), Video-to-Audio (V2A), Video-to-Music (V2M)
- **Editing**: add / remove / extract / style-transfer on existing audio
- **Understanding**: audio/video question answering

## Prerequisites

Download the upstream Audio-Omni release (`Audio-Omni.json` + `model.ckpt`, ~22 GB):

```bash
huggingface-cli download HKUSTAudio/Audio-Omni --local-dir ./audio_omni_weights
```

## Usage

```bash
# Text-to-speech:
python end2end.py --model ./audio_omni_weights --mode tts \
  --prompt "Hello, welcome to Audio-Omni."

# Voice cloning (reference wav + its transcript):
python end2end.py --model ./audio_omni_weights --mode voice_clone \
  --prompt "The quick brown fox jumps over the lazy dog." \
  --voice-prompt ref.wav --voice-ref-text "Transcript of ref.wav."
```

## Arguments

- `--model`: local Audio-Omni bundle dir (or HF id; default `HKUSTAudio/Audio-Omni`).
- `--mode`: `tts` or `voice_clone`.
- `--prompt`: transcript to synthesize (required).
- `--voice-prompt`, `--voice-ref-text`: reference wav and its transcript (`voice_clone`).
- `--num-inference-steps`, `--guidance-scale`, `--seed`, `--output-dir`: generation knobs
  (defaults match upstream: 100 / 7.0 / 42).
- `--no-postprocess`: keep the raw ~10.96 s output instead of the gradio-style trim.

Outputs land in `<output-dir>/<mode>.wav` as 16-bit stereo WAV. For `voice_clone` the
saved file drops the re-spoken reference head and trims silence (the raw waveform is
also written as `<mode>_raw.wav`).
