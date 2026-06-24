# ACE-Step 1.5 — Text to Music

Offline example for generating music with the ACE-Step 1.5 model.

## Requirements

- A diffusers-format ACE-Step checkpoint. The original repo at
  [`ACE-Step/Ace-Step1.5`](https://huggingface.co/ACE-Step/Ace-Step1.5) is **not** in
  diffusers format. Until the converted checkpoint is hosted publicly, run the
  conversion script that ships with diffusers PR
  [`huggingface/diffusers#13095`](https://github.com/huggingface/diffusers/pull/13095)
  against the original repo and point `--model` at the converted directory.
- `soundfile` or `scipy` (for writing the WAV output).

## Quick start

```bash
python text_to_music.py \
    --model /path/to/ace-step-v15-turbo-diffusers \
    --prompt "An upbeat jazz piano piece with a walking bass line."
```

Outputs a WAV file at 48 kHz stereo.

## Defaults match the turbo recipe

| Flag | Default | Notes |
| --- | --- | --- |
| `--num-inference-steps` | `8` | Turbo is designed for 8 steps. |
| `--shift` | `3.0` | Flow-matching timestep shift; choose from `{1, 2, 3}`. |
| `--guidance-scale` | `1.0` | Turbo distills guidance — values >1 are coerced back to 1 with a warning. |
| `--audio-duration` | `30.0` s | |

## Known limitation — flash + sliding window

ACE-Step's DiT uses sliding-window self-attention. vLLM-Omni's flash backend
(`vllm_omni/diffusion/attention/backends/flash_attn.py`) does not currently
expose a `window_size` parameter, so flash silently drops the window constraint
and the model produces noise output. As a workaround, `AceStepAttention`
hard-pins SDPA for sliding-window self-attention sites at construction time
(cross-attention and full self-attention sites keep using flash). This is
automatic — no user flag required. Once `window_size` plumbing lands in
`flash_attn.py` in a follow-up PR, the hard-pin will be removed.

## Out of scope (first PR)

Cover / repaint / extract / lego / complete tasks and their dependencies
(audio tokenizer / detokenizer, reference-audio timbre conditioning,
APG-normalised guidance) are not wired up yet. Track future work in
[issue #1252](https://github.com/vllm-project/vllm-omni/issues/1252).
