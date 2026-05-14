# Covo-Audio-Chat for audio chat on 1x A100 80GB

## Summary

- Vendor: Tencent
- Model: `tencent/Covo-Audio-Chat`
- Task: Audio chat — audio (and optional text) input, interleaved text + audio output
- Mode: Online serving with the OpenAI-compatible API (offline inference also supported)
- Maintainer: Community

## When to use this recipe

Use this recipe when you want a known-good starting point for serving
`tencent/Covo-Audio-Chat` with vLLM-Omni on a single 80 GB-class GPU and
validate the deployment with the bundled client example.

## References

- Related example under `examples/`:
  [`examples/online_serving/covo_audio/README.md`](../../examples/online_serving/covo_audio/README.md)
- Offline end-to-end script:
  [`examples/offline_inference/covo_audio/end2end.py`](../../examples/offline_inference/covo_audio/end2end.py)
- Deploy config:
  [`vllm_omni/deploy/covo_audio.yaml`](../../vllm_omni/deploy/covo_audio.yaml)
- Pipeline topology:
  [`vllm_omni/model_executor/models/covo_audio/pipeline.py`](../../vllm_omni/model_executor/models/covo_audio/pipeline.py)

## Hardware Support

This recipe currently documents one tested reference configuration for
CUDA GPU serving. The fused thinker+talker stage and the BigVGAN code2wav
stage both run on the same physical device.

## GPU

### 1x A100 80GB

#### Environment

- OS: Linux
- Python: 3.12
- Driver / runtime: NVIDIA CUDA 12.x with an A100 80 GB GPU
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from
- Extra Python deps: `torchdiffeq`, `librosa`

#### Command

Install the two runtime-only deps that are not pulled in by vLLM-Omni itself:

```bash
pip install torchdiffeq librosa
```

Start the server from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve tencent/Covo-Audio-Chat --omni \
    --trust-remote-code --port 18091
```

The deploy config at `vllm_omni/deploy/covo_audio.yaml` is loaded
automatically by the model registry.

#### Verification

With the server ready, send a test request via the bundled client:

```bash
cd examples/online_serving/covo_audio
python openai_chat_completion_client.py
```

Expected: the client prints the generated text and writes an output WAV
file to the current directory.

#### Notes

- Memory usage: Stage 0 (7B LLM) ~16 GiB + stage 1 (BigVGAN vocoder)
  ~2 GiB, both resident on the same device.
- Key flags: `--omni` and `--trust-remote-code` are both required.
- System prompt: the bundled `openai_chat_completion_client.py` and
  `end2end.py` already include `COVO_AUDIO_SYSTEM_PROMPT` (defined in
  `vllm_omni/model_executor/models/covo_audio/prompt_utils.py`) in the
  system message by default. When integrating your own client, make sure
  to include it — without it, stage 0 emits text-only tokens and the
  output audio is silent.
- Stage 1 runs with `enforce_eager: true` and `dtype: float32` — see the
  header of `deploy/covo_audio.yaml` for the specific reasons.
- Known limitations: this starter recipe covers the single-GPU online-serving
  path only; multi-GPU layouts and platform-specific overrides are not
  documented here.
