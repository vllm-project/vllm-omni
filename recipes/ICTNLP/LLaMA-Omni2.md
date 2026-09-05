# LLaMA-Omni 2

> Native text/audio-to-text-and-speech inference

## Summary

- Vendor: ICTNLP
- Model: `ICTNLP/LLaMA-Omni2-0.5B`
- Decoder: `ICTNLP/cosy2_decoder`
- Task: Text or speech input to streamed text and 24 kHz speech
- Mode: Online serving or offline `AsyncOmni` inference
- Maintainer: Community

## License

The upstream LLaMA-Omni 2 checkpoint is intended for **academic,
non-commercial use**. Review and comply with the model and decoder repository
licenses before downloading, serving, or redistributing either checkpoint.

## Architecture

vLLM-Omni runs the model as three native stages:

1. Whisper speech encoder/projector plus a vLLM Qwen2 Thinker using paged KV
   cache;
2. a vLLM Qwen2 Talker that generates codec tokens;
3. the CosyVoice2 flow/HiFT decoder, which emits streamed mono 24 kHz audio.

The Thinker and Talker use vLLM tensor-parallel linear and attention layers.
They do not call Transformers `generate()` or use a Transformers
`DynamicCache`.

## Requirements

- Linux with CUDA
- Two 80 GB-class NVIDIA GPUs for the default deployment
- Current vLLM-Omni source checkout
- Access to both Hugging Face repositories listed above

Install the project using its normal development or package installation
workflow. The model implementation also requires the Whisper runtime used by
the checkpoint.

## Start the server

From the repository root:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
vllm serve ICTNLP/LLaMA-Omni2-0.5B \
  --omni \
  --port 8091
```

The model registry selects
`vllm_omni/deploy/llama_omni2.yaml`. That deployment places the Thinker on GPU
0 and the Talker plus Code2Wav stages on GPU 1. To change memory limits,
devices, or tensor parallelism, copy the YAML and pass it with
`--deploy-config`.

## Text input

Request both output modalities:

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ICTNLP/LLaMA-Omni2-0.5B",
    "messages": [
      {"role": "user", "content": "Say hello in one short sentence."}
    ],
    "modalities": ["text", "audio"]
  }'
```

## Speech input

Use the chat-completions endpoint and provide mono audio through the standard
multimodal audio input format. The model's `<speech>` placeholder is expanded
by the native vLLM multimodal processor. The completions endpoint is
intentionally rejected because speech input requires chat message structure.

The repository's CUDA harness provides a deterministic speech smoke test:

```bash
.venv/bin/python tests/e2e/offline_inference/llama_omni2/run_llama_omni2_e2e.py \
  --model ICTNLP/LLaMA-Omni2-0.5B \
  --deploy-config tests/e2e/offline_inference/llama_omni2/llama_omni2_tp1.yaml \
  --output-dir /tmp/llama-omni2-e2e \
  --label tp1 \
  --mode speech
```

The test requires non-empty text tokens, multiple non-empty audio chunks,
finite non-zero waveform samples, and a 24,000 Hz sample rate.

## Tensor parallelism

The Thinker and Talker support tensor parallelism through native vLLM Qwen2
layers. A two-GPU example is provided at:

```text
tests/e2e/offline_inference/llama_omni2/llama_omni2_tp2.yaml
```

Run TP=1 and TP=2 deterministic text-to-speech parity with:

```bash
MODEL=ICTNLP/LLaMA-Omni2-0.5B \
DECODER_MODEL=ICTNLP/cosy2_decoder \
OUTPUT_DIR=/tmp/llama-omni2-tp \
bash tests/e2e/offline_inference/llama_omni2/run_llama_omni2_e2e.sh tp
```

This compares greedy text token IDs and the generated 24 kHz waveform. The
unit suite additionally verifies that packed QKV, gate/up, and Talker
projection parameters load only their local TP shard.

## Validation

Run the focused CPU/configuration suite:

```bash
.venv/bin/python -m pytest -q \
  tests/model_executor/models/llama_omni2 \
  tests/engine/test_arg_utils.py \
  tests/engine/test_prompt_sampling_params_override.py \
  tests/test_config_factory.py
```

For a CUDA environment with two visible GPUs:

```bash
RUN_E2E=1 E2E_CASE=all bash run_validation.sh
```

The implementation was validated against checkpoint revision
`a16aa9a4ea3f2f363c3db728e8e83ee08e60922c` and CosyVoice2 decoder revision
`7ff21e8e641b00cff2e0492651d654d153b21211` on CUDA with the following gates:

- synchronous and asynchronous three-stage text, speech, and concurrent
  requests;
- TP=1 versus TP=2 greedy text-token, codec-token, and waveform parity;
- exact local-shard hashes for packed QKV, gate/up, and Talker projection
  weights on both TP ranks;
- streamed OpenAI chat-completions speech input with non-empty text, 98
  non-empty WAV chunks, 103,680 finite waveform samples at 24 kHz, and one
  terminal `stop`;
- rejection of `/v1/completions` with HTTP 400, because this model requires
  chat message structure for speech input.

The TP parity gate uses `torch.allclose(rtol=1e-3, atol=1e-4)` for the decoded
waveform and exact equality for deterministic text and codec tokens. The
online gate decodes and validates every streamed WAV chunk rather than only
checking the final payload.

## Known limitations

- Only `ICTNLP/LLaMA-Omni2-0.5B` is in the initial validation matrix.
- Image and video inputs are not supported by this speech-language model.
- The default deployment is single-host and uses `SharedMemoryConnector`.
- The CosyVoice2 decoder remains FP32 for numerical stability.
- Audio is streamed at 24 kHz; clients must concatenate chunks in order.
- Checkpoint use remains subject to the upstream academic, non-commercial
  license terms.
