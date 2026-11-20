# Qwen2.5-Omni on NVIDIA H100

> Online serving for multimodal chat (text / image / audio / video → text +
> speech)

## Summary

- Vendor: Qwen
- Model: `Qwen/Qwen2.5-Omni-7B` (also `Qwen/Qwen2.5-Omni-3B`)
- Task: Multimodal chat with text, image, audio, and video input
- Mode: Online serving with the OpenAI-compatible Chat Completions API;
  offline `Omni.generate()`
- Hardware: 2x NVIDIA H100 80GB (default deploy layout)
- Maintainer: Community

## When to use this recipe

Use this recipe as a practical baseline for
`Qwen/Qwen2.5-Omni-7B` with the bundled three-stage pipeline (Thinker →
Talker → Code2Wav). It matches the serving and offline paths already
exercised by repository examples and e2e tests.

The default deploy YAML colocates Thinker and Code2Wav on GPU 0 and places
Talker on GPU 1. The bundled [CUDA deploy configuration](../../vllm_omni/deploy/qwen2_5_omni.yaml)
identifies this as its two-H100 layout.

## Supported model contract

| Task | Shared entrypoint |
| --- | --- |
| Online multimodal chat | `vllm serve <model> --omni` then `POST /v1/chat/completions` |
| Offline batched inference | `examples/offline_inference/qwen2_5_omni/end2end.py` |

| Input | Limits / notes |
| --- | --- |
| Text | Chat messages; thinker owns the tokenizer |
| Image | OpenAI image_url (HTTP(S) or local path encoded by the example client) |
| Audio | OpenAI input_audio / audio_url; default decoder is soundfile |
| Video | OpenAI video_url; local paths are base64-encoded by the example client |

| Output | Notes |
| --- | --- |
| `["text"]` | Thinker only; Talker and Code2Wav are skipped |
| `["audio"]` | 24 kHz speech only from Code2Wav; the text output is filtered out |
| `["text", "audio"]` | Text plus 24 kHz speech from Code2Wav |
| Unspecified modalities | Text + audio (pipeline default) |

| Profile on this recipe | Devices | Qualification |
| --- | --- | --- |
| Bundled default (`qwen2_5_omni.yaml`) | 2 GPUs (Thinker+Code2Wav on 0, Talker on 1) | Two-H100 configuration from the bundled deploy YAML |
| `Qwen/Qwen2.5-Omni-3B` same YAML | 2 GPUs | Configuration-only in this recipe |
| Single-GPU colocation (7B) | 1 CUDA GPU via `--stage-overrides` | Configuration only. A 24 GiB card is too small for this profile |

## References

- Canonical model:
  [Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)
- Online example:
  [`examples/online_serving/qwen2_5_omni/README.md`](../../examples/online_serving/qwen2_5_omni/README.md)
- Offline example:
  [`examples/offline_inference/qwen2_5_omni/README.md`](../../examples/offline_inference/qwen2_5_omni/README.md)
- User guides:
  [`docs/user_guide/examples/online_serving/qwen2_5_omni.md`](../../docs/user_guide/examples/online_serving/qwen2_5_omni.md),
  [`docs/user_guide/examples/offline_inference/qwen2_5_omni.md`](../../docs/user_guide/examples/offline_inference/qwen2_5_omni.md)
- Chat Completions API:
  [`docs/serving/chat_completions_api.md`](../../docs/serving/chat_completions_api.md)
- Supported models:
  [`docs/models/supported_models.md`](../../docs/models/supported_models.md)
- Related issue:
  [#2645](https://github.com/vllm-project/vllm-omni/issues/2645)

## Hardware

- Accelerator model and per-device memory: NVIDIA H100 80GB
- Number of devices: 2
- Device interconnect: NVLink (typical H100 server); the deploy file does not
  require a specific interconnect
- Weight placement: all three stages keep their weights on the GPUs.
- Qualification scope: configuration from
  `vllm_omni/deploy/qwen2_5_omni.yaml` (header: verified on 2x H100). This
  recipe does not add new H100 wall-clock or peak-HBM measurements.

Do not copy these device counts onto ROCm, NPU, or XPU without a separate
hardware-suffixed recipe. Those platforms merge `platforms:` overlays in the
same YAML; they are not qualified here.

## Software environment

- OS: Linux
- Python: 3.12 (3.11–3.13 supported by this checkout)
- Driver / runtime: CUDA matching the vLLM 0.34.0 wheel (PyTorch 2.14 / CUDA
  13.0 for the published CUDA image)
- vLLM version: 0.34.0
- vLLM-Omni version or commit: this source checkout (`main`)

## Command

`--omni` is required.

```bash
vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8091
```

The default deploy config `vllm_omni/deploy/qwen2_5_omni.yaml` is selected
through `default_deploy_config_name`. CUDA, ROCm, NPU, and XPU use Model
Runner V2. `async_chunk` is **off** in this deploy (unlike Qwen3-Omni).

The Thinker stage stays eager: compiling its decode changes
image-conditioned greedy text. The Talker enables CUDA graphs on CUDA
because it emits a long codec-token stream.

Optional overlay:

```bash
vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8091 \
  --deploy-config /path/to/your_qwen2_5_omni_overrides.yaml
```

Text-only serving (skip Talker / Code2Wav work):

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Omni-7B",
    "messages": [{"role": "user", "content": "Describe vLLM-Omni in brief."}],
    "modalities": ["text"]
  }'
```

The single-GPU 7B CUDA profile below is configuration-only, with unmeasured
peak VRAM. The 7B Thinker weights alone need about 16.78 GiB. At a 45%
allocation, that requires about 37.3 GiB of total GPU memory before
activations and the KV cache. The one-card CI profile uses the 3B model on
a 24 GiB Intel Arc Pro B60.

```bash
vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8091 \
  --stage-overrides '{
    "0": {
      "devices": "0", "gpu_memory_utilization": 0.45,
      "max_model_len": 16384, "max_num_batched_tokens": 16384,
      "skip_mm_profiling": true
    },
    "1": {
      "devices": "0", "gpu_memory_utilization": 0.25,
      "max_model_len": 16384, "max_num_batched_tokens": 16384,
      "skip_mm_profiling": true, "enforce_eager": true
    },
    "2": {
      "devices": "0", "gpu_memory_utilization": 0.15,
      "max_model_len": 16384, "max_num_batched_tokens": 16384
    }
  }' \
  --default-sampling-params '{"0": {"max_tokens": 128}}'
```

Offline:

```bash
cd examples/offline_inference/qwen2_5_omni
bash run_single_prompt.sh
```

## Verification

After the server is up:

```bash
python examples/online_serving/openai_chat_completion_client_for_multimodal_generation.py \
  --model Qwen/Qwen2.5-Omni-7B \
  --query-type text \
  --port 8091 \
  --host localhost
```

Expect a non-empty assistant message. For mixed image/audio/video input, use
`--query-type use_mixed_modalities` (see the online example README).

Text + audio:

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Omni-7B",
    "messages": [{"role": "user", "content": "Describe vLLM-Omni in brief."}],
    "modalities": ["text", "audio"]
  }'
```

Expect `choices[0].message.content` plus audio in the multimodal payload.

The weekly CUDA [online suite](../../tests/e2e/online_serving/test_qwen2_5_omni_expansion.py)
uses two L4 GPUs. Its CI configuration places Code2Wav on GPU 1 with Talker.
The [offline suite](../../tests/e2e/offline_inference/test_qwen2_5_omni_expansion.py)
runs on four L4 GPUs. Both suites use dedicated CI deployment settings.

## Notes

- Key flags: `--omni` is required; `--deploy-config` only for overlays.
- Default `max_num_seqs` is 1 per stage in the bundled YAML.
- Prefix caching is off. Thinker `mm_processor_cache_gb` is 0.
- Stage 2 disables FlashInfer autotune so a dummy autotune run does not OOM
  the shared GPU 0.

## Supported features

| Feature | Status | Guide |
| --- | --- | --- |
| Model Runner V2 | Default on CUDA / ROCm / NPU / XPU | [Stage configs](../../docs/configuration/stage_configs.md) |
| Async chunk | Off in the bundled deploy | [Async chunk](../../docs/design/feature/async_chunk.md) |
| Prefix caching | Off | [Prefix caching](../../docs/design/feature/prefix_caching.md) |
| Thinker CUDA graph | Off (eager) | Deploy YAML comments in `qwen2_5_omni.yaml` |
| Talker CUDA graph | On in the bundled CUDA layout. The single-GPU profile, ROCm, and XPU use eager execution | same |
| Tensor-parallel Thinker | Example `strategy_tp2.yaml` needs ≥3 GPUs | [Composable parallel](../../docs/configuration/composable_parallel.md) |
| AutoRound thinker | Supported through AutoRound checkpoints | [Quantization overview](../../docs/user_guide/quantization/overview.md) |
