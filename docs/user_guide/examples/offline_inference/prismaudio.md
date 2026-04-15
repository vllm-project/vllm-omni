# PrismAudio Offline Inference

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/prismaudio>.


PrismAudio support in `vllm-omni` currently targets the first integration scope:

- input: precomputed `video_features`, `text_features`, and `sync_features`
- model path: `PrismAudioPipeline`
- runtime: rectified-flow sampling + VAE decode
- output: stereo audio tensor / waveform

When the upstream preprocessing dependencies are installed, the pipeline can also
start from `video_path` and build conditioning features through the official-style
video preprocessing and feature-extraction adapters.

## Current Scope

Supported:

- official PrismAudio-style model config via `prismaudio_model_config` or `prismaudio_model_config_path`
- runtime checkpoint loading through `model_paths["transformer"]` and `model_paths["vae"]`
- precomputed conditioning features stored as tensors in the request payload
- conditioning fixtures loaded from `prompt.additional_information["conditioning_path"]`
- optional `conditioning_factory` hooks for runtime feature extraction adapters
- optional `video_preprocessor_factory` and `feature_extractor_factory` hooks for higher-level runtime preprocessing
- declarative `video_preprocessor_config` and `feature_extractor_config` paths for the default official preprocessing stack
- batching multiple prompt-local conditioning payloads in one request
- local smoke testing through the Prismaudio offline e2e test

Not supported yet:

- a guaranteed bundled preprocessing runtime in the base `vllm-omni` environment
- a packaged Hugging Face-style Prismaudio model repo layout from upstream

## Expected Config

The upstream Prismaudio config must keep the official schema. In particular, the builder-facing `model_type` remains an upstream value such as `diffusion_cond`; it is not the same thing as the vLLM-Omni pipeline name.

Minimal example:

```json
{
  "model_type": "diffusion_cond",
  "sample_rate": 44100,
  "audio_channels": 2,
  "model": {
    "io_channels": 64,
    "conditioning": {
      "configs": [
        {"id": "video_features", "config": {"dim": 1024}},
        {"id": "text_features", "config": {"dim": 1024}},
        {"id": "sync_features", "config": {"dim": 768}}
      ]
    },
    "diffusion": {
      "type": "dit",
      "diffusion_objective": "rectified_flow",
      "config": {}
    }
  }
}
```

`vllm-omni` resolves the pipeline itself through `model_index.json` with:

```json
{
  "_class_name": "PrismAudioPipeline",
  "_diffusers_version": "0.0.0"
}
```

## Request Contract

Each request must provide precomputed features under `prompt.additional_information`:

```python
prompt = {
    "prompt": "semantic and temporal cot text",
    "additional_information": {
        "video_features": video_features,
        "text_features": text_features,
        "sync_features": sync_features,
    },
}
```

For the base PrismAudio config, those required feature ids are
`video_features`, `text_features`, and `sync_features`. More generally, the
pipeline validates direct conditioning tensors against the official
`model.conditioning.configs[].id` entries from the supplied Prismaudio config.

Or point to a serialized fixture directly:

```python
prompt = {
    "prompt": "semantic and temporal cot text",
    "additional_information": {
        "conditioning_path": "/path/to/demo_features.npz",
    },
}
```

Supported fixture formats are `.npz`, `.pt`, and `.pth`. The file must decode to
a mapping containing at least `video_features`, `text_features`, and `sync_features`.

For follow-up runtime integration work, `vllm-omni` also accepts an optional
`conditioning_factory` in `model_config`. This hook is intended for adapters that
turn higher-level prompt payloads into Prismaudio feature tensors at request time.
The factory can be provided as:

- a Python callable taking `(prompt, runtime_config)`
- an import-path string
- a mapping spec with `path` and `input`

Example:

```python
def build_conditioning(prompt, runtime_config):
    return {
        "video_features": ...,
        "text_features": ...,
        "sync_features": ...,
    }

model_config = {
    "prismaudio_model_config_path": "/path/to/prismaudio.json",
    "conditioning_factory": build_conditioning,
}
```

This hook does not ship a default VideoPrism / Synchformer / T5 runtime stack yet.
It is the extension point for integrating those upstream preprocessors without
changing the core Prismaudio sampling pipeline again.

If you already have an upstream-style feature extractor object, `vllm-omni` also
accepts `feature_extractor_factory`. The constructed extractor is expected to expose
the official-style methods used by ThinkSound/PrismAudio preprocessing:

- `encode_t5_text(captions)`
- `encode_video_and_text_with_videoprism(clip_input, captions)`
- `encode_video_with_sync(sync_input)`

When this hook is used, each prompt can provide higher-level inputs such as:

```python
prompt = {
    "prompt": "fallback prompt text",
    "additional_information": {
        "caption_cot": "semantic and temporal cot text",
        "clip_chunk": clip_tensor_or_array,
        "sync_chunk": sync_tensor_or_array,
    },
}
```

The pipeline will batch those prompt-local inputs, invoke the feature extractor once,
and convert the outputs into `video_features`, `text_features`, and `sync_features`
before diffusion starts.

If your request starts from a higher-level video reference rather than precomputed
chunks, `model_config["video_preprocessor_factory"]` can run one stage earlier and
materialize prompt-local `clip_chunk` / `sync_chunk` inputs before feature extraction.
This hook is intended for adapters that consume fields like `video_path` and return:

```python
{
    "clip_chunk": clip_tensor_or_array,
    "sync_chunk": sync_tensor_or_array,
}
```

The runtime chain then becomes:

`video_preprocessor_factory -> feature_extractor_factory -> Prismaudio diffusion`

For a more declarative setup, `model_config["video_preprocessor_config"]` can trigger
the default official video preprocessing fallback. Minimal example:

```python
model_config = {
    "prismaudio_model_config_path": "/path/to/prismaudio.json",
    "video_preprocessor_config": {},
    "feature_extractor_config": {
        "vae_config": "/path/to/stable_audio_2_0_vae.json",
        "synchformer_ckpt": "/path/to/synchformer_state_dict.pth",
        "need_vae_encoder": False,
    },
}
```

This default path currently targets the official `app.extract_video_frames`
preprocessing entrypoint. If its dependencies are unavailable, the pipeline surfaces
an explicit video-preprocessor dependency error.

For a more declarative setup, `model_config["feature_extractor_config"]` can trigger
the default official `FeaturesUtils` builder. Minimal example:

```python
model_config = {
    "prismaudio_model_config_path": "/path/to/prismaudio.json",
    "feature_extractor_config": {
        "vae_config": "/path/to/stable_audio_2_0_vae.json",
        "synchformer_ckpt": "/path/to/synchformer_state_dict.pth",
        "need_vae_encoder": False,
    },
}
```

This default path currently targets the official
`data_utils.v2a_utils.feature_utils_288.FeaturesUtils` implementation. If its
dependencies are unavailable, the pipeline surfaces an explicit feature-extractor
dependency error rather than silently patching imports.

The pipeline validates these tensors before model execution:

- must be `torch.Tensor`
- must use floating-point dtype
- must have rank 2 or 3
- when a prompt-local tensor uses rank 3, its leading batch size must be `1`
- if the official config declares feature widths, the last dimension must match

Multi-prompt requests are expressed as a prompt list, where each prompt carries its
own `additional_information` mapping. `vllm-omni` batches those prompt-local tensors
before sampling.

## Local Smoke Test

The repo contains an env-driven Prismaudio e2e smoke test:

`tests/e2e/offline_inference/test_prismaudio_model.py`

Required environment variables:

- `PRISMAUDIO_E2E_CONFIG`
- `PRISMAUDIO_E2E_TRANSFORMER_CKPT`
- `PRISMAUDIO_E2E_VAE_CKPT`
- `PRISMAUDIO_E2E_FEATURES`

Optional environment variables:

- `PRISMAUDIO_E2E_NUM_STEPS`
- `PRISMAUDIO_E2E_CFG_SCALE`
- `PRISMAUDIO_E2E_SEED`
- `PRISMAUDIO_E2E_DTYPE`
- `PRISMAUDIO_E2E_VIDEO`
- `PRISMAUDIO_E2E_CAPTION_COT`
- `PRISMAUDIO_E2E_SYNCHFORMER_CKPT`
- `PRISMAUDIO_E2E_PREPROCESS_VAE_CONFIG`

Example:

```bash
export PRISMAUDIO_E2E_CONFIG=/path/to/prismaudio.json
export PRISMAUDIO_E2E_TRANSFORMER_CKPT=/path/to/prismaudio.ckpt
export PRISMAUDIO_E2E_VAE_CKPT=/path/to/vae.ckpt
export PRISMAUDIO_E2E_FEATURES=/path/to/demo_features.npz

python -m pytest tests/e2e/offline_inference/test_prismaudio_model.py -q -s
```

To smoke-test the default official preprocessing chain from `video_path`, also set:

```bash
export PRISMAUDIO_E2E_VIDEO=/path/to/demo.mp4
export PRISMAUDIO_E2E_CAPTION_COT="semantic and temporal cot text"
export PRISMAUDIO_E2E_SYNCHFORMER_CKPT=/path/to/synchformer_state_dict.pth
export PRISMAUDIO_E2E_PREPROCESS_VAE_CONFIG=/path/to/stable_audio_2_0_vae.json

python -m pytest tests/e2e/offline_inference/test_prismaudio_model.py -q -s \
  -k runtime_preprocessing
```

When the resources are present, the test prints a small benchmark line including:

- initialization time
- inference time
- number of steps
- cfg scale
- sample rate
- output audio shape
- peak memory

Example successful smoke result from local development:

```text
[PrismAudio E2E] init_s=7.60 inference_s=1.26 steps=4 cfg_scale=5.00 sample_rate=44100 audio_shape=(1, 2, 397312) peak_memory_mb=3765.90
```

## Notes

- The current offline smoke path is the recommended reviewer-facing validation path for checkpoint-backed PrismAudio integration.
- If the upstream official builder environment is incomplete, related tests skip or fail with explicit dependency / NumPy compatibility errors rather than silently patching imports.
