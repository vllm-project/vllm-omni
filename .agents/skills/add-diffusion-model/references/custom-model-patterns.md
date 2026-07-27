# Custom Model Patterns Reference

Patterns for adding models that don't come from the standard diffusers pipeline format.

## Directory Structure Comparison

### Diffusers-based model (e.g., Wan2.2)

```
vllm_omni/diffusion/models/wan2_2/
├── __init__.py                    # Exports pipeline + transformer + helpers
├── pipeline_wan2_2.py             # Pipeline: loads components via from_pretrained()
├── pipeline_wan2_2_i2v.py         # Variant pipeline for image-to-video
└── wan2_2_transformer.py          # Transformer: ported from diffusers, uses Attention layer
```

The transformer is loaded separately via `weights_sources` + `load_weights()`. Non-transformer components (VAE, text encoder) are loaded in `__init__` via `from_pretrained()`.

### Custom model with external deps (e.g., DreamID-Omni)

```
vllm_omni/diffusion/models/dreamid_omni/
├── __init__.py                    # Exports pipeline only
├── pipeline_dreamid_omni.py       # Pipeline: loads ALL weights in __init__ via custom helpers
├── fusion.py                      # Custom fusion architecture (video + audio cross-attention)
└── wan2_2.py                      # Re-implemented Wan backbone with split API

examples/offline_inference/x_to_video_audio/
└── download_dreamid_omni.py       # Downloads weights from 3 HF repos + clones code repo
```

All weights loaded eagerly in `__init__`. `load_weights()` is a no-op. External dependency (`dreamid_omni` package) imported with try/except.

### Custom model with ported code (e.g., BAGEL)

```
vllm_omni/diffusion/models/bagel/
├── __init__.py
├── pipeline_bagel.py              # Pipeline: instantiates models, uses weights_sources
├── bagel_transformer.py           # Full LLM backbone (Qwen2-MoT) ported into vllm-omni
└── autoencoder.py                 # Custom VAE ported from original repo
```

Model code is fully ported (no external dependency). Uses `weights_sources` and `load_weights()` with custom name remapping to handle non-diffusers safetensors format.

## Weight Loading Patterns

### Pattern 1: Standard diffusers flow (Wan2.2, Z-Image, FLUX)

```
init → create transformer (empty) → set weights_sources → [loader calls load_weights()]
```

- `weights_sources` points to safetensors in HF subfolder (e.g., `transformer/`)
- `load_weights()` receives `(name, tensor)` pairs from the loader
- Name remapping handles diffusers→vllm-omni differences (QKV fusion, Sequential index removal)

### Pattern 2: Custom safetensors at root (BAGEL)

```
init → create all models (empty) → set weights_sources(subfolder=None) → [loader calls load_weights()]
```

- `weights_sources` points to **root** of model directory, not a subfolder
- Weights have non-diffusers names (e.g., `bagel.language_model.model.layers.0.self_attn.q_proj.weight`)
- `load_weights()` does heavy name normalization

```python
self.weights_sources = [
    DiffusersPipelineLoader.ComponentSource(
        model_or_path=od_config.model,
        subfolder=None,      # root directory
        prefix="",           # no prefix stripping
        fall_back_to_pt=False,
    )
]
```

### Pattern 3: Fully custom loading (DreamID-Omni)

```
init → load ALL weights eagerly via custom helpers → load_weights() = no-op
```

- No `weights_sources` attribute — standard loader finds nothing to iterate
- Custom init functions (e.g., `init_wan_vae_2_2()`, `load_fusion_checkpoint()`) handle downloading and loading
- `load_weights()` is `pass`
- Weights may come from multiple HF repos in different formats (`.pth`, `.safetensors`)

Use this when:
- The original model has complex, well-tested loading code you don't want to rewrite
- Weights span multiple HF repos
- Weight format is non-standard (e.g., a single `.pth` file, not sharded safetensors)

## model_index.json for Custom Models

Standard diffusers `model_index.json`:
```json
{
    "_class_name": "WanPipeline",
    "_diffusers_version": "0.35.0.dev0",
    "scheduler": ["diffusers", "UniPCMultistepScheduler"],
    "transformer": ["diffusers", "WanTransformer3DModel"],
    "vae": ["diffusers", "AutoencoderKLWan"]
}
```

Custom model `model_index.json` (minimal):
```json
{
    "_class_name": "DreamIDOmniPipeline",
    "fusion": "DreamID-Omni/dreamid_omni.safetensors"
}
```

The only **required** field is `_class_name` — it must match a key in `_DIFFUSION_MODELS` in `registry.py`. Other fields are model-specific and accessible via `od_config.model_config` dict.

## External Dependency Management

### Git clone + .pth injection (DreamID-Omni pattern)

```python
def download_dependency():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOCK_FILE, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        if not DEPENDENCY_DIR.exists():
            subprocess.run([
                "git", "clone", "--depth", "1",
                REPO_URL, "--branch", BRANCH,
                str(DEPENDENCY_DIR)
            ], check=True)
        fcntl.flock(f, fcntl.LOCK_UN)

    # Add to Python path via .pth file
    site_packages = Path(site.getsitepackages()[0])
    pth_file = site_packages / "vllm_omni_dependency.pth"
    pth_file.write_text(str(DEPENDENCY_DIR))
```

### Direct port (BAGEL pattern)

Copy essential files from the original repo into `vllm_omni/diffusion/models/<name>/`. Adapt imports to use vllm-omni utilities. Benefits: no external dependency, no git clone step. Drawback: must maintain the ported code.

## Multi-Modal Input/Output Protocols

Custom models that handle images, audio, or video I/O should implement protocol classes:

```python
from vllm_omni.diffusion.models.interface import (
    SupportImageInput,    # Model accepts image input
    SupportAudioInput,    # Model accepts audio input
    SupportAudioOutput,   # Model produces audio output
)

class MyPipeline(nn.Module, SupportImageInput, SupportAudioInput, SupportAudioOutput):
    pass  # Protocol markers enable proper engine routing
```

The engine checks `isinstance(pipeline, SupportImageInput)` at startup to configure input validation and warmup behavior.

## Hardcoded Config vs Config Files

Diffusers models use `config.json` in each subfolder. Custom models often use:

**Module-level config dicts** (DreamID-Omni):
```python
VIDEO_CONFIG = {
    "patch_size": [1, 2, 2], "model_type": "ti2v",
    "dim": 3072, "ffn_dim": 14336, "num_heads": 24, "num_layers": 30, ...
}
```

**Loaded from custom JSON** (BAGEL):
```python
cfg_path = os.path.join(model_path, "config.json")
with open(cfg_path) as f:
    bagel_cfg = json.load(f)
vae_cfg = bagel_cfg.get("vae_config", {})
```

## Custom Architecture Patterns

### Split forward API (DreamID-Omni)

When a fusion model needs to interleave blocks from two backbones:

```python
class WanModel(nn.Module):
    def prepare_transformer_block_kwargs(self, x, t, context, ...):
        # Patch embed, time embed, text embed, RoPE
        return x, e, kwargs

    def post_transformer_block_out(self, x, grid_sizes, e):
        # Output projection, unpatchify
        return output

    def forward(self, *args, **kwargs):
        raise NotImplementedError  # Fusion model handles block iteration
```

The `FusionModel` then iterates blocks in lock-step:
```python
for video_block, audio_block in zip(self.video_model.blocks, self.audio_model.blocks):
    video_out = video_block(video_hidden, ...)
    audio_out = audio_block(audio_hidden, ...)
    # Cross-attend between modalities
    video_out = cross_attention(video_out, audio_out)
    audio_out = cross_attention(audio_out, video_out)
```

### LLM-as-denoiser (BAGEL)

When the backbone is a language model that also does diffusion:

```python
class BagelModel(nn.Module):
    def __init__(self):
        self.language_model = Qwen2MoTForCausalLM(config)
        self.vit_model = SiglipVisionModel(vit_config)
```

The LLM processes both text tokens and latent image tokens in a single forward pass, using KV caching for the text portion.

## Pre/Post Processing for Custom Models

Custom models typically handle pre/post processing **inside `forward()`** rather than via registered functions, because the logic is tightly coupled:

```python
def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
    # Inline preprocessing
    image = self._load_and_resize_image(req.prompts[0].get("multi_modal_data", {}).get("image"))
    image_latent = self._vae_encode(image)

    # ... denoising loop ...

    # Inline postprocessing
    pil_image = self._decode_to_pil(latents)
    return DiffusionOutput(output=[pil_image])
```

If pre/post functions are not registered in `_DIFFUSION_PRE_PROCESS_FUNCS` / `_DIFFUSION_POST_PROCESS_FUNCS`, the engine simply skips those steps.

## Download Script Template

```python
# examples/offline_inference/<name>/download_<name>.py
from huggingface_hub import snapshot_download
import json, os

def main(output_dir):
    # Download model weights from HF
    snapshot_download(repo_id="org/model-weights", local_dir=os.path.join(output_dir, "weights"))

    # Download additional components if from separate repos
    snapshot_download(repo_id="org/vae-weights", local_dir=os.path.join(output_dir, "vae"),
        allow_patterns=["*.safetensors"])

    # Generate model_index.json
    config = {"_class_name": "YourPipeline", "custom_key": "weights/model.safetensors"}
    with open(os.path.join(output_dir, "model_index.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Install external code dependency (if needed)
    download_dependency()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./your_model")
    args = parser.parse_args()
    main(args.output_dir)
```
