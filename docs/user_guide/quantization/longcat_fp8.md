# LongCat text encoder FP8

These pipelines opt in to dynamic online FP8 with an explicit `text_encoder`
entry. A global `quantization="fp8"` does **not** enable their HF encoder.

```python
from vllm_omni.entrypoints.omni import Omni

omni = Omni(
    model="meituan-longcat/LongCat-Image",  # or meituan-longcat/LongCat-Image-Edit
    quantization_config={"text_encoder": {"method": "fp8"}},
)
```

Only the Qwen2.5-VL language decoder's attention and MLP linear layers are
quantized. The vision tower, embeddings, normalization and output head retain
their original precision. This setting does not enable DiT or VAE quantization.
For Image-Edit it also applies to language layers processing image-conditioned
features; for Image it also affects prompt rewriting when enabled.

Use an unquantized checkpoint and dynamic activations. Serialized FP8
checkpoints, static activation scales and other encoder quantization methods
are rejected. Hugging Face loads the original encoder before conversion, so
this reduces resident language weights, not the initial encoder loading peak.
The encoder remains replicated; this option does not add encoder tensor
parallelism. Compare image quality on your prompts before enabling it in
production. For controlled comparisons, disable prompt rewriting in both runs
with `OmniDiffusionSamplingParams(extra_args={"enable_prompt_rewrite": False})`.
