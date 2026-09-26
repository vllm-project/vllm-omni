# FLUX.1-Kontext T5 encoder FP8 (experimental)

Enable online FP8 for the T5 encoder explicitly:

```python
from vllm_omni import Omni

omni = Omni(
    model="black-forest-labs/FLUX.1-Kontext-dev",
    quantization_config={"text_encoder_2": {"method": "fp8"}},
)
```

This converts T5 attention projections and FFN input projections at load time.
CLIP, T5 embeddings, relative position bias, norms and FFN output projections
keep their original precision. A global FP8 setting does not enable T5 FP8.
This path accepts an unquantized checkpoint and dynamic activation scaling only.

The initial checkpoint load still precedes conversion, so this does not promise
a reduction in peak loading memory. Tiny-encoder tests cover the conversion;
official Kontext image-editing quality, encoder TP and offload combinations
remain unvalidated.
