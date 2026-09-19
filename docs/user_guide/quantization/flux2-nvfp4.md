# FLUX.2-dev NVFP4 prototype

This initial path loads the static-scale `flux2-dev-nvfp4.safetensors` from
`black-forest-labs/FLUX.2-dev-NVFP4`. It preserves the checkpoint's BF16 layers
and routes only the 144 declared NVFP4 projections through `comfy-kitchen`.

Requires Blackwell, `comfy-kitchen`, TP=1, and local unquantized FLUX.2-dev
components. Blackwell execution and generated-image quality have **not** been
validated. The mixed/dynamic-scale checkpoint, TP>1, CPU/layerwise offload,
compilation and CUDA graphs are outside this initial validation scope.

Prepare a separate model directory (the base components are symlinked):

```bash
python tools/prepare_flux2_nvfp4.py \
  --base-model /models/FLUX.2-dev \
  --checkpoint /models/flux2-dev-nvfp4.safetensors \
  --output /models/FLUX.2-dev-prepared-nvfp4
```

Load the explicit component configuration:

```python
import json
from pathlib import Path
from vllm_omni import Omni

model = Path("/models/FLUX.2-dev-prepared-nvfp4")
config = json.loads((model / "transformer/quantization_config.json").read_text())
omni = Omni(
    model=str(model),
    dtype="bfloat16",
    enforce_eager=True,
    quantization_config={"transformer": config},
)
```

The converter maps BFL parameter names, retains fused QKV/MLP tensors, swaps
the final AdaLN scale/shift halves, and preserves packed weight bytes and FP8
block scales. It does not dequantize and requantize the checkpoint. The initial
loader uses the same TensorCoreNVFP4Layout representation as the upstream
Comfy reference; it does not treat the serialized checkpoint as ModelOpt data.

Keep this experimental until real Blackwell output comparisons are complete.
