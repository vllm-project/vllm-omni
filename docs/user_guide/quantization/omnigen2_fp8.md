# OmniGen2 language encoder FP8

OmniGen2 supports opt-in dynamic online FP8 for the language decoder under
`mllm`. Use an explicit component map:

```python
from vllm_omni.entrypoints.omni import Omni

omni = Omni(
    model="OmniGen2/OmniGen2",
    quantization_config={"mllm": {"method": "fp8"}},
)
```

Add `"transformer": {"method": "fp8"}` to the map to select DiT FP8 as well.
An unspecified component stays unquantized. For compatibility, OmniGen2's
existing global FP8 flag continues to affect only the DiT; it does not opt the
HF language encoder into FP8.

The adapter replaces decoder-block linear layers and respects the backend's
ignored-layer rules. Vision, embeddings, the output head, and normalization
retain their original modules and precision. The HF encoder's activation dtype
is preserved around the FP8 GEMM, including when the checkpoint loads in FP32.
Serialized FP8 encoder checkpoints and static activation scaling are rejected.

The HF checkpoint is loaded before conversion. Smaller persistent weight
allocations do not imply a smaller loading peak, allocator reservation, or
process GPU footprint. Numerical outputs change with quantization; validate
quality for your inputs. The initial full-pipeline smoke coverage is single-GPU,
eager text-to-image generation; image editing, offload combinations and
parallel encoder execution need separate qualification.
