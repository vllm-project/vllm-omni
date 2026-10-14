# Qwen-Image-2.1 shared residual gates

Each Transformer forward shares modulation values across its blocks. The
optional `qwen21_share_residual_gates` setting computes `tanh` and condition-row
selection for the two residual gates once per forward, then passes those
activated gates into the blocks. Scale modulation, residual multiplication,
and addition retain the original implementation and rounding order.

```python
from vllm_omni.entrypoints.omni import Omni

omni = Omni(
    model="Qwen/Qwen-Image-2.1",
    additional_config={"qwen21_share_residual_gates": True},
)
```

The setting defaults to `false` and requires a Boolean. Gates are local to
each forward, including the CUDA decode graph body; they are not cached between timesteps, requests, or CFG branches.
The option is independent of QKV fusion and does not require MindIE-SD.
This changes no AdaLN operator integration.

The CPU regression covers FP32/BF16, batch and condition-row selection, both
residual branches, eager prefill/decode, the production graph body on CPU,
and successive forwards:

```bash
python -m pytest -q \
  tests/diffusion/models/qwen_image_21/test_shared_gates.py
```

Full-model quality, CUDA capture/replay, compile, TP/SP, and Turbo-checkpoint validation are
pending. Historical small latency changes are not a guaranteed speedup for
the rebased code. Set the option to `false` in a fresh engine to restore the
original path.
