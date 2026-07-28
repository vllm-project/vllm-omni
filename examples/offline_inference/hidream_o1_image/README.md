# HiDream-O1-Image

Current support:
- text-to-image only (single GPU, `torch.bfloat16`)

Not supported yet:
- editing / IP / layout / skeleton
- TP / SP / CFG parallel / HSDP / Cache-DiT

This integration currently follows the default upstream text-to-image recipe.
It does not expose recipe selectors such as `model_type` or `variant`.

The checkpoint is identified by its HiDream-O1 pixel prediction head. This keeps ordinary `Qwen3VLForConditionalGeneration` checkpoints out of the diffusion registry.

```bash
python end2end.py --model HiDream-ai/HiDream-O1-Image --output hidream_o1_output.png
```

```bash
hf download HiDream-ai/HiDream-O1-Image --local-dir /workspace/.hf_models_cache/HiDream-O1-Image
python end2end.py --model /workspace/.hf_models_cache/HiDream-O1-Image --output hidream_o1_output.png
```
