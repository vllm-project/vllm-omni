# HiDream-O1-Image

Current support:
- text-to-image only (`full`, single GPU, `torch.bfloat16`)

Not supported yet:
- `dev`
- editing / IP / layout / skeleton
- TP / SP / CFG parallel / HSDP / Cache-DiT

Resolved from `config.json::architectures`; no `model_index.json` rewrite is needed.

```bash
python end2end.py --model HiDream-ai/HiDream-O1-Image --output hidream_o1_output.png
```

```bash
hf download HiDream-ai/HiDream-O1-Image --local-dir /workspace/.hf_models_cache/HiDream-O1-Image
python end2end.py --model /workspace/.hf_models_cache/HiDream-O1-Image --output hidream_o1_output.png
```
