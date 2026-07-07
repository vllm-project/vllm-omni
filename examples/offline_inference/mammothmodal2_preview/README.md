# MammothModa2-Preview

> Text-to-image (T2I) generation has moved to the shared offline image example.
> See the recipe: [`recipes/MammothModa2/MammothModa2-Preview.md`](../../../recipes/MammothModa2/MammothModa2-Preview.md).
>
> This directory retains the image-understanding (image → text) example below,
> which is outside the scope of the diffusion `model_extras` example migration
> (#4548 / RFC #4539: AR / Omni understanding examples keep dedicated scripts).

## Run example (MammothModa2-Preview)

Download model
```bash
hf download bytedance-research/MammothModa2-Preview --local-dir ./MammothModa2-Preview
```

### Image Summary

```bash
python examples/offline_inference/mammothmodal2_preview/run_mammothmoda2_image_summarize.py \
  --model ./MammothModa2-Preview \
  --stage-config ./vllm_omni/model_executor/stage_configs/mammoth_moda2_ar.yaml \
  --question "Summarize this image." \
  --image ./image.png
```
