# MammothModa2-Preview

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/mammothmodal2_preview>.


## Run examples (MammothModa2-Preview)

Download model
```bash
hf download bytedance-research/MammothModa2-Preview --local-dir ./MammothModa2-Preview
```

### Text-to-Image (T2I)

Text-to-image now runs through the shared offline image example
(`examples/offline_inference/text_to_image/text_to_image.py`). See the recipe
`recipes/MammothModa2/MammothModa2-Preview.md` for the full command and the
`extra_body` knobs (`text_guidance_scale`, `cfg_range`, `num_inference_steps`).

### Image Summary

```bash
python examples/offline_inference/mammothmodal2_preview/run_mammothmoda2_image_summarize.py \
  --model ./MammothModa2-Preview \
  --stage-config ./vllm_omni/model_executor/stage_configs/mammoth_moda2_ar.yaml \
  --question "Summarize this image." \
  --image ./image.png
```

## Example materials

??? abstract "run_mammothmoda2_image_summarize.py"
    ``````py
    --8<-- "examples/offline_inference/mammothmodal2_preview/run_mammothmoda2_image_summarize.py"
    ``````
