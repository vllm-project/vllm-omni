# MammothModa2-Preview

## Run examples (MammothModa2-Preview)

Download model
```bash
hf bytedance-research/MammothModa2-Preview --local-dir ./MammothModa2-Preview
```

### Text-to-Image (T2I)

```bash
python examples/offline_inference/mammothmodal2_preview/run_mammothmoda2_t2i.py \
  --model ./MammothModa2-Preview \
  --stage-config ./vllm_omni/model_executor/stage_configs/mammoth_moda2.yaml \
  --prompt "A stylish woman riding a motorcycle in NYC, movie poster style" \
  --height 1024 \
  --width 1024 \
  --num-inference-steps 50 \
  --text-guidance-scale 4.0 \
  --out output.png
```

### Image Summary

```bash
python examples/offline_inference/mammothmodal2_preview/run_mammothmoda2_image_summarize.py \
  --model ./MammothModa2-Preview \
  --stage-config ./vllm_omni/model_executor/stage_configs/mammoth_moda2_ar.yaml \
  --question "Summarize this image." \
  --image ./image.png
```
