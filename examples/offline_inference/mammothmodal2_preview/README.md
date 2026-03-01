# MammothModa2-Preview

## Run examples (MammothModa2-Preview)

Get into the example folder
```bash
cd examples/offline_inference/mammothmodal2_preview
```

Download model
```bash
hf bytedance-research/MammothModa2-Preview --local-dir ./MammothModa2-Preview
```

### Text-to-Image (T2I)

```bash
python run_mammothmoda2_t2i.py \
  --model ./MammothModa2-Preview \
  --stage-config ./mammoth_moda2_t2i.yaml \
  --prompt "A stylish woman riding a motorcycle in NYC, movie poster style" \
  --height 1024 \
  --width 1024 \
  --num-inference-steps 50 \
  --text-guidance-scale 4.0 \
  --out output.png
```

### Image Summary

```bash
python run_mammothmoda2_image_summarize.py \
  --model ./MammothModa2-Preview \
  --stage-config ./mammoth_moda2_image_summarize.yaml \
  --question "Summarize this image." \
  --image ./image.png
```
