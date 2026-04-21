# OmniWeaving Offline Inference Example

This directory contains end-to-end examples of running offline inference with the OmniWeaving (Qwen2.5-VL + HunyuanVideo 1.5) model using the `vllm-omni` unified API.

## Text-to-Video (T2V) Generation
To generate a video from a text prompt only, run:

```bash
python end2end.py \
    --model "Tencent-Hunyuan/OmniWeaving" \
    --prompt "A cute bear wearing a Christmas hat moving naturally." \
    --tensor-parallel-size 1 \
    --output "t2v_output.mp4"
```

Image-to-Video (I2V) Generation
To generate a video conditioned on an input image, provide the --image-path argument:

```bash
python end2end.py \
    --model "Tencent-Hunyuan/OmniWeaving" \
    --prompt "A cute bear wearing a Christmas hat moving naturally." \
    --image-path "path/to/your/reference_image.jpg" \
    --tensor-parallel-size 1 \
    --output "i2v_output.mp4"
```
