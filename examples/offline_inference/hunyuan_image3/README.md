# HunyuanImage-3.0 Image-to-Text Inference

This example demonstrates how to run HunyuanImage-3.0 Image-to-Text with the vLLM-Omni.

## Local CLI Usage

Download the example image:

```bash
wget https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg
```

Run example:

```bash
python image_to_text.py \
  --image cherry_blossom.jpg \
  --prompt "<|startoftext|>You are an assistant that understands images and outputs text.<img>Describe the content of the picture."
```

Key arguments:

- `--model`: Model used. Default is: tencent/HunyuanImage-3.0-Instruct (Optional).
- `--image`: Path to input image (required).
- `--prompt`: Text description used to guide image understanding (required).
