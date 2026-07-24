# HunyuanImage-3.0 Image-to-Text Inference

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/hunyuan_image3>.


This example demonstrates how to run HunyuanImage-3.0 comprehension
(image-to-text / text-to-text) with vLLM-Omni. The image-generating modalities
(text-to-image, image-editing) run through the shared task examples — see the
example directory's README for `--extra-body` recipes.

## Local CLI Usage

Download the example image:

```bash
wget https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg
```

Run example (image-to-text):

```bash
python run_hunyuan_image3_understanding.py \
  --modality image2text \
  --image cherry_blossom.jpg \
  --prompt "Describe the content of the picture."
```

Key arguments:

- `--model`: Model used. Default is: tencent/HunyuanImage-3.0-Instruct (Optional).
- `--modality`: `image2text` or `text2text` (Optional, default `image2text`).
- `--image`: Path to input image (required for `image2text`).
- `--prompt`: Text prompt / question (required).

## Example materials

??? abstract "run_hunyuan_image3_understanding.py"
    ``````py
    --8<-- "examples/offline_inference/hunyuan_image3/run_hunyuan_image3_understanding.py"
    ``````
