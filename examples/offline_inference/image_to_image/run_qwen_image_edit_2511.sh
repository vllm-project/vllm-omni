python image_edit.py \
    --model Qwen/Qwen-Image-Edit-2511 \
    --image qwen_bear.png \
    --prompt "Add a white art board written with colorful text 'vLLM-Omni' on grassland. Add a paintbrush in the bear's hands. position the bear standing in front of the art board as if painting" \
    --output output_image_edit.png \
    --num-inference-steps 50 \
    --cfg-scale 4.0 \
    --cache-backend  cache_dit \
