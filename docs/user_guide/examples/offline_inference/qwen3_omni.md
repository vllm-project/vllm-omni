# Qwen3-Omni

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/qwen3_omni>.


## Run examples (Qwen3-Omni)

### Multiple Prompts
Download dataset from [seed_tts](https://drive.google.com/file/d/1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP/edit). For processing dataset please refer to [Qwen2.5-Omni README.md](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/qwen2_5_omni/README.md)
Get into the example folder
```bash
cd examples/offline_inference/qwen3_omni
```
Then run the command below.
```bash
bash run_multiple_prompts.sh
```
### Single Prompt
Get into the example folder
```bash
cd examples/offline_inference/qwen3_omni
```
Then run the command below.
```bash
bash run_single_prompt.sh
```
If you have not enough memory, you can set thinker with tensor parallel. Just run the command below.
```bash
bash run_single_prompt_tp.sh
```

### Modality control
If you want to control output modalities, e.g. only output text, you can run the command below:
```bash
python end2end.py --output-wav output_audio \
                  --query-type use_audio \
                  --modalities text
```

#### Using Local Media Files
The `end2end.py` script supports local media files (audio, video, image) via command-line arguments:

```bash
# Use local video file
python end2end.py --query-type use_video --video-path /path/to/video.mp4

# Use local image file
python end2end.py --query-type use_image --image-path /path/to/image.jpg

# Use local audio file
python end2end.py --query-type use_audio --audio-path /path/to/audio.wav
```

If media file paths are not provided, the script will use default assets. Supported query types:
- `use_video`: Video input
- `use_image`: Image input
- `use_audio`: Audio input
- `text`: Text-only query


## Example materials

??? abstract "end2end.py"
    ``````py
    --8<-- "examples/offline_inference/qwen3_omni/end2end.py"
    ``````
??? abstract "run_multiple_prompts.sh"
    ``````sh
    --8<-- "examples/offline_inference/qwen3_omni/run_multiple_prompts.sh"
    ``````
??? abstract "run_single_prompt.sh"
    ``````sh
    --8<-- "examples/offline_inference/qwen3_omni/run_single_prompt.sh"
    ``````
??? abstract "run_single_prompt_tp.sh"
    ``````sh
    --8<-- "examples/offline_inference/qwen3_omni/run_single_prompt_tp.sh"
    ``````
