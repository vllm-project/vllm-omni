# Qwen2.5-Omni

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/qwen2_5_omni>.


## Run examples (Qwen2.5-Omni)

### Multiple Prompts
Download dataset from [seed_tts](https://drive.google.com/file/d/1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP/edit). To get the prompt, you can:
```bash
tar -xf <Your Download Path>/seedtts_testset.tar
cp seedtts_testset/en/meta.lst examples/offline_inference/qwen2_5_omni/meta.lst
python3 examples/offline_inference/qwen2_5_omni/extract_prompts.py \
  --input examples/offline_inference/qwen2_5_omni/meta.lst \
  --output examples/offline_inference/qwen2_5_omni/top100.txt \
  --topk 100
```
Get into the example folder
```bash
cd examples/offline_inference/qwen2_5_omni
```
Then run the command below.
```bash
bash run_multiple_prompts.sh
```

### Single Prompt
Get into the example folder
```bash
cd examples/offline_inference/qwen2_5_omni
```
Then run the command below.
```bash
bash run_single_prompt.sh
```

### Modality control
If you want to control output modalities, e.g. only output text, you can run the command below:
```bash
python end2end.py --output-wav output_audio \
                  --query-type mixed_modalities \
                  --modalities text
```

#### Using Local Media Files
The `end2end.py` script supports local media files (audio, video, image) via CLI arguments:

```bash
# Use single local media files
python end2end.py --query-type use_image --image-path /path/to/image.jpg
python end2end.py --query-type use_video --video-path /path/to/video.mp4
python end2end.py --query-type use_audio --audio-path /path/to/audio.wav

# Combine multiple local media files
python end2end.py --query-type mixed_modalities \
    --video-path /path/to/video.mp4 \
    --image-path /path/to/image.jpg \
    --audio-path /path/to/audio.wav

# Use audio from video file
python end2end.py --query-type use_audio_in_video --video-path /path/to/video.mp4

```

If media file paths are not provided, the script will use default assets. Supported query types:
- `use_image`: Image input only
- `use_video`: Video input only
- `use_audio`: Audio input only
- `mixed_modalities`: Audio + image + video
- `use_audio_in_video`: Extract audio from video
- `text`: Text-only query

### FAQ

If you encounter error about backend of librosa, try to install ffmpeg with command below.
```
sudo apt update
sudo apt install ffmpeg
```

## Example materials

??? abstract "end2end.py"
    ``````py
    --8<-- "examples/offline_inference/qwen2_5_omni/end2end.py"
    ``````
??? abstract "extract_prompts.py"
    ``````py
    --8<-- "examples/offline_inference/qwen2_5_omni/extract_prompts.py"
    ``````
??? abstract "run_multiple_prompts.sh"
    ``````sh
    --8<-- "examples/offline_inference/qwen2_5_omni/run_multiple_prompts.sh"
    ``````
??? abstract "run_single_prompt.sh"
    ``````sh
    --8<-- "examples/offline_inference/qwen2_5_omni/run_single_prompt.sh"
    ``````
