# Qwen2.5-Omni: Offline inference

## Setup
Please refer to the [stage configuration documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/stage_configs/) to configure memory allocation appropriately for your hardware setup.

## Run examples

### Multiple Prompts
Get into the example folder
```bash
cd examples/offline_inference/qwen2_5_omni
```
Then run the command below. Note: for processing large volume data, it uses py_generator mode, which will return a python generator from Omni class.
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

### Composable parallelism (strategy configs)

You can shard or replicate a stage with a small composable-parallel
`strategy.yaml` overlaid onto the **bundled default deploy config** -- no bespoke
deploy YAML required. Supply the strategy with `--strategy-config` and any matching
device layout via `--stage-overrides` (explicit kwargs forwarded to `Omni`).

Both examples below require >= 3 GPUs (the thinker uses 2, talker + code2wav share a third):

```bash
# Tensor-parallel the thinker (TP=2)
python end2end.py --query-type text --num-prompts 6 \
    --strategy-config strategy_tp2.yaml \
    --stage-overrides '{"0": {"devices": "0,1"}, "1": {"devices": "2"}, "2": {"devices": "2"}}'

# Replicate the thinker across 2 engines (round-robin load balancing)
python end2end.py --query-type text --num-prompts 6 \
    --strategy-config strategy_stage_replica.yaml \
    --stage-overrides '{"0": {"devices": "0,1"}, "1": {"devices": "2"}, "2": {"devices": "2"}}'
```
