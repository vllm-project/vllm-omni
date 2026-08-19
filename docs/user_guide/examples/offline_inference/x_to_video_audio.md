# X-To-Video-Audio

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/x_to_video_audio>.


The shared runner generates video with optional audio from text, image, and audio
inputs. It can be extended through `vllm_omni.model_extras` without adding
another model-specific runner.

## Shared request and output contract

The example builds one canonical request envelope: `prompt` is text,
`modalities` is `["video"]`, and optional `multi_modal_data` contains RGB PIL
images plus audio `(waveform, sample_rate)` tuples. Audio files are loaded in
full by design; the runner does not apply a time window. Prepare a trimmed input
file explicitly when only part of a recording should condition the model.
DreamID-Omni's audio conditioning sequence grows with the complete input
duration, so long recordings increase denoising memory and step time.

Model-specific request translation (for example, DreamID-Omni's two negative
prompts and its 16 kHz input requirement) is declared in `model_extras`. Model
outputs must provide `fps` and, when audio is present, `audio_sample_rate`.
`--fps` and `--audio-sample-rate` are available as explicit metadata overrides.
The shared encoder normalizes supported video tensor layouts, so this runner
does not branch on a model name or assume a model-specific frame layout.

## Local CLI Usage
### Download the Model locally
Since DreamID-Omni combine multiple models, and without any config, so we need to download them locally.

```bash
python download_dreamid_omni.py --output-dir ./dreamid_omni
```
After download, the model directory will look like this:

```
dreamid_omni/
├── DreamID-Omni/
│   ├── dreamid_omni.safetensors
├── MMAudio/
│   ├── ext_weights/
│   │   ├── best_netG.pt
│   │   ├── v1-16.pth
├── Wan2.2-TI2V-5B/
│   ├── google/*
│   ├── models_t5_umt5-xxl-enc-bf16.pth
│   ├── Wan2.2_VAE.pth
│
├── model_index.json # create by download_dreamid_omni.py
```

### Run the Inference
```python
python x_to_video_audio.py \
  --model /path/to/dreamid_omni \
  --prompt "Two people walking together and singing happily" \
  --image-path ./example0.png ./example1.png \
  --audio-path ./example0.wav ./example1.wav \
  --video-negative-prompt "jitter, bad hands, blur, distortion" \
  --audio-negative-prompt "robotic, muffled, echo, distorted" \
  --cfg-parallel-size 2 \
  --num-inference-steps 45 \
  --height 704 \
  --width 1280 \
  --output out_dreamid_omni_twoip.mp4
```
In the current test scenario (2 images + 2 audio inputs), the VRAM requirement is 72GB, regardless of whether cfg-parallel is enabled or disabled.
The VRAM usage can be reduced by enabling CPU offload via --enable-cpu-offload.
For multi-GPU memory reduction on the fused DreamID-Omni transformer, you can also enable HSDP:

```python
python x_to_video_audio.py \
  --model /path/to/dreamid_omni \
  --prompt "Two people walking together and singing happily" \
  --image-path ./example0.png ./example1.png \
  --audio-path ./example0.wav ./example1.wav \
  --video-negative-prompt "jitter, bad hands, blur, distortion" \
  --audio-negative-prompt "robotic, muffled, echo, distorted" \
  --cfg-parallel-size 2 \
  --num-inference-steps 45 \
  --height 704 \
  --width 1280 \
  --use-hsdp \
  --hsdp-shard-size 2 \
  --output out_dreamid_omni_twoip.mp4
```


You could take reference images/audios from the test cases in the official repo: https://github.com/Guoxu1233/DreamID-Omni

For example, single IP ref resources can be found under https://github.com/Guoxu1233/DreamID-Omni/tree/main/test_case/oneip, you could download them correspondingly to your local and use them for testing.

```python
# Example usage for oneip, ref media from the official repo DreamID-Omni
python x_to_video_audio.py \
  --model /path/to/dreamid_omni \
  --prompt "<img1>: In the frame, a woman with black long hair is identified as <sub1>.\n**Overall Environment/Scene**: A lively open-kitchen café at night; stove flames flare, steam rises, and warm pendant lights swing slightly as staff move behind her. The shot is an upper-body close-up.\n**Main Characters/Subjects Appearance**: <sub1> is a young woman with thick dark wavy hair and a side part. She wears a fitted black top under a light apron, a thin gold chain necklace, and small stud earrings.\n**Main Characters/Subjects Actions**: <sub1> tastes the sauce with a spoon, then turns her face toward the camera while still holding the spoon, her expression shifting from focused to conflicted.\n<sub1> maintains eye contact, swallows as if choosing her words, and says, <S>I keep telling myself I’m fine,but some nights it feels like I’m just performing calm.<E>" \
  --image-path 9.png \
  --audio-path 9.wav \
  --video-negative-prompt "jitter, bad hands, blur, distortion" \
  --audio-negative-prompt "robotic, muffled, echo, distorted" \
  --cfg-parallel-size 2 \
  --num-inference-steps 45 \
  --height 704 \
  --width 1280 \
  --output out_dreamid_omni_oneip.mp4
```


Key arguments:
- `--prompt`: text description (string).
- `--model`: path to the model local directory.
- `--model-class-name`: optional diffusion pipeline class override.
- `--height/--width`: output resolution (defaults 704 * 1280).
- `--image-path`: path to the input image list.
- `--audio-path`: path to the input audio list, indicate the timbre of the output video.
- `--cfg-parallel-size`: number of parallel cfg parallel (defaults 1).
- `--use-hsdp`: enable HSDP weight sharding for DreamID-Omni fused blocks.
- `--hsdp-shard-size`: number of GPUs used for HSDP sharding.
- `--hsdp-replicate-size`: number of HSDP replica groups.
- `--num-inference-steps`: number of denoising steps (defaults 45).
- `--video-negative-prompt`: negative prompt for video generation.
- `--audio-negative-prompt`: negative prompt for audio generation.
- `--extra-body`: JSON object containing parameters declared by the selected model.
- `--fps`: explicit output FPS override when a model does not return that metadata.
- `--audio-sample-rate`: explicit output audio sample-rate override.
- `--enable-cpu-offload`: enable CPU offload (defaults False).

## Example materials

??? abstract "download_dreamid_omni.py"
    ``````py
    --8<-- "examples/offline_inference/x_to_video_audio/download_dreamid_omni.py"
    ``````
??? abstract "x_to_video_audio.py"
    ``````py
    --8<-- "examples/offline_inference/x_to_video_audio/x_to_video_audio.py"
    ``````
