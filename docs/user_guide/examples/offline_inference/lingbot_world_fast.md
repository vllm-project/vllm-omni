# Lingbot World Fast Offline Inference

Lingbot World Fast is an autoregressive diffusion model that uses a reference image, a text prompt and a set of camera positions to generate a video.

## Video Generation

First, download the model weights using `examples/offline_inference/lingbot_world_fast/download_lingbot_world_fast.py`.

The simplest way to run offline generation is to use the script on `examples/offline_inference/lingbot_world_fast/end2end.py`. The core of this script is done by:

```python
from vllm_omni.entrypoints.omni import Omni

if __name__ == "__main__":
    omni = Omni(model="lingbot_world/lingbot-world-base-cam/Lingbot-World-Fast", model_class_name="LingbotWorldFastPipeline")
    outputs = omni.generate(
        {
            "prompt": "A journey along the Great Wall of China",
            "multi_modal_data": {
                "image": "input.png",
                "camera": {
                    "poses": np.load("path/to/poses.npy")
                    "intrinsics": np.load("path/to/intrinsics.npy")
                }
            },
        },
        OmniDiffusionSamplingParams(
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=fps,
        ),
    )
    export_to_video(outputs[0], "output.png")
```

## Generation Parameters

| Parameter             | Type  | Default | Description                         |
| --------------------- | ----- | ------- | ----------------------------------- |
| `height`              | int   | None (computed from image)    | Image height in pixels              |
| `width`               | int   | None (computed from image)    | Image width in pixels               |
| `num_frames`          | int   | 81                            | Number of frames to generate        |
| `fps`                 | int   | 16                            | Frames per second                   |
| `seed`                | int   | 42                            | Optional random seed                |
| `prompt`              | str   | ""                            | Text prompt                         |
| `negative_prompt`     | str   | None                          | Negative prompt                     |
| `image`               | str   | Required                      | Path to reference image             |
|`camera-path`          | str   | Required                      | Path to folder with `poses.npy` and `intrinsics.npy`|
