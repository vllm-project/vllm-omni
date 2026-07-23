# Lingbot World Fast Offline Inference

Lingbot World Fast is an autoregressive diffusion model that uses a reference image, a text prompt and a set of camera positions to generate a video. The online serving model of this model adds a feature that is not implemented in the original model: video extension.

## Quickstart

The easiest way to launch a server running the Lingbot World Fast model is by using the script `examples/online_serving/lingbot_world_fast/run_server.sh`.

Once the server is launched, the client can send requests to its websocket at `/v1/realtime/world/camera`. The easiest way to interact with the server is using the script `examples/online_serving/lingbot_world_fast/openai_client.py`. Its command line options are described below.

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
| `num-calls`               | int   | 1                      | Makes an additional `num-calls - 1` video extension calls with `num_frames` frames         |
| `num-skip-frames`               | int   | 4                      | Extension calls have artifacts on the first couple frames. Discard them. |
| `session-id`               | str   | None                      | Session id to control whether to trigger a video extension call             |

## Video Extension

The idea of video extension is to allow the user to generate further frames for the same video efficiently. This is done by the vllm-omni implementation by storing the KV-cache of the generated video by default. This way, if the next request uses the same session-id, the pipeline will enter extension mode. So, the newly generated frames will use the previously generated frames as context. This is done by storing the KV-cache as mentioned above. No frame information, whether in latent space or RGB values, is kept in the server.

This feature is limited by the fact that the model has not been trained to perform this task. So, the steering capacity of the user is limited. Namely, the reference image and changes to the text prompt are ignored. The best tool the user has is to provide camera positions. In the end, video extension is more of a demonstration of the power and features of VLLM-Omni than of Lingbot World in itself.

## API

The server uses a websocket endpoint located at `/v1/realtime/world/camera`. It makes available two tasks: `infer` and `reset` which can be controlled by the "endpoint" key of the request.

By default, the server uses the `infer` task, which checks the `session-id` field and compares it to the one used on the last infer call. If they are the same, it triggers an extension call at the pipeline level. Note that only the KV-cache of the last request is stored to mitigate Out of Memory problems at the GPU level. Otherwise, it generates the video from scratch. Notice that when doing an extension task, no reference image should be provided (it would be ignored anyway).

The `reset` endpoint does not immediately evict the KV cache in the GPU, but instead it forces a reset on the next `infer` call independently of the value of `session-id`.

The endpoint sends the resulting frames in groups of 4 to mitigate package loss problems. It is the client's role to concatenate the different frames to obtain the final video.

## Example materials

??? abstract "run_server.sh"
    ``````sh
    --8<-- "examples/online_serving/lingbot_world_fast/run_server.sh"
    ``````
??? abstract "openai_client.py"
    ``````sh
    --8<-- "examples/online_serving/lingbot_world_fast/openai_client.py"
    ``````
