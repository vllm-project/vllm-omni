# Invoker-Render Interactive Racing Demo

Start the live proxy service using the [Invoker-Render recipe](../../../recipes/inclusionAI/Invoker-Render.md).
Prepare a 1344×768 reference image that matches the opening camera view of the racing scene, along with the frozen prompt.

Run the following commands from the repository root. Set `CWM_SOURCE` to a local checkout of the
[CodeWorldModel source](https://github.com/inclusionAI/CodeWorldModel/tree/86c952f6a37a82afbc391db2572b4d94ce332438).
The page uses the racing scene, camera, and canvas capture code from that revision.

```bash
CWM_SOURCE=/path/to/CodeWorldModel
DEMO_DIR="$PWD/logs/invoker-render/game"
mkdir -p "$DEMO_DIR"
ln -s "$PWD/examples/online_serving/invoker_render_game/index.html" "$DEMO_DIR/index.html"
ln -s "$PWD/apps/ComfyUI-vLLM-Omni/web/vllm-omni-mark.svg" "$DEMO_DIR/vllm-omni-mark.svg"
ln -s "$PWD/examples/online_serving/invoker_render_game/game.mjs" "$DEMO_DIR/game.mjs"
ln -s "$PWD/examples/online_serving/invoker_render_game/proxy-capture.mjs" "$DEMO_DIR/proxy-capture.mjs"
ln -s "$PWD/examples/online_serving/streaming_video_generation/video-stream-view.js" "$DEMO_DIR/video-stream-view.js"
ln -s "$CWM_SOURCE/src/sglang/projects/minimax_h3/web/playable" "$DEMO_DIR/codeworldmodel"
python -m http.server 18862 --bind 127.0.0.1 --directory "$DEMO_DIR"
```

Open `http://127.0.0.1:18862` and provide the WebSocket address, reference image, and prompt.
Start a session and click the proxy canvas. Press W to accelerate, A or D to steer, and S to brake.
The simulation pauses when input credit runs out or the number of frames awaiting playback reaches its limit.
Use the middle session button to finish and drain the stream. The client captures the final packet at a valid
input boundary. You can start another session when the server reports completion.

```bash
node --test tests/examples/test_invoker_render_proxy_capture.mjs
```
