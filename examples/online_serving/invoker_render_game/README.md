# Invoker-Render 交互赛车

用 [Invoker-Render 配方](../../../recipes/inclusionAI/Invoker-Render.md) 启动实时代理服务。准备与赛车场景起始相机匹配的 1344×768 参考图和冻结提示词。

在仓库根目录运行以下命令。`CWM_SOURCE` 指向 [CodeWorldModel 源码](https://github.com/inclusionAI/CodeWorldModel/tree/86c952f6a37a82afbc391db2572b4d94ce332438) 的本地目录。页面直接使用该版本的赛车、相机和画布采集代码。

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

打开 `http://127.0.0.1:18862`，填写 WebSocket 地址、参考图与提示词。开局后点击代理画布，用 WASD 加速、转向、刹车。输入额度或待播放帧达到上限时，代理模拟等待播放器追上。点击「结束并排空」会在合法输入边界补采尾包。收到完成通知后可以重新开局。

```bash
node --test tests/examples/test_invoker_render_proxy_capture.mjs
```
