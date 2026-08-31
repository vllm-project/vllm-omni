# Demo 使用说明

## 启动方式
```bash
python examples/online_serving/minicpmo/gradio_demo.py \
  --minicpmo45-api-base http://localhost:8091/v1 \
  --minicpmo45-model openbmb/MiniCPM-o-4_5 \
  --port 7862 --host 0.0.0.0
```

## 访问方式
浏览器打开 http://<host>:7862

## 交互流程
1. 文本输入:直接输入文字;勾选 TTS 可听语音回复(零样本音色克隆)
2. 语音输入:上传音频文件
3. 图像输入:上传图片
4. 视频输入:上传视频

## 局域网录制方案(麦克风需要 secure context)
盒端仅开放给转发机时,用 SSH 隧道 + 自签 TLS 暴露 gradio:

```bash
# 盒端:gradio 绑定 7862
# 转发机:/etc/ssh/sshd_config 设 GatewayPorts yes,然后
ssh -N -L 0.0.0.0:7862:<盒内网IP>:7862 user@box
# 录制机浏览器访问 https://<转发机IP>:7862(自签证书,需手动信任)
```

注意:
- 模型参数必须用完整本地路径或 served-model-name,否则 404
- macOS 浏览器不认 clientspecified 类型的证书,需导入系统钥匙串

## 演示视频
TODO:待录制(建议内容:文字→TTS 音色克隆、语音对话、图片问答、视频问答各一段)
