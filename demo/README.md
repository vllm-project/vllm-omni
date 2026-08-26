# Demo 使用说明

## 启动方式
nohup python /vllm-workspace/vllm-omni/examples/online_serving/minicpmo/gradio_demo.py --minicpmo45-api-base http://localhost:8091/v1 --minicpmo45-model openbmb/MiniCPM-o-4_5 --port 7862 --host 0.0.0.0


## 访问方式
浏览器打开 http://<host>:7862

## 交互流程
1. 文本输入：直接输入文字，关闭 TTS 勾选为纯文本回复
2. 语音输入：上传音频文件
3. 图像输入：上传图片
4. 视频输入：上传视频

## 演示视频
https://www.bilibili.com/video/BV1Jw8i6kELe
