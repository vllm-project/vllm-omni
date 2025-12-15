import base64
import io

import requests
from PIL import Image

# 1. 准备本地图片并转为 Base64
image_path = "../text_to_image/coffee.png"  # 替换为你本地的图片路径


def get_resized_base64(image_path, max_size=1024):
    img = Image.open(image_path)

    # 如果图片太大，等比例缩小
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size))
        print(f"图片已压缩至: {img.size}")

    # 转为 Base64
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


base64_image = get_resized_base64(image_path)


# 2. 构造请求
# payload = {
#    "model": "/workspace/ByteDance-Seed/BAGEL-7B-MoT/",
#    "messages": [
#        {
#            "role": "user",
#            "content": [
#                {
#                    "type": "text",
#                    "text": "<|image_pad|>这张图片是什么样子的？有内容么?还是完全没有图片?"
#                },
#                {
#                    "type": "image_url",
#                    "image_url": {
#                        "url": f"data:image/jpeg;base64,{base64_image}"
#                    }
#                }
#            ]
#        }
#    ],
#    "max_tokens": 300
# }

payload = {
    "model": "/workspace/ByteDance-Seed/BAGEL-7B-MoT/",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    # 重点修改这里：
                    # 1. 参考你的例子，只使用 <|image_pad|>
                    # 2. 放在文字的前面 (先看图，再看问题)
                    "text": "<|image_pad|>\n请详细描述这张图片。",
                },
                # {
                #    "type": "image_url",
                #    "image_url": {
                #        "url": f"data:image/jpeg;base64,{base64_image}"
                #    }
                # }
            ],
        }
    ],
    "max_tokens": 300,
    # 既然例子里是标准的 ChatML 格式，我们应该信任 vLLM 的自动模版
    # 如果不行，再尝试设为 False 手动拼写 im_start
}

# 3. 发送请求
response = requests.post("http://localhost:8199/v1/chat/completions", json=payload)
print(response.json())
