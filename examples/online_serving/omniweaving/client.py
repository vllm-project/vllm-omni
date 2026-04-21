import base64

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

print("Sending request to OmniWeaving API...")
response = client.chat.completions.create(
    model="Tencent-Hunyuan/OmniWeaving",
    messages=[{"role": "user", "content": "A cute bear wearing a Christmas hat moving naturally."}],
    # Special sampling parameters for vLLM-Omni
    extra_body={
        "num_inference_steps": 30,
        "height": 256,
        "width": 256,
        "num_frames": 9,
    },
)

video_base64 = response.choices[0].message.content
if video_base64:
    with open("output_online.mp4", "wb") as f:
        f.write(base64.b64decode(video_base64))
    print("✅ Successfully generated and saved output_online.mp4")
