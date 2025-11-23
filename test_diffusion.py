from vllm_omni.diffusion.omni_diffusion import OmniDiffusion

# m = OmniDiffusion.from_pretrained(
#     engine_args={
#         "model_name_or_path": "test-model",
#         "num_gpus": 1,
#     }
# )
# m.generate("A beautiful painting of a sunset over the mountains.")
model_name = "Qwen/Qwen-Image"

m = OmniDiffusion.from_pretrained(
    model="Qwen/Qwen-Image",
    num_gpus=2,  # Adjust based on your hardware
)
m.generate("一只猫坐在公园的长椅上, 超清，电影级构图.")
