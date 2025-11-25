from vllm_omni.diffusion.omni_diffusion import OmniDiffusion

# m = OmniDiffusion.from_pretrained(
#     engine_args={
#         "model_name_or_path": "test-model",
#         "num_gpus": 1,
#     }
# )
# m.generate("A beautiful painting of a sunset over the mountains.")
model_name = "Qwen/Qwen-Image"
prompt = "a cup of coffee on the table"
if __name__ == "__main__":
    m = OmniDiffusion.from_pretrained(
        model="Qwen/Qwen-Image",
        num_gpus=1,  # Adjust based on your hardware
    )
    image = m.generate(prompt)
    image[0].save("qwen_image_output.png")
