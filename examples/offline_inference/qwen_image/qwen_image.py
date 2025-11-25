from vllm_omni.diffusion.omni_diffusion import OmniDiffusion

model_name = "Qwen/Qwen-Image"
prompt = "a cup of coffee on the table"
if __name__ == "__main__":
    m = OmniDiffusion.from_pretrained(
        model="Qwen/Qwen-Image",
        num_gpus=1,
    )
    image = m.generate(prompt)
    image[0].save("qwen_image_output.png")
