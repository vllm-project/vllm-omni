from vllm_omni.entrypoints.omni import Omni

model_name = "Qwen/Qwen-Image"
prompt = "a cup of coffee on the table"
if __name__ == "__main__":
    m = Omni(
        model=model_name,
    )
    image = m.generate(prompt)
    image[0].save("qwen_image_output.png")
