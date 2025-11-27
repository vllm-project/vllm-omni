# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_omni.entrypoints.omni import Omni

model_name = "Qwen/Qwen-Image"
prompt = "a cup of coffee on the table"
if __name__ == "__main__":
    m = Omni(
        model=model_name,
    )
    image = m.generate(
        prompt,
        height=1024,
        width=1024,
        generator=torch.Generator(device="cuda").manual_seed(42),
    )
    image[0].save("qwen_image_output.png")
