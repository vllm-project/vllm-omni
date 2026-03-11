# apps/ComfyUI-vLLM-Omni/comfyui_vllm_omni/utils/models.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from bitsandbytes import nn as bnbnn

def quantize_model(model):
    # Correctly apply quantization using bitsandbytes
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module = bnbnn.Linear4bit(module.in_features, module.out_features, bias=module.bias is not None, compute_dtype=torch.float16)
    return model

def generate_text(model, tokenizer, prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# apps/ComfyUI-vLLM-Omni/comfyui_vllm_omni/__init__.py

from .utils.models import quantize_model, generate_text

# apps/ComfyUI-vLLM-Omni/comfyui_vllm_omni/nodes.py

from comfyui_vllm_omni import generate_text

def text_generation_node(prompt, max_length=50):
    model_name = "black-forest-labs/FLUX.2-dev"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = quantize_model(model)
    return generate_text(model, tokenizer, prompt, max_length)

# apps/ComfyUI-vLLM-Omni/__init__.py

from .comfyui_vllm_omni import text_generation_node

# apps/ComfyUI-vLLM-Omni/web/main.js

// Example usage in JavaScript
async function generateText() {
    try {
        const prompt = "Write a story about a magical forest.";
        const response = await fetch('/api/generate_text', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ prompt })
        });
        const data = await response.json();
        console.log(data.text);
    } catch (error) {
        console.error('Error generating text:', error);
    }
}