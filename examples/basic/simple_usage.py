#!/usr/bin/env python3
"""
Simple vLLM-omni usage example.

This example demonstrates how to use vLLM-omni for basic text generation
with a single AR (Autoregressive) stage.
"""

import asyncio
from vllm_omni.entrypoints.omni_llm import OmniLLM, AsyncOmniLLM
from vllm_omni.config import create_ar_stage_config


def sync_example():
    """Synchronous usage example."""
    print("=== Synchronous vLLM-omni Example ===")
    
    # Create a simple AR stage configuration
    stage_config = create_ar_stage_config(
        stage_id=0,
        model_path="Qwen/Qwen3-0.6B",
        input_modalities=["text"],
        output_modalities=["text"]
    )
    
    # Initialize OmniLLM
    omni_llm = OmniLLM([stage_config])
    
    # Prepare stage arguments
    stage_args = [{
        "prompt": "Hello, how are you today?",
        "max_tokens": 50,
        "temperature": 0.7
    }]
    
    # Generate text
    print(f"Input: {stage_args[0]['prompt']}")
    outputs = omni_llm.generate(stage_args)
    
    # Display results
    for i, output in enumerate(outputs):
        print(f"Output {i}:")
        if hasattr(output, 'outputs') and output.outputs:
            for completion in output.outputs:
                print(f"  Text: {completion.text}")
                print(f"  Finished: {completion.finish_reason != 'length'}")
                print(f"  Tokens: {len(completion.token_ids)} tokens")


async def async_example():
    """Asynchronous usage example."""
    print("\n=== Asynchronous vLLM-omni Example ===")
    
    # Create a simple AR stage configuration
    stage_config = create_ar_stage_config(
        stage_id=0,
        model_path="Qwen/Qwen3-0.6B",
        input_modalities=["text"],
        output_modalities=["text"]
    )
    
    # Initialize AsyncOmniLLM
    omni_llm = AsyncOmniLLM([stage_config])
    
    # Prepare stage arguments
    stage_args = [{
        "prompt": "What is artificial intelligence?",
        "max_tokens": 100,
        "temperature": 0.8
    }]
    
    # Generate text asynchronously
    print(f"Input: {stage_args[0]['prompt']}")
    outputs = await omni_llm.generate_async(stage_args)
    
    # Display results
    for i, output in enumerate(outputs):
        print(f"Output {i}:")
        if hasattr(output, 'outputs') and output.outputs:
            for completion in output.outputs:
                print(f"  Text: {completion.text}")
                print(f"  Finished: {completion.finish_reason != 'length'}")
                print(f"  Tokens: {len(completion.token_ids)} tokens")


def multi_prompt_example():
    """Example with multiple prompts."""
    print("\n=== Multi-Prompt Example ===")
    
    # Create stage configuration
    stage_config = create_ar_stage_config(
        stage_id=0,
        model_path="Qwen/Qwen3-0.6B",
        input_modalities=["text"],
        output_modalities=["text"]
    )
    
    # Initialize OmniLLM
    omni_llm = OmniLLM([stage_config])
    
    # Multiple prompts
    prompts = [
        "Tell me a joke",
        "What's the weather like?",
        "Explain quantum computing"
    ]
    
    for prompt in prompts:
        print(f"\nInput: {prompt}")
        stage_args = [{
            "prompt": prompt,
            "max_tokens": 30,
            "temperature": 0.9
        }]
        
        outputs = omni_llm.generate(stage_args)
        
        if outputs and hasattr(outputs[0], 'outputs') and outputs[0].outputs:
            completion = outputs[0].outputs[0]
            print(f"Output: {completion.text}")


if __name__ == "__main__":
    print("vLLM-omni Simple Usage Examples")
    print("=" * 40)
    
    # Run synchronous example
    sync_example()
    
    # Run asynchronous example
    asyncio.run(async_example())
    
    # Run multi-prompt example
    multi_prompt_example()
    
    print("\n" + "=" * 40)
    print("Examples completed!")
