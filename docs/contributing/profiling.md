# Profiling vLLM-Omni
## This guide provides detailed instructions on how to use the logger system in vllm-omni.

In vllm-omni, there are two different scheduling paths:
• Diffusion/DiT Single diffusion Pipeline[[image_to_image]](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/image_to_image)[[text_to_image]](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/text_to_image)[[image_to_image]](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/image_to_image)[[text_to_video]](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/text_to_video)


• Multi-Stage Pipeline for Multimodal Understanding and Speech Generation[[qwen2_5_omni]](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/qwen2_5_omni)[[qwen3_omni]](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/qwen3_omni)


The logging content and usage methods of the logger system under different scheduling paths are as follows:
## Recording Content and Usage Instructions
### 1. VLLM features
VLLM features it log for root module vllm, and the sub model automatically inherit the parent logger. But the vllm_omni module failed to automatically inherit vllm.So we need to init vllm_omni root logger, witch inherit the parent logger.vLLM config includes communication methods, scheduling modes, parallelism, and runtime scale. It also includes shared memory pressure status, model size, and observed GPU memory usage during runtime.The VLLM config content recorded by Single the Diffusion Pipeline model and the Multi-Stage Pipeline model is the same.
#### How to view vllm features
Before running the scripts in the examples, set the environment variables to view the vLLM config in the logs printed in the terminal.
 ```bash
 export VLLM_LOGGING_LEVEL=DEBUG
 ```
### 2.VLLM-omni features
The vllm-omni feature provides multi-dimensional metrics such as end-to-end performance, IPC communication, pipeline scheduling, and engine passthrough, enabling full observability and detailed performance analysis throughout the entire multimodal inference process. However, since the Diffusion Pipeline model does not schedule the omni feature, only the Multi-Stage Pipeline model can access the omni feature.[[qwen2_5_omni]](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/qwen2_5_omni)[[qwen3_omni]](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/qwen3_omni)
#### How to view VLLM-omni features
During the operation of the Multi-Stage Pipeline model, the Omni feature is automatically invoked. You can directly run the script to view the Omni feature of the model.[[qwen2_5_omni]](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/qwen2_5_omni)[[qwen3_omni]](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/qwen3_omni)
```bash
sh run_multiple_prompts.sh
```

### 3.Diffusion features
• The Multi-Stage Pipeline logs do not directly record the details of the diffusion algorithm. Instead, they abstract a complete diffusion process into a single Stage, indirectly reflecting the overall performance of diffusion through `stage_gen_time_ms`, and focus on recording IPC and scheduling characteristics across different Stages.

• The Diffusion Pipeline logs comprehensively cover the core macro characteristics of diffusion inference, including model loading, CFG, number of inference steps, total diffusion time, average denoising step time, and other parameters.



#### How to view Diffusion features
1.The Multi-Stage Pipeline

##### Setting the log switch:

```python
    omni_llm = Omni(
        model=model_name,
        log_stats=args.enable_stats,#Setting  enable_stats=True 
        log_file=(os.path.join(log_dir, "omni_llm_pipeline.log") if args.enable_stats else None)
    )
```
or
```python
    omni_llm = Omni(
        model=model_name,
        log_stats=True 
        log_file=os.path.join(log_dir, "omni_llm_pipeline.log") 
    )

```
##### Setting the log switch:

```bash
sh run_multiple_prompts.sh
```

2.The Diffusion Pipeline

Run the Diffusion Pipeline script directly to view the model's diffusion properties(Taking image_to_image as an example, the usage method for other models is the same.)[[image_to_image]](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/image_to_image)[[text_to_image]](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/text_to_image)[[image_to_image]](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/image_to_image)[[text_to_video]](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/text_to_video):

```python
python image_edit.py \
        --image input.png \
        --prompt "Let this mascot dance under the moon, surrounded by floating stars and poetic bubbles such as 'Be Kind'" \
        --output output_image_edit.png \
        --num_inference_steps 50 \
        --cfg_scale 4.0

```