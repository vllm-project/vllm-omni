export PYTHONPATH=/workspace/omni/vllm-omni/:$PYTHONPATH
prompt="<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"

CUDA_VISIBLE_DEVICES=6 python end2end.py --model /workspace/ByteDance-Seed/BAGEL-7B-MoT \
                                 --prompt_type text \
                                 --init-sleep-seconds 0 \
				 --stage-configs-path  "/workspace/omni/vllm-omni/vllm_omni/model_executor/stage_configs/bagel.yaml"  \
                                 --prompts ${prompt}
