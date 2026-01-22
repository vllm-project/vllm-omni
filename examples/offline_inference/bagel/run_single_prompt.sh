prompt="<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"

python end2end.py --model ByteDance-Seed/BAGEL-7B-MoT \
                                 --prompt_type text \
                                 --init-sleep-seconds 0 \
                                 --prompts ${prompt}
