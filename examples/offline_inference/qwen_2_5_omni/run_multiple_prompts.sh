export PYTHONPATH=/path/to/vllm-omni:$PYTHONPATH
python end2end.py --model Qwen/Qwen2.5-Omni-7B \
                                 --voice-type "m02" \
                                 --dit-ckpt none \
                                 --bigvgan-ckpt none \
                                 --output-wav output_audio \
                                 --prompt_type text \
                                 --init-sleep-seconds 0 \
                                 --txt-prompts top100.txt
