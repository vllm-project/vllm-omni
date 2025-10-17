python qwen2_5_omni_ckpt_test.py --model Qwen/Qwen2.5-Omni-7B \
                                 --prompts "Explain the system architecture for a scalable audio generation pipeline. Answer in 15 words." \
                                 --voice-type "m02" \
                                 --dit-ckpt none \
                                 --bigvgan-ckpt none \
                                 --output-wav output_audio \
                                 --prompt_type text