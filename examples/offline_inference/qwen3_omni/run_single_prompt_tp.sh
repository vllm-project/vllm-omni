python end2end.py --output-wav output_audio \
                  --query-type use_audio \
                  --init-sleep-seconds 90 \
                  --stage-configs-path qwen3_omni_moe_tp.yaml

# init-sleep-seconds works to avoid two vLLM stages initialized at the same time within a card.
