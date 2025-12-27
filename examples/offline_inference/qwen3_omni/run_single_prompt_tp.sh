python end2end.py --output-wav output_audio \
                  --query-type use_audio \
                  --stage-init-timeout 300

# stage-init-timeout sets the maximum wait to avoid two vLLM stages initializing at the same time on the same card.
