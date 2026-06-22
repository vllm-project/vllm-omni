hf download stepfun-ai/Step-Audio-Tokenizer
export STEP_AUDIO_TOKENIZER_PATH="/path/to/tokenizer"

vllm-omni serve stepfun-ai/Step-Audio-EditX \
    --deploy-config vllm-omni/vllm_omni/deploy/step_audio_editx.yaml \
    --host 0.0.0.0 \
    --port 8091 \
    --trust-remote-code \
    --omni
