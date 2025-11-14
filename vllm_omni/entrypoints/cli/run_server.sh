export PYTHONPATH=/root/gh/vllm_open_release/vllm-omni:$PYTHONPATH
unset HF_HOME
python main.py serve Qwen/Qwen2.5-Omni-7B --omni --port 8091
