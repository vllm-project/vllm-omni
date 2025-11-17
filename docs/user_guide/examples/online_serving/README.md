# Online serving Example of vLLM-omni for Qwen2.5-omni

Source <https://github.com/vllm-project/vllm-omni/blob/main/examples\online_serving\README.md>.


## 🛠️ Installation

Please refer to [README.md](https://github.com/vllm-project/vllm-omni/blob/main/README.md)

## Run examples (Qwen2.5-omni)

Launch the server
```bash
vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8091
```

If you have custom stage configs file, launch the server with command below
```bash
vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8091 --stage-configs-path /path/to/stage_configs_file
```

Get into the example folder
```bash
cd examples/online_serving
```

Send request via python
```bash
python openai_chat_completion_client_for_multimodal_generation.py
```

Send request via curl
```bash
bash run_curl_multimodal_generation.sh
```
