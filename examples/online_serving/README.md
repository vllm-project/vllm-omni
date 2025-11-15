# Online serving Example of vLLM-omni for Qwen2.5-omni

## Run examples (Qwen2.5-omni)

Get into the example folder
```bash
cd examples/online_serving
```

Launch the server
```bash
bash run_server.sh
```

Send request via python
```bash
python openai_chat_completion_client_for_multimodal_generation.py
```

Send request via curl
```bash
bash run_curl_multimodal_generation.sh
```
