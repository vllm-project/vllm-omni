# Online serving Example of vLLM-Omni for Qwen2.5-omni

Source <https://github.com/vllm-project/vllm/tree/main/examples/online_serving/qwen2_5_omni>.


## 🛠️ Installation

Please refer to [README.md](https://github.com/vllm-project/vllm/tree/main/examples/README.md)

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
python openai_chat_completion_client_for_multimodal_generation.py --query-type mixed_modalities
```

Send request via curl
```bash
bash run_curl_multimodal_generation.sh mixed_modalities
```

### FAQ

If you encounter error about backend of librosa, try to install ffmpeg with command below.
```
sudo apt update
sudo apt install ffmpeg
```

## Run Local Web UI Demo

This Web UI demo allows users to interact with the model through a web browser.

### Running Gradio Demo

Once vllm and vllm-omni are installed, you can launch the web service built on AsyncOmni by

```bash
python gradio_demo.py  --model Qwen/Qwen2.5-Omni-7B --port 7861
```

Then open `http://localhost:7861/` on your local browser to interact with the web UI.


### Options

You can customize its basic launch parameters:

```bash
python gradio_demo.py \
    --model Qwen/Qwen2.5-Omni-7B \
    --ip 127.0.0.1 \
    --port 7861 \
    --stage-configs-path /path/to/stage_configs.yaml
```

- `--model`: Local model checkpoint to load (default `Qwen/Qwen2.5-Omni-7B`).
- `--ip`: Host/IP for the Gradio server (default `127.0.0.1`).
- `--port`: Port for the Gradio server (default `7861`).
- `--stage-configs-path`: Optional path to custom stage configs YAML.
- `--share`: Set to expose a temporary public link via Gradio.

## Example materials

??? abstract "gradio_demo.py"
    ``````py
    --8<-- "/mnt/vllm_open_release/vllm-omni-cursor/vllm-omni/examples/online_serving/qwen2_5_omni/gradio_demo.py"
    ``````
??? abstract "openai_chat_completion_client_for_multimodal_generation.py"
    ``````py
    --8<-- "/mnt/vllm_open_release/vllm-omni-cursor/vllm-omni/examples/online_serving/qwen2_5_omni/openai_chat_completion_client_for_multimodal_generation.py"
    ``````
??? abstract "run_curl_multimodal_generation.sh"
    ``````sh
    --8<-- "/mnt/vllm_open_release/vllm-omni-cursor/vllm-omni/examples/online_serving/qwen2_5_omni/run_curl_multimodal_generation.sh"
    ``````
