# Online Serving Example of vLLM-Omni for Qwen2.5-Omni

Source <https://github.com/vllm-project/vllm-omni/blob/main/examples/online_serving/README.md>.


## 🛠️ Installation

Please refer to [installation](../../../getting_started/installation/README.md).

## Deploy Qwen/Qwen2.5-Omni-7B

First launch the OpenAI-compatible inference server
```bash
vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8091
```
If you have custom stage configs file, launch the server with command below
```bash
vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8091 --stage-configs-path /path/to/stage_configs_file
```
## Query the model
Navigate to the example folder
```bash
cd examples/online_serving
```
Query the model server with OpenAI Python API client:
```bash
python openai_chat_completion_client_for_multimodal_generation.py --query-type mixed_modalities
```
??? abstract "openai_chat_completion_client_for_multimodal_generation.py"
    ``````py
    --8<-- "examples/online_serving/openai_chat_completion_client_for_multimodal_generation.py"
    ``````
You can also query the model with `curl` command:
```bash
bash run_curl_multimodal_generation.sh mixed_modalities
```
??? abstract "run_curl_multimodal_generation.sh"
    ``````py
    --8<-- "examples/online_serving/run_curl_multimodal_generation.sh"
    ``````

### FAQ

If you encounter error about backend of librosa, try to install ffmpeg with command below.
```
sudo apt update
sudo apt install ffmpeg
```

## Run Local Web UI Demo

You can also deploy a Gradio Web UI that allows users to interact with the model through a web browser. Below is an example on how to do so with `Qwen/Qwen2.5-Omni-7B`.

### Running Gradio Demo

Install gradio with `uv pip install "gradio>=5.49.1,<6.0.0"`, then you can launch the web service built on AsyncOmni by

```bash
python gradio_demo.py  --model Qwen/Qwen2.5-Omni-7B --port 7861
```
Now you can interact with model via the web UI at `http://localhost:7861/` on your local browser.


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
