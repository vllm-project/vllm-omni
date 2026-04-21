# OmniWeaving Online Serving Example

This directory demonstrates how to deploy OmniWeaving as an OpenAI-compatible API server.

## 1. Start the Server
Run the following script to start the vLLM API server:
```bash
bash run_server.sh
```

## 2. Run the Client
Once the server is ready, send a sample request with:

```bash
python client.py
```

The client writes the generated video to `output_online.mp4`.
