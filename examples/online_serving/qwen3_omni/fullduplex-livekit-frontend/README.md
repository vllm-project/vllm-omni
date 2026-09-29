# Qwen3-Omni LiveKit demo

This directory contains a Docker Compose version of the local LiveKit demo.
It builds the LiveKit server, the Next.js frontend, and the Python agent as
separate images. The vLLM-Omni server runs outside Compose and is configured
with `VLLM_OMNI_HOST`.

Start vLLM-Omni with port `8000` bound to an address reachable from Docker,
for example with `--host 0.0.0.0`, then run:

```bash
cd examples/online_serving/qwen3_omni/fullduplex-livekit-frontend
VLLM_OMNI_HOST=host.docker.internal docker compose up --build
```

Open [http://localhost:3000](http://localhost:3000). The LiveKit server is
available at `ws://localhost:7880`; its development credentials are `devkey`
and `secret`.

For a machine other than the local host, set both the vLLM address and the
address LiveKit advertises to WebRTC clients:

```bash
VLLM_OMNI_HOST=192.168.1.20 \
LIVEKIT_NODE_IP=192.168.1.10 \
LIVEKIT_PUBLIC_URL=ws://192.168.1.10:7880 \
docker compose up --build
```

Use `docker compose up --build --abort-on-container-exit` when the Compose
process should stop the remaining services after one component exits, matching
the behavior of `run-livekit-stack.sh`.
