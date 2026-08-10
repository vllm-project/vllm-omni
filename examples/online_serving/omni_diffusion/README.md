# Omni-Diffusion: Online serving

The server uses one task-specific deploy config per process. Component paths
left as `null` use the official Hugging Face repositories described in the
[offline example](../../offline_inference/omni_diffusion/README.md). Start the
matching task directly with its bundled config:

```bash
# Use the Hugging Face model ID by default. Set MODEL to an existing local
# model directory to skip downloading the main checkpoint.
MODEL=${MODEL:-lijiang/Omni-Diffusion}

bash examples/online_serving/omni_diffusion/run_server.sh \
  "$MODEL" vllm_omni/deploy/omni_diffusion_vqa.yaml
```

Wait for `Application startup complete` before sending a request.

## Send Requests

The same client covers all six tasks and saves the result under
`/tmp/omni_diffusion_online` by default:

```bash
python examples/online_serving/omni_diffusion/client.py \
  --task t2i --model "$MODEL"

python examples/online_serving/omni_diffusion/client.py \
  --task vqa --model "$MODEL"

python examples/online_serving/omni_diffusion/client.py \
  --task asr --model "$MODEL"

python examples/online_serving/omni_diffusion/client.py \
  --task tts --model "$MODEL"

python examples/online_serving/omni_diffusion/client.py \
  --task s2i --model "$MODEL"

python examples/online_serving/omni_diffusion/client.py \
  --task svqa --model "$MODEL"
```

Restart the server with the corresponding deploy config before changing tasks.
The client uses vLLM's cached `cherry_blossom` image and `mary_had_lamb` audio
assets by default. Use `--image-path`, `--audio-path`, `--base-url`, `--prompt`,
`--output`, and `--timeout` to override the client defaults.

S2I currently accepts local files or `data:audio` URLs. The client converts a
local audio file to a data URL so the server can apply its normal request parsing
before forwarding the audio to the model-specific diffusion wrapper.
