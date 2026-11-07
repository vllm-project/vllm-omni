# Independent Stage Execution (experimental)

!!! warning
    This feature is **experimental**. The `POST /v1/run` request and response
    formats may change without notice.

!!! danger
    `POST /v1/run` returns 400 unless `VLLM_OMNI_ENABLE_RUN_ENDPOINT=1` is set. You should only enable it for trusted callers, because it is possible to modify `stage_input` in downstream stages to cause bad behaviors, e.g., crashing individual stages.

`POST /v1/run` runs one stage per call. Instead of submitting a stage's result to the next stage, the server returns it, and the caller sends it back to run that stage. This lets callers route each call themselves, e.g., to independently scaled headless stages.

Each call may include a stage-local `replica_id`. When present, the request is
sent only to that live, attached replica; an unknown or unavailable replica is
rejected without falling back to automatic routing. Omitting `replica_id`
preserves the existing load-balanced behavior. Replica IDs are independent
between stages, so the same numeric ID can name different workers in different
stage pools.

## Deployment

Depending on the situation, it may be beneficial to deploy as one frontend with every stage running headless behind a master server, or with one API server per stage. vLLM-Omni supports either approach, which we will demonstrate using Qwen3-TTS as an example.

### One frontend with headless stages

First, start the frontend, which also brings up the orchestrator in its own thread. This can be accomplished by passing `--stage-id none`. Note that you must configure the Omni master server settings for remote replicas.

```bash
# Starts the API server on port 8000
VLLM_OMNI_ENABLE_RUN_ENDPOINT=1 vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --trust-remote-code \
  --no-async-chunk \
  --stage-id none \
  --omni-master-address 127.0.0.1 \
  --omni-master-port 21212 \
  --omni
```

Wait until you see logs indicating that the addresses for remote stages have been pre-allocated.

```text
(APIServer pid=434098) INFO ... Pre-allocated addresses for stages [0, 1] (master=127.0.0.1:21212)
(APIServer pid=434098) INFO ... [OmniMasterServer] Listening on tcp://127.0.0.1:21212
(APIServer pid=434098) INFO ... [DistStageRuntime] OmniMasterServer started for stages [0, 1]
```

In another terminal, start the talker and code2wav stages as shown below:

```bash
# STAGE_ID=0 should be used to start the talker. Use STAGE_ID=1 for code2wav.
STAGE_ID=0

vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --trust-remote-code \
  --no-async-chunk \
  --stage-id $STAGE_ID \
  --headless \
  --omni-master-address 127.0.0.1 \
  --omni-master-port 21212 \
  --omni
```

You should see both stage 0 and stage 1 register as replicas with the `OmniMasterServer` and that the `DistStageRuntime`s have attached successfully. Once the stages have finished initialization, the server will show as ready.

### One server per stage

Each stage can also run its own API server, without a master server. If you pass `--stage-id` without `--omni-master-address`, the server runs only that stage.

```bash
# Starts the talker (stage 0) on port 8000
VLLM_OMNI_ENABLE_RUN_ENDPOINT=1 vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --trust-remote-code \
  --no-async-chunk \
  --stage-id 0 \
  --port 8000 \
  --omni

# Starts code2wav (stage 1) on port 8001
VLLM_OMNI_ENABLE_RUN_ENDPOINT=1 vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --trust-remote-code \
  --no-async-chunk \
  --stage-id 1 \
  --port 8001 \
  --omni
```

When run in this way, the caller sends the stage 0 call to the talker's server, then sends its response to code2wav's server. These servers only serve `/v1/run`, `/health`, and `/metrics`; other routes return 400.

## `POST /v1/run`

In the preliminary implementation, there is no preprocessing or postprocessing done on the request and response objects, so you need to build the stage 0 inputs and unpack the final response directly.

After the first inference call, we can pipe the output directly back into `/v1/run` to run the second stage. Currently, the output is encoded since it is the raw yielded payload, so we provide a helper to unpack the raw message. Sampling params currently must be set on the stage 0 call and are rejected on direct calls to downstream stages. You can see the full flow below.

```python
import argparse
import json

import httpx
import soundfile as sf
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_run import decode_output
from vllm_omni.entrypoints.openai.tts_adapters.base import conditioning_cache_salt
from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import Qwen3TTSPromptEmbedsBuilder


def build_stage0_input(model: str, text: str, speaker: str) -> dict:
    """Build the talker's input the way the speech endpoint does (CustomVoice, built-in speaker)."""
    tts_params = {"text": [text], "task_type": ["CustomVoice"], "language": ["Auto"], "speaker": [speaker]}
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, padding_side="left")
    with open(hf_hub_download(model, "config.json")) as f:
        talker_config = json.load(f)["talker_config"]
    prompt_len = Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
        additional_information=tts_params,
        task_type="CustomVoice",
        tokenize_prompt=lambda t: tokenizer(t, padding=False)["input_ids"],
        codec_language_id=talker_config.get("codec_language_id"),
        spk_is_dialect=talker_config.get("spk_is_dialect"),
    )
    cache_salt = conditioning_cache_salt(OpenAICreateSpeechRequest(input=text, voice=speaker), tts_params)
    return {"prompt_token_ids": [1] * prompt_len, "additional_information": tts_params, "cache_salt": cache_salt}


def main() -> None:
    parser = argparse.ArgumentParser()
    # NOTE: the port here is your API server port, not the omni master port
    parser.add_argument("--url", default="http://localhost:8000")
    # With one server per stage, set this to stage 1's server, e.g., http://localhost:8001
    parser.add_argument("--stage1-url", default=None)
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    parser.add_argument("--text", default="Hello world")
    parser.add_argument("--speaker", default="vivian")
    parser.add_argument("--out", default="out.wav")
    args = parser.parse_args()

    stage0_url = f"{args.url}/v1/run"
    stage1_url = f"{args.stage1_url or args.url}/v1/run"

    # Stage 0 (talker) returns the request for stage 1.
    stage1_request = httpx.post(
        stage0_url,
        json={
            "stage_input": build_stage0_input(args.model, args.text, args.speaker),
            # NOTE: You can also select a replica manually by ID if you are running one frontend server in front of multiple replicas.
            # "replica_id": 1,
        },
        timeout=600,
    )
    stage1_request.raise_for_status()

    # Stage 1 (code2wav) returns the final output.
    stage1_body = stage1_request.json()

    # Example for picking a code2wav replica by ID the same way.
    # stage1_body["replica_id"] = 0
    final = httpx.post(
        stage1_url,
        json=stage1_body,
        timeout=600,
    )
    final.raise_for_status()

    # For now, since we don't have postprocessing, we call a helper to unpack the message for us
    decoded_output = decode_output(final.json()["output"])
    # Then extract the speech and save it
    audio_output = decoded_output.outputs[0].multimodal_output
    sf.write(args.out, audio_output["audio"].float().numpy().squeeze(), int(audio_output["sr"]))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

## Limitations

This feature is early in its development and currently has the following gaps.

- The server passes `stage_input` to the stages largely as is. Send back the one returned by the previous call unchanged, and only expose `/v1/run` to trusted callers: a modified payload can crash a stage or the server.
- The final response only has the last stage's output.
- No route-specific preprocessing or postprocessing yet, so callers build the stage 0 input and unpack the final output themselves.
- Async chunking isn't supported.
- With one server per stage, pipelines that use native AR-to-DiT KV transfer fail at startup, since stage 0 can't know which server the next call goes to.
