# Independent Stage Execution (experimental)

!!! warning
    This feature is **experimental**. The `POST /v1/run` request and response
    formats may change without notice.

vLLM-Omni supports running stages independently, where the payload is yielded back to the user rather than submitted to the next stage. This provides the groundwork for high-scale distributed serving by decoupling the frontend (API Server & Orchestrator) from independently scalable headless stages, which will eventually allow users to manage their own routing, e.g., based on scoring heuristics.

## Deployment

We will use Qwen3-TTS as an example for independent stage execution, illustrating how to pass the payload from one stage to the next. First, start the frontend, which also brings up the orchestrator in its own thread. For now, this can be accomplished by passing a stage-id of `-1`. Note that you must configure the Omni master server settings for remote replicas.

```bash
# Starts the API server on port 8000
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --trust-remote-code \
  --no-async-chunk \
  --stage-id -1 \
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

## `POST /v1/run`

In the preliminary implementation, there is no preprocessing or postprocessing done on the request and response objects, so you need to build the stage 0 inputs and unpack the final response directly.

After the first inference call, we can pipe the output directly back into `/v1/run` to run the second stage. Currently, the output is encoded since it is the raw yielded payload, so we provide a helper to unpack the raw message. You can see the full flow below.

```python
import argparse
import json

import httpx
import soundfile as sf
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_run import decode_output
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
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
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    parser.add_argument("--text", default="Hello world")
    parser.add_argument("--speaker", default="vivian")
    parser.add_argument("--out", default="out.wav")
    args = parser.parse_args()

    url = f"{args.url}/v1/run"

    # Stage 0 (talker) returns the request for stage 1.
    stage1_request = httpx.post(
        url,
        json={
            "stage_input": build_stage0_input(args.model, args.text, args.speaker)
        },
        timeout=600,
    )
    stage1_request.raise_for_status()

    # Stage 1 (code2wav) returns the final output.
    final = httpx.post(
        url,
        content=stage1_request.content,
        headers={"Content-Type": "application/json"},
        timeout=600,
    )
    final.raise_for_status()

    # For now, since we don't have postprocessing, we call a helper to unpack the message for us
    decoded_output = decode_output(final.json()["output"])
    # Then extract the speech and save it
    audio_output, audio_key = OmniOpenAIServingSpeech._extract_audio_output(decoded_output)
    sf.write(args.out, audio_output[audio_key].float().numpy().squeeze(), int(audio_output["sr"]))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

## Limitations

This feature is early in its development and currently has the following constraints, many of which are being actively worked on.

- Support for emulating different entrypoints' preprocessing and postprocessing through `/v1/run`
- API for passing the `/v1/run` request to a specific replica
- Support for async chunking
