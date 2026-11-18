# Run MiniMax H3 text encoder by using Independent Stage Execution

## 1. Deploy the frontend
Download MiniMax-H3 and use MiniMax-H3/FL2VA for running H3 text encoder
```
export VLLM_OMNI_ENABLE_RUN_ENDPOINT=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800
export VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=3600

vllm serve /MiniMax-H3/FL2VA \
  --trust-remote-code \
  --no-async-chunk \
  --stage-id none \
  --deploy-config [minimax_h3_so_only.yaml](../../vllm_omni/deploy/minimax_h3_so_only.yaml)/MiniMax-H3/xubo/encoder-only.yaml \
  --omni-master-address 127.0.0.1 \
  --omni-master-port 21212 \
  --init-timeout 7200 \
  --stage-init-timeout 7200 \
  --port 8000 \
  --omni 
```

Wait until you see logs indicating that the addresses for remote stages have been pre-allocated.

```text
(APIServer pid=434098) INFO ... Pre-allocated addresses for stages [0] (master=127.0.0.1:21212)
(APIServer pid=434098) INFO ... [OmniMasterServer] Listening on tcp://127.0.0.1:21212
(APIServer pid=434098) INFO ... [DistStageRuntime] OmniMasterServer started for stages [0]
```

## 2. Deploy the MiniMax-H3 text encoder

```
vllm serve /MiniMax-H3/FL2VA \
  --omni \
  --trust-remote-code \
  --no-async-chunk \
  --stage-id 0 \
  --deploy-config [minimax_h3_so_only.yaml](../../vllm_omni/deploy/minimax_h3_so_only.yaml)/MiniMax-H3/xubo/encoder-only.yaml \
  --task-type fl2va \
  --headless \
  --omni-master-address 127.0.0.1 --omni-master-port 21212 \
  --num-gpus 2 \
  --text-encoder-tp-size 2
```

## 3.Send request to /v1/run
```
 time curl --location --request POST 'http://localhost:8000/v1/run'   --header 'Content-Type: application/json'   --data-raw '{
    "stage_id": 0,
    "stage_input": {
      "prompt": "A cat and a dog play a game together in a sunny park."
    },
    "sampling_params": [
      {
        "seed": 1101,
        "extra_args": {
          "task": "t2va",
          "aspect_ratio": "16:9",
          "duration": 4
        }
      }
    ]
  }'
```