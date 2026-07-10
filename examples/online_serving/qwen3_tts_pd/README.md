# Qwen3-TTS PD (Prefill–Decode) Disaggregation Example

This directory contains everything needed to bring up a **same-host
multi-GPU Qwen3-TTS PD deploy** and exercise both the single-prefill
(1P1D) and multiple-prefill (M-P-1-D: 3 prefill + 1 decode) topologies
end-to-end.

The companion deploy YAMLs live under `vllm_omni/deploy/`:

| YAML | Topology | GPUs | Use |
|------|----------|------|-----|
| [`qwen3_tts_pd_1p1d.yaml`](../../../vllm_omni/deploy/qwen3_tts_pd_1p1d.yaml) | 1 prefill + 1 decode + 1 code2wav | 3 | Smallest configuration; matches the legacy 1P1D path exactly. Use this for the [probe script](#1-correctness--latency-probe-1p1d) and as the K=1 regression baseline. |
| [`qwen3_tts_pd_3p1d.yaml`](../../../vllm_omni/deploy/qwen3_tts_pd_3p1d.yaml) | 3 prefill + 1 decode + 1 code2wav | 5 | M-P-1-D recipe with `pd_prefill_pick_strategy: round_robin`. |
| [`qwen3_tts_pd_1p3d.yaml`](../../../vllm_omni/deploy/qwen3_tts_pd_1p3d.yaml) | 1 prefill + 3 decode + 1 code2wav | 5 | One prefill fan-out to three decoders. |
| [`qwen3_tts_pd_1p6d.yaml`](../../../vllm_omni/deploy/qwen3_tts_pd_1p6d.yaml) | 1 prefill + 6 decode + 1 code2wav | 8 | One prefill fan-out to six decoders. |

Both flavours are intentionally **same-host only** -- they do not
require Mooncake RDMA across nodes.

---

## 1. Correctness & latency probe (1P1D)

The probe script [`probe_1p1d.py`](./probe_1p1d.py) drives a baseline
single-process server and a 1P1D PD server side-by-side and compares
the audio-codes streams.  It enforces two contracts:

1. **Greedy bit-equality**: the audio_codes hash from the PD pipeline
   must match the single-process baseline frame-for-frame.
2. **KV-transfer success**: the PD median TTFB must not be more than
   `--ttfb-ratio-threshold` (default 3.0) times the baseline TTFB.

Bring up the two servers:

```bash
# Terminal 1: baseline (no PD)
CUDA_VISIBLE_DEVICES=0,1 \
vllm serve <Qwen3-TTS path> --omni --port 8090 \
    --stage-configs-path vllm_omni/deploy/qwen3_tts.yaml

# Terminal 2: 1P1D PD probe
CUDA_VISIBLE_DEVICES=2,3,4 \
vllm serve <Qwen3-TTS path> --omni --port 8091 \
    --stage-configs-path vllm_omni/deploy/qwen3_tts_pd_1p1d.yaml
```

Run the probe.  The verdict is printed to stdout; pass the optional
`--inventory-path` flag to also append it to a local markdown file:

```bash
python examples/online_serving/qwen3_tts_pd/probe_1p1d.py \
    --baseline-url http://localhost:8090 \
    --pd-url       http://localhost:8091 \
    --runs 3
```

Exit code is `0` on PASS, non-zero on FAIL.

---

## 2. M-P-1-D bring-up (3 prefill + 1 decode + 1 code2wav)

[`qwen3_tts_pd_3p1d.yaml`](../../../vllm_omni/deploy/qwen3_tts_pd_3p1d.yaml)
declares five stages on five GPUs of the same machine (3 prefill +
1 decode + 1 code2wav).  Each Mooncake prefill instance owns a
**unique** bootstrap port; pd_utils validates this at startup so port
collisions fail fast with a clear error.

### GPU & port plan

| Stage | role               | YAML `devices` | `mooncake_bootstrap_port` | Notes |
|-------|--------------------|----------------|----------------------------|-------|
| 0 | Talker prefill #0     | `"0"`          | `25201`                    | kv_producer; decode's `engine_input_source` references this stage. |
| 1 | Talker prefill #1     | `"1"`          | `25202`                    | kv_producer; same port must NOT clash with other prefills. |
| 2 | Talker prefill #2     | `"2"`          | `25203`                    | kv_producer; same port must NOT clash with other prefills. |
| 3 | Talker decode         | `"3"`          | `25204`                    | kv_consumer; `engine_input_source: [0, 1, 2]`. |
| 4 | Code2Wav              | `"7"`          | _n/a_                      | Standard pipeline stage; consumes from stage 3 via SharedMemoryConnector. |

### Launch

The deploy YAML pins each stage to a specific GPU via its `devices`
field, so a single `CUDA_VISIBLE_DEVICES=0,1,2,3,7` is sufficient at the
process level:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,7 \
vllm serve <Qwen3-TTS path> --omni --port 8092 \
    --stage-configs-path vllm_omni/deploy/qwen3_tts_pd_3p1d.yaml
```

### Switching the prefill selection strategy

Two ways, in precedence order (highest first):

1. **YAML field** (recommended) -- top-level `pd_prefill_pick_strategy`:
   ```yaml
   pd_prefill_pick_strategy: least_inflight   # or round_robin (default)
   ```
2. **Environment variable** at process start:
   ```bash
   VLLM_OMNI_PD_PREFILL_PICK_STRATEGY=least_inflight \
   vllm serve ... --stage-configs-path vllm_omni/deploy/qwen3_tts_pd_3p1d.yaml
   ```

Allowed values: `round_robin` (default) and `least_inflight`.  Unknown
values fall back to the default with a single warning at startup.

---

## 3. Calling the server

Once either flavour is running, the `/v1/audio/speech` endpoint behaves
identically to a single-process Qwen3-TTS deploy: clients do not need
to know they are speaking to a PD pipeline.  Streaming PCM example:

```bash
curl -N http://localhost:8092/v1/audio/speech \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer EMPTY" \
    -d '{
      "model": "qwen3_tts",
      "input": "vLLM-Omni Qwen3 TTS multi-prefill demo.",
      "voice": "default",
      "stream": true,
      "response_format": "pcm"
    }' --output ./pd_output.pcm
```

---

## 4. Falling back to single-process

To revert to the original single-process Qwen3-TTS (no PD), simply
launch with `vllm_omni/deploy/qwen3_tts.yaml`:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
vllm serve <Qwen3-TTS path> --omni --port 8092 \
    --stage-configs-path vllm_omni/deploy/qwen3_tts.yaml
```

`pd_utils` detects no `is_prefill_only` / `is_decode_only` stages and
short-circuits all PD-related code paths, so the runtime path is
bit-identical to a vanilla single-process deploy.

---

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `Mooncake bootstrap port collision: stage-X and stage-Y both use port N` | Two prefill stages share `mooncake_bootstrap_port`. | Assign distinct ports under each prefill's `kv_connector_extra_config`. |
| `PD stages must have matching tensor_parallel_size` | Decode TP differs from prefill TP. | Mooncake doesn't support heterogeneous TP across the PD pair; align them. |
| Probe FAIL on bit-equality but TTFB ratio is fine | Some attention KV layer (e.g. the talker code-predictor sub-module) was not captured by Mooncake. | Add a fallback recompute path for the missing KV layer. |
| Probe FAIL on TTFB ratio but bit-equal passes | Decode is silently re-running prefill (KV transfer not engaged). | Inspect Mooncake handshake logs; confirm `kv_role`, ports, and that the patched `MooncakeConnector` is loaded (`mooncake_pd_patch.py` warning at startup). |
