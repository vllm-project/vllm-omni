# Ming-flash-omni Standalone TTS (Online Serving)

This directory contains online e2e examples for **Ming-flash-omni-2.0 standalone talker deployment**.

Server uses:

- `model`: `Jonathan1909/Ming-flash-omni-2.0`
- `deploy config`: `vllm_omni/deploy/ming_flash_omni_tts.yaml`

## Launch the Server

```bash
# from repo root
bash examples/online_serving/ming_flash_omni_tts/run_server.sh
```

Equivalent manual command:

```bash
vllm serve Jonathan1909/Ming-flash-omni-2.0 \
    --deploy-config vllm_omni/deploy/ming_flash_omni_tts.yaml \
    --host 0.0.0.0 \
    --port 8091 \
    --trust-remote-code \
    --omni
```

## Send TTS Request

Python client:

```bash
python examples/online_serving/ming_flash_omni_tts/speech_client.py \
    --text "我们当迎着阳光辛勤耕作，去摘取，去制作，去品尝，去馈赠。" \
    --output ming_online.wav
```

Long-form `instructions` (e.g. ASMR whisper style) via the client:

```bash
python examples/online_serving/ming_flash_omni_tts/speech_client.py \
    --text "我会一直在这里陪着你，直到你慢慢、慢慢地沉入那个最温柔的梦里……好吗？" \
    --instructions "这是一种ASMR耳语，属于一种旨在引发特殊感官体验的创意风格。这个女性使用轻柔的普通话进行耳语，声音气音成分重。音量极低，紧贴麦克风，语速极慢，旨在制造触发听者颅内快感的声学刺激。" \
    --output ming_online_asmr.wav
```

## Notes

- This is the **online serving** counterpart of `examples/offline_inference/ming_flash_omni_tts/`.
- The server uses `use_zero_spk_emb=True` and the default decode args
  (`max_decode_steps=200`, `cfg=2.0`, `sigma=0.25`, `temperature=0.0`).
  For other caption fields (`语速`, `基频`, `IP`, BGM, etc.) or overriding
  decode args, use the offline e2e example where `additional_information`
  is set explicitly.
