#!/bin/bash
set -e
source /cache/caitianchi/miniconda3/etc/profile.d/conda.sh
conda activate vllm

echo "=== Health Check ==="
curl -s http://localhost:8091/health && echo " HEALTH_OK" || echo " HEALTH_FAIL"

echo ""
echo "=== API Text+TTS Test ==="
RESPONSE=$(curl -s http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/cache/caitianchi/model/MiniCPM-o-2_6","messages":[{"role":"user","content":"你好，用一句话介绍北京"}],"max_tokens":128,"temperature":0.0}')

echo "$RESPONSE" | python3 -c "
import sys, json, base64
data = json.load(sys.stdin)
if 'error' in data:
    print('ERROR:', data['error']['message'][:300])
else:
    for i, c in enumerate(data['choices']):
        msg = c['message']
        print(f'Choice {i}: content={msg.get(\"content\",\"N/A\")[:200]}')
        if msg.get('audio') and msg['audio'].get('data'):
            aud = base64.b64decode(msg['audio']['data'])
            print(f'  audio: {len(aud)} bytes')
            if len(aud) > 100:
                with open('/tmp/api_tts.wav','wb') as f: f.write(aud)
                print(f'  >>> REAL AUDIO saved to /tmp/api_tts.wav ({len(aud)} bytes)')
            else:
                print(f'  EMPTY audio (header only)')
print('API_TEST_DONE')
"
