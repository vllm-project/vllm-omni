cd /data/zeyu_bao/vllm-omni

curl --noproxy '*' -fsS http://127.0.0.1:30340/health
curl --noproxy '*' -fsS http://127.0.0.1:30341/health

output="$PWD/outputs/serial50_gpu_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$output"

nohup bash examples/cosmos3/run_serial_profile.sh \
 base 4 30100 http://127.0.0.1:30340 "$output" 50 \
 >"$output/base.controller.log" 2>&1 < /dev/null &

nohup bash examples/cosmos3/run_serial_profile.sh \
 faster 5 30110 http://127.0.0.1:30341 "$output" 50 \
 >"$output/faster.controller.log" 2>&1 < /dev/null &

echo "$output"