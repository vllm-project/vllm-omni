output=$(curl -X POST -s http://localhost:8091/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "'"Qwen/Qwen2.5-Omni-7B"'",
"messages":[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Explain the system architecture for a scalable audio generation pipeline. Answer in 15 words."}
                ]
            }
        ]
}')

echo "Output of request: $output"
