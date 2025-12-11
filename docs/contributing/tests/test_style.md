###online serving


```python
models = ["Qwen/{模型名称}"] #指定对应模型
stage_configs = [str(Path(__file__).parent / "stage_configs" / {模型yaml})] #指定对应模型配置
test_params = [(model, stage_config) for model in models for stage_config in stage_configs]

#OmniServer类，用于拉起OmniServer
class OmniServer:
    xxx



#根据指定参数拉起omni_server
@pytest.fixture
def omni_server(request):
    model, stage_config_path = request.param
    with OmniServer(model, ["--stage-configs-path", stage_config_path]) as server:
        yield server
    
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_video_to_audio(
    client: openai.OpenAI, #指定client类型
    omni_server,
    base64_encoded_video: str,
) -> None:
    #set message
    video_data_url = f"data:video/mp4;base64, {base64_encoded_video}"
    messages = dummy_messages_from_video_data(video_data_url)
    
    #send request
    chat_completion = client.chat.completions.create(
        model=omni_server.model,
        messages=messages,
    )
    
    #verify
    


def test_text_to_audio(

)

def test_audio_to_audio(

)
}
```