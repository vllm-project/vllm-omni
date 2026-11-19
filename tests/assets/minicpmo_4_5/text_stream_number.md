# Streaming text and speech fixture

`text_stream_number_16k.wav` says **"The secret number is seven."** It is
synthetic speech generated with `espeak-ng -v en-us -s 140` and resampled with
FFmpeg to mono 16 kHz PCM16. SHA256:
`aa0d189849262dcc2a921740e5bdc124aebb1412e82bed0c2488099063afa413`.

`test_duplex_streaming_text_input[text-and-speech]` sends this recording in
200 ms chunks, interleaved with separate text appends containing the code
word "blue whale". Neither the instructions nor text inputs contain the
spoken number. The response must contain both facts and non-empty audio.
The text-only case supplies the number through another text fragment.
