# JoyAI spoken instruction

`color_instruction_16k.wav` contains this synthetic instruction:

> For every new image, say the color of only the newest image in English.
> If it is green, also count from one to fifty aloud, saying every number without skipping any.

The ASR duplex E2E submits it with the first image, then submits an image-only
observation to check that the transcribed instruction remains available. It
repeats the audio with a green image, cancels after the model begins replying,
and checks that a fresh text query with the next image still produces speech. The
counting instruction keeps the cancelled response longer than a one-word color
answer, giving the client time to send the cancellation.
The fixture is mono PCM16 at 16 kHz (9.8 seconds). It was generated with
FFmpeg's Flite `slt` voice:

```bash
ffmpeg -f lavfi -i "flite=text='For every new image, say the color of only the newest image in English. If it is green, also count from one to fifty aloud, saying every number without skipping any.':voice=slt" \
  -ar 16000 -ac 1 -c:a pcm_s16le color_instruction_16k.wav
```
