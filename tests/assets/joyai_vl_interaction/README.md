# JoyAI spoken instruction

`color_instruction_16k.wav` contains this synthetic instruction:

> For every new image, say the color of only the newest image in English.

The ASR duplex E2E submits it with the first image, then submits an image-only
observation to check that the transcribed instruction remains available. It
repeats the audio on a later turn, cancels after the model begins replying,
and checks that another image-only observation still produces speech.
The fixture is mono PCM16 at 16 kHz (4.1 seconds). It was generated with
FFmpeg's Flite `slt` voice:

```bash
ffmpeg -f lavfi -i "flite=text='For every new image, say the color of only the newest image in English.':voice=slt" \
  -ar 16000 -ac 1 -c:a pcm_s16le color_instruction_16k.wav
```
