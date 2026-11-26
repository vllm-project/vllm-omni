# MiniCPM-o 4.5 server-VAD barge-in fixtures

These synthetic English utterances were generated with eSpeak NG 1.50 using
`espeak-ng -v en-us -s 160 -w speech.wav "$utterance"` and encoded as mono
16 kHz PCM16 FLAC with FFmpeg 6.1.1. The test adds 100 ms of leading silence
and 1,200 ms of trailing silence and generates one camera frame per audio unit.

| File | Camera frame | Utterance |
| --- | --- | --- |
| `turn1.flac` | Blue circle | Look at this circle. What color is it? Remember the color. Say only the color. |
| `turn2.flac` | Red circle | Please count out loud from one to one hundred, one number after another. Start at one. |
| `turn3.flac` | Green square | Stop counting. Look at the square now. What color is it? Say only the color. |
| `turn4.flac` | Yellow square | What was the color of the circle in the very first picture? What color is the square now? Say both colors in order. |
