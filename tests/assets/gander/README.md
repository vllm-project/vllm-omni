# Gander visual regression questions

These fixtures contain questions without an answer, transcript hint or visual
description. Tests pair the same recording with opposing images or motion and
check the model's response, so completing the streaming protocol alone cannot
satisfy them.

| File | Spoken question |
| --- | --- |
| `color_question_16k.wav` | What color is the square in the picture? Reply with the color. |
| `motion_question_16k.wav` | Which direction did the ball move, left or right? |
| `ocr_question_16k.wav` | Read the number shown in the picture. |

Generated with espeak-ng, `-v en-us -s 145`, then converted with FFmpeg to
mono PCM16 at 16 kHz (`-ar 16000 -ac 1 -c:a pcm_s16le`). The images and ordered
video subframes are generated in the E2E module. Camera stacking excludes each
unit's base frame from its subframe composite, matching the shared video driver.
These small synthetic cases guard visual conditioning; dataset-level accuracy
and performance are measured separately with OmniInteract.
