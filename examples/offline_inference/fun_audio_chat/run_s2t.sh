#!/bin/bash
# Run Fun-Audio-Chat in S2T (Speech-to-Text) mode
# This uses only Stage 0 (Main model) for audio understanding

python end2end.py --mode s2t --output-dir output_s2t
