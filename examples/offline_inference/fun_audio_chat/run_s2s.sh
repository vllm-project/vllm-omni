#!/bin/bash
# Run Fun-Audio-Chat in S2S (Speech-to-Speech) mode on single GPU
# Requires ~35GB VRAM for all 3 stages

python end2end.py --mode s2s --output-dir output_s2s
