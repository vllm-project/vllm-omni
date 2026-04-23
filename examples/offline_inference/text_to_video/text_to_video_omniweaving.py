from __future__ import annotations

from examples.offline_inference.omniweaving.end2end import build_parser, main


def parse_args():
    parser = build_parser()
    parser.description = "Text-to-video offline inference example for OmniWeaving."
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
