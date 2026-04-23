from __future__ import annotations

from examples.offline_inference.omniweaving.end2end import build_parser, main


def parse_args():
    parser = build_parser()
    parser.description = "Image-to-video offline inference example for OmniWeaving."
    parser.add_argument(
        "--image",
        dest="image_path",
        type=str,
        default=None,
        help="Alias of --image-path for consistency with other image-to-video examples.",
    )
    args = parser.parse_args()
    if not args.image_path and not args.image_paths:
        parser.error("image-to-video mode requires --image-path/--image or --image-paths.")
    return args


if __name__ == "__main__":
    main(parse_args())
