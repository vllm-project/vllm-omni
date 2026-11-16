This is an unpublished experimental Gradio6.31 Pillow13 backport.

Local distribution version: 6.31.0+omni.pillow13
Base: official PyPI Gradio6.31.0 source archive, SHA256
40cf094fe8ea1b8ada0fc4c10b685c4c5f09c4061d34c073be25bab342e14e22
Base upstream source:07341609c68bb98701e7db9ba9e2ad6c0fb2acb4

Pillow compatibility changes are backported from the Gradio maintainers'
Apache-2.0 code merged in PR14005 at d6972ed9d1185634cea083e0263839543d5ceb8d:
https://github.com/gradio-app/gradio/pull/14005

Changes: permit Pillow>=8,<14; reject I-mode PNG encoding with the exact
upstream conversion/TIFF guidance; identify this artifact with a local version.
The original Apache-2.0 license and bundled third-party licenses remain intact.
This artifact is not an official Gradio release and has no upstream support
claim. Small CPU probes do not qualify the full Gradio/demos/model feature set.
