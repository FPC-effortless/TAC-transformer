# TAC v0.2 Demo

Composition: `TacV02Demo`

Rendered local output:

```text
demos/tac-v02-demo/out/tac-v02-demo.mp4
```

The video is a 5-minute walkthrough at 10 fps. It covers:

- TAC architecture
- TAC-235 native program bottleneck
- TAC-251 realistic context compression
- TAC-272 causal fix disambiguation
- TAC-274 / v0.2 112M scale-gate protocol

Render command:

```bash
npx remotion render TacV02Demo out/tac-v02-demo.mp4 --codec=h264 --crf=32 --scale=0.5 --concurrency=1 --image-format=jpeg --jpeg-quality=60 --browser-executable="C:\Program Files\Google\Chrome\Application\chrome.exe"
```

The low-footprint render flags are intentional because this machine had less
than 1 GB free on `C:` during rendering.

