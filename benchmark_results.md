# Benchmark Results

## Baseline: PyTorch FP32

- GPU: NVIDIA GeForce RTX 2060 SUPER, 8 GB VRAM
- Model: YOLO26n
- Input: `data/people_sample.mp4`
- Input size: 640
- Warm-up: 10 frames per run
- Runs: 3
- Latency measured: `detector.track()` only; CUDA synchronized around the call

| Run | Median latency (ms) | P95 latency (ms) | Median FPS | Frames | Person boxes | Tracked frames |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.0 | 16.3 | 66.5 | 189 | 423 | 92 |
| 2 | 15.3 | 17.6 | 65.4 | 189 | 423 | 92 |
| 3 | 15.4 | 17.1 | 64.8 | 189 | 423 | 92 |