"""Run repeated local tracking latency measurements."""

import time

import cv2
import torch

from edge_monitor.detector import PersonDetector
from edge_monitor.metrics import summarize_latencies


SOURCE = "data/people_sample.mp4"
RUNS = 3
WARMUP_FRAMES = 10


def measure_once(source: str) -> dict[str, float]:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    detector = PersonDetector()
    latencies_ms = []
    person_boxes_total = 0
    frames_with_track_ids = 0
    frame_count = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_count += 1

            torch.cuda.synchronize()
            start = time.perf_counter()
            result = detector.track(frame)
            torch.cuda.synchronize()

            latency_ms = (time.perf_counter() - start) * 1000

            if frame_count > WARMUP_FRAMES:
                latencies_ms.append(latency_ms)

            if result.boxes is not None and len(result.boxes) > 0:
                person_boxes_total += len(result.boxes)

                if result.boxes.id is not None:
                    frames_with_track_ids += 1
    finally:
        capture.release()

    if not latencies_ms:
        raise RuntimeError("No measured frames. Check the video and warm-up count.")

    summary = summarize_latencies(latencies_ms)
    summary["median_fps"] = 1000.0 / summary["median_ms"]
    summary["frames"] = float(frame_count)
    summary["person_boxes"] = float(person_boxes_total)
    summary["frames_with_track_ids"] = float(frames_with_track_ids)
    return summary


def main() -> None:
    for run_number in range(1, RUNS + 1):
        result = measure_once(SOURCE)
        print(
            f"Run {run_number}/{RUNS}: "
            f"median={result['median_ms']:.1f} ms, "
            f"p95={result['p95_ms']:.1f} ms",
            f"median_fps={result['median_fps']:.1f}, "
            f"frames={int(result['frames'])}, "
            f"person_boxes={int(result['person_boxes'])}, "
            f"tracked_frames={int(result['frames_with_track_ids'])}, "
        )


if __name__ == "__main__":
    main()