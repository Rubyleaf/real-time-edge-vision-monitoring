"""Read and display frames from a video file."""

import time

import cv2

import numpy as np
import torch

from edge_monitor.detector import PersonDetector
from edge_monitor.zone import point_in_zone
from edge_monitor.event_log import log_intrusion

def play_video(source: str) -> None:
    capture = cv2.VideoCapture(source)

    # raise an error if the video source cannot be opened
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = PersonDetector()
    zone = [
        (width // 3, height // 3),
        (2 * width // 3, height // 3),
        (2 * width // 3, 2 * height // 3),
        (width // 3, 2 * height // 3),
    ]
    zone_points = np.array(zone, dtype=np.int32)

    frame_count = 0
    start_time = time.perf_counter()

    # Create a window for displaying the video
    window_name = "Video input"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # cv2.WINDOW_NORMAL allows the window to be resized
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.moveWindow(window_name, 100, 100) # Move the window to a specific position on the screen

    try:
        inference_latencies_ms = []
        previous_inside: dict[int, bool] = {}
        while True:
            ok, frame = capture.read()
            if not ok:
                break
                
            frame_count += 1

            torch.cuda.synchronize() # Wait for all CUDA kernels to finish before measuring inference time
            inference_start = time.perf_counter() # Start measuring inference time

            result = detector.track(frame) # Run person detection and tracking on the current frame

            torch.cuda.synchronize() # Wait for the inference to finish before measuring the time
            inference_ms = (time.perf_counter() - inference_start) * 1000 # time.perf_counter() returns the time in seconds, so we multiply by 1000 to convert to milliseconds

            if frame_count > 10:  # 前 10 幀作為 warm-up，不納入統計
                inference_latencies_ms.append(inference_ms)

            # debug: print the track IDs of detected objects
            # if result.boxes is not None and result.boxes.id is not None:
            #     print("Track IDs:", result.boxes.id.int().cpu().tolist())

            frame = result.plot()

            intrusion_detected = False
            entered_events = []

            # Check if any detected person is inside the zone
            if result.boxes is not None:
                boxes_xyxy = result.boxes.xyxy.cpu().tolist()
                confidences = result.boxes.conf.cpu().tolist()

                if result.boxes.id is not None:
                    track_ids = result.boxes.id.int().cpu().tolist()
                else:
                    track_ids = [None] * len(boxes_xyxy)

                for coords, track_id, confidence in zip(boxes_xyxy, track_ids, confidences):
                    x1, y1, x2, y2 = coords
                    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    inside = point_in_zone(center, zone)

                    if inside:
                        intrusion_detected = True

                    # Draw a circle at the center of the detected person
                    cv2.circle(frame, center, 8, (255, 0, 0), -1)

                    if track_id is not None:
                        was_inside = previous_inside.get(track_id, False)

                        if inside and not was_inside:
                            entered_events.append((track_id, coords, center, confidence))

                        previous_inside[track_id] = inside

            for track_id, bbox, center, confidence in entered_events:
                log_intrusion(
                    track_id=track_id,
                    bbox=bbox,
                    center=center,
                    confidence=confidence,
                )
                print(f"INTRUSION entered: track_id={track_id}")
            
            cv2.polylines(frame, [zone_points], isClosed=True, color=(0, 0, 255), thickness=2)
            elapsed = time.perf_counter() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0.0

            # Resize the frame for display
            display_frame = cv2.resize(
                frame,
                (1280, 720),
                interpolation=cv2.INTER_AREA,
            )

            if intrusion_detected:
                cv2.putText(
                    display_frame,
                    "INTRUSION",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3,
                )
                # print("INTRUSION detected")            

            cv2.putText(
                display_frame,
                f"FPS: {fps:.1f}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                2,
            )

            cv2.imshow(window_name, display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print(f"Displayed {frame_count} frames in {time.perf_counter() - start_time:.2f}s")
    if inference_latencies_ms:
        print(
            "track() latency after warm-up: "
            f"median={np.median(inference_latencies_ms):.1f} ms, "
            f"p95={np.percentile(inference_latencies_ms, 95):.1f} ms"
        )
