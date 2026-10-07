"""Read and display frames from a video file."""

import time

import cv2

from edge_monitor.detector import PersonDetector


def play_video(source: str) -> None:
    capture = cv2.VideoCapture(source)

    # raise an error if the video source cannot be opened
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    detector = PersonDetector()

    frame_count = 0
    start_time = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_count += 1

            result = detector.predict(frame)
            frame = result.plot()
            elapsed = time.perf_counter() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0.0

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Video input", frame)

            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print(f"Displayed {frame_count} frames in {time.perf_counter() - start_time:.2f}s")
