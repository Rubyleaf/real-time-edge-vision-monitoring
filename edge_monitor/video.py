"""Read and display frames from a video file."""

import time

import cv2

from edge_monitor.detector import PersonDetector

import numpy as np

from edge_monitor.zone import point_in_zone

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
        previous_inside: dict[int, bool] = {}
        while True:
            ok, frame = capture.read()
            if not ok:
                break
                
            frame_count += 1

            result = detector.track(frame)

            # debug: print the track IDs of detected objects
            # if result.boxes is not None and result.boxes.id is not None:
            #     print("Track IDs:", result.boxes.id.int().cpu().tolist())

            frame = result.plot()

            intrusion_detected = False

            entered_track_ids = []

            # Check if any detected person is inside the zone
            if result.boxes is not None:
                boxes_xyxy = result.boxes.xyxy.cpu().tolist()

                if result.boxes.id is not None:
                    track_ids = result.boxes.id.int().cpu().tolist()
                else:
                    track_ids = [None] * len(boxes_xyxy)

                for coords, track_id in zip(boxes_xyxy, track_ids):
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
                            entered_track_ids.append(track_id)

                        previous_inside[track_id] = inside

            for track_id in entered_track_ids:
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
            
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print(f"Displayed {frame_count} frames in {time.perf_counter() - start_time:.2f}s")
