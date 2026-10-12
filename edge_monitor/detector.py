"""Run person detection with a pretrained YOLO model."""

from ultralytics import YOLO


class PersonDetector:
    def __init__(self, model_path: str = "yolo26n.pt") -> None:
        self.model = YOLO(model_path)

    def predict(self, frame):
        return self.model.predict(
            source=frame,
            device=0,
            imgsz=640,
            quantize=32,
            classes=[0],
            verbose=False,
        )[0]

    def track(self, frame):
        return self.model.track(
            source=frame,
            device=0,
            imgsz=640,
            classes=[0],
            persist=True,
            verbose=False,
        )[0]
