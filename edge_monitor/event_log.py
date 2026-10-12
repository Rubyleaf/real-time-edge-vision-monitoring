"""Write intrusion events as JSON Lines."""

from datetime import datetime
import json
from pathlib import Path


def log_intrusion(
    track_id: int,
    bbox: list[float],
    center: tuple[int, int],
    confidence: float,
    log_path: str = "logs/events.jsonl",
) -> None:
    """Append one intrusion event to a JSONL file."""
    event = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": "intrusion",
        "track_id": track_id,
        "bbox": [round(float(value), 1) for value in bbox],
        "center": list(center),
        "confidence": round(float(confidence), 3),
    }

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event) + "\n")