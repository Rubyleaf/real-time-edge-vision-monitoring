"""Evaluate whether detection centers are inside a restricted zone."""

import cv2
import numpy as np


def point_in_zone(
    point: tuple[int, int],
    polygon: list[tuple[int, int]],
) -> bool:
    """Return True when point is inside or on the polygon boundary."""
    contour = np.array(polygon, dtype=np.int32)
    result = cv2.pointPolygonTest(contour, point, False) # pointPolygonTest 可以判定點在多邊形內、外或邊界上
    return result >= 0
