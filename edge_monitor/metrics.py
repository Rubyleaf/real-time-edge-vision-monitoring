"""Measure latency, throughput, and resource usage."""
"""Summarize performance measurements."""

import numpy as np


def summarize_latencies(latencies_ms: list[float]) -> dict[str, float]:
    """Return mean, median, and p95 latency in milliseconds."""
    if not latencies_ms:
        raise ValueError("latencies_ms must not be empty")

    values = np.asarray(latencies_ms, dtype=float)

    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
    }