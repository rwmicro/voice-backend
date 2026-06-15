"""
Latency Metrics Tracker
Tracks p50/p95/p99 latency per TTS provider and STT model
"""

import threading
from collections import defaultdict, deque
from typing import Optional

import numpy as np

from .logger import get_logger

logger = get_logger(__name__)

# Maximum number of samples retained per (category, operation) pair
_ROLLING_WINDOW = 1000


class LatencyTracker:
    """Thread-safe rolling-window latency tracker with percentile statistics.

    Measurements are stored per (category, operation) key in a deque of at
    most *_ROLLING_WINDOW* entries so that only recent data influences the
    reported statistics.

    Typical usage::

        tracker = get_metrics_tracker()
        tracker.record("stt", "large-v3-turbo", 312.5)
        tracker.record("tts", "kokoro", 85.0)
        print(tracker.get_stats())
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # {(category, operation): deque[float]}
        self._data: dict[tuple[str, str], deque] = defaultdict(
            lambda: deque(maxlen=_ROLLING_WINDOW)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, category: str, operation: str, latency_ms: float) -> None:
        """Record a single latency measurement.

        Parameters
        ----------
        category:
            Broad grouping, e.g. ``"tts"`` or ``"stt"``.
        operation:
            Provider or model name, e.g. ``"kokoro"`` or ``"large-v3-turbo"``.
        latency_ms:
            Measured latency in milliseconds.
        """
        key = (category, operation)
        with self._lock:
            self._data[key].append(float(latency_ms))

        logger.debug(
            f"[LatencyTracker] {category}/{operation} -> {latency_ms:.1f} ms"
        )

    def get_stats(
        self,
        category: Optional[str] = None,
        operation: Optional[str] = None,
    ) -> dict:
        """Return descriptive statistics for recorded measurements.

        Parameters
        ----------
        category:
            Filter by category.  Pass ``None`` to include all categories.
        operation:
            Filter by operation.  Pass ``None`` to include all operations.

        Returns
        -------
        dict
            Mapping of ``"<category>/<operation>"`` to a stats dict with
            keys: ``count``, ``mean_ms``, ``p50_ms``, ``p95_ms``,
            ``p99_ms``, ``min_ms``, ``max_ms``.
        """
        with self._lock:
            snapshot = {k: list(v) for k, v in self._data.items()}

        result = {}
        for (cat, op), samples in snapshot.items():
            if category is not None and cat != category:
                continue
            if operation is not None and op != operation:
                continue
            if not samples:
                continue

            arr = np.array(samples, dtype=np.float64)
            result[f"{cat}/{op}"] = {
                "count": len(arr),
                "mean_ms": round(float(arr.mean()), 3),
                "p50_ms": round(float(np.percentile(arr, 50)), 3),
                "p95_ms": round(float(np.percentile(arr, 95)), 3),
                "p99_ms": round(float(np.percentile(arr, 99)), 3),
                "min_ms": round(float(arr.min()), 3),
                "max_ms": round(float(arr.max()), 3),
            }

        return result

    def reset(self, category: Optional[str] = None) -> None:
        """Clear stored measurements.

        Parameters
        ----------
        category:
            When provided, only measurements belonging to *category* are
            erased.  Pass ``None`` to wipe all data.
        """
        with self._lock:
            if category is None:
                self._data.clear()
                logger.info("[LatencyTracker] All metrics reset.")
            else:
                keys_to_delete = [k for k in self._data if k[0] == category]
                for k in keys_to_delete:
                    del self._data[k]
                logger.info(f"[LatencyTracker] Metrics reset for category '{category}'.")


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_tracker_instance: Optional[LatencyTracker] = None
_tracker_lock = threading.Lock()


def get_metrics_tracker() -> LatencyTracker:
    """Return the process-wide :class:`LatencyTracker` singleton.

    Thread-safe – safe to call from multiple threads simultaneously.
    """
    global _tracker_instance
    if _tracker_instance is None:
        with _tracker_lock:
            if _tracker_instance is None:
                _tracker_instance = LatencyTracker()
                logger.debug("[LatencyTracker] Singleton created.")
    return _tracker_instance
