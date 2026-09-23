"""data.py – Synthetic bounded in-memory event stream for tumbling-window analysis.

The dataset is a sequence of sensor readings with:
  * an arrival timestamp (monotonically non-decreasing),
  * an event timestamp (may lag arrival, simulating late delivery),
  * a numeric reading value.

Ground-truth window means are computed by grouping on the *event* timestamp
into fixed tumbling windows of size `window_size` seconds.
"""

from __future__ import annotations

import numpy as np


def make_dataset(seed: int = 42, n_samples: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Create a seeded synthetic event stream.

    Parameters
    ----------
    seed : int
        Random seed controlling all stochastic choices.
    n_samples : int
        Number of events to generate.  Must be >= 32.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 3)
        Columns: [arrival_time, event_time, reading_value].
        All values are floats.  arrival_time is monotonically non-decreasing.
        event_time <= arrival_time (late events have event_time < arrival_time).
    y : np.ndarray, shape (n_samples,)
        Per-event ground-truth window index (int) derived from event_time.
        Specifically, y[i] = floor(event_time[i] / window_size).

    Raises
    ------
    ValueError
        If n_samples < 32.
    """
    if n_samples < 32:
        raise ValueError(f"n_samples must be >= 32, got {n_samples}")

    rng = np.random.default_rng(seed)

    window_size = 10.0  # seconds per tumbling window

    # Arrival times: start at 0, each gap is exponential with mean 0.5 s.
    gaps = rng.exponential(scale=0.5, size=n_samples)
    arrival_time = np.cumsum(gaps)
    arrival_time -= arrival_time[0]  # normalise so first arrival is ~0

    # Latency: most events arrive on time (latency=0), ~25 % are late.
    # Late events have event_time = arrival_time - latency, where latency
    # is drawn from a truncated exponential (max 1.5 * window_size).
    is_late = rng.random(n_samples) < 0.25
    latency = np.zeros(n_samples)
    if is_late.any():
        raw = rng.exponential(scale=window_size * 0.4, size=int(is_late.sum()))
        raw = np.clip(raw, 0.0, window_size * 1.5)
        latency[is_late] = raw
    event_time = arrival_time - latency

    # Reading values: smooth sinusoid + noise so window means are meaningful.
    t = event_time
    reading = 5.0 + 2.0 * np.sin(2.0 * np.pi * t / 60.0) + rng.normal(0.0, 0.3, n_samples)

    X = np.column_stack([arrival_time, event_time, reading]).astype(np.float64)

    # Ground-truth window index per event.
    y = (event_time // window_size).astype(np.int64)

    return X, y
