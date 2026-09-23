"""app.py – Bounded in-memory stream: tumbling windows with late-event grace.

Compares a no-grace baseline against a watermark-based grace-period processor
on a seeded synthetic event stream.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from data import make_dataset

WINDOW_SIZE = 10.0  # seconds per tumbling window


def _window_index(event_time: float) -> int:
    return int(math.floor(event_time / WINDOW_SIZE))


def _process_stream(
    X: np.ndarray,
    y: np.ndarray,
    use_grace: bool,
) -> tuple[dict[int, list[float]], dict[int, list[float]], int, int]:
    """Process events in arrival order with optional grace period.

    Parameters
    ----------
    X : np.ndarray, shape (n, 3)
        Columns: [arrival_time, event_time, reading].
    y : np.ndarray, shape (n,)
        Ground-truth window indices.
    use_grace : bool
        If True, apply a grace period of one window_size; late events within
        grace are re-inserted into their window.  If False, late events are
        dropped.

    Returns
    -------
    computed_windows : dict[int, list[float]]
        Window index -> list of reading values actually incorporated.
    true_windows : dict[int, list[float]]
        Window index -> list of reading values per ground truth.
    late_total : int
        Total number of late events (event_time < arrival_time).
    late_retained : int
        Number of late events successfully incorporated into their window.
    """
    n = len(X)
    computed_windows: dict[int, list[float]] = defaultdict(list)
    true_windows: dict[int, list[float]] = defaultdict(list)

    # Track the watermark: the maximum event_time seen so far.
    watermark = float("-inf")
    late_total = 0
    late_retained = 0

    # We process in arrival order (already sorted by construction).
    # For grace: when an event arrives with event_time < watermark,
    # it is "late".  If within grace (event_time >= watermark - WINDOW_SIZE),
    # we still incorporate it.  Otherwise drop.
    # For no-grace: any event with event_time < watermark is dropped.

    for i in range(n):
        arrival_time = X[i, 0]
        event_time = X[i, 1]
        reading = X[i, 2]
        wi = _window_index(event_time)

        # Ground-truth: always assign to the correct window.
        true_windows[wi].append(reading)

        is_late = event_time < watermark
        if is_late:
            late_total += 1

        if use_grace:
            # Grace period: accept if event_time >= watermark - WINDOW_SIZE
            if event_time >= watermark - WINDOW_SIZE:
                computed_windows[wi].append(reading)
                if is_late:
                    late_retained += 1
            # else: drop (beyond grace)
        else:
            # No grace: drop if late
            if not is_late:
                computed_windows[wi].append(reading)

        # Update watermark to max event_time seen so far.
        if event_time > watermark:
            watermark = event_time

    return computed_windows, true_windows, late_total, late_retained


def _compute_metrics(
    computed_windows: dict[int, list[float]],
    true_windows: dict[int, list[float]],
    late_total: int,
    late_retained: int,
) -> dict[str, float]:
    """Compute evaluation metrics from window results.

    Returns a dict with:
      - window_mean_mae: mean absolute error of per-window computed means
        vs ground-truth means (over windows present in ground truth).
      - late_retention_rate: fraction of late events retained.
      - max_window_skew: max absolute difference in event counts between
        computed and true windows.
    """
    # Collect all window indices from ground truth.
    all_wis = set(true_windows.keys())

    mae_errors: list[float] = []
    for wi in all_wis:
        true_vals = true_windows[wi]
        true_mean = float(np.mean(true_vals))
        comp_vals = computed_windows.get(wi, [])
        if comp_vals:
            comp_mean = float(np.mean(comp_vals))
        else:
            comp_mean = 0.0  # no data in this window
        mae_errors.append(abs(comp_mean - true_mean))

    window_mean_mae = float(np.mean(mae_errors)) if mae_errors else 0.0

    late_retention_rate = (late_retained / late_total) if late_total > 0 else 1.0

    # Max window skew: max |len(computed) - len(true)| over all windows.
    all_wis_skew = set(true_windows.keys()) | set(computed_windows.keys())
    max_skew = 0
    for wi in all_wis_skew:
        n_comp = len(computed_windows.get(wi, []))
        n_true = len(true_windows.get(wi, []))
        skew = abs(n_comp - n_true)
        if skew > max_skew:
            max_skew = skew

    return {
        "window_mean_mae": window_mean_mae,
        "late_retention_rate": late_retention_rate,
        "max_window_skew": float(max_skew),
    }


def run_experiment(seed: int = 42, n_samples: int = 256) -> dict:
    """Run the bounded stream windowing experiment.

    Parameters
    ----------
    seed : int
        Random seed for dataset generation.
    n_samples : int
        Number of events.  Must be >= 32.

    Returns
    -------
    dict
        JSON-serializable dict with keys:
          - n_samples (int)
          - metrics (dict[str, float]): grace-period processor metrics
          - baseline_metrics (dict[str, float]): no-grace baseline metrics
          - explanation (str)
    """
    X, y = make_dataset(seed=seed, n_samples=n_samples)

    # Grace-period processor (algorithm).
    comp_g, true_g, lt_g, lr_g = _process_stream(X, y, use_grace=True)
    metrics = _compute_metrics(comp_g, true_g, lt_g, lr_g)

    # No-grace baseline.
    comp_b, true_b, lt_b, lr_b = _process_stream(X, y, use_grace=False)
    baseline_metrics = _compute_metrics(comp_b, true_b, lt_b, lr_b)

    explanation = (
        "In-memory simulation of a bounded event stream with tumbling windows "
        f"(window_size={WINDOW_SIZE}s). Events are processed in arrival order. "
        "The grace-period processor accepts late events (event_time < watermark) "
        "if they fall within one window_size of the watermark, re-inserting them "
        "into the correct window. The no-grace baseline drops all late events. "
        "No train/test split is used; this is a single-pass stream computation. "
        "Metrics: window_mean_mae (lower is better), late_retention_rate "
        "(higher is better), max_window_skew (lower is better)."
    )

    return {
        "n_samples": int(n_samples),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "baseline_metrics": {k: float(v) for k, v in baseline_metrics.items()},
        "explanation": explanation,
    }
