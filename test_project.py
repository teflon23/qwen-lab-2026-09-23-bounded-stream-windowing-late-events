"""test_project.py – Tests for bounded-stream-windowing-late-events project."""

from __future__ import annotations

import numpy as np
import pytest

from data import make_dataset
from app import run_experiment


class TestMakeDataset:
    """Tests for the synthetic dataset generator."""

    def test_different_seeds_produce_different_data(self) -> None:
        """Different seeds must produce different X arrays."""
        X1, _ = make_dataset(seed=1, n_samples=64)
        X2, _ = make_dataset(seed=2, n_samples=64)
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(X1, X2)

    def test_invalid_n_samples_raises_value_error(self) -> None:
        """n_samples < 32 must raise ValueError."""
        with pytest.raises(ValueError):
            make_dataset(seed=42, n_samples=16)

    def test_valid_n_samples_respected(self) -> None:
        """Returned arrays must have length equal to n_samples."""
        X, y = make_dataset(seed=42, n_samples=128)
        assert len(X) == 128
        assert len(y) == 128

    def test_arrival_time_monotonically_non_decreasing(self) -> None:
        """arrival_time column must be monotonically non-decreasing."""
        X, _ = make_dataset(seed=7, n_samples=256)
        arrivals = X[:, 0]
        diffs = np.diff(arrivals)
        assert np.all(diffs >= -1e-12), "arrival_time must be non-decreasing"

    def test_event_time_leq_arrival_time(self) -> None:
        """event_time must be <= arrival_time for every event."""
        X, _ = make_dataset(seed=99, n_samples=256)
        assert np.all(X[:, 1] <= X[:, 0] + 1e-12)


class TestRunExperiment:
    """Tests for the experiment runner and metric properties."""

    def test_output_structure(self) -> None:
        """run_experiment must return required keys with correct types."""
        result = run_experiment(seed=42, n_samples=256)
        assert isinstance(result["n_samples"], int)
        assert result["n_samples"] == 256
        assert isinstance(result["metrics"], dict)
        assert isinstance(result["baseline_metrics"], dict)
        assert len(result["metrics"]) > 0
        assert len(result["baseline_metrics"]) > 0
        for m in (result["metrics"], result["baseline_metrics"]):
            for v in m.values():
                assert isinstance(v, float)
                assert np.isfinite(v)

    def test_grace_period_not_worse_than_baseline(self) -> None:
        """Grace-period metrics must be >= baseline on retention and <= on MAE."""
        result = run_experiment(seed=42, n_samples=256)
        m = result["metrics"]
        b = result["baseline_metrics"]
        assert m["late_retention_rate"] >= b["late_retention_rate"] - 1e-12
        assert m["window_mean_mae"] <= b["window_mean_mae"] + 1e-12

    def test_deterministic_output(self) -> None:
        """Same inputs must produce exactly the same output."""
        r1 = run_experiment(seed=123, n_samples=128)
        r2 = run_experiment(seed=123, n_samples=128)
        assert r1 == r2

    def test_different_n_samples_respected(self) -> None:
        """Different n_samples must be reflected in output."""
        r64 = run_experiment(seed=42, n_samples=64)
        r256 = run_experiment(seed=42, n_samples=256)
        assert r64["n_samples"] == 64
        assert r256["n_samples"] == 256

    def test_edge_case_minimum_n_samples(self) -> None:
        """n_samples=32 (minimum valid) must run without error."""
        result = run_experiment(seed=42, n_samples=32)
        assert result["n_samples"] == 32
        assert len(result["metrics"]) > 0
        assert len(result["baseline_metrics"]) > 0
