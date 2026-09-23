# Bounded In-Memory Stream: Tumbling Windows with Late-Event Grace

This project is a small, CPU-only educational simulation of a bounded in-memory event stream. It demonstrates how a watermark-based grace period can improve the accuracy of tumbling-window aggregations when sensor readings arrive late.

The project is **not** a production-ready streaming system. It does not use Kafka, Flink, or any external service. All processing occurs in a single Python process using seeded synthetic data.

## Problem

In real-time data pipelines, events often arrive out of order due to network latency or sensor delays. When processing a stream with tumbling windows (fixed-size time buckets), a naive processor that drops any event whose timestamp is older than the current "watermark" will lose data. This project compares two strategies:

1.  **Baseline (No Grace):** Drops any event that arrives late (i.e., its `event_time` is less than the current maximum `event_time` seen so far).
2.  **Algorithm (Grace Period):** Allows a grace period of one window size. Late events are retained and incorporated into their correct window if they arrive within the grace period.

## Implementation

The core logic is implemented in `app.py`:

-   **`_process_stream`**: Iterates through events in arrival order. It tracks the `watermark` (the maximum `event_time` seen so far).
    -   If `use_grace` is `True`, an event is accepted if `event_time >= watermark - WINDOW_SIZE`.
    -   If `use_grace` is `False`, an event is accepted only if `event_time >= watermark`.
-   **`_compute_metrics`**: Calculates the performance metrics by comparing the computed window aggregates against the ground-truth window aggregates.

The dataset is generated in `data.py`:

-   **`make_dataset`**: Generates a synthetic stream of sensor readings.
    -   `arrival_time`: Monotonically non-decreasing timestamps.
    -   `event_time`: Timestamps that may lag behind `arrival_time` to simulate late delivery.
    -   `reading_value`: Numeric sensor values generated from a smooth sinusoid plus noise.

## Architecture

```
data.py
  └── make_dataset(seed, n_samples) -> (X, y)
        X: [arrival_time, event_time, reading_value]
        y: [ground_truth_window_index]

app.py
  ├── _process_stream(X, y, use_grace) -> (computed_windows, true_windows, late_total, late_retained)
  ├── _compute_metrics(computed_windows, true_windows, late_total, late_retained) -> metrics_dict
  └── run_experiment(seed, n_samples) -> result_dict

test_project.py
  └── Pytest tests for data generation and experiment logic

main.py
  └── CLI entry point (supplied by host)
```

## Synthetic Dataset Assumptions

-   **Seed Determinism**: The dataset is fully determined by the `seed` and `n_samples` arguments. Different seeds produce different streams.
-   **Latency Model**: ~25% of events are late. Latency is drawn from a truncated exponential distribution, capped at 1.5 times the window size.
-   **Signal Model**: Reading values follow a sinusoidal pattern with added Gaussian noise, ensuring that window means are meaningful and distinct.
-   **Limitations**: This is a simplified model. It does not account for infinite streams, backpressure, or complex watermarking strategies (e.g., allowed lateness based on event time distribution).

## Algorithm vs. Baseline

| Feature | Baseline (No Grace) | Algorithm (Grace Period) |
| :--- | :--- | :--- |
| **Late Event Handling** | Drops all late events. | Retains late events within 1 window size of the watermark. |
| **Window Accuracy** | Lower accuracy due to missing data. | Higher accuracy by including late data. |
| **Complexity** | Simpler logic. | Slightly more complex logic to check grace bounds. |

## Metrics

-   **`window_mean_mae`** (lower is better): Mean Absolute Error between the computed window means and the ground-truth window means.
-   **`late_retention_rate`** (higher is better): Fraction of late events that were successfully incorporated into their correct window.
-   **`max_window_skew`** (lower is better): Maximum absolute difference in the number of events between computed and true windows.

## Reproducibility

The project is designed to be fully reproducible. Given the same `seed` and `n_samples`, the output will be identical.

### Setup

Requires Python 3.11–3.13.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Running the Experiment

```bash
python main.py --seed 42 --n-samples 256 --output results.json
```

### Running Tests

```bash
python -m pytest -q
```

## Limitations

-   **In-Memory Simulation**: This is not a true distributed streaming system. It simulates a bounded stream in a single process.
-   **No Network/External Services**: All data is synthetic and generated locally.
-   **Simplified Watermarking**: The watermark is simply the maximum event time seen so far. Real systems may use more sophisticated watermarking strategies.
-   **No CI/CD**: There is no continuous integration workflow or hosted service.

## Disclaimer

This project was generated by Qwen. Automated tests verify basic functionality and reproducibility but do not prove correctness in all edge cases or production environments. Refer to `validation_report.json` and `example_results.json` for measured outputs. Do not use this code in production without thorough review and additional testing.

## Recorded automated validation

Host contract tests and project tests passed (14 tests, 0 skipped). Demo completed on Python 3.13.15. See `validation_report.json` and `example_results.json`. These checks validate the execution contract, not scientific novelty or every algorithmic claim.
