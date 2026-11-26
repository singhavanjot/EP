"""Tests for Module 3: peak vs. off-peak stats and weather-controlled partial correlation."""

import pandas as pd

from src.correlation_analysis import (
    compute_peak_offpeak_stats,
    partial_correlation_controlling_weather,
)
from src.data_cleaning import tag_peak_hours
from src.data_ingestion import generate_synthetic_data


def test_correlation_returns_expected_structure():
    """Both correlation functions return structured dicts on synthetic data."""
    df = generate_synthetic_data("Ludhiana", "2026-01-01", "2026-01-10", seed=123)
    df = tag_peak_hours(df)

    stats = compute_peak_offpeak_stats(df)
    assert stats["valid"] is True
    for key in ("peak_mean", "offpeak_mean", "mean_diff", "p_value", "significant", "interpretation"):
        assert key in stats

    partial = partial_correlation_controlling_weather(df)
    assert partial["valid"] is True
    for key in ("partial_r", "p_value", "n_samples", "controls_used", "strength", "interpretation"):
        assert key in partial
    assert -1.0 <= partial["partial_r"] <= 1.0


def test_correlation_handles_all_peak_edge_case():
    """All-peak dataset should not crash and should report invalid/insufficient cleanly."""
    df = pd.DataFrame(
        {
            "aqi": [150.0, 160.0, 170.0, 180.0],
            "is_peak_hour": [True, True, True, True],
            "temperature": [20.0, 21.0, 22.0, 23.0],
            "humidity": [55.0, 54.0, 53.0, 52.0],
            "wind_speed": [2.5, 2.6, 2.7, 2.8],
        }
    )

    stats = compute_peak_offpeak_stats(df)
    assert stats["valid"] is False
    assert stats["offpeak_count"] == 0

    partial = partial_correlation_controlling_weather(df)
    assert partial["valid"] is False
    assert partial["partial_r"] is None


def test_correlation_handles_all_offpeak_edge_case():
    """All-off-peak dataset should not crash and should report invalid/insufficient cleanly."""
    df = pd.DataFrame(
        {
            "aqi": [90.0, 95.0, 100.0, 105.0],
            "is_peak_hour": [False, False, False, False],
            "temperature": [20.0, 21.0, 22.0, 23.0],
            "humidity": [55.0, 54.0, 53.0, 52.0],
            "wind_speed": [2.5, 2.6, 2.7, 2.8],
        }
    )

    stats = compute_peak_offpeak_stats(df)
    assert stats["valid"] is False
    assert stats["peak_count"] == 0

    partial = partial_correlation_controlling_weather(df)
    assert partial["valid"] is False
