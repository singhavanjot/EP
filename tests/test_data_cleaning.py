"""Tests for Module 2: data cleaning, merging, and peak-hour tagging."""

import numpy as np
import pandas as pd

from src.data_cleaning import clean_aqi_data, merge_aqi_weather, tag_peak_hours


def test_clean_aqi_removes_invalid_placeholders_and_negatives():
    """Invalid negatives, 999 placeholders, and extreme outliers should be cleaned."""
    timestamps = pd.date_range("2026-01-01", periods=12, freq="h")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "aqi": [110, 115, -5, 120, 999, 125, 130, 9999, 135, 140, 145, 150],
        }
    )

    cleaned = clean_aqi_data(df)
    assert not cleaned.empty
    assert (cleaned["aqi"] >= 0).all()
    assert 999 not in cleaned["aqi"].values
    assert 9999 not in cleaned["aqi"].values
    # Single-hour gaps around -5, 999, 9999 should be interpolated smoothly
    assert len(cleaned) == 12


def test_clean_aqi_handles_long_missing_gaps():
    """Gaps longer than 3 hours should be dropped rather than over-interpolated."""
    timestamps = pd.date_range("2026-01-01", periods=10, freq="h")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "aqi": [100, np.nan, np.nan, np.nan, np.nan, np.nan, 110, 115, 120, 125],
        }
    )

    cleaned = clean_aqi_data(df)
    assert cleaned["aqi"].isna().sum() == 0
    assert len(cleaned) < 10


def test_merge_aqi_weather_and_tag_peak_hours():
    """Merging aligns on hourly timestamps and tags weekday peak windows accurately."""
    # 2026-01-05 is a Monday
    timestamps = pd.date_range("2026-01-05 07:00:00", periods=4, freq="h")
    aqi_df = pd.DataFrame({"timestamp": timestamps, "aqi": [90, 140, 150, 110]})
    weather_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature": [15.0, 16.0, 18.0, 20.0],
            "humidity": [70.0, 68.0, 62.0, 58.0],
            "wind_speed": [2.0, 2.2, 2.5, 3.0],
        }
    )

    merged = merge_aqi_weather(aqi_df, weather_df)
    tagged = tag_peak_hours(merged)

    assert list(tagged["is_peak_hour"]) == [False, True, True, False]
    assert list(tagged["traffic_window"]) == ["Off-Peak", "Peak", "Peak", "Off-Peak"]
