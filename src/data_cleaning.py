"""Module 2: Data cleaning, merging, and peak-hour tagging."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import PEAK_HOUR_WINDOWS, PROCESSED_DATA_DIR, WEEKDAYS_ONLY_PEAK

logger = logging.getLogger(__name__)


def clean_aqi_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw AQI readings by removing invalid values, placeholders, and outliers.

    Steps:
    - Parse and sort timestamps, dropping invalid timestamps.
    - Replace common sensor placeholders (999, 9999, -999) and negative values with NaN.
    - Remove extreme outliers beyond 3.0 * IQR from Q1/Q3 (or > 500 CPCB ceiling).
    - Interpolate short gaps (up to 3 hours) and drop remaining missing AQI rows.

    Args:
        df: Raw AQI DataFrame containing at least 'timestamp' and 'aqi' columns.

    Returns:
        Cleaned DataFrame sorted by timestamp.
    """
    if df.empty:
        return df.copy()

    cleaned = df.copy()
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce")
    cleaned = cleaned.dropna(subset=["timestamp"]).sort_values("timestamp")
    cleaned = cleaned.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    cleaned["aqi"] = pd.to_numeric(cleaned["aqi"], errors="coerce")

    # Remove negative readings, CPCB impossible values, and sensor placeholders
    invalid_mask = (cleaned["aqi"] < 0) | (cleaned["aqi"].isin([999, 999.0, 9999, 9999.0])) | (cleaned["aqi"] > 500)
    cleaned.loc[invalid_mask, "aqi"] = np.nan

    # IQR-based outlier removal on valid readings
    valid_aqi = cleaned["aqi"].dropna()
    if len(valid_aqi) >= 8:
        q1 = valid_aqi.quantile(0.25)
        q3 = valid_aqi.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            lower_bound = max(0.0, q1 - 3.0 * iqr)
            upper_bound = min(500.0, q3 + 3.0 * iqr)
            outlier_mask = (cleaned["aqi"] < lower_bound) | (cleaned["aqi"] > upper_bound)
            cleaned.loc[outlier_mask, "aqi"] = np.nan

    # Interpolate small gaps (max 3 consecutive missing hours), then drop any remaining NaNs
    cleaned["aqi"] = cleaned["aqi"].interpolate(method="linear", limit=3, limit_area="inside")
    cleaned = cleaned.dropna(subset=["aqi"]).reset_index(drop=True)
    cleaned["aqi"] = cleaned["aqi"].round(1)

    return cleaned


def _normalize_timestamps(series: pd.Series, target_tz: str = "Asia/Kolkata") -> pd.Series:
    """Normalize timestamps to timezone-naive local time at hourly resolution."""
    ts = pd.to_datetime(series, errors="coerce")
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert(target_tz).dt.tz_localize(None)
    return ts.dt.floor("h")


def merge_aqi_weather(aqi_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Align and merge cleaned AQI and weather DataFrames on hourly timestamps.

    Args:
        aqi_df: Cleaned AQI DataFrame.
        weather_df: Weather DataFrame with temperature, humidity, wind_speed.

    Returns:
        Merged hourly DataFrame with consistent timestamps.
    """
    if aqi_df.empty:
        return aqi_df.copy()

    left = aqi_df.copy()
    left["timestamp"] = _normalize_timestamps(left["timestamp"])
    left = left.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")

    if weather_df.empty:
        for col in ("temperature", "humidity", "wind_speed"):
            if col not in left.columns:
                left[col] = np.nan
        return left.reset_index(drop=True)

    right = weather_df.copy()
    right["timestamp"] = _normalize_timestamps(right["timestamp"])
    right = right.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")

    weather_cols = ["timestamp"] + [c for c in ("temperature", "humidity", "wind_speed") if c in right.columns]
    merged = pd.merge(left, right[weather_cols], on="timestamp", how="left", suffixes=("", "_wx"))

    # Fill any short weather gaps via interpolation
    for col in ("temperature", "humidity", "wind_speed"):
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").interpolate(method="linear", limit=3)

    return merged.reset_index(drop=True)


def tag_peak_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each row as Peak or Off-Peak traffic window based on config.py rules.

    Args:
        df: DataFrame with a 'timestamp' column.

    Returns:
        DataFrame with 'is_peak_hour' (bool) and 'traffic_window' ('Peak'/'Off-Peak') columns.
    """
    if df.empty:
        tagged = df.copy()
        tagged["is_peak_hour"] = pd.Series(dtype=bool)
        tagged["traffic_window"] = pd.Series(dtype=str)
        return tagged

    tagged = df.copy()
    ts = pd.to_datetime(tagged["timestamp"])
    hours = ts.dt.hour
    is_weekday = ts.dt.weekday < 5

    in_window = pd.Series(False, index=tagged.index)
    for start_h, end_h in PEAK_HOUR_WINDOWS:
        in_window = in_window | ((hours >= start_h) & (hours < end_h))

    if WEEKDAYS_ONLY_PEAK:
        is_peak = in_window & is_weekday
    else:
        is_peak = in_window

    tagged["is_peak_hour"] = is_peak.astype(bool)
    tagged["traffic_window"] = np.where(tagged["is_peak_hour"], "Peak", "Off-Peak")
    return tagged


def save_cleaned_data(df: pd.DataFrame, city: str) -> Path:
    """Save cleaned and tagged dataset to data/processed/{city}_cleaned.csv.

    Args:
        df: Cleaned, merged, and tagged DataFrame.
        city: City name used for the output filename.

    Returns:
        Path to the saved CSV file.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    safe_city = city.lower().replace(" ", "_")
    output_path = PROCESSED_DATA_DIR / f"{safe_city}_cleaned.csv"
    df.to_csv(output_path, index=False)
    return output_path


def run_cleaning_pipeline(
    aqi_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    city: Optional[str] = None,
    save: bool = True,
) -> pd.DataFrame:
    """Execute full cleaning, merging, peak-tagging, and optional CSV persistence.

    Args:
        aqi_df: Raw AQI DataFrame.
        weather_df: Raw weather DataFrame.
        city: Optional city name for saving processed CSV.
        save: Whether to persist the cleaned CSV to data/processed/.

    Returns:
        Cleaned, merged, and peak-tagged DataFrame.
    """
    cleaned_aqi = clean_aqi_data(aqi_df)
    merged = merge_aqi_weather(cleaned_aqi, weather_df)
    tagged = tag_peak_hours(merged)
    if save and city:
        save_cleaned_data(tagged, city)
    return tagged
