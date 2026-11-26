"""Module 6: Severe-AQI hour flagging and alert summarization."""

from typing import Any, Dict

import pandas as pd

from config import SEVERE_AQI_THRESHOLD
from src.health_advisory import aqi_to_category


def flag_severe_hours(df: pd.DataFrame, threshold: float = SEVERE_AQI_THRESHOLD) -> pd.DataFrame:
    """Filter and tag rows where AQI crosses into 'Poor' or worse (>= 201).

    Args:
        df: Cleaned DataFrame containing 'timestamp' and 'aqi'.
        threshold: AQI threshold for flagging (default: config.SEVERE_AQI_THRESHOLD).

    Returns:
        DataFrame of flagged severe-AQI hours with an added 'category' column.
    """
    if df.empty or "aqi" not in df.columns:
        return pd.DataFrame(columns=list(df.columns) + ["category"])

    flagged = df.loc[df["aqi"] >= threshold].copy()
    if flagged.empty:
        flagged["category"] = pd.Series(dtype=str)
        return flagged

    flagged["category"] = flagged["aqi"].apply(aqi_to_category)
    return flagged.sort_values("timestamp", ascending=False).reset_index(drop=True)


def summarize_alerts(df: pd.DataFrame, threshold: float = SEVERE_AQI_THRESHOLD) -> Dict[str, Any]:
    """Compute summary statistics for severe-AQI alerts.

    Args:
        df: Cleaned DataFrame containing 'timestamp', 'aqi', and optionally 'is_peak_hour'.
        threshold: AQI threshold for flagging.

    Returns:
        Dictionary with counts, percentages, peak-hour share, and summary text.
    """
    total_hours = len(df)
    if total_hours == 0 or "aqi" not in df.columns:
        return {
            "total_hours": 0,
            "severe_hours": 0,
            "severe_pct": 0.0,
            "peak_severe_hours": 0,
            "peak_severe_share_pct": 0.0,
            "max_aqi": None,
            "max_aqi_timestamp": None,
            "summary_text": "No observations available to evaluate severe-AQI alerts.",
        }

    flagged = flag_severe_hours(df, threshold=threshold)
    severe_hours = len(flagged)
    severe_pct = round((severe_hours / total_hours) * 100.0, 1)

    if severe_hours > 0 and "is_peak_hour" in flagged.columns:
        peak_severe_hours = int(flagged["is_peak_hour"].sum())
        peak_severe_share_pct = round((peak_severe_hours / severe_hours) * 100.0, 1)
    else:
        peak_severe_hours = 0
        peak_severe_share_pct = 0.0

    max_idx = df["aqi"].idxmax()
    max_aqi = float(df.loc[max_idx, "aqi"])
    max_ts = str(df.loc[max_idx, "timestamp"]) if "timestamp" in df.columns else None

    if severe_hours == 0:
        summary_text = f"0 of {total_hours} hours crossed into Poor or worse AQI (>= {int(threshold)})."
    else:
        summary_text = (
            f"{severe_hours} of {total_hours} hours ({severe_pct}%) reached Poor or worse AQI; "
            f"{peak_severe_share_pct}% of those severe hours occurred during peak traffic windows."
        )

    return {
        "total_hours": total_hours,
        "severe_hours": severe_hours,
        "severe_pct": severe_pct,
        "peak_severe_hours": peak_severe_hours,
        "peak_severe_share_pct": peak_severe_share_pct,
        "max_aqi": max_aqi,
        "max_aqi_timestamp": max_ts,
        "summary_text": summary_text,
    }
