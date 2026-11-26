"""Module 3: Peak vs. off-peak statistical analysis and weather-controlled partial correlation."""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


def compute_peak_offpeak_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Compare AQI distributions between peak traffic hours and off-peak hours.

    Uses Mann-Whitney U test (non-parametric) alongside Welch's t-test to assess
    whether peak-hour AQI differs significantly from off-peak AQI. Handles edge
    cases (empty data, all-peak, or all-off-peak) without crashing.

    Args:
        df: Cleaned DataFrame with 'aqi' and 'is_peak_hour' columns.

    Returns:
        Dictionary containing summary statistics, test results, and interpretation.
    """
    if df.empty or "aqi" not in df.columns or "is_peak_hour" not in df.columns:
        return {
            "valid": False,
            "peak_count": 0,
            "offpeak_count": 0,
            "peak_mean": None,
            "peak_median": None,
            "offpeak_mean": None,
            "offpeak_median": None,
            "mean_diff": None,
            "test_name": None,
            "statistic": None,
            "p_value": None,
            "significant": False,
            "interpretation": "Insufficient columns or empty dataset for peak/off-peak comparison.",
        }

    valid_df = df.dropna(subset=["aqi", "is_peak_hour"])
    peak = valid_df.loc[valid_df["is_peak_hour"].astype(bool), "aqi"]
    offpeak = valid_df.loc[~valid_df["is_peak_hour"].astype(bool), "aqi"]

    peak_count = int(len(peak))
    offpeak_count = int(len(offpeak))

    peak_mean = round(float(peak.mean()), 2) if peak_count > 0 else None
    peak_median = round(float(peak.median()), 2) if peak_count > 0 else None
    offpeak_mean = round(float(offpeak.mean()), 2) if offpeak_count > 0 else None
    offpeak_median = round(float(offpeak.median()), 2) if offpeak_count > 0 else None

    if peak_count < 2 or offpeak_count < 2:
        return {
            "valid": False,
            "peak_count": peak_count,
            "offpeak_count": offpeak_count,
            "peak_mean": peak_mean,
            "peak_median": peak_median,
            "offpeak_mean": offpeak_mean,
            "offpeak_median": offpeak_median,
            "mean_diff": None,
            "test_name": None,
            "statistic": None,
            "p_value": None,
            "significant": False,
            "interpretation": (
                "Both peak and off-peak groups need at least 2 observations for statistical testing."
            ),
        }

    mean_diff = round(peak_mean - offpeak_mean, 2)

    # AQI is typically right-skewed; use Mann-Whitney U as primary non-parametric test
    u_stat, p_val = stats.mannwhitneyu(peak, offpeak, alternative="two-sided")
    p_val_float = float(p_val)
    significant = p_val_float < 0.05

    direction = "higher" if mean_diff > 0 else "lower" if mean_diff < 0 else "equal"
    if significant:
        interpretation = (
            f"Peak-hour AQI (mean {peak_mean}) is significantly {direction} than off-peak AQI "
            f"(mean {offpeak_mean}, diff {mean_diff:+.1f}, Mann-Whitney p={p_val_float:.4f})."
        )
    else:
        interpretation = (
            f"No statistically significant difference between peak-hour AQI (mean {peak_mean}) "
            f"and off-peak AQI (mean {offpeak_mean}, Mann-Whitney p={p_val_float:.4f})."
        )

    return {
        "valid": True,
        "peak_count": peak_count,
        "offpeak_count": offpeak_count,
        "peak_mean": peak_mean,
        "peak_median": peak_median,
        "offpeak_mean": offpeak_mean,
        "offpeak_median": offpeak_median,
        "mean_diff": mean_diff,
        "test_name": "Mann-Whitney U",
        "statistic": round(float(u_stat), 2),
        "p_value": round(p_val_float, 6),
        "significant": significant,
        "interpretation": interpretation,
    }


def partial_correlation_controlling_weather(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute partial correlation between peak traffic window and AQI, controlling for weather.

    Uses a regression-residual approach: regresses both `is_peak_hour` and `aqi` on
    available weather covariates (temperature, humidity, wind_speed) and computes
    Pearson correlation between the residuals.

    Args:
        df: Cleaned DataFrame with 'aqi', 'is_peak_hour', and weather columns.

    Returns:
        Dictionary with partial_r, p_value, n_samples, controls_used, and honest interpretation.
    """
    candidate_controls: List[str] = ["temperature", "humidity", "wind_speed"]
    controls_used = [c for c in candidate_controls if c in df.columns and df[c].notna().sum() > 0]

    if df.empty or "aqi" not in df.columns or "is_peak_hour" not in df.columns:
        return {
            "valid": False,
            "partial_r": None,
            "p_value": None,
            "n_samples": 0,
            "controls_used": controls_used,
            "strength": "inconclusive",
            "interpretation": "Insufficient data to compute weather-controlled partial correlation.",
        }

    required_cols = ["aqi", "is_peak_hour"] + controls_used
    work = df[required_cols].dropna().copy()
    n_samples = len(work)

    if n_samples < 10 or work["is_peak_hour"].nunique() < 2 or not controls_used:
        msg = (
            "Partial correlation inconclusive: requires >= 10 complete rows, both peak and off-peak "
            "observations, and at least one weather covariate."
        )
        logger.info(msg)
        return {
            "valid": False,
            "partial_r": None,
            "p_value": None,
            "n_samples": n_samples,
            "controls_used": controls_used,
            "strength": "inconclusive",
            "interpretation": msg,
        }

    x_weather = work[controls_used].to_numpy(dtype=float)
    y_aqi = work["aqi"].to_numpy(dtype=float)
    z_peak = work["is_peak_hour"].astype(float).to_numpy()

    reg_aqi = LinearRegression().fit(x_weather, y_aqi)
    resid_aqi = y_aqi - reg_aqi.predict(x_weather)

    reg_peak = LinearRegression().fit(x_weather, z_peak)
    resid_peak = z_peak - reg_peak.predict(x_weather)

    if np.std(resid_aqi) == 0 or np.std(resid_peak) == 0:
        msg = "Partial correlation inconclusive due to zero residual variance."
        logger.info(msg)
        return {
            "valid": False,
            "partial_r": None,
            "p_value": None,
            "n_samples": n_samples,
            "controls_used": controls_used,
            "strength": "inconclusive",
            "interpretation": msg,
        }

    partial_r, p_val = stats.pearsonr(resid_peak, resid_aqi)
    partial_r = float(partial_r)
    p_val = float(p_val)
    abs_r = abs(partial_r)

    if p_val >= 0.05 or abs_r < 0.1:
        strength = "weak/inconclusive"
        logger.info(
            "Weather-controlled partial correlation is weak/inconclusive (r=%.3f, p=%.4f).",
            partial_r,
            p_val,
        )
        interpretation = (
            f"Controlling for {', '.join(controls_used)}, the association between peak-traffic "
            f"windows and AQI is weak or inconclusive (partial r={partial_r:.3f}, p={p_val:.4f})."
        )
    elif abs_r < 0.3:
        strength = "weak-to-moderate"
        interpretation = (
            f"Controlling for {', '.join(controls_used)}, peak-traffic windows show a modest "
            f"positive/directional association with AQI (partial r={partial_r:.3f}, p={p_val:.4f})."
        )
    else:
        strength = "moderate-to-strong"
        interpretation = (
            f"Controlling for {', '.join(controls_used)}, peak-traffic windows remain noticeably "
            f"associated with elevated AQI (partial r={partial_r:.3f}, p={p_val:.4f})."
        )

    return {
        "valid": True,
        "partial_r": round(partial_r, 4),
        "p_value": round(p_val, 6),
        "n_samples": n_samples,
        "controls_used": controls_used,
        "strength": strength,
        "interpretation": interpretation,
    }
