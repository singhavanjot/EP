"""Module 8: Generate downloadable summary reports for AirSense findings."""

from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from src.health_advisory import aqi_to_category, get_recommended_action


def generate_summary_report(
    city: str,
    stats: Dict[str, Any],
    forecast: pd.DataFrame,
    alerts: Dict[str, Any],
    partial_corr: Optional[Dict[str, Any]] = None,
    date_range: Optional[str] = None,
) -> str:
    """Generate a formatted Markdown/text summary report for a city.

    Args:
        city: Target city name.
        stats: Output from compute_peak_offpeak_stats().
        forecast: Output DataFrame from predict_next_hours().
        alerts: Output from summarize_alerts().
        partial_corr: Optional output from partial_correlation_controlling_weather().
        date_range: Optional string describing the analyzed date window.

    Returns:
        Report content as a UTF-8 string suitable for download.
    """
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    period_str = date_range or "Recent batch window"

    lines = [
        f"# AirSense AQI Analysis Report — {city}",
        f"Generated: {generated_at}",
        f"Analysis Window: {period_str}",
        "Note: Batch-processed historical data (not a live-streaming feed).",
        "",
        "## 1. Peak vs. Off-Peak Traffic Window Analysis",
        f"- Peak-hour observations: {stats.get('peak_count', 'N/A')} | Mean AQI: {stats.get('peak_mean', 'N/A')} | Median AQI: {stats.get('peak_median', 'N/A')}",
        f"- Off-peak observations: {stats.get('offpeak_count', 'N/A')} | Mean AQI: {stats.get('offpeak_mean', 'N/A')} | Median AQI: {stats.get('offpeak_median', 'N/A')}",
        f"- Mean difference (Peak - Off-Peak): {stats.get('mean_diff', 'N/A')}",
        f"- Statistical test: {stats.get('test_name', 'N/A')} (p-value: {stats.get('p_value', 'N/A')})",
        f"- Finding: {stats.get('interpretation', 'N/A')}",
        "",
    ]

    if partial_corr:
        lines.extend(
            [
                "## 2. Weather-Controlled Partial Correlation",
                f"- Controls: {', '.join(partial_corr.get('controls_used', [])) or 'None'}",
                f"- Partial r: {partial_corr.get('partial_r', 'N/A')} (p-value: {partial_corr.get('p_value', 'N/A')})",
                f"- Finding: {partial_corr.get('interpretation', 'N/A')}",
                "",
            ]
        )

    lines.extend(
        [
            "## 3. Severe-AQI Alerts Summary",
            f"- Total hours analyzed: {alerts.get('total_hours', 0)}",
            f"- Severe/Poor+ hours (AQI >= 201): {alerts.get('severe_hours', 0)} ({alerts.get('severe_pct', 0.0)}%)",
            f"- Severe hours during peak traffic: {alerts.get('peak_severe_hours', 0)} ({alerts.get('peak_severe_share_pct', 0.0)}%)",
            f"- Peak observed AQI: {alerts.get('max_aqi', 'N/A')} at {alerts.get('max_aqi_timestamp', 'N/A')}",
            f"- Summary: {alerts.get('summary_text', 'N/A')}",
            "",
            "## 4. Short-Term Forecast (Next Hours)",
        ]
    )

    if forecast is not None and not forecast.empty:
        peak_fc = float(forecast["yhat"].max())
        min_fc = float(forecast["yhat"].min())
        avg_fc = round(float(forecast["yhat"].mean()), 1)
        fc_cat = aqi_to_category(avg_fc)
        lines.extend(
            [
                f"- Horizon: {len(forecast)} hours ahead",
                f"- Forecasted AQI range: {min_fc:.1f} to {peak_fc:.1f} (mean: {avg_fc:.1f}, {fc_cat})",
                f"- Recommended advisory: {get_recommended_action(fc_cat)}",
                "",
                "| Timestamp | Forecast AQI | Lower (80%) | Upper (80%) | Category |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for _, row in forecast.iterrows():
            cat = aqi_to_category(float(row["yhat"]))
            lines.append(
                f"| {row['ds']} | {row['yhat']:.1f} | {row['yhat_lower']:.1f} | {row['yhat_upper']:.1f} | {cat} |"
            )
    else:
        lines.append("- Forecast unavailable for this selection.")

    return "\n".join(lines) + "\n"
