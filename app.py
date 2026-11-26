"""Module 7: AirSense Interactive Streamlit Dashboard."""

from datetime import date, timedelta
from typing import Dict, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import CITIES
from src.alerts import flag_severe_hours, summarize_alerts
from src.correlation_analysis import (
    compute_peak_offpeak_stats,
    partial_correlation_controlling_weather,
)
from src.data_cleaning import run_cleaning_pipeline
from src.data_ingestion import fetch_aqi_data, fetch_weather_data
from src.db import get_last_updated, save_city_data
from src.forecasting import predict_next_hours, train_forecast_model
from src.health_advisory import aqi_to_category, get_category_color, get_recommended_action
from src.report_export import generate_summary_report

st.set_page_config(
    page_title="AirSense — Multi-City AQI Dashboard",
    page_icon="🌫️",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_city_pipeline(city: str, start_date: str, end_date: str) -> Tuple[pd.DataFrame, str]:
    """Ingest, clean, merge, tag, and store AQI + weather data for a city."""
    aqi_df = fetch_aqi_data(city, start_date, end_date)
    weather_df = fetch_weather_data(city, start_date, end_date)
    cleaned_df = run_cleaning_pipeline(aqi_df, weather_df, city=city, save=True)
    save_city_data(city, cleaned_df)
    last_updated = get_last_updated(city) or "Just now"
    return cleaned_df, last_updated


@st.cache_resource(show_spinner=False)
def get_cached_model(city: str, df_hash: int, df: pd.DataFrame):
    """Train or load a cached Prophet model for a city."""
    _ = df_hash  # cache key invalidation when data changes
    return train_forecast_model(df, city=city, use_cache=True)


def _render_advisory_cards(city_frames: Dict[str, pd.DataFrame]) -> None:
    """Render color-coded Current AQI + Health Advisory cards for selected cities."""
    st.subheader("Current AQI & Health Advisory")
    cols = st.columns(len(city_frames))
    for col, (city, df) in zip(cols, city_frames.items()):
        latest = df.sort_values("timestamp").iloc[-1]
        aqi_val = float(latest["aqi"])
        category = aqi_to_category(aqi_val)
        color = get_category_color(category)
        advisory = get_recommended_action(category)
        ts_str = pd.to_datetime(latest["timestamp"]).strftime("%Y-%m-%d %H:%M")

        text_color = "#111111" if category in ("Satisfactory", "Moderate", "Poor") else "#FFFFFF"
        with col:
            st.markdown(
                f"""
                <div style="background-color:{color}; color:{text_color}; padding:16px; border-radius:10px;">
                    <div style="font-size:1.1rem; font-weight:700;">{city}</div>
                    <div style="font-size:2.4rem; font-weight:800; margin:4px 0;">{aqi_val:.0f} AQI</div>
                    <div style="font-size:1.05rem; font-weight:600;">{category}</div>
                    <div style="font-size:0.8rem; opacity:0.85; margin-bottom:8px;">Latest reading: {ts_str}</div>
                    <div style="font-size:0.9rem;">{advisory}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_historical_trend(city_frames: Dict[str, pd.DataFrame], primary_city: str) -> None:
    """Render Plotly historical AQI trend chart with peak-hour highlight overlay."""
    st.subheader("Historical AQI Trend (with Peak-Hour Highlight Overlay)")
    combined = pd.concat(
        [df.assign(city=city) for city, df in city_frames.items()],
        ignore_index=True,
    )
    fig = px.line(
        combined,
        x="timestamp",
        y="aqi",
        color="city",
        labels={"timestamp": "Timestamp", "aqi": "AQI", "city": "City"},
        title="Hourly AQI Comparison",
    )

    # Overlay peak-hour markers for the primary city
    primary_df = city_frames[primary_city]
    peak_pts = primary_df[primary_df["is_peak_hour"]]
    if not peak_pts.empty:
        fig.add_trace(
            go.Scatter(
                x=peak_pts["timestamp"],
                y=peak_pts["aqi"],
                mode="markers",
                name=f"{primary_city} Peak Hours",
                marker=dict(size=5, color="rgba(220, 38, 38, 0.55)", symbol="circle"),
            )
        )

    fig.add_hline(y=201, line_dash="dash", line_color="orange", annotation_text="Poor / Alert Threshold (201)")
    fig.update_layout(height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)


def _render_correlation_panel(primary_city: str, df: pd.DataFrame):
    """Render peak vs. off-peak stats and weather-controlled partial correlation."""
    st.subheader(f"Peak vs. Off-Peak Traffic Correlation — {primary_city}")
    stats = compute_peak_offpeak_stats(df)
    partial_corr = partial_correlation_controlling_weather(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Peak Mean AQI", f"{stats['peak_mean']}" if stats["peak_mean"] is not None else "N/A")
    c2.metric("Off-Peak Mean AQI", f"{stats['offpeak_mean']}" if stats["offpeak_mean"] is not None else "N/A")
    c3.metric(
        "Mean Difference",
        f"{stats['mean_diff']:+.1f}" if stats["mean_diff"] is not None else "N/A",
    )
    c4.metric(
        "Weather-Controlled Partial r",
        f"{partial_corr['partial_r']:.3f}" if partial_corr["partial_r"] is not None else "N/A",
    )

    st.info(f"**Statistical Comparison:** {stats['interpretation']}")
    st.info(f"**Controlling for Weather:** {partial_corr['interpretation']}")
    return stats, partial_corr


def _render_forecast_panel(primary_city: str, df: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    """Render Prophet short-term AQI forecast with confidence band."""
    st.subheader(f"Short-Term AQI Forecast ({horizon_hours}-Hour Horizon) — {primary_city}")
    df_hash = int(pd.util.hash_pandas_object(df[["timestamp", "aqi"]]).sum())
    model = get_cached_model(primary_city, df_hash, df)
    forecast_df = predict_next_hours(model, hours=horizon_hours)

    eval_metrics = getattr(model, "eval_metrics_", {})
    if eval_metrics.get("mae") is not None:
        st.caption(
            f"Time-aware validation ({eval_metrics.get('holdout_hours')}h held-out tail): "
            f"MAE = {eval_metrics['mae']} | RMSE = {eval_metrics['rmse']}"
        )

    recent_tail = df.sort_values("timestamp").tail(48)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recent_tail["timestamp"],
            y=recent_tail["aqi"],
            mode="lines",
            name="Recent Observed AQI",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["ds"],
            y=forecast_df["yhat_upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["ds"],
            y=forecast_df["yhat_lower"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(255, 127, 14, 0.2)",
            line=dict(width=0),
            name="Confidence Band",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["ds"],
            y=forecast_df["yhat"],
            mode="lines+markers",
            name="Forecasted AQI",
            line=dict(color="#ff7f0e", width=2.5),
        )
    )
    fig.update_layout(height=380, xaxis_title="Timestamp", yaxis_title="AQI")
    st.plotly_chart(fig, use_container_width=True)
    return forecast_df


def _render_seasonal_heatmap(primary_city: str, df: pd.DataFrame) -> None:
    """Render day-of-week x hour-of-day average AQI heatmap."""
    st.subheader(f"Day-of-Week × Hour-of-Day AQI Heatmap — {primary_city}")
    heat_df = df.copy()
    ts = pd.to_datetime(heat_df["timestamp"])
    heat_df["day_name"] = ts.dt.day_name()
    heat_df["hour"] = ts.dt.hour

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = (
        heat_df.pivot_table(index="day_name", columns="hour", values="aqi", aggfunc="mean")
        .reindex(day_order)
    )

    fig = px.imshow(
        pivot,
        labels=dict(x="Hour of Day", y="Day of Week", color="Avg AQI"),
        color_continuous_scale="YlOrRd",
        aspect="auto",
    )
    fig.update_layout(height=360)
    st.plotly_chart(fig, use_container_width=True)


def _render_alerts_panel(primary_city: str, df: pd.DataFrame):
    """Render severe-AQI alerts summary and flagged hours table."""
    st.subheader(f"Severe-AQI Alerts (Poor or Worse, AQI >= 201) — {primary_city}")
    alerts_summary = summarize_alerts(df)
    flagged = flag_severe_hours(df)

    st.warning(alerts_summary["summary_text"])
    if not flagged.empty:
        display_cols = [c for c in ["timestamp", "aqi", "category", "traffic_window", "wind_speed"] if c in flagged.columns]
        st.dataframe(flagged[display_cols].head(50), use_container_width=True)
    else:
        st.success("No hours crossed the Poor/Severe AQI threshold in this window.")
    return alerts_summary


def main() -> None:
    """Run the AirSense Streamlit dashboard."""
    st.title("AirSense — Multi-City Air Quality & Traffic-Window Dashboard")

    # 1. Sidebar: City selector & batch metadata
    st.sidebar.header("Configuration")
    available_cities: List[str] = list(CITIES.keys())
    selected_cities = st.sidebar.multiselect(
        "Select Cities (compare 1–3)",
        options=available_cities,
        default=available_cities[:3],
        max_selections=3,
    )
    if not selected_cities:
        st.warning("Please select at least one city in the sidebar.")
        st.stop()

    default_end = date.today()
    default_start = default_end - timedelta(days=21)
    date_range = st.sidebar.date_input(
        "Historical Date Range",
        value=(default_start, default_end),
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_str, end_str = date_range[0].strftime("%Y-%m-%d"), date_range[1].strftime("%Y-%m-%d")
    else:
        start_str, end_str = default_start.strftime("%Y-%m-%d"), default_end.strftime("%Y-%m-%d")

    horizon_hours = st.sidebar.slider("Forecast Horizon (hours)", min_value=6, max_value=24, value=24, step=6)
    primary_city = st.sidebar.selectbox("Primary City for Deep-Dive & Report", options=selected_cities, index=0)

    city_frames: Dict[str, pd.DataFrame] = {}
    last_updated_map: Dict[str, str] = {}
    with st.spinner("Loading batch AQI & weather datasets..."):
        for city in selected_cities:
            df_city, updated_ts = load_city_pipeline(city, start_str, end_str)
            if not df_city.empty:
                city_frames[city] = df_city
                last_updated_map[city] = updated_ts

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Batch Refresh Mode:** Data is pulled periodically in batch (daily/weekly), "
        "not via live streaming.\n\n"
        f"**Last Updated ({primary_city}):** `{last_updated_map.get(primary_city, 'N/A')}`"
    )

    if primary_city not in city_frames:
        st.error(f"No data available for {primary_city}.")
        st.stop()

    primary_df = city_frames[primary_city]

    # 2. Current AQI + Health Advisory Card
    _render_advisory_cards(city_frames)

    # 3. Historical Trend Chart
    _render_historical_trend(city_frames, primary_city)

    # 4. Peak vs. Off-Peak Correlation Panel
    stats, partial_corr = _render_correlation_panel(primary_city, primary_df)

    # 5. Forecast Panel
    forecast_df = _render_forecast_panel(primary_city, primary_df, horizon_hours)

    # 6. Seasonal Heatmap
    _render_seasonal_heatmap(primary_city, primary_df)

    # 7. Alerts Panel
    alerts_summary = _render_alerts_panel(primary_city, primary_df)

    # 8. Download Report Button
    st.subheader("Export Summary Report")
    report_md = generate_summary_report(
        city=primary_city,
        stats=stats,
        forecast=forecast_df,
        alerts=alerts_summary,
        partial_corr=partial_corr,
        date_range=f"{start_str} to {end_str}",
    )
    st.download_button(
        label=f"Download {primary_city} Report (.md)",
        data=report_md,
        file_name=f"airsense_{primary_city.lower()}_{start_str}_to_{end_str}.md",
        mime="text/markdown",
    )


if __name__ == "__main__":
    main()
