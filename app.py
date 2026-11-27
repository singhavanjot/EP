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
from src.data_ingestion import fetch_aqi_data, fetch_weather_data, get_ingestion_status
from src.db import get_last_updated, save_city_data
from src.forecasting import predict_next_hours, train_forecast_model
from src.health_advisory import aqi_to_category, get_category_color, get_recommended_action
from src.report_export import generate_summary_report

st.set_page_config(
    page_title="AirSense — The Air, Made Visible",
    page_icon="🌫️",
    layout="wide",
)


def _inject_atmospheric_theme() -> None:
    """Apply a crisp white, grey, and black editorial theme to Streamlit."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

        :root {
            --as-bg: #FFFFFF;
            --as-surface: #F5F5F5;
            --as-border: #E2E2E2;
            --as-text: #111111;
            --as-muted: #5C5C5C;
            --as-accent: #111111;
        }

        .stApp {
            background-color: var(--as-bg);
            color: var(--as-text);
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }

        [data-testid="stSidebar"] {
            background-color: #F4F4F4;
            border-right: 1px solid var(--as-border);
        }

        h1, h2, .as-display {
            font-family: 'Instrument Serif', Georgia, serif !important;
            font-weight: 400 !important;
            letter-spacing: -0.02em !important;
            color: var(--as-text) !important;
        }

        h1 { font-size: clamp(2.4rem, 4.2vw, 3.6rem) !important; line-height: 1.06 !important; }
        h2, h3, p, label { color: var(--as-text) !important; }

        [data-testid="stMetric"] {
            background: var(--as-surface);
            border: 1px solid var(--as-border);
            border-radius: 10px;
            padding: 14px 16px;
        }

        [data-testid="stMetricValue"] {
            font-family: 'JetBrains Mono', ui-monospace, monospace !important;
            font-variant-numeric: tabular-nums;
            color: #111111 !important;
        }

        .as-eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.74rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #5C5C5C;
            margin-bottom: 6px;
        }

        .as-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #111111;
            display: inline-block;
        }

        .as-subhead {
            color: var(--as-muted) !important;
            font-size: 1.02rem;
            max-width: 64ch;
            margin-bottom: 18px;
        }

        .as-card {
            background: #FFFFFF;
            border: 1px solid #DCDCDC;
            border-radius: 12px;
            padding: 18px 20px;
            height: 100%;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
        }

        .as-mono {
            font-family: 'JetBrains Mono', ui-monospace, monospace;
            font-variant-numeric: tabular-nums;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _style_plotly(fig: go.Figure, height: int = 400) -> go.Figure:
    """Apply consistent white, grey, and black styling to Plotly figures."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FAFAFA",
        font=dict(family="Inter, sans-serif", color="#111111"),
        margin=dict(l=24, r=24, t=44, b=28),
        xaxis=dict(gridcolor="#EAEAEA", zerolinecolor="#D4D4D4"),
        yaxis=dict(gridcolor="#EAEAEA", zerolinecolor="#D4D4D4"),
    )
    return fig


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
    _ = df_hash
    return train_forecast_model(df, city=city, use_cache=True)


def _render_advisory_cards(city_frames: Dict[str, pd.DataFrame]) -> None:
    """Render semi-circular AQI gauge cards with CPCB health advisories."""
    st.subheader("Current AQI & Health Advisory")
    cols = st.columns(len(city_frames))
    for col, (city, df) in zip(cols, city_frames.items()):
        latest = df.sort_values("timestamp").iloc[-1]
        aqi_val = float(latest["aqi"])
        category = aqi_to_category(aqi_val)
        color = get_category_color(category)
        advisory = get_recommended_action(category)
        ts_str = pd.to_datetime(latest["timestamp"]).strftime("%Y-%m-%d %H:%M")
        source_lbl = str(latest.get("source", "batch"))

        pct = min(1.0, max(0.0, aqi_val / 450.0))
        dash_offset = round(251.2 * (1.0 - pct), 1)
        angle_deg = round(-90.0 + pct * 180.0, 1)

        with col:
            st.markdown(
                f"""
                <div class="as-card" style="border-top: 3px solid #111111;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:600; font-size:1.05rem; color:#111111;">{city}</span>
                        <span class="as-mono" style="font-size:0.72rem; color:#5C5C5C; border:1px solid #D4D4D4; background:#F5F5F5; padding:2px 8px; border-radius:999px;">{source_lbl}</span>
                    </div>
                    <div style="display:flex; flex-direction:column; align-items:center; margin:8px 0 6px;">
                        <svg width="190" height="108" viewBox="0 0 200 112" aria-label="{city} AQI gauge">
                            <path d="M20,100 A80,80 0 0,1 180,100" fill="none" stroke="#E5E5E5" stroke-width="12" stroke-linecap="round" />
                            <path d="M20,100 A80,80 0 0,1 180,100" fill="none" stroke="#262626" stroke-width="12" stroke-linecap="round" stroke-dasharray="251.2" stroke-dashoffset="{dash_offset}" />
                            <g transform="rotate({angle_deg} 100 100)">
                                <line x1="100" y1="100" x2="100" y2="30" stroke="#111111" stroke-width="2.2" stroke-linecap="round" />
                                <circle cx="100" cy="100" r="5" fill="#111111" />
                            </g>
                        </svg>
                        <div class="as-mono" style="font-size:2.15rem; font-weight:600; line-height:1; margin-top:-10px; color:#111111;">{aqi_val:.0f} <span style="font-size:0.9rem; color:#5C5C5C;">AQI</span></div>
                        <div style="font-weight:600; color:#111111; margin-top:4px;">{category}</div>
                        <div class="as-mono" style="font-size:0.72rem; color:#5C5C5C; margin-top:2px;">Latest: {ts_str}</div>
                    </div>
                    <div style="font-size:0.86rem; color:#525252; line-height:1.45; border-top:1px solid #E5E5E5; padding-top:10px;">
                        {advisory}
                    </div>
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
        color_discrete_sequence=["#111111", "#666666", "#A3A3A3"],
        labels={"timestamp": "Timestamp", "aqi": "AQI", "city": "City"},
        title="Hourly AQI Comparison",
    )

    primary_df = city_frames[primary_city]
    peak_pts = primary_df[primary_df["is_peak_hour"]]
    if not peak_pts.empty:
        fig.add_trace(
            go.Scatter(
                x=peak_pts["timestamp"],
                y=peak_pts["aqi"],
                mode="markers",
                name=f"{primary_city} Peak Hours",
                marker=dict(size=5, color="#111111", opacity=0.75, symbol="circle"),
            )
        )

    fig.add_hline(y=201, line_dash="dash", line_color="#525252", annotation_text="Poor / Alert Threshold (201)")
    _style_plotly(fig, height=420)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, width="stretch")


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
    st.caption("Methodology note: Peak traffic is approximated by weekday rush-hour windows (8–10 AM, 5–8 PM), not live GPS.")
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
            line=dict(color="#111111", width=2.2),
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
            fillcolor="rgba(0, 0, 0, 0.10)",
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
            line=dict(color="#525252", width=2.5, dash="dash"),
        )
    )
    _style_plotly(fig, height=380)
    fig.update_layout(xaxis_title="Timestamp", yaxis_title="AQI")
    st.plotly_chart(fig, width="stretch")
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
        color_continuous_scale="Greys",
        aspect="auto",
    )
    _style_plotly(fig, height=360)
    st.plotly_chart(fig, width="stretch")


def _render_alerts_panel(primary_city: str, df: pd.DataFrame):
    """Render severe-AQI alerts summary and flagged hours table."""
    st.subheader(f"Severe-AQI Alerts (Poor or Worse, AQI >= 201) — {primary_city}")
    alerts_summary = summarize_alerts(df)
    flagged = flag_severe_hours(df)

    st.warning(alerts_summary["summary_text"])
    if not flagged.empty:
        display_cols = [c for c in ["timestamp", "aqi", "category", "traffic_window", "wind_speed"] if c in flagged.columns]
        st.dataframe(flagged[display_cols].head(50), width="stretch")
    else:
        st.success("No hours crossed the Poor/Severe AQI threshold in this window.")
    return alerts_summary


def _render_project_info_tabs() -> None:
    """Render concise CPCB scale, architecture, and Chitkara team tabs."""
    st.markdown("---")
    st.subheader("About AirSense — Architecture, CPCB Scale & Team")
    tab1, tab2, tab3 = st.tabs(["Pipeline & Tech Stack", "CPCB 6-Band Scale", "Team & References"])

    with tab1:
        st.markdown(
            "**Pipeline:** `CPCB / OpenAQ v3 / OWM / Open-Meteo` → `IQR Cleaning & Peak Tagging (Pandas, SQLite)` "
            "→ `Weather-Controlled Correlation (SciPy, Scikit-learn)` → `24h Prophet Forecast` → `CPCB Advisories & Alerts`"
        )
        st.caption("Stack: Python 3.10+, Pandas, NumPy, SciPy, Scikit-learn, Prophet, Plotly, Streamlit, SQLite, pytest.")

    with tab2:
        bands = [
            ("Good (0–50)", "#D4D4D4", "Minimal impact; safe for normal outdoor activity."),
            ("Satisfactory (51–100)", "#A3A3A3", "Minor breathing discomfort to sensitive people."),
            ("Moderate (101–200)", "#737373", "Discomfort for lung/heart patients, children, older adults."),
            ("Poor (201–300)", "#525252", "Breathing discomfort on prolonged exposure; limit outdoor exertion."),
            ("Very Poor (301–400)", "#262626", "Respiratory illness risk; keep windows closed."),
            ("Severe (401+)", "#111111", "Health emergency; stay indoors and avoid physical exertion."),
        ]
        cols = st.columns(3)
        for idx, (label, color, desc) in enumerate(bands):
            with cols[idx % 3]:
                st.markdown(
                    f'<div class="as-card" style="border-left:4px solid {color}; margin-bottom:10px;">'
                    f'<strong style="color:#111111;">{label}</strong>'
                    f'<div style="font-size:0.84rem; color:#5C5C5C; margin-top:4px;">{desc}</div></div>',
                    unsafe_allow_html=True,
                )

    with tab3:
        st.markdown(
            "**B.E. CSE Exploratory Project — Chitkara University, Punjab**\n\n"
            "- **Avanjot Singh** (`2310991800`) — Data Engineering\n"
            "- **Devarsh Bajaj** (`2310991814`) — Correlation Analysis & Forecasting\n"
            "- **Akshat Goel** (`2310990355`) — Health Advisory, Alerts & Testing\n"
            "- **Agrim Gupta** (`2310990331`) — Dashboard & Visualization\n\n"
            "References: Sidhu et al. (2024), Natarajan et al. (2024), Liu et al. (2018), "
            "Taylor & Letham (2018, Prophet), Schneider (2003), CPCB, OpenAQ, OpenWeatherMap, Scikit-learn, Streamlit."
        )


def main() -> None:
    """Run the AirSense Streamlit dashboard."""
    _inject_atmospheric_theme()
    st.markdown(
        '<div class="as-eyebrow"><span class="as-dot"></span>AirSense · The Air, Made Visible</div>',
        unsafe_allow_html=True,
    )
    st.title("See what you're breathing. Know what's next.")
    st.markdown(
        '<p class="as-subhead">AirSense connects weekday traffic patterns to air quality across Ludhiana, '
        "Chandigarh, and Amritsar, and forecasts the next 24 hours so numbers become decisions.</p>",
        unsafe_allow_html=True,
    )

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

    if st.sidebar.button("🔄 Refresh Batch Data", width="stretch"):
        load_city_pipeline.clear()
        get_cached_model.clear()
        st.rerun()

    city_frames: Dict[str, pd.DataFrame] = {}
    last_updated_map: Dict[str, str] = {}
    with st.spinner("Loading batch AQI & weather datasets..."):
        for city in selected_cities:
            df_city, updated_ts = load_city_pipeline(city, start_str, end_str)
            if not df_city.empty:
                city_frames[city] = df_city
                last_updated_map[city] = updated_ts

    if primary_city not in city_frames:
        st.error(f"No data available for {primary_city}.")
        st.stop()

    primary_df = city_frames[primary_city]
    aqi_source = str(primary_df["source"].iloc[-1]) if "source" in primary_df.columns else "unknown"
    wx_source = str(primary_df["weather_source"].iloc[-1]) if "weather_source" in primary_df.columns else "unknown"

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Batch Refresh Mode:** Data is pulled periodically in batch (daily/weekly), "
        "not via live streaming.\n\n"
        f"**Last Updated ({primary_city}):** `{last_updated_map.get(primary_city, 'N/A')}`\n\n"
        f"**AQI Source:** `{aqi_source}`\n\n"
        f"**Weather Source:** `{wx_source}`"
    )

    diag = get_ingestion_status(primary_city)
    if diag:
        with st.sidebar.expander("API Diagnostics"):
            for provider, msg in diag.items():
                st.caption(f"**{provider}:** {msg}")

    # 2. Current AQI + Health Advisory Card (Semi-Circular Gauges)
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

    # 9. Architecture, CPCB Scale & Team Tabs
    _render_project_info_tabs()


if __name__ == "__main__":
    main()
