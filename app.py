"""Module 7: AirSense Interactive Streamlit Dashboard."""

from datetime import date, timedelta
from typing import Dict, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

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
    page_title="AirSense — Multi-City AQI Dashboard",
    page_icon="🌫️",
    layout="wide",
)


def _inject_ink_orbit_theme() -> None:
    """Apply Ink-Orbit architectural styling (hatched accents, crisp borders, mono tags)."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }
        code, pre, [data-testid="stMetricValue"] {
            font-family: 'JetBrains Mono', ui-monospace, monospace;
        }
        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 4px;
            padding: 12px 14px;
        }
        .ib-header-tag {
            display: inline-block;
            padding: 3px 10px;
            font-size: 11.5px;
            font-family: 'JetBrains Mono', monospace;
            letter-spacing: 0.04em;
            border: 1px solid rgba(128, 128, 128, 0.35);
            background: rgba(128, 128, 128, 0.08);
            margin-bottom: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_ink_orbit_bento(
    primary_city: str,
    forecast_df: pd.DataFrame,
    aqi_source: str,
    wx_source: str,
    alerts_summary: Dict,
) -> None:
    """Render an interactive Ink-Orbit-style bento overview wired to live AirSense data."""
    vals = (
        [round(float(v), 1) for v in forecast_df["yhat"].tolist()[:12]]
        if forecast_df is not None and not forecast_df.empty
        else [110, 124, 138, 155, 142, 128, 118, 132, 149, 162, 150, 136]
    )
    if len(vals) < 2:
        vals = [vals[0], vals[0]] if vals else [100, 110]

    severe_count = int(alerts_summary.get("severe_hours", 0))
    total_hours = int(alerts_summary.get("total_hours", 0))
    vals_json = str(vals)

    html = f"""
    <!doctype html>
    <html>
    <head>
    <meta charset="utf-8" />
    <style>
      :root {{
        --page:#efefef; --hatch:rgba(0,0,0,.06); --paper:#fbfbfb; --card:#f4f4f4;
        --raise:#ffffff; --ink:#151515; --soft:#3d3d3d; --muted:#7b7b7b;
        --faint:#a8a8a8; --line:#e2e2e2; --line-strong:#cfcfcf; --bracket:#c9c9c9;
        --inv:#161616; --inv-ink:#f5f5f5;
        --sans:"Inter",system-ui,-apple-system,sans-serif;
        --mono:"JetBrains Mono",ui-monospace,monospace;
      }}
      @media (prefers-color-scheme: dark) {{
        :root {{
          --page:#0b0b0b; --hatch:rgba(255,255,255,.05); --paper:#121212; --card:#181818;
          --raise:#1e1e1e; --ink:#eeeeee; --soft:#c9c9c9; --muted:#8d8d8d;
          --faint:#5d5d5d; --line:#262626; --line-strong:#363636; --bracket:#444444;
          --inv:#efefef; --inv-ink:#121212;
        }}
      }}
      * {{ box-sizing:border-box; margin:0; padding:0; }}
      body {{
        font-family:var(--sans); color:var(--ink);
        background-color:var(--page);
        background-image:repeating-linear-gradient(135deg,var(--hatch) 0 1px,transparent 1px 10px);
        padding:18px;
      }}
      .sec {{
        background:var(--paper); border:1px solid var(--line);
        padding:24px; max-width:1180px; margin:0 auto;
      }}
      .head {{ text-align:center; margin-bottom:22px; }}
      .tag {{
        display:inline-block; padding:3px 10px; font-size:11px; font-family:var(--mono);
        color:var(--soft); background:var(--card); border:1px solid var(--line); margin-bottom:8px;
      }}
      .h2 {{ font-size:28px; font-weight:600; letter-spacing:-.02em; line-height:1.12; }}
      .muted {{ color:var(--faint); }}
      .bento {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
      @media (max-width:700px) {{ .bento {{ grid-template-columns:1fr; }} }}
      .wide {{ grid-column:1 / -1; }}
      .frame {{ position:relative; }}
      .c {{ position:absolute; width:10px; height:10px; border:0 solid var(--bracket); pointer-events:none; transition:.25s; }}
      .tl {{ top:-5px; left:-5px; border-top-width:1.5px; border-left-width:1.5px; }}
      .tr {{ top:-5px; right:-5px; border-top-width:1.5px; border-right-width:1.5px; }}
      .bl {{ bottom:-5px; left:-5px; border-bottom-width:1.5px; border-left-width:1.5px; }}
      .br {{ bottom:-5px; right:-5px; border-bottom-width:1.5px; border-right-width:1.5px; }}
      .frame:hover .c {{ border-color:var(--ink); }}
      .card {{
        background:linear-gradient(180deg,var(--raise),var(--card));
        border:1px solid var(--line); padding:16px; display:flex; flex-direction:column; gap:10px;
      }}
      .card h3 {{ font-size:13.5px; font-weight:600; }}
      .card p {{ font-size:12px; color:var(--muted); line-height:1.45; }}
      .wide-copy {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
      svg {{ display:block; width:100%; height:auto; }}
      .flow {{ stroke:var(--line-strong); fill:none; stroke-width:1.3; }}
      .dash {{ stroke-dasharray:3 4; animation:dash 1.2s linear infinite; }}
      @keyframes dash {{ to {{ stroke-dashoffset:-14; }} }}
      .node {{ fill:var(--raise); stroke:var(--line-strong); }}
      .pkt {{ fill:var(--ink); }}
      .tool-pill {{ cursor:pointer; transition:.2s; }}
      .tool-pill:hover rect {{ stroke:var(--ink); }}
      .chart {{ cursor:crosshair; }}
    </style>
    </head>
    <body>
      <div class="sec">
        <div class="head">
          <span class="tag">ARCHITECTURE · {primary_city.upper()}</span>
          <div class="h2">Live <span class="muted">AirSense</span> Pipeline</div>
        </div>
        <div class="bento">
          <div class="frame wide">
            <span class="c tl"></span><span class="c tr"></span><span class="c bl"></span><span class="c br"></span>
            <div class="card">
              <svg viewBox="0 0 640 120" role="img" aria-label="AirSense data pipeline">
                <path id="l1" d="M130,45 C195,45 210,60 262,60" class="flow" />
                <path id="l2" d="M130,75 C195,75 210,60 262,60" class="flow dash" />
                <path id="r1" d="M378,60 C425,60 440,45 490,45" class="flow" />
                <path id="r2" d="M378,60 C425,60 440,75 490,75" class="flow dash" />
                <circle r="2.3" class="pkt"><animateMotion dur="2s" repeatCount="indefinite" path="M130,45 C195,45 210,60 262,60"/></circle>
                <circle r="2.3" class="pkt"><animateMotion dur="2.2s" begin="0.4s" repeatCount="indefinite" path="M378,60 C425,60 440,45 490,45"/></circle>
                <rect x="24" y="25" width="106" height="70" rx="5" class="node" />
                <text x="77" y="52" text-anchor="middle" font-size="9" font-weight="600" fill="var(--ink)" font-family="var(--mono)">{primary_city.upper()}</text>
                <text x="77" y="68" text-anchor="middle" font-size="7.5" fill="var(--muted)" font-family="var(--mono)">{total_hours}h batch</text>
                <rect x="262" y="38" width="116" height="44" rx="6" class="node" />
                <text x="320" y="64" text-anchor="middle" font-size="9" letter-spacing="1.4" font-weight="600" fill="var(--ink)" font-family="var(--mono)">AIRSENSE</text>
                <rect x="490" y="22" width="124" height="76" rx="4" class="node" />
                <text x="502" y="42" font-size="7.5" fill="var(--muted)" font-family="var(--mono)">ALERTS · {severe_count}h</text>
                <rect x="502" y="52" width="84" height="5" rx="1" fill="var(--ink)" opacity=".75"/>
                <rect x="502" y="63" width="98" height="4" rx="1" fill="var(--line-strong)"/>
                <rect x="502" y="72" width="76" height="4" rx="1" fill="var(--line-strong)"/>
              </svg>
              <div class="wide-copy">
                <div>
                  <h3>Batch Ingestion & Cleaning</h3>
                  <p>Hourly AQI ({aqi_source}) and weather ({wx_source}) aligned, outlier-filtered, and tagged for weekday peak traffic.</p>
                </div>
                <div>
                  <h3>Automated Alerts & Reports</h3>
                  <p>{severe_count} of {total_hours} hours flagged at Poor/Severe AQI with one-click Markdown summary export.</p>
                </div>
              </div>
            </div>
          </div>

          <div class="frame">
            <span class="c tl"></span><span class="c tr"></span><span class="c bl"></span><span class="c br"></span>
            <div class="card">
              <svg viewBox="0 0 300 120">
                <g class="tool-pill"><rect x="18" y="18" width="118" height="32" rx="5" class="node"/><text x="77" y="38" text-anchor="middle" font-size="8.5" font-family="var(--mono)" fill="var(--ink)">CPCB / OpenAQ v3</text></g>
                <g class="tool-pill"><rect x="164" y="18" width="118" height="32" rx="5" class="node"/><text x="223" y="38" text-anchor="middle" font-size="8.5" font-family="var(--mono)" fill="var(--ink)">OWM Air Pollution</text></g>
                <g class="tool-pill"><rect x="18" y="68" width="118" height="32" rx="5" class="node"/><text x="77" y="88" text-anchor="middle" font-size="8.5" font-family="var(--mono)" fill="var(--ink)">Open-Meteo AQI</text></g>
                <g class="tool-pill"><rect x="164" y="68" width="118" height="32" rx="5" class="node"/><text x="223" y="88" text-anchor="middle" font-size="8.5" font-family="var(--mono)" fill="var(--ink)">Open-Meteo Weather</text></g>
              </svg>
              <h3>Multi-Source Fallback Hub</h3>
              <p>Active AQI: <strong>{aqi_source}</strong> · Active Weather: <strong>{wx_source}</strong>.</p>
            </div>
          </div>

          <div class="frame">
            <span class="c tl"></span><span class="c tr"></span><span class="c bl"></span><span class="c br"></span>
            <div class="card">
              <svg id="fc-svg" class="chart" viewBox="0 0 300 120"></svg>
              <h3>Interactive Forecast Scrub</h3>
              <p>Hover across the curve to inspect next-hour predicted AQI for {primary_city}.</p>
            </div>
          </div>
        </div>
      </div>
      <script>
        (function() {{
          const vals = {vals_json};
          const svg = document.getElementById("fc-svg");
          const W = 300, H = 112, pad = 14;
          const lo = Math.min(...vals), hi = Math.max(...vals), span = (hi - lo) || 1;
          const pts = vals.map((v, i) => [
            pad + (i / (vals.length - 1)) * (W - pad * 2),
            H - pad - ((v - lo) / span) * (H - pad * 2)
          ]);
          let line = "";
          pts.forEach(([x, y], i) => {{
            if (i === 0) {{ line = `M${{x.toFixed(1)}},${{y.toFixed(1)}}`; return; }}
            const [px, py] = pts[i - 1], cx = (px + x) / 2;
            line += ` C${{cx.toFixed(1)}},${{py.toFixed(1)}} ${{cx.toFixed(1)}},${{y.toFixed(1)}} ${{x.toFixed(1)}},${{y.toFixed(1)}}`;
          }});
          const area = line + ` L${{pts[pts.length-1][0].toFixed(1)}},${{H-pad}} L${{pts[0][0].toFixed(1)}},${{H-pad}} Z`;
          function render(idx) {{
            const i = idx >= 0 ? idx : pts.length - 1;
            const [px, py] = pts[i];
            const tx = Math.min(W - 38, Math.max(38, px));
            const ty = py < 32 ? py + 20 : py - 10;
            svg.innerHTML = `
              <line x1="14" x2="${{W-14}}" y1="${{H*0.33}}" y2="${{H*0.33}}" stroke="var(--line)"/>
              <line x1="14" x2="${{W-14}}" y1="${{H*0.66}}" y2="${{H*0.66}}" stroke="var(--line)"/>
              <path d="${{area}}" fill="var(--ink)" opacity="0.08"/>
              <path d="${{line}}" fill="none" stroke="var(--ink)" stroke-width="1.6"/>
              <line x1="${{px}}" x2="${{px}}" y1="8" y2="${{H-pad}}" stroke="var(--ink)" stroke-opacity="0.3"/>
              <circle cx="${{px}}" cy="${{py}}" r="3.4" fill="var(--raise)" stroke="var(--ink)" stroke-width="1.6"/>
              <g transform="translate(${{tx}},${{ty}})">
                <rect x="-34" y="-11" width="68" height="15" rx="2" fill="var(--inv)"/>
                <text x="0" y="-1" text-anchor="middle" font-size="8" fill="var(--inv-ink)" font-family="var(--mono)">+${{i+1}}h: ${{vals[i]}} AQI</text>
              </g>
            `;
          }}
          render(-1);
          svg.addEventListener("pointermove", (e) => {{
            const r = svg.getBoundingClientRect();
            const mx = ((e.clientX - r.left) / r.width) * W;
            let best = 0;
            for (let j = 1; j < pts.length; j++) if (Math.abs(pts[j][0] - mx) < Math.abs(pts[best][0] - mx)) best = j;
            render(best);
          }});
          svg.addEventListener("pointerleave", () => render(-1));
        }})();
      </script>
    </body>
    </html>
    """
    components.html(html, height=580, scrolling=False)


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
        color_continuous_scale="YlOrRd",
        aspect="auto",
    )
    fig.update_layout(height=360)
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


def main() -> None:
    """Run the AirSense Streamlit dashboard."""
    _inject_ink_orbit_theme()
    st.markdown('<span class="ib-header-tag">AIRSENSE · PUNJAB AQI MONITOR</span>', unsafe_allow_html=True)
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

    # 8. Ink-Orbit Architectural Overview Bento
    st.subheader("Pipeline & Forecast Overview")
    _render_ink_orbit_bento(primary_city, forecast_df, aqi_source, wx_source, alerts_summary)

    # 9. Download Report Button
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
