"""Module 1: Historical AQI and weather data ingestion with API fallbacks."""

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from config import (
    API_CONFIG,
    CITIES,
    MAX_RETRIES,
    RAW_DATA_DIR,
    REQUEST_TIMEOUT_SECONDS,
    RETRY_BACKOFF_FACTOR,
)

load_dotenv()
logger = logging.getLogger(__name__)


def _get_secret(key: str) -> str:
    """Read an API key from environment variables (.env) or Streamlit Cloud secrets."""
    val = os.getenv(key, "").strip()
    if val:
        return val
    try:
        import streamlit as st

        if key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    return ""


def _request_with_retry(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Optional[requests.Response]:
    """Execute an HTTP GET request with exponential backoff retries.

    Args:
        url: Target endpoint URL.
        params: Optional query parameters.
        headers: Optional HTTP headers.

    Returns:
        Response object if status is 200 OK, otherwise None.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code == 200:
                return response
            if response.status_code == 429:
                logger.warning("Rate-limited (429) on %s (attempt %d/%d).", url, attempt, MAX_RETRIES)
            else:
                logger.warning(
                    "HTTP %d from %s (attempt %d/%d).",
                    response.status_code,
                    url,
                    attempt,
                    MAX_RETRIES,
                )
        except requests.RequestException as exc:
            logger.warning("Request error on %s (attempt %d/%d): %s", url, attempt, MAX_RETRIES, exc)

        if attempt < MAX_RETRIES:
            sleep_seconds = RETRY_BACKOFF_FACTOR ** attempt
            time.sleep(sleep_seconds)

    return None


def _save_raw_csv(df: pd.DataFrame, city: str, source: str, start_date: str, end_date: str) -> Path:
    """Persist a raw DataFrame to data/raw/{city}_{source}_{daterange}.csv."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    safe_city = city.lower().replace(" ", "_")
    daterange = f"{start_date}_to_{end_date}".replace(":", "-")
    output_path = RAW_DATA_DIR / f"{safe_city}_{source}_{daterange}.csv"
    df.to_csv(output_path, index=False)
    return output_path


def generate_synthetic_data(
    city: str,
    start_date: str,
    end_date: str,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate realistic hourly AQI and weather data for development and demos.

    Produces diurnal AQI patterns with weekday morning/evening traffic peaks,
    meteorological effects (wind dispersion, humidity), and occasional severe
    episodes.

    Args:
        city: City name (must be in config.CITIES or uses default baseline).
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).
        seed: Optional random seed for reproducibility.

    Returns:
        DataFrame with columns: timestamp, city, aqi, pm25, pm10, temperature,
        humidity, wind_speed, source.
    """
    city_baselines = {
        "Ludhiana": 155.0,
        "Amritsar": 135.0,
        "Chandigarh": 95.0,
    }
    base_aqi = city_baselines.get(city, 120.0)

    if seed is None:
        seed = sum(ord(ch) for ch in city) + 42
    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(start=start_date, end=end_date, freq="h")
    if len(timestamps) == 0:
        timestamps = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=24 * 14, freq="h")

    n = len(timestamps)
    hours = timestamps.hour.to_numpy()
    weekdays = timestamps.weekday.to_numpy() < 5

    # Diurnal temperature (C), humidity (%), wind_speed (m/s)
    temperature = 22.0 + 7.5 * np.sin((hours - 9) * np.pi / 12) + rng.normal(0, 1.5, n)
    humidity = np.clip(62.0 - 16.0 * np.sin((hours - 9) * np.pi / 12) + rng.normal(0, 4.0, n), 20.0, 98.0)
    wind_speed = np.clip(2.8 + 1.4 * np.sin((hours - 10) * np.pi / 12) + rng.normal(0, 0.7, n), 0.3, 12.0)

    # Traffic-window bump (8-10 AM and 5-8 PM on weekdays)
    morning_peak = (hours >= 8) & (hours < 10) & weekdays
    evening_peak = (hours >= 17) & (hours < 20) & weekdays
    traffic_bump = np.where(morning_peak, 32.0, 0.0) + np.where(evening_peak, 45.0, 0.0)

    # Multi-day synoptic episode wave + weather dispersion
    day_index = np.arange(n) / 24.0
    episode_wave = 35.0 * np.sin(day_index * 2 * np.pi / 5.5) + 20.0 * np.cos(day_index * 2 * np.pi / 11.0)
    weather_effect = -8.5 * (wind_speed - 2.8) + 0.45 * (humidity - 60.0)

    # Occasional severe spike episodes
    spikes = np.where(rng.random(n) < 0.03, rng.uniform(70.0, 160.0, n), 0.0)

    aqi = base_aqi + traffic_bump + episode_wave + weather_effect + spikes + rng.normal(0, 14.0, n)
    aqi = np.clip(np.round(aqi, 1), 18.0, 485.0)

    pm25 = np.clip(np.round(aqi * 0.58 + rng.normal(0, 5.0, n), 1), 5.0, 350.0)
    pm10 = np.clip(np.round(aqi * 1.12 + rng.normal(0, 8.0, n), 1), 10.0, 500.0)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "city": city,
            "aqi": aqi,
            "pm25": pm25,
            "pm10": pm10,
            "temperature": np.round(temperature, 1),
            "humidity": np.round(humidity, 1),
            "wind_speed": np.round(wind_speed, 2),
            "source": "synthetic",
        }
    )


def _fetch_cpcb_aqi(city: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Attempt to pull AQI records from the CPCB data.gov.in API."""
    cfg = API_CONFIG["cpcb"]
    api_key = _get_secret(cfg["env_var"])
    if not api_key:
        logger.info("CPCB API key (%s) not configured; skipping CPCB fetch.", cfg["env_var"])
        return None

    city_info = CITIES.get(city, {})
    params = {
        "api-key": api_key,
        "format": "json",
        "limit": 1000,
        "filters[city]": city,
    }
    if city_info.get("cpcb_station"):
        params["filters[station]"] = city_info["cpcb_station"]

    resp = _request_with_retry(cfg["base_url"], params=params)
    if resp is None:
        return None

    try:
        records = resp.json().get("records", [])
        if not records:
            logger.warning("CPCB returned 0 records for %s.", city)
            return None

        rows = []
        for rec in records:
            ts = rec.get("last_update") or rec.get("date")
            pollutant_avg = rec.get("pollutant_avg")
            if ts and pollutant_avg is not None:
                rows.append(
                    {
                        "timestamp": pd.to_datetime(ts, errors="coerce"),
                        "city": city,
                        "aqi": pd.to_numeric(pollutant_avg, errors="coerce"),
                        "source": "cpcb",
                    }
                )
        df = pd.DataFrame(rows).dropna(subset=["timestamp", "aqi"])
        if df.empty:
            return None
        mask = (df["timestamp"] >= pd.to_datetime(start_date)) & (
            df["timestamp"] <= pd.to_datetime(end_date) + pd.Timedelta(days=1)
        )
        filtered = df.loc[mask].copy()
        return filtered if not filtered.empty else df
    except Exception as exc:
        logger.warning("Failed parsing CPCB response for %s: %s", city, exc)
        return None


def _fetch_openaq_aqi(city: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Fallback AQI pull from the OpenAQ v2 API."""
    cfg = API_CONFIG["openaq"]
    api_key = _get_secret(cfg["env_var"])
    if not api_key:
        logger.info("OpenAQ API key (%s) not configured; skipping OpenAQ fetch.", cfg["env_var"])
        return None
    headers = {"X-API-Key": api_key}

    city_info = CITIES.get(city, {})
    params: Dict[str, Any] = {
        "date_from": f"{start_date}T00:00:00Z",
        "date_to": f"{end_date}T23:59:59Z",
        "parameter": "pm25",
        "limit": 1000,
    }
    if city_info.get("openaq_location_id"):
        params["location_id"] = city_info["openaq_location_id"]
    else:
        params["city"] = city

    resp = _request_with_retry(cfg["base_url"], params=params, headers=headers)
    if resp is None:
        return None

    try:
        results = resp.json().get("results", [])
        if not results:
            logger.warning("OpenAQ returned 0 results for %s.", city)
            return None

        rows = []
        for item in results:
            ts = item.get("date", {}).get("utc")
            val = item.get("value")
            if ts is not None and val is not None:
                # Approximate Indian AQI from PM2.5 when raw AQI is unavailable
                pm25 = float(val)
                approx_aqi = pm25 * 1.7
                rows.append(
                    {
                        "timestamp": pd.to_datetime(ts, utc=True).tz_convert("Asia/Kolkata").tz_localize(None),
                        "city": city,
                        "aqi": round(approx_aqi, 1),
                        "pm25": round(pm25, 1),
                        "source": "openaq",
                    }
                )
        df = pd.DataFrame(rows).dropna(subset=["timestamp", "aqi"])
        return df if not df.empty else None
    except Exception as exc:
        logger.warning("Failed parsing OpenAQ response for %s: %s", city, exc)
        return None


def fetch_aqi_data(city: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical AQI data for a city with CPCB -> OpenAQ -> synthetic fallback.

    Args:
        city: Target city name.
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).

    Returns:
        DataFrame containing timestamp, city, aqi, and source columns.
    """
    df = _fetch_cpcb_aqi(city, start_date, end_date)
    source = "cpcb"

    if df is None:
        df = _fetch_openaq_aqi(city, start_date, end_date)
        source = "openaq"

    if df is None:
        logger.info("Using synthetic AQI fallback for %s (%s to %s).", city, start_date, end_date)
        synth = generate_synthetic_data(city, start_date, end_date)
        df = synth[["timestamp", "city", "aqi", "pm25", "pm10", "source"]].copy()
        source = "synthetic"

    _save_raw_csv(df, city, f"aqi_{source}", start_date, end_date)
    return df


def fetch_weather_data(city: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical weather data from OpenWeatherMap with synthetic fallback.

    Args:
        city: Target city name.
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).

    Returns:
        DataFrame with timestamp, city, temperature, humidity, wind_speed, source.
    """
    cfg = API_CONFIG["openweather"]
    api_key = _get_secret(cfg["env_var"])
    city_info = CITIES.get(city)

    if api_key and city_info:
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp()) + 86399
        params = {
            "lat": city_info["lat"],
            "lon": city_info["lon"],
            "type": "hour",
            "start": start_ts,
            "end": end_ts,
            "units": "metric",
            "appid": api_key,
        }
        resp = _request_with_retry(cfg["base_url"], params=params)
        if resp is not None:
            try:
                items = resp.json().get("list", [])
                rows = []
                for item in items:
                    dt = pd.to_datetime(item.get("dt"), unit="s", utc=True)
                    local_dt = dt.tz_convert("Asia/Kolkata").tz_localize(None)
                    main = item.get("main", {})
                    wind = item.get("wind", {})
                    rows.append(
                        {
                            "timestamp": local_dt,
                            "city": city,
                            "temperature": main.get("temp"),
                            "humidity": main.get("humidity"),
                            "wind_speed": wind.get("speed"),
                            "source": "openweathermap",
                        }
                    )
                df = pd.DataFrame(rows).dropna(subset=["timestamp"])
                if not df.empty:
                    _save_raw_csv(df, city, "weather_openweathermap", start_date, end_date)
                    return df
            except Exception as exc:
                logger.warning("Failed parsing OpenWeatherMap response for %s: %s", city, exc)
    else:
        logger.info("OpenWeatherMap key not configured; using synthetic weather fallback for %s.", city)

    synth = generate_synthetic_data(city, start_date, end_date)
    weather_df = synth[["timestamp", "city", "temperature", "humidity", "wind_speed", "source"]].copy()
    _save_raw_csv(weather_df, city, "weather_synthetic", start_date, end_date)
    return weather_df
