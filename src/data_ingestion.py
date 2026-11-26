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
            if response.status_code in (401, 403, 404, 410):
                logger.warning("Permanent HTTP %d from %s; skipping retries.", response.status_code, url)
                return None
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
    """Fallback AQI pull from the OpenAQ v3 API."""
    cfg = API_CONFIG["openaq"]
    api_key = _get_secret(cfg["env_var"])
    if not api_key:
        logger.info("OpenAQ API key (%s) not configured; skipping OpenAQ fetch.", cfg["env_var"])
        return None
    headers = {"X-API-Key": api_key}

    city_info = CITIES.get(city, {})
    location_id = city_info.get("openaq_location_id")
    if not location_id:
        return None

    # OpenAQ v3: fetch sensors for the location, then pull PM2.5 measurements
    loc_url = f"{cfg['base_url']}/locations/{location_id}/sensors"
    loc_resp = _request_with_retry(loc_url, headers=headers)
    if loc_resp is None:
        return None

    try:
        sensors = loc_resp.json().get("results", [])
        pm25_sensor_id = None
        for s in sensors:
            param = (s.get("parameter") or {}).get("name", "").lower()
            if param == "pm25":
                pm25_sensor_id = s.get("id")
                break
        if not pm25_sensor_id:
            return None

        meas_url = f"{cfg['base_url']}/sensors/{pm25_sensor_id}/measurements"
        params = {
            "datetime_from": f"{start_date}T00:00:00Z",
            "datetime_to": f"{end_date}T23:59:59Z",
            "limit": 1000,
        }
        resp = _request_with_retry(meas_url, params=params, headers=headers)
        if resp is None:
            return None

        results = resp.json().get("results", [])
        if not results:
            logger.warning("OpenAQ v3 returned 0 results for %s.", city)
            return None

        rows = []
        for item in results:
            ts = (item.get("period") or {}).get("datetimeTo", {}).get("utc")
            val = item.get("value")
            if ts is not None and val is not None:
                pm25 = float(val)
                rows.append(
                    {
                        "timestamp": pd.to_datetime(ts, utc=True).tz_convert("Asia/Kolkata").tz_localize(None),
                        "city": city,
                        "aqi": round(pm25 * 1.7, 1),
                        "pm25": round(pm25, 1),
                        "source": "openaq_v3",
                    }
                )
        df = pd.DataFrame(rows).dropna(subset=["timestamp", "aqi"])
        return df if not df.empty else None
    except Exception as exc:
        logger.warning("Failed parsing OpenAQ v3 response for %s: %s", city, exc)
        return None


def _fetch_openweather_aqi(city: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Free-tier OpenWeatherMap Air Pollution History fallback."""
    cfg = API_CONFIG["openweather"]
    api_key = _get_secret(cfg["env_var"])
    city_info = CITIES.get(city)
    if not api_key or not city_info:
        return None

    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp()) + 86399
    params = {
        "lat": city_info["lat"],
        "lon": city_info["lon"],
        "start": start_ts,
        "end": end_ts,
        "appid": api_key,
    }
    resp = _request_with_retry(cfg["air_pollution_url"], params=params)
    if resp is None:
        return None

    try:
        items = resp.json().get("list", [])
        rows = []
        for item in items:
            dt = pd.to_datetime(item.get("dt"), unit="s", utc=True).tz_convert("Asia/Kolkata").tz_localize(None)
            comps = item.get("components", {})
            pm25 = comps.get("pm2_5")
            pm10 = comps.get("pm10")
            if pm25 is not None:
                # Approximate Indian CPCB AQI from PM2.5 when station AQI isn't provided
                approx_aqi = round(float(pm25) * 1.7, 1)
                rows.append(
                    {
                        "timestamp": dt,
                        "city": city,
                        "aqi": approx_aqi,
                        "pm25": round(float(pm25), 1),
                        "pm10": round(float(pm10), 1) if pm10 is not None else None,
                        "source": "openweathermap_air",
                    }
                )
        df = pd.DataFrame(rows).dropna(subset=["timestamp", "aqi"])
        return df if not df.empty else None
    except Exception as exc:
        logger.warning("Failed parsing OpenWeatherMap Air Pollution response for %s: %s", city, exc)
        return None


def fetch_aqi_data(city: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical AQI data with CPCB -> OpenAQ v3 -> OWM Air -> synthetic fallback.

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
        df = _fetch_openweather_aqi(city, start_date, end_date)
        source = "openweathermap_air"

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
        logger.info("OpenWeatherMap key not configured; trying Open-Meteo archive fallback for %s.", city)

    # Free Open-Meteo historical weather fallback (works without API key or on OWM free tier)
    if city_info:
        om_params = {
            "latitude": city_info["lat"],
            "longitude": city_info["lon"],
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            "timezone": "Asia/Kolkata",
        }
        om_resp = _request_with_retry(API_CONFIG["openmeteo"]["base_url"], params=om_params)
        if om_resp is not None:
            try:
                hourly = om_resp.json().get("hourly", {})
                times = hourly.get("time", [])
                if times:
                    om_df = pd.DataFrame(
                        {
                            "timestamp": pd.to_datetime(times),
                            "city": city,
                            "temperature": hourly.get("temperature_2m"),
                            "humidity": hourly.get("relative_humidity_2m"),
                            "wind_speed": hourly.get("wind_speed_10m"),
                            "source": "openmeteo",
                        }
                    ).dropna(subset=["timestamp"])
                    if not om_df.empty:
                        _save_raw_csv(om_df, city, "weather_openmeteo", start_date, end_date)
                        return om_df
            except Exception as exc:
                logger.warning("Failed parsing Open-Meteo response for %s: %s", city, exc)

    synth = generate_synthetic_data(city, start_date, end_date)
    weather_df = synth[["timestamp", "city", "temperature", "humidity", "wind_speed", "source"]].copy()
    _save_raw_csv(weather_df, city, "weather_synthetic", start_date, end_date)
    return weather_df
