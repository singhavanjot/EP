"""Configuration settings for the AirSense multi-city AQI dashboard."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROCESSED_DATA_DIR / "models"
DB_PATH = DATA_DIR / "airsense.db"

# Ensure directories exist at runtime
for _dir in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# Target cities (Punjab region, extensible to others)
CITIES: Dict[str, Dict[str, Any]] = {
    "Ludhiana": {
        "lat": 30.9010,
        "lon": 75.8573,
        "state": "Punjab",
        "cpcb_station": "Punjab Agricultural University, Ludhiana - PPCB",
        "openaq_location_id": 8118,
        "timezone": "Asia/Kolkata",
    },
    "Chandigarh": {
        "lat": 30.7333,
        "lon": 76.7794,
        "state": "Chandigarh",
        "cpcb_station": "Sector-25, Chandigarh - CPCC",
        "openaq_location_id": 8244,
        "timezone": "Asia/Kolkata",
    },
    "Amritsar": {
        "lat": 31.6340,
        "lon": 74.8723,
        "state": "Punjab",
        "cpcb_station": "Golden Temple, Amritsar - PPCB",
        "openaq_location_id": 8125,
        "timezone": "Asia/Kolkata",
    },
}

# Peak traffic-hour windows (24-hour clock, inclusive start, exclusive end)
# Applied on weekdays (Monday=0 through Friday=4)
PEAK_HOUR_WINDOWS: List[Tuple[int, int]] = [
    (8, 10),   # 8:00 AM - 10:00 AM
    (17, 20),  # 5:00 PM - 8:00 PM
]
WEEKDAYS_ONLY_PEAK: bool = True

# CPCB official AQI health category bands: (min_aqi, max_aqi, category, color_hex)
AQI_CATEGORIES: List[Tuple[float, float, str, str]] = [
    (0, 50, "Good", "#00B050"),
    (51, 100, "Satisfactory", "#92D050"),
    (101, 200, "Moderate", "#FFFF00"),
    (201, 300, "Poor", "#FF9900"),
    (301, 400, "Very Poor", "#FF0000"),
    (401, float("inf"), "Severe", "#990000"),
]

# Severe-AQI alert threshold (CPCB "Poor" starts at 201)
SEVERE_AQI_THRESHOLD: float = 201.0

# API base URLs and environment variable names (never hardcode keys)
API_CONFIG: Dict[str, Dict[str, str]] = {
    "cpcb": {
        "base_url": "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69",
        "env_var": "CPCB_API_KEY",
    },
    "openaq": {
        "base_url": "https://api.openaq.org/v3",
        "env_var": "OPENAQ_API_KEY",
    },
    "openweather": {
        "base_url": "https://history.openweathermap.org/data/2.5/history/city",
        "air_pollution_url": "https://api.openweathermap.org/data/2.5/air_pollution/history",
        "env_var": "OPENWEATHER_API_KEY",
    },
    "openmeteo": {
        "base_url": "https://archive-api.open-meteo.com/v1/archive",
    },
}

# Request retry configuration
REQUEST_TIMEOUT_SECONDS: int = 15
MAX_RETRIES: int = 3
RETRY_BACKOFF_FACTOR: float = 1.5
