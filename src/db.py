"""SQLite persistence helpers for AirSense processed city datasets."""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from config import DB_PATH


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection, ensuring the parent directory exists."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(db_path))


def init_db(db_path: Path = DB_PATH) -> None:
    """Initialize the SQLite schema for AQI observations and batch metadata."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS aqi_observations (
                city TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                aqi REAL NOT NULL,
                pm25 REAL,
                pm10 REAL,
                temperature REAL,
                humidity REAL,
                wind_speed REAL,
                is_peak_hour INTEGER,
                traffic_window TEXT,
                source TEXT,
                PRIMARY KEY (city, timestamp)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS batch_metadata (
                city TEXT PRIMARY KEY,
                last_updated TEXT NOT NULL,
                record_count INTEGER NOT NULL
            )
            """
        )
        conn.commit()


def save_city_data(city: str, df: pd.DataFrame, db_path: Path = DB_PATH) -> int:
    """Upsert cleaned city observations into SQLite and update batch metadata.

    Args:
        city: Target city name.
        df: Cleaned DataFrame with timestamp, aqi, weather, and peak columns.
        db_path: Optional path to SQLite database.

    Returns:
        Number of rows written.
    """
    init_db(db_path)
    if df.empty:
        return 0

    records = df.copy()
    records["city"] = city
    records["timestamp"] = pd.to_datetime(records["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    records["is_peak_hour"] = records.get("is_peak_hour", False).astype(int)

    expected_cols = [
        "city",
        "timestamp",
        "aqi",
        "pm25",
        "pm10",
        "temperature",
        "humidity",
        "wind_speed",
        "is_peak_hour",
        "traffic_window",
        "source",
    ]
    for col in expected_cols:
        if col not in records.columns:
            records[col] = None

    rows = records[expected_cols].where(pd.notnull(records[expected_cols]), None).values.tolist()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with get_connection(db_path) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO aqi_observations (
                city, timestamp, aqi, pm25, pm10, temperature,
                humidity, wind_speed, is_peak_hour, traffic_window, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO batch_metadata (city, last_updated, record_count)
            VALUES (?, ?, ?)
            """,
            (city, now_str, len(rows)),
        )
        conn.commit()

    return len(rows)


def load_city_data(city: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Load stored observations for a city from SQLite.

    Args:
        city: City name to query.
        db_path: Optional path to SQLite database.

    Returns:
        DataFrame sorted by timestamp, or empty DataFrame if not found.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM aqi_observations WHERE city = ? ORDER BY timestamp ASC",
            conn,
            params=(city,),
        )
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["is_peak_hour"] = df["is_peak_hour"].astype(bool)
    return df


def get_last_updated(city: str, db_path: Path = DB_PATH) -> Optional[str]:
    """Return the last batch-update timestamp string for a city, if recorded."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cur = conn.execute("SELECT last_updated FROM batch_metadata WHERE city = ?", (city,))
        row = cur.fetchone()
    return row[0] if row else None
