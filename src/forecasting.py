"""Module 4: Short-term AQI forecasting (6-24 hours) using Prophet."""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from config import MODELS_DIR

logger = logging.getLogger(__name__)


class _FourierTrendFallbackModel:
    """Lightweight Prophet-compatible fallback when CmdStan cannot execute in the environment."""

    def __init__(self) -> None:
        self._reg = Ridge(alpha=1.0)
        self._t0: Optional[pd.Timestamp] = None
        self._last_ds: Optional[pd.Timestamp] = None
        self._resid_std: float = 15.0
        self.eval_metrics_: Dict[str, Optional[float]] = {}

    @staticmethod
    def _features(ds: pd.Series, t0: pd.Timestamp) -> np.ndarray:
        hours_since = (pd.to_datetime(ds) - t0).dt.total_seconds().to_numpy() / 3600.0
        hour_of_day = pd.to_datetime(ds).dt.hour.to_numpy()
        day_of_week = pd.to_datetime(ds).dt.weekday.to_numpy()
        return np.column_stack(
            [
                hours_since / 24.0,
                np.sin(2 * np.pi * hour_of_day / 24.0),
                np.cos(2 * np.pi * hour_of_day / 24.0),
                np.sin(4 * np.pi * hour_of_day / 24.0),
                np.cos(4 * np.pi * hour_of_day / 24.0),
                np.sin(2 * np.pi * day_of_week / 7.0),
                np.cos(2 * np.pi * day_of_week / 7.0),
            ]
        )

    def fit(self, df: pd.DataFrame) -> "_FourierTrendFallbackModel":
        self._t0 = pd.to_datetime(df["ds"].min())
        self._last_ds = pd.to_datetime(df["ds"].max())
        x = self._features(df["ds"], self._t0)
        y = df["y"].to_numpy(dtype=float)
        self._reg.fit(x, y)
        preds = self._reg.predict(x)
        self._resid_std = max(5.0, float(np.std(y - preds)))
        return self

    def make_future_dataframe(self, periods: int, freq: str = "h", include_history: bool = False) -> pd.DataFrame:
        assert self._last_ds is not None
        future_ds = pd.date_range(start=self._last_ds + pd.Timedelta(hours=1), periods=periods, freq=freq)
        return pd.DataFrame({"ds": future_ds})

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self._t0 is not None
        x = self._features(df["ds"], self._t0)
        yhat = self._reg.predict(x)
        margin = 1.28 * self._resid_std  # ~80% interval matching Prophet default
        return pd.DataFrame(
            {
                "ds": pd.to_datetime(df["ds"]),
                "yhat": yhat,
                "yhat_lower": yhat - margin,
                "yhat_upper": yhat + margin,
            }
        )


def _model_cache_path(city: str) -> Path:
    """Return the pickle cache path for a city's trained Prophet model."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    safe_city = city.lower().replace(" ", "_")
    return MODELS_DIR / f"{safe_city}_prophet.pkl"


def _prepare_prophet_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert cleaned AQI DataFrame into Prophet's (ds, y) format sorted chronologically."""
    prophet_df = df[["timestamp", "aqi"]].dropna().copy()
    prophet_df = prophet_df.rename(columns={"timestamp": "ds", "aqi": "y"})
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
    prophet_df = prophet_df.sort_values("ds").drop_duplicates(subset=["ds"]).reset_index(drop=True)
    return prophet_df


def _fit_prophet_or_fallback(train_df: pd.DataFrame) -> Any:
    """Fit Prophet, falling back to Fourier+Trend Ridge if CmdStan fails at runtime."""
    try:
        model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
        model.fit(train_df)
        return model
    except (RuntimeError, OSError) as exc:
        logger.warning("Prophet/CmdStan execution failed (%s); using Fourier-trend fallback.", exc)
        return _FourierTrendFallbackModel().fit(train_df)


def train_forecast_model(
    df: pd.DataFrame,
    city: str,
    use_cache: bool = True,
    holdout_hours: int = 24,
) -> Prophet:
    """Train a Prophet model on the cleaned AQI time series with time-aware validation.

    Evaluates MAE and RMSE on a chronological held-out tail split (never shuffled),
    then fits the final model on the full series and caches it to disk. Evaluation
    metrics are attached to `model.eval_metrics_`.

    Args:
        df: Cleaned DataFrame containing 'timestamp' and 'aqi'.
        city: City name used for model caching.
        use_cache: Load cached model from data/processed/models/ if available.
        holdout_hours: Number of trailing hours reserved for time-aware validation.

    Returns:
        Trained Prophet (or fallback) model instance with `eval_metrics_` attribute.
    """
    cache_path = _model_cache_path(city)
    if use_cache and cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached_model: Prophet = pickle.load(f)
            logger.info("Loaded cached forecast model for %s from %s.", city, cache_path)
            return cached_model
        except Exception as exc:
            logger.warning("Failed loading cached model for %s (%s); retraining.", city, exc)

    prophet_df = _prepare_prophet_df(df)
    if len(prophet_df) < 12:
        raise ValueError(f"Need at least 12 hourly observations to train Prophet for {city}.")

    eval_metrics: Dict[str, Optional[float]] = {"mae": None, "rmse": None, "holdout_hours": 0}

    # Time-aware chronological split (no shuffling)
    effective_holdout = min(holdout_hours, max(6, len(prophet_df) // 5))
    if len(prophet_df) > effective_holdout + 24:
        train_df = prophet_df.iloc[:-effective_holdout].copy()
        test_df = prophet_df.iloc[-effective_holdout:].copy()

        val_model = _fit_prophet_or_fallback(train_df)
        val_pred = val_model.predict(test_df[["ds"]])
        y_true = test_df["y"].to_numpy()
        y_hat = np.clip(val_pred["yhat"].to_numpy(), 0.0, 500.0)

        eval_metrics = {
            "mae": round(float(mean_absolute_error(y_true, y_hat)), 2),
            "rmse": round(float(root_mean_squared_error(y_true, y_hat)), 2),
            "holdout_hours": int(effective_holdout),
        }
        logger.info(
            "Forecast validation for %s (%dh holdout): MAE=%.2f, RMSE=%.2f",
            city,
            effective_holdout,
            eval_metrics["mae"],
            eval_metrics["rmse"],
        )

    final_model = _fit_prophet_or_fallback(prophet_df)
    final_model.eval_metrics_ = eval_metrics
    final_model.last_observed_ds_ = prophet_df["ds"].max()

    try:
        with open(cache_path, "wb") as f:
            pickle.dump(final_model, f)
    except Exception as exc:
        logger.warning("Could not cache forecast model for %s: %s", city, exc)

    return final_model


def predict_next_hours(model: Prophet, hours: int = 24) -> pd.DataFrame:
    """Forecast AQI for the next N hours (clamped to 1-24 hours) with confidence intervals.

    Args:
        model: Trained Prophet (or fallback) model.
        hours: Forecast horizon in hours (typically 6 to 24).

    Returns:
        DataFrame with columns: ds, yhat, yhat_lower, yhat_upper.
    """
    horizon = max(1, min(int(hours), 24))
    future = model.make_future_dataframe(periods=horizon, freq="h", include_history=False)
    forecast = model.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()

    for col in ("yhat", "yhat_lower", "yhat_upper"):
        forecast[col] = np.clip(np.round(forecast[col], 1), 0.0, 500.0)

    return forecast.reset_index(drop=True)
