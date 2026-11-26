# AirSense — Multi-City Air Quality & Traffic-Window Dashboard

AirSense is a Python-based batch-processing Air Quality Index (AQI) analytics and forecasting platform covering Ludhiana, Chandigarh, and Amritsar (and extensible to other cities). It ingests historical AQI and weather data, cleans and merges hourly series, evaluates peak-traffic-hour correlations while controlling for meteorology, forecasts 6–24 hour AQI with Prophet, and presents health advisories, severe-AQI alerts, and downloadable reports in an interactive Streamlit dashboard.

## Architecture

```
Data Sources (CPCB / OpenAQ / OpenWeatherMap / Synthetic Fallback)
        │
        ▼
Ingestion & Cleaning (data_ingestion.py → data_cleaning.py → db.py)
        │
        ├──► Correlation Analysis (correlation_analysis.py)
        │
        └──► Short-Term Forecasting (forecasting.py, Prophet)
                    │
                    ▼
        Health Advisory & Alerts (health_advisory.py + alerts.py)
                    │
                    ▼
        Interactive Dashboard & Report Export (app.py + report_export.py)
```

## Setup

1. Create and activate a virtual environment (Python 3.10+):
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # macOS/Linux:
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables (optional — leave blank to run with synthetic demo data):
   ```bash
   cp .env.example .env
   # Add CPCB_API_KEY, OPENAQ_API_KEY, and OPENWEATHER_API_KEY to .env
   ```

## Running the Dashboard

```bash
streamlit run app.py
```

No API keys are required for a first run: AirSense automatically falls back to realistic synthetic AQI and weather data when keys are absent or rate limits are reached.

## Running Tests

```bash
pytest
```

## Known Limitations

- **Traffic proxy:** Peak traffic is modeled using weekday time-of-day windows (`8–10 AM` and `5–8 PM`) rather than live GPS or road-sensor traffic counts.
- **Forecast horizon:** Short-term Prophet predictions are capped at 24 hours ahead because synoptic weather shifts beyond 1 day require numerical weather prediction inputs.
- **Batch refresh:** Data is pulled and processed periodically in batch rather than via a real-time streaming pipeline.
