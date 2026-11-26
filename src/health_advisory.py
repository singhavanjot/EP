"""Module 5: CPCB AQI-to-category mapping and health advisory messages."""

from typing import Dict

from config import AQI_CATEGORIES

_ADVISORIES: Dict[str, str] = {
    "Good": (
        "Air quality is satisfactory and poses little or no risk. "
        "Enjoy normal outdoor activities."
    ),
    "Satisfactory": (
        "Air quality is acceptable; unusually sensitive individuals may experience "
        "minor breathing discomfort."
    ),
    "Moderate": (
        "People with lung or heart disease, children, and older adults should reduce "
        "prolonged or heavy outdoor exertion."
    ),
    "Poor": (
        "Breathing discomfort likely for most people on prolonged exposure. "
        "Avoid strenuous outdoor exercise and consider wearing an N95 mask."
    ),
    "Very Poor": (
        "Respiratory illness risk on prolonged exposure. Keep windows closed, "
        "avoid outdoor exertion, and use air purifiers if available."
    ),
    "Severe": (
        "Health emergency: affects healthy people and seriously impacts those with "
        "existing conditions. Stay indoors and avoid all outdoor physical activity."
    ),
}


def aqi_to_category(aqi_value: float) -> str:
    """Map a numeric AQI value to its official CPCB health category.

    Args:
        aqi_value: Non-negative AQI value.

    Returns:
        CPCB category string: Good, Satisfactory, Moderate, Poor, Very Poor, or Severe.

    Raises:
        ValueError: If aqi_value is negative or NaN.
    """
    if aqi_value is None or aqi_value != aqi_value:  # NaN check
        raise ValueError("AQI value cannot be None or NaN.")
    if aqi_value < 0:
        raise ValueError(f"AQI value cannot be negative: {aqi_value}")

    if aqi_value <= 50:
        return "Good"
    if aqi_value <= 100:
        return "Satisfactory"
    if aqi_value <= 200:
        return "Moderate"
    if aqi_value <= 300:
        return "Poor"
    if aqi_value <= 400:
        return "Very Poor"
    return "Severe"


def get_category_color(category: str) -> str:
    """Return the hex color associated with a CPCB AQI category.

    Args:
        category: CPCB category name.

    Returns:
        Hex color string (defaults to neutral gray if unknown).
    """
    for _, _, cat_name, color_hex in AQI_CATEGORIES:
        if cat_name.lower() == category.strip().lower():
            return color_hex
    return "#808080"


def get_recommended_action(category: str) -> str:
    """Return human-readable health advisory text for a CPCB category.

    Args:
        category: CPCB category name.

    Returns:
        Recommended action string.
    """
    return _ADVISORIES.get(
        category,
        "Monitor local air quality updates and limit outdoor exposure if symptoms occur.",
    )
