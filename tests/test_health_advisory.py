"""Tests for Module 5: CPCB AQI-to-category mapping and health advisories."""

import pytest

from src.health_advisory import aqi_to_category, get_category_color, get_recommended_action


@pytest.mark.parametrize(
    "aqi,expected_category",
    [
        (0, "Good"),
        (50, "Good"),
        (51, "Satisfactory"),
        (100, "Satisfactory"),
        (101, "Moderate"),
        (200, "Moderate"),
        (201, "Poor"),
        (300, "Poor"),
        (301, "Very Poor"),
        (400, "Very Poor"),
        (401, "Severe"),
        (500, "Severe"),
    ],
)
def test_aqi_to_category_cpcb_boundaries(aqi, expected_category):
    """Verify exact CPCB boundary thresholds map to the expected category."""
    assert aqi_to_category(aqi) == expected_category


def test_aqi_to_category_rejects_negative_and_nan():
    """Negative or NaN AQI values should raise ValueError."""
    with pytest.raises(ValueError):
        aqi_to_category(-1)
    with pytest.raises(ValueError):
        aqi_to_category(float("nan"))


def test_advisory_and_color_lookup():
    """Every CPCB category has a non-empty advisory and hex color."""
    for cat in ("Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"):
        assert get_category_color(cat).startswith("#")
        assert len(get_recommended_action(cat)) > 10
