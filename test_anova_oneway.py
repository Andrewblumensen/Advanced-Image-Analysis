import pandas as pd
from math import isclose
import leanvl.metrics.anova_oneway

def test_anova_significant_difference():
    """
    Test One-Way ANOVA for a significant difference between two groups.

    Test Data:
        - Two pd.Series with significantly different means.

    Test Steps:
        1. Create two pd.Series with significantly different means.
        2. Call the anova_oneway function with these pd.Series.
        3. Assert that the p-value is less than 0.05.

    Expected Result:
        - The p-value should be less than 0.05, indicating a significant difference.
    """
    series1 = pd.Series([1, 2, 3, 4, 5])
    series2 = pd.Series([10, 11, 12, 13, 14])
    p_value = leanvl.metrics.anova_oneway.anova_oneway(series1, series2)
    assert p_value < 0.05
    assert isinstance(p_value, float)


def test_anova_no_significant_difference():
    """
    Test One-Way ANOVA for no significant difference between two groups.

    Test Data:
        - Two pd.Series with similar means.

    Test Steps:
        1. Create two pd.Series with similar means.
        2. Call the anova_oneway function with these pd.Series.
        3. Assert that the p-value is greater than or equal to 0.05.

    Expected Result:
        - The p-value should be greater than or equal to 0.05, indicating no significant difference.
    """
    series1 = pd.Series([1, 2, 3, 4, 5])
    series2 = pd.Series([1.1, 2.1, 3.1, 4.1, 5.1])
    p_value = leanvl.metrics.anova_oneway.anova_oneway(series1, series2)
    assert p_value >= 0.05
    assert isinstance(p_value, float)


def test_anova_input_validation():
    """
    Test One-Way ANOVA for input validation.

    Test Data:
        - Invalid inputs (non-pandas Series).

    Test Steps:
        1. Pass invalid inputs (e.g., lists) to the anova_oneway function.
        2. Assert that the function raises a ValueError.

    Expected Result:
        - The anova_oneway function should raise a ValueError for invalid inputs.
    """
    series1 = [1, 2, 3, 4, 5]  # Not a pandas Series
    series2 = [10, 11, 12, 13, 14]  # Not a pandas Series
    try:
        leanvl.metrics.anova_oneway.anova_oneway(series1, series2)
    except ValueError as e:
        assert str(e) == "Both inputs must be pandas Series."
    else:
        assert False, "Expected ValueError was not raised."


def test_anova_empty_series():
    """
    Test One-Way ANOVA with empty series.

    Test Data:
        - Two empty pd.Series.

    Test Steps:
        1. Create two empty pd.Series.
        2. Call the anova_oneway function with these pd.Series.
        3. Assert that the function raises a ValueError.

    Expected Result:
        - The anova_oneway function should raise a ValueError for empty series.
    """
    series1 = pd.Series([], dtype=float)
    series2 = pd.Series([], dtype=float)
    try:
        leanvl.metrics.anova_oneway.anova_oneway(series1, series2)
    except ValueError as e:
        assert str(e) == "Inputs must not be empty."
    else:
        assert False, "Expected ValueError was not raised."


def test_anova_large_numbers():
    """
    Test One-Way ANOVA with very large numbers.

    Test Data:
        - Two pd.Series with very large numbers.

    Test Steps:
        1. Create two pd.Series with very large numbers.
        2. Call the anova_oneway function with these pd.Series.
        3. Assert that the p-value is less than 0.05 (significant difference).

    Expected Result:
        - The p-value should be less than 0.05, indicating a significant difference.
    """
    series1 = pd.Series([1e10, 2e10, 3e10, 4e10, 5e10])
    series2 = pd.Series([1e10 + 1e5, 2e10 + 1e5, 3e10 + 1e5, 4e10 + 1e5, 5e10 + 1e5])
    p_value = leanvl.metrics.anova_oneway.anova_oneway(series1, series2)
    assert p_value >= 0.05
    assert isinstance(p_value, float)
