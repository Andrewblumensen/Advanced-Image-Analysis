import pandas as pd
from math import isclose
import leanvl.metrics.spearman

def test_spearman_perfect_positive_correlation():
    """
    Test Spearman correlation for a perfect positive monotonic relationship.

    Test Data:
        - Two pd.Series with a perfect positive monotonic relationship.

    Test Steps:
        1. Create two pd.Series with a perfect positive monotonic relationship.
        2. Call the spearman function with these pd.Series.
        3. Assert that the result is close to 1.0.

    Expected Result:
        - The Spearman correlation should return a value close to 1.0.
    """
    series1 = pd.Series([1, 2, 3, 4, 5])
    series2 = pd.Series([10, 20, 30, 40, 50])
    result = leanvl.metrics.spearman.spearman(series1, series2)
    assert isclose(result, 1.0, rel_tol=1e-9)
    assert isinstance(result, float)


def test_spearman_perfect_negative_correlation():
    """
    Test Spearman correlation for a perfect negative monotonic relationship.

    Test Data:
        - Two pd.Series with a perfect negative monotonic relationship.

    Test Steps:
        1. Create two pd.Series with a perfect negative monotonic relationship.
        2. Call the spearman function with these pd.Series.
        3. Assert that the result is close to -1.0.

    Expected Result:
        - The Spearman correlation should return a value close to -1.0.
    """
    series1 = pd.Series([1, 2, 3, 4, 5])
    series2 = pd.Series([50, 40, 30, 20, 10])
    result = leanvl.metrics.spearman.spearman(series1, series2)
    assert isclose(result, -1.0, rel_tol=1e-9)
    assert isinstance(result, float)


def test_spearman_small_correlation():
    """
    Test Spearman correlation for small monotonic relationship.

    Test Data:
        - Two pd.Series with small monotonic relationship.

    Test Steps:
        1. Create two pd.Series with small monotonic relationship.
        2. Call the spearman function with these pd.Series.
        3. Assert that the result is close to 0.05129.

    Expected Result:
        - The Spearman correlation should return a value close to 0.05129.
    """
    series1 = pd.Series([1, 2, 3, 4, 5])
    series2 = pd.Series([10, 88, 4.7, 10, 20])
    result = leanvl.metrics.spearman.spearman(series1, series2)
    assert isclose(result, 0.05129, rel_tol=1e-3)
    assert isinstance(result, float)


def test_spearman_input_validation():
    """
    Test Spearman correlation for input validation.

    Test Data:
        - Invalid inputs (non-pandas Series).

    Test Steps:
        1. Pass invalid inputs (e.g., lists) to the spearman function.
        2. Assert that the function raises a ValueError.

    Expected Result:
        - The Spearman function should raise a ValueError for invalid inputs.
    """
    series1 = [1, 2, 3, 4, 5]  # Not a pandas Series
    series2 = [10, 20, 30, 40, 50]  # Not a pandas Series
    try:
        leanvl.metrics.spearman.spearman(series1, series2)
    except ValueError as e:
        assert str(e) == "Both inputs must be pandas Series."
    else:
        assert False, "Expected ValueError was not raised."


def test_spearman_empty_series():
    """
    Test Spearman correlation with empty series.

    Test Data:
        - Two empty pd.Series.

    Test Steps:
        1. Create two empty pd.Series.
        2. Call the spearman function with these pd.Series.
        3. Assert that the function raises a ValueError.

    Expected Result:
        - The Spearman function should raise a ValueError for empty series.
    """
    series1 = pd.Series([], dtype=float)
    series2 = pd.Series([], dtype=float)
    try:
        leanvl.metrics.spearman.spearman(series1, series2)
    except ValueError as e:
        assert str(e) == "Inputs must not be empty."
    else:
        assert False, "Expected ValueError was not raised."


def test_spearman_large_numbers():
    """
    Test Spearman correlation with very large numbers.

    Test Data:
        - Two pd.Series with very large numbers.

    Test Steps:
        1. Create two pd.Series with very large numbers.
        2. Call the spearman function with these pd.Series.
        3. Assert that the result is close to 1.0 (perfect positive monotonic relationship).

    Expected Result:
        - The Spearman correlation should return a value close to 1.0.
    """
    series1 = pd.Series([1e10, 2e10, 3e10, 4e10, 5e10])
    series2 = pd.Series([1e10, 2e10, 3e10, 4e10, 5e10])
    result = leanvl.metrics.spearman.spearman(series1, series2)
    assert isclose(result, 1.0, rel_tol=1e-9)
    assert isinstance(result, float)