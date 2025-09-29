
import pandas as pd
from math import isclose
import leanvl.metrics.pearson

def test_pearson_perfect_positive_correlation():
    """
    Test Pearson correlation for a perfect positive linear relationship.

    Test Data:
        - Two pd.Series with a perfect positive linear relationship.

    Test Steps:
        1. Create two pd.Series with a perfect positive linear relationship.
        2. Call the pearson function with these pd.Series.
        3. Assert that the result is close to 1.0.

    Expected Result:
        - The Pearson correlation should return a value close to 1.0.
    """
    series1 = pd.Series([1, 2, 3, 4, 5])
    series2 = pd.Series([2, 4, 6, 8, 10])
    result = leanvl.metrics.pearson.pearson(series1, series2)
    assert isclose(result, 1.0, rel_tol=1e-9)
    assert isinstance(result, float)
    
def test_pearson_perfect_negative_correlation():
    """
    Test Pearson correlation for a perfect negative linear relationship.

    Test Data:
        - Two pd.Series with a perfect negative linear relationship.

    Test Steps:
        1. Create two pd.Series with a perfect negative linear relationship.
        2. Call the pearson function with these pd.Series.
        3. Assert that the result is close to -1.0.

    Expected Result:
        - The Pearson correlation should return a value close to -1.0.
    """
    series1 = pd.Series([1, 2, 3, 4, 5])
    series2 = pd.Series([10, 8, 6, 4, 2])
    result = leanvl.metrics.pearson.pearson(series1, series2)
    assert isclose(result, -1.0, rel_tol=1e-9)
    assert isinstance(result, float)
    

def test_pearson_small_correlation():
    """
    Test Pearson correlation for no linear relationship.

    Test Data:
        - Two pd.Series with no linear relationship.

    Test Steps:
        1. Create two pd.Series with small linear relationship.
        2. Call the pearson function with these pd.Series.
        3. Assert that the result is close to -0.0081411.

    Expected Result:
        - The Pearson correlation should return a value close to -0.0081411.
    """
    series1 = pd.Series([1, 2, 3, 4, 5])
    series2 = pd.Series([2, 7, 5, 4.9, 3])
    result = leanvl.metrics.pearson.pearson(series1, series2)
    assert isclose(result, -0.0081411, rel_tol=1e-5)
    assert isinstance(result, float)
    
    
def test_pearson_input_validation():
    """
    Test Pearson correlation for input validation.

    Test Data:
        - Invalid inputs (non-pandas Series).

    Test Steps:
        1. Pass invalid inputs (e.g., lists) to the pearson function.
        2. Assert that the function raises a ValueError.

    Expected Result:
        - The Pearson function should raise a ValueError for invalid inputs.
    """
    series1 = [1, 2, 3, 4, 5]  # Not a pandas Series
    series2 = [2, 4, 6, 8, 10]  # Not a pandas Series
    try:
        leanvl.metrics.pearson.pearson(series1, series2)
    except ValueError as e:
        assert str(e) == "Both inputs must be pandas Series."
    else:
        assert False, "Expected ValueError was not raised."


def test_pearson_large_numbers():
    """
    Test Pearson correlation with very large numbers.

    Test Data:
        - Two pd.Series with very large numbers.

    Test Steps:
        1. Create two pd.Series with very large numbers.
        2. Call the pearson function with these pd.Series.
        3. Assert that the result is close to 1.0 (perfect positive correlation).

    Expected Result:
        - The Pearson correlation should return a value close to 1.0.
    """
    series1 = pd.Series([1e10, 2e10, 3e10, 4e10, 5e10])
    series2 = pd.Series([1e10, 2e10, 3e10, 4e10, 5e10])
    result = leanvl.metrics.pearson.pearson(series1, series2)
    assert isclose(result, 1.0, rel_tol=1e-9)
    assert isinstance(result, float)
    
def test_pearson_empty_series():
    """
    Test Pearson correlation with empty series.

    Test Data:
        - Two empty pd.Series.

    Test Steps:
        1. Create two empty pd.Series.
        2. Call the pearson function with these pd.Series.
        3. Assert that the function raises a ValueError.

    Expected Result:
        - The Pearson function should raise a ValueError for empty series.
    """
    series1 = pd.Series([], dtype=float)
    series2 = pd.Series([], dtype=float)
    try:
        leanvl.metrics.pearson.pearson(series1, series2)
    except ValueError as e:
        assert str(e) == "Inputs must not be empty."
    else:
        assert False, "Expected ValueError was not raised."
    
    