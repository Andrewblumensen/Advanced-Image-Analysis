"""
Spearman correlation Coefficient Calculation
============================

Calculates the Spearman rank-order correlation coefficient between two pandas Series.

This function evaluates the strength and direction of a monotonic relationship between two variables by ranking the data and computing the correlation coefficient based on the ranked values.

Formula:

    .. math::

        r_s = 1 - \\frac{6 \sum d_i^2}{n(n^2 - 1)}

Where:
    - :math:d_i = The difference between the ranks of corresponding values in the two datasets.
    - :math:n = The number of observations.


The resulting coefficient (r_s) ranges from -1 to 1:
    r_s = 1: Perfect positive monotonic relationship.
    r_s = -1: Perfect negative monotonic relationship.
    r_s = 0: No monotonic relationship.


Unit testing:
    See the unit test: :file:`tests/metrics/test_spearman.py`
"""


import pandas as pd
from scipy.stats import spearmanr

def spearman(series1: pd.Series, series2: pd.Series) -> float:
    """
    Calculates the Spearman correlation coefficient between two pandas Series.

    Args:
        series1 (pd.Series): The first dataset.
        series2 (pd.Series): The second dataset.

    Returns:
        float: The Spearman correlation coefficient (r_s), ranging from -1 to 1.

    Example:
        >>> series1 = pd.Series([1, 2, 3, 4, 5])
        >>> series2 = pd.Series([5, 6, 7, 8, 9])
        >>> result = spearman(series1, series2)
        >>> print(f"The Spearman correlation is: {result:.4f}")
    """
    # Ensure inputs are pandas Series
    if not isinstance(series1, pd.Series) or not isinstance(series2, pd.Series):
        raise ValueError("Both inputs must be pandas Series.")
    
    # Check for empty inputs
    if series1.empty or series2.empty:
        raise ValueError("Inputs must not be empty.")
    
    # Perform Spearman correlation
    r_s, p_value = spearmanr(series1, series2)
    
    return r_s