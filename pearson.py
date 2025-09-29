"""
Pearson correlation Coefficient Calculation
============================

Calculates the Pearson product-moment correlation coefficient between two numerical pandas Series.

The Pearson correlation coefficient measures the strength and direction of the linear relationship between two continuous variables.

Formula:

    .. math::

        r_p = \frac{\text{Cov}(X, Y)}{\sigma_X \cdot \sigma_Y}

Where:
    - :math:\text{Cov}(X, Y) = Covariance between X and Y.
    - :math:\sigma_X = Standard deviation of X.
    - :math:\sigma_Y = Standard deviation of Y.


The resulting coefficient (r_p) ranges from -1 to 1:
    r_p = 1: Perfect positive linear relationship.
    r_p = -1: Perfect negative linear relationship.
    r_p = 0: No linear relationship.


Unit testing:
    See the unit test: :file:`tests/metrics/test_pearson.py`
"""

import pandas as pd
from scipy.stats import pearsonr

def pearson(series1: pd.Series, series2: pd.Series) -> float:
    """
    Calculates the Pearson correlation between two pandas Series.

    Args:
        series1 (pd.Series): The first dataset.
        series2 (pd.Series): The second dataset.

    Returns:
        float: Pearson correlation Coefficient (r_p), ranging from -1 to 1.

    Example:
        >>> series1 = pd.Series([1, 2, 3, 4, 5])
        >>> series2 = pd.Series([5, 6, 7, 8, 9])
        >>> result = pearson(series1, series2)
        >>> print(f"The Pearson correlation is: {result:.4f}")
    """
    # Ensure inputs are pandas Series
    if not isinstance(series1, pd.Series) or not isinstance(series2, pd.Series):
        raise ValueError("Both inputs must be pandas Series.")
    
    # Check for empty inputs
    if series1.empty or series2.empty:
        raise ValueError("Inputs must not be empty.")
    
    # Perform Pearson correlation
    r_p, p_value = pearsonr(series1, series2)
    
    return r_p