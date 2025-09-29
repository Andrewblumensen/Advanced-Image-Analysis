"""
One-Way ANOVA
============================

Performs a one-way Analysis of Variance (ANOVA) test on two pandas Series.

The one-way ANOVA test is used to determine whether there are statistically significant differences between the means of two independent groups.
It compares the variance between groups to the variance within groups to assess whether the group means are significantly different.

The null hypothesis for the one-way ANOVA test is that all group means are equal.

Formula:

    .. math::

        F = \frac{\frac{\text{SS}_{\text{between}}}{k - 1}{\frac{\text{SS}_{\text{within}}}{n - k}}

Where:
    - :math:\text{SS}_{\text{between}} = Sum of squares between groups.
    - :math: k = Number of groups.
    - :math: \text{SS}_{\text{within}} = Sum of squares within groups.
    - :math: n = Total number of observations.


The resulting F-statistic is compared against the F-distribution with (k-1, n-k) degrees of freedom to compute the p-value:


Unit testing:
    See the unit test: :file:`tests/metrics/test_anova_oneway.py`
"""

import pandas as pd
from scipy.stats import f_oneway

def anova_oneway(series1: pd.Series, series2: pd.Series) -> float:
    """
    Performs a one-way ANOVA test on two pandas Series.

    Args:
        series1 (pd.Series): The first group of data.
        series2 (pd.Series): The second group of data.

    Returns:
        float: The p-value from the ANOVA test, indicating whether the means of the two groups are significantly different.

    Example:
        >>> series1 = pd.Series([1, 2, 3, 4, 5])
        >>> series2 = pd.Series([5, 6, 7, 8, 9])
        >>> result = anova(series1, series2)
        >>> print(f"The p-value is: {result:.4f}")
    """
    # Ensure inputs are pandas Series
    if not isinstance(series1, pd.Series) or not isinstance(series2, pd.Series):
        raise ValueError("Both inputs must be pandas Series.")
    
    # Check for empty inputs
    if series1.empty or series2.empty:
        raise ValueError("Inputs must not be empty.")
    
    # Perform one-way ANOVA
    f_stat, p_value = f_oneway(series1, series2)
    
    return p_value


