"""
Two-Way ANOVA
============================

Performs a two-way Analysis of Variance (ANOVA) test on a pandas DataFrame.

The two-way ANOVA test is used to determine whether there are statistically significant differences between the means of a dependent variable across two independent factors, as well as their interaction effect.

The null hypotheses for the two-way ANOVA test are:
    1. The means of the dependent variable are equal across levels of the first factor.
    2. The means of the dependent variable are equal across levels of the second factor.
    3. There is no interaction effect between the two factors.

Formula:

    .. math::

        F = \frac{\frac{\text{SS}_{\text{factor}}}{df_{\text{factor}}}}{\frac{\text{SS}_{\text{residual}}}{df_{\text{residual}}}}

Where:
    - :math:\text{SS}_{\text{factor}} = Sum of squares for the factor or interaction.
    - :math:df_{\text{factor}} = Degrees of freedom for the factor or interaction.
    - :math:\text{SS}_{\text{residual}} = Sum of squares for the residuals (error).
    - :math:df_{\text{residual}} = Degrees of freedom for the residuals.

The resulting F-statistic is compared against the F-distribution to compute the p-value.

Unit testing:
    See the unit test: :file:`tests/metrics/test_anova_two_way.py`
"""

import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

def anova_twoway(data: pd.DataFrame, dependent_var: str, factor1: str, factor2: str) -> pd.DataFrame:
    """
    Performs a two-way ANOVA test on a pandas DataFrame.

    Args:
        data (pd.DataFrame): The dataset containing the dependent variable and factors.
        dependent_var (str): The name of the dependent variable (response variable).
        factor1 (str): The name of the first independent variable (factor).
        factor2 (str): The name of the second independent variable (factor).

    Returns:
        pd.DataFrame: A DataFrame containing the ANOVA table with F-statistics and p-values.

    Example:
        >>> data = pd.DataFrame({
        >>>     'Score': [85, 90, 88, 78, 92, 80, 75, 85],
        >>>     'Teaching_Method': ['Traditional', 'Traditional', 'Traditional', 'Traditional', 
        >>>                         'Modern', 'Modern', 'Modern', 'Modern'],
        >>>     'Gender': ['Male', 'Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Female']
        >>> })
        >>> result = anova_twoway(data, dependent_var='Score', factor1='Teaching_Method', factor2='Gender')
        >>> print(result)
    """
    
    # Ensure inputs are valid
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    if dependent_var not in data.columns or factor1 not in data.columns or factor2 not in data.columns:
        raise ValueError("Dependent variable and factors must be columns in the DataFrame.")
    # Check for empty inputs
    if data.empty:
        raise ValueError("Input data must not be empty.")
    
    # Fit the two-way ANOVA model
    formula = f"{dependent_var} ~ C({factor1}) + C({factor2}) + C({factor1}):C({factor2})"
    model = ols(formula, data=data).fit()
    anova_table = anova_lm(model, type=2)

    return anova_table