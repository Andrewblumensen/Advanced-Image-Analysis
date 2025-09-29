import pandas as pd
import leanvl.metrics.anova_twoway

def test_anova_twoway_significant_main_effects():
    """
    Test Two-Way ANOVA for significant main effects.

    Test Data:
        - A dataset with significant main effects for both factors.

    Test Steps:
        1. Create a dataset with significant differences in the dependent variable across levels of two factors.
        2. Call the anova_twoway function with this dataset.
        3. Assert that the p-values for the main effects are less than 0.05.

    Expected Result:
        - The p-values for the main effects should be less than 0.05.
    """
    data = pd.DataFrame({
        'Score': [100, 98, 70, 85, 80, 77, 20, 30],
        'Teaching_Method': ['Traditional', 'Traditional', 'Traditional', 'Traditional', 
                            'Modern', 'Modern', 'Modern', 'Modern'],
        'Gender': ['Male', 'Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Female']
    })
    result = leanvl.metrics.anova_twoway.anova_twoway(data, dependent_var='Score', factor1='Teaching_Method', factor2='Gender')
    assert result.loc['C(Teaching_Method)', 'PR(>F)'] < 0.05
    assert result.loc['C(Gender)', 'PR(>F)'] < 0.05


def test_anova_twoway_no_significant_interaction():
    """
    Test Two-Way ANOVA for no significant interaction effect.

    Test Data:
        - A dataset with no significant interaction effect between the two factors.

    Test Steps:
        1. Create a dataset with no significant interaction effect.
        2. Call the anova_twoway function with this dataset.
        3. Assert that the p-value for the interaction term is greater than or equal to 0.05.

    Expected Result:
        - The p-value for the interaction term should be greater than or equal to 0.05.
    """
    data = pd.DataFrame({
        'Score': [85, 90, 88, 78, 92, 80, 75, 85],
        'Teaching_Method': ['Traditional', 'Traditional', 'Traditional', 'Traditional', 
                            'Modern', 'Modern', 'Modern', 'Modern'],
        'Gender': ['Male', 'Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Female']
    })
    result = leanvl.metrics.anova_twoway.anova_twoway(data, dependent_var='Score', factor1='Teaching_Method', factor2='Gender')
    assert result.loc['C(Teaching_Method):C(Gender)', 'PR(>F)'] >= 0.05
    


def test_anova_twoway_input_validation():
    """
    Test Two-Way ANOVA for input validation.

    Test Data:
        - Invalid inputs (non-pandas DataFrame or missing columns).

    Test Steps:
        1. Pass invalid inputs (e.g., lists or DataFrame without required columns) to the anova_twoway function.
        2. Assert that the function raises a ValueError.

    Expected Result:
        - The anova_twoway function should raise a ValueError for invalid inputs.
    """
    # Invalid input: not a DataFrame
    data = [85, 90, 88, 78, 92, 80, 75, 85]
    try:
        leanvl.metrics.anova_twoway.anova_twoway(data, dependent_var='Score', factor1='Teaching_Method', factor2='Gender')
    except ValueError as e:
        assert str(e) == "Input data must be a pandas DataFrame."
    else:
        assert False, "Expected ValueError was not raised."

    # Invalid input: missing columns
    data = pd.DataFrame({'Score': [85, 90, 88, 78, 92, 80, 75, 85]})
    try:
        leanvl.metrics.anova_twoway.anova_twoway(data, dependent_var='Score', factor1='Teaching_Method', factor2='Gender')
    except ValueError as e:
        assert str(e) == "Dependent variable and factors must be columns in the DataFrame."
    else:
        assert False, "Expected ValueError was not raised."


def test_anova_twoway_empty_data():
    """
    Test Two-Way ANOVA with an empty DataFrame.

    Test Data:
        - An empty pandas DataFrame.

    Test Steps:
        1. Create an empty DataFrame.
        2. Call the anova_twoway function with this DataFrame.
        3. Assert that the function raises a ValueError.

    Expected Result:
        - The anova_twoway function should raise a ValueError for empty data.
    """
    data = pd.DataFrame(columns=['Score', 'Teaching_Method', 'Gender'])
    try:
        leanvl.metrics.anova_twoway.anova_twoway(data, dependent_var='Score', factor1='Teaching_Method', factor2='Gender')
    except ValueError as e:
        assert str(e) == "Input data must not be empty."
    else:
        assert False, "Expected ValueError was not raised."

