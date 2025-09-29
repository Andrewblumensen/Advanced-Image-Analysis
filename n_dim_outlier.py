"""
Outlier Count in N-Dimensions
=============================

Calculates the number of outliers in an n-dimensional dataset using various methods.

This metric quantifies how many data points in the input dataset are considered outliers based on the selected method. It supports robust statistical techniques for multivariate outlier detection.

Methods
-------

1. **Minimum Covariance Determinant (MCD)**:
    - Identifies outliers using robust estimates of location and scatter.
    - Outliers are determined based on Mahalanobis distances exceeding a chi-squared threshold.

2. **K-Nearest Neighbors (KNN)**:
    - Identifies outliers based on the average distance to the k-nearest neighbors.
    - Outliers are determined based on distances exceeding the 95th percentile.

Formulas
--------

1. **Mahalanobis Distance**:
    .. math::

        D^2 = (x - \mu)^T \Sigma^{-1} (x - \mu)

    Where:
        - :math:`x` = Data point.
        - :math:`\mu` = Mean vector of the dataset.
        - :math:`\Sigma` = Covariance matrix of the dataset.

2. **KNN Distance**:
    .. math::

        d_i = \\frac{1}{k} \sum_{j=1}^k d(x_i, x_j)

    Where:
        - :math:`d_i` = Average distance to the k-nearest neighbors.
        - :math:`d(x_i, x_j)` = Distance between point :math:`x_i` and its neighbor :math:`x_j`.

Unit testing:
    See the unit test: :file:`tests/metrics/test_n_outlier_ndim.py`
"""

import pandas as pd
import numpy as np
from sklearn.covariance import MinCovDet
from sklearn.neighbors import NearestNeighbors
from scipy.stats import chi2


def n_dim_outlier(data: pd.DataFrame, method: str, **kwargs) -> int:
    """
    Detects and counts outliers in n-dimensional data using various methods.

    Parameters:
        data (pd.DataFrame): The input dataset (n-dimensional).
        method (str): The outlier detection method. Options include:
                      'mcd', 'knn'.
        **kwargs: Additional parameters for specific methods (e.g., `k` for KNN).

    Returns:
        int: The number of outliers detected.

    Example:
        >>> import pandas as pd
        >>> data = pd.DataFrame({
        >>>     'Feature1': [1, 2, 3, 4, 100],
        >>>     'Feature2': [1, 2, 3, 4, 100]
        >>> })
        >>> n_outlier_ndim(data, method='mcd')
        1
    """
    # Ensure the input data is valid
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("Input data must not be empty.")

    if method == 'mcd':
        # Minimum Covariance Determinant (MCD)
        mcd = MinCovDet().fit(data)
        mahal_dist = mcd.mahalanobis(data)  # Mahalanobis distances
        threshold = chi2.ppf(0.95, data.shape[1])  # 95% confidence level
        outliers = mahal_dist > threshold
        return np.sum(outliers)


    elif method == 'knn':
        # K-Nearest Neighbors (KNN)
        k = kwargs.get('k', 5)  # Default to 5 neighbors if not specified
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(data)
        distances, _ = nn.kneighbors(data)
        avg_distances = distances.mean(axis=1)
        threshold = np.percentile(avg_distances, 95)  # 95th percentile as threshold
        outliers = avg_distances > threshold
        return np.sum(outliers)
    
    elif method == 'isolation_forest':
        # Isolation Forest
        from sklearn.ensemble import IsolationForest

        contamination = kwargs.get('contamination', 0.05)  # Default to 5% contamination
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(data)

        return outlier_labels

    else:
        raise ValueError(f"Unsupported method: {method}. Choose from 'mcd', or 'knn'.")