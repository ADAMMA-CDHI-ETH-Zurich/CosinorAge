import pandas as pd
import numpy as np


def min_max_scaling_exclude_outliers(data, upper_quantile=0.999):
    """
    Scales the data using min-max scaling, excluding outliers based on quantiles.
    
    Parameters:
        data (pd.Series or np.ndarray): Input data to be scaled.
        lower_quantile (float): Lower quantile threshold for excluding outliers.
        upper_quantile (float): Upper quantile threshold for excluding outliers.
    
    Returns:
        np.ndarray: Scaled data with min-max scaling applied.
    """
    # Convert to pandas Series if input is numpy array
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    # Calculate the lower and upper bounds based on quantiles
    upper_bound = data.quantile(upper_quantile)
    
    # Filter data to exclude outliers
    filtered_data = data[data <= upper_bound]
    
    # Calculate min and max of the filtered data
    min_val = 0
    max_val = filtered_data.max()
    
    # Apply min-max scaling to [0,100] - outliers may overshoot
    scaled_data = 100 * (data - min_val) / (max_val - min_val)
    
    return scaled_data