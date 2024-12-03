import pandas as pd
import numpy as np


def min_max_scaling_exclude_outliers(data, upper_quantile=0.999):
    """
    Scales the data using min-max scaling, excluding outliers based on quantiles.
    
    Parameters:
        data (pd.Series or np.ndarray): Input data to be scaled.
        upper_quantile (float): Upper quantile threshold for excluding outliers.
    
    Returns:
        pd.Series: Scaled data with min-max scaling applied.
        
    Raises:
        ValueError: If input data is empty.
    """
    # Convert to pandas Series if input is numpy array
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
        
    # Check for empty input
    if len(data) == 0:
        raise ValueError("Input data cannot be empty")
        
    # Handle single value or constant values
    if len(data.unique()) == 1:
        return pd.Series(np.zeros(len(data)))

    # Calculate the upper bound based on quantiles
    upper_bound = data.quantile(upper_quantile)
    
    # Filter data to exclude outliers
    filtered_data = data[data <= upper_bound]
    
    # Calculate min and max of the filtered data
    min_val = filtered_data.min()
    max_val = filtered_data.max()
    
    # Handle zero division case
    if max_val == min_val:
        return pd.Series(np.zeros(len(data)))
    
    # Apply min-max scaling to [0,100] - outliers may overshoot
    scaled_data = 100 * (data - min_val) / (max_val - min_val)
    
    return scaled_data