import numpy as np
import pandas as pd


def calculate_enmo(acc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Euclidean Norm Minus One (ENMO) metric from accelerometer data.

    Args:
        acc_df (pd.DataFrame): DataFrame containing accelerometer data with columns
            'X', 'Y', and 'Z' for accelerometer readings along the three axes, and 'TIMESTAMP' for time information.

    Returns:
        pd.DataFrame: DataFrame containing columns 'TIMESTAMP' and 'ENMO'. The frequency of the records is the same as for the input data.
    """

    try:
        _acc_vectors = acc_df[['X', 'Y', 'Z']].values
        _enmo_vals = np.linalg.norm(_acc_vectors, axis=1) - 1
        _enmo_vals = np.maximum(_enmo_vals, 0)
    except Exception as e:
        print(f"Error calculating ENMO: {e}")
        _enmo_vals = np.nan

    return _enmo_vals


def calculate_minute_level_enmo(data: pd.DataFrame) -> pd.DataFrame:
    """
    Resample high-frequency ENMO data to minute-level by averaging over each minute.

    Args:
        data (pd.DataFrame): DataFrame with 'TIMESTAMP' and 'ENMO' column containing high-frequency ENMO data.

    Returns:
        pd.DataFrame: DataFrame containing columns 'TIMESTAMP' and 'ENMO'. The records are aggregated at the minute level.
    """

    minute_level_enmo_df = data['ENMO'].resample('min').mean().to_frame(name='ENMO')
    return minute_level_enmo_df
