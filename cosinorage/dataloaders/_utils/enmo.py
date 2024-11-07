import numpy as np
import pandas as pd


def calculate_enmo(acc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Euclidean Norm Minus One (ENMO) metric from raw
    accelerometer data.

    The ENMO metric is computed as the Euclidean norm of the accelerometer
    vector
    (X, Y, Z) minus 1, with negative values set to zero. This metric is commonly
    used in physical activity research to quantify movement intensity.

    Args:
        data (pd.DataFrame): DataFrame containing accelerometer data with
            columns 'X', 'Y', and 'Z' representing the raw accelerometer
            readings
            along the three axes.

    Returns:
        pd.DataFrame: The input DataFrame with an additional 'ENMO' column
        containing the ENMO values calculated for each row.
    """

    try:
        _acc_vectors = acc_df[['X', 'Y', 'Z']].values
        _enmo_vals = np.linalg.norm(_acc_vectors, axis=1) - 1
        acc_df['ENMO'] = np.maximum(_enmo_vals, 0)
    except Exception as e:
        print(f"Error calculating ENMO: {e}")
        acc_df['ENMO'] = np.nan

    return acc_df[['TIMESTAMP', 'ENMO']]


def calculate_minute_level_enmo(data: pd.DataFrame) -> pd.DataFrame:
    """
    Resample ENMO data to minute-level by averaging over 1-minute intervals.

    This function resamples high-frequency ENMO data to a minute-level
    resolution
    by taking the mean of the ENMO values within each 1-minute period.

    Args:
        data (pd.DataFrame): DataFrame containing timestamped ENMO data with
            'TIMESTAMP' and 'ENMO' columns, where 'TIMESTAMP' is in a
            datetime format.

    Returns:
        pd.DataFrame: DataFrame with minute-level averaged ENMO values,
        containing
        'TIMESTAMP' and 'ENMO' columns where each row represents a 1-minute
        interval.
    """

    data.set_index('TIMESTAMP', inplace=True)
    minute_enmo = data['ENMO'].resample('min').mean().reset_index()
    return minute_enmo
