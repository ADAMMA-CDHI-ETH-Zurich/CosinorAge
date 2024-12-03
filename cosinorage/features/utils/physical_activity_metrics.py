import pandas as pd

cutpoints = {
    "SB": 0.00001,
    "LIPA": 0.01
}



def activity_metrics(data: pd.Series) -> pd.DataFrame:
    r"""Calculate Sedentary Behavior (SB), Light Physical Activity (LIPA), and 
    Moderate-to-Vigorous Physical Activity (MVPA) durations in hours for each day.

    Parameters
    ----------
    data : pd.Series
        A pandas Series with a DatetimeIndex and ENMO (Euclidean Norm Minus One) values.
        The index should be datetime with minute-level resolution.
        The values should be float numbers representing acceleration in g units.

    Returns
    -------
    pd.DataFrame
        DataFrame with daily physical activity metrics:
        - Index: date (datetime.date)
        - Columns:
            - SB: Hours spent in sedentary behavior (ENMO ≤ 0.00001g)
            - LIPA: Hours spent in light physical activity (0.00001g < ENMO ≤ 0.01g)
            - MVPA: Hours spent in moderate-to-vigorous physical activity (ENMO > 0.01g)

    Notes
    -----
    - The function assumes minute-level data when converting to hours
    - ENMO cutpoints are based on established thresholds:
        - SB: ≤ 0.00001g
        - LIPA: > 0.00001g and ≤ 0.01g
        - MVPA: > 0.01g
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> dates = pd.date_range('2023-01-01', periods=1440, freq='min')  # One day
    >>> enmo = pd.Series(np.random.uniform(0, 0.1, 1440), index=dates)
    >>> activity_metrics(enmo)
              SB      LIPA      MVPA
    2023-01-01  8.2  10.5  5.3
    """

    if data.empty:
        return pd.DataFrame(columns=['SB', 'LIPA', 'MVPA'])

    data_ = data.copy()[['ENMO']]

    # Group data by day
    daily_groups = data_.groupby(data_.index.date)

    # Initialize list to store results
    daily_metrics = []

    for date, day_data in daily_groups:
        # Calculate time spent in each activity category
        sb_hours = (day_data['ENMO'] <= cutpoints["SB"]).sum() / 60  # Assuming data is minute-level
        lipa_hours = ((day_data['ENMO'] > cutpoints["SB"]) & (day_data['ENMO'] <= cutpoints["LIPA"])).sum() / 60
        mvpa_hours = (day_data['ENMO'] > cutpoints["LIPA"]).sum() / 60

        daily_metrics.append({'date': date, 'SB': sb_hours, 'LIPA': lipa_hours, 'MVPA': mvpa_hours})

    # Create DataFrame with results
    metrics_df = pd.DataFrame(daily_metrics)
    metrics_df.set_index('date', inplace=True)

    return metrics_df