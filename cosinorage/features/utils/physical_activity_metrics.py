import pandas as pd

cutpoints = {
    "SB": 0.00001,
    "LIPA": 0.01
}



def activity_metrics(data: pd.Series) -> pd.DataFrame:
    r"""Calculate SB, LIPA, and MVPA (in hours) for each day.

    Parameters
    ----------
    data : pd.Series
        A pandas Series with a DatetimeIndex and ENMO values.

    Returns
    -------
    pd.DataFrame
        DataFrame with date index and columns for SB, LIPA, and MVPA (in hours).
        
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