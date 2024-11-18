import pandas as pd


def IV(data: pd.Series) -> float:
    r"""Calculate the intradaily variability"""

    c_1h = data.diff(1).pow(2).mean()

    d_1h = data.var()

    return (c_1h / d_1h)  


def IS(data: pd.Series) -> float:
    r"""Calculate the interdaily stability"""

    d_24h = data.groupby([
        data.index.hour,
        data.index.minute,
        data.index.second]
    ).mean().var()

    d_1h = data.var()

    return (d_24h / d_1h)


def RA(data: pd.Series):
        r"""Relative rest/activity amplitude

        Relative amplitude between the mean activity during the 10 most active
        hours of the day and the mean activity during the 5 least active hours
        of the day.

        Parameters
        ----------
        binarize: bool, optional
            If set to True, the data are binarized.
            Default is True.
        threshold: int, optional
            If binarize is set to True, data above this threshold are set to 1
            and to 0 otherwise.
            Default is 4.

        Returns
        -------
        ra: float


        Notes
        -----

        The RA [1]_ variable is calculated as:

        .. math::

            RA = \frac{M10 - L5}{M10 + L5}

        References
        ----------

        .. [1] Van Someren, E.J.W., Lijzenga, C., Mirmiran, M., Swaab, D.F.
               (1997). Long-Term Fitness Training Improves the Circadian
               Rest-Activity Rhythm in Healthy Elderly Males.
               Journal of Biological Rhythms, 12(2), 146–156.
               http://doi.org/10.1177/074873049701200206

        Examples
        --------

            >>> import pyActigraphy
            >>> rawAWD = pyActigraphy.io.read_raw_awd(fpath + 'SUBJECT_01.AWD')
            >>> rawAWD.RA()
            0.XXXX
            >>> rawAWD.RA(binarize=False)
            0.XXXX
        """

        _data = data.copy()

        # n_epochs = int(pd.Timedelta('5H')/self.frequency)

        _, l5 = _lmx(_data, '5H', lowest=True)
        _, m10 = _lmx(_data, '10H', lowest=False)

        return (m10-l5)/(m10+l5)


def M10(data: pd.Series):
        r"""M10

        Mean activity during the 10 most active hours of the day.

        Parameters
        ----------
        binarize: bool, optional
            If set to True, the data are binarized.
            Default is True.
        threshold: int, optional
            If binarize is set to True, data above this threshold are set to 1
            and to 0 otherwise.
            Default is 4.

        Returns
        -------
        m10: float


        Notes
        -----

        The M10 [1]_ variable is calculated as the mean, per acquisition period
        , of the average daily activities during the 10 most active hours.

        .. warning:: The value of this variable depends on the length of the
                     acquisition period.

        References
        ----------

        .. [1] Van Someren, E.J.W., Lijzenga, C., Mirmiran, M., Swaab, D.F.
               (1997). Long-Term Fitness Training Improves the Circadian
               Rest-Activity Rhythm in Healthy Elderly Males.
               Journal of Biological Rhythms, 12(2), 146–156.
               http://doi.org/10.1177/074873049701200206

        Examples
        --------

            >>> import pyActigraphy
            >>> rawAWD = pyActigraphy.io.read_raw_awd(fpath + 'SUBJECT_01.AWD')
            >>> rawAWD.M10()
            0.XXXX
            >>> rawAWD.M10(binarize=False)
            0.XXXX
        """

        _data = data.copy()

        # n_epochs = int(pd.Timedelta('10H')/self.frequency)

        _, m10 = _lmx(_data, '10H', lowest=False)

        return m10


def L5(data: pd.Series):
        r"""L5

        Mean activity during the 5 least active hours of the day.

        Parameters
        ----------
        binarize: bool, optional
            If set to True, the data are binarized.
            Default is True.
        threshold: int, optional
            If binarize is set to True, data above this threshold are set to 1
            and to 0 otherwise.
            Default is 4.

        Returns
        -------
        l5: float


        Notes
        -----

        The L5 [1]_ variable is calculated as the mean, per acquisition period,
        of the average daily activities during the 5 least active hours.

        .. warning:: The value of this variable depends on the length of the
                     acquisition period.

        References
        ----------

        .. [1] Van Someren, E.J.W., Lijzenga, C., Mirmiran, M., Swaab, D.F.
               (1997). Long-Term Fitness Training Improves the Circadian
               Rest-Activity Rhythm in Healthy Elderly Males.
               Journal of Biological Rhythms, 12(2), 146–156.
               http://doi.org/10.1177/074873049701200206

        Examples
        --------

            >>> import pyActigraphy
            >>> rawAWD = pyActigraphy.io.read_raw_awd(fpath + 'SUBJECT_01.AWD')
            >>> rawAWD.L5()
            0.XXXX
            >>> rawAWD.L5(binarize=False)
            0.XXXX
        """

        _data = data.copy()

        # n_epochs = int(pd.Timedelta('5H')/self.frequency)

        _, l5 = _lmx(_data, '5H', lowest=True)

        return l5


def _lmx(data, period, lowest=True):
    """Calculate the start time and mean activity of the period of
    lowest/highest activity"""

    avgdaily = _average_daily_activity(data=data, cyclic=True)

    n_epochs = int(pd.Timedelta(period)/avgdaily.index.freq)

    mean_activity = avgdaily.rolling(period).sum().shift(-n_epochs+1)

    if lowest:
        t_start = mean_activity.idxmin()
    else:
        t_start = mean_activity.idxmax()

    lmx = mean_activity[t_start]/n_epochs
    return t_start, lmx


def _average_daily_activity(data: pd.Series, cyclic=False):
    """Calculate the average daily activity distribution"""

    avgdaily = data.groupby([
        data.index.hour,
        data.index.minute,
        data.index.second
    ]).mean()

    if cyclic:
        avgdaily = pd.concat([avgdaily, avgdaily])
        avgdaily.index = pd.timedelta_range(
            start='0 day',
            end='2 days',
            freq=data.index.freq,
            closed='left'
        )
    else:
        avgdaily.index = pd.timedelta_range(
            start='0 day',
            end='1 day',
            freq=data.index.freq,
            closed='left'
        )

    return avgdaily