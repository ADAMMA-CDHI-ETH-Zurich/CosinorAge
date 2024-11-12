import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.linear_model import LinearRegression


def read_smartwatch_data():
    pass


def preprocess_smartwatch_data():
    pass


def auto_calibrate(df: pd.DataFrame, sf: float, epoch_size: int = 10) -> pd.DataFrame:
    # Initialize calibration parameters
    scale = np.array([1.0, 1.0, 1.0])
    offset = np.array([0.0, 0.0, 0.0])
    calib_error_start = 0
    calib_error_end = 0

    # Rolling calculations
    gx, gy, gz = df['X'].values, df['Y'].values, df['Z'].values
    en = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)

    # Roll mean and standard deviation
    window_size = int(sf * epoch_size)
    mean_en = roll_mean(en, window_size)
    mean_gx = roll_mean(gx, window_size)
    mean_gy = roll_mean(gy, window_size)
    mean_gz = roll_mean(gz, window_size)

    sd_gx = roll_sd(gx, window_size)
    sd_gy = roll_sd(gy, window_size)
    sd_gz = roll_sd(gz, window_size)

    # Auto-calibration
    calib_error_start = np.mean(np.abs(mean_en - 1))

    # Calibration loop - adjust offset and scale iteratively
    max_iter = 1000
    tol = 1e-5
    for _ in range(max_iter):
        # Adjust based on offsets and scaling
        adjusted = (np.array([mean_gx, mean_gy, mean_gz]) - offset[:, None]) * scale[:, None]
        print(adjusted.shape)
        calib_error_end = np.mean(np.abs(np.sqrt(adjusted[0] ** 2 + adjusted[1] ** 2 + adjusted[2] ** 2) - 1))

        # Check for convergence
        if abs(calib_error_end - calib_error_start) < tol:
            print("Convergence reached")
            break
        print("Improvement:", calib_error_start - calib_error_end)

        # Update offset and scale slightly to reduce calibration error
        offset -= np.mean(adjusted - 1, axis=1) * 0.1  # Step size for offset adjustment
        scale *= (1 + 0.1 * (np.std(adjusted, axis=1) - 1))  # Step size for scale adjustment
        calib_error_start = calib_error_end

    # Apply calibration
    calibrated_df = df.copy()
    calibrated_df[['X', 'Y', 'Z']] = (df[['X', 'Y', 'Z']] - offset) * scale

    print("Calibration complete. Error start:", calib_error_start, "Error end:", calib_error_end)
    print("Offset:", offset, "Scale:", scale)
    return calibrated_df


def roll_mean(df, window_size):
    return np.convolve(df, np.ones(window_size) / window_size, mode='valid')


def roll_sd(df, window_size):
    return pd.Series(df).rolling(window=window_size).std().dropna().values


def remove_noise(df: pd.DataFrame, cutoff: float = 2.5, fs: float = 50, order: int = 2) -> pd.DataFrame:
    def butter_lowpass_filter(data, cutoff, fs, order=2):
        # Design Butterworth filter
        nyquist = 0.5 * fs  # Nyquist frequency
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)

        # Apply filter to data
        return filtfilt(b, a, data)

    df['X_filtered'] = butter_lowpass_filter(df['X'], cutoff, fs)
    df['Y_filtered'] = butter_lowpass_filter(df['Y'], cutoff, fs)
    df['Z_filtered'] = butter_lowpass_filter(df['Z'], cutoff, fs)

    return df


def detect_non_wear():
    pass


def calc_weartime():
    pass
