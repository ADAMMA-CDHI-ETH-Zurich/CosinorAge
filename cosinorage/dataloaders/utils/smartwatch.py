import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.linear_model import LinearRegression


def read_smartwatch_data():
    pass


def preprocess_smartwatch_data():
    pass


def auto_calibrate(df: pd.DataFrame, sf: float, epoch_size: int = 10):

    """
    Perform autocalibration on accelerometer data, adjusting offset and scale to reduce calibration error.
    Parameters:
        data (np.ndarray): Accelerometer data (N x 3), where each column is an axis (x, y, z)
        sf (int): Sampling frequency of the data
        params_rawdata (dict): Dictionary with calibration parameters
        calib_epoch_size (int): Epoch size for calibration in seconds
        block_resolution (int): Resolution of data blocks in seconds
        verbose (bool): If True, prints calibration summary
    Returns:
        dict: Calibration results including scale, offset, and calibration error metrics
    """
    # Initialize calibration parameters for scale and offset
    scale = np.array([1.0, 1.0, 1.0])  # Start with no scaling (1.0) for x, y, z
    offset = np.array([0.0, 0.0, 0.0])  # Start with no offset for x, y, z
    calib_error_start, calib_error_end = None, None  # Track calibration error before and after adjustment

    # Step 1: Calculate features for calibration
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

    # Step 2: Filter features for nonmovement periods based on low standard deviation
    sd_criter = 0.013  # Example threshold for standard deviation

    nonmovement_idx = np.where((sd_gx < sd_criter) & (sd_gy < sd_criter) & (sd_gz < sd_criter) & (mean_gx < 2) & (mean_gy < 2) & (mean_gz < 2))[0]

    mean_en = mean_en[nonmovement_idx]
    mean_gx = mean_gx[nonmovement_idx]
    mean_gy = mean_gy[nonmovement_idx]
    mean_gz = mean_gz[nonmovement_idx]
    sd_gx = sd_gx[nonmovement_idx]
    sd_gy = sd_gy[nonmovement_idx]
    sd_gz = sd_gz[nonmovement_idx]

    # Step 3: Ensure enough data points for calibration
    if len(mean_en) > 10:
        # Calculate initial calibration error based on distance from expected 1g magnitude
        npoints = len(mean_en)
        calib_error_start = np.mean(
            np.abs(np.sqrt(mean_gx ** 2 + mean_gy ** 2 + mean_gz ** 2) - 1))
        # Step 4: Iterative adjustment to minimize calibration error

        input_data = np.vstack((mean_gx, mean_gy, mean_gz)).T
        weights = np.ones(len(input_data))  # Initialize weights for each data point
        res = float('inf')
        max_iter = 1000
        tol = 1e-10
        # Iterative loop to adjust scale and offset
        for iteration in range(max_iter):
            # Apply current offset and scale to data
            adjusted = (input_data - offset) * scale
            norms = np.linalg.norm(adjusted, axis=1, keepdims=True)  # Compute norms for each row
            closest_point = adjusted / norms  # Normalize to project onto the unit sphere
            # Offset and scale changes for each axis
            offset_change = np.zeros(3)
            scale_change = np.ones(3)
            # Adjust offset and scale for each axis using linear regression
            for k in range(3):
                model = LinearRegression()
                model.fit(adjusted[:, k].reshape(-1, 1), closest_point[:, k], sample_weight=weights)
                offset_change[k] = model.intercept_  # Intercept represents offset adjustment
                scale_change[k] = model.coef_[0]  # Coefficient represents scale adjustment
                adjusted[:, k] = model.predict(adjusted[:, k].reshape(-1, 1))  # Update adjusted data
            # Apply changes to offset and scale
            offset += offset_change / (scale * scale_change)
            scale *= scale_change
            # Check convergence based on change in residuals
            new_res = np.mean(weights * np.sum((adjusted - closest_point) ** 2, axis=1))
            weights = np.minimum(1 / np.sqrt(np.sum((adjusted - closest_point) ** 2, axis=1)), 1 / 0.01)
            if abs(new_res - res) < tol:
                break  # Stop iteration if convergence criterion met
            res = new_res  # Update residual for next iteration
            print(f"Iteration {iteration + 1}: Residual = {res}")
        # Calculate final calibration error
        calib_error_end = np.mean(
            np.abs(np.sqrt(mean_gx ** 2 + mean_gy ** 2 + mean_gz ** 2) - 1))
        # Output summary of calibration results if verbose is enabled
    else:
        print("Insufficient nonmovement data for calibration.")
    # Return calibration results
    return {
        "scale": scale,
        "offset": offset,
        "calib_error_start": calib_error_start,
        "calib_error_end": calib_error_end,
        "npoints": npoints
    }

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
