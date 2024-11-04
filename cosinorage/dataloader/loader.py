import pandas as pd
from .utility import concatenate_csv, get_posix_timestamps, filter_incomplete_days
from .enmo import calculate_enmo, calculate_minute_level_enmo

class DataLoader:
    def __init__(self):
        self.enmo_per_minute = pd.DataFrame()

    def load_data(self):
        """
        Method intended to be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def save_data(self, output_path):
        """
        Method intended to be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")
class AccelerometerDataLoader(DataLoader):
    def __init__(self, input_dir_path):
        super().__init__()

        self.input_dir_path = input_dir_path
        self.data = pd.DataFrame()

    def load_data(self):
        self.data = concatenate_csv(self.input_dir_path)
        self.data["TIMESTAMP"] = get_posix_timestamps(self.data["HEADER_TIMESTAMP"])

        self.data = calculate_enmo(self.data)
        self.data = filter_incomplete_days(self.data)
        self.enmo_per_minute = calculate_minute_level_enmo(self.data)

    def save_data(self, output_file_path):
        self.enmo_per_minute.to_csv(output_file_path, index=False)

class ENMODataLoader(DataLoader):
    def __init__(self, input_file_path):
        super().__init__()

        self.input_file_path = input_file_path

    def load_data(self):
        self.enmo_per_minute = pd.read_csv(self.input_file_path)
        self.enmo_per_minute = self.enmo_per_minute[['time', 'ENMO_t']]
        self.enmo_per_minute['TIMESTAMP'] = get_posix_timestamps(self.enmo_per_minute['time'], sample_rate=1/60)
        self.enmo_per_minute = self.enmo_per_minute.rename(columns={'ENMO_t': 'ENMO'})
        self.enmo_per_minute = self.enmo_per_minute.drop(columns=['time'])

        self.enmo_per_minute = filter_incomplete_days(self.enmo_per_minute)


    def save_data(self, output_file_path):
        self.enmo_per_minute.to_csv(output_file_path, index=False)




