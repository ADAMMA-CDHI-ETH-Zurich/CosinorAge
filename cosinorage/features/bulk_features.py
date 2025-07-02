###########################################################################
# Copyright (C) 2025 ETH Zurich
# CosinorAge: Prediction of biological age based on accelerometer data
# using the CosinorAge method proposed by Shim, Fleisch and Barata
# (https://www.nature.com/articles/s41746-024-01111-x)
# 
# Authors: Jacob Leo Oskar Hunecke
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#         http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

"""
bulk_features.py
----------------

Provides the BulkWearableFeatures class for batch computation and statistical analysis
of wearable-derived features across multiple datasets. This is useful for studies
involving cohorts or large-scale data, enabling summary statistics, correlation analysis,
and robust error handling for failed data handlers.

Typical usage example::

    handlers = [DataHandler1, DataHandler2, ...]
    bulk = BulkWearableFeatures(handlers)
    stats = bulk.get_distribution_stats()
    summary_df = bulk.get_summary_dataframe()
    corr = bulk.get_feature_correlation_matrix()

"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict
from ..datahandlers import DataHandler
from .features import WearableFeatures

class BulkWearableFeatures:
    """A class for computing and managing features from multiple wearable accelerometer datasets.

    This class processes multiple DataHandler instances to compute features for each
    and then calculates statistical distributions (mean, std, quartiles, etc.) across
    all datasets.

    Attributes:
        handlers (List[DataHandler]): List of DataHandler instances
        features_args (dict): Arguments for feature computation
        individual_features (List[dict]): List of feature dictionaries for each handler
        distribution_stats (dict): Statistical distributions across all features
    """

    def __init__(self, 
                 handlers: List[DataHandler],
                 features_args: dict = {},
                 compute_distributions: bool = True):
        """Initialize BulkWearableFeatures with multiple DataHandler instances.

        Args:
            handlers (List[DataHandler]): List of DataHandler instances containing ENMO data
            features_args (dict): Arguments for feature computation
            compute_distributions (bool): Whether to compute statistical distributions
        """
        self.handlers = handlers
        self.features_args = features_args
        self.individual_features = []
        self.distribution_stats = {}
        self.failed_handlers = []
        
        self.__run(compute_distributions)

    def __run(self, compute_distributions: bool = True):
        """Compute features for all handlers and optionally compute distributions."""
        
        # Compute features for each handler
        for i, handler in enumerate(self.handlers):
            try:
                wearable_features = WearableFeatures(handler, self.features_args)
                self.individual_features.append(wearable_features.get_features())
            except Exception as e:
                print(f"Failed to compute features for handler {i}: {str(e)}")
                self.failed_handlers.append((i, str(e)))
                self.individual_features.append(None)
        
        # Compute distributions if requested and we have successful computations
        if compute_distributions and len(self.individual_features) > 0:
            self.__compute_distributions()

    def __compute_distributions(self):
        """Compute statistical distributions across all features."""
        
        # Filter out None values (failed computations)
        valid_features = [f for f in self.individual_features if f is not None]
        
        if len(valid_features) == 0:
            print("No valid features found for distribution computation")
            return
        
        # Flatten all features into a single DataFrame
        flattened_features = self.__flatten_features(valid_features)
        
        # Compute statistics for each feature
        self.distribution_stats = self.__compute_feature_statistics(flattened_features)

    def __flatten_features(self, features_list: List[dict]) -> pd.DataFrame:
        """Flatten nested feature dictionaries into a DataFrame.
        
        Args:
            features_list (List[dict]): List of feature dictionaries
            
        Returns:
            pd.DataFrame: Flattened features DataFrame
        """
        flattened_data = []
        
        for i, features in enumerate(features_list):
            row = {'handler_index': i}
            
            # Flatten nested dictionaries
            for category, category_features in features.items():
                if isinstance(category_features, dict):
                    for feature_name, feature_value in category_features.items():
                        # Skip flag features
                        if feature_name.endswith('_flag'):
                            continue
                        
                        # Handle different data types
                        if isinstance(feature_value, (list, np.ndarray)):
                            # Only aggregate if all elements are numeric
                            if len(feature_value) > 0 and all(isinstance(x, (int, float, np.number, np.floating, np.integer)) for x in feature_value):
                                row[f"{category}_{feature_name}"] = np.mean(feature_value)
                            else:
                                # Skip non-numeric lists (e.g., Timestamps)
                                continue
                        elif isinstance(feature_value, (int, float, np.number, np.floating, np.integer)):
                            row[f"{category}_{feature_name}"] = feature_value
                        else:
                            # Skip non-numeric features
                            continue
                else:
                    # Direct feature value
                    if isinstance(category_features, (int, float, np.number, np.floating, np.integer)):
                        row[category] = category_features
            
            flattened_data.append(row)
        
        return pd.DataFrame(flattened_data)

    def __compute_feature_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compute statistical measures for each feature.
        
        Args:
            df (pd.DataFrame): Flattened features DataFrame
            
        Returns:
            Dict[str, Dict[str, float]]: Statistics for each feature
        """
        stats = {}
        
        # Exclude non-numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != 'handler_index']
        
        for column in numeric_columns:
            values = df[column].dropna()
            
            if len(values) == 0:
                continue
                
            column_stats = {
                'count': len(values),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'q25': float(np.percentile(values, 25)),
                'q75': float(np.percentile(values, 75)),
                'iqr': float(np.percentile(values, 75) - np.percentile(values, 25))
            }
            
            # Compute mode (most frequent value)
            try:
                mode_values = values.mode()
                if len(mode_values) > 0:
                    column_stats['mode'] = float(mode_values.iloc[0])
                else:
                    column_stats['mode'] = np.nan
            except:
                column_stats['mode'] = np.nan
            
            # Compute skewness and kurtosis
            try:
                column_stats['skewness'] = float(pd.Series(values).skew())
                column_stats['kurtosis'] = float(pd.Series(values).kurtosis())
            except:
                column_stats['skewness'] = np.nan
                column_stats['kurtosis'] = np.nan
            
            stats[column] = column_stats
        
        return stats

    def get_individual_features(self) -> List[dict]:
        """Returns the individual feature dictionaries for each handler.
        
        Returns:
            List[dict]: List of feature dictionaries, one per handler. If a handler failed, its entry is None.
        """
        return self.individual_features

    def get_distribution_stats(self) -> Dict[str, Dict[str, float]]:
        """Returns the statistical distributions across all features.
        
        Returns:
            Dict[str, Dict[str, float]]: Statistical distributions (mean, std, min, max, etc.) for each feature across all handlers.
        """
        return self.distribution_stats

    def get_failed_handlers(self) -> List[tuple]:
        """Returns information about handlers that failed during feature computation.
        
        Returns:
            List[tuple]: List of (handler_index, error_message) tuples for handlers that failed.
        """
        return self.failed_handlers

    def get_summary_dataframe(self) -> pd.DataFrame:
        """Returns a summary DataFrame with all statistical measures for each feature.
        
        Returns:
            pd.DataFrame: Summary DataFrame with features as rows and statistics as columns.
        """
        if not self.distribution_stats:
            return pd.DataFrame()
        
        # Convert to DataFrame
        summary_df = pd.DataFrame.from_dict(self.distribution_stats, orient='index')
        summary_df.index.name = 'feature'
        summary_df.reset_index(inplace=True)
        
        return summary_df

    def get_feature_correlation_matrix(self) -> pd.DataFrame:
        """Returns correlation matrix between features across all handlers.
        
        Returns:
            pd.DataFrame: Correlation matrix of features (empty if insufficient data).
        """
        # Flatten features and create DataFrame
        valid_features = [f for f in self.individual_features if f is not None]
        if len(valid_features) == 0:
            return pd.DataFrame()
        
        flattened_df = self.__flatten_features(valid_features)
        
        # Select only numeric columns and compute correlation
        numeric_columns = flattened_df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != 'handler_index']
        
        if len(numeric_columns) < 2:
            return pd.DataFrame()
        
        correlation_matrix = flattened_df[numeric_columns].corr()
        return correlation_matrix 