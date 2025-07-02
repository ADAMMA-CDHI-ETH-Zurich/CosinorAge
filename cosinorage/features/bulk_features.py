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
of wearable-derived features across multiple datasets. This module is essential for
cohort studies and large-scale data analysis, enabling comprehensive feature extraction,
statistical summarization, and correlation analysis across multiple participants.

The BulkWearableFeatures class processes multiple DataHandler instances simultaneously,
computes features for each using the WearableFeatures class, and provides statistical
distributions and correlation matrices across all datasets. It includes robust error
handling for failed computations and supports both individual feature access and
summary statistics.

Typical usage example::

    # Create multiple data handlers
    handlers = [DataHandler1, DataHandler2, DataHandler3]
    
    # Initialize bulk feature computation
    bulk = BulkWearableFeatures(handlers, compute_distributions=True)
    
    # Access individual features
    individual_features = bulk.get_individual_features()
    
    # Get statistical distributions
    stats = bulk.get_distribution_stats()
    
    # Get summary DataFrame
    summary_df = bulk.get_summary_dataframe()
    
    # Get correlation matrix
    corr_matrix = bulk.get_feature_correlation_matrix()
    
    # Check for failed handlers
    failed = bulk.get_failed_handlers()

Features computed include:
    - Cosinor analysis (MESOR, amplitude, acrophase)
    - Non-parametric measures (IV, IS, RA, M10, L5)
    - Physical activity metrics (sedentary, light, moderate, vigorous)
    - Sleep metrics (TST, WASO, PTA, NWB, SOL, SRI)

Statistical measures provided:
    - count, mean, std, min, max, median
    - q25, q75, iqr (interquartile range)
    - mode, skewness, kurtosis
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
    all datasets. It provides comprehensive analysis capabilities for cohort studies
    and large-scale wearable data analysis.

    The class handles feature computation failures gracefully, allowing analysis to
    continue even when some datasets fail to process. It provides both individual
    feature access and aggregated statistical summaries.

    Args:
        handlers (List[DataHandler]): List of DataHandler instances containing ENMO data.
            Each handler should have been properly initialized and loaded with data.
        features_args (dict, optional): Arguments for feature computation passed to
            WearableFeatures. Common arguments include:
            - 'pa_params': Physical activity parameters
            - 'sleep_params': Sleep detection parameters
            Defaults to empty dict.
        compute_distributions (bool, optional): Whether to compute statistical distributions
            across all features. If False, only individual features are computed.
            Defaults to True.

    Attributes:
        handlers (List[DataHandler]): List of DataHandler instances provided during initialization
        features_args (dict): Arguments for feature computation
        individual_features (List[dict]): List of feature dictionaries for each handler.
            Failed computations are represented as None.
        distribution_stats (dict): Statistical distributions across all features.
            Only populated if compute_distributions=True.
        failed_handlers (List[tuple]): List of (handler_index, error_message) tuples
            for handlers that failed during feature computation.

    Example:
        >>> from cosinorage.datahandlers import GalaxyDataHandler
        >>> from cosinorage.features import BulkWearableFeatures
        >>>
        >>> # Create multiple handlers
        >>> handlers = []
        >>> for i in range(3):
        ...     handler = GalaxyDataHandler(f"data/participant_{i}.csv")
        ...     handler.load_data()
        ...     handlers.append(handler)
        >>>
        >>> # Compute bulk features
        >>> bulk = BulkWearableFeatures(handlers, compute_distributions=True)
        >>>
        >>> # Get statistical summary
        >>> stats = bulk.get_distribution_stats()
        >>> print(f"Computed features for {len(stats)} feature types")
        >>>
        >>> # Check for failures
        >>> failed = bulk.get_failed_handlers()
        >>> if failed:
        ...     print(f"Failed handlers: {len(failed)}")
    """

    def __init__(self, 
                 handlers: List[DataHandler],
                 features_args: dict = {},
                 compute_distributions: bool = True):
        """Initialize BulkWearableFeatures with multiple DataHandler instances.

        Args:
            handlers (List[DataHandler]): List of DataHandler instances containing ENMO data.
                Each handler should have been properly initialized and loaded with data.
            features_args (dict, optional): Arguments for feature computation passed to
                WearableFeatures. Common arguments include:
                - 'pa_params': Physical activity parameters
                - 'sleep_params': Sleep detection parameters
                Defaults to empty dict.
            compute_distributions (bool, optional): Whether to compute statistical distributions
                across all features. If False, only individual features are computed.
                Defaults to True.

        Note:
            Empty handlers list is allowed and will result in empty individual_features
            and distribution_stats.
        """
        
        self.handlers = handlers
        self.features_args = features_args
        self.individual_features = []
        self.distribution_stats = {}
        self.failed_handlers = []
        
        self.__run(compute_distributions)

    def __run(self, compute_distributions: bool = True):
        """Compute features for all handlers and optionally compute distributions.
        
        This method processes each handler sequentially, computing features using
        the WearableFeatures class. Failed computations are logged and stored
        for later inspection.
        
        Args:
            compute_distributions (bool): Whether to compute statistical distributions
                after individual feature computation.
        """
        
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
        """Compute statistical distributions across all features.
        
        This method flattens all individual features into a single DataFrame and
        computes comprehensive statistical measures for each feature across all
        successful computations.
        """
        
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
        
        This method converts the nested structure of individual feature dictionaries
        into a flat DataFrame where each row represents one handler and each column
        represents one feature. Nested features are flattened using the pattern
        'category_feature_name'.
        
        Args:
            features_list (List[dict]): List of feature dictionaries from successful
                computations. Each dictionary contains nested feature categories.
            
        Returns:
            pd.DataFrame: Flattened features DataFrame with handler_index column and
                one column per feature. Non-numeric features are excluded.
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
        
        This method computes comprehensive statistical measures for each numeric
        feature across all handlers. It includes descriptive statistics, distribution
        measures, and handles edge cases like empty data or single values.
        
        Args:
            df (pd.DataFrame): Flattened features DataFrame with handler_index column
                and numeric feature columns.
            
        Returns:
            Dict[str, Dict[str, float]]: Dictionary where keys are feature names and
                values are dictionaries containing statistical measures:
                - count: Number of non-null values
                - mean: Arithmetic mean
                - std: Standard deviation
                - min: Minimum value
                - max: Maximum value
                - median: Median value
                - q25: 25th percentile
                - q75: 75th percentile
                - iqr: Interquartile range (q75 - q25)
                - mode: Most frequent value (if available)
                - skewness: Distribution skewness (if available)
                - kurtosis: Distribution kurtosis (if available)
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
                    column_stats['mode'] = float('nan')
            except:
                column_stats['mode'] = float('nan')
            
            # Compute skewness and kurtosis
            try:
                column_stats['skewness'] = float(pd.Series(values).skew())
                column_stats['kurtosis'] = float(pd.Series(values).kurtosis())
            except:
                column_stats['skewness'] = float('nan')
                column_stats['kurtosis'] = float('nan')
            
            stats[column] = column_stats
        
        return stats

    def get_individual_features(self) -> List[dict]:
        """Returns the individual feature dictionaries for each handler.
        
        This method provides access to the raw feature dictionaries computed for
        each handler. Failed computations are represented as None entries in the list.
        
        Returns:
            List[dict]: List of feature dictionaries, one per handler. Each dictionary
                contains nested feature categories (cosinor, nonparam, physical_activity, sleep).
                If a handler failed during computation, its entry is None.
                
        Example:
            >>> features = bulk.get_individual_features()
            >>> for i, feat in enumerate(features):
            ...     if feat is not None:
            ...         print(f"Handler {i}: MESOR = {feat['cosinor']['mesor']:.3f}")
            ...     else:
            ...         print(f"Handler {i}: Failed")
        """
        return self.individual_features

    def get_distribution_stats(self) -> Dict[str, Dict[str, float]]:
        """Returns the statistical distributions across all features.
        
        This method provides comprehensive statistical measures for each feature
        across all successful computations. The statistics include descriptive
        measures, distribution characteristics, and quartile information.
        
        Returns:
            Dict[str, Dict[str, float]]: Statistical distributions for each feature.
                Keys are feature names (e.g., 'cosinor_mesor', 'nonparam_IS').
                Values are dictionaries containing statistical measures:
                - count, mean, std, min, max, median
                - q25, q75, iqr (interquartile range)
                - mode, skewness, kurtosis
                
        Example:
            >>> stats = bulk.get_distribution_stats()
            >>> mesor_stats = stats['cosinor_mesor']
            >>> print(f"MESOR: mean={mesor_stats['mean']:.3f}, std={mesor_stats['std']:.3f}")
        """
        return self.distribution_stats

    def get_failed_handlers(self) -> List[tuple]:
        """Returns information about handlers that failed during feature computation.
        
        This method provides details about which handlers failed and why, allowing
        for debugging and quality control in large-scale analyses.
        
        Returns:
            List[tuple]: List of (handler_index, error_message) tuples for handlers
                that failed during feature computation. Empty list if all handlers
                succeeded.
                
        Example:
            >>> failed = bulk.get_failed_handlers()
            >>> for idx, error in failed:
            ...     print(f"Handler {idx} failed: {error}")
        """
        return self.failed_handlers

    def get_summary_dataframe(self) -> pd.DataFrame:
        """Returns a summary DataFrame with all statistical measures for each feature.
        
        This method converts the statistical distributions into a pandas DataFrame
        format, making it easy to export, analyze, or visualize the results.
        
        Returns:
            pd.DataFrame: Summary DataFrame with features as rows and statistics as columns.
                Columns include: feature, count, mean, std, min, max, median, q25, q75,
                iqr, mode, skewness, kurtosis. Empty DataFrame if no distributions
                were computed.
                
        Example:
            >>> summary_df = bulk.get_summary_dataframe()
            >>> print(summary_df.head())
            >>> # Export to CSV
            >>> summary_df.to_csv('feature_summary.csv', index=False)
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
        
        This method computes pairwise correlations between all numeric features
        across all successful computations. This is useful for understanding
        feature relationships and identifying redundant or highly correlated features.
        
        Returns:
            pd.DataFrame: Correlation matrix of features. Values range from -1 to 1,
                where 1 indicates perfect positive correlation, -1 indicates perfect
                negative correlation, and 0 indicates no correlation. Empty DataFrame
                if insufficient data (less than 2 features or no successful computations).
                
        Example:
            >>> corr_matrix = bulk.get_feature_correlation_matrix()
            >>> print(corr_matrix['cosinor_mesor']['nonparam_IS'])  # Correlation between MESOR and IS
            >>> # Visualize with heatmap
            >>> import seaborn as sns
            >>> sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
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