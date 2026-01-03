"""
Data preprocessing module for SpectraLab.

Implements standard preprocessing techniques for high-dimensional data:
- Normalization and standardization
- Missing value handling
- Feature filtering
- Data transformations
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Literal
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocesses high-dimensional datasets for PCA analysis.
    
    Key operations:
    - Handling missing values
    - Feature scaling (standardization/normalization)
    - Log transformations for skewed data
    - Low-variance feature removal
    - Maintains reproducibility through stored parameters
    """
    
    def __init__(self):
        self.scaler: Optional[object] = None
        self.scaling_method: Optional[str] = None
        self.removed_features: List[str] = []
        self.preprocessing_log: List[str] = []
        
    def handle_missing_values(
        self,
        data: pd.DataFrame,
        strategy: Literal['drop', 'mean', 'median', 'zero'] = 'mean',
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Handle missing values in dataset.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        strategy : {'drop', 'mean', 'median', 'zero'}, default 'mean'
            How to handle missing values:
            - 'drop': Remove samples with any missing values
            - 'mean': Impute with feature mean
            - 'median': Impute with feature median
            - 'zero': Replace with zero
        threshold : float, default 0.5
            For 'drop' strategy: remove features with more than this fraction missing
            
        Returns
        -------
        pd.DataFrame
            Data with missing values handled
        """
        data = data.copy()
        n_missing = data.isna().sum().sum()
        
        if n_missing == 0:
            logger.info("No missing values found")
            return data
        
        logger.info(f"Handling {n_missing} missing values using '{strategy}' strategy")
        
        if strategy == 'drop':
            # Remove features with too many missing values
            missing_fraction = data.isna().mean()
            features_to_drop = missing_fraction[missing_fraction > threshold].index.tolist()
            
            if features_to_drop:
                data = data.drop(columns=features_to_drop)
                self.removed_features.extend(features_to_drop)
                logger.info(f"Removed {len(features_to_drop)} features with >{threshold*100}% missing")
            
            # Remove samples with any remaining missing values
            n_samples_before = len(data)
            data = data.dropna()
            n_dropped = n_samples_before - len(data)
            
            if n_dropped > 0:
                logger.info(f"Removed {n_dropped} samples with missing values")
                
        elif strategy == 'mean':
            data = data.fillna(data.mean())
            logger.info("Imputed missing values with feature means")
            
        elif strategy == 'median':
            data = data.fillna(data.median())
            logger.info("Imputed missing values with feature medians")
            
        elif strategy == 'zero':
            data = data.fillna(0)
            logger.info("Replaced missing values with zeros")
        
        self.preprocessing_log.append(f"Missing values: {strategy}")
        
        return data
    
    def scale_features(
        self,
        data: pd.DataFrame,
        method: Literal['standard', 'minmax', 'none'] = 'standard'
    ) -> pd.DataFrame:
        """
        Scale features for PCA analysis.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        method : {'standard', 'minmax', 'none'}, default 'standard'
            Scaling method:
            - 'standard': Z-score normalization (mean=0, std=1)
            - 'minmax': Scale to [0, 1] range
            - 'none': No scaling
            
        Returns
        -------
        pd.DataFrame
            Scaled data
        """
        if method == 'none':
            logger.info("No feature scaling applied")
            return data
        
        data = data.copy()
        logger.info(f"Applying {method} scaling")
        
        if method == 'standard':
            self.scaler = StandardScaler()
            scaled_values = self.scaler.fit_transform(data)
            logger.info("Applied z-score standardization (mean=0, std=1)")
            
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
            scaled_values = self.scaler.fit_transform(data)
            logger.info("Applied min-max scaling to [0, 1]")
        
        self.scaling_method = method
        scaled_data = pd.DataFrame(
            scaled_values,
            index=data.index,
            columns=data.columns
        )
        
        self.preprocessing_log.append(f"Scaling: {method}")
        
        return scaled_data
    
    def apply_log_transform(
        self,
        data: pd.DataFrame,
        offset: float = 1.0
    ) -> pd.DataFrame:
        """
        Apply log transformation to handle skewed intensity data.
        
        Useful for mass spectrometry and other exponential-scale measurements.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        offset : float, default 1.0
            Offset to add before log (to handle zeros): log(x + offset)
            
        Returns
        -------
        pd.DataFrame
            Log-transformed data
        """
        data = data.copy()
        logger.info(f"Applying log transform with offset={offset}")
        
        # Check for negative values
        if (data < 0).any().any():
            logger.warning("Negative values detected - they will remain negative after log transform")
            # Apply log only to positive values
            positive_mask = data > 0
            data[positive_mask] = np.log(data[positive_mask] + offset)
        else:
            data = np.log(data + offset)
        
        self.preprocessing_log.append(f"Log transform: offset={offset}")
        
        return data
    
    def remove_low_variance_features(
        self,
        data: pd.DataFrame,
        threshold: float = 0.01
    ) -> pd.DataFrame:
        """
        Remove features with low variance (near-constant).
        
        Low-variance features contribute little information and can cause
        numerical issues in PCA.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        threshold : float, default 0.01
            Variance threshold (features with variance < threshold are removed)
            
        Returns
        -------
        pd.DataFrame
            Data with low-variance features removed
        """
        data = data.copy()
        variances = data.var()
        
        low_var_features = variances[variances < threshold].index.tolist()
        
        if low_var_features:
            data = data.drop(columns=low_var_features)
            self.removed_features.extend(low_var_features)
            logger.info(f"Removed {len(low_var_features)} low-variance features "
                       f"(variance < {threshold})")
            self.preprocessing_log.append(f"Low variance removal: threshold={threshold}")
        else:
            logger.info(f"No features below variance threshold {threshold}")
        
        return data
    
    def preprocess_pipeline(
        self,
        data: pd.DataFrame,
        missing_strategy: Literal['drop', 'mean', 'median', 'zero'] = 'mean',
        scaling: Literal['standard', 'minmax', 'none'] = 'standard',
        log_transform: bool = False,
        log_offset: float = 1.0,
        remove_low_variance: bool = True,
        variance_threshold: float = 0.01
    ) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Applies preprocessing steps in proper order:
        1. Handle missing values
        2. Log transform (if requested)
        3. Remove low variance features
        4. Scale features
        
        Parameters
        ----------
        data : pd.DataFrame
            Raw input data
        missing_strategy : str, default 'mean'
            Strategy for missing values
        scaling : str, default 'standard'
            Scaling method
        log_transform : bool, default False
            Whether to apply log transformation
        log_offset : float, default 1.0
            Offset for log transform
        remove_low_variance : bool, default True
            Whether to remove low-variance features
        variance_threshold : float, default 0.01
            Threshold for low-variance removal
            
        Returns
        -------
        pd.DataFrame
            Preprocessed data ready for PCA
        """
        logger.info("="*60)
        logger.info("Starting preprocessing pipeline")
        logger.info("="*60)
        
        self.preprocessing_log = []
        self.removed_features = []
        
        original_shape = data.shape
        
        # Step 1: Handle missing values
        data = self.handle_missing_values(data, strategy=missing_strategy)
        
        # Step 2: Log transform (before scaling)
        if log_transform:
            data = self.apply_log_transform(data, offset=log_offset)
        
        # Step 3: Remove low variance features
        if remove_low_variance:
            data = self.remove_low_variance_features(data, threshold=variance_threshold)
        
        # Step 4: Scale features
        data = self.scale_features(data, method=scaling)
        
        final_shape = data.shape
        
        logger.info("="*60)
        logger.info(f"Preprocessing complete: {original_shape} → {final_shape}")
        logger.info("="*60)
        
        return data
    
    def get_preprocessing_summary(self) -> dict:
        """
        Get summary of preprocessing operations performed.
        
        Returns
        -------
        dict
            Summary of preprocessing steps
        """
        return {
            'steps': self.preprocessing_log,
            'scaling_method': self.scaling_method,
            'removed_features': self.removed_features,
            'n_removed_features': len(self.removed_features)
        }