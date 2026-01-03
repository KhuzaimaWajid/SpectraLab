"""
Data ingestion module for SpectraLab.

Handles loading datasets from various formats, with robust error handling
and validation for high-dimensional scientific data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads and validates high-dimensional datasets for analysis.
    
    Supports:
    - CSV/TSV format (samples × features)
    - Missing value detection and reporting
    - Data shape validation
    - Metadata extraction
    """
    
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.sample_ids: Optional[List[str]] = None
        self.feature_names: Optional[List[str]] = None
        self.metadata: dict = {}
        
    def load_csv(
        self, 
        filepath: str,
        sample_col: Optional[str] = None,
        sep: str = ',',
        **kwargs
    ) -> pd.DataFrame:
        """
        Load dataset from CSV file.
        
        Parameters
        ----------
        filepath : str
            Path to CSV file
        sample_col : str, optional
            Column to use as sample identifiers (becomes index)
        sep : str, default ','
            Delimiter
        **kwargs
            Additional arguments passed to pandas.read_csv
            
        Returns
        -------
        pd.DataFrame
            Loaded dataset with samples as rows, features as columns
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
            
        logger.info(f"Loading data from {filepath}")
        
        try:
            self.data = pd.read_csv(filepath, sep=sep, **kwargs)
            
            # Set sample identifiers if specified
            if sample_col and sample_col in self.data.columns:
                self.data.set_index(sample_col, inplace=True)
                logger.info(f"Using '{sample_col}' as sample identifiers")
            
            self.sample_ids = list(self.data.index)
            self.feature_names = list(self.data.columns)
            
            # Store metadata
            self.metadata = {
                'n_samples': len(self.data),
                'n_features': len(self.data.columns),
                'source_file': str(filepath),
                'missing_values': self.data.isna().sum().sum(),
                'missing_percentage': (self.data.isna().sum().sum() / self.data.size) * 100
            }
            
            logger.info(f"Loaded {self.metadata['n_samples']} samples × "
                       f"{self.metadata['n_features']} features")
            
            if self.metadata['missing_values'] > 0:
                logger.warning(f"Found {self.metadata['missing_values']} missing values "
                             f"({self.metadata['missing_percentage']:.2f}%)")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def generate_synthetic_data(
        self,
        n_samples: int = 100,
        n_features: int = 50,
        n_informative: int = 10,
        noise_level: float = 0.1,
        random_state: Optional[int] = 42
    ) -> pd.DataFrame:
        """
        Generate synthetic high-dimensional dataset for testing.
        
        Creates data with controllable structure suitable for PCA analysis:
        - Informative features with correlation structure
        - Noise features
        - Optional clustering structure
        
        Parameters
        ----------
        n_samples : int, default 100
            Number of samples
        n_features : int, default 50
            Total number of features
        n_informative : int, default 10
            Number of informative features (rest are noise)
        noise_level : float, default 0.1
            Standard deviation of Gaussian noise
        random_state : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        pd.DataFrame
            Synthetic dataset
        """
        np.random.seed(random_state)
        logger.info(f"Generating synthetic data: {n_samples} samples × {n_features} features")
        
        # Generate latent factors (true underlying signals)
        n_factors = min(5, n_informative)
        latent_factors = np.random.randn(n_samples, n_factors)
        
        # Create informative features as linear combinations of latent factors
        informative_data = np.zeros((n_samples, n_informative))
        for i in range(n_informative):
            # Random weights for combining latent factors
            weights = np.random.randn(n_factors)
            informative_data[:, i] = latent_factors @ weights
        
        # Add noise to informative features
        informative_data += np.random.randn(n_samples, n_informative) * noise_level
        
        # Generate pure noise features
        n_noise = n_features - n_informative
        noise_data = np.random.randn(n_samples, n_noise) * (noise_level * 2)
        
        # Combine informative and noise features
        data = np.hstack([informative_data, noise_data])
        
        # Create DataFrame with proper labels
        sample_ids = [f"Sample_{i+1:03d}" for i in range(n_samples)]
        feature_names = (
            [f"Feature_{i+1:03d}" for i in range(n_informative)] +
            [f"Noise_{i+1:03d}" for i in range(n_noise)]
        )
        
        self.data = pd.DataFrame(data, index=sample_ids, columns=feature_names)
        self.sample_ids = sample_ids
        self.feature_names = feature_names
        
        self.metadata = {
            'n_samples': n_samples,
            'n_features': n_features,
            'n_informative': n_informative,
            'noise_level': noise_level,
            'synthetic': True,
            'missing_values': 0,
            'missing_percentage': 0.0
        }
        
        logger.info(f"Generated synthetic data with {n_informative} informative features")
        
        return self.data
    
    def get_summary(self) -> dict:
        """
        Get summary statistics of loaded data.
        
        Returns
        -------
        dict
            Summary statistics including shape, missing values, value ranges
        """
        if self.data is None:
            raise ValueError("No data loaded")
        
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        summary = {
            **self.metadata,
            'numeric_features': len(numeric_data.columns),
            'value_range': {
                'min': float(numeric_data.min().min()),
                'max': float(numeric_data.max().max()),
                'mean': float(numeric_data.mean().mean()),
                'std': float(numeric_data.std().mean())
            }
        }
        
        return summary
    
    def validate_data(self) -> Tuple[bool, List[str]]:
        """
        Validate loaded data for analysis readiness.
        
        Returns
        -------
        tuple
            (is_valid, list of issues)
        """
        if self.data is None:
            return False, ["No data loaded"]
        
        issues = []
        
        # Check for minimum samples
        if len(self.data) < 3:
            issues.append("Need at least 3 samples for PCA")
        
        # Check for minimum features
        if len(self.data.columns) < 2:
            issues.append("Need at least 2 features for PCA")
        
        # Check for non-numeric data
        non_numeric = self.data.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            issues.append(f"Found {len(non_numeric)} non-numeric columns: {list(non_numeric)[:5]}")
        
        # Check for constant features
        numeric_data = self.data.select_dtypes(include=[np.number])
        constant_features = numeric_data.columns[numeric_data.std() == 0].tolist()
        if constant_features:
            issues.append(f"Found {len(constant_features)} constant features (zero variance)")
        
        # Check for infinite values
        if np.isinf(numeric_data.values).any():
            issues.append("Dataset contains infinite values")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info("Data validation passed ✓")
        else:
            logger.warning(f"Data validation found {len(issues)} issue(s)")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        return is_valid, issues