"""
Principal Component Analysis module for SpectraLab.

Implements comprehensive PCA workflow:
- Component computation
- Variance analysis
- Feature importance (loadings)
- Sample scoring
- Outlier detection
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class PCAAnalyzer:
    """
    Performs Principal Component Analysis with comprehensive interpretation.
    
    Key capabilities:
    - Dimensionality reduction with variance preservation
    - Component interpretation via loadings
    - Sample clustering and outlier detection
    - Statistical significance testing
    """
    
    def __init__(self, n_components: Optional[int] = None, variance_threshold: float = 0.95):
        """
        Initialize PCA analyzer.
        
        Parameters
        ----------
        n_components : int, optional
            Number of components to compute. If None, uses variance_threshold
        variance_threshold : float, default 0.95
            Cumulative variance threshold for automatic component selection
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.pca: Optional[PCA] = None
        self.transformed_data: Optional[pd.DataFrame] = None
        self.loadings: Optional[pd.DataFrame] = None
        self.feature_names: Optional[list] = None
        self.sample_ids: Optional[list] = None
        
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit PCA model and transform data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Preprocessed data (samples × features)
            
        Returns
        -------
        pd.DataFrame
            Transformed data (samples × principal components)
        """
        logger.info("="*60)
        logger.info("Performing PCA")
        logger.info("="*60)
        
        self.feature_names = list(data.columns)
        self.sample_ids = list(data.index)
        
        n_samples, n_features = data.shape
        max_components = min(n_samples, n_features)
        
        # Determine number of components
        if self.n_components is None:
            # Use all components initially to find variance threshold
            logger.info(f"Finding components to explain {self.variance_threshold*100}% variance")
            self.pca = PCA(n_components=max_components)
            self.pca.fit(data)
            
            # Find number of components for variance threshold
            cumsum = np.cumsum(self.pca.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= self.variance_threshold) + 1
            
            logger.info(f"Selected {n_components} components (explaining "
                       f"{cumsum[n_components-1]*100:.2f}% variance)")
            
            # Refit with selected components
            self.pca = PCA(n_components=n_components)
        else:
            n_components = min(self.n_components, max_components)
            logger.info(f"Using {n_components} components")
            self.pca = PCA(n_components=n_components)
        
        # Fit and transform
        transformed = self.pca.fit_transform(data)
        
        # Create DataFrame with PC labels
        pc_labels = [f"PC{i+1}" for i in range(n_components)]
        self.transformed_data = pd.DataFrame(
            transformed,
            index=data.index,
            columns=pc_labels
        )
        
        # Compute loadings (feature contributions to PCs)
        self.loadings = pd.DataFrame(
            self.pca.components_.T,
            index=data.columns,
            columns=pc_labels
        )
        
        # Log variance explained
        var_explained = self.pca.explained_variance_ratio_
        cumsum = np.cumsum(var_explained)
        
        logger.info(f"\nVariance explained by component:")
        for i, (var, cum) in enumerate(zip(var_explained[:5], cumsum[:5]), 1):
            logger.info(f"  PC{i}: {var*100:.2f}% (cumulative: {cum*100:.2f}%)")
        
        if len(var_explained) > 5:
            logger.info(f"  ... ({len(var_explained)} components total)")
        
        logger.info(f"\nTotal variance explained: {cumsum[-1]*100:.2f}%")
        logger.info("="*60)
        
        return self.transformed_data
    
    def get_explained_variance(self) -> pd.DataFrame:
        """
        Get explained variance statistics.
        
        Returns
        -------
        pd.DataFrame
            Variance statistics for each component
        """
        if self.pca is None:
            raise ValueError("PCA not fitted yet")
        
        n_components = len(self.pca.explained_variance_ratio_)
        
        return pd.DataFrame({
            'Component': [f"PC{i+1}" for i in range(n_components)],
            'Variance_Explained': self.pca.explained_variance_ratio_,
            'Cumulative_Variance': np.cumsum(self.pca.explained_variance_ratio_),
            'Eigenvalue': self.pca.explained_variance_
        })
    
    def get_top_contributing_features(
        self,
        component: int = 1,
        n_features: int = 10,
        by_abs_value: bool = True
    ) -> pd.DataFrame:
        """
        Get features with strongest contribution to a principal component.
        
        Parameters
        ----------
        component : int, default 1
            Component number (1-indexed)
        n_features : int, default 10
            Number of top features to return
        by_abs_value : bool, default True
            If True, rank by absolute loading value
            
        Returns
        -------
        pd.DataFrame
            Top contributing features with loadings
        """
        if self.loadings is None:
            raise ValueError("PCA not fitted yet")
        
        pc_label = f"PC{component}"
        if pc_label not in self.loadings.columns:
            raise ValueError(f"Component {component} not available")
        
        loadings_series = self.loadings[pc_label]
        
        if by_abs_value:
            # Sort by absolute value
            top_features = loadings_series.abs().nlargest(n_features)
            # Get original values
            top_features = pd.DataFrame({
                'Feature': top_features.index,
                'Loading': loadings_series[top_features.index].values,
                'Abs_Loading': top_features.values
            })
        else:
            # Sort by value (can be negative)
            top_features = loadings_series.nlargest(n_features)
            top_features = pd.DataFrame({
                'Feature': top_features.index,
                'Loading': top_features.values,
                'Abs_Loading': np.abs(top_features.values)
            })
        
        return top_features.reset_index(drop=True)
    
    def detect_outliers(
        self,
        n_components: int = 2,
        threshold: float = 3.0,
        method: str = 'mahalanobis'
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Detect outliers in PCA space.
        
        Parameters
        ----------
        n_components : int, default 2
            Number of components to use for outlier detection
        threshold : float, default 3.0
            Standard deviations from center for outlier classification
        method : str, default 'mahalanobis'
            Distance metric ('mahalanobis' or 'euclidean')
            
        Returns
        -------
        tuple
            (outlier_info DataFrame, outlier_mask array)
        """
        if self.transformed_data is None:
            raise ValueError("PCA not fitted yet")
        
        # Use first n_components
        pc_data = self.transformed_data.iloc[:, :n_components].values
        
        if method == 'mahalanobis':
            # Mahalanobis distance (accounts for variance in each component)
            center = pc_data.mean(axis=0)
            cov = np.cov(pc_data.T)
            
            # Handle singular covariance matrix
            try:
                cov_inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                logger.warning("Singular covariance matrix, adding regularization")
                cov_inv = np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-6)
            
            distances = np.array([
                np.sqrt((x - center).T @ cov_inv @ (x - center))
                for x in pc_data
            ])
            
        else:  # euclidean
            # Euclidean distance from center
            center = pc_data.mean(axis=0)
            distances = np.sqrt(((pc_data - center) ** 2).sum(axis=1))
        
        # Normalize distances
        mean_dist = distances.mean()
        std_dist = distances.std()
        z_scores = (distances - mean_dist) / std_dist
        
        # Identify outliers
        is_outlier = z_scores > threshold
        
        outlier_info = pd.DataFrame({
            'Sample': self.sample_ids,
            'Distance': distances,
            'Z_Score': z_scores,
            'Is_Outlier': is_outlier
        })
        
        n_outliers = is_outlier.sum()
        logger.info(f"Detected {n_outliers} outliers using {method} distance "
                   f"(threshold={threshold} std)")
        
        return outlier_info, is_outlier
    
    def get_sample_contributions(self, component: int = 1, n_samples: int = 10) -> pd.DataFrame:
        """
        Get samples with strongest contribution to variance in a component.
        
        Parameters
        ----------
        component : int, default 1
            Component number (1-indexed)
        n_samples : int, default 10
            Number of top samples to return
            
        Returns
        -------
        pd.DataFrame
            Top contributing samples
        """
        if self.transformed_data is None:
            raise ValueError("PCA not fitted yet")
        
        pc_label = f"PC{component}"
        scores = self.transformed_data[pc_label].abs()
        top_samples = scores.nlargest(n_samples)
        
        return pd.DataFrame({
            'Sample': top_samples.index,
            'PC_Score': self.transformed_data.loc[top_samples.index, pc_label].values,
            'Abs_Score': top_samples.values
        }).reset_index(drop=True)
    
    def analyze_feature_importance(self, n_components: int = 3) -> pd.DataFrame:
        """
        Compute overall feature importance across multiple components.
        
        Combines loadings from multiple PCs to rank overall feature importance.
        
        Parameters
        ----------
        n_components : int, default 3
            Number of components to consider
            
        Returns
        -------
        pd.DataFrame
            Feature importance scores
        """
        if self.loadings is None:
            raise ValueError("PCA not fitted yet")
        
        n_components = min(n_components, len(self.loadings.columns))
        pc_labels = [f"PC{i+1}" for i in range(n_components)]
        
        # Weight by explained variance
        weights = self.pca.explained_variance_ratio_[:n_components]
        
        # Compute weighted importance
        importance_scores = np.zeros(len(self.loadings))
        
        for i, pc_label in enumerate(pc_labels):
            # Contribution = abs(loading) * variance_explained
            importance_scores += np.abs(self.loadings[pc_label].values) * weights[i]
        
        importance_df = pd.DataFrame({
            'Feature': self.loadings.index,
            'Importance_Score': importance_scores,
            'Rank': range(1, len(importance_scores) + 1)
        })
        
        importance_df = importance_df.sort_values('Importance_Score', ascending=False)
        importance_df['Rank'] = range(1, len(importance_df) + 1)
        
        return importance_df.reset_index(drop=True)
    
    def get_biplot_data(self, pc_x: int = 1, pc_y: int = 2, n_vectors: int = 10) -> Dict:
        """
        Get data for biplot visualization (samples + feature vectors).
        
        Parameters
        ----------
        pc_x : int, default 1
            Component for x-axis
        pc_y : int, default 2
            Component for y-axis
        n_vectors : int, default 10
            Number of feature vectors to include
            
        Returns
        -------
        dict
            Biplot data including sample scores and feature loadings
        """
        if self.transformed_data is None or self.loadings is None:
            raise ValueError("PCA not fitted yet")
        
        pc_x_label = f"PC{pc_x}"
        pc_y_label = f"PC{pc_y}"
        
        # Sample scores
        sample_scores = self.transformed_data[[pc_x_label, pc_y_label]]
        
        # Feature loadings (scaled for visualization)
        loadings_x = self.loadings[pc_x_label]
        loadings_y = self.loadings[pc_y_label]
        
        # Select top features by combined loading magnitude
        combined_magnitude = np.sqrt(loadings_x**2 + loadings_y**2)
        top_features = combined_magnitude.nlargest(n_vectors).index
        
        feature_vectors = pd.DataFrame({
            'Feature': top_features,
            f'{pc_x_label}_loading': loadings_x[top_features].values,
            f'{pc_y_label}_loading': loadings_y[top_features].values,
            'Magnitude': combined_magnitude[top_features].values
        })
        
        return {
            'sample_scores': sample_scores,
            'feature_vectors': feature_vectors,
            'variance_explained': {
                pc_x_label: self.pca.explained_variance_ratio_[pc_x - 1],
                pc_y_label: self.pca.explained_variance_ratio_[pc_y - 1]
            }
        }
    
    def get_reconstruction_error(self, original_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute reconstruction error for each sample.
        
        Measures how well PCA captures each sample's variance.
        
        Parameters
        ----------
        original_data : pd.DataFrame
            Original preprocessed data
            
        Returns
        -------
        pd.DataFrame
            Reconstruction errors per sample
        """
        if self.pca is None:
            raise ValueError("PCA not fitted yet")
        
        # Reconstruct data from PCA
        reconstructed = self.pca.inverse_transform(self.transformed_data)
        
        # Compute reconstruction error per sample
        errors = np.sqrt(((original_data.values - reconstructed) ** 2).sum(axis=1))
        
        return pd.DataFrame({
            'Sample': self.sample_ids,
            'Reconstruction_Error': errors,
            'Relative_Error': errors / np.sqrt((original_data.values ** 2).sum(axis=1))
        }).sort_values('Reconstruction_Error', ascending=False).reset_index(drop=True)