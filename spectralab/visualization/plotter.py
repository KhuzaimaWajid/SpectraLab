"""
Visualization module for SpectraLab.

Creates publication-quality plots for PCA analysis:
- Scree plots (variance explained)
- Score plots (sample clustering)
- Loading plots (feature importance)
- Biplots (combined view)
- 3D visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class Plotter:
    """
    Creates comprehensive visualizations for PCA analysis.
    
    Designed for exploratory data analysis with emphasis on:
    - Clear, interpretable plots
    - Publication-quality output
    - Flexible customization
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize plotter.
        
        Parameters
        ----------
        output_dir : str, optional
            Directory to save plots. If None, plots are shown but not saved.
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Plots will be saved to: {self.output_dir}")
    
    def _save_or_show(self, fig: plt.Figure, filename: str):
        """Save figure if output_dir is set, otherwise show."""
        if self.output_dir:
            filepath = self.output_dir / filename
            fig.savefig(filepath, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {filename}")
        else:
            plt.show()
        plt.close(fig)
    
    def plot_scree(
        self,
        variance_df: pd.DataFrame,
        title: str = "Scree Plot",
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Create scree plot showing variance explained by each component.
        
        Parameters
        ----------
        variance_df : pd.DataFrame
            Variance data from PCAAnalyzer.get_explained_variance()
        title : str
            Plot title
        figsize : tuple
            Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        n_components = len(variance_df)
        components = range(1, n_components + 1)
        
        # Individual variance
        ax1.bar(components, variance_df['Variance_Explained'] * 100, 
                alpha=0.7, color='steelblue', edgecolor='black')
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Variance Explained (%)')
        ax1.set_title('Variance per Component')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative variance
        ax2.plot(components, variance_df['Cumulative_Variance'] * 100, 
                marker='o', linewidth=2, markersize=6, color='steelblue')
        ax2.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='95% threshold')
        ax2.axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='90% threshold')
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Cumulative Variance Explained (%)')
        ax2.set_title('Cumulative Variance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        fig.tight_layout()
        
        self._save_or_show(fig, 'scree_plot.png')
    
    def plot_scores_2d(
        self,
        scores: pd.DataFrame,
        pc_x: int = 1,
        pc_y: int = 2,
        color_by: Optional[pd.Series] = None,
        labels: Optional[List[str]] = None,
        title: str = "PCA Score Plot",
        figsize: Tuple[int, int] = (10, 8),
        show_labels: bool = False,
        variance_explained: Optional[dict] = None
    ):
        """
        Create 2D scatter plot of PCA scores.
        
        Parameters
        ----------
        scores : pd.DataFrame
            PCA scores from PCAAnalyzer
        pc_x : int
            Component for x-axis
        pc_y : int
            Component for y-axis
        color_by : pd.Series, optional
            Values to color points by (e.g., groups, outliers)
        labels : list, optional
            Sample labels to display
        title : str
            Plot title
        figsize : tuple
            Figure size
        show_labels : bool
            Whether to show sample labels
        variance_explained : dict, optional
            Variance explained by each PC for axis labels
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        pc_x_label = f"PC{pc_x}"
        pc_y_label = f"PC{pc_y}"
        
        x = scores[pc_x_label]
        y = scores[pc_y_label]
        
        # Create scatter plot
        if color_by is not None:
            scatter = ax.scatter(x, y, c=color_by, cmap='viridis', 
                               s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
            plt.colorbar(scatter, ax=ax, label='Color Scale')
        else:
            ax.scatter(x, y, s=80, alpha=0.7, color='steelblue',
                      edgecolors='black', linewidth=0.5)
        
        # Add sample labels if requested
        if show_labels:
            for i, (xi, yi) in enumerate(zip(x, y)):
                label = labels[i] if labels else scores.index[i]
                ax.annotate(label, (xi, yi), fontsize=8, alpha=0.7,
                           xytext=(5, 5), textcoords='offset points')
        
        # Axis labels with variance explained
        if variance_explained:
            xlabel = f"{pc_x_label} ({variance_explained[pc_x_label]*100:.1f}%)"
            ylabel = f"{pc_y_label} ({variance_explained[pc_y_label]*100:.1f}%)"
        else:
            xlabel = pc_x_label
            ylabel = pc_y_label
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.5)
        
        fig.tight_layout()
        
        self._save_or_show(fig, f'scores_PC{pc_x}_PC{pc_y}.png')
    
    def plot_scores_3d(
        self,
        scores: pd.DataFrame,
        pc_x: int = 1,
        pc_y: int = 2,
        pc_z: int = 3,
        color_by: Optional[pd.Series] = None,
        title: str = "3D PCA Score Plot",
        figsize: Tuple[int, int] = (12, 9)
    ):
        """
        Create 3D scatter plot of PCA scores.
        
        Parameters
        ----------
        scores : pd.DataFrame
            PCA scores
        pc_x, pc_y, pc_z : int
            Components for each axis
        color_by : pd.Series, optional
            Values to color points by
        title : str
            Plot title
        figsize : tuple
            Figure size
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        pc_x_label = f"PC{pc_x}"
        pc_y_label = f"PC{pc_y}"
        pc_z_label = f"PC{pc_z}"
        
        x = scores[pc_x_label]
        y = scores[pc_y_label]
        z = scores[pc_z_label]
        
        if color_by is not None:
            scatter = ax.scatter(x, y, z, c=color_by, cmap='viridis',
                               s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
            plt.colorbar(scatter, ax=ax, label='Color Scale', shrink=0.5)
        else:
            ax.scatter(x, y, z, s=80, alpha=0.7, color='steelblue',
                      edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel(pc_x_label, fontsize=11)
        ax.set_ylabel(pc_y_label, fontsize=11)
        ax.set_zlabel(pc_z_label, fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        fig.tight_layout()
        
        self._save_or_show(fig, f'scores_3D_PC{pc_x}_PC{pc_y}_PC{pc_z}.png')
    
    def plot_loadings(
        self,
        loadings_df: pd.DataFrame,
        component: int = 1,
        n_features: int = 15,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Create bar plot of feature loadings for a component.
        
        Parameters
        ----------
        loadings_df : pd.DataFrame
            Top loadings from PCAAnalyzer.get_top_contributing_features()
        component : int
            Component number
        n_features : int
            Number of features to show
        title : str, optional
            Plot title
        figsize : tuple
            Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        data = loadings_df.head(n_features).copy()
        
        # Sort by loading value for better visualization
        data = data.sort_values('Loading')
        
        # Color bars by sign
        colors = ['steelblue' if x >= 0 else 'coral' for x in data['Loading']]
        
        ax.barh(range(len(data)), data['Loading'], color=colors, 
                edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data['Feature'], fontsize=9)
        ax.set_xlabel('Loading Value', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        
        if title is None:
            title = f"Top {n_features} Feature Loadings - PC{component}"
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.axvline(x=0, color='black', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')
        
        fig.tight_layout()
        
        self._save_or_show(fig, f'loadings_PC{component}.png')
    
    def plot_biplot(
        self,
        biplot_data: dict,
        pc_x: int = 1,
        pc_y: int = 2,
        title: str = "PCA Biplot",
        figsize: Tuple[int, int] = (12, 10),
        scale_vectors: float = 3.0
    ):
        """
        Create biplot showing both samples and feature vectors.
        
        Parameters
        ----------
        biplot_data : dict
            Biplot data from PCAAnalyzer.get_biplot_data()
        pc_x, pc_y : int
            Components to plot
        title : str
            Plot title
        figsize : tuple
            Figure size
        scale_vectors : float
            Scaling factor for feature vectors
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        scores = biplot_data['sample_scores']
        vectors = biplot_data['feature_vectors']
        variance = biplot_data['variance_explained']
        
        pc_x_label = f"PC{pc_x}"
        pc_y_label = f"PC{pc_y}"
        
        # Plot sample scores
        ax.scatter(scores[pc_x_label], scores[pc_y_label],
                  alpha=0.6, s=80, color='steelblue', 
                  edgecolors='black', linewidth=0.5, label='Samples')
        
        # Plot feature vectors
        for _, row in vectors.iterrows():
            x = row[f'{pc_x_label}_loading'] * scale_vectors
            y = row[f'{pc_y_label}_loading'] * scale_vectors
            
            ax.arrow(0, 0, x, y, head_width=0.1, head_length=0.1,
                    fc='red', ec='red', alpha=0.7, linewidth=1.5)
            ax.text(x * 1.1, y * 1.1, row['Feature'], 
                   fontsize=9, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
        
        # Axis labels with variance
        xlabel = f"{pc_x_label} ({variance[pc_x_label]*100:.1f}%)"
        ylabel = f"{pc_y_label} ({variance[pc_y_label]*100:.1f}%)"
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.5)
        ax.legend()
        
        fig.tight_layout()
        
        self._save_or_show(fig, f'biplot_PC{pc_x}_PC{pc_y}.png')
    
    def plot_outliers(
        self,
        scores: pd.DataFrame,
        outlier_info: pd.DataFrame,
        pc_x: int = 1,
        pc_y: int = 2,
        title: str = "Outlier Detection",
        figsize: Tuple[int, int] = (12, 5)
    ):
        """
        Visualize outlier detection results.
        
        Parameters
        ----------
        scores : pd.DataFrame
            PCA scores
        outlier_info : pd.DataFrame
            Outlier detection results from PCAAnalyzer.detect_outliers()
        pc_x, pc_y : int
            Components to plot
        title : str
            Plot title
        figsize : tuple
            Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        pc_x_label = f"PC{pc_x}"
        pc_y_label = f"PC{pc_y}"
        
        # Score plot with outliers highlighted
        outliers = outlier_info['Is_Outlier'].values
        
        ax1.scatter(scores.loc[~outliers, pc_x_label], 
                   scores.loc[~outliers, pc_y_label],
                   alpha=0.6, s=80, color='steelblue',
                   edgecolors='black', linewidth=0.5, label='Normal')
        ax1.scatter(scores.loc[outliers, pc_x_label], 
                   scores.loc[outliers, pc_y_label],
                   alpha=0.8, s=120, color='red',
                   edgecolors='black', linewidth=1, label='Outlier', marker='^')
        
        ax1.set_xlabel(pc_x_label, fontsize=11)
        ax1.set_ylabel(pc_y_label, fontsize=11)
        ax1.set_title('Score Plot with Outliers', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
        ax1.axvline(x=0, color='k', linewidth=0.5, alpha=0.5)
        
        # Distance distribution
        ax2.hist(outlier_info.loc[~outliers, 'Z_Score'], bins=30, 
                alpha=0.7, color='steelblue', edgecolor='black', label='Normal')
        ax2.hist(outlier_info.loc[outliers, 'Z_Score'], bins=10,
                alpha=0.7, color='red', edgecolor='black', label='Outlier')
        ax2.axvline(x=3, color='red', linestyle='--', linewidth=2, label='Threshold')
        ax2.set_xlabel('Z-Score (Standardized Distance)', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('Distance Distribution', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        fig.tight_layout()
        
        self._save_or_show(fig, 'outlier_detection.png')
    
    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        n_features: int = 20,
        title: str = "Feature Importance",
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Plot overall feature importance across components.
        
        Parameters
        ----------
        importance_df : pd.DataFrame
            Feature importance from PCAAnalyzer.analyze_feature_importance()
        n_features : int
            Number of top features to show
        title : str
            Plot title
        figsize : tuple
            Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        data = importance_df.head(n_features)
        
        ax.barh(range(len(data)), data['Importance_Score'],
               color='steelblue', edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data['Feature'], fontsize=9)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Invert y-axis so highest importance is at top
        ax.invert_yaxis()
        
        fig.tight_layout()
        
        self._save_or_show(fig, 'feature_importance.png')
    
    def create_summary_report(
        self,
        variance_df: pd.DataFrame,
        scores: pd.DataFrame,
        top_features: pd.DataFrame,
        figsize: Tuple[int, int] = (16, 12)
    ):
        """
        Create comprehensive summary figure with multiple plots.
        
        Parameters
        ----------
        variance_df : pd.DataFrame
            Variance data
        scores : pd.DataFrame
            PCA scores
        top_features : pd.DataFrame
            Top contributing features for PC1
        figsize : tuple
            Figure size
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Scree plot
        ax1 = fig.add_subplot(gs[0, 0])
        components = range(1, len(variance_df) + 1)
        ax1.bar(components, variance_df['Variance_Explained'] * 100,
               alpha=0.7, color='steelblue', edgecolor='black')
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Variance Explained (%)')
        ax1.set_title('Variance per Component', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative variance
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(components, variance_df['Cumulative_Variance'] * 100,
                marker='o', linewidth=2, markersize=6, color='steelblue')
        ax2.axhline(y=95, color='red', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Cumulative Variance (%)')
        ax2.set_title('Cumulative Variance', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Score plot
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.scatter(scores['PC1'], scores['PC2'], s=80, alpha=0.6,
                   color='steelblue', edgecolors='black', linewidth=0.5)
        ax3.set_xlabel('PC1')
        ax3.set_ylabel('PC2')
        ax3.set_title('Score Plot (PC1 vs PC2)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
        ax3.axvline(x=0, color='k', linewidth=0.5, alpha=0.5)
        
        # Loadings
        ax4 = fig.add_subplot(gs[1, 1])
        data = top_features.head(10).sort_values('Loading')
        colors = ['steelblue' if x >= 0 else 'coral' for x in data['Loading']]
        ax4.barh(range(len(data)), data['Loading'], color=colors,
                edgecolor='black', linewidth=0.5)
        ax4.set_yticks(range(len(data)))
        ax4.set_yticklabels(data['Feature'], fontsize=9)
        ax4.set_xlabel('Loading Value')
        ax4.set_title('Top 10 Feature Loadings - PC1', fontweight='bold')
        ax4.axvline(x=0, color='black', linewidth=1)
        ax4.grid(True, alpha=0.3, axis='x')
        
        fig.suptitle('PCA Analysis Summary', fontsize=16, fontweight='bold')
        
        self._save_or_show(fig, 'summary_report.png')