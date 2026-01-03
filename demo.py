"""
SpectraLab Demo Script

Demonstrates all major features of the platform with synthetic data.
"""

import sys
from pathlib import Path

# IMPORTANT: Add parent directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Now import SpectraLab modules
from spectralab import DataLoader, DataPreprocessor, PCAAnalyzer, Plotter
import numpy as np


def main():
    """Run complete demo workflow."""
    
    print("="*70)
    print("SPECTRALAB DEMO - Comprehensive PCA Analysis")
    print("="*70)
    
    # ... rest of your code stays the same
    
    # Create output directory
    output_dir = Path('./demo_results')
    output_dir.mkdir(exist_ok=True)
    
    # ============================================================================
    # STEP 1: DATA GENERATION
    # ============================================================================
    print("\n" + "="*70)
    print("STEP 1: Generating Synthetic Dataset")
    print("="*70)
    
    loader = DataLoader()
    data = loader.generate_synthetic_data(
        n_samples=100,
        n_features=50,
        n_informative=10,
        noise_level=0.15,
        random_state=42
    )
    
    print(f"\nDataset shape: {data.shape}")
    print(f"Sample IDs: {data.index[:5].tolist()} ...")
    print(f"Features: {data.columns[:5].tolist()} ...")
    
    # Get and display summary
    summary = loader.get_summary()
    print(f"\nData Summary:")
    print(f"  - Samples: {summary['n_samples']}")
    print(f"  - Features: {summary['n_features']}")
    print(f"  - Value range: [{summary['value_range']['min']:.3f}, {summary['value_range']['max']:.3f}]")
    
    # Validate data
    is_valid, issues = loader.validate_data()
    print(f"\nValidation: {'✓ Passed' if is_valid else '✗ Failed'}")
    
    # ============================================================================
    # STEP 2: PREPROCESSING
    # ============================================================================
    print("\n" + "="*70)
    print("STEP 2: Data Preprocessing")
    print("="*70)
    
    preprocessor = DataPreprocessor()
    
    processed_data = preprocessor.preprocess_pipeline(
        data,
        missing_strategy='mean',
        scaling='standard',
        log_transform=False,
        remove_low_variance=True,
        variance_threshold=0.01
    )
    
    print(f"\nProcessed data shape: {processed_data.shape}")
    
    preproc_summary = preprocessor.get_preprocessing_summary()
    print(f"Preprocessing steps:")
    for step in preproc_summary['steps']:
        print(f"  - {step}")
    print(f"Removed features: {preproc_summary['n_removed_features']}")
    
    # ============================================================================
    # STEP 3: PCA ANALYSIS
    # ============================================================================
    print("\n" + "="*70)
    print("STEP 3: Principal Component Analysis")
    print("="*70)
    
    analyzer = PCAAnalyzer(n_components=None, variance_threshold=0.95)
    scores = analyzer.fit_transform(processed_data)
    
    print(f"\nPCA scores shape: {scores.shape}")
    print(f"Components computed: {len(scores.columns)}")
    
    # Variance explained
    variance_df = analyzer.get_explained_variance()
    print(f"\nVariance Explained:")
    for i in range(min(5, len(variance_df))):
        row = variance_df.iloc[i]
        print(f"  PC{i+1}: {row['Variance_Explained']*100:.2f}% "
              f"(cumulative: {row['Cumulative_Variance']*100:.2f}%)")
    
    # Save variance data
    variance_df.to_csv(output_dir / 'explained_variance.csv', index=False)
    print(f"\n✓ Saved variance data to {output_dir / 'explained_variance.csv'}")
    
    # ============================================================================
    # STEP 4: FEATURE ANALYSIS
    # ============================================================================
    print("\n" + "="*70)
    print("STEP 4: Feature Importance Analysis")
    print("="*70)
    
    # Top features for PC1
    print("\nTop 10 Contributing Features for PC1:")
    top_features_pc1 = analyzer.get_top_contributing_features(component=1, n_features=10)
    for i, row in top_features_pc1.iterrows():
        print(f"  {i+1}. {row['Feature']}: {row['Loading']:.4f}")
    
    top_features_pc1.to_csv(output_dir / 'top_features_PC1.csv', index=False)
    
    # Overall feature importance
    print("\nOverall Feature Importance (across top 3 PCs):")
    importance_df = analyzer.analyze_feature_importance(n_components=3)
    print(importance_df.head(10).to_string(index=False))
    
    importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)
    print(f"\n✓ Saved feature importance to {output_dir / 'feature_importance.csv'}")
    
    # ============================================================================
    # STEP 5: SAMPLE ANALYSIS
    # ============================================================================
    print("\n" + "="*70)
    print("STEP 5: Sample Analysis")
    print("="*70)
    
    # Top contributing samples for PC1
    print("\nTop 5 Samples with Highest PC1 Scores:")
    top_samples = analyzer.get_sample_contributions(component=1, n_samples=5)
    print(top_samples.to_string(index=False))
    
    # Outlier detection
    print("\nOutlier Detection:")
    outlier_info, outlier_mask = analyzer.detect_outliers(
        n_components=2,
        threshold=3.0,
        method='mahalanobis'
    )
    
    n_outliers = outlier_mask.sum()
    print(f"  Outliers detected: {n_outliers} / {len(data)} samples")
    
    if n_outliers > 0:
        print("\nOutlier samples:")
        outliers = outlier_info[outlier_info['Is_Outlier']]
        print(outliers[['Sample', 'Z_Score']].to_string(index=False))
    
    outlier_info.to_csv(output_dir / 'outlier_detection.csv', index=False)
    
    # Reconstruction error
    print("\nReconstruction Quality:")
    recon_error = analyzer.get_reconstruction_error(processed_data)
    print(f"  Mean reconstruction error: {recon_error['Reconstruction_Error'].mean():.4f}")
    print(f"  Max reconstruction error: {recon_error['Reconstruction_Error'].max():.4f}")
    
    # ============================================================================
    # STEP 6: VISUALIZATION
    # ============================================================================
    print("\n" + "="*70)
    print("STEP 6: Creating Visualizations")
    print("="*70)
    
    plot_dir = output_dir / 'plots'
    plotter = Plotter(output_dir=plot_dir)
    
    print("\nGenerating plots...")
    
    # Scree plot
    print("  - Scree plot")
    plotter.plot_scree(variance_df, title="Scree Plot - Variance Explained")
    
    # Score plots
    print("  - Score plots (2D)")
    variance_dict_12 = {
        'PC1': variance_df.iloc[0]['Variance_Explained'],
        'PC2': variance_df.iloc[1]['Variance_Explained']
    }
    plotter.plot_scores_2d(
        scores,
        pc_x=1,
        pc_y=2,
        variance_explained=variance_dict_12,
        title="PCA Score Plot: PC1 vs PC2"
    )
    
    if len(scores.columns) >= 3:
        variance_dict_13 = {
            'PC1': variance_df.iloc[0]['Variance_Explained'],
            'PC3': variance_df.iloc[2]['Variance_Explained']
        }
        plotter.plot_scores_2d(
            scores,
            pc_x=1,
            pc_y=3,
            variance_explained=variance_dict_13,
            title="PCA Score Plot: PC1 vs PC3"
        )
    
    # 3D score plot
    if len(scores.columns) >= 3:
        print("  - Score plot (3D)")
        plotter.plot_scores_3d(scores, pc_x=1, pc_y=2, pc_z=3)
    
    # Loading plots
    print("  - Loading plots")
    for i in range(1, min(4, len(variance_df) + 1)):
        top_feat = analyzer.get_top_contributing_features(component=i, n_features=15)
        plotter.plot_loadings(top_feat, component=i, n_features=15)
    
    # Biplot
    print("  - Biplot")
    biplot_data = analyzer.get_biplot_data(pc_x=1, pc_y=2, n_vectors=10)
    plotter.plot_biplot(biplot_data, pc_x=1, pc_y=2)
    
    # Outlier plot
    print("  - Outlier detection plot")
    plotter.plot_outliers(scores, outlier_info, pc_x=1, pc_y=2)
    
    # Feature importance
    print("  - Feature importance plot")
    plotter.plot_feature_importance(importance_df, n_features=20)
    
    # Summary report
    print("  - Summary report")
    plotter.create_summary_report(variance_df, scores, top_features_pc1)
    
    print(f"\n✓ All plots saved to {plot_dir}")
    
    # ============================================================================
    # STEP 7: SAVE RESULTS
    # ============================================================================
    print("\n" + "="*70)
    print("STEP 7: Saving Results")
    print("="*70)
    
    # Save scores and loadings
    scores.to_csv(output_dir / 'pca_scores.csv')
    analyzer.loadings.to_csv(output_dir / 'pca_loadings.csv')
    
    print(f"\n✓ Results saved to: {output_dir}")
    print(f"  - PCA scores: pca_scores.csv")
    print(f"  - PCA loadings: pca_loadings.csv")
    print(f"  - Variance explained: explained_variance.csv")
    print(f"  - Feature importance: feature_importance.csv")
    print(f"  - Outlier detection: outlier_detection.csv")
    print(f"  - Plots: plots/")
    
    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "="*70)
    print("DEMO COMPLETE - ANALYSIS SUMMARY")
    print("="*70)
    
    print(f"\nDataset:")
    print(f"  - {summary['n_samples']} samples × {summary['n_features']} features")
    print(f"  - {preproc_summary['n_removed_features']} features removed during preprocessing")
    
    print(f"\nPCA Results:")
    print(f"  - {len(scores.columns)} principal components")
    print(f"  - {variance_df['Cumulative_Variance'].iloc[-1]*100:.2f}% variance explained")
    print(f"  - {n_outliers} outliers detected")
    
    print(f"\nTop 3 Most Important Features:")
    for i in range(3):
        row = importance_df.iloc[i]
        print(f"  {i+1}. {row['Feature']}: {row['Importance_Score']:.4f}")
    
    print("\n" + "="*70)
    print("Thank you for using SpectraLab!")
    print("="*70)


if __name__ == '__main__':
    main()