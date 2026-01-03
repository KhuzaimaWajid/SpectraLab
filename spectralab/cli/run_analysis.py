"""
Command-line interface for SpectraLab.

Provides reproducible batch analysis with config-driven runs.
"""

import argparse
import yaml
import json
from pathlib import Path
import sys
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from spectralab import DataLoader, DataPreprocessor, PCAAnalyzer, Plotter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        else:
            config = json.load(f)
    return config


def run_analysis(config: dict):
    """
    Run complete PCA analysis pipeline from config.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary with keys:
        - data: data loading config
        - preprocessing: preprocessing config
        - pca: PCA config
        - visualization: visualization config
        - output: output directory config
    """
    logger.info("="*70)
    logger.info("SPECTRALAB - PCA ANALYSIS PIPELINE")
    logger.info("="*70)
    
    # Setup output directory
    output_dir = Path(config.get('output', {}).get('directory', './results'))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # 1. LOAD DATA
    logger.info("\n" + "="*70)
    logger.info("STEP 1: DATA LOADING")
    logger.info("="*70)
    
    loader = DataLoader()
    data_config = config.get('data', {})
    
    if data_config.get('type') == 'synthetic':
        data = loader.generate_synthetic_data(
            n_samples=data_config.get('n_samples', 100),
            n_features=data_config.get('n_features', 50),
            n_informative=data_config.get('n_informative', 10),
            noise_level=data_config.get('noise_level', 0.1),
            random_state=data_config.get('random_state', 42)
        )
    else:
        data = loader.load_csv(
            filepath=data_config['filepath'],
            sample_col=data_config.get('sample_col'),
            sep=data_config.get('sep', ',')
        )
    
    # Save data summary
    summary = loader.get_summary()
    with open(output_dir / 'data_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Validate data
    is_valid, issues = loader.validate_data()
    if not is_valid:
        logger.error("Data validation failed. Please fix issues before proceeding.")
        return
    
    # 2. PREPROCESSING
    logger.info("\n" + "="*70)
    logger.info("STEP 2: PREPROCESSING")
    logger.info("="*70)
    
    preprocessor = DataPreprocessor()
    preproc_config = config.get('preprocessing', {})
    
    processed_data = preprocessor.preprocess_pipeline(
        data,
        missing_strategy=preproc_config.get('missing_strategy', 'mean'),
        scaling=preproc_config.get('scaling', 'standard'),
        log_transform=preproc_config.get('log_transform', False),
        log_offset=preproc_config.get('log_offset', 1.0),
        remove_low_variance=preproc_config.get('remove_low_variance', True),
        variance_threshold=preproc_config.get('variance_threshold', 0.01)
    )
    
    # Save preprocessing summary
    preproc_summary = preprocessor.get_preprocessing_summary()
    with open(output_dir / 'preprocessing_summary.json', 'w') as f:
        json.dump(preproc_summary, f, indent=2)
    
    # Save preprocessed data
    processed_data.to_csv(output_dir / 'preprocessed_data.csv')
    logger.info(f"Saved preprocessed data: {output_dir / 'preprocessed_data.csv'}")
    
    # 3. PCA ANALYSIS
    logger.info("\n" + "="*70)
    logger.info("STEP 3: PCA ANALYSIS")
    logger.info("="*70)
    
    pca_config = config.get('pca', {})
    
    analyzer = PCAAnalyzer(
        n_components=pca_config.get('n_components'),
        variance_threshold=pca_config.get('variance_threshold', 0.95)
    )
    
    # Fit and transform
    scores = analyzer.fit_transform(processed_data)
    
    # Save results
    scores.to_csv(output_dir / 'pca_scores.csv')
    analyzer.loadings.to_csv(output_dir / 'pca_loadings.csv')
    logger.info(f"Saved PCA scores and loadings")
    
    # Get variance explained
    variance_df = analyzer.get_explained_variance()
    variance_df.to_csv(output_dir / 'explained_variance.csv', index=False)
    
    # Get top features for each component
    n_top_features = pca_config.get('n_top_features', 10)
    for i in range(1, min(4, len(variance_df) + 1)):
        top_features = analyzer.get_top_contributing_features(
            component=i,
            n_features=n_top_features
        )
        top_features.to_csv(output_dir / f'top_features_PC{i}.csv', index=False)
    
    # Feature importance analysis
    importance_df = analyzer.analyze_feature_importance(n_components=3)
    importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)
    
    # Outlier detection
    if pca_config.get('detect_outliers', True):
        outlier_info, outlier_mask = analyzer.detect_outliers(
            n_components=pca_config.get('outlier_n_components', 2),
            threshold=pca_config.get('outlier_threshold', 3.0)
        )
        outlier_info.to_csv(output_dir / 'outlier_detection.csv', index=False)
    
    # Reconstruction error
    recon_error = analyzer.get_reconstruction_error(processed_data)
    recon_error.to_csv(output_dir / 'reconstruction_error.csv', index=False)
    
    # 4. VISUALIZATION
    logger.info("\n" + "="*70)
    logger.info("STEP 4: VISUALIZATION")
    logger.info("="*70)
    
    viz_config = config.get('visualization', {})
    plot_dir = output_dir / 'plots'
    
    plotter = Plotter(output_dir=plot_dir)
    
    # Scree plot
    if viz_config.get('scree_plot', True):
        plotter.plot_scree(variance_df)
    
    # Score plots
    if viz_config.get('score_plots', True):
        # 2D score plots
        for pc_pair in viz_config.get('score_plot_pairs', [[1, 2], [1, 3], [2, 3]]):
            if len(scores.columns) >= max(pc_pair):
                variance_dict = {
                    f"PC{pc_pair[0]}": variance_df.iloc[pc_pair[0]-1]['Variance_Explained'],
                    f"PC{pc_pair[1]}": variance_df.iloc[pc_pair[1]-1]['Variance_Explained']
                }
                plotter.plot_scores_2d(
                    scores,
                    pc_x=pc_pair[0],
                    pc_y=pc_pair[1],
                    variance_explained=variance_dict
                )
        
        # 3D score plot
        if viz_config.get('3d_plot', False) and len(scores.columns) >= 3:
            plotter.plot_scores_3d(scores, pc_x=1, pc_y=2, pc_z=3)
    
    # Loading plots
    if viz_config.get('loading_plots', True):
        for i in range(1, min(4, len(variance_df) + 1)):
            top_features = analyzer.get_top_contributing_features(component=i, n_features=15)
            plotter.plot_loadings(top_features, component=i)
    
    # Biplot
    if viz_config.get('biplot', True) and len(scores.columns) >= 2:
        biplot_data = analyzer.get_biplot_data(pc_x=1, pc_y=2, n_vectors=10)
        plotter.plot_biplot(biplot_data, pc_x=1, pc_y=2)
    
    # Outlier plot
    if viz_config.get('outlier_plot', True) and pca_config.get('detect_outliers', True):
        plotter.plot_outliers(scores, outlier_info, pc_x=1, pc_y=2)
    
    # Feature importance plot
    if viz_config.get('feature_importance_plot', True):
        plotter.plot_feature_importance(importance_df, n_features=20)
    
    # Summary report
    if viz_config.get('summary_report', True):
        top_features_pc1 = analyzer.get_top_contributing_features(component=1, n_features=10)
        plotter.create_summary_report(variance_df, scores, top_features_pc1)
    
    # 5. FINAL SUMMARY
    logger.info("\n" + "="*70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*70)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"  - Data summary: data_summary.json")
    logger.info(f"  - Preprocessed data: preprocessed_data.csv")
    logger.info(f"  - PCA scores: pca_scores.csv")
    logger.info(f"  - PCA loadings: pca_loadings.csv")
    logger.info(f"  - Plots: plots/")
    logger.info("="*70)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='SpectraLab: Data Science & PCA Analysis Platform',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with config file
  python -m spectralab.cli.run_analysis --config config.yaml
  
  # Generate example config
  python -m spectralab.cli.run_analysis --generate-config
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (YAML or JSON)'
    )
    
    parser.add_argument(
        '--generate-config',
        action='store_true',
        help='Generate example configuration file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory (overrides config)'
    )
    
    args = parser.parse_args()
    
    if args.generate_config:
        # Generate example config
        example_config = {
            'data': {
                'type': 'synthetic',  # or 'csv'
                'n_samples': 100,
                'n_features': 50,
                'n_informative': 10,
                'noise_level': 0.1,
                'random_state': 42,
                # For CSV:
                # 'filepath': 'data/my_data.csv',
                # 'sample_col': 'Sample_ID',
                # 'sep': ','
            },
            'preprocessing': {
                'missing_strategy': 'mean',  # 'drop', 'mean', 'median', 'zero'
                'scaling': 'standard',  # 'standard', 'minmax', 'none'
                'log_transform': False,
                'log_offset': 1.0,
                'remove_low_variance': True,
                'variance_threshold': 0.01
            },
            'pca': {
                'n_components': None,  # None = auto-select based on variance_threshold
                'variance_threshold': 0.95,
                'n_top_features': 10,
                'detect_outliers': True,
                'outlier_n_components': 2,
                'outlier_threshold': 3.0
            },
            'visualization': {
                'scree_plot': True,
                'score_plots': True,
                'score_plot_pairs': [[1, 2], [1, 3], [2, 3]],
                '3d_plot': False,
                'loading_plots': True,
                'biplot': True,
                'outlier_plot': True,
                'feature_importance_plot': True,
                'summary_report': True
            },
            'output': {
                'directory': './results'
            }
        }
        
        config_path = Path('example_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(example_config, f, default_flow_style=False, sort_keys=False)
        
        print(f"Generated example configuration: {config_path}")
        print("\nEdit this file and run:")
        print(f"  python -m spectralab.cli.run_analysis --config {config_path}")
        return
    
    if not args.config:
        parser.print_help()
        return
    
    # Load and run config
    config = load_config(args.config)
    
    # Override output directory if specified
    if args.output:
        config['output']['directory'] = args.output
    
    run_analysis(config)


if __name__ == '__main__':
    main()