"""
SpectraLab Streamlit Dashboard

Interactive web interface for exploratory PCA analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import io

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spectralab import DataLoader, DataPreprocessor, PCAAnalyzer, Plotter

# Page config
st.set_page_config(
    page_title="SpectraLab - PCA Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'pca_fitted' not in st.session_state:
    st.session_state.pca_fitted = False


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<div class="main-header">🔬 SpectraLab</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #7f8c8d;">Data Science & PCA Analysis Platform</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Select Module",
            ["Data Loading", "Preprocessing", "PCA Analysis", "Visualization", "Export Results"]
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info(
            "SpectraLab provides comprehensive PCA analysis for "
            "high-dimensional datasets with focus on interpretability "
            "and statistical insight."
        )
    
    # Page routing
    if page == "Data Loading":
        data_loading_page()
    elif page == "Preprocessing":
        preprocessing_page()
    elif page == "PCA Analysis":
        pca_analysis_page()
    elif page == "Visualization":
        visualization_page()
    elif page == "Export Results":
        export_page()


def data_loading_page():
    """Data loading interface."""
    st.markdown('<div class="sub-header">📊 Data Loading</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Upload CSV", "Generate Synthetic Data"])
    
    with tab1:
        st.markdown("### Upload Your Dataset")
        st.markdown("Upload a CSV file with samples as rows and features as columns.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        col1, col2 = st.columns(2)
        with col1:
            sep = st.selectbox("Separator", [',', '\t', ';', '|'], index=0)
        with col2:
            sample_col = st.text_input("Sample ID Column (optional)", "")
        
        if uploaded_file is not None:
            try:
                loader = DataLoader()
                data = loader.load_csv(
                    io.StringIO(uploaded_file.getvalue().decode('utf-8')),
                    sample_col=sample_col if sample_col else None,
                    sep=sep
                )
                
                st.session_state.raw_data = data
                st.session_state.data_loaded = True
                st.session_state.loader = loader
                
                st.success(f"✅ Loaded {len(data)} samples × {len(data.columns)} features")
                
                # Display data preview
                st.markdown("#### Data Preview")
                st.dataframe(data.head(10), use_container_width=True)
                
                # Data summary
                summary = loader.get_summary()
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Samples", summary['n_samples'])
                with col2:
                    st.metric("Features", summary['n_features'])
                with col3:
                    st.metric("Missing Values", summary['missing_values'])
                with col4:
                    st.metric("Missing %", f"{summary['missing_percentage']:.2f}%")
                
                # Validation
                is_valid, issues = loader.validate_data()
                if not is_valid:
                    st.warning("⚠️ Data Validation Issues:")
                    for issue in issues:
                        st.write(f"  - {issue}")
                else:
                    st.success("✅ Data validation passed")
                    
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    
    with tab2:
        st.markdown("### Generate Synthetic Dataset")
        st.markdown("Create a synthetic dataset for testing and demonstration.")
        
        col1, col2 = st.columns(2)
        with col1:
            n_samples = st.number_input("Number of Samples", min_value=10, max_value=1000, value=100)
            n_features = st.number_input("Number of Features", min_value=5, max_value=500, value=50)
        with col2:
            n_informative = st.number_input("Informative Features", min_value=1, max_value=n_features, value=10)
            noise_level = st.slider("Noise Level", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        
        random_state = st.number_input("Random State", min_value=0, max_value=9999, value=42)
        
        if st.button("Generate Dataset", type="primary"):
            loader = DataLoader()
            data = loader.generate_synthetic_data(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_informative,
                noise_level=noise_level,
                random_state=random_state
            )
            
            st.session_state.raw_data = data
            st.session_state.data_loaded = True
            st.session_state.loader = loader
            
            st.success(f"✅ Generated {n_samples} samples × {n_features} features")
            
            # Display preview
            st.markdown("#### Data Preview")
            st.dataframe(data.head(10), use_container_width=True)
            
            # Summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Samples", n_samples)
            with col2:
                st.metric("Features", n_features)
            with col3:
                st.metric("Informative", n_informative)


def preprocessing_page():
    """Preprocessing interface."""
    st.markdown('<div class="sub-header">⚙️ Data Preprocessing</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first in the Data Loading page.")
        return
    
    data = st.session_state.raw_data
    
    st.markdown("### Preprocessing Configuration")
    st.markdown("Configure preprocessing steps to prepare data for PCA analysis.")
    
    # Preprocessing options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Missing Values")
        missing_strategy = st.selectbox(
            "Strategy",
            ['mean', 'median', 'zero', 'drop'],
            help="How to handle missing values"
        )
        
        st.markdown("#### Scaling")
        scaling = st.selectbox(
            "Method",
            ['standard', 'minmax', 'none'],
            help="Feature scaling method"
        )
    
    with col2:
        st.markdown("#### Transformations")
        log_transform = st.checkbox("Apply Log Transform", help="Useful for skewed intensity data")
        if log_transform:
            log_offset = st.number_input("Log Offset", min_value=0.0, max_value=10.0, value=1.0)
        else:
            log_offset = 1.0
        
        st.markdown("#### Feature Filtering")
        remove_low_variance = st.checkbox("Remove Low Variance Features", value=True)
        if remove_low_variance:
            variance_threshold = st.number_input("Variance Threshold", min_value=0.0, max_value=1.0, value=0.01, step=0.01)
        else:
            variance_threshold = 0.01
    
    if st.button("Run Preprocessing", type="primary"):
        with st.spinner("Preprocessing data..."):
            preprocessor = DataPreprocessor()
            
            processed_data = preprocessor.preprocess_pipeline(
                data,
                missing_strategy=missing_strategy,
                scaling=scaling,
                log_transform=log_transform,
                log_offset=log_offset,
                remove_low_variance=remove_low_variance,
                variance_threshold=variance_threshold
            )
            
            st.session_state.processed_data = processed_data
            st.session_state.preprocessor = preprocessor
            
            st.success("✅ Preprocessing complete!")
            
            # Show summary
            summary = preprocessor.get_preprocessing_summary()
            
            st.markdown("#### Preprocessing Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Original Features", len(data.columns))
                st.metric("Processed Features", len(processed_data.columns))
            with col2:
                st.metric("Removed Features", summary['n_removed_features'])
                st.metric("Scaling Method", summary['scaling_method'] or 'None')
            
            if summary['removed_features']:
                with st.expander("View Removed Features"):
                    st.write(summary['removed_features'])
            
            # Show data comparison
            st.markdown("#### Data Comparison")
            tab1, tab2 = st.tabs(["Original", "Processed"])
            with tab1:
                st.dataframe(data.head(10), use_container_width=True)
            with tab2:
                st.dataframe(processed_data.head(10), use_container_width=True)


def pca_analysis_page():
    """PCA analysis interface."""
    st.markdown('<div class="sub-header">🔍 PCA Analysis</div>', unsafe_allow_html=True)
    
    if not hasattr(st.session_state, 'processed_data'):
        st.warning("⚠️ Please preprocess data first in the Preprocessing page.")
        return
    
    processed_data = st.session_state.processed_data
    
    st.markdown("### PCA Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_components = st.number_input(
            "Number of Components",
            min_value=1,
            max_value=min(len(processed_data), len(processed_data.columns)),
            value=None,
            help="Leave empty for automatic selection"
        )
        
        variance_threshold = st.slider(
            "Variance Threshold",
            min_value=0.5,
            max_value=1.0,
            value=0.95,
            step=0.05,
            help="Cumulative variance to preserve (if n_components not specified)"
        )
    
    with col2:
        detect_outliers = st.checkbox("Detect Outliers", value=True)
        if detect_outliers:
            outlier_threshold = st.slider("Outlier Threshold (std)", min_value=2.0, max_value=5.0, value=3.0, step=0.5)
    
    if st.button("Run PCA", type="primary"):
        with st.spinner("Performing PCA analysis..."):
            analyzer = PCAAnalyzer(
                n_components=n_components,
                variance_threshold=variance_threshold
            )
            
            scores = analyzer.fit_transform(processed_data)
            
            st.session_state.analyzer = analyzer
            st.session_state.scores = scores
            st.session_state.pca_fitted = True
            
            st.success("✅ PCA analysis complete!")
            
            # Variance explained
            variance_df = analyzer.get_explained_variance()
            
            st.markdown("#### Variance Explained")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Components", len(variance_df))
            with col2:
                st.metric("PC1 Variance", f"{variance_df.iloc[0]['Variance_Explained']*100:.2f}%")
            with col3:
                st.metric("Total Variance", f"{variance_df['Cumulative_Variance'].iloc[-1]*100:.2f}%")
            
            st.dataframe(variance_df, use_container_width=True)
            
            # Top features
            st.markdown("#### Top Contributing Features (PC1)")
            top_features = analyzer.get_top_contributing_features(component=1, n_features=10)
            st.dataframe(top_features, use_container_width=True)
            
            # Feature importance
            st.markdown("#### Overall Feature Importance")
            importance_df = analyzer.analyze_feature_importance(n_components=3)
            st.dataframe(importance_df.head(15), use_container_width=True)
            
            # Outlier detection
            if detect_outliers:
                st.markdown("#### Outlier Detection")
                outlier_info, outlier_mask = analyzer.detect_outliers(threshold=outlier_threshold)
                n_outliers = outlier_mask.sum()
                
                st.metric("Outliers Detected", n_outliers)
                
                if n_outliers > 0:
                    outliers = outlier_info[outlier_info['Is_Outlier']]
                    st.dataframe(outliers, use_container_width=True)


def visualization_page():
    """Visualization interface."""
    st.markdown('<div class="sub-header">📈 Visualization</div>', unsafe_allow_html=True)
    
    if not st.session_state.pca_fitted:
        st.warning("⚠️ Please run PCA analysis first.")
        return
    
    analyzer = st.session_state.analyzer
    scores = st.session_state.scores
    
    # Plot selection
    plot_type = st.selectbox(
        "Select Plot Type",
        ["Scree Plot", "Score Plot (2D)", "Score Plot (3D)", "Loading Plot", "Biplot", "Feature Importance"]
    )
    
    import matplotlib.pyplot as plt
    
    if plot_type == "Scree Plot":
        st.markdown("### Scree Plot")
        variance_df = analyzer.get_explained_variance()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        components = range(1, len(variance_df) + 1)
        
        ax1.bar(components, variance_df['Variance_Explained'] * 100, color='steelblue', edgecolor='black')
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Variance Explained (%)')
        ax1.set_title('Variance per Component')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(components, variance_df['Cumulative_Variance'] * 100, marker='o', linewidth=2, color='steelblue')
        ax2.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='95%')
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Cumulative Variance (%)')
        ax2.set_title('Cumulative Variance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
        st.pyplot(fig)
    
    elif plot_type == "Score Plot (2D)":
        st.markdown("### 2D Score Plot")
        
        col1, col2 = st.columns(2)
        with col1:
            pc_x = st.selectbox("X-axis", range(1, len(scores.columns) + 1), index=0)
        with col2:
            pc_y = st.selectbox("Y-axis", range(1, len(scores.columns) + 1), index=1)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        x = scores[f'PC{pc_x}']
        y = scores[f'PC{pc_y}']
        
        ax.scatter(x, y, s=80, alpha=0.6, color='steelblue', edgecolors='black', linewidth=0.5)
        ax.set_xlabel(f'PC{pc_x}')
        ax.set_ylabel(f'PC{pc_y}')
        ax.set_title(f'Score Plot: PC{pc_x} vs PC{pc_y}')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.5)
        
        fig.tight_layout()
        st.pyplot(fig)
    
    elif plot_type == "Loading Plot":
        st.markdown("### Loading Plot")
        
        component = st.selectbox("Select Component", range(1, len(scores.columns) + 1))
        n_features = st.slider("Number of Features", min_value=5, max_value=30, value=15)
        
        top_features = analyzer.get_top_contributing_features(component=component, n_features=n_features)
        data = top_features.sort_values('Loading')
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['steelblue' if x >= 0 else 'coral' for x in data['Loading']]
        ax.barh(range(len(data)), data['Loading'], color=colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data['Feature'], fontsize=9)
        ax.set_xlabel('Loading Value')
        ax.set_title(f'Top {n_features} Feature Loadings - PC{component}')
        ax.axvline(x=0, color='black', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')
        
        fig.tight_layout()
        st.pyplot(fig)
    
    elif plot_type == "Feature Importance":
        st.markdown("### Feature Importance")
        
        n_features = st.slider("Number of Features", min_value=10, max_value=50, value=20)
        
        importance_df = analyzer.analyze_feature_importance(n_components=3)
        data = importance_df.head(n_features)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.barh(range(len(data)), data['Importance_Score'], color='steelblue', edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data['Feature'], fontsize=9)
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance (Across Top 3 PCs)')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        fig.tight_layout()
        st.pyplot(fig)


def export_page():
    """Export results interface."""
    st.markdown('<div class="sub-header">💾 Export Results</div>', unsafe_allow_html=True)
    
    if not st.session_state.pca_fitted:
        st.warning("⚠️ No results to export. Please run PCA analysis first.")
        return
    
    st.markdown("### Download Analysis Results")
    
    analyzer = st.session_state.analyzer
    scores = st.session_state.scores
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### PCA Scores")
        csv = scores.to_csv().encode('utf-8')
        st.download_button(
            label="Download Scores (CSV)",
            data=csv,
            file_name="pca_scores.csv",
            mime="text/csv"
        )
        
        st.markdown("#### Variance Explained")
        variance_df = analyzer.get_explained_variance()
        csv = variance_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Variance (CSV)",
            data=csv,
            file_name="explained_variance.csv",
            mime="text/csv"
        )
    
    with col2:
        st.markdown("#### PCA Loadings")
        csv = analyzer.loadings.to_csv().encode('utf-8')
        st.download_button(
            label="Download Loadings (CSV)",
            data=csv,
            file_name="pca_loadings.csv",
            mime="text/csv"
        )
        
        st.markdown("#### Feature Importance")
        importance_df = analyzer.analyze_feature_importance(n_components=3)
        csv = importance_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Importance (CSV)",
            data=csv,
            file_name="feature_importance.csv",
            mime="text/csv"
        )
    
    st.success("✅ Ready to download results!")


if __name__ == '__main__':
    main()