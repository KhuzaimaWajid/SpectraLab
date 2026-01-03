# 🔬 SpectraLab

**A Production-Grade Data Science Platform for PCA Analysis of High-Dimensional Datasets**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

> Transform complex, high-dimensional scientific data into actionable insights through Principal Component Analysis. Built for researchers, data scientists, and analysts working with mass spectrometry, genomics, spectroscopy, and metabolomics data.

---

## 🎯 What is SpectraLab?

SpectraLab is a comprehensive data science platform that makes Principal Component Analysis (PCA) accessible and powerful. Whether you're analyzing mass spectrometry peaks, gene expression profiles, or spectroscopic measurements, SpectraLab provides the tools to:

- 📊 **Reduce Dimensionality** - Transform hundreds of features into interpretable principal components
- 🔍 **Discover Patterns** - Identify sample clustering and relationships in your data
- 📈 **Rank Features** - Find which measurements drive variance in your experiments
- 🚨 **Detect Outliers** - Identify anomalous samples using statistical methods
- 🎨 **Visualize Results** - Generate publication-quality plots automatically

---

## ✨ Key Features

### 🎛️ **Three User Interfaces**
- **Python API** - For programmatic analysis and integration
- **Command-Line Interface (CLI)** - For batch processing and automation
- **Streamlit Web Dashboard** - Interactive, no-code interface for exploration

### 🧪 **Complete Analysis Pipeline**
- **Data Loading** - CSV import with validation and synthetic data generation
- **Preprocessing** - Missing value handling, scaling, transformations, filtering
- **PCA Analysis** - Automatic component selection, variance analysis, feature importance
- **Visualization** - 10+ plot types including scree plots, biplots, 3D views
- **Outlier Detection** - Mahalanobis distance and z-score methods

### 📊 **Professional Outputs**
- Publication-ready plots (300 DPI, customizable styles)
- Comprehensive CSV exports (scores, loadings, statistics)
- Reproducible analysis with configuration files
- Detailed logging of all operations

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/spectralab.git
cd spectralab

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### 30-Second Demo

```bash
# Run the comprehensive demo
python demo.py
```

This generates synthetic data, runs the full analysis pipeline, and creates 11 visualizations in `demo_results/`.

---

## 📖 Usage Examples

### 🐍 Python API

```python
from spectralab import DataLoader, DataPreprocessor, PCAAnalyzer, Plotter

# 1. Load your data
loader = DataLoader()
data = loader.load_csv('your_data.csv', sample_col='Sample_ID')

# 2. Preprocess
preprocessor = DataPreprocessor()
clean_data = preprocessor.preprocess_pipeline(
    data,
    missing_strategy='mean',
    scaling='standard',
    remove_low_variance=True
)

# 3. Run PCA
analyzer = PCAAnalyzer(variance_threshold=0.95)
scores = analyzer.fit_transform(clean_data)

# 4. Analyze results
variance_df = analyzer.get_explained_variance()
top_features = analyzer.get_top_contributing_features(component=1, n_features=10)
outliers, mask = analyzer.detect_outliers(threshold=3.0)

# 5. Visualize
plotter = Plotter(output_dir='./results')
plotter.plot_scree(variance_df)
plotter.plot_scores_2d(scores, pc_x=1, pc_y=2)
plotter.plot_loadings(analyzer.loadings, component=1)
```

### 💻 Command-Line Interface

```bash
# Generate example configuration
python -m spectralab.cli.run_analysis --generate-config

# Edit config.yaml with your settings, then run
python -m spectralab.cli.run_analysis --config config.yaml
```

**Example config.yaml:**
```yaml
data:
  type: 'csv'
  filepath: 'data.csv'
  sample_col: 'Sample_ID'

preprocessing:
  missing_strategy: 'mean'
  scaling: 'standard'
  remove_low_variance: true

pca:
  variance_threshold: 0.95
  detect_outliers: true

visualization:
  scree_plot: true
  score_plots_2d: [[1, 2], [1, 3]]
  biplot: [1, 2]
  
output:
  directory: './results'
```

### 🌐 Interactive Web Dashboard

```bash
# Launch the Streamlit app
streamlit run app.py
```

**Features:**
- 📤 Upload CSV files or generate synthetic data
- ⚙️ Configure preprocessing with dropdowns and sliders
- 🔬 Run PCA with one click
- 📊 View interactive plots in your browser
- 💾 Download all results as CSV files

The browser opens automatically at `http://localhost:8501`

---

## 📊 Output Files

### CSV Files
- `pca_scores.csv` - Sample coordinates in PC space
- `pca_loadings.csv` - Feature contributions to each PC
- `explained_variance.csv` - Variance statistics per component
- `feature_importance.csv` - Overall feature rankings
- `top_features_PC{n}.csv` - Top features for each component
- `outlier_detection.csv` - Outlier analysis results
- `preprocessing_summary.json` - Log of preprocessing steps

### Visualizations (PNG, 300 DPI)
- **Scree Plot** - Variance explained per component
- **Score Plots (2D/3D)** - Sample clustering visualization
- **Loading Plots** - Feature importance bars
- **Biplots** - Combined sample + feature vectors
- **Outlier Detection** - Samples with anomalous patterns
- **Summary Report** - Comprehensive 4-panel overview

---

## 🎓 Use Cases

### 🧬 **Genomics & Transcriptomics**
- Identify gene expression patterns across conditions
- Cluster samples by molecular profile
- Discover biomarkers driving experimental variance

### 🔬 **Mass Spectrometry**
- Reduce thousands of m/z peaks to interpretable components
- Quality control and batch effect detection
- Feature selection for metabolomics

### 🌈 **Spectroscopy**
- Identify key wavelengths in spectral data
- Sample classification (NIR, Raman, FTIR)
- Dimensionality reduction for ML pipelines

### 💊 **Drug Discovery**
- Analyze high-throughput screening data
- Identify compound clusters by activity profile
- Feature engineering for QSAR models

---

## 🏗️ Project Structure

```
spectralab/
├── spectralab/               # Main package
│   ├── data/
│   │   ├── loader.py        # Data loading & validation
│   │   └── preprocessor.py  # Preprocessing pipeline
│   ├── analysis/
│   │   └── pca_analyzer.py  # PCA implementation
│   ├── visualization/
│   │   └── plotter.py       # Plot generation
│   └── cli/
│       └── run_analysis.py  # Command-line interface
│
├── tests/                    # Unit tests
├── demo.py                   # Demonstration script
├── app.py                    # Streamlit dashboard
├── setup.py                  # Package configuration
├── requirements.txt          # Dependencies
└── config_example.yaml       # Example configuration

```

---

## 🔧 Requirements

- Python 3.8+
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- pyyaml >= 5.4.0
- streamlit >= 1.20.0 (for web dashboard)

---

## 📚 Documentation

- **[Quick Start Guide](QUICKSTART.md)** - Get running in 5 minutes
- **[Project Structure](PROJECT_STRUCTURE.md)** - Architecture deep-dive
- **[Resume Guide](RESUME_GUIDE.md)** - Interview preparation

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=spectralab tests/

# Run specific test file
pytest tests/test_loader.py
```

---

## 🎨 Example Workflow: Mass Spectrometry Analysis

```python
from spectralab import DataLoader, DataPreprocessor, PCAAnalyzer, Plotter

# Load mass spec peak intensities (samples × m/z values)
loader = DataLoader()
data = loader.load_csv('mass_spec_peaks.csv', sample_col='Sample_ID')
print(f"Loaded {data.shape[0]} samples with {data.shape[1]} m/z peaks")

# Preprocess: log-transform intensities, scale, remove low-variance peaks
preprocessor = DataPreprocessor()
clean_data = preprocessor.preprocess_pipeline(
    data,
    log_transform=True,       # Handle exponential intensities
    scaling='standard',        # Z-score normalization
    remove_low_variance=True,  # Remove uninformative peaks
    variance_threshold=0.01
)

# PCA: auto-select components for 95% variance
analyzer = PCAAnalyzer(variance_threshold=0.95)
scores = analyzer.fit_transform(clean_data)
print(f"Selected {len(scores.columns)} components")

# Find most important m/z peaks
importance = analyzer.analyze_feature_importance(n_components=3)
print("Top 10 discriminating peaks:")
print(importance.head(10))

# Detect outlier samples
outliers, mask = analyzer.detect_outliers(method='mahalanobis', threshold=3.0)
print(f"Found {mask.sum()} outlier samples")

# Generate all visualizations
plotter = Plotter(output_dir='./mass_spec_results')
variance_df = analyzer.get_explained_variance()

plotter.plot_scree(variance_df)
plotter.plot_scores_2d(scores, pc_x=1, pc_y=2)
plotter.plot_biplot(
    analyzer.get_biplot_data(pc_x=1, pc_y=2, n_vectors=15),
    pc_x=1, pc_y=2
)
plotter.plot_outliers(scores, outliers, pc_x=1, pc_y=2)

print("✓ Analysis complete! Results in ./mass_spec_results/")
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with [scikit-learn](https://scikit-learn.org/) for PCA implementation
- Visualization powered by [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/)
- Interactive dashboard built with [Streamlit](https://streamlit.io/)

---

<p align="center">
  Made with ❤️ for the data science community
</p>
