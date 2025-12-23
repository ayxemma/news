# News Sentiment Analysis for SPY Returns Prediction

A comprehensive machine learning pipeline for predicting next-day SPY (S&P 500 ETF) returns using news sentiment, volume, complexity, and uncertainty features extracted from news articles.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Key Components](#key-components)
- [Data Requirements](#data-requirements)
- [Model Details](#model-details)
- [Results](#results)
- [Contributing](#contributing)

## 🎯 Overview

This project implements an end-to-end pipeline for:
1. **Feature Extraction**: Extracting sentiment, complexity, volume, and uncertainty features from news articles
2. **Feature Engineering**: Aggregating features to daily frequency, normalizing with rolling z-scores, and aligning with SPY returns
3. **Model Training**: Training multiple models (Random Forest, XGBoost, LightGBM) with time-series cross-validation
4. **Strategy Backtesting**: Constructing and backtesting long-short trading strategies based on model predictions
5. **Factor-Neutral Analysis**: Analyzing portfolio performance after neutralizing common risk factors (Fama-French factors)

## ✨ Features

### Feature Extraction
- **Sentiment Analysis**: Using RoBERTa-based sentiment classification model
- **Text Complexity**: Flesch Reading Ease, Gunning Fog Index, Dale-Chall Readability Score
- **Volume Metrics**: Article counts, token lengths
- **Uncertainty Indicators**: Linguistic uncertainty markers

### Feature Engineering
- Daily aggregation (overall and by category)
- Rolling window z-score normalization (90-day window)
- One-day lag alignment with SPY returns
- Feature clipping at 1st and 99th percentiles

### Modeling
- **Random Forest**: Tree-based ensemble model
- **XGBoost**: Gradient boosting model
- **LightGBM**: Light gradient boosting model
- **Ensemble**: Weighted combination (80% XGBoost, 10% LightGBM, 10% Random Forest)
- Time-series cross-validation (5-fold)
- Hyperparameter tuning with GridSearchCV

### Strategy Backtesting
- Z-score signal normalization
- Long-short portfolio construction
- Volatility targeting
- Transaction cost modeling
- Performance metrics: Sharpe ratio, max drawdown, turnover, hit rate
- Factor-neutral analysis using Fama-French factors

### Analysis Tools
- Feature correlation analysis (Pearson, Spearman)
- Feature distribution analysis
- Horizon selection analysis (1-14 days)
- Top feature visualization
- Model performance comparison

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/ayxemma/news.git
cd news
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

**Note**: If you encounter NumPy compatibility issues with PyTorch:
```bash
pip install "numpy<2.0.0"
```

3. **Download required data**:
   - Place your news dataset in `data/News_Category_Dataset_v3.json`
   - Place SPY returns data in `data/SP500.csv`
   - Download Fama-French factors from [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) and save as `data/F-F_Research_Data_5_Factors_2x3_daily.csv`

## 📁 Project Structure

```
news/
├── data/                          # Data directory (excluded from git)
│   ├── News_Category_Dataset_v3.json
│   ├── SP500.csv
│   ├── F-F_Research_Data_5_Factors_2x3_daily.csv
│   └── cache/                     # Feature extraction cache
├── data_loader.py                 # Data loading utilities
├── feature_extractor.py           # Feature extraction from articles
├── feature_analyzer.py            # Feature aggregation, normalization, analysis
├── model.py                       # Model training and evaluation
├── strategy_backtest.py           # Strategy construction and backtesting
├── notebook.ipynb                 # Main analysis notebook
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## 💻 Usage

### Quick Start

The easiest way to run the complete pipeline is through the Jupyter notebook:

```bash
jupyter notebook notebook.ipynb
```

The notebook includes:
1. **Data Loading**: Loading news articles and SPY returns
2. **Feature Extraction**: Computing sentiment, complexity, and volume features
3. **Feature Engineering**: Aggregation, normalization, and alignment
4. **Model Training**: Training models with cross-validation
5. **Strategy Backtesting**: Constructing and evaluating trading strategies
6. **Factor-Neutral Analysis**: Analyzing risk-adjusted returns

### Programmatic Usage

#### Feature Extraction

```python
from data_loader import DataLoader
from feature_extractor import ArticleFeatureExtractor

# Load data
data_loader = DataLoader()
df_news = data_loader.load_news_dataset()

# Extract features
extractor = ArticleFeatureExtractor(data_loader=data_loader)
df_features = extractor.compute_all_features(df_news, reload_cache=True)
```

#### Feature Engineering

```python
from feature_analyzer import FeatureAnalyzer

feature_analyzer = FeatureAnalyzer()
df_spy = data_loader.load_spy_returns()

# Complete pipeline: aggregate, normalize, align
df_clean = feature_analyzer.prepare_features_for_modeling(
    df_features, 
    df_spy,
    window_size=90,
    min_periods=30
)
```

#### Model Training

```python
from model import NewsSentimentModeler

modeler = NewsSentimentModeler()

# Run complete pipeline
results_summary = modeler.run_full_pipeline(
    df_clean,
    cv_folds=5,
    hyperparameters_file='model_hyperparameters.json',
    overwrite=False
)

# Visualize performance
modeler.visualize_model_performance()
```

#### Strategy Backtesting

```python
from strategy_backtest import StrategyBacktester

backtester = StrategyBacktester()

# Generate predictions
predictions = modeler.generate_predictions(X_backtest)

# Backtest strategy
for model_name, pred in predictions.items():
    results = backtester.backtest_strategy(
        predictions=pred,
        spy_returns=df_backtest['spy_return'].values,
        dates=df_backtest['date'],
        normalization='zscore',
        window=60,
        k=0.5,
        w_max=1.0,
        strategy_name=model_name
    )

# Factor-neutral analysis
factor_results = backtester.run_factor_neutral_analysis(
    french_factors_path='data/F-F_Research_Data_5_Factors_2x3_daily.csv'
)
```

## 🔧 Key Components

### `data_loader.py`
- `DataLoader`: Loads news dataset, SPY returns, and Fama-French factors
- Handles date parsing and data cleaning

### `feature_extractor.py`
- `ArticleFeatureExtractor`: Extracts features from news articles
  - Sentiment scores (positive, negative, neutral)
  - Text complexity metrics
  - Token counts and lengths
  - Uncertainty indicators
- Caches results for faster re-runs

### `feature_analyzer.py`
- `FeatureAnalyzer`: Feature engineering and analysis
  - Daily aggregation (overall and by category)
  - Rolling z-score normalization
  - Feature alignment with SPY returns
  - Correlation analysis
  - Distribution analysis
  - Horizon selection analysis

### `model.py`
- `NewsSentimentModeler`: Model training and evaluation
  - Time-series cross-validation
  - Hyperparameter tuning
  - Model persistence (save/load hyperparameters)
  - Ensemble prediction (weighted average)
  - Performance visualization

### `strategy_backtest.py`
- `StrategyBacktester`: Strategy construction and backtesting
  - Signal normalization (z-score, rank)
  - Position sizing (continuous, discrete)
  - Volatility targeting
  - Transaction cost modeling
  - Performance metrics calculation
  - Factor-neutral analysis

## 📊 Data Requirements

### News Dataset
- Format: JSON file with articles containing:
  - `headline`: Article headline
  - `category`: Article category
  - `date`: Publication date
  - `short_description`: Article text

### SPY Returns
- Format: CSV file with columns:
  - `Date`: Trading date
  - `Close`: Closing price (or returns)

### Fama-French Factors (Optional)
- Format: CSV file from Kenneth French Data Library
- Required factors: MKT-RF, SMB, HML, RF
- Used for factor-neutral analysis

## 🤖 Model Details

### Training Period
- **Training**: 2012-2019
- **Test**: 2020-2021
- **Out-of-Sample**: 2022+

### Models

1. **Random Forest**
   - Hyperparameters: `n_estimators`, `max_depth`, `min_samples_split`
   - No feature scaling required

2. **XGBoost**
   - Hyperparameters: `n_estimators`, `max_depth`, `learning_rate`
   - No feature scaling required

3. **LightGBM**
   - Hyperparameters: `n_estimators`, `max_depth`, `learning_rate`, `num_leaves`
   - No feature scaling required

4. **Ensemble**
   - Weighted average: 80% XGBoost, 10% LightGBM, 10% Random Forest

### Cross-Validation
- Method: TimeSeriesSplit (5-fold)
- Prevents look-ahead bias
- Expanding window approach

## 📈 Results

The pipeline evaluates models on multiple metrics:
- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **Directional Accuracy**: Percentage of correct direction predictions
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Maximum peak-to-trough decline
- **Turnover**: Average daily position changes

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) for sentiment analysis models
- [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) for factor data
- [scikit-learn](https://scikit-learn.org/) for machine learning tools
- [XGBoost](https://xgboost.readthedocs.io/) and [LightGBM](https://lightgbm.readthedocs.io/) for gradient boosting

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This project is for research and educational purposes. Past performance does not guarantee future results. Always conduct thorough backtesting and risk analysis before deploying any trading strategy.

