# Predictive Modeling for Bitcoin Volatility Forecasting

## Overview
This project develops and evaluates a Bitcoin volatility forecasting pipeline using both traditional machine learning and deep learning approaches. The system is designed to support short-horizon risk analysis by transforming historical Bitcoin market data into volatility-focused features and comparing multiple forecasting strategies under a chronological evaluation framework.

The project includes two main workflows:

- `main.py` for baseline methods and Random Forest forecasting
- `main_lstm.py` for sequence-based LSTM forecasting

The goal is not simply to apply more advanced models, but to test whether increased model complexity improves predictive performance relative to strong volatility baselines. This creates a more defensible and realistic forecasting framework for academic analysis and model comparison. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

## Research Focus
Bitcoin is known for price instability, volatility clustering, and rapid regime shifts. These characteristics make volatility forecasting a meaningful problem for financial risk assessment and decision support. This project examines whether machine learning and sequence modeling approaches can forecast future Bitcoin volatility more effectively than simpler benchmark methods.

## Project Objectives
The project is organized around the following objectives:

- build a clean and reproducible volatility forecasting pipeline
- engineer meaningful features from raw Bitcoin OHLCV market data
- compare baseline volatility methods against machine learning models
- evaluate predictive performance using a realistic time-ordered split
- generate exportable results, metrics tables, and training diagnostics for reporting

## Dataset
The project uses historical Bitcoin market data stored as:

`data/raw/btc_kaggle.csv`

The pipeline expects the dataset to include the following columns:

- `Date`
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`

The scripts validate these columns before modeling begins. 

## Modeling Workflows

### 1. Baseline + Random Forest Pipeline (`main.py`)
This script performs the Week 2 forecasting workflow:

- loads and validates the dataset
- computes returns
- computes realized volatility
- computes EWMA volatility
- builds a forecasting target
- optionally merges context-aware features
- creates the modeling dataset
- performs a chronological 80/20 train-test split
- evaluates:
  - Historical Volatility
  - EWMA
  - Random Forest
- saves prediction outputs and metrics tables

This workflow establishes benchmark performance and tests whether Random Forest improves on traditional volatility estimators. :contentReference[oaicite:3]{index=3}

### 2. LSTM Forecasting Pipeline (`main_lstm.py`)
This script extends the project with a sequence-learning approach:

- uses the same data loading and preprocessing pipeline
- standardizes the feature set with `StandardScaler`
- creates fixed-length sequences for temporal modeling
- splits training data into training and validation subsets
- trains an LSTM model
- evaluates LSTM predictions against the same baseline methods
- saves prediction outputs, metrics tables, and a loss curve figure

The sequence length is set to 14, meaning the model uses 14 time steps to forecast subsequent volatility behavior. :contentReference[oaicite:4]{index=4}

## Evaluation Strategy
This project uses a **chronological train-test split** rather than a random split. That design choice is important because volatility forecasting is a time-series problem. The model should learn from past data and be evaluated on future data, not on randomly mixed observations.

Performance is measured using:

- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**

These metrics are implemented in `src/metrics.py`, along with a helper function for saving metrics tables. :contentReference[oaicite:5]{index=5}

## Current Model Comparison
Based on the current pipeline output from `main.py`, the model ranking is:

1. Historical Volatility
2. Random Forest
3. EWMA

Example reported results:

| Model | RMSE | MAE |
|---|---:|---:|
| Historical Volatility | 0.002478 | 0.001370 |
| Random Forest | 0.002705 | 0.001747 |
| EWMA | 0.004779 | 0.003844 |

These results suggest that, in the current configuration, the simple historical volatility baseline remains the strongest performer, while Random Forest improves over EWMA but does not yet surpass the best baseline. This comparison strengthens the analysis because it prevents the project from assuming that higher model complexity automatically produces better forecasting performance. :contentReference[oaicite:6]{index=6}

## Project Structure
```text
crypto_project/
├── data/
│   └── raw/
│       └── btc_kaggle.csv
├── outputs/
│   ├── charts/
│   ├── exports/
│   └── tables/
├── src/
│   ├── baseline_models.py
│   ├── context_features.py
│   ├── lstm_model.py
│   ├── metrics.py
│   ├── preprocessing.py
│   └── rf_model.py
├── main.py
├── main_lstm.py
├── requirements.txt
└── README.md
