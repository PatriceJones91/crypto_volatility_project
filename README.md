# Predictive Modeling for Bitcoin Volatility Forecasting

## Overview

This project develops a reproducible Bitcoin volatility forecasting framework for short-horizon risk analysis. The goal is not to predict exact future Bitcoin prices, create a trading bot, or provide investment advice. Instead, the project focuses on forecasting Bitcoin volatility and comparing how different modeling strategies perform under a time-ordered evaluation process.

The current direction of the project centers on comparing a static forecasting model with a periodically refreshed forecasting model. This is important because cryptocurrency markets change over time. A model trained once may perform well during one market period but become less reliable as volatility patterns shift. A periodically refreshed model allows the forecasting process to update with newer data so the project can evaluate whether model refresh strategies improve short-term volatility prediction.

The project includes two main workflows:

- `main.py` for baseline volatility methods and Random Forest forecasting
- `main_lstm.py` for sequence-based LSTM forecasting

The purpose of these workflows is to test whether additional model complexity improves forecasting performance compared to simpler volatility baselines.

## Research Focus

Bitcoin is known for rapid price movement, volatility clustering, and changing market behavior. Because of this, forecasting volatility can be more meaningful than attempting to predict exact future prices. Volatility forecasting supports risk interpretation by estimating how uncertain or unstable future price movement may be over a short horizon.

This project is guided by the following research question:

> How does a periodically refreshed forecasting model compare to a static forecasting model when predicting short-term Bitcoin volatility?

The project examines whether updating a model over time can produce better forecasting performance than relying on a model trained once and evaluated on future data without retraining.

## Project Objectives

The project is organized around the following objectives:

- Build a clean and reproducible Bitcoin volatility forecasting pipeline.
- Engineer meaningful volatility-focused features from historical Bitcoin OHLCV data.
- Compare baseline volatility methods against machine learning models.
- Compare static forecasting behavior against periodically refreshed forecasting behavior.
- Evaluate model performance using a chronological train-test split.
- Export results, metrics tables, and diagnostic visuals for academic reporting.
- Support advisor review, committee discussion, and future capstone development.

## What This Project Is Not

This project is not intended to be:

- A Bitcoin price prediction system.
- A trading application.
- A financial advice platform.
- A guaranteed investment decision tool.
- A live cryptocurrency buying or selling system.

The project is focused on forecasting, model comparison, evaluation, and risk interpretation.

## Dataset

The project uses historical Bitcoin market data stored as:

-data/raw/btc_kaggle.csv
