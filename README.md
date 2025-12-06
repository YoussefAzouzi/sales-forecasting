# Store Sales Forecasting (Time Series)

This project forecasts daily product-family sales per store using the Corporación Favorita “Store Sales – Time Series Forecasting” dataset.[web:3][web:8][web:18]

## Setup

1. Download the Kaggle competition data files into `data/raw/`:
   - `train.csv`, `test.csv`, `stores.csv`, `oil.csv`, `holidays_events.csv`, `transactions.csv`.[web:18]

2. Create a virtual environment and install requirements:
pip install -r requirements.txt

text

3. Run training:
python -m src.train

text

4. Generate predictions:
python -m src.predict

text

The pipeline covers merging external time series, feature engineering (lags, rolling stats, calendar and promotion features), model training with XGBoost, and CSV submission generation.
This gives you a complete, ready‑to‑run sales forecasting / time‑series project with a real Kaggle dataset, clean folder structure, and full code for every file.