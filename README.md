# Store Sales Forecasting (Time Series)

This project builds an end-to-end time series forecasting pipeline to predict daily grocery sales for multiple stores and product families using the **Store Sales – Time Series Forecasting** dataset.[web:18][web:62] It demonstrates data engineering, feature engineering, model training with XGBoost, and automated prediction generation suitable for Kaggle-style workflows.[web:49][web:59]

---

## Dataset

The project uses the Kaggle **Store Sales – Time Series Forecasting** dataset (Corporación Favorita grocery sales).[web:18][web:62]

Download it from Kaggle (requires account and competition rules acceptance):  
https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data[web:18]

Place the downloaded CSV files in:

data/raw/
train.csv
test.csv
stores.csv
oil.csv
holidays_events.csv
transactions.csv

---

## Project Structure

sales-forecasting-store-sales/
├── data/
│ ├── raw/ # Original Kaggle CSVs (not tracked by git)
│ └── processed/ # Parquet files with merged and feature-engineered data
├── notebooks/
│ └── 01_eda.ipynb # Optional exploratory analysis
├── src/
│ ├── config.py # Paths and global config
│ ├── data_loader.py # Load and merge raw data sources
│ ├── features.py # Feature engineering (lags, rolling stats, promos, calendar)
│ ├── models.py # XGBoost training utilities
│ ├── train.py # End-to-end training pipeline
│ └── predict.py # Generate predictions and submission.csv
├── tests/
│ └── test_pipeline.py # Simple end-to-end pipeline test
├── .gitignore
├── requirements.txt
└── README.md

---

## Setup

1. **Clone the repository**

git clone https://github.com/YoussefAzouzi/sales-forecasting.git
cd sales-forecasting

text

2. **Create and activate a virtual environment (optional but recommended)**

python -m venv .venv
.venv\Scripts\activate # Windows

source .venv/bin/activate # macOS/Linux
text

3. **Install dependencies**

pip install -r requirements.txt

4. **Download and place the dataset**

- Go to the Kaggle competition page:  
  https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data[web:18]  
- Download all required CSVs and place them under `data/raw/` as described above.[web:18][web:62]

---

## Training the Model

Run the full pipeline (data merge → feature engineering → XGBoost training):

python -m src.train

This will:

- Merge `train.csv` with `stores.csv`, `oil.csv`, `holidays_events.csv`, and `transactions.csv`.[web:18][web:59]  
- Create lag features, rolling mean/std features, promotion aggregates, and calendar features.[web:49][web:59]  
- Train an XGBoost regression model and save it under `models/xgb_store_sales.json`.

---

## Generating Predictions

After training, generate predictions for the Kaggle `test.csv`:

python -m src.predict

text

This will:

- Build test features aligned with the training feature set.  
- Load the trained XGBoost model.  
- Produce a `submission.csv` file in the project root with columns:

  - `id`: as defined in Kaggle `test.csv`.  
  - `sales`: predicted daily sales for each (date, store, family) combination.[web:18][web:59]

You can upload `submission.csv` to the Kaggle competition page to get a score.[web:18][web:62]

---

## Technologies Used

- **Python**, **Pandas**, **NumPy** for data manipulation  
- **XGBoost** for regression modeling  
- **Parquet** for efficient intermediate storage  
- **Git/GitHub** for version control and collaboration  

---

## Notes

- Raw data files under `data/raw/` are intentionally excluded from version control via `.gitignore` to keep the repository lightweight and to respect Kaggle’s data distribution rules.[web:94][web:95]  
- The project is intended as a learning and portfolio piece for time series forecasting, data engineering, and ML model deployment on structured data.[web:49][web:59]
