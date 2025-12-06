import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure directories exist
for d in [PROCESSED_DIR, MODELS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Files
TRAIN_RAW_PATH = RAW_DIR / "train.csv"
TEST_RAW_PATH = RAW_DIR / "test.csv"
STORES_PATH = RAW_DIR / "stores.csv"
OIL_PATH = RAW_DIR / "oil.csv"
HOLIDAYS_PATH = RAW_DIR / "holidays_events.csv"
TRANSACTIONS_PATH = RAW_DIR / "transactions.csv"

TRAIN_MERGED_PATH = PROCESSED_DIR / "train_merged.parquet"
TRAIN_FEATURES_PATH = PROCESSED_DIR / "train_features.parquet"
TEST_FEATURES_PATH = PROCESSED_DIR / "test_features.parquet"

MODEL_PATH = MODELS_DIR / "xgb_store_sales.json"
